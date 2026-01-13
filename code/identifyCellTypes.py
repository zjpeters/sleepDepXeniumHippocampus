#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 15:15:41 2025

@author: zjpeters
"""
import os
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.insert(0, "/home/zjpeters/Documents/stanly/code")
import stanly
import pandas as pd
import scipy.stats as scipy_stats
import umap
import matplotlib.patches as mpatches
import matplotlib.cm as cm

# rawdata=os.path.join('/','media','zjpeters','Expansion','sleepDepXenium','rawdata')
# derivatives=os.path.join('/','media','zjpeters','Expansion','sleepDepXeniumHippocampus','derivatives')

rawdata=os.path.join('/','home','zjpeters','Documents','sleepDepXenium','rawdata')
derivatives=os.path.join('/','home','zjpeters','Documents','sleepDepXeniumHippocampus','derivatives')

#%% import data to use for cell type identification
experiment = stanly.loadParticipantsTsv(os.path.join('/','home','zjpeters','Documents','sleepDepXeniumHippocampus', 'participants.tsv'))
processedSamples = {}
# for actSample in range(len(experiment['sample-id'])):
#     sample = stanly.importXeniumData(os.path.join(rawdata, experiment['sample-id'][actSample]))
#     sampleProcessed = stanly.processXeniumData(sample, experiment['rotation'][actSample], derivatives)
#     processedSamples[actSample] = sampleProcessed
for sampleIdx in range(len(experiment['sample-id'] )):
    processedSamples[sampleIdx] = stanly.loadProcessedXeniumSample(os.path.join(derivatives, f"{experiment['sample-id'][sampleIdx]}_hippocampus"))
    # clusterInfo = pd.read_csv(os.path.join(derivatives, f'{processedSamples[sampleIdx]["sampleID"]}_cluster_information.csv'))
    # processedSamples[sampleIdx]['cluster_labels'] = clusterInfo['cluster_labels']
    # processedSamples[sampleIdx]['silhouette_values'] = clusterInfo['silhouette_values']
    # processedSamples[sampleIdx]['cluster_colors'] = np.array([clusterInfo['color_r'], clusterInfo['color_g'], clusterInfo['color_b'], clusterInfo['color_alpha']]).T
    # cluster_regions = pd.read_csv(os.path.join(derivatives, f'{processedSamples[sampleIdx]["sampleID"]}_cluster_associations.csv'), header=None)
    # processedSamples[sampleIdx]['cluster_region'] = np.squeeze(np.array(cluster_regions[1]))

sampleToCluster = processedSamples[0]
#%% create dictionary with information about different interneuron types, from csv from Junko
interneuron_information = dict.fromkeys(["Pvalb", "Sst", "Vip", "Sncg", "Lamp5"])
for i in interneuron_information.keys():
    interneuron_information[i] = dict.fromkeys(['geneList', 'geneIdx'])
interneuron_information["Pvalb"]["geneList"] = ['Btbd11', 'Cntnap4', 'Eya4', 'Kcnmb2', 'Pvalb', 'Slit2']
interneuron_information["Sst"]["geneList"] = ['Calb1', 'Lypd6', 'Pdyn', 'Rab3b', 'Rbp4']
interneuron_information["Vip"]["geneList"] = ['Chat', 'Crh', 'Igf1', 'Penk', 'Pthlh', 'Sorcs3', 'Thsd7a', 'Vip']
interneuron_information["Sncg"]["geneList"] = ['Col19a1', 'Kctd12', 'Necab1', 'Slc44a5']
interneuron_information["Lamp5"]["geneList"] = ['Dner', 'Gad1', 'Gad2', 'Hapln1', 'Lamp5', 'Pde11a', 'Rasgrf2']

# load gene lists from paper
"""
identify cell type genes present in data that are also present in:
    "Brain Cell Type Specific Gene Expression and Co-expression Network Architectures"
    https://www.nature.com/articles/s41598-018-27293-5
"""
cellTypeSpreadsheetLocation = os.path.join('/','home','zjpeters','Documents','stanly','code','data','cellTypeMarkerGeneInfo','Brain Cell Type Specific Gene Expression and Co-expression Network Architectures_41598_2018_27293_MOESM2_ESM_mouse_specificity.csv')
cellTypeGeneExpressionList = pd.read_csv(cellTypeSpreadsheetLocation)

#%% identify overlapping genes
for interneuron_type in interneuron_information.keys():
    interneuron_gene_idx = []
    for gene in interneuron_information[interneuron_type]['geneList']:
        geneIdx = sampleToCluster['geneList'].index(gene)
        interneuron_gene_idx.append(geneIdx)
    interneuron_information[interneuron_type]['geneIdx'] = np.array(interneuron_gene_idx)
    
#%% loop over gene list from data and find matching genes in cell type specificity list
xeniumCasefoldGeneList = []
for gene in sampleToCluster['geneList']:
    xeniumCasefoldGeneList.append(gene.casefold())
xeniumCasefoldGeneList = list(xeniumCasefoldGeneList)
cellTypeCasefoldList = []
for gene in cellTypeGeneExpressionList['gene']:
    cellTypeCasefoldList.append(gene.casefold())

sampleCasefoldList = []
for gene in sampleToCluster['geneList']:
    sampleCasefoldList.append(gene.casefold())

cellTypeGeneIdx = [x in sampleCasefoldList for x in cellTypeCasefoldList]

cellTypeGenesInSample = cellTypeGeneExpressionList[cellTypeGeneIdx]

# create lists of cell type genes
cellTypes = np.unique(cellTypeGenesInSample['Celltype'])
cellTypeGeneLists = {}
for i in cellTypes:
    cellTypeDF = cellTypeGenesInSample[cellTypeGenesInSample['Celltype'] == i]
    singleCellTypeGeneList = []
    for j in cellTypeDF['gene']:
        try:
            geneIdx = xeniumCasefoldGeneList.index(j.casefold())
            singleCellTypeGeneList.append([sampleToCluster['geneList'][geneIdx], geneIdx])
            cellTypeGeneLists[i] = np.array(singleCellTypeGeneList)

        except ValueError:
            # code above should work well, though might need to consider if there
            # are situations where a casefold gene name would lead to duplicates
            print('Gene not found')

for i in interneuron_information:
    singleCellTypeGeneList = np.array([interneuron_information[i]['geneList'], interneuron_information[i]['geneIdx']])
    cellTypeGeneLists[i] = singleCellTypeGeneList.T
    
#%% use gene lists to identify cell type of each cell
"""
neurons have the largest number of genes included, would explain the low z-score
this is becasuse of the variety of neurons, and should be somewhat fixed by 
inclusion of interneurons, but will still need to consider other options
"""
geneMatrixZScore = sampleToCluster['geneMatrixLog2'].todense()
geneMatrixZScore = (geneMatrixZScore - np.mean(geneMatrixZScore, axis=0))/np.std(geneMatrixZScore, axis=0)
plt.close('all')
meanZScoreMatrixCellTypes = np.zeros([sampleToCluster['geneMatrixLog2'].shape[1], len(cellTypeGeneLists)])
cellTypeProbs = np.zeros([sampleToCluster['geneMatrixLog2'].shape[1], len(cellTypeGeneLists)])
for i in range(sampleToCluster['geneMatrixLog2'].shape[1]):
    for j in enumerate(cellTypeGeneLists): 
        geneMask = np.array(cellTypeGeneLists[j[1]][:,1], dtype='int32')
        cellTypeMatrix = np.squeeze(np.array(geneMatrixZScore[geneMask, i]))
        # cellTypeMatrix = cellTypeMatrix[geneMask, :]
        # cellTypeMatrixMeanZScore = np.squeeze(np.array(np.mean(cellTypeMatrix, axis=0)))
        meanZScore = np.mean(cellTypeMatrix)
        meanZScoreMatrixCellTypes[i, j[0]] = meanZScore
        # for the sake of calculating cell type probability, we don't want negative z-scores
        if meanZScore < 0:
            cellTypeProbs[i,j[0]] = 0
        else:
            cellTypeProbs[i,j[0]] = scipy_stats.norm.cdf(np.abs(meanZScore))

meanZScoreMatrixInterneurons = np.zeros([sampleToCluster['geneMatrixLog2'].shape[1], len(cellTypes)])
for i in range(sampleToCluster['geneMatrixLog2'].shape[1]):
    for j in enumerate(interneuron_information): 
        geneMask = np.array(interneuron_information[j[1]]['geneIdx'], dtype='int32')
        cellTypeMatrix = np.squeeze(np.array(geneMatrixZScore[geneMask, i]))
        # cellTypeMatrix = cellTypeMatrix[geneMask, :]
        # cellTypeMatrixMeanZScore = np.squeeze(np.array(np.mean(cellTypeMatrix, axis=0)))
        meanZScore = np.mean(cellTypeMatrix)
        meanZScoreMatrixInterneurons[i, j[0]] = meanZScore


#%% plot the probabilities for X number of cells
plt.close('all')
plt.figure()
for i in range(len(cellTypeProbs)):
    plt.plot(cellTypeProbs[i,:])
plt.show()
plt.xlabel(list(cellTypeGeneLists.keys()))
#%% try to look at what the average probability is across groups
averageCellTypeProb = np.mean(cellTypeProbs, axis=0)
plt.close('all')
plt.figure()
plt.plot(averageCellTypeProb)
plt.show()
#%% use the mean z-score matrix to identify the probability of each cell type
cellTypeMax = np.max(meanZScoreMatrixCellTypes, axis=1)
cellTypeMaxIdx = np.argmax(meanZScoreMatrixCellTypes, axis=1)
cellTypeColumnNames = list(cellTypeGeneLists.keys())
cellTypeProbDF = pd.DataFrame(cellTypeProbs, columns=cellTypeColumnNames)
cellTypeProbMax = np.max(cellTypeProbs, axis=1)
cellTypeProbMaxIdx = np.argmax(cellTypeProbs, axis=1)

cellTypeEstimates = []
cellTypeColors = []
for i in enumerate(cellTypeProbMaxIdx):
    cellTypeEstimates.append([cellTypeProbMax[i[0]], cellTypeColumnNames[i[1]]])
    # add color for each cell type
    cellTypeColors.append(cm.tab10(i[1]))
cellTypeEstimates = np.array(cellTypeEstimates)
cellTypeColors = np.array(cellTypeColors)
plt.plot(cellTypeProbMax)

#%% 
plt.close('all')
plt.figure()
plt.imshow(sampleToCluster['tissueImageProcessed'], cmap='gray_r')
plt.scatter(sampleToCluster['processedTissuePositionList'][:,0], sampleToCluster['processedTissuePositionList'][:,1], c=cellTypeColors, s=4)
plt.show()

#%% Perform umap
reducer = umap.UMAP()

embedding = reducer.fit_transform(sampleToCluster['geneMatrixLog2'].T)

#%% plot umap using cell type lists
fig = plt.figure()
umapAxs = fig.add_subplot()

umapAxs.scatter(embedding[:,0], embedding[:,1], c=cellTypeColors, s=5)
umapAxs.set_title("UMAP of all samples")
umapAxs.set_xticks([])
umapAxs.set_yticks([])
# create handles and patches to generate labeled legend
handles, labels = umapAxs.get_legend_handles_labels()
patch = mpatches.Patch(color=cm.tab10(0), label='astrocytes')
handles.append(patch) 
patch = mpatches.Patch(color=cm.tab10(1), label='endothelial cells')
handles.append(patch) 
patch = mpatches.Patch(color=cm.tab10(2), label='microglia')
handles.append(patch) 
patch = mpatches.Patch(color=cm.tab10(3), label='neurons')
handles.append(patch) 
patch = mpatches.Patch(color=cm.tab10(4), label='oligodendrocytes')
handles.append(patch) 
patch = mpatches.Patch(color=cm.tab10(5), label='Pvalb interneurons')
handles.append(patch)
patch = mpatches.Patch(color=cm.tab10(6), label='Sst interneurons')
handles.append(patch)
patch = mpatches.Patch(color=cm.tab10(7), label='Vip interneurons')
handles.append(patch)
patch = mpatches.Patch(color=cm.tab10(8), label='Sncg interneurons')
handles.append(patch)
patch = mpatches.Patch(color=cm.tab10(9), label='Lamp5 interneurons')
handles.append(patch) 
umapAxs.legend(handles=handles, loc='upper left',bbox_to_anchor=(1, 1))
plt.show()
#%% plot the two possible sections as lines
"""
the below plots will be odd because the interneurons are included in the cell types
"""
plt.close('all')
cellTypeMax = np.max(meanZScoreMatrixCellTypes, axis=1)
cellTypeMaxIdx = np.argmax(meanZScoreMatrixCellTypes, axis=1)
neuronMax = np.max(meanZScoreMatrixInterneurons, axis=1)
diffOfMax = neuronMax - cellTypeMax
plt.figure()
plt.plot(diffOfMax)
plt.show()
x = np.abs(diffOfMax) > 0.5
print(sum(x))
plt.figure()
plt.plot(cellTypeMax)
plt.plot(neuronMax)
plt.show()
cellTypeAndNeuronMax = np.array([cellTypeMax, neuronMax]).T
plt.figure()
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
plt.scatter(cellTypeMax, neuronMax)
plt.axis('equal')
plt.show()

#%% plot cell types with colors
plt.close('all')
astColor = np.squeeze(np.array([0.9019607843137255, 0.3333333333333333, 0.050980392156862744, 1.0]))
endColor = np.squeeze(np.array([0.6196078431372549, 0.792156862745098, 0.8823529411764706, 1.0]))
micColor = np.squeeze(np.array([0.19215686274509805, 0.6392156862745098, 0.32941176470588235, 1.0]))
neuColor = np.squeeze(np.array([0.6313725490196078, 0.8509803921568627, 0.6078431372549019, 1.0]))
oliColor = np.squeeze(np.array([0.7372549019607844, 0.7411764705882353, 0.8627450980392157, 1.0]))

cellTypeColors = np.zeros([sampleToCluster['geneMatrixLog2'].shape[1], 4])
colorIdx = np.where(cellTypeMaxIdx == 0)
cellTypeColors[colorIdx, :] = astColor
colorIdx = np.where(cellTypeMaxIdx == 1)
cellTypeColors[colorIdx, :] = endColor
colorIdx = np.where(cellTypeMaxIdx == 2)
cellTypeColors[colorIdx, :] = micColor
colorIdx = np.where(cellTypeMaxIdx == 3)
cellTypeColors[colorIdx, :] = neuColor
colorIdx = np.where(cellTypeMaxIdx == 4)
cellTypeColors[colorIdx, :] = oliColor

fig = plt.figure()
cellTypeScatter = fig.add_subplot()
cellTypeScatter.scatter(cellTypeMax, neuronMax, c=cellTypeColors)
# create handles and patches to generate labeled legend
handles, labels = cellTypeScatter.get_legend_handles_labels()
patch = mpatches.Patch(color=astColor, label='astrocytes')
handles.append(patch) 
patch = mpatches.Patch(color=endColor, label='endothelial cells')
handles.append(patch) 
patch = mpatches.Patch(color=micColor, label='microglia')
handles.append(patch) 
patch = mpatches.Patch(color=neuColor, label='neurons')
handles.append(patch) 
patch = mpatches.Patch(color=oliColor, label='oligodendrocytes')
handles.append(patch) 
cellTypeScatter.legend(handles=handles, loc='upper right',bbox_to_anchor=(1, 1))
plt.xlabel('Cell type z-score')
plt.ylabel('Neuronal z-score')
plt.show()

#%% plot UMAP
fig = plt.figure()
umapAxs = fig.add_subplot()
umapAxs.scatter(embedding[:,0], embedding[:,1], c=cellTypeColors, s=3)
umapAxs.set_title("UMAP of all samples")
umapAxs.set_xticks([])
umapAxs.set_yticks([])
plt.show()