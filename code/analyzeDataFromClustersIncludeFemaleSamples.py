#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:48:31 2025

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
import csv

rawdata=os.path.join('/','media','zjpeters','Expansion','sleepDepXenium','rawdata')
derivatives=os.path.join('/','media','zjpeters','Expansion','sleepDepXeniumHippocampus','derivatives')
# colors to use in the output figures, with scatter color being slightly darker
colorHCHex = '#f3766e'
colorHCScatterHex = '#fe4d34'
colorMDDHex = '#1cbdc2'
colorMDDScatterHex = '#14a7c2'
#%% load newly processed samples and check selection

locOfTsvFile = os.path.join('/','home','zjpeters','Documents','sleepDepXeniumHippocampus', 'participants.tsv')
participants = pd.read_csv(locOfTsvFile, delimiter='\t')

experiment = {'sample-id': participants['participant_id'].to_numpy(),
                    'rotation': participants['deg_rot'].to_numpy(),
                    'experimental-group': participants['sleep_dep'].to_numpy(),
                    'flip': participants['flip'].to_numpy(),
                    'sex': participants['sex'].to_numpy()}
processedSamples = {}

for sampleIdx in range(len(experiment['sample-id'] )):
    processedSamples[sampleIdx] = stanly.loadProcessedXeniumSample(os.path.join(derivatives, f"{experiment['sample-id'][sampleIdx]}_hippocampus"))
    clusterInfo = pd.read_csv(os.path.join(derivatives, f'{processedSamples[sampleIdx]["sampleID"]}_cluster_information.csv'))
    processedSamples[sampleIdx]['cluster_labels'] = clusterInfo['cluster_labels']
    processedSamples[sampleIdx]['silhouette_values'] = clusterInfo['silhouette_values']
    processedSamples[sampleIdx]['cluster_colors'] = np.array([clusterInfo['color_r'], clusterInfo['color_g'], clusterInfo['color_b'], clusterInfo['color_alpha']]).T
    cluster_regions = pd.read_csv(os.path.join(derivatives, f'{processedSamples[sampleIdx]["sampleID"]}_cluster_associations.csv'), header=None)
    processedSamples[sampleIdx]['cluster_region'] = np.squeeze(np.array(cluster_regions[1]))

#%% plot each cluster individually
actK = 15
sampleToCluster = processedSamples[0]
# plt.close('all')

# plt.figure()
# plt.imshow(sampleToCluster['tissueImageProcessed'],cmap='gray_r')
# plt.scatter(sampleToCluster['processedTissuePositionList'][:,0], sampleToCluster['processedTissuePositionList'][:,1],c=sampleToCluster['cluster_colors'], s=5)
# plt.title("The visualization of the clustered control data.")
# plt.axis('off')
# plt.show()
# for i in range(actK):
#     plt.figure()
#     clusterIdx = np.where(sampleToCluster['cluster_labels'] == i)[0]
#     plt.imshow(sampleToCluster['tissueImageProcessed'],cmap='gray_r')
#     plt.scatter(sampleToCluster['processedTissuePositionList'][clusterIdx,0], sampleToCluster['processedTissuePositionList'][clusterIdx,1],c=sampleToCluster['cluster_colors'][clusterIdx], s=5)
#     plt.title(f"{sampleToCluster['sampleID']} cluster {i}.")
#     plt.axis('off')
#     plt.show()

#%% create function to restrict to plotting specific clusters from each sample

def findRelevantClusters(processedSample, cluster_region):
    matchingClusters = np.where(processedSample['cluster_region'] == cluster_region)[0]
    clusterMask = []
    for cellIdx in range(len(processedSample['cluster_labels'])):
        # since it's possible that more than one cluster correspond to cluster of interest, use for loop
        for clusterIdx in matchingClusters:
            if processedSample['cluster_labels'][cellIdx] == clusterIdx:
                clusterMask.append(cellIdx)
    return clusterMask

#%% generate male and female samples dictionaries
maleSamples = []
maleSamplesIdx = [0,1,2,3,4,5]
for i in maleSamplesIdx:
    maleSamples.append(processedSamples[i])

femaleSamples = []
femaleSamplesIdx = [6,7,8,9,10,11]
for i in femaleSamplesIdx:
    femaleSamples.append(processedSamples[i])

designMatrix = [0,1,0,1,0,1]
# create empty arrays the height of the number of genes in the gene matrix
controlCA1Cells = np.empty([len(maleSamples[0]['geneList']),0])
experimentalCA1Cells = np.empty([len(maleSamples[0]['geneList']),0])
# loop over all samples and place all cells positive for gene of interest in array
regionOfInterest = 'CA3'
for actSample in range(len(maleSamples)):
    clusterIdx = findRelevantClusters(maleSamples[actSample], regionOfInterest)
    if designMatrix[actSample] == 0:
        controlCA1Cells = np.append(controlCA1Cells, maleSamples[actSample]['geneMatrixLog2'][:,clusterIdx].todense(), axis=1)
    elif designMatrix[actSample] == 1:
        experimentalCA1Cells = np.append(experimentalCA1Cells, maleSamples[actSample]['geneMatrixLog2'][:,clusterIdx].todense(), axis=1)
        

#%% write code that performs the comparison and displays the results for both CA1 and CA3 overlayed on tissue

controlCA1Cells = np.empty([len(maleSamples[0]['geneList']),0])
experimentalCA1Cells = np.empty([len(maleSamples[0]['geneList']),0])
controlCA3Cells = np.empty([len(maleSamples[0]['geneList']),0])
experimentalCA3Cells = np.empty([len(maleSamples[0]['geneList']),0])
# loop over all samples and place all cells positive for gene of interest in array
for actSample in range(len(maleSamples)):
    ca1Idx = findRelevantClusters(maleSamples[actSample], 'CA1')
    ca3Idx = findRelevantClusters(maleSamples[actSample], 'CA3')
    if designMatrix[actSample] == 0:
        controlCA1Cells = np.append(controlCA1Cells, maleSamples[actSample]['geneMatrixLog2'][:,ca1Idx].todense(), axis=1)
        controlCA3Cells = np.append(controlCA3Cells, maleSamples[actSample]['geneMatrixLog2'][:,ca3Idx].todense(), axis=1)
    elif designMatrix[actSample] == 1:
        experimentalCA1Cells = np.append(experimentalCA1Cells, maleSamples[actSample]['geneMatrixLog2'][:,ca1Idx].todense(), axis=1)
        experimentalCA3Cells = np.append(experimentalCA3Cells, maleSamples[actSample]['geneMatrixLog2'][:,ca3Idx].todense(), axis=1)
        
#%% perform umap of single sample
sampleToCluster = processedSamples[0]
# load cluster regions with cell types included
cluster_regions = pd.read_csv(os.path.join(derivatives, 'clusterAssociationCsvs', f'{sampleToCluster["sampleID"]}_cluster_associations_cell_types.csv'), header=None)
reducer = umap.UMAP()

embedding = reducer.fit_transform(sampleToCluster['geneMatrixLog2'].T)

#%% plot umap of single sample
plt.close('all')
fig, ax = plt.subplots(1,2, figsize=(12,8))
ax[0].imshow(sampleToCluster['tissueImageProcessed'],cmap='gray_r')
ax[0].scatter(sampleToCluster['processedTissuePositionList'][:,0], sampleToCluster['processedTissuePositionList'][:,1],c=sampleToCluster['cluster_colors'], s=5)
ax[0].axis('off')
ax[0].set_title("Clustering of sample at k=15")
ax[1].scatter(embedding[:,0], embedding[:,1], c=sampleToCluster['cluster_colors'])
ax[1].axis('equal')
ax[1].set_title("UMAP of sample with cluster labels")
plt.savefig(os.path.join(derivatives, f'xenium_cluster{actK}_with_UMAP.png'), bbox_inches='tight', dpi=300)
plt.show()

#%% try running umap on combined NSD samples
"""
Want to standardize cluster colors so that each sample uses the same colors
additionally set it so that regions represented with multiple clusters share only one color
"""
sample01ClustersWithColors = pd.read_csv(os.path.join(derivatives, 'YW-1_ROI_A1_hippocampus_cluster_associations_with_colors.csv'), header=None)
plt.close('all')
for i in range(sample01ClustersWithColors.shape[0]):
    plt.scatter(i,i, c=np.squeeze(np.array(sample01ClustersWithColors)[i,-4:]), label=np.array(sample01ClustersWithColors)[i,1])
plt.scatter(15,15,c=[0.7,0.7,0.7,1])
plt.legend()

#%% run umap embedding using all cells, SD and NSD
plt.close('all')
allCellsMale = np.empty([len(maleSamples[0]['geneList']),0])

ca1Color = np.squeeze(np.array([0.807843137254902, 0.427450980392157, 0.741176470588235, 1]))
ca2Color = np.squeeze(np.array([0.83921568627451, 0.380392156862745, 0.419607843137255, 1]))
ca3Color = np.squeeze(np.array([0.741176470588235, 0.619607843137255, 0.223529411764706, 1]))
ca4dgColor = np.squeeze(np.array([0.388235294117647	, 0.474509803921569, 0.223529411764706, 1]))
dgColor = np.squeeze(np.array([0.223529411764706, 0.231372549019608, 0.474509803921569, 1]))
sparseColor = np.squeeze(np.array([0.7,0.7,0.7,1]))
regionList = ['CA1','CA2','CA3','DG/CA4','DG']
colorsAllCellsMale = np.empty([0,4])
for actSample in range(len(maleSamples)):
    allCellsMale  = np.append(allCellsMale , maleSamples[actSample]['geneMatrixLog2'].todense(), axis=1)
    sampleColors = maleSamples[actSample]['cluster_colors']
    ca1Idx = findRelevantClusters(maleSamples[actSample], 'CA1')
    sampleColors[ca1Idx,:] = ca1Color
    ca2Idx = findRelevantClusters(maleSamples[actSample], 'CA2')
    sampleColors[ca2Idx,:] = ca2Color
    ca3Idx = findRelevantClusters(maleSamples[actSample], 'CA3')
    sampleColors[ca3Idx,:] = ca3Color
    ca4dgIdx = findRelevantClusters(maleSamples[actSample], 'DG/CA4')
    sampleColors[ca4dgIdx,:] = ca4dgColor
    dgIdx = findRelevantClusters(maleSamples[actSample], 'DG')
    sampleColors[dgIdx,:] = dgColor
    sparseIdx = findRelevantClusters(maleSamples[actSample], 'sparse')
    sampleColors[sparseIdx,:] = sparseColor
    colorsAllCellsMale = np.append(colorsAllCellsMale, sampleColors, axis=0)
    
#%% run umap embedding using all cells, SD and NSD
plt.close('all')
allCellsFemale = np.empty([len(femaleSamples[0]['geneList']),0])

colorsAllCellsFemale = np.empty([0,4])
for actSample in range(len(maleSamples)):
    allCellsFemale  = np.append(allCellsFemale , femaleSamples[actSample]['geneMatrixLog2'].todense(), axis=1)
    sampleColors = femaleSamples[actSample]['cluster_colors']
    ca1Idx = findRelevantClusters(femaleSamples[actSample], 'CA1')
    sampleColors[ca1Idx,:] = ca1Color
    ca2Idx = findRelevantClusters(femaleSamples[actSample], 'CA2')
    sampleColors[ca2Idx,:] = ca2Color
    ca3Idx = findRelevantClusters(femaleSamples[actSample], 'CA3')
    sampleColors[ca3Idx,:] = ca3Color
    ca4dgIdx = findRelevantClusters(femaleSamples[actSample], 'DG/CA4')
    sampleColors[ca4dgIdx,:] = ca4dgColor
    dgIdx = findRelevantClusters(femaleSamples[actSample], 'DG')
    sampleColors[dgIdx,:] = dgColor
    sparseIdx = findRelevantClusters(femaleSamples[actSample], 'sparse')
    sampleColors[sparseIdx,:] = sparseColor
    colorsAllCellsFemale = np.append(colorsAllCellsFemale, sampleColors, axis=0)
#%% perform umap plot on combined cells

reducer = umap.UMAP()

embeddingMale = reducer.fit_transform(np.array(allCellsMale).T)
embeddingFemale = reducer.fit_transform(np.array(allCellsFemale).T)

#%% plot each cluster individually to look for interneuron genes
# create dictionary with information about different interneuron types
interneuron_information = dict.fromkeys(["Pvalb", "Sst", "Vip", "Sncg", "Lamp5"])
for i in interneuron_information.keys():
    interneuron_information[i] = dict.fromkeys(['geneList', 'geneIdx'])
interneuron_information["Pvalb"]["geneList"] = ['Btbd11', 'Cntnap4', 'Eya4', 'Kcnmb2', 'Pvalb', 'Slit2']
interneuron_information["Sst"]["geneList"] = ['Calb1', 'Lypd6', 'Pdyn', 'Rab3b', 'Rbp4']
interneuron_information["Vip"]["geneList"] = ['Chat', 'Crh', 'Igf1', 'Penk', 'Pthlh', 'Sorcs3', 'Thsd7a', 'Vip']
interneuron_information["Sncg"]["geneList"] = ['Col19a1', 'Kctd12', 'Necab1', 'Slc44a5']
interneuron_information["Lamp5"]["geneList"] = ['Dner', 'Gad1', 'Gad2', 'Hapln1', 'Lamp5', 'Pde11a', 'Rasgrf2']

for interneuron_type in interneuron_information.keys():
    interneuron_gene_idx = []
    for gene in interneuron_information[interneuron_type]['geneList']:
        geneIdx = sampleToCluster['geneList'].index(gene)
        interneuron_gene_idx.append(geneIdx)
    interneuron_information[interneuron_type]['geneIdx'] = np.array(interneuron_gene_idx)
    
# load gene lists from paper
"""
identify genes present in allen data that are also present in:
    "Brain Cell Type Specific Gene Expression and Co-expression Network Architectures"
    https://www.nature.com/articles/s41598-018-27293-5
"""
cellTypeSpreadsheetLocation = os.path.join('/','home','zjpeters','Documents','stanly','code','data','cellTypeMarkerGeneInfo','Brain Cell Type Specific Gene Expression and Co-expression Network Architectures_41598_2018_27293_MOESM2_ESM_mouse_specificity.csv')
cellTypeGeneExpressionList = pd.read_csv(cellTypeSpreadsheetLocation)


"""
need to use inclusion exclusion criteria that checks for multiple genes expressed
within cell
"""
#%% loop over gene list from allen and find matching genes in cell type specificity list

cellTypeCasefoldList = []
for gene in cellTypeGeneExpressionList['gene']:
    cellTypeCasefoldList.append(gene.casefold())

sampleCasefoldList = []
for gene in sampleToCluster['geneList']:
    sampleCasefoldList.append(gene.casefold())

cellTypeGeneIdx = [x in sampleCasefoldList for x in cellTypeCasefoldList]

cellTypeGenesInSample = cellTypeGeneExpressionList[cellTypeGeneIdx]

#%% create lists of cell type genes
cellTypes = np.unique(cellTypeGenesInSample['Celltype'])
cellTypeGeneLists = {}
for i in cellTypes:
    cellTypeDF = cellTypeGenesInSample[cellTypeGenesInSample['Celltype'] == i]
    singleCellTypeGeneList = []
    for j in cellTypeDF['gene']:
        try:
            geneIdx = sampleToCluster['geneList'].index(j.title())
            singleCellTypeGeneList.append([j.title(), geneIdx])
            cellTypeGeneLists[i] = np.array(singleCellTypeGeneList)

        except ValueError:
            #### need to update the search because it's not getting the case correct
            #### doing this in the meantime
            print('Gene not found')                    
#%% look for whether the sparse clusters correlate with any specific gene set
sampleToCluster = femaleSamples[5]
sparseClusters = np.where(sampleToCluster['cluster_region'] == 'sparse')[0]
geneMatrixZScore = sampleToCluster['geneMatrixLog2'].todense()
geneMatrixZScore = (geneMatrixZScore - np.mean(geneMatrixZScore, axis=0))/np.std(geneMatrixZScore, axis=0)
plt.close('all')
fig,ax = plt.subplots(len(cellTypes), len(sparseClusters), figsize=(21, 11))
meanZScoreMatrix = np.zeros([len(cellTypes), len(sparseClusters)])
for i in range(len(sparseClusters)):
    for j in enumerate(cellTypeGeneLists):
        clusterIdx = np.where(sampleToCluster['cluster_labels'] == sparseClusters[i])[0]    
        geneMask = np.array(cellTypeGeneLists[j[1]][:,1], dtype='int32')
        cellTypeMatrix = geneMatrixZScore[:, clusterIdx]
        cellTypeMatrix = cellTypeMatrix[geneMask, :]
        cellTypeMatrixMeanZScore = np.squeeze(np.array(np.mean(cellTypeMatrix, axis=0)))
        meanZScore = np.mean(cellTypeMatrixMeanZScore)
        meanZScoreMatrix[j[0], i] = meanZScore
        # print(f"Mean z-score for cell type {j[1]} in cluster {sparseClusters[i]}: {meanZScore}")
        ax[0, i].set_title(f'Cluster {sparseClusters[i]}')
        ax[j[0], 0].set_ylabel(j[1], rotation='horizontal', horizontalalignment='right')
        ax[j[0],i].imshow(sampleToCluster['tissueImageProcessed'],cmap='gray_r')
        scatterAx = ax[j[0],i].scatter(sampleToCluster['processedTissuePositionList'][clusterIdx,0], sampleToCluster['processedTissuePositionList'][clusterIdx,1],c=cellTypeMatrixMeanZScore, cmap='seismic', s=3, vmin=-2, vmax=2)
        ax[j[0],i].tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False,
            left=False,
            labelleft=False)
        # ax[i,j[0].set_title(f"Expression of {j[1]} in\n{sampleToCluster['sampleID']} cluster {sparseClusters[i]}.\nMean z-score = {meanZScore}")
        # plt.axis('off')
plt.suptitle('Mean z-score for cell type marker genes')
cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
fig.colorbar(scatterAx, cax=cbar_ax, fraction=0.01, pad=0.04)
plt.show()
plt.savefig(os.path.join(derivatives, f'xenium_mean_z-score_for_cell_type_marker_genes_{sampleToCluster["sampleID"]}.pdf'), dpi=300)
#%% create a plot with clustering, UMAP, and t-test for Rbm3 and Gpr161
"""
create a figure with two sets of subfigures, 
one horizontal for the cluster + umap
one vertical for Rbm3 + Gpr161
"""
sampleForDisplay = maleSamples[0]
ca1Idx = findRelevantClusters(sampleForDisplay, 'CA1')
ca3Idx = findRelevantClusters(sampleForDisplay, 'CA3')
desiredPval = 0.05
alphaFdr = 1 - np.power((1 - desiredPval),(1/(len(maleSamples[0]["geneList"]))))
plt.close('all')

#%%
plt.close('all')
fig = plt.figure(figsize=(14,7))
clustAxs = plt.subplot2grid((4,7), (1,0), rowspan=2, colspan=2)
umapAxs = plt.subplot2grid((4,7), (0,2), colspan=3, rowspan=4)
ttestAxs1 = plt.subplot2grid((4,7), (0,5), rowspan=2, colspan=2)
ttestAxs2 = plt.subplot2grid((4,7), (2,5), rowspan=2, colspan=2)
plt.show()

clustAxs.imshow(sampleToCluster['tissueImageProcessed'],cmap='gray_r')
clustAxs.scatter(sampleToCluster['processedTissuePositionList'][:,0], sampleToCluster['processedTissuePositionList'][:,1],c=sampleToCluster['cluster_colors'], s=3)
clustAxs.axis('off')
clustAxs.set_title("Clustering, k=15")
# create handles and patches to generate labeled legend
handles, labels = clustAxs.get_legend_handles_labels()
patch = mpatches.Patch(color=ca1Color, label='CA1')
handles.append(patch) 
patch = mpatches.Patch(color=ca2Color, label='CA2')
handles.append(patch) 
patch = mpatches.Patch(color=ca3Color, label='CA3')
handles.append(patch) 
patch = mpatches.Patch(color=ca4dgColor, label='CA4/DG')
handles.append(patch) 
patch = mpatches.Patch(color=dgColor, label='DG')
handles.append(patch) 
patch = mpatches.Patch(color=sparseColor, label='nonspecific cells')
handles.append(patch) 
# plot the legend
clustAxs.legend(handles=handles, loc='lower right',bbox_to_anchor=(1.1, 1.2))
umapAxs.scatter(embeddingMale[:,0], embeddingMale[:,1], c=colorsAllCellsMale, s=3)
umapAxs.set_title("UMAP of all samples")
umapAxs.set_xticks([])
umapAxs.set_yticks([])
# plt.tight_layout()
# calculate t-statistics for Rbm3 and Gpr161
rbm3Idx = maleSamples[0]["geneList"].index('Rbm3')
conCA1Rbm3 = np.squeeze(np.array(controlCA1Cells[rbm3Idx,:]))
expCA1Rbm3 = np.squeeze(np.array(experimentalCA1Cells[rbm3Idx,:]))
conCA3Rbm3 = np.squeeze(np.array(controlCA3Cells[rbm3Idx,:]))
expCA3Rbm3 = np.squeeze(np.array(experimentalCA3Cells[rbm3Idx,:]))
tStatCA1Rbm3, pValCA1Rbm3 = scipy_stats.ttest_ind(expCA1Rbm3, conCA1Rbm3)
tStatCA3Rbm3, pValCA3Rbm3 = scipy_stats.ttest_ind(expCA3Rbm3, conCA3Rbm3)
# use the maximum value from both regions as the max for scatter plot
displayMax = np.max(np.array([np.max(conCA1Rbm3), np.max(expCA1Rbm3),np.max(conCA3Rbm3), np.max(expCA3Rbm3)]))
tStatMax = np.max(np.array([tStatCA1Rbm3, tStatCA3Rbm3]))
# output image for each gene, with significance marked by asterisk next to gene name
if pValCA1Rbm3 < alphaFdr or pValCA3Rbm3 < alphaFdr:
    geneSig = '*'
else:
    geneSig = ''
if pValCA1Rbm3 < alphaFdr:
    ca1Sig = '*'
else:
    ca1Sig = ''
if pValCA3Rbm3 < alphaFdr:
    ca3Sig = '*'
else:
    ca3Sig = ''
ca1Rbm3TstatColors = np.empty_like(ca1Idx)
ca1Rbm3TstatColors[:] = tStatCA1Rbm3
ca3Rbm3TstatColors = np.empty_like(ca3Idx)
ca3Rbm3TstatColors[:] = tStatCA3Rbm3
# display t-statistic between data for CA1 and CA3 overlaid on first sample
ttestAxs1.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
ttestAxs1.scatter(sampleForDisplay['processedTissuePositionList'][ca1Idx,0],sampleForDisplay['processedTissuePositionList'][ca1Idx,1], cmap='seismic', c=ca1Rbm3TstatColors, vmax=4, vmin=-4)
axScatter = ttestAxs1.scatter(sampleForDisplay['processedTissuePositionList'][ca3Idx,0],sampleForDisplay['processedTissuePositionList'][ca3Idx,1], cmap='seismic', c=ca3Rbm3TstatColors, vmax=4, vmin=-4, s=3)
ttestAxs1.axis('off')
#bar([f'NSD, mean={np.mean(conGroup)}', f'SD, mean={np.mean(expGroup)}'], [np.mean(conGroup), np.mean(expGroup)], yerr=[scipy_stats.sem(conGroup), scipy_stats.sem(expGroup)], capsize=3, width=w, label=f'p={pVal}', color=[colorHCHex,colorMDDHex])
ttestAxs1.set_title(f'Rbm3{geneSig}, t-statistic SD > NSD\n CA1{ca1Sig}, CA3{ca3Sig}')
plt.colorbar(axScatter,fraction=0.02, pad=0.04)
# do the same for Gpr161
gpr161Idx = maleSamples[0]["geneList"].index('Gpr161')
conCA1Gpr161 = np.squeeze(np.array(controlCA1Cells[gpr161Idx,:]))
expCA1Gpr161 = np.squeeze(np.array(experimentalCA1Cells[gpr161Idx,:]))
conCA3Gpr161 = np.squeeze(np.array(controlCA3Cells[gpr161Idx,:]))
expCA3Gpr161 = np.squeeze(np.array(experimentalCA3Cells[gpr161Idx,:]))
tStatCA1Gpr161, pValCA1Gpr161 = scipy_stats.ttest_ind(expCA1Gpr161, conCA1Gpr161)
tStatCA3Gpr161, pValCA3Gpr161 = scipy_stats.ttest_ind(expCA3Gpr161, conCA3Gpr161)

# output image for each gene, with significance marked by asterisk next to gene name
if pValCA1Gpr161 < alphaFdr or pValCA3Gpr161 < alphaFdr:
    geneSig = '*'
else:
    geneSig = ''
if pValCA1Gpr161 < alphaFdr:
    ca1Sig = '*'
else:
    ca1Sig = ''
if pValCA3Gpr161 < alphaFdr:
    ca3Sig = '*'
else:
    ca3Sig = ''

ca1Gpr161TstatColors = np.empty_like(ca1Idx)
ca1Gpr161TstatColors[:] = tStatCA1Gpr161
ca3Gpr161TstatColors = np.empty_like(ca3Idx)
ca3Gpr161TstatColors[:] = tStatCA3Gpr161
# display t-statistic between data for CA1 and CA3 overlaid on first sample
ttestAxs2.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
ttestAxs2.scatter(sampleForDisplay['processedTissuePositionList'][ca1Idx,0],sampleForDisplay['processedTissuePositionList'][ca1Idx,1], cmap='seismic', c=ca1Gpr161TstatColors, vmax=4, vmin=-4)
axScatter = ttestAxs2.scatter(sampleForDisplay['processedTissuePositionList'][ca3Idx,0],sampleForDisplay['processedTissuePositionList'][ca3Idx,1], cmap='seismic', c=ca3Gpr161TstatColors, vmax=4, vmin=-4, s=3)
ttestAxs2.axis('off')
#bar([f'NSD, mean={np.mean(conGroup)}', f'SD, mean={np.mean(expGroup)}'], [np.mean(conGroup), np.mean(expGroup)], yerr=[scipy_stats.sem(conGroup), scipy_stats.sem(expGroup)], capsize=3, width=w, label=f'p={pVal}', color=[colorHCHex,colorMDDHex])
ttestAxs2.set_title(f'Gpr161{geneSig}, t-statistic SD > NSD\n CA1{ca1Sig}, CA3{ca3Sig}')
plt.colorbar(axScatter,fraction=0.02, pad=0.04)
plt.show()
# plt.savefig(os.path.join(derivatives, 'xenium_figure_k15cluster_umap_rbm3ttest_gpr161ttest.svg'), bbox_inches='tight', dpi=300)
# plt.savefig(os.path.join(derivatives, 'xenium_figure_k15cluster_umap_rbm3ttest_gpr161ttest.png'), bbox_inches='tight', dpi=300)
# plt.savefig(os.path.join(derivatives, 'xenium_figure_k15cluster_umap_rbm3ttest_gpr161ttest.pdf'), bbox_inches='tight', dpi=300)

#%% try to find clusters that correlated with different interneuron types
# interneuron_matrices = {}
# for i in sparseClusters:
#     clusterIdx = np.where(sampleToCluster['cluster_labels'] == i)[0]
#     for interneuron in interneuron_information.keys():
#         cluster_interneuron_matrix = sampleToCluster['geneMatrixLog2'].todense()[interneuron_information[interneuron]['geneIdx'],:]
#         cluster_interneuron_matrix = cluster_interneuron_matrix[:,clusterIdx]
#         interneuron_matrices[interneuron] = cluster_interneuron_matrix
#         cluster_interneuron_sum = np.squeeze(np.array(np.sum(cluster_interneuron_matrix, axis=0)))
#         plt.figure()
#         plt.imshow(sampleToCluster['tissueImageProcessed'],cmap='gray_r')
#         plt.scatter(sampleToCluster['processedTissuePositionList'][clusterIdx,0], sampleToCluster['processedTissuePositionList'][clusterIdx,1],c=cluster_interneuron_sum, cmap='Reds', s=5)
#         plt.title(f"Expression of {interneuron} in\n{sampleToCluster['sampleID']} cluster {i}.")
#         plt.axis('off')
#         plt.show()
        

# cellTypeGenesInSample.to_csv(os.path.join(derivatives,'cell_type_genes_in_xenium.csv'), index=False)


#%% should next do the same as cell above, but focused on interneurons

# plt.close('all')
fig,ax = plt.subplots(len(interneuron_information), len(sparseClusters), figsize=(21, 11))
meanZScoreMatrix = np.zeros([len(interneuron_information), len(sparseClusters)])
for i in range(len(sparseClusters)):
    for j in enumerate(interneuron_information):
        clusterIdx = np.where(sampleToCluster['cluster_labels'] == sparseClusters[i])[0]    
        geneMask = np.array(interneuron_information[j[1]]['geneIdx'], dtype='int32')
        cellTypeMatrix = geneMatrixZScore[:, clusterIdx]
        cellTypeMatrix = cellTypeMatrix[geneMask, :]
        cellTypeMatrixMeanZScore = np.squeeze(np.array(np.mean(cellTypeMatrix, axis=0)))
        meanZScore = np.mean(cellTypeMatrixMeanZScore)
        meanZScoreMatrix[j[0], i] = meanZScore
        ax[0, i].set_title(f'Cluster {sparseClusters[i]}')
        ax[j[0], 0].set_ylabel(j[1], rotation='horizontal', horizontalalignment='right')
        ax[j[0],i].imshow(sampleToCluster['tissueImageProcessed'],cmap='gray_r')
        scatterAx = ax[j[0],i].scatter(sampleToCluster['processedTissuePositionList'][clusterIdx,0], sampleToCluster['processedTissuePositionList'][clusterIdx,1],c=cellTypeMatrixMeanZScore, cmap='seismic', s=3, vmin=-2, vmax=2)
        ax[j[0],i].tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False,
            left=False,
            labelleft=False)
        # ax[i,j[0].set_title(f"Expression of {j[1]} in\n{sampleToCluster['sampleID']} cluster {sparseClusters[i]}.\nMean z-score = {meanZScore}")
plt.suptitle('Mean z-score for interneuron subtype marker genes')
cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
fig.colorbar(scatterAx, cax=cbar_ax, fraction=0.01, pad=0.04)
plt.show()
plt.savefig(os.path.join(derivatives, f'xenium_mean_z-score_for_interneuron_subtype_marker_genes_{sampleToCluster["sampleID"]}.pdf'), dpi=300)
# plt.colorbar(scatterAx)

#%% add cell types to the umap plotting

for sampleIdx in range(len(maleSamples)):
    cluster_regions = pd.read_csv(os.path.join(derivatives, 'clusterAssociationCsvs', f'{maleSamples[sampleIdx]["sampleID"]}_cluster_associations_cell_types.csv'), header=None)
    maleSamples[sampleIdx]['cluster_region'] = np.squeeze(np.array(cluster_regions[1]))
    
for sampleIdx in range(len(femaleSamples)):
    cluster_regions = pd.read_csv(os.path.join(derivatives, 'clusterAssociationCsvs', f'{femaleSamples[sampleIdx]["sampleID"]}_cluster_associations_cell_types.csv'), header=None)
    femaleSamples[sampleIdx]['cluster_region'] = np.squeeze(np.array(cluster_regions[1]))

#%% identify interneurons within neuron clusters
neuronIdx = np.where(sampleToCluster['cluster_region'] == 'neurons')[0]
clusterIdx = np.where(sampleToCluster['cluster_labels'] == neuronIdx[0])[0]  
geneMatrixZScore = sampleToCluster['geneMatrixLog2'].todense()
geneMatrixZScore = (geneMatrixZScore - np.mean(geneMatrixZScore, axis=0))/np.std(geneMatrixZScore, axis=0)
plt.close('all')

# identify the z scores of of each cell to identify if it is an interneuron
meanZScoreMatrixInterneurons = np.zeros([len(clusterIdx), len(interneuron_type)])
for i in enumerate(clusterIdx):
    print(i)
    for j in enumerate(interneuron_information): 
        geneMask = np.array(interneuron_information[j[1]]['geneIdx'], dtype='int32')
        cellTypeMatrix = np.squeeze(np.array(geneMatrixZScore[geneMask, i[1]]))
        # cellTypeMatrix = cellTypeMatrix[geneMask, :]
        # cellTypeMatrixMeanZScore = np.squeeze(np.array(np.mean(cellTypeMatrix, axis=0)))
        meanZScore = np.mean(cellTypeMatrix)
        meanZScoreMatrixInterneurons[i[0], j[0]] = meanZScore
# find the cell type with the highest z score to identify the type of interneuron
maxZScoreInterneurons = np.max(meanZScoreMatrixInterneurons, axis=1)
maxZScoreInterneuronsIdx = np.argmax(meanZScoreMatrixInterneurons, axis=1)

# identify as interneurons the cells which have a z-score above 1 
interneuronZScoreMask = maxZScoreInterneurons > 1
interneuronMaskIdx = clusterIdx[interneuronZScoreMask]

#%% perform umap plot on combined cells

reducer = umap.UMAP()

embeddingMale = reducer.fit_transform(np.array(allCellsMale).T)

#%% next need to use the mask for interneurons to assign to cluster label list
for sampleIdx in range(len(maleSamples)):
    neuronIdx = np.where(maleSamples[sampleIdx]['cluster_region'] == 'neurons')[0]
    clusterIdx = np.where(maleSamples[sampleIdx]['cluster_labels'] == neuronIdx[0])[0]  
    geneMatrixZScore = maleSamples[sampleIdx]['geneMatrixLog2'].todense()
    geneMatrixZScore = (geneMatrixZScore - np.mean(geneMatrixZScore, axis=0))/np.std(geneMatrixZScore, axis=0)
    plt.close('all')

    # identify the z scores of of each cell to identify if it is an interneuron
    meanZScoreMatrixInterneurons = np.zeros([len(clusterIdx), len(interneuron_type)])
    for i in enumerate(clusterIdx):
        print(i)
        for j in enumerate(interneuron_information): 
            geneMask = np.array(interneuron_information[j[1]]['geneIdx'], dtype='int32')
            cellTypeMatrix = np.squeeze(np.array(geneMatrixZScore[geneMask, i[1]]))
            # cellTypeMatrix = cellTypeMatrix[geneMask, :]
            # cellTypeMatrixMeanZScore = np.squeeze(np.array(np.mean(cellTypeMatrix, axis=0)))
            meanZScore = np.mean(cellTypeMatrix)
            meanZScoreMatrixInterneurons[i[0], j[0]] = meanZScore
    # find the cell type with the highest z score to identify the type of interneuron
    maxZScoreInterneurons = np.max(meanZScoreMatrixInterneurons, axis=1)
    maxZScoreInterneuronsIdx = np.argmax(meanZScoreMatrixInterneurons, axis=1)

    # identify as interneurons the cells which have a z-score above 1 
    interneuronZScoreMask = maxZScoreInterneurons > 1
    interneuronMaskIdx = clusterIdx[interneuronZScoreMask]

    maleSamples[sampleIdx]['cluster_labels'][interneuronMaskIdx] = 15
#%% try to identify specific top interneuron
sortZScoresInterneurons = np.sort(meanZScoreMatrixInterneurons, axis=1)
diffFirstAndSecondRank = sortZScoresInterneurons[:,4] - sortZScoresInterneurons[:,3]
sortMaxZScores = np.sort(maxZScoreInterneurons)
sortMaxSecondZScores = np.sort(sortZScoresInterneurons[:,3])
plt.close('all')
plt.plot(sortMaxZScores)
plt.plot(sortMaxSecondZScores)
plt.show()
#%% testing general colors from tab20 list
import matplotlib.cm as cm
plt.figure()
tab20c_colors = []
rangeToPlot = 20
for i in range(0,rangeToPlot+1,1):
    plt.scatter(i,i, c=cm.tab20(i), label=cm.tab20(i))
    tab20c_colors.append(cm.tab20(i))
plt.legend()
plt.show()
#%%
plt.close('all')
controlAllCells = np.empty([len(maleSamples[0]['geneList']),0])
### uses combination of tab20 and tab20c
ca1Color = np.squeeze(np.array([0.807843137254902, 0.427450980392157, 0.741176470588235, 1]))
ca2Color = np.squeeze(np.array([0.83921568627451, 0.380392156862745, 0.419607843137255, 1]))
ca3Color = np.squeeze(np.array([0.741176470588235, 0.619607843137255, 0.223529411764706, 1]))
ca4dgColor = np.squeeze(np.array([0.388235294117647	, 0.474509803921569, 0.223529411764706, 1]))
dgColor = np.squeeze(np.array([0.223529411764706, 0.231372549019608, 0.474509803921569, 1]))
astColor = np.squeeze(np.array([0.9019607843137255, 0.3333333333333333, 0.050980392156862744, 1.0]))
endColor = np.squeeze(np.array([0.6196078431372549, 0.792156862745098, 0.8823529411764706, 1.0]))
micColor = np.squeeze(np.array([0.19215686274509805, 0.6392156862745098, 0.32941176470588235, 1.0]))
neuColor = np.squeeze(np.array([0.5176470588235295, 0.23529411764705882, 0.2235294117647059, 1.]))
oliColor = np.squeeze(np.array([0.7372549019607844, 0.7411764705882353, 0.8627450980392157, 1.0]))
sparseColor = np.squeeze(np.array([0.7,0.7,0.7,1]))
regionList = ['CA1','CA2','CA3','DG/CA4','DG']

### uses tab10
ca1Color = cm.tab10(0)
ca2Color = cm.tab10(1)
ca3Color = cm.tab10(2)
ca4dgColor = cm.tab20(7)
dgColor = cm.tab10(4)
astColor = cm.tab20(1)
endColor = cm.tab20c(7)
micColor = cm.tab20(5)
neuColor = cm.Accent(5)
oliColor = cm.Set2(5)

plt.scatter(1,1, c=ca1Color)
plt.scatter(2,2, c=ca2Color)
plt.scatter(3,3, c=ca3Color)
plt.scatter(4,4, c=ca4dgColor)
plt.scatter(5,5, c=dgColor)
plt.scatter(6,6, c=astColor)
plt.scatter(7,7, c=endColor)
plt.scatter(8,8, c=micColor)
plt.scatter(9,9, c=neuColor)
plt.scatter(10,10, c=oliColor)

#%% create color list for single sample
sampleToClusterColors = np.empty([sampleToCluster['geneMatrixLog2'].shape[1],4])
ca1Idx = findRelevantClusters(sampleToCluster, 'CA1')
sampleToClusterColors[ca1Idx,:] = ca1Color
ca2Idx = findRelevantClusters(sampleToCluster, 'CA2')
sampleToClusterColors[ca2Idx,:] = ca2Color
ca3Idx = findRelevantClusters(sampleToCluster, 'CA3')
sampleToClusterColors[ca3Idx,:] = ca3Color
ca4dgIdx = findRelevantClusters(sampleToCluster, 'DG/CA4')
sampleToClusterColors[ca4dgIdx,:] = ca4dgColor
dgIdx = findRelevantClusters(sampleToCluster, 'DG')
sampleToClusterColors[dgIdx,:] = dgColor
astIdx = findRelevantClusters(sampleToCluster, 'astrocytes')
sampleToClusterColors[astIdx,:] = astColor
endIdx = findRelevantClusters(sampleToCluster, 'endothelial')
sampleToClusterColors[endIdx,:] = endColor
micIdx = findRelevantClusters(sampleToCluster, 'microglia')
sampleToClusterColors[micIdx,:] = micColor
neuIdx = findRelevantClusters(sampleToCluster, 'neurons')
sampleToClusterColors[neuIdx,:] = sparseColor
# sampleToClusterColors[neuIdx,:] = neuColor
oliIdx = findRelevantClusters(sampleToCluster, 'oligodendrocytes')
sampleToClusterColors[oliIdx,:] = oliColor
sparseIdx = findRelevantClusters(sampleToCluster, 'sparse')
sampleToClusterColors[sparseIdx,:] = sparseColor
intIdx = np.where(sampleToCluster['cluster_labels'] == 15)[0]
sampleToClusterColors[intIdx, :] = neuColor
# colorsAllCells = np.append(colorsAllCells, sampleColors, axis=0)

#%% perform plotting with all samples
colorsAllCells = np.empty([0,4])
for actSample in range(len(maleSamples)):
        controlAllCells = np.append(controlAllCells, maleSamples[actSample]['geneMatrixLog2'].todense(), axis=1)
        sampleColors = maleSamples[actSample]['cluster_colors']
        ca1Idx = findRelevantClusters(maleSamples[actSample], 'CA1')
        sampleColors[ca1Idx,:] = ca1Color
        ca2Idx = findRelevantClusters(maleSamples[actSample], 'CA2')
        sampleColors[ca2Idx,:] = ca2Color
        ca3Idx = findRelevantClusters(maleSamples[actSample], 'CA3')
        sampleColors[ca3Idx,:] = ca3Color
        ca4dgIdx = findRelevantClusters(maleSamples[actSample], 'DG/CA4')
        sampleColors[ca4dgIdx,:] = ca4dgColor
        dgIdx = findRelevantClusters(maleSamples[actSample], 'DG')
        sampleColors[dgIdx,:] = dgColor
        astIdx = findRelevantClusters(maleSamples[actSample], 'astrocytes')
        sampleColors[astIdx,:] = astColor
        endIdx = findRelevantClusters(maleSamples[actSample], 'endothelial')
        sampleColors[endIdx,:] = endColor
        micIdx = findRelevantClusters(maleSamples[actSample], 'microglia')
        sampleColors[micIdx,:] = micColor
        neuIdx = findRelevantClusters(maleSamples[actSample], 'neurons')
        sampleColors[neuIdx,:] = sparseColor
        # sampleColors[neuIdx,:] = neuColor
        oliIdx = findRelevantClusters(maleSamples[actSample], 'oligodendrocytes')
        sampleColors[oliIdx,:] = oliColor
        sparseIdx = findRelevantClusters(maleSamples[actSample], 'sparse')
        sampleColors[sparseIdx,:] = sparseColor
        intIdx = np.where(maleSamples[actSample]['cluster_labels'] == 15)[0]
        sampleColors[intIdx, :] = neuColor
        colorsAllCells = np.append(colorsAllCells, sampleColors, axis=0)
        
#%%
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Arial'
ca1Idx = findRelevantClusters(sampleForDisplay, 'CA1')
ca3Idx = findRelevantClusters(sampleForDisplay, 'CA3')
plt.close('all')
fig = plt.figure(figsize=(14,7))
clustAxs = plt.subplot2grid((4,7), (1,0), rowspan=2, colspan=2)
umapAxs = plt.subplot2grid((4,7), (0,2), colspan=3, rowspan=4)
ttestAxs1 = plt.subplot2grid((4,7), (0,5), rowspan=2, colspan=2)
ttestAxs2 = plt.subplot2grid((4,7), (2,5), rowspan=2, colspan=2)
plt.show()

clustAxs.imshow(sampleToCluster['tissueImageProcessed'],cmap='gray_r')
clustAxs.scatter(sampleToCluster['processedTissuePositionList'][:,0], sampleToCluster['processedTissuePositionList'][:,1],c=sampleToClusterColors, s=1)
clustAxs.axis('off')
clustAxs.set_title("Clustering, k=15",fontsize='medium', verticalalignment='top', fontfamily='serif',
fontweight='bold')
# create handles and patches to generate labeled legend
handles, labels = clustAxs.get_legend_handles_labels()
patch = mpatches.Patch(color=ca1Color, label='CA1')
handles.append(patch) 
patch = mpatches.Patch(color=ca2Color, label='CA2')
handles.append(patch) 
patch = mpatches.Patch(color=ca3Color, label='CA3')
handles.append(patch) 
patch = mpatches.Patch(color=ca4dgColor, label='CA4/DG')
handles.append(patch) 
patch = mpatches.Patch(color=dgColor, label='DG')
handles.append(patch) 
patch = mpatches.Patch(color=sparseColor, label='nonspecific cells')
handles.append(patch) 
# plot the legend
clustAxs.legend(handles=handles, loc='lower right',bbox_to_anchor=(1.1, 1.2))
# create handles and patches to generate labeled legend
handles, labels = clustAxs.get_legend_handles_labels()
patch = mpatches.Patch(color=ca1Color, label='CA1')
handles.append(patch) 
patch = mpatches.Patch(color=ca2Color, label='CA2')
handles.append(patch) 
patch = mpatches.Patch(color=ca3Color, label='CA3')
handles.append(patch) 
patch = mpatches.Patch(color=ca4dgColor, label='CA4/DG')
handles.append(patch) 
patch = mpatches.Patch(color=dgColor, label='DG')
handles.append(patch) 
patch = mpatches.Patch(color=astColor, label='astrocytes')
handles.append(patch) 
patch = mpatches.Patch(color=endColor, label='endothelial cells')
handles.append(patch) 
patch = mpatches.Patch(color=micColor, label='microglia')
handles.append(patch) 
patch = mpatches.Patch(color=neuColor, label='interneurons')
handles.append(patch) 
patch = mpatches.Patch(color=oliColor, label='oligodendrocytes')
handles.append(patch) 
### can comment out legend command for 'nonspecific cells' to make even # of rows
patch = mpatches.Patch(color=sparseColor, label='nonspecific cells')
handles.append(patch) 
clustAxs.legend(handles=handles, loc='lower right',bbox_to_anchor=(1.1, 1.2), ncols=2)
umapAxs.scatter(embedding[:,0], embedding[:,1], c=colorsAllCells, s=2)
umapAxs.set_title("UMAP of all samples", fontsize='medium', verticalalignment='top', fontfamily='serif',
fontweight='bold')
umapAxs.set_xticks([])
umapAxs.set_yticks([])
# plt.tight_layout()
# calculate t-statistics for Rbm3 and Gpr161
rbm3Idx = maleSamples[0]["geneList"].index('Rbm3')
conCA1Rbm3 = np.squeeze(np.array(controlCA1Cells[rbm3Idx,:]))
expCA1Rbm3 = np.squeeze(np.array(experimentalCA1Cells[rbm3Idx,:]))
conCA3Rbm3 = np.squeeze(np.array(controlCA3Cells[rbm3Idx,:]))
expCA3Rbm3 = np.squeeze(np.array(experimentalCA3Cells[rbm3Idx,:]))
tStatCA1Rbm3, pValCA1Rbm3 = scipy_stats.ttest_ind(expCA1Rbm3, conCA1Rbm3)
tStatCA3Rbm3, pValCA3Rbm3 = scipy_stats.ttest_ind(expCA3Rbm3, conCA3Rbm3)
# use the maximum value from both regions as the max for scatter plot
displayMax = np.max(np.array([np.max(conCA1Rbm3), np.max(expCA1Rbm3),np.max(conCA3Rbm3), np.max(expCA3Rbm3)]))
tStatMax = np.max(np.array([tStatCA1Rbm3, tStatCA3Rbm3]))
# output image for each gene, with significance marked by asterisk next to gene name
if pValCA1Rbm3 < alphaFdr or pValCA3Rbm3 < alphaFdr:
    geneSig = '*'
else:
    geneSig = ''
if pValCA1Rbm3 < alphaFdr:
    ca1Sig = '*'
else:
    ca1Sig = ''
if pValCA3Rbm3 < alphaFdr:
    ca3Sig = '*'
else:
    ca3Sig = ''
ca1Rbm3TstatColors = np.empty_like(ca1Idx)
ca1Rbm3TstatColors[:] = tStatCA1Rbm3
ca3Rbm3TstatColors = np.empty_like(ca3Idx)
ca3Rbm3TstatColors[:] = tStatCA3Rbm3
# display t-statistic between data for CA1 and CA3 overlaid on first sample
ttestAxs1.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
ttestAxs1.scatter(sampleForDisplay['processedTissuePositionList'][ca1Idx,0],sampleForDisplay['processedTissuePositionList'][ca1Idx,1], cmap='seismic', c=ca1Rbm3TstatColors, vmax=4, vmin=-4, s=2)
axScatter = ttestAxs1.scatter(sampleForDisplay['processedTissuePositionList'][ca3Idx,0],sampleForDisplay['processedTissuePositionList'][ca3Idx,1], cmap='seismic', c=ca3Rbm3TstatColors, vmax=4, vmin=-4, s=2)
ttestAxs1.axis('off')
#bar([f'NSD, mean={np.mean(conGroup)}', f'SD, mean={np.mean(expGroup)}'], [np.mean(conGroup), np.mean(expGroup)], yerr=[scipy_stats.sem(conGroup), scipy_stats.sem(expGroup)], capsize=3, width=w, label=f'p={pVal}', color=[colorHCHex,colorMDDHex])
ttestAxs1.set_title(f'Rbm3{geneSig}, t-statistic SD > NSD\n CA1{ca1Sig}, CA3{ca3Sig}',
                    fontsize='medium', verticalalignment='top', fontfamily='serif',
                    fontweight='bold')
plt.colorbar(axScatter,fraction=0.02, pad=0.04)
# do the same for Gpr161
gpr161Idx = maleSamples[0]["geneList"].index('Gpr161')
conCA1Gpr161 = np.squeeze(np.array(controlCA1Cells[gpr161Idx,:]))
expCA1Gpr161 = np.squeeze(np.array(experimentalCA1Cells[gpr161Idx,:]))
conCA3Gpr161 = np.squeeze(np.array(controlCA3Cells[gpr161Idx,:]))
expCA3Gpr161 = np.squeeze(np.array(experimentalCA3Cells[gpr161Idx,:]))
tStatCA1Gpr161, pValCA1Gpr161 = scipy_stats.ttest_ind(expCA1Gpr161, conCA1Gpr161)
tStatCA3Gpr161, pValCA3Gpr161 = scipy_stats.ttest_ind(expCA3Gpr161, conCA3Gpr161)

# output image for each gene, with significance marked by asterisk next to gene name
if pValCA1Gpr161 < alphaFdr or pValCA3Gpr161 < alphaFdr:
    geneSig = '*'
else:
    geneSig = ''
if pValCA1Gpr161 < alphaFdr:
    ca1Sig = '*'
else:
    ca1Sig = ''
if pValCA3Gpr161 < alphaFdr:
    ca3Sig = '*'
else:
    ca3Sig = ''

ca1Gpr161TstatColors = np.empty_like(ca1Idx)
ca1Gpr161TstatColors[:] = tStatCA1Gpr161
ca3Gpr161TstatColors = np.empty_like(ca3Idx)
ca3Gpr161TstatColors[:] = tStatCA3Gpr161
# display t-statistic between data for CA1 and CA3 overlaid on first sample
ttestAxs2.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
ttestAxs2.scatter(sampleForDisplay['processedTissuePositionList'][ca1Idx,0],sampleForDisplay['processedTissuePositionList'][ca1Idx,1], cmap='seismic', c=ca1Gpr161TstatColors, vmax=4, vmin=-4, s=2)
axScatter = ttestAxs2.scatter(sampleForDisplay['processedTissuePositionList'][ca3Idx,0],sampleForDisplay['processedTissuePositionList'][ca3Idx,1], cmap='seismic', c=ca3Gpr161TstatColors, vmax=4, vmin=-4, s=2)
ttestAxs2.axis('off')
#bar([f'NSD, mean={np.mean(conGroup)}', f'SD, mean={np.mean(expGroup)}'], [np.mean(conGroup), np.mean(expGroup)], yerr=[scipy_stats.sem(conGroup), scipy_stats.sem(expGroup)], capsize=3, width=w, label=f'p={pVal}', color=[colorHCHex,colorMDDHex])
ttestAxs2.set_title(f'Gpr161{geneSig}, t-statistic SD > NSD\n CA1{ca1Sig}, CA3{ca3Sig}',
                    fontsize='medium', verticalalignment='top', fontfamily='serif',
                    fontweight='bold')
plt.colorbar(axScatter,fraction=0.02, pad=0.04)
plt.show()
plt.savefig(os.path.join(derivatives, 'xenium_figure_k15cluster_umap_with_cell_types_rbm3ttest_gpr161ttest.svg'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(derivatives, 'xenium_figure_k15cluster_umap_with_cell_types_rbm3ttest_gpr161ttest.png'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(derivatives, 'xenium_figure_k15cluster_umap_with_cell_types_rbm3ttest_gpr161ttest.pdf'), bbox_inches='tight', dpi=300)
