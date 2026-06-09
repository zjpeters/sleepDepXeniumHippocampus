#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:48:31 2025

@author: zjpeters
"""
import os
if os.path.exists(os.path.join('/','media','zjpeters','Expansion')):
    externalDrivePath = os.path.join('/','media','zjpeters','Expansion')
else:
    externalDrivePath = os.path.join('D:',os.sep)

from matplotlib import pyplot as plt
import numpy as np
import sys
if os.path.exists(os.path.join("/", "home", "zjpeters", "Documents", "stanly", "code")):
    sys.path.insert(0, "/home/zjpeters/Documents/stanly/code")
else:
    sys.path.insert(0, os.path.join('C:',os.sep, 'Users','onyh19ug', 'Documents', 'STANLY','code'))
import stanly
import pandas as pd
import scipy.stats as scipy_stats
import umap
import matplotlib.patches as mpatches
import upsetplot
if os.path.exists(os.path.join('/', 'home', 'zjpeters', 'Documents', 'stanly', 'code')):
    stanlyLoc = os.path.join('/', 'home', 'zjpeters', 'Documents', 'stanly', 'code')
else:
    # windows workstation locations
    stanlyLoc = os.path.join('C:',os.sep, 'Users','onyh19ug', 'Documents', 'STANLY','code')
# from adjustText import adjust_text

# function to restrict cells to only those clustered to a particular label

def findRelevantClusters(processedSample, cluster_region):
    matchingClusters = np.where(processedSample['cluster_region'] == cluster_region)[0]
    clusterMask = []
    for cellIdx in range(len(processedSample['cluster_labels'])):
        # since it's possible that more than one cluster correspond to cluster of interest, use for loop
        for clusterIdx in matchingClusters:
            if processedSample['cluster_labels'][cellIdx] == clusterIdx:
                clusterMask.append(cellIdx)
    return clusterMask

# setting runtime parameters for pyplot plotting    
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['text.usetex'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 6

rawdata = os.path.join(externalDrivePath,'sleepDepXenium','rawdata')
derivatives = os.path.join(externalDrivePath,'sleepDepXeniumHippocampus','derivatives')
figureFolder = os.path.join(externalDrivePath,'sleepDepXeniumHippocampus','writing','figures')
#%% put together color list

cellsOfInterest = ['CA1', 'CA2', 'CA3', 'DG', 'DG/CA4', 'astrocytes', 'endothelial', 'microglia', 'neurons', 'oligodendrocytes', 'sparse']
### uses "Normal, 12 colors" from: https://tsitsul.in/blog/coloropt/
dgColor = np.squeeze(np.array([235, 172, 35, 255]))/255
ca2Color = np.squeeze(np.array([184, 0, 88, 255]))/255
ca3Color = np.squeeze(np.array([0, 140, 249, 255]))/255
ca4dgColor = np.squeeze(np.array([0, 110, 0, 255]))/255
ca1Color = np.squeeze(np.array([0, 187, 173, 255]))/255
astColor = np.squeeze(np.array([209, 99, 230, 255]))/255
endColor = np.squeeze(np.array([178, 69, 2, 255]))/255
micColor = np.squeeze(np.array([255, 146, 135, 255]))/255
neuColor = np.squeeze(np.array([89, 84, 214, 255]))/255
oliColor = np.squeeze(np.array([0, 198, 248, 255]))/255
extraColor1 = np.squeeze(np.array([135, 133, 0, 255]))/255
extraColor2 = np.squeeze(np.array([0, 167, 108, 255]))/255
sparseColor = np.squeeze(np.array([189, 189, 189, 255]))/255

ca1ColorOp = np.squeeze(np.array([0, 187, 173, 40]))/255
ca2ColorOp = np.squeeze(np.array([184, 0, 88, 40]))/255
ca3ColorOp = np.squeeze(np.array([0, 140, 249, 40]))/255
dgColorOp = np.squeeze(np.array([235, 172, 35, 40]))/255
ca4dgColorOp = np.squeeze(np.array([0, 110, 0, 40]))/255
astColorOp = np.squeeze(np.array([209, 99, 230, 40]))/255
endColorOp = np.squeeze(np.array([178, 69, 2, 40]))/255
micColorOp = np.squeeze(np.array([255, 146, 135, 40]))/255
neuColorOp = np.squeeze(np.array([89, 84, 214, 40]))/255
oliColorOp = np.squeeze(np.array([0, 198, 248, 40]))/255

cellsOfInterestColorList = [ca1Color, ca2Color, ca3Color, dgColor, ca4dgColor, astColor, endColor, micColor, neuColor, oliColor, sparseColor]

#%% load newly processed samples and check selection

locOfTsvFile = os.path.join(externalDrivePath,'sleepDepXeniumHippocampus', 'participants.tsv')
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
    

#%% create dictionary with information about different interneuron types
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
identify genes present in allen data that are also present in:
    "Brain Cell Type Specific Gene Expression and Co-expression Network Architectures"
    https://www.nature.com/articles/s41598-018-27293-5
"""
cellTypeSpreadsheetLocation = os.path.join(stanlyLoc,'data','cellTypeMarkerGeneInfo','Brain Cell Type Specific Gene Expression and Co-expression Network Architectures_41598_2018_27293_MOESM2_ESM_mouse_specificity.csv')
cellTypeGeneExpressionList = pd.read_csv(cellTypeSpreadsheetLocation)
# identify overlapping genes
interneuronGenesInSample = dict.fromkeys(["Pvalb", "Sst", "Vip", "Sncg", "Lamp5"])
for i in interneuronGenesInSample.keys():
    interneuronGenesInSample[i] = dict.fromkeys(['geneList', 'geneIdx'])
for interneuron_type in interneuronGenesInSample.keys():
    interneuron_gene_idx = []
    interneuron_gene_list = []
    for gene in interneuron_information[interneuron_type]['geneList']:
        try:
            geneIdx = processedSamples[0]['geneList'].index(gene)
            interneuron_gene_list.append(gene)
            interneuron_gene_idx.append(geneIdx)
        except ValueError:
            print('Gene not in list')
    interneuronGenesInSample[interneuron_type]['geneIdx'] = np.array(interneuron_gene_idx)
    interneuronGenesInSample[interneuron_type]['geneList'] = interneuron_gene_list
    
cellTypeCasefoldList = []
for gene in cellTypeGeneExpressionList['gene']:
    cellTypeCasefoldList.append(gene.casefold())

sampleCasefoldList = []
for gene in processedSamples[0]['geneList']:
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
            geneIdx = sampleCasefoldList.index(j.casefold())
            singleCellTypeGeneList.append([processedSamples[0]['geneList'][geneIdx], geneIdx])
            cellTypeGeneLists[i] = np.array(singleCellTypeGeneList)

        except ValueError:
            # code above should work well, though might need to consider if there
            # are situations where a casefold gene name would lead to duplicates
            print('Gene not found')

for interneuron_type in interneuron_information.keys():
    singleCellTypeGeneList = []
    for gene in interneuron_information[interneuron_type]['geneList']:
        try:
            geneIdx = processedSamples[0]['geneList'].index(gene)
            singleCellTypeGeneList.append([gene, geneIdx])
            cellTypeGeneLists[interneuron_type] = np.array(singleCellTypeGeneList)
        except ValueError:
            print('Gene not in list')

writer = pd.ExcelWriter(os.path.join(derivatives, 'genes_for_cell_type_ID.xlsx'))

for cellType in cellTypeGeneLists.keys():
    cellTypeDF = pd.DataFrame(cellTypeGeneLists[cellType])
    if cellType == 'DG/CA4':
        cellTypeDF.to_excel(writer, sheet_name='DG-CA4', index=False)
    else:
        cellTypeDF.to_excel(writer, sheet_name=cellType, index=False)
writer.close()
#%% generate male samples
maleSamples = []
maleSamplesIdx = [0,1,2,3,4,5]
for i in maleSamplesIdx:
    maleSamples.append(processedSamples[i])

designMatrix = [0,1,0,1,0,1]
#%% add cell types to the umap plotting

for sampleIdx in range(len(maleSamples)):
    cluster_regions = pd.read_csv(os.path.join(derivatives, 'clusterAssociationCsvs', f'{maleSamples[sampleIdx]["sampleID"]}_cluster_associations_cell_types.csv'), header=None)
    maleSamples[sampleIdx]['cluster_region'] = np.squeeze(np.array(cluster_regions[1]))
    sampleColors = processedSamples[sampleIdx]['cluster_colors']
    ca1Idx = findRelevantClusters(processedSamples[sampleIdx], 'CA1')
    sampleColors[ca1Idx,:] = ca1Color
    ca2Idx = findRelevantClusters(processedSamples[sampleIdx], 'CA2')
    sampleColors[ca2Idx,:] = ca2Color
    ca3Idx = findRelevantClusters(processedSamples[sampleIdx], 'CA3')
    sampleColors[ca3Idx,:] = ca3Color
    ca4dgIdx = findRelevantClusters(processedSamples[sampleIdx], 'DG/CA4')
    sampleColors[ca4dgIdx,:] = ca4dgColor
    dgIdx = findRelevantClusters(processedSamples[sampleIdx], 'DG')
    sampleColors[dgIdx,:] = dgColor
    astIdx = findRelevantClusters(processedSamples[sampleIdx], 'astrocytes')
    sampleColors[astIdx,:] = astColor
    endIdx = findRelevantClusters(processedSamples[sampleIdx], 'endothelial')
    sampleColors[endIdx,:] = endColor
    micIdx = findRelevantClusters(processedSamples[sampleIdx], 'microglia')
    sampleColors[micIdx,:] = micColor
    neuIdx = findRelevantClusters(processedSamples[sampleIdx], 'neurons')
    sampleColors[neuIdx,:] = neuColor
    # sampleColors[neuIdx,:] = neuColor
    oliIdx = findRelevantClusters(processedSamples[sampleIdx], 'oligodendrocytes')
    sampleColors[oliIdx,:] = oliColor
    sparseIdx = findRelevantClusters(processedSamples[sampleIdx], 'sparse')
    sampleColors[sparseIdx,:] = sparseColor
    intIdx = np.where(processedSamples[sampleIdx]['cluster_labels'] == 15)[0]
    sampleColors[intIdx, :] = neuColor
    processedSamples[sampleIdx]['normalizedColors'] = sampleColors
    
#%% create lists of all cells and all clusters for use in later plotting
plt.close('all')

allCellsMale = np.empty([len(maleSamples[0]['geneList']),0])
allClustersMale = np.empty(0, dtype='str')
sdNSDIdxMale = np.empty(0, dtype='int32')
sampleListMale = np.empty(0,dtype='int32')
allCoordinatesMale = np.empty([0,2])
for actSample in range(len(maleSamples)):
    allCellsMale = np.append(allCellsMale, maleSamples[actSample]['geneMatrixLog2'].todense(), axis=1)
    clusterListMale = [] #np.empty([len(maleSamples[actSample]['cluster_labels']),1], dtype='str')
    for cluster in maleSamples[actSample]['cluster_labels'].to_numpy():
        clusterListMale.append(maleSamples[actSample]['cluster_region'][cluster])
    allClustersMale = np.append(allClustersMale, clusterListMale)
    # binary distinction of each cell for sd or nsd
    if designMatrix[actSample] == 0:
        sdNSDIdxMale = np.append(sdNSDIdxMale, np.zeros(len(maleSamples[actSample]['processedTissuePositionList']), dtype='int32').T)
    elif designMatrix[actSample] == 1:
        sdNSDIdxMale = np.append(sdNSDIdxMale, np.ones(len(maleSamples[actSample]['processedTissuePositionList']), dtype='int32').T)
    sampleListMale = np.append(sampleListMale, np.zeros(len(maleSamples[actSample]['processedTissuePositionList']), dtype='int32').T + actSample)
    allCoordinatesMale = np.append(allCoordinatesMale, maleSamples[actSample]['processedTissuePositionList'], axis=0)
    sampleColorsMale = maleSamples[actSample]['cluster_colors']
clusterListMale = np.array(clusterListMale, dtype='str')
#%% assign colors to cells
# controlAllCellsMale = np.empty([len(maleSamples[0]['geneList']),0])
colorsAllCellsMale = np.empty([0,4])
for actSample in range(len(maleSamples)):
    # controlAllCellsMale = np.append(controlAllCellsMale, maleSamples[actSample]['geneMatrixLog2'].todense(), axis=1)
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
    sampleColors[neuIdx,:] = neuColor
    # sampleColors[neuIdx,:] = neuColor
    oliIdx = findRelevantClusters(maleSamples[actSample], 'oligodendrocytes')
    sampleColors[oliIdx,:] = oliColor
    sparseIdx = findRelevantClusters(maleSamples[actSample], 'sparse')
    sampleColors[sparseIdx,:] = sparseColor
    intIdx = np.where(maleSamples[actSample]['cluster_labels'] == 15)[0]
    sampleColors[intIdx, :] = neuColor
    colorsAllCellsMale = np.append(colorsAllCellsMale, sampleColors, axis=0)


#%% plot all samples with whole images but only the hippocampal cells along with just tissue
# supplemental figure to show the clustering of all samples

marker_size = 0.1
plt.close('all')
# rotationInfo = stanly.rotateTissuePoints(sample['tissuePositionList'], sample['imageData'], 270)
fig, ax = plt.subplots(4,3, figsize=(7.2,4.8))

# plot NSD first
for i in [0,2,4]:
    nC = int(i/2)
    sampleForDisplay = processedSamples[i]
    sample = stanly.importXeniumData(os.path.join(rawdata, experiment['sample-id'][i]))
    barcodeIdx = []
    for barcode in sampleForDisplay['barcodeList']:
        barcodeIdx.append(sample['barcodeList'].index(barcode))
    sample['hippMask'] = np.array(barcodeIdx, dtype='int32')
    maleSamples[i]['hippMask'] = sample['hippMask']
    rotationInfo = stanly.rotateTissuePoints(sample['tissuePositionList'], sample['imageData'], experiment['rotation'][i])
    # plot tissue
    ax[0, nC].imshow(rotationInfo[1], cmap='gray_r')
    ax[0, nC].axis('off')
    ax[0, nC].set_title(sample['sampleID'])
    # plot tissue with clusters
    ax[1, nC].imshow(rotationInfo[1], cmap='gray_r')
    ax[1, nC].scatter(rotationInfo[0][sample['hippMask'],0], rotationInfo[0][sample['hippMask'],1], c=processedSamples[i]['normalizedColors'], alpha=1, s=marker_size, linewidth=0)
    ax[1, nC].axis('off')
    # ax[1, nC].set_title(sample['sampleID'])

# plot SD first
for i in enumerate([1,3,5]):
    nC = i[0]
    sampleForDisplay = processedSamples[i[1]]
    sample = stanly.importXeniumData(os.path.join(rawdata, experiment['sample-id'][i[1]]))
    barcodeIdx = []
    for barcode in sampleForDisplay['barcodeList']:
        barcodeIdx.append(sample['barcodeList'].index(barcode))
    sample['hippMask'] = np.array(barcodeIdx, dtype='int32')
    maleSamples[i[1]]['hippMask'] = sample['hippMask']
    rotationInfo = stanly.rotateTissuePoints(sample['tissuePositionList'], sample['imageData'], experiment['rotation'][i[1]])
    # plot tissue
    ax[2, nC].imshow(rotationInfo[1], cmap='gray_r')
    ax[2, nC].axis('off')
    ax[2, nC].set_title(sample['sampleID'])
    # plot tissue with clusters
    ax[3, nC].imshow(rotationInfo[1], cmap='gray_r')
    ax[3, nC].scatter(rotationInfo[0][sample['hippMask'],0], rotationInfo[0][sample['hippMask'],1], c=processedSamples[i[1]]['normalizedColors'], alpha=1, s=marker_size, linewidth=0)
    ax[3, nC].axis('off')
    # ax[3, nC].set_title(sample['sampleID'])
    
plt.show()
# output pdf and svg
plt.savefig(os.path.join(figureFolder, f'supp_figure01_opt2_xenium_slice_clustering_all_samples_marker_size_{marker_size}_resized.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(figureFolder, f'supp_figure01_opt2_xenium_slice_clustering_all_samples_marker_size_{marker_size}_resized.svg'), bbox_inches='tight', dpi=300)
plt.close('all')
#%% plot all samples with whole images but only the hippocampal cells
# alternate form of supplemental figure showing clustering on samples

marker_size = 0.1
plt.close('all')
# rotationInfo = stanly.rotateTissuePoints(sample['tissuePositionList'], sample['imageData'], 270)
fig, ax = plt.subplots(2,3, figsize=(7.2,3.6))
nR = 0
nC = 0
for i in range(len(maleSamples)):
    sampleForDisplay = processedSamples[i]
    sample = stanly.importXeniumData(os.path.join(rawdata, experiment['sample-id'][i]))
    barcodeIdx = []
    for barcode in sampleForDisplay['barcodeList']:
        barcodeIdx.append(sample['barcodeList'].index(barcode))
    sample['hippMask'] = np.array(barcodeIdx, dtype='int32')
    maleSamples[i]['hippMask'] = sample['hippMask']
    rotationInfo = stanly.rotateTissuePoints(sample['tissuePositionList'], sample['imageData'], experiment['rotation'][i])
    # a
    ax[nR, nC].imshow(rotationInfo[1], cmap='gray_r')
    ax[nR, nC].scatter(rotationInfo[0][sample['hippMask'],0], rotationInfo[0][sample['hippMask'],1], c=processedSamples[i]['normalizedColors'], alpha=1, s=marker_size, linewidth=0)
    ax[nR, nC].axis('off')
    ax[nR, nC].set_title(sample['sampleID'])
    if nC < 2:
        nC += 1
    else:
        nR += 1
        nC = 0
plt.show()
# output pdf and svg
# plt.savefig(os.path.join(figureFolder, f'supp_figure01_xenium_slice_clustering_all_samples_marker_size_{marker_size}_resized.pdf'), bbox_inches='tight', dpi=300)
# plt.savefig(os.path.join(figureFolder, f'supp_figure01_xenium_slice_clustering_all_samples_marker_size_{marker_size}_resized.svg'), bbox_inches='tight', dpi=300)
# plt.close('all')
#%% perform umap plot on combined cells

reducer = umap.UMAP(random_state=42)

embeddingMale = reducer.fit_transform(np.array(allCellsMale).T)

#%% create a plot of umap for SD, umap for NSD, and UMAP for all cells in male samples
# supplemental figure showing that there is no obvious difference between sd and nsd in clustering
plt.close('all')

fig = plt.figure(figsize=(7.2,4))
umapAxsNSD = plt.subplot2grid((4,9), (0,0), colspan=3, rowspan=4)
umapAxsSD = plt.subplot2grid((4,9), (0,3), colspan=3, rowspan=4)
umapAxsAllCells = plt.subplot2grid((4,9), (0,6), colspan=3, rowspan=4)
# plot the NSD samples
nsdIdx = np.where(sdNSDIdxMale == 0)[0]
umapAxsNSD.scatter(embeddingMale[nsdIdx,0], embeddingMale[nsdIdx,1], c=colorsAllCellsMale[nsdIdx,:], s=marker_size, linewidth=0)
umapAxsNSD.set_title("UMAP of male NSD samples")
umapAxsNSD.set_xticks([])
umapAxsNSD.set_yticks([])

# plot the SD samples
sdIdx = np.where(sdNSDIdxMale == 1)[0]
umapAxsSD.scatter(embeddingMale[sdIdx,0], embeddingMale[sdIdx,1], c=colorsAllCellsMale[sdIdx,:], s=marker_size, linewidth=0)
umapAxsSD.set_title("UMAP of male SD samples")
umapAxsSD.set_xticks([])
umapAxsSD.set_yticks([])

# plot the legend
# create handles and patches to generate labeled legend
handles, labels = umapAxsSD.get_legend_handles_labels()
patch = mpatches.Patch(color=ca1Color, label='CA1')
handles.append(patch) 
patch = mpatches.Patch(color=ca2Color, label='CA2')
handles.append(patch) 
patch = mpatches.Patch(color=ca3Color, label='CA3')
handles.append(patch) 
patch = mpatches.Patch(color=dgColor, label='DG')
handles.append(patch) 
patch = mpatches.Patch(color=ca4dgColor, label='DG hilus')
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

umapAxsSD.legend(handles=handles, ncols=2, fontsize=5)

# plot all cells
umapAxsAllCells.scatter(embeddingMale[:,0], embeddingMale[:,1], c=colorsAllCellsMale, s=marker_size, linewidth=0)
umapAxsAllCells.set_title("UMAP of all male samples")
umapAxsAllCells.set_xticks([])
umapAxsAllCells.set_yticks([])

plt.show()

### uncomment below to save new figure pdf and svg
plt.savefig(os.path.join(figureFolder, 'supp_figure_k15cluster_umap_with_sd_nsd_split_males_resized.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(figureFolder, 'supp_figure_k15cluster_umap_with_sd_nsd_split_males_resized.svg'), bbox_inches='tight', dpi=300)
plt.close('all')
#%% create figure 1 image for all samples with schematic image
# figure 01, showing clustering and tissue image of a single slice along with umap of all samples
schematic_img = plt.imread(os.path.join(figureFolder, 'biorender_schematic.png'))
sample = stanly.importXeniumData(os.path.join(rawdata, experiment['sample-id'][0]))
rotationInfo = stanly.rotateTissuePoints(sample['tissuePositionList'], sample['imageData'], 270)

plt.close('all')
fig = plt.figure(figsize=(6.14,6.69))
schematicImage = plt.subplot2grid((6,7), (0,0), colspan=7, rowspan=2)
tissueImage = plt.subplot2grid((6,7), (2,0), colspan=3, rowspan=2)
clusterImage = plt.subplot2grid((6,7), (4,0), colspan=3, rowspan=2)
umapAxsAllCells = plt.subplot2grid((6,7), (2,3), colspan=4, rowspan=4)

# plot schematic image from biorender
schematicImage.imshow(schematic_img)
schematicImage.set_xticks([])
schematicImage.set_yticks([])
schematicImage.spines['top'].set_visible(False)
schematicImage.spines['bottom'].set_visible(False)
schematicImage.spines['left'].set_visible(False)
schematicImage.spines['right'].set_visible(False)
# plot the tissue image from sample
tissueImage.imshow(rotationInfo[1], cmap='gray_r')
tissueImage.set_xticks([])
tissueImage.set_yticks([])

# plot the clustering results on top of the image
clusterImage.imshow(rotationInfo[1], cmap='gray_r')
clusterImage.scatter(rotationInfo[0][maleSamples[0]['hippMask'],0], rotationInfo[0][maleSamples[0]['hippMask'],1], c=maleSamples[0]['cluster_colors'], alpha=1, s=marker_size, linewidth=0)
clusterImage.set_xticks([])
clusterImage.set_yticks([])

# plot all cells
umapAxsAllCells.scatter(embeddingMale[:,0], embeddingMale[:,1], c=colorsAllCellsMale, s=3, linewidth=0)
umapAxsAllCells.set_xticks([])
umapAxsAllCells.set_yticks([])

# plot the legend
# create handles and patches to generate labeled legend
handles, labels = umapAxsAllCells.get_legend_handles_labels()
patch = mpatches.Patch(color=ca1Color, label='CA1')
handles.append(patch) 
patch = mpatches.Patch(color=ca2Color, label='CA2')
handles.append(patch) 
patch = mpatches.Patch(color=ca3Color, label='CA3')
handles.append(patch) 
patch = mpatches.Patch(color=dgColor, label='DG')
handles.append(patch) 
patch = mpatches.Patch(color=ca4dgColor, label='DG hilus')
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

umapAxsAllCells.legend(handles=handles, ncols=2)

plt.show()
# output pdf and svg
plt.savefig(os.path.join(figureFolder, f'figure01_xenium_slice_clustering_and_umap_marker_size_{marker_size}_with_schematic_resized.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(figureFolder, f'figure01_xenium_slice_clustering_and_umap_marker_size_{marker_size}_with_schematic_resized.svg'), bbox_inches='tight', dpi=300)

#%% create figure 1 image for all samples
# figure 01, showing clustering and tissue image of a single slice along with umap of all samples
plt.close('all')
sample = stanly.importXeniumData(os.path.join(rawdata, experiment['sample-id'][0]))
rotationInfo = stanly.rotateTissuePoints(sample['tissuePositionList'], sample['imageData'], 270)

fig = plt.figure(figsize=(7.2,5.89))
tissueImage = plt.subplot2grid((4,7), (0,0), colspan=3, rowspan=2)
clusterImage = plt.subplot2grid((4,7), (2,0), colspan=3, rowspan=2)
umapAxsAllCells = plt.subplot2grid((4,7), (0,3), colspan=4, rowspan=4)
# plot the tissue image from sample
tissueImage.imshow(rotationInfo[1], cmap='gray_r')
tissueImage.set_xticks([])
tissueImage.set_yticks([])

# plot the clustering results on top of the image
clusterImage.imshow(rotationInfo[1], cmap='gray_r')
clusterImage.scatter(rotationInfo[0][maleSamples[0]['hippMask'],0], rotationInfo[0][maleSamples[0]['hippMask'],1], c=maleSamples[0]['cluster_colors'], alpha=1, s=marker_size, linewidth=0)
clusterImage.set_xticks([])
clusterImage.set_yticks([])

# plot all cells
umapAxsAllCells.scatter(embeddingMale[:,0], embeddingMale[:,1], c=colorsAllCellsMale, s=3, linewidth=0)
umapAxsAllCells.set_xticks([])
umapAxsAllCells.set_yticks([])

# plot the legend
# create handles and patches to generate labeled legend
handles, labels = umapAxsAllCells.get_legend_handles_labels()
patch = mpatches.Patch(color=ca1Color, label='CA1')
handles.append(patch) 
patch = mpatches.Patch(color=ca2Color, label='CA2')
handles.append(patch) 
patch = mpatches.Patch(color=ca3Color, label='CA3')
handles.append(patch) 
patch = mpatches.Patch(color=dgColor, label='DG')
handles.append(patch) 
patch = mpatches.Patch(color=ca4dgColor, label='DG hilus')
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

umapAxsAllCells.legend(handles=handles, ncols=2)

plt.show()
# output pdf and svg
# plt.savefig(os.path.join(figureFolder, f'figure01_xenium_slice_clustering_and_umap_marker_size_{marker_size}_resized.pdf'), bbox_inches='tight', dpi=300)
# plt.savefig(os.path.join(figureFolder, f'figure01_xenium_slice_clustering_and_umap_marker_size_{marker_size}_resized.svg'), bbox_inches='tight', dpi=300)

#%% perform t-test on each region/cell type
# use the first sample as the one to plot to
"""
this should be moved into the analysis script and this code should just load the 
p-values and t-statistics from a file
"""
desiredPval = 0.05
alphaFdr = 1 - np.power((1 - desiredPval),(1/(len(maleSamples[0]["geneList"])*len(cellsOfInterest))))
plt.close('all')
sigGenesPerCells = {}

# prepare lists for BH fdr
tStatList = np.empty([len(sampleForDisplay['geneList']), len(cellsOfInterest)])
pValList = np.empty([len(sampleForDisplay['geneList']), len(cellsOfInterest)])
foldChangeList = np.empty([len(sampleForDisplay['geneList']), len(cellsOfInterest)])
logFCList = np.empty([len(sampleForDisplay['geneList']), len(cellsOfInterest)])
sigColorList = np.empty([len(sampleForDisplay['geneList']), len(cellsOfInterest)], dtype='int32')
# loop over each of the regions/cell types and perform a t-test and plot results
for regionN, region in enumerate(cellsOfInterest):
    regionCellsToPlot = findRelevantClusters(sampleForDisplay, region)
    regionIdx = np.where(allClustersMale == region)[0]
    sdNSDRegionIdx = sdNSDIdxMale[regionIdx]
    regionCells = allCellsMale[:, regionIdx]
    nsdCells = regionCells[:, sdNSDRegionIdx == 0]
    sdCells = regionCells[:, sdNSDRegionIdx == 1]
    sigGenesPerCells[region] = []
    for gene in range(len(sampleForDisplay['geneList'])):
        tStat, pVal = scipy_stats.ttest_ind(np.squeeze(np.array(sdCells[gene,:]))**2 - 1, np.squeeze(np.array(nsdCells[gene,:]))**2 - 1)
        tStatList[gene, regionN] = tStat
        pValList[gene, regionN] = pVal
        # foldChange = np.mean(sdCells[gene,:]) - np.mean(nsdCells[gene,:])
        foldChange = np.mean(sdCells[gene,:]) / np.mean(nsdCells[gene,:])
        logFC = np.log2(np.mean(np.squeeze(np.array(sdCells[gene,:]))**2 - 1) / np.mean(np.squeeze(np.array(nsdCells[gene,:]))**2 - 1))
        # logFC = np.mean(sdCells[gene,:]) - np.mean(nsdCells[gene,:])
        # if foldChange < 1:
        #     foldChange = -1/foldChange
        #     logFC = -1/logFC
        foldChangeList[gene, regionN] = foldChange
        logFCList[gene, regionN] = logFC
        if pVal < alphaFdr:
            nsdColor = np.empty([len(regionCellsToPlot)])
            nsdColor[:] = np.mean(nsdCells[gene,:]) 
            sdColor = np.empty([len(regionCellsToPlot)])
            sdColor[:] = np.mean(sdCells[gene,:]) 
            tStatColor = np.empty([len(regionCellsToPlot)])
            tStatColor[:] = tStat
            fig, ax = plt.subplots(1,3)
            ax[0].imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
            ax[0].scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=nsdColor, cmap='Reds', s=2, vmin=0, vmax=4)
            ax[0].set_title(f'Mean of NSD \n mean={np.mean(nsdCells[gene,:])}')
            ax[0].axis('off')
            ax[1].imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
            ax[1].scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=sdColor, cmap='Reds', s=2, vmin=0, vmax=4)
            ax[1].set_title(f'Mean of SD \n mean={np.mean(sdCells[gene,:])}')
            ax[1].axis('off')
            ax[2].imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
            ax[2].scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=tStatColor, cmap='seismic', s=2, vmin=-4, vmax=4)
            ax[2].set_title(f'T-statistic for SD > NSD \n p-value={pVal}')
            ax[2].axis('off')
            if region == "DG/CA4":
                plt.suptitle(f'T-statistic for gene: {sampleForDisplay["geneList"][gene]} \n in DG hilus \n fold-change={foldChange}')
            else:
                plt.suptitle(f'T-statistic for gene: {sampleForDisplay["geneList"][gene]} \n in {region} \n fold-change={foldChange}')
            plt.show()
            if region == "DG/CA4":
                plt.savefig(os.path.join(derivatives, 'clusterDEGsMales', f'ttest_nsd_sd_males_{sampleForDisplay["geneList"][gene]}_in_DG-CA4.png'), bbox_inches='tight', dpi=300)
            else:
                plt.savefig(os.path.join(derivatives, 'clusterDEGsMales', f'ttest_nsd_sd_males_{sampleForDisplay["geneList"][gene]}_in_{region}.png'), bbox_inches='tight', dpi=300)
            plt.close()
            sigGenesPerCells[region].append(sampleForDisplay["geneList"][gene])
            if logFC < 0:
                sigColorList[gene, regionN] = -1# '#1f77b4'
            if logFC > 0:
                sigColorList[gene, regionN] = 1# '#FF0000'
        else:
            sigColorList[gene, regionN] = 0 #'#808080'
sigColorList = np.array(sigColorList)

#%% perform BH fdr correction
q = 0.05
# writer = pd.ExcelWriter(os.path.join(derivatives, 'degs_per_cell-type_and_regions_male_SD_BH.xlsx'))
bhCorrList = np.empty([len(sampleForDisplay['geneList']), len(cellsOfInterest)])
for cellTypeIdx, cellType in enumerate(cellsOfInterest):
    regionIdx = np.where(allClustersMale == cellType)[0]
    regionCells = allCellsMale[:, regionIdx]
    sortedPvals = np.sort(pValList[:,cellTypeIdx])
    sortedIdx = np.array(np.argsort(pValList[:,cellTypeIdx]))
    sigGenesPerCells[cellType] = []
    for i, pVal in enumerate(sortedPvals):
        p_i = ((i+1)/(pValList.shape[0]*pValList.shape[1]))*q
        # p_i = ((i+1)/allCellsMale.shape[1])*q
        if pVal <= p_i:
            p_idx = i
    print(f"{cellType}:\n  Number of Cells: {regionCells.shape[1]}\n  Number of DEGs: {p_idx}")
    GeneID = []
    BH_FDR = []
    bhBool = np.zeros(pValList.shape[0], dtype='int32')
    for geneIdx in sortedIdx[0:p_idx]:
        GeneID.append(sampleForDisplay['geneList'][geneIdx])
        BH_FDR.append(pValList[geneIdx,cellTypeIdx])
        bhBool[geneIdx] = 1
    GeneID = np.array(GeneID).T
    BH_FDR = np.array(BH_FDR).T
    bhCorrList[:, cellTypeIdx] = bhBool
#     cellTypeDF = pd.DataFrame({"Gene_ID": GeneID, "BH_FDR": BH_FDR})
#     if cellType == 'DG/CA4':
#         cellTypeDF.to_excel(writer, sheet_name='DG-CA4', index=False)
#     else:
#         cellTypeDF.to_excel(writer, sheet_name=cellType, index=False)
# # writer.save()
# writer.close()

#%% create color list based on BH list

sigColorList = np.empty([len(sampleForDisplay['geneList']), len(cellsOfInterest)], dtype='int32')

for regionN, region in enumerate(cellsOfInterest):
    for gene in range(len(sampleForDisplay['geneList'])):
        if bhCorrList[gene, regionN] == 1 and logFCList[gene, regionN] < 0:
            sigColorList[gene, regionN] = -1# '#1f77b4'
        elif bhCorrList[gene, regionN] == 1 and logFCList[gene, regionN] > 0:
            sigColorList[gene, regionN] = 1# '#FF0000'
        else:
            sigColorList[gene, regionN] = 0
sigColorList = np.array(sigColorList)

#%% write t-stats, p-vals, and fold-change to excel
# writer = pd.ExcelWriter(os.path.join(derivatives, 't-stat_p-val_fold-change_cell-type_and_regions_male_SD_BH.xlsx'))
# tStatDF = pd.DataFrame(tStatList, index=sampleForDisplay['geneList'], columns=cellsOfInterest)
# pValDF = pd.DataFrame(pValList, index=sampleForDisplay['geneList'], columns=cellsOfInterest)
# foldChangeDF = pd.DataFrame(foldChangeList, index=sampleForDisplay['geneList'], columns=cellsOfInterest)
# tStatDF.to_excel(writer, sheet_name='t-statistic')
# pValDF.to_excel(writer, sheet_name='p-value')
# foldChangeDF.to_excel(writer, sheet_name='fold-change')
# writer.close()

#%% create volcano plot of regions with bar plot of degs per region at top

volcanoMarkerSize = 0.7
volcanoGeneFontSize = 5
volcanoTitleFontSize = 7

plt.close('all')

fig = plt.figure(figsize=(6,4.28))
# for subplot2grid: size of plot (rows, columns) plot location (rows, columns) 
barPlot = plt.subplot2grid((10,7), (0,2), colspan=3, rowspan=2)
ca1Plot = plt.subplot2grid((10,7), (3,0), colspan=3, rowspan=3)
ca2Plot = plt.subplot2grid((10,7), (3,4), colspan=3, rowspan=3)
ca3Plot = plt.subplot2grid((10,7), (7,0), colspan=3, rowspan=3)
dgPlot = plt.subplot2grid((10,7), (7,4), colspan=3, rowspan=3)

# calculate the number of degs per region
degPerRegion = np.sum(bhCorrList, axis=0)
cellsOfInterestNoSparse = np.array(cellsOfInterest[0:10])
degPerRegionNoSparse = np.array(degPerRegion[0:10])
cellsOfInterestColorListNoSparse = np.array(cellsOfInterestColorList[0:10])

barPlot.bar(cellsOfInterestNoSparse[0:4], degPerRegionNoSparse[0:4], color=cellsOfInterestColorListNoSparse[0:4])
barPlot.set_ylabel('Number of DEGs', fontweight='bold')
barPlot.set_xlabel('Region/cell type', fontweight='bold')
# barPlot.set_xticks(barPlot, rotation=20)

# CA1
cellTypeIdx = 0
nOfGenesToPlot = 10
colorArray = np.zeros([sigColorList[:,cellTypeIdx].shape[0], 3])
greyIdx = np.where(sigColorList[:,cellTypeIdx] == 0)
blueIdx = np.where(sigColorList[:,cellTypeIdx] == -1)
redIdx = np.where(sigColorList[:,cellTypeIdx] == 1)
colorArray[greyIdx, :] = np.array([0.5, 0.5, 0.5])
colorArray[blueIdx, :] = np.array([31/255, 119/255, 180/255])
colorArray[redIdx, :] = np.array([1, 0, 0])

ca1Plot.scatter(logFCList[:,cellTypeIdx], -np.log10(pValList[:,cellTypeIdx]), c=colorArray, s=volcanoMarkerSize, linewidth=0)
xabs_max = abs(max(ca1Plot.get_xlim(), key=abs))
ca1Plot.set_xlim(xmin=-xabs_max, xmax=xabs_max)
sortedPval = np.argsort(pValList[:,cellTypeIdx])
# plot only the names of the top ten genes
for j in enumerate(sortedPval[:nOfGenesToPlot]):
    if bhCorrList[j[1],cellTypeIdx] == 1:
        a = ca1Plot.annotate(sampleForDisplay["geneList"][j[1]], (logFCList[j[1],cellTypeIdx], -np.log10(pValList[j[1],cellTypeIdx])), fontsize=volcanoGeneFontSize)
        a.draggable()
# for j in enumerate(sigColorList[:,cellTypeIdx]):
#     if j[1] != 0:
#         ax[0,0].annotate(sampleForDisplay["geneList"][j[0]], (logFCList[j[0],cellTypeIdx], -np.log10(pValList[j[0],cellTypeIdx])))
if cellsOfInterest[cellTypeIdx] == 'DG/CA4':
    ca1Plot.set_title('DG hilus', fontsize=volcanoTitleFontSize, fontweight='bold')
else:
    ca1Plot.set_title(f'{cellsOfInterest[cellTypeIdx]}', fontsize=volcanoTitleFontSize, fontweight='bold')

# CA2
cellTypeIdx = 1
nOfGenesToPlot = 10
colorArray = np.zeros([sigColorList[:,cellTypeIdx].shape[0], 3])
greyIdx = np.where(sigColorList[:,cellTypeIdx] == 0)
blueIdx = np.where(sigColorList[:,cellTypeIdx] == -1)
redIdx = np.where(sigColorList[:,cellTypeIdx] == 1)
colorArray[greyIdx, :] = np.array([0.5, 0.5, 0.5])
colorArray[blueIdx, :] = np.array([31/255, 119/255, 180/255])
colorArray[redIdx, :] = np.array([1, 0, 0])

ca2Plot.scatter(logFCList[:,cellTypeIdx], -np.log10(pValList[:,cellTypeIdx]), c=colorArray, s=volcanoMarkerSize, linewidth=0)
xabs_max = abs(max(ca2Plot.get_xlim(), key=abs))
ca2Plot.set_xlim(xmin=-xabs_max, xmax=xabs_max)
sortedPval = np.argsort(pValList[:,cellTypeIdx])
# plot only the names of the top ten genes
for j in enumerate(sortedPval[:nOfGenesToPlot]):
    if bhCorrList[j[1],cellTypeIdx] == 1:
        a = ca2Plot.annotate(sampleForDisplay["geneList"][j[1]], (logFCList[j[1],cellTypeIdx], -np.log10(pValList[j[1],cellTypeIdx])), fontsize=volcanoGeneFontSize)
        a.draggable()
# for j in enumerate(sigColorList[:,cellTypeIdx]):
#     if j[1] != 0:
#         ax[0,0].annotate(sampleForDisplay["geneList"][j[0]], (logFCList[j[0],cellTypeIdx], -np.log10(pValList[j[0],cellTypeIdx])))
if cellsOfInterest[cellTypeIdx] == 'DG/CA4':
    ca2Plot.set_title('DG hilus', fontsize=volcanoTitleFontSize, fontweight='bold')
else:
    ca2Plot.set_title(f'{cellsOfInterest[cellTypeIdx]}', fontsize=volcanoTitleFontSize, fontweight='bold')

# CA3
cellTypeIdx = 2
nOfGenesToPlot = 10
colorArray = np.zeros([sigColorList[:,cellTypeIdx].shape[0], 3])
greyIdx = np.where(sigColorList[:,cellTypeIdx] == 0)
blueIdx = np.where(sigColorList[:,cellTypeIdx] == -1)
redIdx = np.where(sigColorList[:,cellTypeIdx] == 1)
colorArray[greyIdx, :] = np.array([0.5, 0.5, 0.5])
colorArray[blueIdx, :] = np.array([31/255, 119/255, 180/255])
colorArray[redIdx, :] = np.array([1, 0, 0])

ca3Plot.scatter(logFCList[:,cellTypeIdx], -np.log10(pValList[:,cellTypeIdx]), c=colorArray, s=volcanoMarkerSize, linewidth=0)
xabs_max = abs(max(ca3Plot.get_xlim(), key=abs))
ca3Plot.set_xlim(xmin=-xabs_max, xmax=xabs_max)
sortedPval = np.argsort(pValList[:,cellTypeIdx])
# plot only the names of the top ten genes
for j in enumerate(sortedPval[:nOfGenesToPlot]):
    if bhCorrList[j[1],cellTypeIdx] == 1:
        a = ca3Plot.annotate(sampleForDisplay["geneList"][j[1]], (logFCList[j[1],cellTypeIdx], -np.log10(pValList[j[1],cellTypeIdx])), fontsize=volcanoGeneFontSize)
        a.draggable()
# for j in enumerate(sigColorList[:,cellTypeIdx]):
#     if j[1] != 0:
#         ax[0,0].annotate(sampleForDisplay["geneList"][j[0]], (logFCList[j[0],cellTypeIdx], -np.log10(pValList[j[0],cellTypeIdx])))
if cellsOfInterest[cellTypeIdx] == 'DG/CA4':
    ca3Plot.set_title('DG hilus', fontsize=volcanoTitleFontSize, fontweight='bold')
else:
    ca3Plot.set_title(f'{cellsOfInterest[cellTypeIdx]}', fontsize=volcanoTitleFontSize, fontweight='bold')

# DG
cellTypeIdx = 3
nOfGenesToPlot = 10
colorArray = np.zeros([sigColorList[:,cellTypeIdx].shape[0], 3])
greyIdx = np.where(sigColorList[:,cellTypeIdx] == 0)
blueIdx = np.where(sigColorList[:,cellTypeIdx] == -1)
redIdx = np.where(sigColorList[:,cellTypeIdx] == 1)
colorArray[greyIdx, :] = np.array([0.5, 0.5, 0.5])
colorArray[blueIdx, :] = np.array([31/255, 119/255, 180/255])
colorArray[redIdx, :] = np.array([1, 0, 0])

dgPlot.scatter(logFCList[:,cellTypeIdx], -np.log10(pValList[:,cellTypeIdx]), c=colorArray, s=volcanoMarkerSize, linewidth=0)
xabs_max = abs(max(dgPlot.get_xlim(), key=abs))
dgPlot.set_xlim(xmin=-xabs_max, xmax=xabs_max)
sortedPval = np.argsort(pValList[:,cellTypeIdx])
# plot only the names of the top ten genes
for j in enumerate(sortedPval[:nOfGenesToPlot]):
    if bhCorrList[j[1],cellTypeIdx] == 1:
        a = dgPlot.annotate(sampleForDisplay["geneList"][j[1]], (logFCList[j[1],cellTypeIdx], -np.log10(pValList[j[1],cellTypeIdx])), fontsize=volcanoGeneFontSize)
        a.draggable()
# for j in enumerate(sigColorList[:,cellTypeIdx]):
#     if j[1] != 0:
#         ax[0,0].annotate(sampleForDisplay["geneList"][j[0]], (logFCList[j[0],cellTypeIdx], -np.log10(pValList[j[0],cellTypeIdx])))
if cellsOfInterest[cellTypeIdx] == 'DG/CA4':
    dgPlot.set_title('DG hilus', fontsize=volcanoTitleFontSize, fontweight='bold')
else:
    dgPlot.set_title(f'{cellsOfInterest[cellTypeIdx]}', fontsize=volcanoTitleFontSize, fontweight='bold')

fig.text(0.55, 0.04, 'logFC', ha='center', fontsize=volcanoTitleFontSize, fontweight='bold')
fig.text(0.04, 0.37, '-log10(p-value)', va='center', rotation='vertical', fontsize=volcanoTitleFontSize, fontweight='bold')

plt.show()

### edited locations of gene annotations to prevent overlap in inkscape, so only generate new if necessary
# output pdf and svg
# plt.savefig(os.path.join(figureFolder, f'figure02_regional_volcano_plots_volcano_marker_{volcanoMarkerSize}.pdf'), bbox_inches='tight', dpi=300)
# plt.savefig(os.path.join(figureFolder, f'figure02_regional_volcano_plots_volcano_marker_{volcanoMarkerSize}.svg'), bbox_inches='tight', dpi=300)
# plt.close('all')

#%% volcano plot for all cell types in one plot, reorganized

plt.close('all')
# uses -log10(pValue) as y axis, logFC as x axis

fig = plt.figure(figsize=(6,6.69))
# for subplot2grid: size of plot (rows, columns) plot location (rows, columns) 
barPlot = plt.subplot2grid((14,7), (0,1), colspan=5, rowspan=2)
astPlot = plt.subplot2grid((14,7), (3,0), colspan=3, rowspan=3)
oliPlot = plt.subplot2grid((14,7), (3,4), colspan=3, rowspan=3)
micPlot = plt.subplot2grid((14,7), (7,0), colspan=3, rowspan=3)
endPlot = plt.subplot2grid((14,7), (7,4), colspan=3, rowspan=3)
dgIntPlot = plt.subplot2grid((14,7), (11,0), colspan=3, rowspan=3)
neuPlot = plt.subplot2grid((14,7), (11,4), colspan=3, rowspan=3)

# mask in order to reorder the bar plots based on desired order
reorderMask = np.array([5, 9, 7, 6, 4, 8], dtype='int32')

cellsOfInterestNoSparse[4] = 'DG hilus'
barPlot.bar(cellsOfInterestNoSparse[reorderMask], degPerRegionNoSparse[reorderMask], color=cellsOfInterestColorListNoSparse[reorderMask])
barPlot.set_ylabel('Number of DEGs', fontweight='bold')
# barPlot.set_xlabel('Region/cell type')
barPlot.set_xticks(np.arange(0,6), cellsOfInterestNoSparse[reorderMask], rotation=15)
# neurons
cellTypeIdx = 8

colorArray = np.zeros([sigColorList[:,cellTypeIdx].shape[0], 3])
greyIdx = np.where(sigColorList[:,cellTypeIdx] == 0)
blueIdx = np.where(sigColorList[:,cellTypeIdx] == -1)
redIdx = np.where(sigColorList[:,cellTypeIdx] == 1)
colorArray[greyIdx, :] = np.array([0.5, 0.5, 0.5])
colorArray[blueIdx, :] = np.array([31/255, 119/255, 180/255])
colorArray[redIdx, :] = np.array([1, 0, 0])

neuPlot.scatter(logFCList[:,cellTypeIdx], -np.log10(pValList[:,cellTypeIdx]), c=colorArray, s=volcanoMarkerSize, linewidth=0)
xabs_max = abs(max(neuPlot.get_xlim(), key=abs))
neuPlot.set_xlim(xmin=-xabs_max, xmax=xabs_max)
sortedPval = np.argsort(pValList[:,cellTypeIdx])
# plot only the names of the top ten genes
for j in enumerate(sortedPval[:10]):
    if bhCorrList[j[1],cellTypeIdx] == 1:
        a = neuPlot.annotate(sampleForDisplay["geneList"][j[1]], (logFCList[j[1],cellTypeIdx], -np.log10(pValList[j[1],cellTypeIdx])))
        a.draggable()
# for j in enumerate(sigColorList[:,cellTypeIdx]):
#     if j[1] != 0:
#         ax[0,0].annotate(sampleForDisplay["geneList"][j[0]], (logFCList[j[0],cellTypeIdx], -np.log10(pValList[j[0],cellTypeIdx])))
if cellsOfInterest[cellTypeIdx] == 'DG/CA4':
    neuPlot.set_title('DG hilus', fontweight='bold')
else:
    neuPlot.set_title('interneurons', fontweight='bold')

# DG hilus
cellTypeIdx = 4

colorArray = np.zeros([sigColorList[:,cellTypeIdx].shape[0], 3])
greyIdx = np.where(sigColorList[:,cellTypeIdx] == 0)
blueIdx = np.where(sigColorList[:,cellTypeIdx] == -1)
redIdx = np.where(sigColorList[:,cellTypeIdx] == 1)
colorArray[greyIdx, :] = np.array([0.5, 0.5, 0.5])
colorArray[blueIdx, :] = np.array([31/255, 119/255, 180/255])
colorArray[redIdx, :] = np.array([1, 0, 0])

dgIntPlot.scatter(logFCList[:,cellTypeIdx], -np.log10(pValList[:,cellTypeIdx]), c=colorArray, s=volcanoMarkerSize, linewidth=0)
xabs_max = abs(max(dgIntPlot.get_xlim(), key=abs))
dgIntPlot.set_xlim(xmin=-xabs_max, xmax=xabs_max)
sortedPval = np.argsort(pValList[:,cellTypeIdx])
# plot only the names of the top ten genes
for j in enumerate(sortedPval[:10]):
    if bhCorrList[j[1],cellTypeIdx] == 1:
        a = dgIntPlot.annotate(sampleForDisplay["geneList"][j[1]], (logFCList[j[1],cellTypeIdx], -np.log10(pValList[j[1],cellTypeIdx])))
        a.draggable()
# for j in enumerate(sigColorList[:,cellTypeIdx]):
#     if j[1] != 0:
#         ax[0,1].annotate(sampleForDisplay["geneList"][j[0]], (logFCList[j[0],cellTypeIdx], -np.log10(pValList[j[0],cellTypeIdx])))
if cellsOfInterest[cellTypeIdx] == 'DG/CA4':
    dgIntPlot.set_title('DG hilus', fontweight='bold')
else:
    dgIntPlot.set_title(f'{cellsOfInterest[cellTypeIdx]}', fontweight='bold')

# Astrocytes
cellTypeIdx = 5

colorArray = np.zeros([sigColorList[:,cellTypeIdx].shape[0], 3])
greyIdx = np.where(sigColorList[:,cellTypeIdx] == 0)
blueIdx = np.where(sigColorList[:,cellTypeIdx] == -1)
redIdx = np.where(sigColorList[:,cellTypeIdx] == 1)
colorArray[greyIdx, :] = np.array([0.5, 0.5, 0.5])
colorArray[blueIdx, :] = np.array([31/255, 119/255, 180/255])
colorArray[redIdx, :] = np.array([1, 0, 0])

astPlot.scatter(logFCList[:,cellTypeIdx], -np.log10(pValList[:,cellTypeIdx]), c=colorArray, s=volcanoMarkerSize, linewidth=0)
xabs_max = abs(max(astPlot.get_xlim(), key=abs))
astPlot.set_xlim(xmin=-xabs_max, xmax=xabs_max)
sortedPval = np.argsort(pValList[:,cellTypeIdx])
# plot only the names of the top ten genes
for j in enumerate(sortedPval[:10]):
    if bhCorrList[j[1],cellTypeIdx] == 1:
        astPlot.annotate(sampleForDisplay["geneList"][j[1]], (logFCList[j[1],cellTypeIdx], -np.log10(pValList[j[1],cellTypeIdx])))
# for j in enumerate(sigColorList[:,cellTypeIdx]):
#     if j[1] != 0:
#         ax[1,1].annotate(sampleForDisplay["geneList"][j[0]], (logFCList[j[0],cellTypeIdx], -np.log10(pValList[j[0],cellTypeIdx])))
if cellsOfInterest[cellTypeIdx] == 'DG/CA4':
    astPlot.set_title('DG hilus', fontweight='bold')
else:
    astPlot.set_title(f'{cellsOfInterest[cellTypeIdx]}', fontweight='bold')

# microglia
cellTypeIdx = 7

colorArray = np.zeros([sigColorList[:,cellTypeIdx].shape[0], 3])
greyIdx = np.where(sigColorList[:,cellTypeIdx] == 0)
blueIdx = np.where(sigColorList[:,cellTypeIdx] == -1)
redIdx = np.where(sigColorList[:,cellTypeIdx] == 1)
colorArray[greyIdx, :] = np.array([0.5, 0.5, 0.5])
colorArray[blueIdx, :] = np.array([31/255, 119/255, 180/255])
colorArray[redIdx, :] = np.array([1, 0, 0])

micPlot.scatter(logFCList[:,cellTypeIdx], -np.log10(pValList[:,cellTypeIdx]), c=colorArray, s=volcanoMarkerSize, linewidth=0)
xabs_max = abs(max(micPlot.get_xlim(), key=abs))
micPlot.set_xlim(xmin=-xabs_max, xmax=xabs_max)
sortedPval = np.argsort(pValList[:,cellTypeIdx])
# plot only the names of the top ten genes
for j in enumerate(sortedPval[:10]):
    if bhCorrList[j[1],cellTypeIdx] == 1:
        micPlot.annotate(sampleForDisplay["geneList"][j[1]], (logFCList[j[1],cellTypeIdx], -np.log10(pValList[j[1],cellTypeIdx])))
# for j in enumerate(sigColorList[:,cellTypeIdx]):
#     if j[1] != 0:
#         ax[1,0].annotate(sampleForDisplay["geneList"][j[0]], (logFCList[j[0],cellTypeIdx], -np.log10(pValList[j[0],cellTypeIdx])))
if cellsOfInterest[cellTypeIdx] == 'DG/CA4':
    micPlot.set_title('DG hilus', fontweight='bold')
else:
    micPlot.set_title(f'{cellsOfInterest[cellTypeIdx]}', fontweight='bold')

# oligodendrocytes
cellTypeIdx = 9

colorArray = np.zeros([sigColorList[:,cellTypeIdx].shape[0], 3])
greyIdx = np.where(sigColorList[:,cellTypeIdx] == 0)
blueIdx = np.where(sigColorList[:,cellTypeIdx] == -1)
redIdx = np.where(sigColorList[:,cellTypeIdx] == 1)
colorArray[greyIdx, :] = np.array([0.5, 0.5, 0.5])
colorArray[blueIdx, :] = np.array([31/255, 119/255, 180/255])
colorArray[redIdx, :] = np.array([1, 0, 0])

oliPlot.scatter(logFCList[:,cellTypeIdx], -np.log10(pValList[:,cellTypeIdx]), c=colorArray, s=volcanoMarkerSize, linewidth=0)
xabs_max = abs(max(oliPlot.get_xlim(), key=abs))
oliPlot.set_xlim(xmin=-xabs_max, xmax=xabs_max)
sortedPval = np.argsort(pValList[:,cellTypeIdx])
# plot only the names of the top ten genes
for j in enumerate(sortedPval[:10]):
    if bhCorrList[j[1],cellTypeIdx] == 1:
        oliPlot.annotate(sampleForDisplay["geneList"][j[1]], (logFCList[j[1],cellTypeIdx], -np.log10(pValList[j[1],cellTypeIdx])))
# for j in enumerate(sigColorList[:,cellTypeIdx]):
#     if j[1] != 0:
#         ax[1,1].annotate(sampleForDisplay["geneList"][j[0]], (logFCList[j[0],cellTypeIdx], -np.log10(pValList[j[0],cellTypeIdx])))
if cellsOfInterest[cellTypeIdx] == 'DG/CA4':
    oliPlot.set_title('DG hilus', fontweight='bold')
else:
    oliPlot.set_title(f'{cellsOfInterest[cellTypeIdx]}', fontweight='bold')

# endothelial
cellTypeIdx = 6

colorArray = np.zeros([sigColorList[:,cellTypeIdx].shape[0], 3])
greyIdx = np.where(sigColorList[:,cellTypeIdx] == 0)
blueIdx = np.where(sigColorList[:,cellTypeIdx] == -1)
redIdx = np.where(sigColorList[:,cellTypeIdx] == 1)
colorArray[greyIdx, :] = np.array([0.5, 0.5, 0.5])
colorArray[blueIdx, :] = np.array([31/255, 119/255, 180/255])
colorArray[redIdx, :] = np.array([1, 0, 0])

endPlot.scatter(logFCList[:,cellTypeIdx], -np.log10(pValList[:,cellTypeIdx]), c=colorArray, s=volcanoMarkerSize, linewidth=0)
xabs_max = abs(max(endPlot.get_xlim(), key=abs))
endPlot.set_xlim(xmin=-xabs_max, xmax=xabs_max)
sortedPval = np.argsort(pValList[:,cellTypeIdx])
# plot only the names of the top ten genes
for j in enumerate(sortedPval[:10]):
    if bhCorrList[j[1],cellTypeIdx] == 1:
        endPlot.annotate(sampleForDisplay["geneList"][j[1]], (logFCList[j[1],cellTypeIdx], -np.log10(pValList[j[1],cellTypeIdx])))
# for j in enumerate(sigColorList[:,cellTypeIdx]):
#     if j[1] != 0:
#         ax[1,1].annotate(sampleForDisplay["geneList"][j[0]], (logFCList[j[0],cellTypeIdx], -np.log10(pValList[j[0],cellTypeIdx])))
if cellsOfInterest[cellTypeIdx] == 'DG/CA4':
    endPlot.set_title('DG hilus', fontweight='bold')
else:
    endPlot.set_title(f'{cellsOfInterest[cellTypeIdx]}', fontweight='bold')

fig.text(0.5, 0.04, 'logFC', ha='center')
fig.text(0.04, 0.42, '-log10(p-value)', va='center', rotation='vertical')

plt.show()

### edited locations of gene annotations to prevent overlap in inkscape, so only generate new if necessary
# output pdf and svg
# plt.savefig(os.path.join(figureFolder, f'figure02_cell_type_volcano_plots_volcano_marker_{volcanoMarkerSize}.pdf'), bbox_inches='tight', dpi=300)
# plt.savefig(os.path.join(figureFolder, f'figure02_cell_type_volcano_plots_volcano_marker_{volcanoMarkerSize}.svg'), bbox_inches='tight', dpi=300)
# plt.close('all')

#%% combine two portions of figure 2 into single figure
# create volcano plots of regions with bar plot of degs per region at top

# volcanoMarkerSize = 15
# volcanoGeneFontSize = 12
# volcanoTitleFontSize = 16

# plt.close('all')

# fig = plt.figure(figsize=(18,10))
# # for subplot2grid: size of plot (rows, columns) plot location (rows, columns) 
# barPlot1 = plt.subplot2grid((14,16), (0,2), colspan=3, rowspan=2)
# ca1Plot = plt.subplot2grid((14,16), (3,0), colspan=3, rowspan=3)
# ca2Plot = plt.subplot2grid((14,16), (3,4), colspan=3, rowspan=3)
# ca3Plot = plt.subplot2grid((14,16), (7,0), colspan=3, rowspan=3)
# dgPlot = plt.subplot2grid((14,16), (7,4), colspan=3, rowspan=3)

# barPlot2 = plt.subplot2grid((14,16), (0,9), colspan=5, rowspan=2)
# astPlot = plt.subplot2grid((14,16), (3,8), colspan=3, rowspan=3)
# oliPlot = plt.subplot2grid((14,16), (3,12), colspan=3, rowspan=3)
# micPlot = plt.subplot2grid((14,16), (7,8), colspan=3, rowspan=3)
# endPlot = plt.subplot2grid((14,16), (7,12), colspan=3, rowspan=3)
# dgIntPlot = plt.subplot2grid((14,16), (11,8), colspan=3, rowspan=3)
# neuPlot = plt.subplot2grid((14,16), (11,12), colspan=3, rowspan=3)


# # calculate the number of degs per region
# degPerRegion = np.sum(bhCorrList, axis=0)
# cellsOfInterestNoSparse = np.array(cellsOfInterest[0:10])
# degPerRegionNoSparse = np.array(degPerRegion[0:10])
# cellsOfInterestColorListNoSparse = np.array(cellsOfInterestColorList[0:10])

# barPlot1.bar(cellsOfInterestNoSparse[0:4], degPerRegionNoSparse[0:4], color=cellsOfInterestColorListNoSparse[0:4])
# barPlot1.set_ylabel('Number of DEGs', fontweight='bold')
# barPlot1.set_xlabel('Region/cell type', fontweight='bold')
# # barPlot.set_xticks(barPlot, rotation=20)

# # CA1
# cellTypeIdx = 0
# nOfGenesToPlot = 10
# colorArray = np.zeros([sigColorList[:,cellTypeIdx].shape[0], 3])
# greyIdx = np.where(sigColorList[:,cellTypeIdx] == 0)
# blueIdx = np.where(sigColorList[:,cellTypeIdx] == -1)
# redIdx = np.where(sigColorList[:,cellTypeIdx] == 1)
# colorArray[greyIdx, :] = np.array([0.5, 0.5, 0.5])
# colorArray[blueIdx, :] = np.array([31/255, 119/255, 180/255])
# colorArray[redIdx, :] = np.array([1, 0, 0])

# ca1Plot.scatter(logFCList[:,cellTypeIdx], -np.log10(pValList[:,cellTypeIdx]), c=colorArray, s=volcanoMarkerSize, linewidth=0)
# xabs_max = abs(max(ca1Plot.get_xlim(), key=abs))
# ca1Plot.set_xlim(xmin=-xabs_max, xmax=xabs_max)
# sortedPval = np.argsort(pValList[:,cellTypeIdx])
# # plot only the names of the top ten genes
# for j in enumerate(sortedPval[:nOfGenesToPlot]):
#     if bhCorrList[j[1],cellTypeIdx] == 1:
#         a = ca1Plot.annotate(sampleForDisplay["geneList"][j[1]], (logFCList[j[1],cellTypeIdx], -np.log10(pValList[j[1],cellTypeIdx])), fontsize=volcanoGeneFontSize)
#         a.draggable()
# # for j in enumerate(sigColorList[:,cellTypeIdx]):
# #     if j[1] != 0:
# #         ax[0,0].annotate(sampleForDisplay["geneList"][j[0]], (logFCList[j[0],cellTypeIdx], -np.log10(pValList[j[0],cellTypeIdx])))
# if cellsOfInterest[cellTypeIdx] == 'DG/CA4':
#     ca1Plot.set_title('DG hilus', fontsize=volcanoTitleFontSize, fontweight='bold')
# else:
#     ca1Plot.set_title(f'{cellsOfInterest[cellTypeIdx]}', fontsize=volcanoTitleFontSize, fontweight='bold')

# # CA2
# cellTypeIdx = 1
# nOfGenesToPlot = 10
# colorArray = np.zeros([sigColorList[:,cellTypeIdx].shape[0], 3])
# greyIdx = np.where(sigColorList[:,cellTypeIdx] == 0)
# blueIdx = np.where(sigColorList[:,cellTypeIdx] == -1)
# redIdx = np.where(sigColorList[:,cellTypeIdx] == 1)
# colorArray[greyIdx, :] = np.array([0.5, 0.5, 0.5])
# colorArray[blueIdx, :] = np.array([31/255, 119/255, 180/255])
# colorArray[redIdx, :] = np.array([1, 0, 0])

# ca2Plot.scatter(logFCList[:,cellTypeIdx], -np.log10(pValList[:,cellTypeIdx]), c=colorArray, s=volcanoMarkerSize, linewidth=0)
# xabs_max = abs(max(ca2Plot.get_xlim(), key=abs))
# ca2Plot.set_xlim(xmin=-xabs_max, xmax=xabs_max)
# sortedPval = np.argsort(pValList[:,cellTypeIdx])
# # plot only the names of the top ten genes
# for j in enumerate(sortedPval[:nOfGenesToPlot]):
#     if bhCorrList[j[1],cellTypeIdx] == 1:
#         a = ca2Plot.annotate(sampleForDisplay["geneList"][j[1]], (logFCList[j[1],cellTypeIdx], -np.log10(pValList[j[1],cellTypeIdx])), fontsize=volcanoGeneFontSize)
#         a.draggable()
# # for j in enumerate(sigColorList[:,cellTypeIdx]):
# #     if j[1] != 0:
# #         ax[0,0].annotate(sampleForDisplay["geneList"][j[0]], (logFCList[j[0],cellTypeIdx], -np.log10(pValList[j[0],cellTypeIdx])))
# if cellsOfInterest[cellTypeIdx] == 'DG/CA4':
#     ca2Plot.set_title('DG hilus', fontsize=volcanoTitleFontSize, fontweight='bold')
# else:
#     ca2Plot.set_title(f'{cellsOfInterest[cellTypeIdx]}', fontsize=volcanoTitleFontSize, fontweight='bold')

# # CA3
# cellTypeIdx = 2
# nOfGenesToPlot = 10
# colorArray = np.zeros([sigColorList[:,cellTypeIdx].shape[0], 3])
# greyIdx = np.where(sigColorList[:,cellTypeIdx] == 0)
# blueIdx = np.where(sigColorList[:,cellTypeIdx] == -1)
# redIdx = np.where(sigColorList[:,cellTypeIdx] == 1)
# colorArray[greyIdx, :] = np.array([0.5, 0.5, 0.5])
# colorArray[blueIdx, :] = np.array([31/255, 119/255, 180/255])
# colorArray[redIdx, :] = np.array([1, 0, 0])

# ca3Plot.scatter(logFCList[:,cellTypeIdx], -np.log10(pValList[:,cellTypeIdx]), c=colorArray, s=volcanoMarkerSize, linewidth=0)
# xabs_max = abs(max(ca3Plot.get_xlim(), key=abs))
# ca3Plot.set_xlim(xmin=-xabs_max, xmax=xabs_max)
# sortedPval = np.argsort(pValList[:,cellTypeIdx])
# # plot only the names of the top ten genes
# for j in enumerate(sortedPval[:nOfGenesToPlot]):
#     if bhCorrList[j[1],cellTypeIdx] == 1:
#         a = ca3Plot.annotate(sampleForDisplay["geneList"][j[1]], (logFCList[j[1],cellTypeIdx], -np.log10(pValList[j[1],cellTypeIdx])), fontsize=volcanoGeneFontSize)
#         a.draggable()
# # for j in enumerate(sigColorList[:,cellTypeIdx]):
# #     if j[1] != 0:
# #         ax[0,0].annotate(sampleForDisplay["geneList"][j[0]], (logFCList[j[0],cellTypeIdx], -np.log10(pValList[j[0],cellTypeIdx])))
# if cellsOfInterest[cellTypeIdx] == 'DG/CA4':
#     ca3Plot.set_title('DG hilus', fontsize=volcanoTitleFontSize, fontweight='bold')
# else:
#     ca3Plot.set_title(f'{cellsOfInterest[cellTypeIdx]}', fontsize=volcanoTitleFontSize, fontweight='bold')

# # DG
# cellTypeIdx = 3
# nOfGenesToPlot = 10
# colorArray = np.zeros([sigColorList[:,cellTypeIdx].shape[0], 3])
# greyIdx = np.where(sigColorList[:,cellTypeIdx] == 0)
# blueIdx = np.where(sigColorList[:,cellTypeIdx] == -1)
# redIdx = np.where(sigColorList[:,cellTypeIdx] == 1)
# colorArray[greyIdx, :] = np.array([0.5, 0.5, 0.5])
# colorArray[blueIdx, :] = np.array([31/255, 119/255, 180/255])
# colorArray[redIdx, :] = np.array([1, 0, 0])

# dgPlot.scatter(logFCList[:,cellTypeIdx], -np.log10(pValList[:,cellTypeIdx]), c=colorArray, s=volcanoMarkerSize, linewidth=0)
# xabs_max = abs(max(dgPlot.get_xlim(), key=abs))
# dgPlot.set_xlim(xmin=-xabs_max, xmax=xabs_max)
# sortedPval = np.argsort(pValList[:,cellTypeIdx])
# # plot only the names of the top ten genes
# for j in enumerate(sortedPval[:nOfGenesToPlot]):
#     if bhCorrList[j[1],cellTypeIdx] == 1:
#         a = dgPlot.annotate(sampleForDisplay["geneList"][j[1]], (logFCList[j[1],cellTypeIdx], -np.log10(pValList[j[1],cellTypeIdx])), fontsize=volcanoGeneFontSize)
#         a.draggable()
# # for j in enumerate(sigColorList[:,cellTypeIdx]):
# #     if j[1] != 0:
# #         ax[0,0].annotate(sampleForDisplay["geneList"][j[0]], (logFCList[j[0],cellTypeIdx], -np.log10(pValList[j[0],cellTypeIdx])))
# if cellsOfInterest[cellTypeIdx] == 'DG/CA4':
#     dgPlot.set_title('DG hilus', fontsize=volcanoTitleFontSize)
# else:
#     dgPlot.set_title(f'{cellsOfInterest[cellTypeIdx]}', fontsize=volcanoTitleFontSize)

# # volcano plot for all cell types in one plot, reorganized
# # mask in order to reorder the bar plots based on desired order
# reorderMask = np.array([5, 9, 7, 6, 4, 8], dtype='int32')

# cellsOfInterestNoSparse[4] = 'DG hilus'
# cellsOfInterestNoSparse[8] = 'interneurons'
# barPlot2.bar(cellsOfInterestNoSparse[reorderMask], degPerRegionNoSparse[reorderMask], color=cellsOfInterestColorListNoSparse[reorderMask])
# barPlot2.set_ylabel('Number of DEGs', fontweight='bold')
# barPlot2.set_ylim(barPlot1.get_ylim())
# barPlot2.set_xticks(np.arange(0,6), cellsOfInterestNoSparse[reorderMask], rotation=15)
# # neurons
# cellTypeIdx = 8

# colorArray = np.zeros([sigColorList[:,cellTypeIdx].shape[0], 3])
# greyIdx = np.where(sigColorList[:,cellTypeIdx] == 0)
# blueIdx = np.where(sigColorList[:,cellTypeIdx] == -1)
# redIdx = np.where(sigColorList[:,cellTypeIdx] == 1)
# colorArray[greyIdx, :] = np.array([0.5, 0.5, 0.5])
# colorArray[blueIdx, :] = np.array([31/255, 119/255, 180/255])
# colorArray[redIdx, :] = np.array([1, 0, 0])

# neuPlot.scatter(logFCList[:,cellTypeIdx], -np.log10(pValList[:,cellTypeIdx]), c=colorArray, s=volcanoMarkerSize, linewidth=0)
# xabs_max = abs(max(neuPlot.get_xlim(), key=abs))
# neuPlot.set_xlim(xmin=-xabs_max, xmax=xabs_max)
# sortedPval = np.argsort(pValList[:,cellTypeIdx])
# # plot only the names of the top ten genes
# for j in enumerate(sortedPval[:10]):
#     if bhCorrList[j[1],cellTypeIdx] == 1:
#         a = neuPlot.annotate(sampleForDisplay["geneList"][j[1]], (logFCList[j[1],cellTypeIdx], -np.log10(pValList[j[1],cellTypeIdx])), fontsize=volcanoGeneFontSize)
#         a.draggable()
# # for j in enumerate(sigColorList[:,cellTypeIdx]):
# #     if j[1] != 0:
# #         ax[0,0].annotate(sampleForDisplay["geneList"][j[0]], (logFCList[j[0],cellTypeIdx], -np.log10(pValList[j[0],cellTypeIdx])))
# if cellsOfInterest[cellTypeIdx] == 'DG/CA4':
#     neuPlot.set_title('DG hilus', fontweight='bold')
# else:
#     neuPlot.set_title('interneurons', fontweight='bold')

# # DG hilus
# cellTypeIdx = 4

# colorArray = np.zeros([sigColorList[:,cellTypeIdx].shape[0], 3])
# greyIdx = np.where(sigColorList[:,cellTypeIdx] == 0)
# blueIdx = np.where(sigColorList[:,cellTypeIdx] == -1)
# redIdx = np.where(sigColorList[:,cellTypeIdx] == 1)
# colorArray[greyIdx, :] = np.array([0.5, 0.5, 0.5])
# colorArray[blueIdx, :] = np.array([31/255, 119/255, 180/255])
# colorArray[redIdx, :] = np.array([1, 0, 0])

# dgIntPlot.scatter(logFCList[:,cellTypeIdx], -np.log10(pValList[:,cellTypeIdx]), c=colorArray, s=volcanoMarkerSize, linewidth=0)
# xabs_max = abs(max(dgIntPlot.get_xlim(), key=abs))
# dgIntPlot.set_xlim(xmin=-xabs_max, xmax=xabs_max)
# sortedPval = np.argsort(pValList[:,cellTypeIdx])
# # plot only the names of the top ten genes
# for j in enumerate(sortedPval[:10]):
#     if bhCorrList[j[1],cellTypeIdx] == 1:
#         a = dgIntPlot.annotate(sampleForDisplay["geneList"][j[1]], (logFCList[j[1],cellTypeIdx], -np.log10(pValList[j[1],cellTypeIdx])), fontsize=volcanoGeneFontSize)
#         a.draggable()
# # for j in enumerate(sigColorList[:,cellTypeIdx]):
# #     if j[1] != 0:
# #         ax[0,1].annotate(sampleForDisplay["geneList"][j[0]], (logFCList[j[0],cellTypeIdx], -np.log10(pValList[j[0],cellTypeIdx])))
# if cellsOfInterest[cellTypeIdx] == 'DG/CA4':
#     dgIntPlot.set_title('DG hilus', fontweight='bold')
# else:
#     dgIntPlot.set_title(f'{cellsOfInterest[cellTypeIdx]}', fontweight='bold')

# # Astrocytes
# cellTypeIdx = 5

# colorArray = np.zeros([sigColorList[:,cellTypeIdx].shape[0], 3])
# greyIdx = np.where(sigColorList[:,cellTypeIdx] == 0)
# blueIdx = np.where(sigColorList[:,cellTypeIdx] == -1)
# redIdx = np.where(sigColorList[:,cellTypeIdx] == 1)
# colorArray[greyIdx, :] = np.array([0.5, 0.5, 0.5])
# colorArray[blueIdx, :] = np.array([31/255, 119/255, 180/255])
# colorArray[redIdx, :] = np.array([1, 0, 0])

# astPlot.scatter(logFCList[:,cellTypeIdx], -np.log10(pValList[:,cellTypeIdx]), c=colorArray, s=volcanoMarkerSize, linewidth=0)
# xabs_max = abs(max(astPlot.get_xlim(), key=abs))
# astPlot.set_xlim(xmin=-xabs_max, xmax=xabs_max)
# sortedPval = np.argsort(pValList[:,cellTypeIdx])
# # plot only the names of the top ten genes
# for j in enumerate(sortedPval[:10]):
#     if bhCorrList[j[1],cellTypeIdx] == 1:
#         astPlot.annotate(sampleForDisplay["geneList"][j[1]], (logFCList[j[1],cellTypeIdx], -np.log10(pValList[j[1],cellTypeIdx])), fontsize=volcanoGeneFontSize)
# # for j in enumerate(sigColorList[:,cellTypeIdx]):
# #     if j[1] != 0:
# #         ax[1,1].annotate(sampleForDisplay["geneList"][j[0]], (logFCList[j[0],cellTypeIdx], -np.log10(pValList[j[0],cellTypeIdx])))
# if cellsOfInterest[cellTypeIdx] == 'DG/CA4':
#     astPlot.set_title('DG hilus', fontweight='bold')
# else:
#     astPlot.set_title(f'{cellsOfInterest[cellTypeIdx]}', fontweight='bold')

# # microglia
# cellTypeIdx = 7

# colorArray = np.zeros([sigColorList[:,cellTypeIdx].shape[0], 3])
# greyIdx = np.where(sigColorList[:,cellTypeIdx] == 0)
# blueIdx = np.where(sigColorList[:,cellTypeIdx] == -1)
# redIdx = np.where(sigColorList[:,cellTypeIdx] == 1)
# colorArray[greyIdx, :] = np.array([0.5, 0.5, 0.5])
# colorArray[blueIdx, :] = np.array([31/255, 119/255, 180/255])
# colorArray[redIdx, :] = np.array([1, 0, 0])

# micPlot.scatter(logFCList[:,cellTypeIdx], -np.log10(pValList[:,cellTypeIdx]), c=colorArray, s=volcanoMarkerSize, linewidth=0)
# xabs_max = abs(max(micPlot.get_xlim(), key=abs))
# micPlot.set_xlim(xmin=-xabs_max, xmax=xabs_max)
# sortedPval = np.argsort(pValList[:,cellTypeIdx])
# # plot only the names of the top ten genes
# for j in enumerate(sortedPval[:10]):
#     if bhCorrList[j[1],cellTypeIdx] == 1:
#         micPlot.annotate(sampleForDisplay["geneList"][j[1]], (logFCList[j[1],cellTypeIdx], -np.log10(pValList[j[1],cellTypeIdx])), fontsize=volcanoGeneFontSize)
# # for j in enumerate(sigColorList[:,cellTypeIdx]):
# #     if j[1] != 0:
# #         ax[1,0].annotate(sampleForDisplay["geneList"][j[0]], (logFCList[j[0],cellTypeIdx], -np.log10(pValList[j[0],cellTypeIdx])))
# if cellsOfInterest[cellTypeIdx] == 'DG/CA4':
#     micPlot.set_title('DG hilus', fontweight='bold')
# else:
#     micPlot.set_title(f'{cellsOfInterest[cellTypeIdx]}', fontweight='bold')

# # oligodendrocytes
# cellTypeIdx = 9

# colorArray = np.zeros([sigColorList[:,cellTypeIdx].shape[0], 3])
# greyIdx = np.where(sigColorList[:,cellTypeIdx] == 0)
# blueIdx = np.where(sigColorList[:,cellTypeIdx] == -1)
# redIdx = np.where(sigColorList[:,cellTypeIdx] == 1)
# colorArray[greyIdx, :] = np.array([0.5, 0.5, 0.5])
# colorArray[blueIdx, :] = np.array([31/255, 119/255, 180/255])
# colorArray[redIdx, :] = np.array([1, 0, 0])

# oliPlot.scatter(logFCList[:,cellTypeIdx], -np.log10(pValList[:,cellTypeIdx]), c=colorArray, s=volcanoMarkerSize, linewidth=0)
# xabs_max = abs(max(oliPlot.get_xlim(), key=abs))
# oliPlot.set_xlim(xmin=-xabs_max, xmax=xabs_max)
# sortedPval = np.argsort(pValList[:,cellTypeIdx])
# # plot only the names of the top ten genes
# for j in enumerate(sortedPval[:10]):
#     if bhCorrList[j[1],cellTypeIdx] == 1:
#         oliPlot.annotate(sampleForDisplay["geneList"][j[1]], (logFCList[j[1],cellTypeIdx], -np.log10(pValList[j[1],cellTypeIdx])), fontsize=volcanoGeneFontSize)
# # for j in enumerate(sigColorList[:,cellTypeIdx]):
# #     if j[1] != 0:
# #         ax[1,1].annotate(sampleForDisplay["geneList"][j[0]], (logFCList[j[0],cellTypeIdx], -np.log10(pValList[j[0],cellTypeIdx])))
# if cellsOfInterest[cellTypeIdx] == 'DG/CA4':
#     oliPlot.set_title('DG hilus', fontweight='bold')
# else:
#     oliPlot.set_title(f'{cellsOfInterest[cellTypeIdx]}', fontweight='bold')

# # endothelial
# cellTypeIdx = 6

# colorArray = np.zeros([sigColorList[:,cellTypeIdx].shape[0], 3])
# greyIdx = np.where(sigColorList[:,cellTypeIdx] == 0)
# blueIdx = np.where(sigColorList[:,cellTypeIdx] == -1)
# redIdx = np.where(sigColorList[:,cellTypeIdx] == 1)
# colorArray[greyIdx, :] = np.array([0.5, 0.5, 0.5])
# colorArray[blueIdx, :] = np.array([31/255, 119/255, 180/255])
# colorArray[redIdx, :] = np.array([1, 0, 0])

# endPlot.scatter(logFCList[:,cellTypeIdx], -np.log10(pValList[:,cellTypeIdx]), c=colorArray, s=volcanoMarkerSize, linewidth=0)
# xabs_max = abs(max(endPlot.get_xlim(), key=abs))
# endPlot.set_xlim(xmin=-xabs_max, xmax=xabs_max)
# sortedPval = np.argsort(pValList[:,cellTypeIdx])
# # plot only the names of the top ten genes
# for j in enumerate(sortedPval[:10]):
#     if bhCorrList[j[1],cellTypeIdx] == 1:
#         endPlot.annotate(sampleForDisplay["geneList"][j[1]], (logFCList[j[1],cellTypeIdx], -np.log10(pValList[j[1],cellTypeIdx])), fontsize=volcanoGeneFontSize)
# # for j in enumerate(sigColorList[:,cellTypeIdx]):
# #     if j[1] != 0:
# #         ax[1,1].annotate(sampleForDisplay["geneList"][j[0]], (logFCList[j[0],cellTypeIdx], -np.log10(pValList[j[0],cellTypeIdx])))
# if cellsOfInterest[cellTypeIdx] == 'DG/CA4':
#     endPlot.set_title('DG hilus', fontweight='bold')
# else:
#     endPlot.set_title(f'{cellsOfInterest[cellTypeIdx]}', fontweight='bold')

# fig.text(0.47, 0.04, 'logFC', ha='center', fontsize=volcanoTitleFontSize, fontweight='bold')
# fig.text(0.04, 0.5, '-log10(p-value)', va='center', rotation='vertical', fontsize=volcanoTitleFontSize, fontweight='bold')

# # fig.text(0.5, 0.04, 'logFC', ha='center')
# # fig.text(0.04, 0.42, '-log10(p-value)', va='center', rotation='vertical')

# plt.show()

### edited locations of gene annotations to prevent overlap in inkscape, so only generate new if necessary
# output pdf and svg
# plt.savefig(os.path.join(figureFolder, f'figure02_combined_volcano_plots_volcano_marker_{volcanoMarkerSize}.pdf'), bbox_inches='tight', dpi=300)
# plt.savefig(os.path.join(figureFolder, f'figure02_combined_volcano_plots_volcano_marker_{volcanoMarkerSize}.svg'), bbox_inches='tight', dpi=300)
# plt.close('all')

#%% look for overlapping genes
sheetNames = pd.ExcelFile(os.path.join(derivatives, 'degs_per_cell-type_and_regions_male_SD_BH.xlsx'))
degDict = {}
for cellType in sheetNames.sheet_names:
    degDict[cellType] = pd.read_excel(os.path.join(derivatives, 'degs_per_cell-type_and_regions_male_SD_BH.xlsx'), sheet_name=cellType)
# remove unlabeled cells from data
del degDict['sparse']

allDegs = np.empty(0)
for cellType in degDict.keys():
    allDegs = np.append(allDegs, np.array(degDict[cellType]['Gene_ID']))
    
allDegs = np.unique(allDegs)

overlappingBool = np.full([len(allDegs), len(degDict)], False)
for cellType in enumerate(degDict):
    for deg in enumerate(allDegs):
        if deg[1] in np.array(degDict[cellType[1]]['Gene_ID']):
            overlappingBool[deg[0], cellType[0]] = True
regionsPerGene = np.sum(overlappingBool, axis=1)
uniqueDegMask = np.where(regionsPerGene == 1)[0]
uniqueDegs = {}
for cellType in enumerate(degDict):
    uniqueDegs[cellType[1]] = []

for cellType in enumerate(degDict):
    for deg in enumerate(allDegs[uniqueDegMask]):   
        if deg[1] in np.array(degDict[cellType[1]]['Gene_ID']):
            uniqueDegs[cellType[1]].append(deg[1])
            
plt.close('all')
if not os.path.exists(os.path.join(derivatives, 'uniqueDEGsPerRegion')):
    os.makedirs(os.path.join(derivatives, 'uniqueDEGsPerRegion'))
for cellType in enumerate(uniqueDegs):
    uniqueGeneList = uniqueDegs[cellType[1]]
    print(len(uniqueGeneList))
    if len(uniqueGeneList) < 1:
        continue
    if not os.path.exists(os.path.join(derivatives, 'uniqueDEGsPerRegion', cellType[1])):
        os.makedirs(os.path.join(derivatives, 'uniqueDEGsPerRegion', cellType[1]))
    for gene in uniqueGeneList:
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
        geneIdx = sampleForDisplay['geneList'].index(gene)
        maxExp = np.max(allCellsMale[geneIdx, :])
        geneSig = ''
        for regionN, region in enumerate(cellsOfInterest):
            regionCellsToPlot = findRelevantClusters(sampleForDisplay, region)
            regionIdx = np.where(allClustersMale == region)[0]
            sdNSDRegionIdx = sdNSDIdxMale[regionIdx]
            regionCells = allCellsMale[:, regionIdx]
            nsdCells = regionCells[:, sdNSDRegionIdx == 0]
            sdCells = regionCells[:, sdNSDRegionIdx == 1]
            tStat = tStatList[geneIdx, regionN]
            pVal = pValList[geneIdx, regionN]
            nsdColor = np.empty([len(regionCellsToPlot)])
            nsdColor[:] = np.mean(nsdCells[geneIdx,:]) 
            sdColor = np.empty([len(regionCellsToPlot)])
            sdColor[:] = np.mean(sdCells[geneIdx,:]) 
            tStatColor = np.empty([len(regionCellsToPlot)])
            tStatColor[:] = tStat
            ax[0].scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=nsdColor, cmap='Reds', s=2, vmin=0, vmax=maxExp, linewidth=0)
            ax[0].set_title('Mean of NSD')
            ax[0].axis('off')
            ax[1].imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
            meanScatter = ax[1].scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=sdColor, cmap='Reds', s=2, vmin=0, vmax=maxExp, linewidth=0)
            ax[1].set_title('Mean of SD')
            ax[1].axis('off')
            ax[2].imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
            tstatScatter = ax[2].scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=tStatColor, cmap='seismic', s=2, vmin=-4, vmax=4, linewidth=0)
            ax[2].set_title('T-statistic for SD > NSD')
            ax[2].axis('off')
        plt.colorbar(meanScatter,fraction=0.02, pad=0.04)
        plt.colorbar(tstatScatter,fraction=0.02, pad=0.04)
        plt.suptitle(f'{gene} in {cellType[1]}')
        plt.show()
        plt.savefig(os.path.join(derivatives, 'uniqueDEGsPerRegion', cellType[1], f'{gene}_mean_and_t-stat.png'), bbox_inches='tight', dpi=300)
        plt.close('all')
        
#%% look for shared DEGs

regionsPerGene = np.sum(overlappingBool, axis=1)
sharedDEGMask = np.where(regionsPerGene > 5)[0]
sharedGeneList = allDegs[sharedDEGMask]
            
plt.close('all')
if not os.path.exists(os.path.join(derivatives, 'sharedDEGsPerRegion')):
    os.makedirs(os.path.join(derivatives, 'sharedDEGsPerRegion'))

for gene in sharedGeneList:
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
    geneIdx = sampleForDisplay['geneList'].index(gene)
    maxExp = np.max(allCellsMale[geneIdx, :])
    for regionN, region in enumerate(cellsOfInterest):
        regionCellsToPlot = findRelevantClusters(sampleForDisplay, region)
        regionIdx = np.where(allClustersMale == region)[0]
        sdNSDRegionIdx = sdNSDIdxMale[regionIdx]
        regionCells = allCellsMale[:, regionIdx]
        nsdCells = regionCells[:, sdNSDRegionIdx == 0]
        sdCells = regionCells[:, sdNSDRegionIdx == 1]
        tStat = tStatList[geneIdx, regionN]
        pVal = pValList[geneIdx, regionN]
        nsdColor = np.empty([len(regionCellsToPlot)])
        nsdColor[:] = np.mean(nsdCells[geneIdx,:]) 
        sdColor = np.empty([len(regionCellsToPlot)])
        sdColor[:] = np.mean(sdCells[geneIdx,:]) 
        tStatColor = np.empty([len(regionCellsToPlot)])
        tStatColor[:] = tStat
        ax[0].scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=nsdColor, cmap='Reds', s=2, vmin=0, vmax=maxExp, linewidth=0)
        ax[0].set_title('Mean of NSD')
        ax[0].axis('off')
        ax[1].imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
        meanScatter = ax[1].scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=sdColor, cmap='Reds', s=2, vmin=0, vmax=maxExp, linewidth=0)
        ax[1].set_title('Mean of SD')
        ax[1].axis('off')
        ax[2].imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
        tstatScatter = ax[2].scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=tStatColor, cmap='seismic', s=2, vmin=-4, vmax=4, linewidth=0)
        ax[2].set_title('T-statistic for SD > NSD')
        ax[2].axis('off')
    plt.colorbar(meanScatter,fraction=0.02, pad=0.04)
    plt.colorbar(tstatScatter,fraction=0.02, pad=0.04)
    plt.suptitle(f'{gene}')
    plt.show()
    plt.savefig(os.path.join(derivatives, 'sharedDEGsPerRegion', f'{gene}_mean_and_t-stat.png'), bbox_inches='tight', dpi=300)
    plt.close('all')
#%% create t-statistic plots for Cirbp and Rbm3 in one figure
# broad changes
gene01 = 'Cirbp'
gene02 = 'Rbm3'
# ast
gene03 = 'Dcn'
# ca1
gene04 = 'Gpr161'
# ca3
gene05 = 'Jdp2'
# DG
gene06 = 'Egr3'
# plot Cirbp t-statistics
plt.close('all')

fig = plt.figure(figsize=(7.2,3.6))
# for subplot2grid: size of plot (rows, columns) plot location (rows, columns) 
gene01NSDMean = plt.subplot2grid((10,19), (0,0), colspan=3, rowspan=2)
gene01SDMean = plt.subplot2grid((10,19), (0,3), colspan=3, rowspan=2)
gene01Tstat = plt.subplot2grid((10,19), (0,6), colspan=3, rowspan=2)

gene02NSDMean = plt.subplot2grid((10,19), (0,10), colspan=3, rowspan=2)
gene02SDMean = plt.subplot2grid((10,19), (0,13), colspan=3, rowspan=2)
gene02Tstat = plt.subplot2grid((10,19), (0,16), colspan=3, rowspan=2)

gene03NSDMean = plt.subplot2grid((10,19), (3,0), colspan=3, rowspan=2)
gene03SDMean = plt.subplot2grid((10,19), (3,3), colspan=3, rowspan=2)
gene03Tstat = plt.subplot2grid((10,19), (3,6), colspan=3, rowspan=2)

gene04NSDMean = plt.subplot2grid((10,19), (3,10), colspan=3, rowspan=2)
gene04SDMean = plt.subplot2grid((10,19), (3,13), colspan=3, rowspan=2)
gene04Tstat = plt.subplot2grid((10,19), (3,16), colspan=3, rowspan=2)

gene05NSDMean = plt.subplot2grid((10,19), (6,0), colspan=3, rowspan=2)
gene05SDMean = plt.subplot2grid((10,19), (6,3), colspan=3, rowspan=2)
gene05Tstat = plt.subplot2grid((10,19), (6,6), colspan=3, rowspan=2)

gene06NSDMean = plt.subplot2grid((10,19), (6,10), colspan=3, rowspan=2)
gene06SDMean = plt.subplot2grid((10,19), (6,13), colspan=3, rowspan=2)
gene06Tstat = plt.subplot2grid((10,19), (6,16), colspan=3, rowspan=2)

# gene 1, top left
gene01NSDMean.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
gene01SDMean.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
gene01Tstat.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')

geneIdx = sampleForDisplay['geneList'].index(gene01)
maxExp = np.max(regionCells[geneIdx, :])
geneSig = ''
for regionN, region in enumerate(cellsOfInterest):
    regionCellsToPlot = findRelevantClusters(sampleForDisplay, region)
    regionIdx = np.where(allClustersMale == region)[0]
    sdNSDRegionIdx = sdNSDIdxMale[regionIdx]
    regionCells = allCellsMale[:, regionIdx]
    nsdCells = regionCells[:, sdNSDRegionIdx == 0]
    sdCells = regionCells[:, sdNSDRegionIdx == 1]
    tStat = tStatList[geneIdx, regionN]
    pVal = pValList[geneIdx, regionN]
    if pVal < alphaFdr:
        geneSig += '*'
    nsdColor = np.empty([len(regionCellsToPlot)])
    nsdColor[:] = np.mean(nsdCells[geneIdx,:]) 
    sdColor = np.empty([len(regionCellsToPlot)])
    sdColor[:] = np.mean(sdCells[geneIdx,:]) 
    tStatColor = np.empty([len(regionCellsToPlot)])
    tStatColor[:] = tStat
    gene01NSDMean.scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=nsdColor, cmap='Reds', s=marker_size, linewidth=0, vmin=0, vmax=maxExp)
    gene01NSDMean.set_title('Mean of NSD', fontweight='bold')
    gene01NSDMean.set_xticks([])
    gene01NSDMean.set_yticks([])
    # gene01NSDMean.axis('off')
    
    meanScatter = gene01SDMean.scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=sdColor, cmap='Reds', s=marker_size, linewidth=0, vmin=0, vmax=maxExp)
    gene01SDMean.set_title('Mean of SD', fontweight='bold')
    gene01SDMean.set_xticks([])
    gene01SDMean.set_yticks([])
    
    tstatScatter = gene01Tstat.scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=tStatColor, cmap='seismic', s=marker_size, linewidth=0, vmin=-4, vmax=4)
    gene01Tstat.set_title('T-statistic for SD > NSD', fontweight='bold')
    gene01Tstat.set_xticks([])
    gene01Tstat.set_yticks([])
gene01NSDMean.set_ylabel(gene01)
plt.colorbar(meanScatter, ax=[gene01NSDMean, gene01SDMean], orientation='horizontal', fraction=0.05, pad=0.04)
plt.colorbar(tstatScatter, orientation='horizontal',fraction=0.05, pad=0.04)

# gene 2, top right
gene02NSDMean.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
gene02SDMean.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
gene02Tstat.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')

geneIdx = sampleForDisplay['geneList'].index(gene02)
maxExp = np.max(regionCells[geneIdx, :])
geneSig = ''
for regionN, region in enumerate(cellsOfInterest):
    regionCellsToPlot = findRelevantClusters(sampleForDisplay, region)
    regionIdx = np.where(allClustersMale == region)[0]
    sdNSDRegionIdx = sdNSDIdxMale[regionIdx]
    regionCells = allCellsMale[:, regionIdx]
    nsdCells = regionCells[:, sdNSDRegionIdx == 0]
    sdCells = regionCells[:, sdNSDRegionIdx == 1]
    tStat = tStatList[geneIdx, regionN]
    pVal = pValList[geneIdx, regionN]
    if pVal < alphaFdr:
        geneSig += '*'
    nsdColor = np.empty([len(regionCellsToPlot)])
    nsdColor[:] = np.mean(nsdCells[geneIdx,:]) 
    sdColor = np.empty([len(regionCellsToPlot)])
    sdColor[:] = np.mean(sdCells[geneIdx,:]) 
    tStatColor = np.empty([len(regionCellsToPlot)])
    tStatColor[:] = tStat
    gene02NSDMean.scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=nsdColor, cmap='Reds', s=marker_size, linewidth=0, vmin=0, vmax=maxExp)
    gene02NSDMean.set_title('Mean of NSD', fontweight='bold')
    gene02NSDMean.set_xticks([])
    gene02NSDMean.set_yticks([])
    
    meanScatter = gene02SDMean.scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=sdColor, cmap='Reds', s=marker_size, linewidth=0, vmin=0, vmax=maxExp)
    gene02SDMean.set_title('Mean of SD', fontweight='bold')
    gene02SDMean.set_xticks([])
    gene02SDMean.set_yticks([])
    
    tstatScatter = gene02Tstat.scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=tStatColor, cmap='seismic', s=marker_size, linewidth=0, vmin=-4, vmax=4)
    gene02Tstat.set_title('T-statistic for SD > NSD', fontweight='bold')
    gene02Tstat.set_xticks([])
    gene02Tstat.set_yticks([])
gene02NSDMean.set_ylabel(gene02)
plt.colorbar(meanScatter, ax=[gene02NSDMean, gene02SDMean], orientation='horizontal', fraction=0.05, pad=0.04)
plt.colorbar(tstatScatter, orientation='horizontal',fraction=0.05, pad=0.04)

# gene 3, middle left
gene03NSDMean.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
gene03SDMean.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
gene03Tstat.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')

geneIdx = sampleForDisplay['geneList'].index(gene03)
maxExp = np.max(regionCells[geneIdx, :])
geneSig = ''
for regionN, region in enumerate(cellsOfInterest):
    regionCellsToPlot = findRelevantClusters(sampleForDisplay, region)
    regionIdx = np.where(allClustersMale == region)[0]
    sdNSDRegionIdx = sdNSDIdxMale[regionIdx]
    regionCells = allCellsMale[:, regionIdx]
    nsdCells = regionCells[:, sdNSDRegionIdx == 0]
    sdCells = regionCells[:, sdNSDRegionIdx == 1]
    tStat = tStatList[geneIdx, regionN]
    pVal = pValList[geneIdx, regionN]
    if pVal < alphaFdr:
        geneSig += '*'
    nsdColor = np.empty([len(regionCellsToPlot)])
    nsdColor[:] = np.mean(nsdCells[geneIdx,:]) 
    sdColor = np.empty([len(regionCellsToPlot)])
    sdColor[:] = np.mean(sdCells[geneIdx,:]) 
    tStatColor = np.empty([len(regionCellsToPlot)])
    tStatColor[:] = tStat
    gene03NSDMean.scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=nsdColor, cmap='Reds', s=marker_size, linewidth=0, vmin=0, vmax=maxExp)
    gene03NSDMean.set_title('Mean of NSD', fontweight='bold')
    gene03NSDMean.set_xticks([])
    gene03NSDMean.set_yticks([])
    
    meanScatter = gene03SDMean.scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=sdColor, cmap='Reds', s=marker_size, linewidth=0, vmin=0, vmax=maxExp)
    gene03SDMean.set_title('Mean of SD', fontweight='bold')
    gene03SDMean.set_xticks([])
    gene03SDMean.set_yticks([])
    
    tstatScatter = gene03Tstat.scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=tStatColor, cmap='seismic', s=marker_size, linewidth=0, vmin=-4, vmax=4)
    gene03Tstat.set_title('T-statistic for SD > NSD', fontweight='bold')
    gene03Tstat.set_xticks([])
    gene03Tstat.set_yticks([])
gene03NSDMean.set_ylabel(gene03)
plt.colorbar(meanScatter, ax=[gene03NSDMean, gene03SDMean], orientation='horizontal', fraction=0.05, pad=0.04)
plt.colorbar(tstatScatter, orientation='horizontal',fraction=0.05, pad=0.04)

# gene 4, middle right
gene04NSDMean.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
gene04SDMean.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
gene04Tstat.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')

geneIdx = sampleForDisplay['geneList'].index(gene04)
maxExp = np.max(regionCells[geneIdx, :])
geneSig = ''
for regionN, region in enumerate(cellsOfInterest):
    regionCellsToPlot = findRelevantClusters(sampleForDisplay, region)
    regionIdx = np.where(allClustersMale == region)[0]
    sdNSDRegionIdx = sdNSDIdxMale[regionIdx]
    regionCells = allCellsMale[:, regionIdx]
    nsdCells = regionCells[:, sdNSDRegionIdx == 0]
    sdCells = regionCells[:, sdNSDRegionIdx == 1]
    tStat = tStatList[geneIdx, regionN]
    pVal = pValList[geneIdx, regionN]
    if pVal < alphaFdr:
        geneSig += '*'
    nsdColor = np.empty([len(regionCellsToPlot)])
    nsdColor[:] = np.mean(nsdCells[geneIdx,:]) 
    sdColor = np.empty([len(regionCellsToPlot)])
    sdColor[:] = np.mean(sdCells[geneIdx,:]) 
    tStatColor = np.empty([len(regionCellsToPlot)])
    tStatColor[:] = tStat
    gene04NSDMean.scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=nsdColor, cmap='Reds', s=marker_size, linewidth=0, vmin=0, vmax=maxExp)
    gene04NSDMean.set_title('Mean of NSD', fontweight='bold')
    gene04NSDMean.set_xticks([])
    gene04NSDMean.set_yticks([])
    
    meanScatter = gene04SDMean.scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=sdColor, cmap='Reds', s=marker_size, linewidth=0, vmin=0, vmax=maxExp)
    gene04SDMean.set_title('Mean of SD', fontweight='bold')
    gene04SDMean.set_xticks([])
    gene04SDMean.set_yticks([])
    
    tstatScatter = gene04Tstat.scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=tStatColor, cmap='seismic', s=marker_size, linewidth=0, vmin=-4, vmax=4)
    gene04Tstat.set_title('T-statistic for SD > NSD', fontweight='bold')
    gene04Tstat.set_xticks([])
    gene04Tstat.set_yticks([])
gene04NSDMean.set_ylabel(gene04)
plt.colorbar(meanScatter, ax=[gene04NSDMean, gene04SDMean], orientation='horizontal', fraction=0.05, pad=0.04)
plt.colorbar(tstatScatter, orientation='horizontal',fraction=0.05, pad=0.04)

# gene 5, bottom left
gene05NSDMean.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
gene05SDMean.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
gene05Tstat.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')

geneIdx = sampleForDisplay['geneList'].index(gene05)
maxExp = np.max(regionCells[geneIdx, :])
geneSig = ''
for regionN, region in enumerate(cellsOfInterest):
    regionCellsToPlot = findRelevantClusters(sampleForDisplay, region)
    regionIdx = np.where(allClustersMale == region)[0]
    sdNSDRegionIdx = sdNSDIdxMale[regionIdx]
    regionCells = allCellsMale[:, regionIdx]
    nsdCells = regionCells[:, sdNSDRegionIdx == 0]
    sdCells = regionCells[:, sdNSDRegionIdx == 1]
    tStat = tStatList[geneIdx, regionN]
    pVal = pValList[geneIdx, regionN]
    if pVal < alphaFdr:
        geneSig += '*'
    nsdColor = np.empty([len(regionCellsToPlot)])
    nsdColor[:] = np.mean(nsdCells[geneIdx,:]) 
    sdColor = np.empty([len(regionCellsToPlot)])
    sdColor[:] = np.mean(sdCells[geneIdx,:]) 
    tStatColor = np.empty([len(regionCellsToPlot)])
    tStatColor[:] = tStat
    gene05NSDMean.scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=nsdColor, cmap='Reds', s=marker_size, linewidth=0, vmin=0, vmax=maxExp)
    gene05NSDMean.set_title('Mean of NSD')
    gene05NSDMean.set_xticks([])
    gene05NSDMean.set_yticks([])
    
    meanScatter = gene05SDMean.scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=sdColor, cmap='Reds', s=marker_size, linewidth=0, vmin=0, vmax=maxExp)
    gene05SDMean.set_title('Mean of SD', fontweight='bold')
    gene05SDMean.set_xticks([])
    gene05SDMean.set_yticks([])
    
    tstatScatter = gene05Tstat.scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=tStatColor, cmap='seismic', s=marker_size, linewidth=0, vmin=-4, vmax=4)
    gene05Tstat.set_title('T-statistic for SD > NSD', fontweight='bold')
    gene05Tstat.set_xticks([])
    gene05Tstat.set_yticks([])
gene05NSDMean.set_ylabel(gene05)
plt.colorbar(meanScatter, ax=[gene05NSDMean, gene05SDMean], orientation='horizontal', fraction=0.05, pad=0.04)
plt.colorbar(tstatScatter, orientation='horizontal',fraction=0.05, pad=0.04)

# gene 6, bottom right
gene06NSDMean.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
gene06SDMean.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
gene06Tstat.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')

geneIdx = sampleForDisplay['geneList'].index(gene06)
maxExp = np.max(regionCells[geneIdx, :])
geneSig = ''
for regionN, region in enumerate(cellsOfInterest):
    regionCellsToPlot = findRelevantClusters(sampleForDisplay, region)
    regionIdx = np.where(allClustersMale == region)[0]
    sdNSDRegionIdx = sdNSDIdxMale[regionIdx]
    regionCells = allCellsMale[:, regionIdx]
    nsdCells = regionCells[:, sdNSDRegionIdx == 0]
    sdCells = regionCells[:, sdNSDRegionIdx == 1]
    tStat = tStatList[geneIdx, regionN]
    pVal = pValList[geneIdx, regionN]
    if pVal < alphaFdr:
        geneSig += '*'
    nsdColor = np.empty([len(regionCellsToPlot)])
    nsdColor[:] = np.mean(nsdCells[geneIdx,:]) 
    sdColor = np.empty([len(regionCellsToPlot)])
    sdColor[:] = np.mean(sdCells[geneIdx,:]) 
    tStatColor = np.empty([len(regionCellsToPlot)])
    tStatColor[:] = tStat
    gene06NSDMean.scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=nsdColor, cmap='Reds', s=marker_size, linewidth=0, vmin=0, vmax=maxExp)
    gene06NSDMean.set_title('Mean of NSD', fontweight='bold')
    gene06NSDMean.set_xticks([])
    gene06NSDMean.set_yticks([])
    
    meanScatter = gene06SDMean.scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=sdColor, cmap='Reds', s=marker_size, linewidth=0, vmin=0, vmax=maxExp)
    gene06SDMean.set_title('Mean of SD', fontweight='bold')
    gene06SDMean.set_xticks([])
    gene06SDMean.set_yticks([])
    
    tstatScatter = gene06Tstat.scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=tStatColor, cmap='seismic', s=marker_size, linewidth=0, vmin=-4, vmax=4)
    gene06Tstat.set_title('T-statistic for SD > NSD', fontweight='bold')
    gene06Tstat.set_xticks([])
    gene06Tstat.set_yticks([])
gene06NSDMean.set_ylabel(gene06)
plt.colorbar(meanScatter, ax=[gene06NSDMean, gene06SDMean], orientation='horizontal', fraction=0.05, pad=0.04)
plt.colorbar(tstatScatter, orientation='horizontal',fraction=0.05, pad=0.04)

plt.show()

# output pdf and svg
# plt.savefig(os.path.join(figureFolder, 'figure04_sig_gene_t-statistic.pdf'), bbox_inches='tight', dpi=300)
# plt.savefig(os.path.join(figureFolder, 'figure04_sig_gene_t-statistic.svg'), bbox_inches='tight', dpi=300)
# plt.close('all')

