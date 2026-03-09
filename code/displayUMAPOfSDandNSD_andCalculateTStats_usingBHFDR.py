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
import matplotlib.cm as cm
import scipy.sparse as sp_sparse
import scipy
from sklearn.cluster import KMeans

rawdata=os.path.join('/','media','zjpeters','Expansion','sleepDepXenium','rawdata')
derivatives=os.path.join('/','media','zjpeters','Expansion','sleepDepXeniumHippocampus','derivatives')

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
plt.close('all')

plt.figure()
plt.imshow(sampleToCluster['tissueImageProcessed'],cmap='gray_r')
plt.scatter(sampleToCluster['processedTissuePositionList'][:,0], sampleToCluster['processedTissuePositionList'][:,1],c=sampleToCluster['cluster_colors'], s=5)
plt.title("The visualization of the clustered control data.")
plt.axis('off')
plt.show()
for i in range(actK):
    plt.figure()
    clusterIdx = np.where(sampleToCluster['cluster_labels'] == i)[0]
    plt.imshow(sampleToCluster['tissueImageProcessed'],cmap='gray_r')
    plt.scatter(sampleToCluster['processedTissuePositionList'][clusterIdx,0], sampleToCluster['processedTissuePositionList'][clusterIdx,1],c=sampleToCluster['cluster_colors'][clusterIdx], s=5)
    plt.title(f"{sampleToCluster['sampleID']} cluster {i}.")
    plt.axis('off')
    plt.show()

#%% code to restrict cells to only those clustered to a particular label

def findRelevantClusters(processedSample, cluster_region):
    matchingClusters = np.where(processedSample['cluster_region'] == cluster_region)[0]
    clusterMask = []
    for cellIdx in range(len(processedSample['cluster_labels'])):
        # since it's possible that more than one cluster correspond to cluster of interest, use for loop
        for clusterIdx in matchingClusters:
            if processedSample['cluster_labels'][cellIdx] == clusterIdx:
                clusterMask.append(cellIdx)
    return clusterMask

ca3Mask = findRelevantClusters(processedSamples[0], 'CA3')
plt.figure()
plt.imshow(processedSamples[0]['tissueImageProcessed'], cmap='gray_r')
plt.scatter(processedSamples[0]['processedTissuePositionList'][ca3Mask,0],processedSamples[0]['processedTissuePositionList'][ca3Mask,1])
plt.show()
#%% generate male samples
maleSamples = []
maleSamplesIdx = [0,1,2,3,4,5]
for i in maleSamplesIdx:
    maleSamples.append(processedSamples[i])

femaleSamples = []
femaleSamplesIdx = [6,7,8,9,10,11]
for i in femaleSamplesIdx:
    femaleSamples.append(processedSamples[i])

designMatrix = [0,1,0,1,0,1]
#%% add cell types to the umap plotting

for sampleIdx in range(len(maleSamples)):
    cluster_regions = pd.read_csv(os.path.join(derivatives, 'clusterAssociationCsvs', f'{maleSamples[sampleIdx]["sampleID"]}_cluster_associations_cell_types.csv'), header=None)
    maleSamples[sampleIdx]['cluster_region'] = np.squeeze(np.array(cluster_regions[1]))
    
for sampleIdx in range(len(femaleSamples)):
    cluster_regions = pd.read_csv(os.path.join(derivatives, 'clusterAssociationCsvs', f'{femaleSamples[sampleIdx]["sampleID"]}_cluster_associations_cell_types.csv'), header=None)
    femaleSamples[sampleIdx]['cluster_region'] = np.squeeze(np.array(cluster_regions[1]))

#%% run umap embedding using all cells, SD and NSD
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

allCellsFemale = np.empty([len(femaleSamples[0]['geneList']),0])
allClustersFemale = np.empty(0, dtype='str')
sdNSDIdxFemale = np.empty(0, dtype='int32')
sampleListFemale = np.empty(0,dtype='int32')
allCoordinatesFemale = np.empty([0,2])
for actSample in range(len(femaleSamples)):
    allCellsFemale = np.append(allCellsFemale, femaleSamples[actSample]['geneMatrixLog2'].todense(), axis=1)
    clusterListFemale = [] #np.empty([len(maleSamples[actSample]['cluster_labels']),1], dtype='str')
    for cluster in femaleSamples[actSample]['cluster_labels'].to_numpy():
        clusterListFemale.append(femaleSamples[actSample]['cluster_region'][cluster])
    allClustersFemale = np.append(allClustersFemale, clusterListFemale)
    # binary distinction of each cell for sd or nsd
    if designMatrix[actSample] == 0:
        sdNSDIdxFemale = np.append(sdNSDIdxFemale, np.zeros(len(femaleSamples[actSample]['processedTissuePositionList']), dtype='int32').T)
    elif designMatrix[actSample] == 1:
        sdNSDIdxFemale = np.append(sdNSDIdxFemale, np.ones(len(femaleSamples[actSample]['processedTissuePositionList']), dtype='int32').T)
    sampleListFemale = np.append(sampleListFemale, np.zeros(len(femaleSamples[actSample]['processedTissuePositionList']), dtype='int32').T + actSample)
    allCoordinatesFemale = np.append(allCoordinatesFemale, femaleSamples[actSample]['processedTissuePositionList'], axis=0)
    sampleColorsFemale = femaleSamples[actSample]['cluster_colors']
clusterListFemale = np.array(clusterListFemale, dtype='str')
#%% put together color list
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
sparseColor = np.squeeze(np.array([0.7,0.7,0.7,1]))
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
    sampleColors[neuIdx,:] = sparseColor
    # sampleColors[neuIdx,:] = neuColor
    oliIdx = findRelevantClusters(maleSamples[actSample], 'oligodendrocytes')
    sampleColors[oliIdx,:] = oliColor
    sparseIdx = findRelevantClusters(maleSamples[actSample], 'sparse')
    sampleColors[sparseIdx,:] = sparseColor
    intIdx = np.where(maleSamples[actSample]['cluster_labels'] == 15)[0]
    sampleColors[intIdx, :] = neuColor
    colorsAllCellsMale = np.append(colorsAllCellsMale, sampleColors, axis=0)
        
# controlAllCellsFemale = np.empty([len(femaleSamples[0]['geneList']),0])
colorsAllCellsFemale = np.empty([0,4])
for actSample in range(len(femaleSamples)):
    # controlAllCellsFemale = np.append(controlAllCellsFemale, femaleSamples[actSample]['geneMatrixLog2'].todense(), axis=1)
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
    astIdx = findRelevantClusters(femaleSamples[actSample], 'astrocytes')
    sampleColors[astIdx,:] = astColor
    endIdx = findRelevantClusters(femaleSamples[actSample], 'endothelial')
    sampleColors[endIdx,:] = endColor
    micIdx = findRelevantClusters(femaleSamples[actSample], 'microglia')
    sampleColors[micIdx,:] = micColor
    neuIdx = findRelevantClusters(femaleSamples[actSample], 'neurons')
    sampleColors[neuIdx,:] = sparseColor
    # sampleColors[neuIdx,:] = neuColor
    oliIdx = findRelevantClusters(femaleSamples[actSample], 'oligodendrocytes')
    sampleColors[oliIdx,:] = oliColor
    sparseIdx = findRelevantClusters(femaleSamples[actSample], 'sparse')
    sampleColors[sparseIdx,:] = sparseColor
    intIdx = np.where(femaleSamples[actSample]['cluster_labels'] == 15)[0]
    sampleColors[intIdx, :] = neuColor
    colorsAllCellsFemale = np.append(colorsAllCellsFemale, sampleColors, axis=0)
#%% perform umap plot on combined cells

reducer = umap.UMAP()

embeddingMale = reducer.fit_transform(np.array(allCellsMale).T)
embeddingFemale = reducer.fit_transform(np.array(allCellsFemale).T)


#%% create a plot of umap for SD, umap for NSD, and UMAP for all cells in male samples
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Arial'
plt.close('all')

fig = plt.figure(figsize=(16,9))
umapAxsNSD = plt.subplot2grid((4,9), (0,0), colspan=3, rowspan=4)
umapAxsSD = plt.subplot2grid((4,9), (0,3), colspan=3, rowspan=4)
umapAxsAllCells = plt.subplot2grid((4,9), (0,6), colspan=3, rowspan=4)
# plot the NSD samples
nsdIdx = np.where(sdNSDIdxMale == 0)[0]
umapAxsNSD.scatter(embeddingMale[nsdIdx,0], embeddingMale[nsdIdx,1], c=colorsAllCellsMale[nsdIdx,:], s=3)
umapAxsNSD.set_title("UMAP of male NSD samples")
umapAxsNSD.set_xticks([])
umapAxsNSD.set_yticks([])

# plot the SD samples
sdIdx = np.where(sdNSDIdxMale == 1)[0]
umapAxsSD.scatter(embeddingMale[sdIdx,0], embeddingMale[sdIdx,1], c=colorsAllCellsMale[sdIdx,:], s=3)
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

umapAxsSD.legend(handles=handles, ncols=2)

# plot all cells
umapAxsAllCells.scatter(embeddingMale[:,0], embeddingMale[:,1], c=colorsAllCellsMale, s=3)
umapAxsAllCells.set_title("UMAP of all male samples")
umapAxsAllCells.set_xticks([])
umapAxsAllCells.set_yticks([])

plt.show()

### uncomment below to save new figure
# plt.savefig(os.path.join(derivatives, 'xenium_figure_k15cluster_umap_with_sd_nsd_split_males.pdf'), bbox_inches='tight', dpi=300)

#%% create a plot of umap for SD, umap for NSD, and UMAP for all cells in female samples
plt.close('all')

fig = plt.figure(figsize=(16,9))
umapAxsNSD = plt.subplot2grid((4,9), (0,0), colspan=3, rowspan=4)
umapAxsSD = plt.subplot2grid((4,9), (0,3), colspan=3, rowspan=4)
umapAxsAllCells = plt.subplot2grid((4,9), (0,6), colspan=3, rowspan=4)
# plot the NSD samples
nsdIdx = np.where(sdNSDIdxFemale == 0)[0]
umapAxsNSD.scatter(embeddingFemale[nsdIdx,0], embeddingFemale[nsdIdx,1], c=colorsAllCellsFemale[nsdIdx,:], s=3)
umapAxsNSD.set_title("UMAP of Female NSD samples")
umapAxsNSD.set_xticks([])
umapAxsNSD.set_yticks([])

# plot the SD samples
sdIdx = np.where(sdNSDIdxFemale == 1)[0]
umapAxsSD.scatter(embeddingFemale[sdIdx,0], embeddingFemale[sdIdx,1], c=colorsAllCellsFemale[sdIdx,:], s=3)
umapAxsSD.set_title("UMAP of Female SD samples")
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

umapAxsSD.legend(handles=handles, ncols=2)

# plot all cells
umapAxsAllCells.scatter(embeddingFemale[:,0], embeddingFemale[:,1], c=colorsAllCellsFemale, s=3)
umapAxsAllCells.set_title("UMAP of all Female samples")
umapAxsAllCells.set_xticks([])
umapAxsAllCells.set_yticks([])

plt.show()

### uncomment below to save new figure
# plt.savefig(os.path.join(derivatives, 'xenium_figure_k15cluster_umap_with_sd_nsd_split_females.pdf'), bbox_inches='tight', dpi=300)

#%% try to perform umap and clustering of CA1 data from all samples
plt.close('all')
maleCA1Idx = np.where(allClustersMale == 'CA1')[0]
ca1MaleEmbedding = reducer.fit_transform(np.array(allCellsMale[:,maleCA1Idx]).T)
plt.figure()
plt.scatter(ca1MaleEmbedding[:,0], ca1MaleEmbedding[:,1], s=3)
plt.show()

#%% umap doesn't seem to discriminate between ca1 subregions, try spectral clustering
simMatrixSavePath = os.path.join(derivatives, 'similarityMatrix_all_male_CA1_cells.npz')
ca1MaleGeneMatrix = allCellsMale[:,maleCA1Idx]
# check whether similarity matrix has already been calculated
if os.path.exists(simMatrixSavePath):
    similarityMatrix = sp_sparse.load_npz(simMatrixSavePath)
else:
    similarityMatrix = stanly.measureTranscriptomicSimilarity(ca1MaleGeneMatrix, axis=1, denseMatrix=True)
    sp_sparse.save_npz(simMatrixSavePath, similarityMatrix) 

#%% cluster using similarity matrix calculated above
k = 8
Wcontrol = similarityMatrix.todense()

#% create laplacian for control
Wcontrol = (Wcontrol - np.nanmin(Wcontrol))/(np.nanmax(Wcontrol) - np.nanmin(Wcontrol))
Wcontrol[Wcontrol==1] = 0
# Wcontrol[np.isnan(Wcontrol)] = 0
Dcontrol = np.diag(sum(Wcontrol))
Lcontrol = Dcontrol - Wcontrol
eigvalControl,eigvecControl = scipy.sparse.linalg.eigs(Lcontrol, k=550)
eigvalControlSort = np.sort(np.real(eigvalControl))[::-1]
eigvalControlSortIdx = np.argsort(np.real(eigvalControl))[::-1]
eigvecControlSort = np.real(eigvecControl[:,eigvalControlSortIdx])
clusters = KMeans(n_clusters=k, init='random', n_init=500, tol=1e-8,)
cluster_labels = clusters.fit_predict(np.real(np.array(eigvecControlSort)[:,0:k+1]))
colors = cm.tab20b(cluster_labels.astype(float) / k)
#%% plot the ca1 clusters on top of sample images

maleCA1Coordinates = allCoordinatesMale[maleCA1Idx,:]
sampleListMaleCA1 = sampleListMale[maleCA1Idx]
fig, ax = plt.subplots(2,3)
nR = 0
nC = 0
for i in range(len(maleSamples)):
    sampleCA1Idx = np.where(sampleListMaleCA1 == i)[0]
    sampleCA1Coordinates = maleCA1Coordinates[sampleCA1Idx, :]
    sampleCA1Clusters = cluster_labels[sampleCA1Idx]
    sampleCA1Colors = colors[sampleCA1Idx]
    ax[nR,nC].imshow(maleSamples[i]['tissueImageProcessed'], cmap='gray_r')
    ax[nR,nC].scatter(sampleCA1Coordinates[:,0], sampleCA1Coordinates[:,1], c=sampleCA1Colors, s=3)
    if nC < 2:
        nC += 1
    else:
        nC = 0
        nR += 1
plt.show()

#%% try running different k in clustering CA1
plt.close('all')
k = 8
for k in range(2,10):
    Wcontrol = similarityMatrix.todense()
    
    #% create laplacian for control
    Wcontrol = (Wcontrol - np.nanmin(Wcontrol))/(np.nanmax(Wcontrol) - np.nanmin(Wcontrol))
    Wcontrol[Wcontrol==1] = 0
    # Wcontrol[np.isnan(Wcontrol)] = 0
    Dcontrol = np.diag(sum(Wcontrol))
    Lcontrol = Dcontrol - Wcontrol
    eigvalControl,eigvecControl = scipy.sparse.linalg.eigs(Lcontrol, k=297)
    eigvalControlSort = np.sort(np.real(eigvalControl))[::-1]
    eigvalControlSortIdx = np.argsort(np.real(eigvalControl))[::-1]
    eigvecControlSort = np.real(eigvecControl[:,eigvalControlSortIdx])
    clusters = KMeans(n_clusters=k, init='random', n_init=500, tol=1e-8,)
    cluster_labels = clusters.fit_predict(np.real(np.array(eigvecControlSort)[:,0:k+1]))
    colors = cm.tab20b(cluster_labels.astype(float) / k)
    # plot umap using clustering colors
    plt.figure()
    plt.scatter(ca1MaleEmbedding[:,0], ca1MaleEmbedding[:,1], c=colors, s=3)
    plt.show()
    plt.savefig(os.path.join(derivatives, 'clusterCA1Males', f'cluster_ca1_males_k{k}_umap.pdf'), bbox_inches='tight', dpi=300)
    plt.title(f'UMAP for CA1 with labels for k={k}')
    plt.close()
    # plot the ca1 clusters on top of sample images    
    maleCA1Coordinates = allCoordinatesMale[maleCA1Idx,:]
    sampleListMaleCA1 = sampleListMale[maleCA1Idx]
    # plt.figure(figsize=(16,6))
    fig, ax = plt.subplots(2,3, figsize=(16,6))
    nR = 0
    nC = 0
    for i in range(len(maleSamples)):
        sampleCA1Idx = np.where(sampleListMaleCA1 == i)[0]
        sampleCA1Coordinates = maleCA1Coordinates[sampleCA1Idx, :]
        sampleCA1Clusters = cluster_labels[sampleCA1Idx]
        sampleCA1Colors = colors[sampleCA1Idx]
        ax[nR,nC].imshow(maleSamples[i]['tissueImageProcessed'], cmap='gray_r')
        ax[nR,nC].scatter(sampleCA1Coordinates[:,0], sampleCA1Coordinates[:,1], c=sampleCA1Colors, s=1, marker='.')
        if designMatrix[i] == 0:
            sdStatus = 'SD'
        else:
            sdStatus = 'NSD'
        ax[nR,nC].set_title(f'{maleSamples[i]["sampleID"]}, {sdStatus}')
        ax[nR,nC].axis('off')
        if nC < 2:
            nC += 1
        else:
            nC = 0
            nR += 1
    plt.suptitle(f'Clustering with k={k}')
    plt.show()
    plt.savefig(os.path.join(derivatives, 'clusterCA1Males', f'cluster_ca1_males_k{k}.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

#%% try to perform umap and clustering of CA1 data from all samples
plt.close('all')
ca1MaleEmbedding = reducer.fit_transform(np.array(allCellsMale[:,maleCA1Idx]).T)
plt.figure()
plt.scatter(ca1MaleEmbedding[:,0], ca1MaleEmbedding[:,1], c=colors, s=3)
plt.show()
#%% perform t-test on each region/cell type
cellsOfInterest = ['CA1', 'CA2', 'CA3', 'DG', 'DG/CA4', 'astrocytes', 'endothelial', 'microglia', 'neurons', 'oligodendrocytes', 'sparse']
# use the first sample as the one to plot to
sampleForDisplay = maleSamples[0]
desiredPval = 0.05
alphaFdr = 1 - np.power((1 - desiredPval),(1/(len(maleSamples[0]["geneList"])*len(cellsOfInterest))))
plt.close('all')
sigGenesPerCells = {}

# prepare lists for BH fdr
tStatList = np.empty([len(sampleForDisplay['geneList']), len(cellsOfInterest)])
pValList = np.empty([len(sampleForDisplay['geneList']), len(cellsOfInterest)])

# loop over each of the regions/cell types and perform a t-test and plot results
for regionN, region in enumerate(cellsOfInterest):
    regionCellsToPlot = findRelevantClusters(sampleForDisplay, region)
    regionIdx = np.where(allClustersMale == region)[0]
    sdNSDRegionIdx = sdNSDIdxMale[regionIdx]
    regionCells = allCellsMale[:, regionIdx]
    nsdCells = regionCells[:, sdNSDRegionIdx == 0]
    sdCells = regionCells[:, sdNSDRegionIdx == 1]
    sigGenesPerCells[region] = []
    for gene in range(len(sampleToCluster['geneList'])):
        tStat, pVal = scipy_stats.ttest_ind(np.squeeze(np.array(sdCells[gene,:])), np.squeeze(np.array(nsdCells[gene,:])))
        tStatList[gene, regionN] = tStat
        pValList[gene, regionN] = pVal
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
            plt.suptitle(f'T-statistic for gene: {sampleForDisplay["geneList"][gene]} \n in {region}')
            plt.show()
            if region == "DG/CA4":
                plt.savefig(os.path.join(derivatives, 'clusterDEGsMales', f'ttest_nsd_sd_males_{sampleForDisplay["geneList"][gene]}_in_DG-CA4.png'), bbox_inches='tight', dpi=300)
            else:
                plt.savefig(os.path.join(derivatives, 'clusterDEGsMales', f'ttest_nsd_sd_males_{sampleForDisplay["geneList"][gene]}_in_{region}.png'), bbox_inches='tight', dpi=300)
            plt.close()
            sigGenesPerCells[region].append(sampleForDisplay["geneList"][gene])

#%% perform BH fdr correction
q = 0.05
writer = pd.ExcelWriter(os.path.join(derivatives, 'degs_per_cell-type_and_regions_male_SD_BH.xlsx'))
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
    for geneIdx in sortedIdx[0:p_idx]:
        GeneID.append(sampleForDisplay['geneList'][geneIdx])
        BH_FDR.append(pValList[geneIdx,cellTypeIdx])
    GeneID = np.array(GeneID).T
    BH_FDR = np.array(BH_FDR).T
    cellTypeDF = pd.DataFrame({"Gene_ID": GeneID, "BH_FDR": BH_FDR})
    if cellType == 'DG/CA4':
        cellTypeDF.to_excel(writer, sheet_name='DG-CA4', index=False)
    else:
        cellTypeDF.to_excel(writer, sheet_name=cellType, index=False)
# writer.save()
writer.close()
#%% write sig genes to excel file
writer = pd.ExcelWriter(os.path.join(derivatives, 'degs_per_cell-type_and_regions_male_SD_sidak.xlsx'))
for i in sigGenesPerCells.keys():
    cellTypeDF = pd.DataFrame(sigGenesPerCells[i])
    if i == 'DG/CA4':
        cellTypeDF.to_excel(writer, sheet_name='DG-CA4', index=False)
    else:
        cellTypeDF.to_excel(writer, sheet_name=i, index=False)
# writer.save()
writer.close()
#%% perform t-test on each region/cell type same as above, but try plotting all regions together
plt.close('all')
sigGenesPerCells = {}
# prepare lists for BH fdr
tStatList = np.empty([len(sampleForDisplay['geneList']), len(cellsOfInterest)])
pValList = np.empty([len(sampleForDisplay['geneList']), len(cellsOfInterest)])
# loop over each of the regions/cell types and perform a t-test and plot results
for gene in range(len(sampleToCluster['geneList'])):
    fig = plt.figure(figsize=(16,6))
    nsdMean = plt.subplot2grid((4,9), (0,0), colspan=3, rowspan=4)
    sdMean = plt.subplot2grid((4,9), (0,3), colspan=3, rowspan=4)
    sdNSDTstat = plt.subplot2grid((4,9), (0,6), colspan=3, rowspan=4)
    nsdMean.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
    nsdMean.set_title('Mean of NSD')
    nsdMean.axis('off')
    sdMean.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
    sdMean.set_title('Mean of SD')
    sdMean.axis('off')
    sdNSDTstat.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
    sdNSDTstat.set_title('T-statistic for SD > NSD')
    sdNSDTstat.axis('off')
    sigAst = ""
    for regionN, region in enumerate(cellsOfInterest):
        regionCellsToPlot = findRelevantClusters(sampleForDisplay, region)
        regionIdx = np.where(allClustersMale == region)[0]
        sdNSDRegionIdx = sdNSDIdxMale[regionIdx]
        regionCells = allCellsMale[:, regionIdx]
        nsdCells = regionCells[:, sdNSDRegionIdx == 0]
        sdCells = regionCells[:, sdNSDRegionIdx == 1]
        sigGenesPerCells[region] = []
        tStat, pVal = scipy_stats.ttest_ind(np.squeeze(np.array(sdCells[gene,:])), np.squeeze(np.array(nsdCells[gene,:])))
        tStatList[gene, regionN] = tStat
        pValList[gene, regionN] = pVal
        if pVal < alphaFdr:
            sigAst = sigAst + "*"
        nsdColor = np.empty([len(regionCellsToPlot)])
        nsdColor[:] = np.mean(nsdCells[gene,:]) 
        sdColor = np.empty([len(regionCellsToPlot)])
        sdColor[:] = np.mean(sdCells[gene,:]) 
        tStatColor = np.empty([len(regionCellsToPlot)])
        tStatColor[:] = tStat
        nsdMean.scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=nsdColor, cmap='Reds', s=1, vmin=0, vmax=4)
        sdMean.scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=sdColor, cmap='Reds', s=1, vmin=0, vmax=4)
        sdNSDTstat.scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=tStatColor, cmap='seismic', s=1, vmin=-4, vmax=4)
    plt.suptitle(f'T-statistic for gene: {sampleToCluster["geneList"][gene]}{sigAst}')
    plt.show()
    """
    need to add colorbars, ideally set max of each mean section to the max value
    per gene, and the t-stat max value between -4 and 4
    
    set mean colorbar on bottom between NSD and SD, with t-stat below t-stat
    """
    plt.savefig(os.path.join(derivatives, 'clusterDEGsAllRegionsMales', f'ttest_nsd_sd_males_{sampleToCluster["geneList"][gene]}_in_allRegions.png'), bbox_inches='tight', dpi=300)
    plt.close()

#%% perform t-test on each region/cell type in female samples
# use the first sample as the one to plot to
sampleForDisplay = femaleSamples[0]
# desiredPval = 0.05
# alphaFdr = 1 - np.power((1 - desiredPval),(1/(len(femaleSamples[0]["geneList"]))))
plt.close('all')
sigGenesPerCells = {}
# loop over each of the regions/cell types and perform a t-test and plot results
for region in cellsOfInterest:
    regionCellsToPlot = findRelevantClusters(sampleForDisplay, region)
    regionIdx = np.where(allClustersFemale == region)[0]
    sdNSDRegionIdx = sdNSDIdxFemale[regionIdx]
    regionCells = allCellsFemale[:, regionIdx]
    nsdCells = regionCells[:, sdNSDRegionIdx == 0]
    sdCells = regionCells[:, sdNSDRegionIdx == 1]
    sigGenesPerCells[region] = []
    for gene in range(len(sampleForDisplay['geneList'])):
        tStat, pVal = scipy_stats.ttest_ind(np.squeeze(np.array(sdCells[gene,:])), np.squeeze(np.array(nsdCells[gene,:])))
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
            plt.suptitle(f'T-statistic for gene: {sampleForDisplay["geneList"][gene]} \n in {region}')
            plt.show()
            if region == "DG/CA4":
                plt.savefig(os.path.join(derivatives, 'clusterDEGsFemales', f'ttest_nsd_sd_females_{sampleForDisplay["geneList"][gene]}_in_DG-CA4.png'), bbox_inches='tight', dpi=300)
            else:
                plt.savefig(os.path.join(derivatives, 'clusterDEGsFemales', f'ttest_nsd_sd_females_{sampleForDisplay["geneList"][gene]}_in_{region}.png'), bbox_inches='tight', dpi=300)
            plt.close()
            sigGenesPerCells[region].append(sampleForDisplay["geneList"][gene])
            
#%% perform t-test on each region/cell type same as above, but try plotting all regions together
plt.close('all')
sigGenesPerCells = {}
# loop over each of the regions/cell types and perform a t-test and plot results

for gene in range(len(sampleForDisplay['geneList'])):
    fig = plt.figure(figsize=(16,6))
    nsdMean = plt.subplot2grid((4,9), (0,0), colspan=3, rowspan=4)
    sdMean = plt.subplot2grid((4,9), (0,3), colspan=3, rowspan=4)
    sdNSDTstat = plt.subplot2grid((4,9), (0,6), colspan=3, rowspan=4)
    nsdMean.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
    nsdMean.set_title('Mean of NSD')
    nsdMean.axis('off')
    sdMean.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
    sdMean.set_title('Mean of SD')
    sdMean.axis('off')
    sdNSDTstat.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
    sdNSDTstat.set_title('T-statistic for SD > NSD')
    sdNSDTstat.axis('off')
    sigAst = ""
    for region in cellsOfInterest:
        regionCellsToPlot = findRelevantClusters(sampleForDisplay, region)
        regionIdx = np.where(allClustersFemale == region)[0]
        sdNSDRegionIdx = sdNSDIdxFemale[regionIdx]
        regionCells = allCellsFemale[:, regionIdx]
        nsdCells = regionCells[:, sdNSDRegionIdx == 0]
        sdCells = regionCells[:, sdNSDRegionIdx == 1]
        sigGenesPerCells[region] = []
        tStat, pVal = scipy_stats.ttest_ind(np.squeeze(np.array(sdCells[gene,:])), np.squeeze(np.array(nsdCells[gene,:])))
        if pVal < alphaFdr:
            sigAst = sigAst + "*"
        nsdColor = np.empty([len(regionCellsToPlot)])
        nsdColor[:] = np.mean(nsdCells[gene,:]) 
        sdColor = np.empty([len(regionCellsToPlot)])
        sdColor[:] = np.mean(sdCells[gene,:]) 
        tStatColor = np.empty([len(regionCellsToPlot)])
        tStatColor[:] = tStat
        nsdMean.scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=nsdColor, cmap='Reds', s=1, vmin=0, vmax=4)
        sdMean.scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=sdColor, cmap='Reds', s=1, vmin=0, vmax=4)
        sdNSDTstat.scatter(sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,0], sampleForDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=tStatColor, cmap='seismic', s=1, vmin=-4, vmax=4)
    plt.suptitle(f'T-statistic for gene: {sampleToCluster["geneList"][gene]}{sigAst}')
    plt.show()
    """
    need to add colorbars, ideally set max of each mean section to the max value
    per gene, and the t-stat max value between -4 and 4
    
    set mean colorbar on bottom between NSD and SD, with t-stat below t-stat
    """
    plt.savefig(os.path.join(derivatives, 'clusterDEGsAllRegionsFemales', f'ttest_nsd_sd_females_{sampleToCluster["geneList"][gene]}_in_allRegions.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
#%% try to plot female and male results side by side (using code directly above)
plt.close('all')
sigGenesPerCells = {}
# loop over each of the regions/cell types and perform a t-test and plot results
femaleSampleToDisplay = femaleSamples[0]
maleSampleToDisplay = maleSamples[0]
for gene in range(len(sampleToCluster['geneList'])):
    fig = plt.figure(figsize=(12,6))
    femaleTstat = plt.subplot2grid((4,9), (0,0), colspan=3, rowspan=4)
    maleTstat = plt.subplot2grid((4,9), (0,3), colspan=3, rowspan=4)
    femaleTstat.imshow(femaleSampleToDisplay['tissueImageProcessed'], cmap='gray_r')
    femaleTstat.axis('off')
    femaleSigAst = ""
    maleTstat.imshow(maleSampleToDisplay['tissueImageProcessed'], cmap='gray_r')
    maleTstat.axis('off')
    maleSigAst = ""
    for region in cellsOfInterest:
        regionCellsToPlot = findRelevantClusters(femaleSampleToDisplay, region)
        regionIdx = np.where(allClustersFemale == region)[0]
        sdNSDRegionIdx = sdNSDIdxFemale[regionIdx]
        regionCells = allCellsFemale[:, regionIdx]
        nsdCells = regionCells[:, sdNSDRegionIdx == 0]
        sdCells = regionCells[:, sdNSDRegionIdx == 1]
        sigGenesPerCells[region] = []
        tStat, pVal = scipy_stats.ttest_ind(np.squeeze(np.array(sdCells[gene,:])), np.squeeze(np.array(nsdCells[gene,:])))
        if pVal < alphaFdr:
            femaleSigAst = femaleSigAst + "*"
        tStatColor = np.empty([len(regionCellsToPlot)])
        tStatColor[:] = tStat
        femaleTstat.scatter(femaleSampleToDisplay['processedTissuePositionList'][regionCellsToPlot,0], femaleSampleToDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=tStatColor, cmap='seismic', s=1, vmin=-4, vmax=4)
        femaleTstat.set_title(f'T-statistic for {femaleSampleToDisplay["geneList"][gene]}{femaleSigAst}\n females SD > NSD')
    for region in cellsOfInterest:
        regionCellsToPlot = findRelevantClusters(maleSampleToDisplay, region)
        regionIdx = np.where(allClustersMale == region)[0]
        sdNSDRegionIdx = sdNSDIdxMale[regionIdx]
        regionCells = allCellsMale[:, regionIdx]
        nsdCells = regionCells[:, sdNSDRegionIdx == 0]
        sdCells = regionCells[:, sdNSDRegionIdx == 1]
        sigGenesPerCells[region] = []
        tStat, pVal = scipy_stats.ttest_ind(np.squeeze(np.array(sdCells[gene,:])), np.squeeze(np.array(nsdCells[gene,:])))
        if pVal < alphaFdr:
            maleSigAst = maleSigAst + "*"
        tStatColor = np.empty([len(regionCellsToPlot)])
        tStatColor[:] = tStat
        maleTstat.scatter(maleSampleToDisplay['processedTissuePositionList'][regionCellsToPlot,0], maleSampleToDisplay['processedTissuePositionList'][regionCellsToPlot,1], c=tStatColor, cmap='seismic', s=1, vmin=-4, vmax=4)
        maleTstat.set_title(f'T-statistic for {maleSampleToDisplay["geneList"][gene]}{maleSigAst}\n males SD > NSD')
    plt.show()
    """
    need to add colorbars, ideally set max of each mean section to the max value
    per gene, and the t-stat max value between -4 and 4
    
    set mean colorbar on bottom between NSD and SD, with t-stat below t-stat
    """
    plt.savefig(os.path.join(derivatives, 'clusterDEGsAllRegionsFemalesAndMales', f'ttest_nsd_sd_females_and_males_{sampleToCluster["geneList"][gene]}_in_allRegions.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
#%% create a plot of umap for female and male all cells, along with clustering info for display image from each
plt.close('all')

fig = plt.figure(figsize=(16,9))
clustFemale = plt.subplot2grid((4,9), (0,0), colspan=2, rowspan=2)
clustMale = plt.subplot2grid((4,9), (2,0), colspan=2, rowspan=2)
umapAxsFemale = plt.subplot2grid((4,9), (0,3), colspan=3, rowspan=4)
umapAxMale = plt.subplot2grid((4,9), (0,6), colspan=3, rowspan=4)
# plot female clusters for single sample
clustFemale.imshow(femaleSampleToDisplay['tissueImageProcessed'], cmap='gray_r')
clustFemale.scatter(femaleSampleToDisplay['processedTissuePositionList'][:,0], femaleSampleToDisplay['processedTissuePositionList'][:,1], c=colorsAllCellsFemale[0:femaleSampleToDisplay['processedTissuePositionList'].shape[0],:], s=3)
clustFemale.set_title(f"Clustering of female sample {femaleSampleToDisplay['sampleID']}")
clustFemale.set_xticks([])
clustFemale.set_yticks([])
# plot male clusters for single sample
clustMale.imshow(maleSampleToDisplay['tissueImageProcessed'], cmap='gray_r')
clustMale.scatter(maleSampleToDisplay['processedTissuePositionList'][:,0], maleSampleToDisplay['processedTissuePositionList'][:,1], c=colorsAllCellsMale[0:maleSampleToDisplay['processedTissuePositionList'].shape[0],:], s=3)
clustMale.set_title(f"Clustering of male sample {maleSampleToDisplay['sampleID']}")
clustMale.set_xticks([])
clustMale.set_yticks([])

# plot the female samples
umapAxsFemale.scatter(embeddingFemale[:,0], embeddingFemale[:,1], c=colorsAllCellsFemale, s=3)
umapAxsFemale.set_title("UMAP of female samples")
umapAxsFemale.set_xticks([])
umapAxsFemale.set_yticks([])

# plot the male samples
umapAxMale.scatter(embeddingMale[:,0], embeddingMale[:,1], c=colorsAllCellsMale, s=3)
umapAxMale.set_title("UMAP of male samples")
umapAxMale.set_xticks([])
umapAxMale.set_yticks([])

# plot the legend
# create handles and patches to generate labeled legend
handles, labels = umapAxsFemale.get_legend_handles_labels()
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

umapAxsFemale.legend(handles=handles, ncols=2)

plt.show()
### uncomment below to save new figure
plt.savefig(os.path.join(derivatives, 'xenium_figure_k15cluster_umap_with_female_and_male.pdf'), bbox_inches='tight', dpi=300)
#%% Old code below

"""
code below is all old and was used to create figure, only keeping around for taking useful pieces
"""

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
        
#%% create a plot with clustering, UMAP, and t-test for Rbm3 and Gpr161
"""
create a figure with two sets of subfigures, 
one horizontal for the cluster + umap
one vertical for Rbm3 + Gpr161
"""
sampleForDisplay = maleSamples[0]
ca1Idx = findRelevantClusters(sampleForDisplay, 'CA1')
ca3Idx = findRelevantClusters(sampleForDisplay, 'CA3')
# desiredPval = 0.05
# alphaFdr = 1 - np.power((1 - desiredPval),(1/(len(maleSamples[0]["geneList"]))))
plt.close('all')
# sfigs = fig.subfigures(1, 2)
# clustAxs = sfigs[0].subplots(1, 2)
# ttestAxs = sfigs[1].subplots(2, 1)

#%%
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Arial'
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
umapAxs.scatter(embedding[:,0], embedding[:,1], c=colorsAllCells, s=3)
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
ttestAxs1.scatter(sampleForDisplay['processedTissuePositionList'][ca1Idx,0],sampleForDisplay['processedTissuePositionList'][ca1Idx,1], cmap='seismic', c=ca1Rbm3TstatColors, vmax=4, vmin=-4, s=3)
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
ttestAxs2.scatter(sampleForDisplay['processedTissuePositionList'][ca1Idx,0],sampleForDisplay['processedTissuePositionList'][ca1Idx,1], cmap='seismic', c=ca1Gpr161TstatColors, vmax=4, vmin=-4, s=3)
axScatter = ttestAxs2.scatter(sampleForDisplay['processedTissuePositionList'][ca3Idx,0],sampleForDisplay['processedTissuePositionList'][ca3Idx,1], cmap='seismic', c=ca3Gpr161TstatColors, vmax=4, vmin=-4, s=3)
ttestAxs2.axis('off')
#bar([f'NSD, mean={np.mean(conGroup)}', f'SD, mean={np.mean(expGroup)}'], [np.mean(conGroup), np.mean(expGroup)], yerr=[scipy_stats.sem(conGroup), scipy_stats.sem(expGroup)], capsize=3, width=w, label=f'p={pVal}', color=[colorHCHex,colorMDDHex])
ttestAxs2.set_title(f'Gpr161{geneSig}, t-statistic SD > NSD\n CA1{ca1Sig}, CA3{ca3Sig}')
plt.colorbar(axScatter,fraction=0.02, pad=0.04)
plt.show()

##### uncomment 3 lines below to recreate figures
# plt.savefig(os.path.join(derivatives, 'xenium_figure_k15cluster_umap_rbm3ttest_gpr161ttest.svg'), bbox_inches='tight', dpi=300)
# plt.savefig(os.path.join(derivatives, 'xenium_figure_k15cluster_umap_rbm3ttest_gpr161ttest.png'), bbox_inches='tight', dpi=300)
# plt.savefig(os.path.join(derivatives, 'xenium_figure_k15cluster_umap_rbm3ttest_gpr161ttest.pdf'), bbox_inches='tight', dpi=300)

#%% replace titles with A, B, C
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
clustAxs.set_title('Clustering, k=15')
clustAxs.annotate('A', xy=(0,1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top', fontfamily='serif',
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
umapAxs.scatter(embedding[:,0], embedding[:,1], c=colorsAllCells, s=3)
umapAxs.set_title("UMAP of all samples")
umapAxs.set_xticks([])
umapAxs.set_yticks([])
umapAxs.annotate('B', xy=(0,1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top', fontfamily='serif',
        fontweight='bold')
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
ttestAxs1.scatter(sampleForDisplay['processedTissuePositionList'][ca1Idx,0],sampleForDisplay['processedTissuePositionList'][ca1Idx,1], cmap='seismic', c=ca1Rbm3TstatColors, vmax=4, vmin=-4, s=3)
axScatter = ttestAxs1.scatter(sampleForDisplay['processedTissuePositionList'][ca3Idx,0],sampleForDisplay['processedTissuePositionList'][ca3Idx,1], cmap='seismic', c=ca3Rbm3TstatColors, vmax=4, vmin=-4, s=3)
ttestAxs1.axis('off')
#bar([f'NSD, mean={np.mean(conGroup)}', f'SD, mean={np.mean(expGroup)}'], [np.mean(conGroup), np.mean(expGroup)], yerr=[scipy_stats.sem(conGroup), scipy_stats.sem(expGroup)], capsize=3, width=w, label=f'p={pVal}', color=[colorHCHex,colorMDDHex])
ttestAxs1.set_title(f'Rbm3{geneSig}, t-statistic SD > NSD\n CA1{ca1Sig}, CA3{ca3Sig}')
ttestAxs1.annotate('C', xy=(0,1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
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
ttestAxs2.scatter(sampleForDisplay['processedTissuePositionList'][ca1Idx,0],sampleForDisplay['processedTissuePositionList'][ca1Idx,1], cmap='seismic', c=ca1Gpr161TstatColors, vmax=4, vmin=-4, s=3)
axScatter = ttestAxs2.scatter(sampleForDisplay['processedTissuePositionList'][ca3Idx,0],sampleForDisplay['processedTissuePositionList'][ca3Idx,1], cmap='seismic', c=ca3Gpr161TstatColors, vmax=4, vmin=-4, s=3)
ttestAxs2.axis('off')
#bar([f'NSD, mean={np.mean(conGroup)}', f'SD, mean={np.mean(expGroup)}'], [np.mean(conGroup), np.mean(expGroup)], yerr=[scipy_stats.sem(conGroup), scipy_stats.sem(expGroup)], capsize=3, width=w, label=f'p={pVal}', color=[colorHCHex,colorMDDHex])
ttestAxs2.set_title(f'Gpr161{geneSig}, t-statistic SD > NSD\n CA1{ca1Sig}, CA3{ca3Sig}')
ttestAxs2.annotate('D', xy=(0,1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top', fontfamily='serif',
        fontweight='bold')
plt.colorbar(axScatter,fraction=0.02, pad=0.04)
plt.show()

##### uncomment 3 lines below to recreate figures
# plt.savefig(os.path.join(derivatives, 'xenium_figure_k15cluster_umap_rbm3ttest_gpr161ttest.svg'), bbox_inches='tight', dpi=300)
# plt.savefig(os.path.join(derivatives, 'xenium_figure_k15cluster_umap_rbm3ttest_gpr161ttest.png'), bbox_inches='tight', dpi=300)
# plt.savefig(os.path.join(derivatives, 'xenium_figure_k15cluster_umap_rbm3ttest_gpr161ttest.pdf'), bbox_inches='tight', dpi=300)
