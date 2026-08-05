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
import scipy
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import scipy.stats as scipy_stats
import umap
import matplotlib.patches as mpatches
if os.path.exists(os.path.join('/', 'home', 'zjpeters', 'Documents', 'stanly', 'code')):
    stanlyLoc = os.path.join('/', 'home', 'zjpeters', 'Documents', 'stanly', 'code')
else:
    # windows workstation locations
    stanlyLoc = os.path.join('C:',os.sep, 'Users','onyh19ug', 'Documents', 'STANLY','code')
from sklearn.metrics import silhouette_samples, silhouette_score
import upsetplot
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
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6
plt.rcParams['text.usetex'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 6

# rawdata = os.path.join('/','home','zjpeters','Documents','sleepDepXenium','rawdata')
derivatives = os.path.join('/','home','zjpeters','Documents','sleepDepXeniumHippocampus','derivatives')
locOfTsvFile = os.path.join('/','home','zjpeters','Documents','sleepDepXeniumHippocampus', 'participants.tsv')
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
cellTypeColorListDict = {'astrocytes': astColor, 'endothelial': endColor, 'microglia': micColor, 'oligodendrocytes': oliColor, 'neurons': neuColor, 'sparse': sparseColor}
regionalColorListDict = {'CA1': ca1Color, 'CA2': ca2Color, 'CA3': ca3Color, 'DG': dgColor, 'DG/CA4': ca4dgColor}
#%% load newly processed samples and check selection
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
    


#%% generate male samples
maleSamples = []
maleSamplesIdx = [0,1,2,3,4,5]
for i in maleSamplesIdx:
    maleSamples.append(processedSamples[i])

designMatrix = [0,1,0,1,0,1]
#%% add cell types to the umap plotting
colorsAllCellsMale = np.empty([0,4])
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
    colorsAllCellsMale = np.append(colorsAllCellsMale, sampleColors, axis=0)
    
#%% create lists of all cells and all clusters for use in later plotting
plt.close('all')

allCellsMale = np.empty([len(maleSamples[0]['geneList']),0])
allClustersMale = np.empty(0, dtype='str')
sdNSDIdxMale = np.empty(0, dtype='int32')
sampleListMale = np.empty(0,dtype='int32')
allCoordinatesMale = np.empty([0,2])
sampleColorsMale = np.empty([0,4])
allBarcodesMale = np.empty(0, dtype='str')
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
    sampleColorsMale = np.append(sampleColorsMale, maleSamples[actSample]['cluster_colors'], axis=0)
    allBarcodesMale = np.append(allBarcodesMale, np.array(maleSamples[actSample]['barcodeList']))
clusterListMale = np.array(clusterListMale, dtype='str')

#%% create dataset containing only dentate gyrus 

dgMask = np.where(allClustersMale == 'DG')[0]
dgCells = allCellsMale[:,dgMask]
dgSdNSDIdx = sdNSDIdxMale[dgMask]
dgCoordinates = allCoordinatesMale[dgMask, :]
dgSampleList = sampleListMale[dgMask]
dgColorList = sampleColorsMale[dgMask, :]

#%% perform umap of DG cells
reducer = umap.UMAP(random_state=42)

embedding = reducer.fit_transform(np.array(dgCells).T)

#%% spectral clustering of all data, first calculate similarity matrix

similarityMatrix = stanly.measureTranscriptomicSimilarity(dgCells, savePath=os.path.join(derivatives, 'dgAnalysis', 'all_sample_dg_sim_matrix.npz'), axis=1, denseMatrix=True)
#%% perform clustering

Wcontrol = similarityMatrix.todense()

#% create laplacian for control
Wcontrol = (Wcontrol - np.nanmin(Wcontrol))/(np.nanmax(Wcontrol) - np.nanmin(Wcontrol))
Wcontrol[Wcontrol==1] = 0
# Wcontrol[np.isnan(Wcontrol)] = 0
Dcontrol = np.diag(sum(Wcontrol))
Lcontrol = Dcontrol - Wcontrol
eigvalControl,eigvecControl = scipy.sparse.linalg.eigs(Lcontrol, k=500)
eigvalControlSort = np.sort(np.real(eigvalControl))[::-1]
eigvalControlSortIdx = np.argsort(np.real(eigvalControl))[::-1]
eigvecControlSort = np.real(eigvecControl[:,eigvalControlSortIdx])

k = 5
clusters = KMeans(n_clusters=k, init='random', n_init=500, tol=1e-8,)
cluster_labels = clusters.fit_predict(np.real(np.array(eigvecControlSort)[:,0:k+1]))

#%% plot spectral clusters on UMAP

colors = cm.tab20b(cluster_labels.astype(float) / k)
plt.close('all')
plt.figure()
plt.scatter(embedding[:,0], embedding[:,1], c=colors, s=5)
plt.show()

sampleBasedColors = cm.tab20b(dgSampleList.astype(float) / k)
plt.figure()
plt.scatter(embedding[:,0], embedding[:,1], c=sampleBasedColors, s=5)
plt.show()

#%% extract genes most associated with clusters
clusterHeatmap = []
for actCluster in range(k):
    clusterIdx = np.where(cluster_labels == actCluster)[0]
    clusterCells = dgCells[:,clusterIdx]
    clusterHeatmap.append(np.var(clusterCells, axis=1))
clusterHeatmap = np.squeeze(np.array(clusterHeatmap)).T

maxVarianceIdx = np.argmax(clusterHeatmap, axis=0)

#%% display genes for each of the max variance per cluster

for i in maxVarianceIdx:
    stanly.viewGeneInProcessedSample(maleSamples[0], maleSamples[0]['geneList'][i])

plt.figure()
for i in maxVarianceIdx:
    plt.figure()
    plt.scatter(embedding[:,0], embedding[:,1], c=np.squeeze(np.array(dgCells[i])), cmap='Reds', s=5)
    plt.show()