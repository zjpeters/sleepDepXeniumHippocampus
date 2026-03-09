#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 11:10:59 2025

@author: zjpeters
"""

import os
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.insert(0, "/home/zjpeters/Documents/stanly/code")
import stanly
import scipy
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
import scipy.sparse as sp_sparse
import time
import matplotlib.cm as cm
import seaborn as sns
import scipy.spatial as sp_spatial
import pandas as pd
import umap

rawdata=os.path.join('/','media','zjpeters','Expansion','sleepDepXenium','rawdata')
derivatives=os.path.join('/','media','zjpeters','Expansion','sleepDepXeniumHippocampus','derivatives')

def findRelevantClusters(processedSample, cluster_region):
    matchingClusters = np.where(processedSample['cluster_region'] == cluster_region)[0]
    clusterMask = []
    for cellIdx in range(len(processedSample['cluster_labels'])):
        # since it's possible that more than one cluster correspond to cluster of interest, use for loop
        for clusterIdx in matchingClusters:
            if processedSample['cluster_labels'][cellIdx] == clusterIdx:
                clusterMask.append(cellIdx)
    return clusterMask

#%% load processed samples

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

#%% load similarity matrix for ca1 cells

ca1Regions = findRelevantClusters(processedSamples[0], 'CA1')
ca1GeneMatrix = processedSamples[0]['geneMatrixLog2'][:,ca1Regions]
ca1Coors = processedSamples[0]['processedTissuePositionList'][ca1Regions, :]
ca1SimMatrix = stanly.measureTranscriptomicSimilarity(ca1GeneMatrix, os.path.join(derivatives, f"similarityMatrix_{processedSamples[0]['sampleID']}.npz"))
ca1SimMatrix = ca1SimMatrix[ca1Regions, :]
ca1SimMatrix = ca1SimMatrix[:, ca1Regions]

#%% perform clustering
actK = 8

nDigitalSpots = ca1SimMatrix.shape[0]
Wcontrol = ca1SimMatrix.todense()

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

clusters = KMeans(n_clusters=actK, init='random', n_init=500, tol=1e-8,)
cluster_labels = clusters.fit_predict(np.real(np.array(eigvecControlSort)[:,0:actK+1]))

#%% plot clustering 

plt.close('all')
plt.figure()
plt.imshow(processedSamples[0]['tissueImageProcessed'], cmap='gray_r')
plt.scatter(ca1Coors[:,0], ca1Coors[:,1], c=cluster_labels, cmap='tab20', s=2)
plt.show()

#%% create function that combines all of above code to cluster and plot for specific region
def subclusterDefinedRegion(processedSample, regionToCluster, k=8, marker_size=2):
    regionIdx = findRelevantClusters(processedSample, regionToCluster)
    regionGeneMatrix = processedSample['geneMatrixLog2'][:,regionIdx]
    regionCoors = processedSample['processedTissuePositionList'][regionIdx, :]
    regionSimMatrix = stanly.measureTranscriptomicSimilarity(regionGeneMatrix, os.path.join(derivatives, f"similarityMatrix_{processedSample['sampleID']}.npz"))
    regionSimMatrix = regionSimMatrix[regionIdx, :]
    regionSimMatrix = regionSimMatrix[:, regionIdx]
    
    Wcontrol = regionSimMatrix.todense()
    
    #% create laplacian for control
    Wcontrol = (Wcontrol - np.nanmin(Wcontrol))/(np.nanmax(Wcontrol) - np.nanmin(Wcontrol))
    Wcontrol[Wcontrol==1] = 0
    # Wcontrol[np.isnan(Wcontrol)] = 0
    Dcontrol = np.diag(sum(Wcontrol))
    Lcontrol = Dcontrol - Wcontrol
    eigvalControl,eigvecControl = scipy.sparse.linalg.eigs(Lcontrol, k=550)
    eigvalControlSortIdx = np.argsort(np.real(eigvalControl))[::-1]
    eigvecControlSort = np.real(eigvecControl[:,eigvalControlSortIdx])
    
    clusters = KMeans(n_clusters=k, init='random', n_init=500, tol=1e-8,)
    cluster_labels = clusters.fit_predict(np.real(np.array(eigvecControlSort)[:,0:k+1]))
    
    plt.figure()
    plt.imshow(processedSample['tissueImageProcessed'], cmap='gray_r')
    plt.scatter(regionCoors[:,0], regionCoors[:,1], c=cluster_labels, cmap='tab20', s=2)
    plt.show()
    label_dict = {'cluster_labels': cluster_labels, 'cluster_coordinates': regionCoors, 'cluster_idx': np.array(regionIdx)}
    return label_dict
    
labelInfo = subclusterDefinedRegion(processedSamples[2], 'CA3', marker_size=4)

#%% try using different k's to test clusterings

for k in range(4, 17):
    subclusterDefinedRegion(processedSamples[0], 'CA1', k=k, marker_size=6)
    
#%% create z-scored matrix for hippocampal sample
labelInfo = subclusterDefinedRegion(processedSamples[0], 'CA1', marker_size=4)

z_score_gene_matrix = (processedSamples[0]['geneMatrixLog2'].todense() - np.mean(processedSamples[0]['geneMatrixLog2'].todense(), axis=1))/np.std(processedSamples[0]['geneMatrixLog2'].todense(), axis=1)

#%% check for genes enriched in ca1 clusters

ca1ZScore = z_score_gene_matrix[:,labelInfo['cluster_idx']]
ca1MeanZScore = np.mean(ca1ZScore, axis=1)
for i in range(8):
    subClusterIdx = np.where(labelInfo['cluster_labels'] == i)[0]
    subClusterZScore = z_score_gene_matrix[:,labelInfo['cluster_idx'][subClusterIdx]]
    subClusterMeanZScore = np.squeeze(np.array(np.mean(subClusterZScore, axis=1)))
    subClusterMeanZScoreSorted = np.sort(subClusterMeanZScore)
    subClusterMeanZScoreSortedIdx = np.argsort(subClusterMeanZScore)
    print(f'Top genes for cluster {i}')
    for geneIdx in subClusterMeanZScoreSortedIdx[-10:-1]:
        print(processedSamples[0]['geneList'][geneIdx], subClusterMeanZScore[geneIdx])
        
#%% similar as above, but perform z-score on ca1 restricted matrix
z_score_gene_matrix = (ca1GeneMatrix.todense() - np.mean(ca1GeneMatrix.todense(), axis=1))/np.std(ca1GeneMatrix.todense(), axis=1)

#%% check for genes enriched in ca1 clusters

ca1ZScore = z_score_gene_matrix
ca1MeanZScore = np.mean(ca1ZScore, axis=1)
topGenesPerCluster = {}
for i in range(8):
    subClusterIdx = np.where(labelInfo['cluster_labels'] == i)[0]
    subClusterZScore = z_score_gene_matrix[:,subClusterIdx]
    subClusterMeanZScore = np.squeeze(np.array(np.mean(subClusterZScore, axis=1)))
    subClusterMeanZScoreSorted = np.sort(subClusterMeanZScore)
    subClusterMeanZScoreSortedIdx = np.argsort(subClusterMeanZScore)
    print(f'Top genes for cluster {i}')
    topGenesPerCluster[f'cluster{i}'] = []
    for geneIdx in subClusterMeanZScoreSortedIdx[-10:-1]:
        topGenesPerCluster[f'cluster{i}'].append(processedSamples[0]['geneList'][geneIdx])
        print(processedSamples[0]['geneList'][geneIdx], subClusterMeanZScore[geneIdx])

#%% display gene expression of top genes per cluster

for gene in topGenesPerCluster['cluster0']:
    geneIdx = processedSamples[0]['geneList'].index(gene)
    geneArray = np.squeeze(np.array(processedSamples[0]['geneMatrixLog2'][geneIdx,:].todense()))
    plt.figure()
    plt.imshow(processedSamples[0]['tissueImageProcessed'], cmap='gray_r')
    plt.scatter(processedSamples[0]['processedTissuePositionList'][:,0], processedSamples[0]['processedTissuePositionList'][:,1], c=geneArray, cmap='Reds', vmin=0, vmax=4, s=2)
    plt.show()

#%% display gene expression of top genes per cluster only within CA1 using z-score
plt.close('all')
z_score_gene_matrix = (ca1GeneMatrix.todense() - np.mean(ca1GeneMatrix.todense(), axis=1))/np.std(ca1GeneMatrix.todense(), axis=1)
for gene in topGenesPerCluster['cluster0']:
    geneIdx = processedSamples[0]['geneList'].index(gene)
    geneArray = np.squeeze(np.array(z_score_gene_matrix[geneIdx,:]))
    plt.figure()
    plt.imshow(processedSamples[0]['tissueImageProcessed'], cmap='gray_r')
    plt.scatter(labelInfo['cluster_coordinates'][:,0], labelInfo['cluster_coordinates'][:,1], c=geneArray, cmap='Reds', vmin=0, vmax=4, s=2)
    plt.title(f'Expression of {gene}')
    plt.show()
    
#%% display gene expression of all top genes in cluster as one figure
plt.close('all')
cluster_k = 0
for cluster_k in range(8):
    color = cm.tab20b(float(cluster_k) / 8)
    cluster_idx = np.where(labelInfo['cluster_labels'] == cluster_k)[0]
    fig, ax = plt.subplots(2,5)
    nR = 0
    nC = 1
    ax[0,0].imshow(processedSamples[0]['tissueImageProcessed'], cmap='gray_r')
    ax[0,0].scatter(labelInfo['cluster_coordinates'][cluster_idx,0], labelInfo['cluster_coordinates'][cluster_idx,1], c=color, s=2)
    ax[0,0].axis('off')
    for gene in topGenesPerCluster[f'cluster{cluster_k}']:
        geneIdx = processedSamples[0]['geneList'].index(gene)
        geneArray = np.squeeze(np.array(z_score_gene_matrix[geneIdx,:]))
        ax[nR,nC].imshow(processedSamples[0]['tissueImageProcessed'], cmap='gray_r')
        ax[nR,nC].scatter(labelInfo['cluster_coordinates'][:,0], labelInfo['cluster_coordinates'][:,1], c=geneArray, cmap='seismic', vmin=-2, vmax=2, s=2)
        ax[nR,nC].set_title(f'Expression of {gene}')
        ax[nR,nC].axis('off')
        if nC < 4:
            nC += 1
        else:
            nR += 1
            nC = 0
    plt.suptitle(f'Top genes by z-score for cluster {cluster_k}')
    plt.show()
    
#%% check for genes enriched in ca1 clusters, this time accounting for decreased z-score

ca1ZScore = z_score_gene_matrix
ca1MeanZScore = np.mean(ca1ZScore, axis=1)
topGenesPerCluster = {}
for i in range(8):
    subClusterIdx = np.where(labelInfo['cluster_labels'] == i)[0]
    subClusterZScore = z_score_gene_matrix[:,subClusterIdx]
    subClusterMeanZScore = np.squeeze(np.array(np.mean(subClusterZScore, axis=1)))
    subClusterMeanZScoreSorted = np.sort(np.abs(subClusterMeanZScore))
    subClusterMeanZScoreSortedIdx = np.argsort(np.abs(subClusterMeanZScore))
    print(f'Top genes for cluster {i}')
    topGenesPerCluster[f'cluster{i}'] = []
    for geneIdx in subClusterMeanZScoreSortedIdx[-10:-1]:
        topGenesPerCluster[f'cluster{i}'].append(processedSamples[0]['geneList'][geneIdx])
        print(processedSamples[0]['geneList'][geneIdx], subClusterMeanZScore[geneIdx])

#%% display gene expression of all top genes in cluster as one figure

plt.close('all')
plt.figure()
plt.imshow(processedSamples[0]['tissueImageProcessed'], cmap='gray_r')
for cluster_k in range(8):
    color = cm.tab20b(float(cluster_k) / 8)
    cluster_idx = np.where(labelInfo['cluster_labels'] == cluster_k)[0]
    plt.scatter(labelInfo['cluster_coordinates'][cluster_idx,0], labelInfo['cluster_coordinates'][cluster_idx,1], c=color, s=2)
plt.axis('off')
plt.show()
    
for cluster_k in range(8):
    color = cm.tab20b(float(cluster_k) / 8)
    cluster_idx = np.where(labelInfo['cluster_labels'] == cluster_k)[0]
    fig, ax = plt.subplots(2,5)
    nR = 0
    nC = 1
    ax[0,0].imshow(processedSamples[0]['tissueImageProcessed'], cmap='gray_r')
    ax[0,0].scatter(labelInfo['cluster_coordinates'][cluster_idx,0], labelInfo['cluster_coordinates'][cluster_idx,1], c=color, s=2)
    ax[0,0].axis('off')
    for gene in topGenesPerCluster[f'cluster{cluster_k}']:
        geneIdx = processedSamples[0]['geneList'].index(gene)
        geneArray = np.squeeze(np.array(z_score_gene_matrix[geneIdx,:]))
        ax[nR,nC].imshow(processedSamples[0]['tissueImageProcessed'], cmap='gray_r')
        ax[nR,nC].scatter(labelInfo['cluster_coordinates'][:,0], labelInfo['cluster_coordinates'][:,1], c=geneArray, cmap='seismic', vmin=-2, vmax=2, s=2)
        ax[nR,nC].set_title(f'Expression of {gene}')
        ax[nR,nC].axis('off')
        if nC < 4:
            nC += 1
        else:
            nR += 1
            nC = 0
    plt.suptitle(f'Top genes by z-score for cluster {cluster_k}')
    plt.show()
#%% think about how to better distinguish between CA1 regions
# could look for genes with high variability

ca1Variance = np.squeeze(np.array(np.var(ca1GeneMatrix.todense(), axis=1)))
ca1SortVar = np.sort(ca1Variance)
ca1SortVarIdx = np.argsort(ca1Variance)

topN = 40
ca1VarIdxTopN = ca1SortVarIdx[-topN-1:-1]
short_gene_matrix = ca1GeneMatrix[ca1VarIdxTopN, :]

#%% cluster using condensed gene matrix
k = 8
regionSimMatrix = stanly.measureTranscriptomicSimilarity(short_gene_matrix, os.path.join(derivatives, "similarityMatrix_test_top_variance.npz"))

Wcontrol = regionSimMatrix.todense()

#% create laplacian for control
Wcontrol = (Wcontrol - np.nanmin(Wcontrol))/(np.nanmax(Wcontrol) - np.nanmin(Wcontrol))
Wcontrol[Wcontrol==1] = 0
# Wcontrol[np.isnan(Wcontrol)] = 0
Dcontrol = np.diag(sum(Wcontrol))
Lcontrol = Dcontrol - Wcontrol
eigvalControl,eigvecControl = scipy.sparse.linalg.eigs(Lcontrol, k=550)
eigvalControlSortIdx = np.argsort(np.real(eigvalControl))[::-1]
eigvecControlSort = np.real(eigvecControl[:,eigvalControlSortIdx])

clusters = KMeans(n_clusters=k, init='random', n_init=500, tol=1e-8,)
cluster_labels = clusters.fit_predict(np.real(np.array(eigvecControlSort)[:,0:k+1]))

#%% display clustering
plt.close('all')
plt.figure()
plt.imshow(processedSamples[0]['tissueImageProcessed'], cmap='gray_r')
plt.scatter(ca1Coors[:,0], ca1Coors[:,1], c=cluster_labels, cmap='tab20', s=2)
plt.show()

#%% try reducing gene matrix but using all cells
designMatrix = [0,1,0,1,0,1]
allCells = np.empty([len(processedSamples[0]['geneList']),0])
allClusters = np.empty(0, dtype='str')
sdNSDIdx = np.empty(0, dtype='int32')
sampleList = np.empty(0,dtype='int32')
allCoordinates = np.empty([0,2])
for actSample in range(6):
    allCells = np.append(allCells, processedSamples[actSample]['geneMatrixLog2'].todense(), axis=1)
    clusterList = [] #np.empty([len(maleSamples[actSample]['cluster_labels']),1], dtype='str')
    for cluster in processedSamples[actSample]['cluster_labels'].to_numpy():
        clusterList.append(processedSamples[actSample]['cluster_region'][cluster])
    allClusters = np.append(allClusters, clusterList)
    # binary distinction of each cell for sd or nsd
    if designMatrix[actSample] == 0:
        sdNSDIdx = np.append(sdNSDIdx, np.zeros(len(processedSamples[actSample]['processedTissuePositionList']), dtype='int32').T)
    elif designMatrix[actSample] == 1:
        sdNSDIdx = np.append(sdNSDIdx, np.ones(len(processedSamples[actSample]['processedTissuePositionList']), dtype='int32').T)
    sampleList = np.append(sampleList, np.zeros(len(processedSamples[actSample]['processedTissuePositionList']), dtype='int32').T + actSample)
    allCoordinates = np.append(allCoordinates, processedSamples[actSample]['processedTissuePositionList'], axis=0)
    sampleColors = processedSamples[actSample]['cluster_colors']
    
#%% calculate variance across all cells

ca1Variance = np.squeeze(np.array(np.var(allCells, axis=1)))
ca1SortVar = np.sort(ca1Variance)
ca1SortVarIdx = np.argsort(ca1Variance)

topN = 40
ca1VarIdxTopN = ca1SortVarIdx[-topN-1:-1]
short_gene_matrix = allCells[ca1VarIdxTopN, :]

#%% perform umap on reduced gene matrix

reducer = umap.UMAP()

embedding = reducer.fit_transform(np.array(allCells).T)

#%% plot umap
plt.close('all')
plt.figure()
plt.scatter(embedding[:,0], embedding[:,1], s=3)
plt.show()

#%% try clustering umap data

labels = HDBSCAN(min_samples=10, min_cluster_size=500).fit_predict(embedding)

#%% try plotting this on all samples
# create a 2 row by 3 column figure
fig, ax = plt.subplots(2,3)
nR = 0
nC = 0
for i in range(6):
    cellIdx = np.where(sampleList == i)[0]
    sampleCoors = allCoordinates[cellIdx,:]
    sampleLabels = labels[cellIdx]
    ax[nR,nC].imshow(processedSamples[i]['tissueImageProcessed'], cmap='gray_r')
    ax[nR,nC].scatter(sampleCoors[:,0], sampleCoors[:,1], c=sampleLabels, cmap='tab20', s=3)
    if nC < 2:
        nC += 1
    else:
        nR += 1
        nC = 0
        
#%% perform same on CA1 data

ca1Idx = np.where(allClusters == 'CA1')[0]
ca1Cells = allCells[:,ca1Idx]
ca1Coors = allCoordinates[ca1Idx,:]
ca1SampleList = sampleList[ca1Idx]

#%% calculate variance across all cells

ca1Variance = np.squeeze(np.array(np.var(ca1Cells, axis=1)))
ca1SortVar = np.sort(ca1Variance)
ca1SortVarIdx = np.argsort(ca1Variance)

topN = 297
ca1VarIdxTopN = ca1SortVarIdx[-topN-1:-1]
short_gene_matrix = ca1Cells[ca1VarIdxTopN, :]

#%% perform umap on reduced gene matrix

reducer = umap.UMAP()

embedding = reducer.fit_transform(np.array(ca1Cells).T)

#%% plot umap
plt.close('all')
plt.figure()
plt.scatter(embedding[:,0], embedding[:,1], s=3)
plt.show()

#%% try clustering umap data

labels = HDBSCAN(min_samples=10, min_cluster_size=50).fit_predict(embedding)
#%% plot umap
plt.close('all')
plt.figure()
plt.scatter(embedding[:,0], embedding[:,1], c=labels, cmap='tab20', s=3)
plt.show()
#%% try plotting this on all samples
# create a 2 row by 3 column figure
fig, ax = plt.subplots(2,3)
nR = 0
nC = 0
for i in range(6):
    cellIdx = np.where(ca1SampleList == i)[0]
    sampleCoors = ca1Coors[cellIdx,:]
    sampleLabels = labels[cellIdx]
    ax[nR,nC].imshow(processedSamples[i]['tissueImageProcessed'], cmap='gray_r')
    ax[nR,nC].scatter(sampleCoors[:,0], sampleCoors[:,1], c=sampleLabels, cmap='tab20', s=3)
    if nC < 2:
        nC += 1
    else:
        nR += 1
        nC = 0
#%% try spectral clustering on ca1 cells
k = 8
regionSimMatrix = stanly.measureTranscriptomicSimilarity(ca1Cells, os.path.join(derivatives, "similarityMatrix_CA1_cells.npz"), denseMatrix=True)

Wcontrol = regionSimMatrix.todense()

#% create laplacian for control
Wcontrol = (Wcontrol - np.nanmin(Wcontrol))/(np.nanmax(Wcontrol) - np.nanmin(Wcontrol))
Wcontrol[Wcontrol==1] = 0
# Wcontrol[np.isnan(Wcontrol)] = 0
Dcontrol = np.diag(sum(Wcontrol))
Lcontrol = Dcontrol - Wcontrol
eigvalControl,eigvecControl = scipy.sparse.linalg.eigs(Lcontrol, k=550)
eigvalControlSortIdx = np.argsort(np.real(eigvalControl))[::-1]
eigvecControlSort = np.real(eigvecControl[:,eigvalControlSortIdx])

clusters = KMeans(n_clusters=k, init='random', n_init=500, tol=1e-8,)
cluster_labels = clusters.fit_predict(np.real(np.array(eigvecControlSort)[:,0:k+1]))

#%% try plotting this on all samples
# create a 2 row by 3 column figure
fig, ax = plt.subplots(2,3, figsize=(16,8))
nR = 0
nC = 0
for i in range(6):
    cellIdx = np.where(ca1SampleList == i)[0]
    sampleCoors = ca1Coors[cellIdx,:]
    sampleLabels = cluster_labels[cellIdx]
    ax[nR,nC].imshow(processedSamples[i]['tissueImageProcessed'], cmap='gray_r')
    ax[nR,nC].scatter(sampleCoors[:,0], sampleCoors[:,1], c=sampleLabels, cmap='tab20', s=2)
    ax[nR,nC].axis('off')
    if nC < 2:
        nC += 1
    else:
        nR += 1
        nC = 0
plt.show()
plt.savefig(os.path.join(derivatives, 'subclustering_of_CA1.pdf'), bbox_inches='tight', dpi=300)