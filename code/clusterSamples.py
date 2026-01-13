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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import scipy.sparse as sp_sparse
import time
import matplotlib.cm as cm
import seaborn as sns
import scipy.spatial as sp_spatial
import pandas as pd

rawdata=os.path.join('/','home','zjpeters','Documents','sleepDepXenium','rawdata')
derivatives=os.path.join('/','home','zjpeters','Documents','sleepDepXeniumHippocampus','derivatives')

#%% load newly processed samples and check selection

participant_list = ['YW-1_ROI_A1_hippocampus', 'YW-1_ROI_C1_hippocampus', 
                    'YW-2_ROI_B1_hippocampus', 'YW-1_ROI_A2_hippocampus', 
                    'YW-1_ROI_C2_hippocampus', 'YW-2_ROI_B2_hippocampus',
                    'YW-1_ROI_B1_hippocampus', 'YW-2_ROI_A1_hippocampus', 
                    'YW-2_ROI_C1_hippocampus', 'YW-1_ROI_B2_hippocampus', 
                    'YW-2_ROI_A2_hippocampus', 'YW-2_ROI_C2_hippocampus']

processedSamples = {}
for sampleIdx in range(len(participant_list)):
    processedSamples[sampleIdx] = stanly.loadProcessedXeniumSample(os.path.join(derivatives, f"{participant_list[sampleIdx]}"))

#%% run clustering on all samples at a given resolution

actK = 15
for actSample in range(0, 12):
    sampleToCluster = processedSamples[actSample]
    nDigitalSpots = sampleToCluster['geneMatrixLog2'].shape[1]
    
    start_time = time.time()
    simMatrixSavePath = os.path.join(derivatives, f'similarityMatrix_{sampleToCluster["sampleID"]}.npz')
    # check whether similarity matrix has already been calculated
    if os.path.exists(simMatrixSavePath):
        similarityMatrix = sp_sparse.load_npz(simMatrixSavePath)
    else:
        similarityMatrix = stanly.measureTranscriptomicSimilarity(sampleToCluster['geneMatrixLog2'], axis=1)
        sp_sparse.save_npz(simMatrixSavePath, similarityMatrix)
    print("--- %s seconds ---" % (time.time() - start_time))

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
    
    # run sillhoutte analysis on control clustering
    plt.close('all')
    clusterRange = np.array(range(10,35))
    # for actK in clusterRange:
        
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    
    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, nDigitalSpots + (actK + 1) * 10])
    
    clusters = KMeans(n_clusters=actK, init='random', n_init=500, tol=1e-8,)
    cluster_labels = clusters.fit_predict(np.real(np.array(eigvecControlSort)[:,0:actK+1]))
    
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(np.real(np.array(eigvecControlSort)[:,0:actK]), cluster_labels)
    print(
        "For n_clusters =",
        actK,
        "The average silhouette_score is :",
        silhouette_avg,
    )
    
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(np.real(np.array(eigvecControlSort)[:,0:actK]), cluster_labels)
    
    y_lower = 10
    for i in range(actK):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    
        ith_cluster_silhouette_values.sort()
    
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
    
        color = cm.tab20b(float(i) / actK)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
    
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    
    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    # 2nd Plot showing the actual clusters formed
    colors = cm.tab20b(cluster_labels.astype(float) / actK)
    ax2.imshow(sampleToCluster['tissueImageProcessed'],cmap='gray_r')
    ax2.scatter(sampleToCluster['processedTissuePositionList'][:,0], sampleToCluster['processedTissuePositionList'][:,1],c=colors, s=5)
    ax2.set_title("The visualization of the clustered control data.")
    ax2.axis('off')
    
    plt.suptitle(
        "Silhouette analysis for KMeans clustering on control sample data with n_clusters = %d"
        % actK,
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(os.path.join(derivatives, f'clusteringAndSilhouetteSleepDepControlK{actK}_hippocampus_{sampleToCluster["sampleID"]}.png'), bbox_inches='tight', dpi=300)
    plt.show()
    clusterDF = {'barcodes': sampleToCluster['barcodeList'], 'cluster_labels': cluster_labels, 'silhouette_values': sample_silhouette_values, 'color_r': colors[:,0], 'color_g': colors[:,1], 'color_b': colors[:,2], 'color_alpha': colors[:,3]}
    clusterDF = pd.DataFrame(clusterDF)
    clusterDF.to_csv(os.path.join(derivatives, f'{sampleToCluster["sampleID"]}_cluster_information.csv'), index=False)

#%% write clustering function

def performSpectralClustering(sampleToCluster, k=15, n_eigenvectors=250, kNN_for_similarity='FullyConnected'):
    """
    Perform a spectral clustering of a processed sample using transcriptomic 
    gene matrix.

    Parameters
    ----------
    sampleToCluster : dict
        processedSample from one of the processXData.
    k : int, optional
        The number of clusters to calculate. The default is 15.
    n_eigenvectors : int, optional
        Number of eigenvectors to calculate for clustering. The default is 250.
    kNN_for_similarity : 
        How many nearest neighbors to use in calculation. The default is 
        'FullyConnected'
    Returns
    -------
    clusterDF : TYPE
        DESCRIPTION.

    """
    nDigitalSpots = sampleToCluster['geneMatrixLog2'].shape[1]
    
    start_time = time.time()
    # location for saving similarity matrix
    simMatrixSavePath = os.path.join(sampleToCluster['derivativesPath'], f'similarityMatrix_{sampleToCluster["sampleID"]}.npz')
    # check whether similarity matrix has already been calculated
    if os.path.exists(simMatrixSavePath):
        similarityMatrix = sp_sparse.load_npz(simMatrixSavePath)
    else:
        similarityMatrix = stanly.measureTranscriptomicSimilarity(sampleToCluster['geneMatrixLog2'], axis=1)
        sp_sparse.save_npz(simMatrixSavePath, similarityMatrix)
    print("--- %s seconds ---" % (time.time() - start_time))

    Wcontrol = similarityMatrix.todense()
    
    #% create laplacian for control
    Wcontrol = (Wcontrol - np.nanmin(Wcontrol))/(np.nanmax(Wcontrol) - np.nanmin(Wcontrol))
    Wcontrol[Wcontrol==1] = 0
    # Wcontrol[np.isnan(Wcontrol)] = 0
    Dcontrol = np.diag(sum(Wcontrol))
    Lcontrol = Dcontrol - Wcontrol
    eigvalControl,eigvecControl = scipy.sparse.linalg.eigs(Lcontrol, k=n_eigenvectors)
    eigvalControlSort = np.sort(np.real(eigvalControl))[::-1]
    eigvalControlSortIdx = np.argsort(np.real(eigvalControl))[::-1]
    eigvecControlSort = np.real(eigvecControl[:,eigvalControlSortIdx])
    
    # run sillhoutte analysis on control clustering
        
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    
    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, nDigitalSpots + (k + 1) * 10])
    
    clusters = KMeans(n_clusters=k, init='random', n_init=500, tol=1e-8,)
    cluster_labels = clusters.fit_predict(np.real(np.array(eigvecControlSort)[:,0:k+1]))
    
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(np.real(np.array(eigvecControlSort)[:,0:k]), cluster_labels)
    print(
        "For n_clusters =",
        k,
        "The average silhouette_score is :",
        silhouette_avg,
    )
    
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(np.real(np.array(eigvecControlSort)[:,0:k]), cluster_labels)
    
    y_lower = 10
    for i in range(k):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    
        ith_cluster_silhouette_values.sort()
    
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
    
        color = cm.tab20b(float(i) / k)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
    
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    
    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    # 2nd Plot showing the actual clusters formed
    colors = cm.tab20b(cluster_labels.astype(float) / k)
    ax2.imshow(sampleToCluster['tissueImageProcessed'],cmap='gray_r')
    ax2.scatter(sampleToCluster['processedTissuePositionList'][:,0], sampleToCluster['processedTissuePositionList'][:,1],c=colors, s=5)
    ax2.set_title("The visualization of the clustered control data.")
    ax2.axis('off')
    
    plt.suptitle(
        "Silhouette analysis for KMeans clustering on control sample data with n_clusters = %d"
        % k,
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(os.path.join(sampleToCluster['derivativesPath'], f'clusteringAndSilhouette_K{k}_{sampleToCluster["sampleID"]}.png'), bbox_inches='tight', dpi=300)
    plt.show()
    clusterDF = {'barcodes': sampleToCluster['barcodeList'], 'cluster_labels': cluster_labels, 'silhouette_values': sample_silhouette_values, 'color_r': colors[:,0], 'color_g': colors[:,1], 'color_b': colors[:,2], 'color_alpha': colors[:,3]}
    clusterDF = pd.DataFrame(clusterDF)
    clusterDF.to_csv(os.path.join(sampleToCluster['derivativesPath'], f'clusteringAndSilhouette_K{k}_{sampleToCluster["sampleID"]}_cluster_information.csv'), index=False)
    return clusterDF
#%% test function above
clusterInfo = performSpectralClustering(processedSamples[0])
#%% extract top genes from clustering
actK = 15
sampleToCluster = processedSamples[0]
nDigitalSpots = sampleToCluster['geneMatrixLog2'].shape[1]

start_time = time.time()
simMatrixSavePath = os.path.join(derivatives, f'similarityMatrix_{sampleToCluster["sampleID"]}.npz')
# check whether similarity matrix has already been calculated
if os.path.exists(simMatrixSavePath):
    similarityMatrix = sp_sparse.load_npz(simMatrixSavePath)
else:
    similarityMatrix = stanly.measureTranscriptomicSimilarity(sampleToCluster['geneMatrixLog2'], axis=1)
    sp_sparse.save_npz(simMatrixSavePath, similarityMatrix)
print("--- %s seconds ---" % (time.time() - start_time))

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

# run sillhoutte analysis on control clustering
plt.close('all')
clusterRange = np.array(range(10,35))
# for actK in clusterRange:
    
# Create a subplot with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)

# The 1st subplot is the silhouette plot
# The silhouette coefficient can range from -1, 1 but in this example all
# lie within [-0.1, 1]
ax1.set_xlim([-1, 1])
# The (n_clusters+1)*10 is for inserting blank space between silhouette
# plots of individual clusters, to demarcate them clearly.
ax1.set_ylim([0, nDigitalSpots + (actK + 1) * 10])

clusters = KMeans(n_clusters=actK, init='random', n_init=500, tol=1e-8,)
cluster_labels = clusters.fit_predict(np.real(np.array(eigvecControlSort)[:,0:actK+1]))

# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed
# clusters
silhouette_avg = silhouette_score(np.real(np.array(eigvecControlSort)[:,0:actK]), cluster_labels)
print(
    "For n_clusters =",
    actK,
    "The average silhouette_score is :",
    silhouette_avg,
)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(np.real(np.array(eigvecControlSort)[:,0:actK]), cluster_labels)

y_lower = 10
for i in range(actK):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.tab20b(float(i) / actK)
    ax1.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,
    )

    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

# 2nd Plot showing the actual clusters formed
colors = cm.tab20b(cluster_labels.astype(float) / actK)
ax2.imshow(sampleToCluster['tissueImageProcessed'],cmap='gray_r')
ax2.scatter(sampleToCluster['processedTissuePositionList'][:,0], sampleToCluster['processedTissuePositionList'][:,1],c=colors, s=5)
ax2.set_title("The visualization of the clustered control data.")
ax2.axis('off')

plt.suptitle(
    "Silhouette analysis for KMeans clustering on control sample data with n_clusters = %d"
    % actK,
    fontsize=14,
    fontweight="bold",
)
plt.show()

clusterDF = {'barcodes': sampleToCluster['barcodeList'], 'cluster_labels': cluster_labels, 'silhouette_values': sample_silhouette_values, 'color_r': colors[:,0], 'color_g': colors[:,1], 'color_b': colors[:,2], 'color_alpha': colors[:,3]}
clusterDF = pd.DataFrame(clusterDF)
clusterDF.to_csv(os.path.join(derivatives, f'{sampleToCluster["sampleID"]}_cluster_information.csv'), index=False)
#%% save clustering data and input information
"""
Need to save certain data in order to be able to re-run clustering:
    eigenvectors/eigenvalues
    cluster_labels
    cluster colors
"""

#%% plot each cluster individually
plt.close('all')
plt.figure()
plt.imshow(sampleToCluster['tissueImageProcessed'],cmap='gray_r')
plt.scatter(sampleToCluster['processedTissuePositionList'][:,0], sampleToCluster['processedTissuePositionList'][:,1],c=colors, s=5)
plt.title("The visualization of the clustered control data.")
plt.axis('off')
plt.show()
for i in range(actK):
    plt.figure()
    clusterIdx = np.where(cluster_labels == i)[0]
    plt.imshow(sampleToCluster['tissueImageProcessed'],cmap='gray_r')
    plt.scatter(sampleToCluster['processedTissuePositionList'][clusterIdx,0], sampleToCluster['processedTissuePositionList'][clusterIdx,1],c=colors[clusterIdx], s=5)
    plt.title(f"The visualization of the clustered control data for cluster {i}.")
    plt.axis('off')
    plt.show()

#%% try to correlate gene expression with cluster label
plt.close('all')
gene_matrix = sampleToCluster['geneMatrixLog2'].todense()
# create a z-scored gene matrix 
z_score_gene_matrix = (gene_matrix - np.mean(gene_matrix, axis=1))/np.std(gene_matrix, axis=1)
clusterSpecificGeneList = []
clusterSpecificGeneIdx = []
for i in range(actK):
    clusterIdx = np.where(cluster_labels == i)
    # create cluster specific gene matrix from z-score matrix
    cluster_gene_matrix = z_score_gene_matrix[:,clusterIdx[0]]
    # sort cluster gene matrix to find genes with higher z-score within cluster compared to whole slice
    sorted_cluster_matrix_mean = np.argsort(np.squeeze(np.array(np.mean(cluster_gene_matrix, axis=1))))
    clusterSpecificGeneList.append(sampleToCluster['geneList'][sorted_cluster_matrix_mean[-1]])
    clusterSpecificGeneIdx.append(sorted_cluster_matrix_mean[-1])
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(sampleToCluster['tissueImageProcessed'],cmap='gray_r')
    ax[0].scatter(sampleToCluster['processedTissuePositionList'][clusterIdx,0], sampleToCluster['processedTissuePositionList'][clusterIdx,1],c=colors[clusterIdx], s=5)
    ax[0].axis('off')
    ax[0].set_title(f'Cluster {i}')
    ax[1].imshow(sampleToCluster['tissueImageProcessed'],cmap='gray_r')
    ax[1].scatter(sampleToCluster['processedTissuePositionList'][:,0], sampleToCluster['processedTissuePositionList'][:,1],c=np.squeeze(np.array(gene_matrix[sorted_cluster_matrix_mean[-1], :])), s=5, cmap='Reds')
    ax[1].axis('off')
    ax[1].set_title(f'{sampleToCluster["geneList"][sorted_cluster_matrix_mean[-1]]}')
    fig.set_size_inches(10,14)
    plt.show()
    plt.savefig(os.path.join(derivatives, f'{sampleToCluster["sampleID"]}_cluster{i}_hippocampus_top_gene_{sampleToCluster["geneList"][sorted_cluster_matrix_mean[-1]]}.png'), bbox_inches='tight', dpi=300)
    plt.close()
    # plt.title(f"The visualization of the clustered control data for cluster {i}.")
    # stanly.viewGeneInProcessedSample(sampleToCluster, sampleToCluster['geneList'][sorted_cluster_matrix_mean[-2]])

#%% calculate nearest neighbor cells to sparsify matrix
# to be incorporated into measureTranscriptomicSimilarity function
geneMatrix = processedSamples[0]['geneMatrixLog2']
# number of nearest neighbors to calculate
# kNN will only work along the cell axis, 
kNN = 50
axis = 1
spotCdist = sp_spatial.distance.cdist(processedSamples[0]['processedTissuePositionList'], processedSamples[0]['processedTissuePositionList'], 'euclidean')
sortedCdist = np.argsort(spotCdist, axis=1)
# create list of indices of nearest neighbors
kNNIdx = sortedCdist[:, 1:kNN+1]
def findKNearestNeighbors(tissuePositionList, kNN=50):
    spotCdist = sp_spatial.distance.cdist(tissuePositionList, tissuePositionList, 'euclidean')    
    sortedCdist = np.argsort(spotCdist, axis=1)
    kNNIdx = sortedCdist[:, 1:kNN+1]
    edgeList = []
    for i in range(len(tissuePositionList)):
        for j in range(kNN):
            edgeList.append([i, kNNIdx[i,j]])
    return np.array(edgeList, dtype='int32')

x = findKNearestNeighbors(processedSamples[0]['processedTissuePositionList'])
#%% update measureTranscriptomicSimilarity
def measureTranscriptomicSimilarity(geneMatrix, edgeList='FullyConnected', measurement='cosine', axis=1):
    """
    measure the transcriptomic similarity/distance between two spots
    ----------
    Parameters
    ----------
    geneMatrix: float array
        2D matrix of genetic or transcriptomic data, organized [gene,spot]
    edgeList: Nx2 int list
        List of edges to be measured for distance, [[spot1,spot2],[spot1,spot3],...]
    measurement: str
        Choice of measurement metric, default='cosine'
        'cosine' - cosine similarity
        'pearson' - Pearson's R correlation (**not yet implemented**)
    axis: int
        Which axis to run the similarity metric on, default=1, for cells/spots
    """
    
    # dataSimMatrix = []   
    n = geneMatrix.shape[axis]
    nInc = n/20
    percentLocations = np.arange(0, n, nInc, dtype='int32')
    percents = np.arange(5,101, 5)
    if axis == 0:
        geneMatrix = geneMatrix.tocsr()
    if edgeList == 'FullyConnected':
        dataSimMatrix = sp_sparse.lil_matrix((geneMatrix.shape[axis],geneMatrix.shape[axis]))
        for i in range(geneMatrix.shape[axis]):
            for j in range(i, geneMatrix.shape[axis]):
                if axis==1:
                    I = np.ravel(geneMatrix[:,i].todense())
                    J = np.ravel(geneMatrix[:,j].todense())
                elif axis==0:
                    I = np.ravel(geneMatrix[i,:].todense())
                    J = np.ravel(geneMatrix[j,:].todense())
                cs = sp_spatial.distance.cosine(I,J)
                dataSimMatrix[i,j] = float(cs)
                dataSimMatrix[j,i] = float(cs)
            if i in percentLocations:
                print(f"Calculation {percents[list(percentLocations).index(i)]}% completed")
        dataSimMatrix = dataSimMatrix.tocsc()
    return dataSimMatrix
#%% try clustering with reduced gene set, first remove genes with mean expression < 1
expression_mask = np.squeeze(np.array(np.mean(gene_matrix, axis=1) > 1))
high_expression_matrix = sp_sparse.csc_matrix(gene_matrix[np.array(clusterSpecificGeneIdx), :])
similarityMatrix = stanly.measureTranscriptomicSimilarity(high_expression_matrix, axis=1)
sp_sparse.save_npz(os.path.join(derivatives, f'{sampleToCluster["sampleID"]}_reduced_matrix.npz'), similarityMatrix)

#%%

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

# run sillhoutte analysis on control clustering
plt.close('all')
clusterRange = np.array(range(10,35))
# for actK in clusterRange:
    
# Create a subplot with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)

# The 1st subplot is the silhouette plot
# The silhouette coefficient can range from -1, 1 but in this example all
# lie within [-0.1, 1]
ax1.set_xlim([-1, 1])
# The (n_clusters+1)*10 is for inserting blank space between silhouette
# plots of individual clusters, to demarcate them clearly.
ax1.set_ylim([0, nDigitalSpots + (actK + 1) * 10])

clusters = KMeans(n_clusters=actK, init='random', n_init=500, tol=1e-8,)
cluster_labels = clusters.fit_predict(np.real(np.array(eigvecControlSort)[:,0:actK+1]))

# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed
# clusters
silhouette_avg = silhouette_score(np.real(np.array(eigvecControlSort)[:,0:actK]), cluster_labels)
print(
    "For n_clusters =",
    actK,
    "The average silhouette_score is :",
    silhouette_avg,
)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(np.real(np.array(eigvecControlSort)[:,0:actK]), cluster_labels)

y_lower = 10
for i in range(actK):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.tab20b(float(i) / actK)
    ax1.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,
    )

    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

# 2nd Plot showing the actual clusters formed
colors = cm.tab20b(cluster_labels.astype(float) / actK)
ax2.imshow(sampleToCluster['tissueImageProcessed'],cmap='gray_r')
ax2.scatter(sampleToCluster['processedTissuePositionList'][:,0], sampleToCluster['processedTissuePositionList'][:,1],c=colors, s=5)
ax2.set_title("The visualization of the clustered control data.")
ax2.axis('off')

plt.suptitle(
    "Silhouette analysis for KMeans clustering on control sample data with n_clusters = %d"
    % actK,
    fontsize=14,
    fontweight="bold",
)
plt.show()
#%%
"""
not presently working since it's still setup for clustering genes, not spots
"""
geneListForImage = []
geneCellCount = []
for i in range(actK):
    clusterIdx = np.where(cluster_labels == i)
    silhouettePerCluster = sample_silhouette_values[clusterIdx]
    topSilhouetteIdx = clusterIdx[0][np.argmax(silhouettePerCluster)]
    geneListForImage.append(sampleToCluster['geneList'][topSilhouetteIdx])
    geneCellCount.append(np.sum(sampleToCluster['geneMatrixLog2'][topSilhouetteIdx,:].todense() > 0))