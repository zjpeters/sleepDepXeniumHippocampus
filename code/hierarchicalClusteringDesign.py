#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 11:06:28 2025

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

"""
For hierarchical clustering, need to consider the stability of n-1 level clustering
before moving on to n level clustering, so should have a repeated step that then
chooses a cluster that maximizes consistency across runs therefore:
    
Begin by setting starting K for highest level cluster. Run this clustering for 
r repeats, collecting the cluster IDs with each r and finding which nodes cluster
together regularly and use these as the K-level cluster

Move to next K, a pre-defined distance away and repeat same approach of repeating
clustering and finding nodes that cluster together

Can do this by creating a square matrix to show associations with each node
With first iteration add 1 to show which nodes shared a cluster
continue until r has been completed and calculate a probability of shared cluster
"""

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

clusterRange = np.array(range(10,35))
startingK = 15
sampleToCluster = processedSamples[0]
nDigitalSpots = sampleToCluster['geneMatrixLog2'].shape[1]

probMatrix = np.zeros([nDigitalSpots, nDigitalSpots])

actK = startingK
nRepeats = 10
for r in range(nRepeats):
    start_time = time.time()
    simMatrixSavePath = os.path.join(derivatives, f'similarityMatrix_{sampleToCluster["sampleID"]}.npz')
    # check whether similarity matrix has already been calculated
    if os.path.exists(simMatrixSavePath):
        similarityMatrix = sp_sparse.load_npz(simMatrixSavePath)
    else:
        similarityMatrix = stanly.measureTranscriptomicSimilarity(sampleToCluster['geneMatrixLog2'], axis=1)
        sp_sparse.save_npz(simMatrixSavePath, similarityMatrix)
    
    Wcontrol = similarityMatrix.todense()
    
    #% create laplacian for control
    Wcontrol = (Wcontrol - np.nanmin(Wcontrol))/(np.nanmax(Wcontrol) - np.nanmin(Wcontrol))
    Wcontrol[Wcontrol==1] = 0
    # Wcontrol[np.isnan(Wcontrol)] = 0
    Dcontrol = np.diag(sum(Wcontrol))
    Lcontrol = Dcontrol - Wcontrol
    """
    k of eigvecs/vals should be related to the direction being clustered against, 
    i.e. since clustering cells by genes expressed in cells, k should be <= # of genes
    """
    eigvalControl,eigvecControl = scipy.sparse.linalg.eigs(Lcontrol, k=250)
    eigvalControlSort = np.sort(np.real(eigvalControl))[::-1]
    eigvalControlSortIdx = np.argsort(np.real(eigvalControl))[::-1]
    eigvecControlSort = np.real(eigvecControl[:,eigvalControlSortIdx])
    
    # run sillhoutte analysis on control clustering
    plt.close('all')
    # for actK in clusterRange:
        
    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)
    
    clusters = KMeans(n_clusters=actK, init='random', n_init=500, tol=1e-8,)
    cluster_labels = clusters.fit_predict(np.real(np.array(eigvecControlSort)[:,0:actK+1]))
    
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    
    # 2nd Plot showing the actual clusters formed
    colors = cm.tab20b(cluster_labels.astype(float) / actK)
    ax1.imshow(sampleToCluster['tissueImageProcessed'],cmap='gray_r')
    ax1.scatter(sampleToCluster['processedTissuePositionList'][:,0], sampleToCluster['processedTissuePositionList'][:,1],c=colors, s=5)
    ax1.axis('off')
    
    plt.suptitle(
        "KMeans clustering on sample data with n_clusters = %d"
        % actK,
        fontsize=14,
        fontweight="bold",
    )
    plt.show()


    print("--- %s seconds ---" % (time.time() - start_time))
    
    for i in range(actK):
        clusterIdx = np.where(cluster_labels == i)[0]
        for j in clusterIdx:
            for k in clusterIdx:
                probMatrix[j,k] += 1

probMatrix = probMatrix/nRepeats - np.eye(nDigitalSpots)
#%% testing creating clusters from repeated clusterings
for cell in range(nDigitalSpots):
    
    x = np.where(probMatrix[cell,:] > 0)[0]
