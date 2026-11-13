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
if os.path.exists(os.path.join("/", "home", "zjpeters", "Documents", "stanly", "code")):
    sys.path.insert(0, "/home/zjpeters/Documents/stanly/code")
else:
    sys.path.insert(0, os.path.join('C:',os.sep, 'Users','onyh19ug', 'Documents', 'STANLY','code'))
import stanly
import pandas as pd
import scipy.stats as scipy_stats
import umap
import matplotlib.patches as mpatches
import scipy.sparse as sp_sparse
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
plt.rcParams['text.usetex'] = False
plt.rcParams['svg.fonttype'] = 'none'

derivatives = os.path.join('/','home','zjpeters','Documents','sleepDepXeniumHippocampus','derivatives')
figureFolder = os.path.join('/','home','zjpeters','Documents','ovguMasters','sauvageRotationReport','figures')
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

cellsOfInterestColorList = [ca1Color, ca2Color, ca3Color, dgColor, ca4dgColor, astColor, endColor, micColor, neuColor, oliColor, sparseColor]

#%% load newly processed samples and check selection

locOfTsvFile = os.path.join('/','home','zjpeters','Documents','sleepDepXeniumHippocampus', 'participants.tsv')
participants = pd.read_csv(locOfTsvFile, delimiter='\t')

experiment = {'sample-id': participants['participant_id'].to_numpy(),
                    'rotation': participants['deg_rot'].to_numpy(),
                    'experimental-group': participants['sleep_dep'].to_numpy(),
                    'flip': participants['flip'].to_numpy(),
                    'sex': participants['sex'].to_numpy()}

processedSamples = {}

for sampleIdx in range(len(experiment['sample-id'])):
    processedSamples[sampleIdx] = stanly.loadProcessedXeniumSample(os.path.join(derivatives, f"{experiment['sample-id'][sampleIdx]}_hippocampus"))
    # clusterInfo = pd.read_csv(os.path.join(derivatives, f'{processedSamples[sampleIdx]["sampleID"]}_cluster_information.csv'))
    # processedSamples[sampleIdx]['cluster_labels'] = clusterInfo['cluster_labels']
    # processedSamples[sampleIdx]['silhouette_values'] = clusterInfo['silhouette_values']
    # processedSamples[sampleIdx]['cluster_colors'] = np.array([clusterInfo['color_r'], clusterInfo['color_g'], clusterInfo['color_b'], clusterInfo['color_alpha']]).T
    # cluster_regions = pd.read_csv(os.path.join(derivatives, f'{processedSamples[sampleIdx]["sampleID"]}_cluster_associations.csv'), header=None)
    # processedSamples[sampleIdx]['cluster_region'] = np.squeeze(np.array(cluster_regions[1]))

#%% plot marker genes from Jyoti
markerGeneDict = {
    "DG": ["Cdh9","Prox1"],                          
    "Oligo": ["Sox10", "Opalin", "Sgk1"],                       
    "Astro": ["Aqp4", "Gli3"], 
    "Endothelial": ["Cldn5", "Sox17"],              
    "CA1": ["Wfs1", "Fibcd1"],
    "VLMC": ["Igf2", "Dcn"],     
    "CA3": ["Neurod6", "Npy2r"],
    "Microglia": ["Cd53", "Trem2"],
    "OPC": ["Pdgfra"],
    "Interneuron": ["Gad2", "Gad1"],       
    "Spag16": ["Spag16"],
    "Thalamic Neurons": ["Slc17a6", "Rims3"],
    "CA2": ["Prdm8", "Igfbp4"],
    "Syt6": ["Syt6"],
    "Cortical Neuron": ["Hs3st2","Myl4"],
    "Cajal Retzius": ["Ebf3", "Cdh4"]
}

for cellType in markerGeneDict.keys():
    for gene in markerGeneDict[cellType]:
        stanly.viewGeneInProcessedSample(processedSamples[0], gene)

sampleForDisplay = processedSamples[0]
#%% calculate average number of cells per slice
cellNumberArray = []
for i in range(len(processedSamples)):
    cellNumberArray.append(len(processedSamples[i]['barcodeList']))
cellNumberArray = np.array(cellNumberArray)
print(np.mean(cellNumberArray))

#%% create variable containing all cells
maleSamples = []
maleSamplesIdx = [0,1,2,3,4,5]
for i in maleSamplesIdx:
    maleSamples.append(processedSamples[i])

designMatrix = [0,1,0,1,0,1]
allCellsMale = np.empty([len(maleSamples[0]['geneList']),0])
allClustersMale = np.empty(0, dtype='str')
sdNSDIdxMale = np.empty(0, dtype='int32')
sampleListMale = np.empty(0,dtype='int32')
allCoordinatesMale = np.empty([0,2])
for actSample in range(len(maleSamples)):
    allCellsMale = np.append(allCellsMale, maleSamples[actSample]['geneMatrixLog2'].todense(), axis=1)

#%% perform umap plot on combined cells

reducer = umap.UMAP(random_state=42)

embeddingMale = reducer.fit_transform(np.array(allCellsMale).T)

#%% display overall umap

# plt.scatter(embeddingMale[:,0], embeddingMale[:,1], s=2, linewidth=0)
# plt.show()

# look at ndst4 vs calb1 expression in all cells
ndst4Idx = processedSamples[0]['geneList'].index('Ndst4')
ndst4GeneExp = np.squeeze(np.array(allCellsMale[ndst4Idx,:]))

# Calb1 has higher expression in the inferior CA1
calb1Idx = processedSamples[0]['geneList'].index('Calb1')
calb1GeneExp = np.squeeze(np.array(allCellsMale[calb1Idx,:]))

# find shared cells
sharedMask = np.where((ndst4GeneExp > 0) & (calb1GeneExp > 0))[0]
ndst4GeneExp = ndst4GeneExp[sharedMask]
calb1GeneExp = calb1GeneExp[sharedMask]
maskedCoors = embeddingMale[sharedMask,:]
ndst4Mask = [] # np.zeros_like(ndst4GeneExp, dtype='int32')
for i in range(len(ndst4GeneExp)):
    if ndst4GeneExp[i] > calb1GeneExp[i]:
        # ndst4Mask[i] = 1
        ndst4Mask.append(i)
        
ndst4Mask = np.array(ndst4Mask)
plt.close('all')
fig, ax = plt.subplots(1,1)
ax.scatter(maskedCoors[:,0], maskedCoors[:,1], s=3, linewidth=0, vmin=-1, vmax=1)
ax.scatter(maskedCoors[ndst4Mask,0], maskedCoors[ndst4Mask,1], s=3, linewidth=0, vmin=-1, vmax=1)
plt.show()
#%% display overlay of "Gad1", "Sst", "Pvalb", "Gfap"

markerGeneList = ['Gad1', 'Sst', 'Pvalb', 'Gfap']

plt.close('all')
fig, ax = plt.subplots(1,1)
ax.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray')
for gene in range(len(markerGeneList)):
    geneName = markerGeneList[gene]
    geneIdx = sampleForDisplay['geneList'].index(geneName)
    geneExp = np.squeeze(np.array(sampleForDisplay['geneMatrixLog2'][geneIdx, :].todense()))
    geneMask = np.where(geneExp > 0)[0]
    #, c=geneExp[geneMask]
    ax.scatter(sampleForDisplay['processedTissuePositionList'][geneMask,0], sampleForDisplay['processedTissuePositionList'][geneMask,1], label=geneName, s=3, linewidth=0)
plt.legend()

#%% look for genes related to each of the clusters
# dataSimMatrix = sp_sparse.load_npz(os.path.join(derivatives, 'YW-1_ROI_A1_hippocampus', '')
geneListForGeneImage = stanly.selectGenePatterns(processedSamples[2])

plt.close('all')
for gene in geneListForGeneImage:
    stanly.viewGeneInProcessedSample(processedSamples[2], gene)
#%% display genes that should be specific to different regions of hippocampus
"""
 identified from the paper:
     Spatial Gene-Expression Gradients Underlie Prominent Heterogeneity of CA1 Pyramidal Neurons
"""
plt.close('all')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Gad1')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Gad2')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Gfap')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Npy2r')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Penk')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Sst')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Ntsr2')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Pvalb')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Calb1')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Calb2')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Wfs1')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Epha4')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Ndst4')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Nr2f2')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Car4')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Cpne4')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Cpne6')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Cpne8')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Slit2')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Nrp2')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Pde11a')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Col6a1')

#%% visualize gradient differences
# Ndst4 has higher expression in the deep CA1 (outer layer)
ndst4Idx = sampleForDisplay['geneList'].index('Ndst4')
ndst4GeneExp = np.squeeze(np.array(sampleForDisplay['geneMatrixLog2'][ndst4Idx,:].todense()))
ndst4GeneExp = (ndst4GeneExp - np.nanmin(ndst4GeneExp))/(np.nanmax(ndst4GeneExp) - np.nanmin(ndst4GeneExp))
# Calb1 has higher expression in the superficial CA1 (inner layer)
calb1Idx = sampleForDisplay['geneList'].index('Calb1')
calb1GeneExp = np.squeeze(np.array(sampleForDisplay['geneMatrixLog2'][calb1Idx,:].todense()))
calb1GeneExp = (calb1GeneExp - np.nanmin(calb1GeneExp))/(np.nanmax(calb1GeneExp) - np.nanmin(calb1GeneExp))
expRatio = calb1GeneExp/ndst4GeneExp
expRatio[expRatio == np.inf] = 0
expRatio[expRatio == np.nan] = 0
plt.close('all')
plt.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
plt.scatter(sampleForDisplay['processedTissuePositionList'][:,0], sampleForDisplay['processedTissuePositionList'][:,1], c=expRatio, cmap='seismic', s=3, linewidth=0, vmin=-1, vmax=1)
plt.show()

#%% display same ratio but using z-score
# Ndst4 has higher expression in the superior CA1
ndst4Idx = sampleForDisplay['geneList'].index('Ndst4')
ndst4GeneExp = np.squeeze(np.array(sampleForDisplay['geneMatrixLog2'][ndst4Idx,:].todense()))

# Calb1 has higher expression in the inferior CA1
calb1Idx = sampleForDisplay['geneList'].index('Calb1')
calb1GeneExp = np.squeeze(np.array(sampleForDisplay['geneMatrixLog2'][calb1Idx,:].todense()))

# find shared cells
sharedMask = np.where((ndst4GeneExp > 0) & (calb1GeneExp > 0))[0]
ndst4GeneExp = ndst4GeneExp[sharedMask]
calb1GeneExp = calb1GeneExp[sharedMask]
# calculate z-scores
# ndst4GeneExp = (ndst4GeneExp - np.mean(ndst4GeneExp))/np.std(ndst4GeneExp)
# calb1GeneExp = (calb1GeneExp - np.mean(calb1GeneExp))/np.std(calb1GeneExp)
expRatio = calb1GeneExp - ndst4GeneExp
expRatio[expRatio == np.inf] = 0
expRatio[expRatio == np.nan] = 0
plt.close('all')
plt.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
plt.scatter(sampleForDisplay['processedTissuePositionList'][sharedMask,0], sampleForDisplay['processedTissuePositionList'][sharedMask,1], c=expRatio, cmap='seismic', s=3, linewidth=0, vmin=-1, vmax=1)
plt.show()

#%% just use a relative comparison between the two
# Ndst4 has higher expression in the superior CA1
ndst4Idx = sampleForDisplay['geneList'].index('Ndst4')
ndst4GeneExp = np.squeeze(np.array(sampleForDisplay['geneMatrixLog2'][ndst4Idx,:].todense()))

# Calb1 has higher expression in the inferior CA1
calb1Idx = sampleForDisplay['geneList'].index('Calb1')
calb1GeneExp = np.squeeze(np.array(sampleForDisplay['geneMatrixLog2'][calb1Idx,:].todense()))

# find shared cells
sharedMask = np.where((ndst4GeneExp > 0) & (calb1GeneExp > 0))[0]
ndst4GeneExp = ndst4GeneExp[sharedMask]
calb1GeneExp = calb1GeneExp[sharedMask]
maskedCoors = sampleForDisplay['processedTissuePositionList'][sharedMask,:]
ndst4Mask = [] # np.zeros_like(ndst4GeneExp, dtype='int32')
for i in range(len(ndst4GeneExp)):
    if ndst4GeneExp[i] > calb1GeneExp[i]:
        # ndst4Mask[i] = 1
        ndst4Mask.append(i)
        
ndst4Mask = np.array(ndst4Mask)
plt.close('all')
plt.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
plt.scatter(maskedCoors[ndst4Mask,0], maskedCoors[ndst4Mask,1], s=3, linewidth=0, vmin=-1, vmax=1)
plt.show()



# calculate z-scores
# ndst4GeneExp = (ndst4GeneExp - np.mean(ndst4GeneExp))/np.std(ndst4GeneExp)
# calb1GeneExp = (calb1GeneExp - np.mean(calb1GeneExp))/np.std(calb1GeneExp)
expRatio = calb1GeneExp - ndst4GeneExp
expRatio[expRatio == np.inf] = 0
expRatio[expRatio == np.nan] = 0
plt.close('all')
plt.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
plt.scatter(sampleForDisplay['processedTissuePositionList'][sharedMask,0], sampleForDisplay['processedTissuePositionList'][sharedMask,1], c=expRatio, cmap='seismic', s=3, linewidth=0, vmin=-1, vmax=1)
plt.show()
#%% display genes that should be specific to different regions of hippocampus
"""
 identified from the paper:
     Genomic Anatomy of the Hippocampus
"""
plt.close('all')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Dkk3')

#%% display genes that should be specific to different regions of hippocampus
"""
 identified from the paper:
     Hipposeq: a comprehensive RNA-seq database of gene expression in hippocampal principal neurons
"""
plt.close('all')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Gad1')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Gad2')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Sst')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Pvalb')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Gfap')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Fibcd1')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Slc17a7')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Prox1')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Dkk3')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Calb2')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Fibcd1')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Gsg1l')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Nr2f2')

#%% display genes that should be specific to different regions of hippocampus
"""
 identified from the paper:
     Genomic-anatomic evidence for distinct functional domains in hippocampal field CA1 
"""
plt.close('all')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Wfs1')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Ndst4')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Dcn')
stanly.viewGeneInProcessedSample(processedSamples[0], 'Nr4a1')
# mentions Prss23, but Prss35 shows CA3 specificity
stanly.viewGeneInProcessedSample(processedSamples[0], 'Prss35')

#%% create figure showing genes with regional specificity and regional gradients
# to be used in report for time in Sauvage lab
sampleForDisplay = processedSamples[0]
# CA1
gene01 = 'Wfs1'
# CA1 inferior
gene02 = 'Npy2r'
# CA1 superior
gene03 = 'Ndst4'
# ca3
gene04 = 'Penk'

# plot Cirbp t-statistics
plt.close('all')

# fig = plt.figure(figsize=(6,4))
# for subplot2grid: size of plot (rows, columns) plot location (rows, columns) 
# gene01Plot = plt.subplot2grid((6,4), (0,0), colspan=3, rowspan=2)
# gene01Plot.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
# gene02Plot = plt.subplot2grid((6,4), (0,3), colspan=3, rowspan=2)
# gene02Plot.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
# gene03Plot = plt.subplot2grid((6,4), (2,0), colspan=3, rowspan=2)
# gene03Plot.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
# gene04Plot = plt.subplot2grid((6,4), (2,3), colspan=3, rowspan=2)
# gene04Plot.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')

fig, ax = plt.subplots(2,2, figsize=(4.8,3.6))

geneIdx = sampleForDisplay['geneList'].index(gene01)

geneArray = np.squeeze(np.array(sampleForDisplay['geneMatrixLog2'][geneIdx,:].todense()))
ax[0,0].imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
ax[0,0].scatter(sampleForDisplay['processedTissuePositionList'][:,0], sampleForDisplay['processedTissuePositionList'][:,1], c=geneArray, cmap='Reds', s=2, linewidth=0)
ax[0,0].set_title(gene01, style='italic')
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])

geneIdx = sampleForDisplay['geneList'].index(gene02)

geneArray = np.squeeze(np.array(sampleForDisplay['geneMatrixLog2'][geneIdx,:].todense()))
ax[0,1].imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
ax[0,1].scatter(sampleForDisplay['processedTissuePositionList'][:,0], sampleForDisplay['processedTissuePositionList'][:,1], c=geneArray, cmap='Reds', s=2, linewidth=0)
ax[0,1].set_title(gene02, style='italic')
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])

geneIdx = sampleForDisplay['geneList'].index(gene03)

geneArray = np.squeeze(np.array(sampleForDisplay['geneMatrixLog2'][geneIdx,:].todense()))
ax[1,0].imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
ax[1,0].scatter(sampleForDisplay['processedTissuePositionList'][:,0], sampleForDisplay['processedTissuePositionList'][:,1], c=geneArray, cmap='Reds', s=2, linewidth=0)
ax[1,0].set_title(gene03, style='italic')
ax[1,0].set_xticks([])
ax[1,0].set_yticks([])

geneIdx = sampleForDisplay['geneList'].index(gene04)

geneArray = np.squeeze(np.array(sampleForDisplay['geneMatrixLog2'][geneIdx,:].todense()))
ax[1,1].imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
ax[1,1].scatter(sampleForDisplay['processedTissuePositionList'][:,0], sampleForDisplay['processedTissuePositionList'][:,1], c=geneArray, cmap='Reds', s=2, linewidth=0)
ax[1,1].set_title(gene04, style='italic')
ax[1,1].set_xticks([])
ax[1,1].set_yticks([])

fig.text(0.10, 0.88, 'A', fontweight='bold', size=9)
fig.text(0.53, 0.88, 'B', fontweight='bold', size=9)
fig.text(0.10, 0.45, 'C', fontweight='bold', size=9)
fig.text(0.53, 0.45, 'D', fontweight='bold', size=9)

plt.show()
plt.savefig(os.path.join(figureFolder, 'comparison_of_specific_and_gradient_expression_4_genes.pdf'), bbox_inches='tight', dpi=300)

#%% create figure showing genes with regional specificity and regional gradients
# to be used in report for time in Sauvage lab
sampleForDisplay = processedSamples[0]
# CA1
gene01 = 'Wfs1'
# CA1 inferior
gene02 = 'Npy2r'
# CA1 superior
gene03 = 'Ndst4'
# ca3
gene04 = 'Penk'
# gradient expression
gene05 = 'Cpne8'
#
gene06 = 'Prox1'
# plot Cirbp t-statistics
plt.close('all')

# fig = plt.figure(figsize=(6,4))
# for subplot2grid: size of plot (rows, columns) plot location (rows, columns) 
# gene01Plot = plt.subplot2grid((6,4), (0,0), colspan=3, rowspan=2)
# gene01Plot.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
# gene02Plot = plt.subplot2grid((6,4), (0,3), colspan=3, rowspan=2)
# gene02Plot.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
# gene03Plot = plt.subplot2grid((6,4), (2,0), colspan=3, rowspan=2)
# gene03Plot.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
# gene04Plot = plt.subplot2grid((6,4), (2,3), colspan=3, rowspan=2)
# gene04Plot.imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')

fig, ax = plt.subplots(3,2, figsize=(4.8,4.5))

geneIdx = sampleForDisplay['geneList'].index(gene01)

geneArray = np.squeeze(np.array(sampleForDisplay['geneMatrixLog2'][geneIdx,:].todense()))
ax[0,0].imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
ax[0,0].scatter(sampleForDisplay['processedTissuePositionList'][:,0], sampleForDisplay['processedTissuePositionList'][:,1], c=geneArray, cmap='Reds', s=2, linewidth=0)
ax[0,0].set_title(gene01, style='italic')
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])

geneIdx = sampleForDisplay['geneList'].index(gene02)

geneArray = np.squeeze(np.array(sampleForDisplay['geneMatrixLog2'][geneIdx,:].todense()))
ax[0,1].imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
ax[0,1].scatter(sampleForDisplay['processedTissuePositionList'][:,0], sampleForDisplay['processedTissuePositionList'][:,1], c=geneArray, cmap='Reds', s=2, linewidth=0)
ax[0,1].set_title(gene02, style='italic')
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])

geneIdx = sampleForDisplay['geneList'].index(gene03)

geneArray = np.squeeze(np.array(sampleForDisplay['geneMatrixLog2'][geneIdx,:].todense()))
ax[1,0].imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
ax[1,0].scatter(sampleForDisplay['processedTissuePositionList'][:,0], sampleForDisplay['processedTissuePositionList'][:,1], c=geneArray, cmap='Reds', s=2, linewidth=0)
ax[1,0].set_title(gene03, style='italic')
ax[1,0].set_xticks([])
ax[1,0].set_yticks([])

geneIdx = sampleForDisplay['geneList'].index(gene04)

geneArray = np.squeeze(np.array(sampleForDisplay['geneMatrixLog2'][geneIdx,:].todense()))
ax[1,1].imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
ax[1,1].scatter(sampleForDisplay['processedTissuePositionList'][:,0], sampleForDisplay['processedTissuePositionList'][:,1], c=geneArray, cmap='Reds', s=2, linewidth=0)
ax[1,1].set_title(gene04, style='italic')
ax[1,1].set_xticks([])
ax[1,1].set_yticks([])

geneIdx = sampleForDisplay['geneList'].index(gene05)

geneArray = np.squeeze(np.array(sampleForDisplay['geneMatrixLog2'][geneIdx,:].todense()))
ax[2,0].imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
ax[2,0].scatter(sampleForDisplay['processedTissuePositionList'][:,0], sampleForDisplay['processedTissuePositionList'][:,1], c=geneArray, cmap='Reds', s=2, linewidth=0)
ax[2,0].set_title(gene05, style='italic')
ax[2,0].set_xticks([])
ax[2,0].set_yticks([])

geneIdx = sampleForDisplay['geneList'].index(gene06)

geneArray = np.squeeze(np.array(sampleForDisplay['geneMatrixLog2'][geneIdx,:].todense()))
ax[2,1].imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
ax[2,1].scatter(sampleForDisplay['processedTissuePositionList'][:,0], sampleForDisplay['processedTissuePositionList'][:,1], c=geneArray, cmap='Reds', s=2, linewidth=0)
ax[2,1].set_title(gene06, style='italic')
ax[2,1].set_xticks([])
ax[2,1].set_yticks([])


fig.text(0.10, 0.88, 'A', fontweight='bold', size=9)
fig.text(0.53, 0.88, 'B', fontweight='bold', size=9)
fig.text(0.10, 0.61, 'C', fontweight='bold', size=9)
fig.text(0.53, 0.61, 'D', fontweight='bold', size=9)
fig.text(0.10, 0.34, 'E', fontweight='bold', size=9)
fig.text(0.53, 0.34, 'F', fontweight='bold', size=9)

plt.show()
plt.savefig(os.path.join(figureFolder, 'comparison_of_specific_and_gradient_expression_6_genes.pdf'), bbox_inches='tight', dpi=300)
#%% identify gradient of gene expression
plt.close('all')
# Cpne8 shows gradient from CA1 to CA2 to CA3
geneIdx = processedSamples[0]['geneList'].index('Cpne8')
geneExp = np.squeeze(np.array(processedSamples[0]['geneMatrixLog2'][geneIdx,:].todense()))
arrayCoordinates = np.round(processedSamples[0]['processedTissuePositionList']*5)
# account for potentially out of bounds data
if np.min(arrayCoordinates[:,0]) < 0:
    print(np.min(arrayCoordinates[:,0]))
    arrayCoordinates[:,0] = arrayCoordinates[:,0] - np.min(arrayCoordinates[:,0])

if np.min(arrayCoordinates[:,1]) < 0:
    print(np.min(arrayCoordinates[:,1]))
    arrayCoordinates[:,1] = arrayCoordinates[:,1] - np.min(arrayCoordinates[:,1])
    
# checking that all coordinates are properly displayed
print(len(np.unique(arrayCoordinates, axis=0)))
# geneArray = np.zeros([processedSamples[0]['tissueImageProcessed'].shape[0]*5, processedSamples[0]['tissueImageProcessed'].shape[1]*5])
geneArray = sp_sparse.coo_array((geneExp, arrayCoordinates.T), [(processedSamples[0]['tissueImageProcessed'].shape[1]*5)+2, (processedSamples[0]['tissueImageProcessed'].shape[0]*5)+2]).todense()

geneArrayGradient = np.gradient(geneArray)
fig, ax = plt.subplots(1,2)
ax[0].imshow(geneArrayGradient[0], cmap='seismic', vmin=-1, vmax=1)
ax[1].imshow(geneArrayGradient[1], cmap='seismic', vmin=-1, vmax=1)
plt.show()
# Ndst4 shows gradient from Deep CA1 to superficial CA1
geneIdx = processedSamples[0]['geneList'].index('Ndst4')
geneExp = np.squeeze(np.array(processedSamples[0]['geneMatrixLog2'][geneIdx,:].todense()))
arrayCoordinates = np.round(processedSamples[0]['processedTissuePositionList']*5)
# account for potentially out of bounds data
if np.min(arrayCoordinates[:,0]) < 0:
    print(np.min(arrayCoordinates[:,0]))
    arrayCoordinates[:,0] = arrayCoordinates[:,0] - np.min(arrayCoordinates[:,0])

if np.min(arrayCoordinates[:,1]) < 0:
    print(np.min(arrayCoordinates[:,1]))
    arrayCoordinates[:,1] = arrayCoordinates[:,1] - np.min(arrayCoordinates[:,1])
    
# checking that all coordinates are properly displayed
print(len(np.unique(arrayCoordinates, axis=0)))
# geneArray = np.zeros([processedSamples[0]['tissueImageProcessed'].shape[0]*5, processedSamples[0]['tissueImageProcessed'].shape[1]*5])
geneArray = sp_sparse.coo_array((geneExp, arrayCoordinates.T), [(processedSamples[0]['tissueImageProcessed'].shape[1]*5)+2, (processedSamples[0]['tissueImageProcessed'].shape[0]*5)+2]).todense()

geneArrayGradient = np.gradient(geneArray)
fig, ax = plt.subplots(1,2)
ax[0].imshow(geneArrayGradient[0], cmap='seismic', vmin=-1, vmax=1)
ax[1].imshow(geneArrayGradient[1], cmap='seismic', vmin=-1, vmax=1)
plt.show()

# Calb1 shows gradient from superficial CA1 to Deep CA1 
geneIdx = processedSamples[0]['geneList'].index('Calb1')
geneExp = np.squeeze(np.array(processedSamples[0]['geneMatrixLog2'][geneIdx,:].todense()))
arrayCoordinates = np.round(processedSamples[0]['processedTissuePositionList']*5)
# account for potentially out of bounds data
if np.min(arrayCoordinates[:,0]) < 0:
    print(np.min(arrayCoordinates[:,0]))
    arrayCoordinates[:,0] = arrayCoordinates[:,0] - np.min(arrayCoordinates[:,0])

if np.min(arrayCoordinates[:,1]) < 0:
    print(np.min(arrayCoordinates[:,1]))
    arrayCoordinates[:,1] = arrayCoordinates[:,1] - np.min(arrayCoordinates[:,1])
    
# checking that all coordinates are properly displayed
print(len(np.unique(arrayCoordinates, axis=0)))
# geneArray = np.zeros([processedSamples[0]['tissueImageProcessed'].shape[0]*5, processedSamples[0]['tissueImageProcessed'].shape[1]*5])
geneArray = sp_sparse.coo_array((geneExp, arrayCoordinates.T), [(processedSamples[0]['tissueImageProcessed'].shape[1]*5)+2, (processedSamples[0]['tissueImageProcessed'].shape[0]*5)+2]).todense()

geneArrayGradient = np.gradient(geneArray)
fig, ax = plt.subplots(1,2)
ax[0].imshow(geneArrayGradient[0], cmap='seismic', vmin=-1, vmax=1)
ax[1].imshow(geneArrayGradient[1], cmap='seismic', vmin=-1, vmax=1)
plt.show()

# Wfs1 shows gradient from distal to proximal CA1
geneIdx = processedSamples[0]['geneList'].index('Wfs1')
geneExp = np.squeeze(np.array(processedSamples[0]['geneMatrixLog2'][geneIdx,:].todense()))
arrayCoordinates = np.round(processedSamples[0]['processedTissuePositionList']*5)
# account for potentially out of bounds data
if np.min(arrayCoordinates[:,0]) < 0:
    print(np.min(arrayCoordinates[:,0]))
    arrayCoordinates[:,0] = arrayCoordinates[:,0] - np.min(arrayCoordinates[:,0])

if np.min(arrayCoordinates[:,1]) < 0:
    print(np.min(arrayCoordinates[:,1]))
    arrayCoordinates[:,1] = arrayCoordinates[:,1] - np.min(arrayCoordinates[:,1])
    
# checking that all coordinates are properly displayed
print(len(np.unique(arrayCoordinates, axis=0)))
# geneArray = np.zeros([processedSamples[0]['tissueImageProcessed'].shape[0]*5, processedSamples[0]['tissueImageProcessed'].shape[1]*5])
geneArray = sp_sparse.coo_array((geneExp, arrayCoordinates.T), [(processedSamples[0]['tissueImageProcessed'].shape[1]*5)+2, (processedSamples[0]['tissueImageProcessed'].shape[0]*5)+2]).todense()

geneArrayGradient = np.gradient(geneArray)
fig, ax = plt.subplots(1,2)
ax[0].imshow(geneArrayGradient[0], cmap='seismic', vmin=-1, vmax=1)
ax[1].imshow(geneArrayGradient[1], cmap='seismic', vmin=-1, vmax=1)
plt.show()

#%% additive array
plt.close('all')
geneIdx = processedSamples[0]['geneList'].index('Cpne8')
geneArray = np.zeros([processedSamples[0]['tissueImageProcessed'].shape[0]+1, processedSamples[0]['tissueImageProcessed'].shape[1]+1])
arrayCoordinates = np.array(np.round(processedSamples[0]['processedTissuePositionList']), dtype='int32')

for i in enumerate(arrayCoordinates):
    geneArray[i[1][1], i[1][0]] += 1

plt.figure()
plt.imshow(geneArray, cmap='Reds')
plt.show()

geneArrayGradient = np.gradient(geneArray)
fig, ax = plt.subplots(1,2)
ax[0].imshow(geneArrayGradient[0], cmap='seismic', vmin=-1, vmax=1)
ax[1].imshow(geneArrayGradient[1], cmap='seismic', vmin=-1, vmax=1)
plt.show()

#%% load visium sample to count number of spots per hippocampus
template = stanly.chooseTemplateSlice(70)
visiumLocation = os.path.join('/','home','zjpeters','Documents','stanly','rawdata','sample-05')
visiumSample = stanly.importVisiumData(visiumLocation)
visiumProcessed = stanly.processVisiumData(visiumSample, template, 90, os.path.join('/','home','zjpeters','Documents','ovguMasters','sauvageRotationReport'))
stanly.viewGeneInProcessedSample(visiumProcessed, 'Arc')

#%% 
# lasso = stanly.SelectUsingLasso(visiumProcessed, "sample-05_hippocampus", os.path.join('/','home','zjpeters','Documents','ovguMasters','sauvageRotationReport'))
# lasso.applyLasso()
# lassoSample = lasso.outputMaskedSample(visiumProcessed)

lassoSample = stanly.loadProcessedVisiumSample(os.path.join('/','home','zjpeters','Documents','ovguMasters','sauvageRotationReport', 'sample-05_hippocampus'))
#%% 
visiumCoorFlipped, visiumImageFlipped, _ = stanly.rotateTissuePoints(lassoSample['processedTissuePositionList'], lassoSample['tissueImageProcessed'], 0, flip=True)
plt.close('all')
fig, ax = plt.subplots(1,2, figsize=(6.8,2.6))
visiumArcIdx = lassoSample['geneListMasked'].index('Arc')
visiumGeneArray = np.squeeze(np.array(lassoSample['geneMatrixLog2'][visiumArcIdx,:].todense()))
ax[0].imshow(visiumImageFlipped, cmap='gray_r')
ax[0].scatter(visiumCoorFlipped[:,0], visiumCoorFlipped[:,1], c=visiumGeneArray, cmap='Reds', s=6)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title(f"Visium, {visiumCoorFlipped.shape[0]} spots")

xeniumArcIdx = sampleForDisplay['geneList'].index('Arc')
xeniumGeneArray = np.squeeze(np.array(sampleForDisplay['geneMatrixLog2'][xeniumArcIdx,:].todense()))
ax[1].imshow(sampleForDisplay['tissueImageProcessed'], cmap='gray_r')
ax[1].scatter(sampleForDisplay['processedTissuePositionList'][:,0], sampleForDisplay['processedTissuePositionList'][:,1], c=xeniumGeneArray, cmap='Reds', s=1, linewidth=0)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title(f"Xenium, {sampleForDisplay['processedTissuePositionList'].shape[0]} cells")

plt.suptitle("Expression of Arc")
plt.show()
plt.savefig(os.path.join(figureFolder, 'comparison_of_xenium_and_visium.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(figureFolder, 'comparison_of_xenium_and_visium.png'), bbox_inches='tight', dpi=300)