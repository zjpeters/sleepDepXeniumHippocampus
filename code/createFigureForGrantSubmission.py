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
rawdata=os.path.join('/','media','zjpeters','Expansion','sleepDepXenium','rawdata')
derivatives=os.path.join('/','media','zjpeters','Expansion','sleepDepXeniumHippocampus','derivatives')
# colors to use in the output figures, with scatter color being slightly darker
colorHCHex = '#f3766e'
colorHCScatterHex = '#fe4d34'
colorMDDHex = '#1cbdc2'
colorMDDScatterHex = '#14a7c2'
#%% load newly processed samples and check selection
# participant_list = ['YW-1_ROI_A1_hippocampus', 'YW-1_ROI_C1_hippocampus', 
#                     'YW-2_ROI_B1_hippocampus', 'YW-1_ROI_A2_hippocampus', 
#                     'YW-1_ROI_C2_hippocampus', 'YW-2_ROI_B2_hippocampus',
#                     'YW-1_ROI_B1_hippocampus', 'YW-2_ROI_A1_hippocampus', 
#                     'YW-2_ROI_C1_hippocampus', 'YW-1_ROI_B2_hippocampus', 
#                     'YW-2_ROI_A2_hippocampus', 'YW-2_ROI_C2_hippocampus']
# processedSamples = {}
experiment = stanly.loadParticipantsTsv(os.path.join('/','home','zjpeters','Documents','sleepDepXeniumHippocampus', 'participants.tsv'))
processedSamples = {}
# for actSample in range(len(experiment['sample-id'])):
#     sample = stanly.importXeniumData(os.path.join(rawdata, experiment['sample-id'][actSample]))
#     sampleProcessed = stanly.processXeniumData(sample, experiment['rotation'][actSample], derivatives)
#     processedSamples[actSample] = sampleProcessed
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
maleSamplesIdx = [0,1,2,4,5]
designMatrix = [0,1,0,0,1]
for i in maleSamplesIdx:
    maleSamples.append(processedSamples[i])

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
        
#%% run umap embedding using all cells, SD and NSD
plt.close('all')

# define colors for regions using uniformly distributed colors from original colormap
ca1Color = np.squeeze(np.array([0.807843137254902, 0.427450980392157, 0.741176470588235, 1]))
ca2Color = np.squeeze(np.array([0.83921568627451, 0.380392156862745, 0.419607843137255, 1]))
ca3Color = np.squeeze(np.array([0.741176470588235, 0.619607843137255, 0.223529411764706, 1]))
ca4dgColor = np.squeeze(np.array([0.388235294117647	, 0.474509803921569, 0.223529411764706, 1]))
dgColor = np.squeeze(np.array([0.223529411764706, 0.231372549019608, 0.474509803921569, 1]))
sparseColor = np.squeeze(np.array([0.7,0.7,0.7,1]))
colorsAllCells = np.empty([0,4])

allCells = np.empty([len(maleSamples[0]['geneList']),0])
for actSample in range(len(maleSamples)):
    allCells  = np.append(allCells , maleSamples[actSample]['geneMatrixLog2'].todense(), axis=1)
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
    colorsAllCells = np.append(colorsAllCells, sampleColors, axis=0)
#%% perform umap plot on combined cells

reducer = umap.UMAP()

embedding = reducer.fit_transform(np.array(allCells ).T)

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
# sfigs = fig.subfigures(1, 2)
# clustAxs = sfigs[0].subplots(1, 2)
# ttestAxs = sfigs[1].subplots(2, 1)

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
plt.savefig(os.path.join(derivatives, 'xenium_figure_k15cluster_umap_rbm3ttest_gpr161ttest.svg'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(derivatives, 'xenium_figure_k15cluster_umap_rbm3ttest_gpr161ttest.png'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(derivatives, 'xenium_figure_k15cluster_umap_rbm3ttest_gpr161ttest.pdf'), bbox_inches='tight', dpi=300)
