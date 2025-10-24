#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 12:27:19 2025

@author: zjpeters
"""
import os
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.insert(0, "/home/zjpeters/Documents/stanly/code")
import stanly
import tifffile
import itk
import scipy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import scipy.sparse as sp_sparse
import json 
import ants
import scipy.stats as scipy_stats

rawdata=os.path.join('/','media','zjpeters','Expansion','sleepDepXenium','rawdata')
derivatives=os.path.join('/','media','zjpeters','Expansion','sleepDepXeniumHippocampus','derivatives')

#%% load newly processed samples and check selection
# sampleXenium = stanly.importXeniumData(os.path.join(rawdata,'YW-1_ROI_A1'))
experiment = stanly.loadParticipantsTsv(os.path.join(rawdata, 'participants.tsv'))
processedSamples = {}
for sampleIdx in range(len(experiment['sample-id'])):
    processedSamples[sampleIdx] = stanly.loadProcessedXeniumSample(os.path.join(derivatives, f"{experiment['sample-id'][sampleIdx]}_hippocampus"))
    plt.figure()
    plt.imshow(processedSamples[sampleIdx]['tissueImageProcessed'], cmap='gray')
    plt.show()
    
#%% look for gene list
geneListForImageSorted = stanly.selectGenePatterns(processedSamples[0], k=16)

#%% visualize all of the gene list genes
plt.close('all')
for i in geneListForImageSorted:
    stanly.viewGeneInProcessedSample(processedSamples[0], i)
    
"""
After looking at images, the following genes are most instructive for Hippocampus
['Jdp2','Neurod6','Npnt','Gpr161']
"""

#%% display genes from hippocapal gene list
"""
After examining the gene patterns collected above, found the following four:
    'Jdp2' : Expressed throughout entirety of hippocampal pyramidal cells
    'Nwd2' : Strongly expressed in CA2/3
    'Npnt' : Strongly expressed in DG
    'Gpr161' : Strongly expresed in CA1
"""

plt.close('all')
hippocampalGeneList = ['Jdp2', 'Nwd2','Npnt','Gpr161']
for i in hippocampalGeneList:
    stanly.viewGeneInProcessedSample(processedSamples[0], i)

#%% create gene image
for i in range(len(processedSamples)):
    geneImage = stanly.createGeneImageFromProcessedSample(processedSamples[i], hippocampalGeneList, displayImage=False, pixelCombination='additive')
    processedSamples[i]['geneImage'] = geneImage
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(processedSamples[i]['tissueImageProcessed'])
    ax[1].imshow(processedSamples[i]['geneImage'])
    # plt.scatter(processedSamples[i]['processedTissuePositionList'][:,0], processedSamples[i]['processedTissuePositionList'][:,1], alpha=0.4)
    plt.title(processedSamples[i]['sampleID'])
    plt.show()
    
#%% test some ants functions
# plt.close('all')
# ccMetric = ['CC', ants.from_numpy(processedSamples[0]['tissueImageProcessed']), ants.from_numpy(processedSamples[1]['tissueImageProcessed']), 2, 1 ]
# secondMetric = ['CC', ants.from_numpy(processedSamples[0]['placeHolder']), ants.from_numpy(processedSamples[1]['placeHolder']), 2, 1 ]
# metrics = list()
# metrics.append(ccMetric)
# metrics.append(secondMetric)
# reg = ants.registration(ants.from_numpy(processedSamples[0]['geneImage']), ants.from_numpy(processedSamples[1]['geneImage']), multivariate_extras = metrics)
# plt.imshow(processedSamples[0]['geneImage'])
# plt.imshow(reg['warpedmovout'].numpy(), alpha=0.5, cmap='gray_r')
#print(ants.image_mutual_information(ants.from_numpy(processedSamples[0]['geneImage']), reg['warpedmovout']))
#%% test registration with gene image
experimentalResults = {}
for actSample in range(len(processedSamples)):
    sampleRegistered = stanly.runANTsInterSampleRegistration(processedSamples[actSample], processedSamples[0], regParams='xenium_to_regional', imageToRegister='geneImage')
    experimentalResults[actSample] = sampleRegistered

#%% display registered images
plt.close('all')
for i in range(len(experimentalResults)):
    plt.figure()
    plt.imshow(processedSamples[0]['tissueImageProcessed'])
    plt.imshow(experimentalResults[i]['tissueImageRegistered'], cmap='gray_r', alpha=0.5)
    plt.title(experimentalResults[i]['sampleID'])
    plt.show()

#%%

stanly.viewGeneInProcessedSample(experimentalResults[11], 'Homer1')
#%% split groups into male and female
# samples 3 and 9 are not registering well, so left out
femaleSamples = []
femaleSamplesIdx = [6,7,8,10,11]
maleSamples = []
maleSamplesIdx = [0,1,2,4,5]
for i in maleSamplesIdx:
    maleSamples.append(experimentalResults[i])

for i in femaleSamplesIdx:
    femaleSamples.append(experimentalResults[i])
        
#%% create FOVs
"""
male analysis below
"""
nFOVs = 5000
fovLists = []
fovCentroids = []
for i in range(len(maleSamples)):
    fovList, fovCentroid, _ = stanly.createFOVsForMerscope(maleSamples[i], nFOVs, displayImage=False)
    fovLists.append(fovList)
    fovCentroids.append(fovCentroid)
    
#%% check for shared fovs
sharedFOVs = set(fovLists[0])
for i in fovLists:
    sharedFOVs = sharedFOVs.intersection(set(i))

sharedFOVs = np.sort(np.array(list(sharedFOVs)))
# collect coordinates of shared fovs
sharedFOVsCoor = []
for fov in sharedFOVs:
    sharedFOVsCoor.append(fovCentroid[fovList.index(fov),:])
sharedFOVsCoor = np.array(sharedFOVsCoor)

#%% testing bootstrapping/permutation testing of data
def ttest(control, experimental):
    actTStats, actPvals = scipy.stats.ttest_ind(control, experimental, axis=1, nan_policy='omit')
    return actTStats

designMatrix = [0,1,0,0,1]
nOfGenes = len(maleSamples[0]['geneListMasked'])
fovPValMatrix = np.empty([len(sharedFOVs), nOfGenes])
fovPValMatrix[:] = np.nan
fovTStatMatrix = np.empty([len(sharedFOVs), nOfGenes])
fovTStatMatrix[:] = np.nan
for actFov in enumerate(sharedFOVs):
    digitalSamplesControl = np.empty((nOfGenes, 0))
    digitalSamplesExperimental = np.empty((nOfGenes, 0))
    for actSample in enumerate(maleSamples):
        fovIdxs = np.where(np.array(np.array(fovLists[actSample[0]]) == actFov[1]))[0]
        if designMatrix[actSample[0]] == 1:
            digitalSamplesControl = np.append(digitalSamplesControl, np.array(maleSamples[actSample[0]]['geneMatrixLog2'][:,fovIdxs].todense()), axis=1)
        else:
            digitalSamplesExperimental = np.append(digitalSamplesExperimental, np.array(maleSamples[actSample[0]]['geneMatrixLog2'][:,fovIdxs].todense()), axis=1)
    actTStats, actPvals = scipy_stats.permutation_test((digitalSamplesControl, digitalSamplesExperimental), ttest, n_resamples=10000, random_state=12345)
    fovPValMatrix[actFov[0],:] = actPvals
    fovTStatMatrix[actFov[0],:] = actTStats
#%% remove the loop over genes and instead calculat t-test for all genes simultaneously

designMatrix = [0,1,0,0,1]
nOfGenes = len(maleSamples[0]['geneListMasked'])
fovPValMatrix = np.empty([len(sharedFOVs), nOfGenes])
fovPValMatrix[:] = np.nan
fovTStatMatrix = np.empty([len(sharedFOVs), nOfGenes])
fovTStatMatrix[:] = np.nan
for actFov in enumerate(sharedFOVs):
    digitalSamplesControl = np.empty((nOfGenes, 0))
    digitalSamplesExperimental = np.empty((nOfGenes, 0))
    for actSample in enumerate(maleSamples):
        fovIdxs = np.where(np.array(np.array(fovLists[actSample[0]]) == actFov[1]))[0]
        if designMatrix[actSample[0]] == 1:
            digitalSamplesControl = np.append(digitalSamplesControl, np.array(maleSamples[actSample[0]]['geneMatrixLog2'][:,fovIdxs].todense()), axis=1)
        else:
            digitalSamplesExperimental = np.append(digitalSamplesExperimental, np.array(maleSamples[actSample[0]]['geneMatrixLog2'][:,fovIdxs].todense()), axis=1)
    actTStats, actPvals = scipy.stats.ttest_ind(digitalSamplesControl,digitalSamplesExperimental, axis=1, nan_policy='omit')
    fovPValMatrix[actFov[0],:] = actPvals
    fovTStatMatrix[actFov[0],:] = actTStats

#%% check for significant genes
plt.close('all')
desiredPval = 0.05
alphaFdr = 1 - np.power((1 - desiredPval),(1/(len(sharedFOVs))))
checkForSig = np.nanmin(fovPValMatrix, axis=0)
nSigGenes = 0
for actP in enumerate(checkForSig):
    if actP[1] > 0 and actP[1] < alphaFdr:
        actGene = maleSamples[0]['geneListMasked'][actP[0]]
        print(actP)
        nSigGenes += 1
        plt.figure()
        plt.imshow(maleSamples[0]['tissueImageRegistered'], cmap='gray_r')
        plt.scatter(sharedFOVsCoor[:,0], sharedFOVsCoor[:,1], c=fovTStatMatrix[:,actP[0]], cmap='seismic',alpha=0.7,vmin=-4,vmax=4,plotnonfinite=False, s=10)
        plt.show()
        plt.title(f"T-statistic for {actGene}, p-value={actP[1]}**")
        plt.savefig(os.path.join(derivatives,f'xeniumTStat_male_SDvsNSD_{actGene}_nFOVs{nFOVs}_uncorrPVal{desiredPval}.png'), bbox_inches='tight', dpi=300)
        plt.close()
    else:
        actGene = maleSamples[0]['geneListMasked'][actP[0]]
        print(actP)
        nSigGenes += 1
        plt.figure()
        plt.imshow(maleSamples[0]['tissueImageRegistered'], cmap='gray_r')
        plt.scatter(sharedFOVsCoor[:,0], sharedFOVsCoor[:,1], c=fovTStatMatrix[:,actP[0]], cmap='seismic',alpha=0.7,vmin=-4,vmax=4,plotnonfinite=False, s=10)
        plt.show()
        plt.title(f"T-statistic for {actGene}, p-value={actP[1]}")
        plt.savefig(os.path.join(derivatives,f'xeniumTStat_male_SDvsNSD_{actGene}_nFOVs{nFOVs}_uncorrPVal{desiredPval}.png'), bbox_inches='tight', dpi=300)
        plt.close()

#%% create FOVs for female
"""
female analysis below
"""
nFOVs = 5000
fovLists = []
fovCentroids = []
for i in range(len(femaleSamples)):
    fovList, fovCentroid, _ = stanly.createFOVsForMerscope(femaleSamples[i], nFOVs, displayImage=False)
    fovLists.append(fovList)
    fovCentroids.append(fovCentroid)
    
#%% check for shared fovs
sharedFOVs = set(fovLists[0])
for i in fovLists:
    sharedFOVs = sharedFOVs.intersection(set(i))

sharedFOVs = np.sort(np.array(list(sharedFOVs)))
# collect coordinates of shared fovs
sharedFOVsCoor = []
for fov in sharedFOVs:
    sharedFOVsCoor.append(fovCentroid[fovList.index(fov),:])
sharedFOVsCoor = np.array(sharedFOVsCoor)
#%% remove the loop over genes and instead calculat t-test for all genes simultaneously

designMatrix = [0,1,0,0,1]
nOfGenes = len(femaleSamples[0]['geneListMasked'])
fovPValMatrix = np.empty([len(sharedFOVs), nOfGenes])
fovPValMatrix[:] = np.nan
fovTStatMatrix = np.empty([len(sharedFOVs), nOfGenes])
fovTStatMatrix[:] = np.nan
for actFov in enumerate(sharedFOVs):
    digitalSamplesControl = np.empty((nOfGenes, 0))
    digitalSamplesExperimental = np.empty((nOfGenes, 0))
    for actSample in enumerate(femaleSamples):
        fovIdxs = np.where(np.array(np.array(fovLists[actSample[0]]) == actFov[1]))[0]
        if designMatrix[actSample[0]] == 1:
            digitalSamplesControl = np.append(digitalSamplesControl, np.array(femaleSamples[actSample[0]]['geneMatrixLog2'][:,fovIdxs].todense()), axis=1)
        else:
            digitalSamplesExperimental = np.append(digitalSamplesExperimental, np.array(femaleSamples[actSample[0]]['geneMatrixLog2'][:,fovIdxs].todense()), axis=1)
    actTStats, actPvals = scipy.stats.ttest_ind(digitalSamplesControl,digitalSamplesExperimental, axis=1, nan_policy='omit')
    fovPValMatrix[actFov[0],:] = actPvals
    fovTStatMatrix[actFov[0],:] = actTStats

#%% check for significant genes
plt.close('all')
desiredPval = 0.05
alphaFdr = 1 - np.power((1 - desiredPval),(1/(len(sharedFOVs))))
checkForSig = np.nanmin(fovPValMatrix, axis=0)
nSigGenes = 0
for actP in enumerate(checkForSig):
    if actP[1] > 0 and actP[1] < alphaFdr:
        actGene = femaleSamples[0]['geneListMasked'][actP[0]]
        print(actP)
        nSigGenes += 1
        plt.figure()
        plt.imshow(femaleSamples[0]['tissueImageRegistered'], cmap='gray_r')
        plt.scatter(sharedFOVsCoor[:,0], sharedFOVsCoor[:,1], c=fovTStatMatrix[:,actP[0]], cmap='seismic',alpha=0.7,vmin=-4,vmax=4,plotnonfinite=False, s=10)
        plt.show()
        plt.title(f"T-statistic for {actGene}, p-value={actP[1]}")
        plt.savefig(os.path.join(derivatives,f'xeniumTStat_female_SDvsNSD_{actGene}_nFOVs{nFOVs}_uncorrPVal{desiredPval}.png'), bbox_inches='tight', dpi=300)
        plt.close()
        
#%% whole sample
# samples 3 and 9 are not registering well, so left out
allSamples = []
allSamplesIdx = [0,1,2,4,5,6,7,8,10,11]
for i in allSamplesIdx:
    allSamples.append(experimentalResults[i])
    
#%% create FOVs
"""
all sample analysis below
"""
nFOVs = 5000
fovLists = []
fovCentroids = []
for i in range(len(allSamples)):
    fovList, fovCentroid, _ = stanly.createFOVsForMerscope(allSamples[i], nFOVs, displayImage=False)
    fovLists.append(fovList)
    fovCentroids.append(fovCentroid)
    
#%% check for shared fovs
sharedFOVs = set(fovLists[0])
for i in fovLists:
    sharedFOVs = sharedFOVs.intersection(set(i))

sharedFOVs = np.sort(np.array(list(sharedFOVs)))
# collect coordinates of shared fovs
sharedFOVsCoor = []
for fov in sharedFOVs:
    sharedFOVsCoor.append(fovCentroid[fovList.index(fov),:])
sharedFOVsCoor = np.array(sharedFOVsCoor)
#%% remove the loop over genes and instead calculat t-test for all genes simultaneously

designMatrix = [0,1,0,0,1,0,1,0,0,1]
nOfGenes = len(allSamples[0]['geneListMasked'])
fovPValMatrix = np.empty([len(sharedFOVs), nOfGenes])
fovPValMatrix[:] = np.nan
fovTStatMatrix = np.empty([len(sharedFOVs), nOfGenes])
fovTStatMatrix[:] = np.nan
for actFov in enumerate(sharedFOVs):
    digitalSamplesControl = np.empty((nOfGenes, 0))
    digitalSamplesExperimental = np.empty((nOfGenes, 0))
    for actSample in enumerate(allSamples):
        fovIdxs = np.where(np.array(np.array(fovLists[actSample[0]]) == actFov[1]))[0]
        if designMatrix[actSample[0]] == 1:
            digitalSamplesControl = np.append(digitalSamplesControl, np.array(allSamples[actSample[0]]['geneMatrixLog2'][:,fovIdxs].todense()), axis=1)
        else:
            digitalSamplesExperimental = np.append(digitalSamplesExperimental, np.array(allSamples[actSample[0]]['geneMatrixLog2'][:,fovIdxs].todense()), axis=1)
    actTStats, actPvals = scipy.stats.ttest_ind(digitalSamplesControl,digitalSamplesExperimental, axis=1, nan_policy='omit')
    fovPValMatrix[actFov[0],:] = actPvals
    fovTStatMatrix[actFov[0],:] = actTStats

#%% check for significant genes
plt.close('all')
desiredPval = 0.05
alphaFdr = 1 - np.power((1 - desiredPval),(1/(len(sharedFOVs))))
checkForSig = np.nanmin(fovPValMatrix, axis=0)
nSigGenes = 0
for actP in enumerate(checkForSig):
    if actP[1] > 0 and actP[1] < alphaFdr:
        actGene = allSamples[0]['geneListMasked'][actP[0]]
        print(actP)
        nSigGenes += 1
        plt.figure()
        plt.imshow(allSamples[0]['tissueImageRegistered'], cmap='gray_r')
        plt.scatter(sharedFOVsCoor[:,0], sharedFOVsCoor[:,1], c=fovTStatMatrix[:,actP[0]], cmap='seismic',alpha=0.7,vmin=-4,vmax=4,plotnonfinite=False, s=10)
        plt.show()
        plt.title(f"T-statistic for {actGene}, p-value={actP[1]}")
        plt.savefig(os.path.join(derivatives,f'xeniumTStat_maleAndFemale_SDvsNSD_{actGene}_nFOVs{nFOVs}_uncorrPVal{desiredPval}.png'), bbox_inches='tight', dpi=300)
        plt.close()
