#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 11:56:56 2025

@author: zjpeters
"""
import os
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, "/home/zjpeters/Documents/stanly/code")
import stanly

rawdata=os.path.join('/','media','zjpeters','Expansion','sleepDepXenium','rawdata')
derivatives=os.path.join('/','media','zjpeters','Expansion','sleepDepXeniumHippocampus','derivatives')

#%% import samples
# sampleXenium = stanly.importXeniumData(os.path.join(rawdata,'YW-1_ROI_A1'))
experiment = stanly.loadParticipantsTsv(os.path.join(rawdata, 'participants.tsv'))
processedSamples = {}
for actSample in range(len(experiment['sample-id'])):
    sample = stanly.importXeniumData(os.path.join(rawdata, experiment['sample-id'][actSample]))
    sampleProcessed = stanly.processXeniumData(sample, experiment['rotation'][actSample], derivatives)
    processedSamples[actSample] = sampleProcessed

#%% use lasso tool to select only hippocampal region
sampleIdx = 0
lasso = stanly.SelectUsingLasso(processedSamples[sampleIdx], f"{processedSamples[sampleIdx]['sampleID']}_hippocampus", derivatives)
lasso.applyLasso()
hippSample = lasso.outputMaskedSample(processedSamples[sampleIdx])

sampleIdx = 1
lasso = stanly.SelectUsingLasso(processedSamples[sampleIdx], f"{processedSamples[sampleIdx]['sampleID']}_hippocampus", derivatives)
lasso.applyLasso()
hippSample = lasso.outputMaskedSample(processedSamples[sampleIdx])

sampleIdx = 2
lasso = stanly.SelectUsingLasso(processedSamples[sampleIdx], f"{processedSamples[sampleIdx]['sampleID']}_hippocampus", derivatives)
lasso.applyLasso()
hippSample = lasso.outputMaskedSample(processedSamples[sampleIdx])

sampleIdx = 3
lasso = stanly.SelectUsingLasso(processedSamples[sampleIdx], f"{processedSamples[sampleIdx]['sampleID']}_hippocampus", derivatives)
lasso.applyLasso()
hippSample = lasso.outputMaskedSample(processedSamples[sampleIdx])

sampleIdx = 4
lasso = stanly.SelectUsingLasso(processedSamples[sampleIdx], f"{processedSamples[sampleIdx]['sampleID']}_hippocampus", derivatives)
lasso.applyLasso()
hippSample = lasso.outputMaskedSample(processedSamples[sampleIdx])

sampleIdx = 5
lasso = stanly.SelectUsingLasso(processedSamples[sampleIdx], f"{processedSamples[sampleIdx]['sampleID']}_hippocampus", derivatives)
lasso.applyLasso()
hippSample = lasso.outputMaskedSample(processedSamples[sampleIdx])

sampleIdx = 6
lasso = stanly.SelectUsingLasso(processedSamples[sampleIdx], f"{processedSamples[sampleIdx]['sampleID']}_hippocampus", derivatives)
lasso.applyLasso()
hippSample = lasso.outputMaskedSample(processedSamples[sampleIdx])

sampleIdx = 7
lasso = stanly.SelectUsingLasso(processedSamples[sampleIdx], f"{processedSamples[sampleIdx]['sampleID']}_hippocampus", derivatives)
lasso.applyLasso()
hippSample = lasso.outputMaskedSample(processedSamples[sampleIdx])

sampleIdx = 8
lasso = stanly.SelectUsingLasso(processedSamples[sampleIdx], f"{processedSamples[sampleIdx]['sampleID']}_hippocampus", derivatives)
lasso.applyLasso()
hippSample = lasso.outputMaskedSample(processedSamples[sampleIdx])

sampleIdx = 9
lasso = stanly.SelectUsingLasso(processedSamples[sampleIdx], f"{processedSamples[sampleIdx]['sampleID']}_hippocampus", derivatives)
lasso.applyLasso()
hippSample = lasso.outputMaskedSample(processedSamples[sampleIdx])

sampleIdx = 10
lasso = stanly.SelectUsingLasso(processedSamples[sampleIdx], f"{processedSamples[sampleIdx]['sampleID']}_hippocampus", derivatives)
lasso.applyLasso()
hippSample = lasso.outputMaskedSample(processedSamples[sampleIdx])

sampleIdx = 11
lasso = stanly.SelectUsingLasso(processedSamples[sampleIdx], f"{processedSamples[sampleIdx]['sampleID']}_hippocampus", derivatives)
lasso.applyLasso()
hippSample = lasso.outputMaskedSample(processedSamples[sampleIdx])

#%% load newly processed samples and check selection
hippSamples = {}
for sampleIdx in range(len(processedSamples)):
    hippSamples[sampleIdx] = stanly.loadProcessedXeniumSample(os.path.join(derivatives, f"{processedSamples[sampleIdx]['sampleID']}_hippocampus"))
    plt.figure()
    plt.imshow(hippSamples[sampleIdx]['tissueImageProcessed'], cmap='gray')
    plt.show()