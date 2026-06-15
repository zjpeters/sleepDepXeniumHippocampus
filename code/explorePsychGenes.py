#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:48:31 2025

@author: zjpeters
"""
import os
import numpy as np
import sys
if os.path.exists(os.path.join("/", "home", "zjpeters", "Documents", "stanly", "code")):
    sys.path.insert(0, "/home/zjpeters/Documents/stanly/code")
else:
    sys.path.insert(0, os.path.join('C:',os.sep, 'Users','onyh19ug', 'Documents', 'STANLY','code'))
import stanly
import pandas as pd

derivatives = os.path.join('/','media','zjpeters','Expansion','sleepDepXeniumHippocampus','derivatives')
figureFolder = os.path.join('/','media','zjpeters','Expansion','sleepDepXeniumHippocampus','writing','figures')
geneListLocation = os.path.join('/','media','zjpeters','Expansion','sleepDepXeniumHippocampus','geneLists')

#%% load processed samples and check selection

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

#%% import deg lists       
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

#%% create dictionary containing only brain diseases of interest
"""
identify genes present in xenium data that are also present in:
    "A comparison of anatomic and cellular transcriptome structures across 40 human brain diseases"
    https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002058#pbio.3002058.s002
"""

zeighamiGeneList = pd.read_excel(os.path.join(geneListLocation,"journal.pbio.3002058.s002.xlsx"), sheet_name='Associated genes')

disOfInterest = ['Autistic Disorder', 'Bipolar Disorder', 'Depressive Disorder', 'Schizophrenia', 'Sleep disorders']
zeighamiGenes = {}
for disease in disOfInterest:
    zeighamiGenes[disease] = zeighamiGeneList[disease].dropna()
    
zeighamiGenesCasefold = {}
for disease in zeighamiGenes.keys():
    tempGeneList = []
    for gene in np.array(zeighamiGenes[disease]):
        tempGeneList.append(gene.casefold())
    zeighamiGenesCasefold[disease] = tempGeneList

# look for which DEGs are potentially present in disease list
zeighamiDegDict = {}
for disease in zeighamiGenesCasefold.keys():
    zeighamiDegDict[disease] = []
    for deg in allDegs:
        if deg.casefold() in np.array(zeighamiGenesCasefold[disease]):
            zeighamiDegDict[disease].append(deg)
            
#%% import and check SFARI gene list
sfariGeneList = pd.read_csv(os.path.join(geneListLocation, 'SFARI-Gene_genes_05-01-2026release_06-13-2026export.csv'))

sfariGenes = sfariGeneList['gene-symbol']
    
sfariGenesCasefold = []
for gene in sfariGenes:
    sfariGenesCasefold.append(gene.casefold())

# look for which DEGs are potentially present in disease list
sfariDegList = []

for deg in allDegs:
    if deg.casefold() in np.array(sfariGenesCasefold):
        sfariDegList.append(deg)

#%% import and check schizophrenia gene list
schizGeneList = pd.read_csv(os.path.join(geneListLocation, 'INT-17_SCZ_High_Confidence_Gene_List.csv'))

schizGenes = schizGeneList['sczgenenames'].dropna()
    
schizGenesCasefold = []
for gene in schizGenes:
    schizGenesCasefold.append(gene.casefold())

# look for which DEGs are potentially present in disease list
schizDegList = []

for deg in allDegs:
    if deg.casefold() in np.array(schizGenesCasefold):
        schizDegList.append(deg)

#%% import and check bipolar gene list
bpGeneList = pd.read_excel(os.path.join(geneListLocation, 'NIHMS1687813-supplement-Supplementary_Tables.xlsx'), sheet_name='Table S4', skiprows=1)

bpGenes = bpGeneList['Gene '].dropna()
    
bpCasefold = []
for gene in bpGenes:
    bpCasefold.append(gene.casefold())

# look for which DEGs are potentially present in disease list
bpDegList = []

for deg in allDegs:
    if deg.casefold() in np.array(bpCasefold):
        bpDegList.append(deg)

#%% import and check MDD gene list
mddGeneList = pd.read_excel(os.path.join(geneListLocation, '41588_2026_2638_MOESM4_ESM.xlsx'), sheet_name='Supplementary Table 1')

mddGenes = mddGeneList['Risk gene'].dropna()
    
mddCasefold = []
for gene in mddGenes:
    mddCasefold.append(gene.casefold())

# look for which DEGs are potentially present in disease list
mddDegList = []

for deg in allDegs:
    if deg.casefold() in np.array(mddCasefold):
        mddDegList.append(deg)

#%% write results to excel file

writer = pd.ExcelWriter(os.path.join(derivatives, 'Xenium_SD_DEGs_in_psych_risk_gene_lists.xlsx'))

degListDF = pd.DataFrame(allDegs, columns=['gene-name'])
degListDF.to_excel(writer, sheet_name='Xenium SD unique DEG list', index=False)
for disease in zeighamiDegDict.keys():
    if len(zeighamiDegDict[disease]) > 0:
        degListDF = pd.DataFrame(zeighamiDegDict[disease], columns=['gene-name'])
        degListDF.to_excel(writer, sheet_name=f'Zeighami {disease}', index=False)

degListDF = pd.DataFrame(sfariDegList, columns=['gene-name'])
degListDF.to_excel(writer, sheet_name='SFARI', index=False)

degListDF = pd.DataFrame(schizDegList, columns=['gene-name'])
degListDF.to_excel(writer, sheet_name='Schizophrenia', index=False)

degListDF = pd.DataFrame(bpDegList, columns=['gene-name'])
degListDF.to_excel(writer, sheet_name='Bipolar disorder', index=False)

degListDF = pd.DataFrame(mddDegList, columns=['gene-name'])
degListDF.to_excel(writer, sheet_name='Major depressive disorder', index=False)
writer.close()