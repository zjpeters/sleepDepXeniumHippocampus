#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 06:45:27 2026

@author: zjpeters
"""
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import upsetplot
import sys
if os.path.exists(os.path.join("/", "home", "zjpeters", "Documents", "stanly", "code")):
    sys.path.insert(0, "/home/zjpeters/Documents/stanly/code")
else:
    sys.path.insert(0, os.path.join('C:',os.sep, 'Users','onyh19ug', 'Documents', 'STANLY','code'))
import stanly
# setting runtime parameters for pyplot plotting    
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['text.usetex'] = False
plt.rcParams['svg.fonttype'] = 'none'


derivatives = os.path.join('/','home','zjpeters','Documents','sleepDepXeniumHippocampus','derivatives')
figureFolder = os.path.join('/','home','zjpeters','Documents','sleepDepXeniumHippocampus','writing','figures')
columnNames = ['CA1', 'CA2', 'CA3', 'DG', 'DG hilus', 'astrocytes', 'endothelial', 'microglia', 'interneurons', 'oligodendrocytes']
#%% create dictionary containing all of the DEG information per region/cell type
sheetNames = pd.ExcelFile(os.path.join(derivatives, 'degs_per_cell-type_and_regions_male_SD_BH.xlsx'))
degDict = {}
for cellType in sheetNames.sheet_names:
    degDict[cellType] = pd.read_excel(os.path.join(derivatives, 'degs_per_cell-type_and_regions_male_SD_BH.xlsx'), sheet_name=cellType)
# remove unlabeled cells from data
del degDict['sparse']
#%% look for overlapping genes

allDegs = np.empty(0)
for cellType in degDict.keys():
    allDegs = np.append(allDegs, np.array(degDict[cellType]['Gene_ID']))
    
allDegs = np.unique(allDegs)

overlappingBool = np.full([len(allDegs), len(degDict)], False)
for cellType in enumerate(degDict):
    for deg in enumerate(allDegs):
        if deg[1] in np.array(degDict[cellType[1]]['Gene_ID']):
            overlappingBool[deg[0], cellType[0]] = True

#%% count number of shared overlapps (i.e. all genes shared between CA1 and CA3)
uniqueSharePatterns = np.unique(overlappingBool, axis=0)
nOfRegionOverlaps = []
degsPerOverlap = {}
for i in range(uniqueSharePatterns.shape[0]):
    degsPerOverlap[i] = {'gene_list': [], 'region_list': []}

nOfGenesPerOverlap = []
for n, actPattern in enumerate(uniqueSharePatterns):
    nOfShares = len(np.where((overlappingBool == actPattern).all(axis=1))[0])
    nOfRegionOverlaps.append(nOfShares)
    degsPerOverlap[n]['gene_list'] = allDegs[np.where((overlappingBool == actPattern).all(axis=1))[0]]
    degsPerOverlap[n]['region_list'] = np.array(columnNames)[actPattern]
    nOfGenesPerOverlap.append(len(degsPerOverlap[n]['gene_list']))

#%% color assignment

ca1Color = np.squeeze(np.array([0, 187, 173, 255]))/255
ca2Color = np.squeeze(np.array([184, 0, 88, 255]))/255
ca3Color = np.squeeze(np.array([0, 140, 249, 255]))/255
dgColor = np.squeeze(np.array([235, 172, 35, 255]))/255
ca4dgColor = np.squeeze(np.array([0, 110, 0, 255]))/255
astColor = np.squeeze(np.array([209, 99, 230, 255]))/255
endColor = np.squeeze(np.array([178, 69, 2, 255]))/255
micColor = np.squeeze(np.array([255, 146, 135, 255]))/255
neuColor = np.squeeze(np.array([89, 84, 214, 255]))/255
oliColor = np.squeeze(np.array([0, 198, 248, 255]))/255

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

cellsOfInterestColorList = np.array([ca1Color, ca2Color, ca3Color, dgColor, ca4dgColor, astColor, endColor, micColor, neuColor, oliColor])
#%% create dataframe of overlapping degs
plt.close('all')

overlappingDF = pd.DataFrame(overlappingBool, index=allDegs, columns=columnNames)
overlappingDF = overlappingDF.set_index(columnNames)

upset = upsetplot.UpSet(overlappingDF, show_counts='%d')
# , absent=['CA1', 'CA2', 'CA3', 'DG hilus', 'astrocytes', 'endothelial', 'microglia', 'interneurons', 'oligodendrocytes']
# upset.style_subsets(present="DG", facecolor="red")

# set the colors of the bars for the regional association
upset.style_categories('CA1', bar_facecolor=ca1Color, shading_facecolor=ca1ColorOp)
upset.style_categories('CA2', bar_facecolor=ca2Color, shading_facecolor=ca2ColorOp)
upset.style_categories('CA3', bar_facecolor=ca3Color, shading_facecolor=ca3ColorOp)
upset.style_categories('DG', bar_facecolor=dgColor, shading_facecolor=dgColorOp)
upset.style_categories('DG hilus', bar_facecolor=ca4dgColor, shading_facecolor=ca4dgColorOp)
upset.style_categories('astrocytes', bar_facecolor=astColor, shading_facecolor=astColorOp)
upset.style_categories('endothelial', bar_facecolor=endColor, shading_facecolor=endColorOp)
upset.style_categories('microglia', bar_facecolor=micColor, shading_facecolor=micColorOp)
upset.style_categories('interneurons', bar_facecolor=neuColor, shading_facecolor=neuColorOp)
upset.style_categories('oligodendrocytes', bar_facecolor=oliColor, shading_facecolor=oliColorOp)
# upset.style_subsets(present="CA1", facecolor="lightblue", present='CA2', facecolor='red')

fig = upset.plot()
fig['totals'].set_title('Number of DEGs', fontweight='bold')
plt.show()

# output pdf and svg
# plt.savefig(os.path.join(figureFolder, 'upset_plot_of_DEGS_color_underlay.pdf'), bbox_inches='tight', dpi=300)
# plt.savefig(os.path.join(figureFolder, 'upset_plot_of_DEGS_color_underlay.svg'), bbox_inches='tight', dpi=300)
# plt.close('all')
#%% create dataframe of overlapping degs
plt.close('all')

# overlappingDF = pd.DataFrame(overlappingBool, index=allDegs, columns=columnNames)
# overlappingDF = overlappingDF.set_index(columnNames)

upset = upsetplot.UpSet(overlappingDF, show_counts='%d')
# , absent=['CA1', 'CA2', 'CA3', 'DG hilus', 'astrocytes', 'endothelial', 'microglia', 'interneurons', 'oligodendrocytes']
# upset.style_subsets(present="DG", facecolor="red")

# set the colors of the bars for the regional association
upset.style_categories('CA1', bar_facecolor=ca1Color)
upset.style_categories('CA2', bar_facecolor=ca2Color)
upset.style_categories('CA3', bar_facecolor=ca3Color)
upset.style_categories('DG', bar_facecolor=dgColor)
upset.style_categories('DG hilus', bar_facecolor=ca4dgColor)
upset.style_categories('astrocytes', bar_facecolor=astColor)
upset.style_categories('endothelial', bar_facecolor=endColor)
upset.style_categories('microglia', bar_facecolor=micColor)
upset.style_categories('interneurons', bar_facecolor=neuColor)
upset.style_categories('oligodendrocytes', bar_facecolor=oliColor)

# upset.style_subsets(present=columnNames, facecolor='blue')
# upset.style_subsets(present="CA1", facecolor="lightblue", present='CA2', facecolor='red')

fig = upset.plot()
fig['totals'].set_title('Number of DEGs', fontweight='bold')

plt.show()
# output pdf and svg
# plt.savefig(os.path.join(figureFolder, 'upset_plot_of_DEGS.pdf'), bbox_inches='tight', dpi=300)
# plt.savefig(os.path.join(figureFolder, 'upset_plot_of_DEGS.svg'), bbox_inches='tight', dpi=300)
# plt.close('all')

#%% create dataframe of overlapping degs 
plt.close('all')

# overlappingDF = pd.DataFrame(overlappingBool, index=allDegs, columns=columnNames)
# overlappingDF = overlappingDF.set_index(columnNames)

upset = upsetplot.UpSet(overlappingDF, show_counts='%d')
# , absent=['CA1', 'CA2', 'CA3', 'DG hilus', 'astrocytes', 'endothelial', 'microglia', 'interneurons', 'oligodendrocytes']
# upset.style_subsets(present="DG", facecolor="red")

# set the colors of the bars for the regional association
upset.style_categories('CA1', bar_facecolor=ca1Color, shading_facecolor=ca1ColorOp)
upset.style_categories('CA2', bar_facecolor=ca2Color, shading_facecolor=ca2ColorOp)
upset.style_categories('CA3', bar_facecolor=ca3Color, shading_facecolor=ca3ColorOp)
upset.style_categories('DG', bar_facecolor=dgColor, shading_facecolor=dgColorOp)
upset.style_categories('DG hilus', bar_facecolor=ca4dgColor, shading_facecolor=ca4dgColorOp)
upset.style_categories('astrocytes', bar_facecolor=astColor, shading_facecolor=astColorOp)
upset.style_categories('endothelial', bar_facecolor=endColor, shading_facecolor=endColorOp)
upset.style_categories('microglia', bar_facecolor=micColor, shading_facecolor=micColorOp)
upset.style_categories('interneurons', bar_facecolor=neuColor, shading_facecolor=neuColorOp)
upset.style_categories('oligodendrocytes', bar_facecolor=oliColor, shading_facecolor=oliColorOp)

### use dark grey for fill to replace with correct color scheme in editing tool
upset.style_subsets(min_subset_size=1, facecolor=np.squeeze(np.array([0.1,0.1,0.1,1])), edgecolor='black')
# upset.style_subsets(present="CA1", facecolor="lightblue", present='CA2', facecolor='red')

fig = upset.plot()
fig['totals'].set_title('Number of DEGs', fontweight='bold')
plt.show()

# output pdf and svg
# plt.savefig(os.path.join(figureFolder, 'upset_plot_of_DEGS_color_underlay_black_lines.pdf'), bbox_inches='tight', dpi=300)
# plt.savefig(os.path.join(figureFolder, 'upset_plot_of_DEGS_color_underlay_black_lines.svg'), bbox_inches='tight', dpi=300)
# plt.close('all')

#%% create list of genes unique to regions
regionsPerGene = np.sum(overlappingBool, axis=1)
uniqueDegMask = np.where(regionsPerGene == 1)[0]
uniqueDegs = {}
for cellType in enumerate(degDict):
    uniqueDegs[cellType[1]] = []

for cellType in enumerate(degDict):
    for deg in enumerate(allDegs[uniqueDegMask]):   
        if deg[1] in np.array(degDict[cellType[1]]['Gene_ID']):
            uniqueDegs[cellType[1]].append(deg[1])
            
