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

# setting runtime parameters for pyplot plotting  
# these are settings about font and plotting information  
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['text.usetex'] = False
plt.rcParams['svg.fonttype'] = 'none'

# settings about where things are or will be stored 
derivatives = os.path.join('/','home','zjpeters','Documents','sleepDepXeniumHippocampus','derivatives')
figureFolder = os.path.join('/','home','zjpeters','Documents','sleepDepXeniumHippocampus','writing','figures')

#%% create dictionary containing cluster and color info
# a list of the cluster names that will be used
clusterNames = ['CA1', 'CA2', 'CA3', 'DG', 'DG hilus', 'astrocytes', 'endothelial', 'microglia', 'interneurons', 'oligodendrocytes']
# color assignment for the main colors, i.e. bar plots
# these colors are sorted in the same order as the clusters, otherwise they would be incorrectly assigned
foregroundColors = np.array([np.array([0, 187, 173, 255])/255, 
                            np.array([184, 0, 88, 255])/255, 
                            np.array([0, 140, 249, 255])/255,
                            np.array([235, 172, 35, 255])/255,
                            np.array([0, 110, 0, 255])/255,
                            np.array([209, 99, 230, 255])/255,
                            np.array([178, 69, 2, 255])/255,
                            np.array([255, 146, 135, 255])/255,
                            np.array([89, 84, 214, 255])/255,
                            np.array([0, 198, 248, 255])/255])
# color assignment for the opaque, used as the background for the rows
backgroundColors = np.array([np.array([0, 187, 173, 40])/255,
                             np.array([184, 0, 88, 40])/255,
                             np.array([0, 140, 249, 40])/255,
                             np.array([235, 172, 35, 40])/255,
                             np.array([0, 110, 0, 40])/255,
                             np.array([209, 99, 230, 40])/255,
                             np.array([178, 69, 2, 40])/255,
                             np.array([255, 146, 135, 40])/255,
                             np.array([89, 84, 214, 40])/255,
                             np.array([0, 198, 248, 40])/255])

clusterInfo = {'clusterName':[], 'foreColor':[], 'backColor':[]}
for i in range(len(clusterNames)):
    clusterInfo['clusterName'].append(clusterNames[i])
    clusterInfo['foreColor'].append(foregroundColors[i,:])
    clusterInfo['backColor'].append(backgroundColors[i,:])
#%% create dictionary containing all of the DEG information per region/cell type
# load the information about the sheet names from the excel file
sheetNames = pd.ExcelFile(os.path.join(derivatives, 'degs_per_cell-type_and_regions_male_SD_BH.xlsx'))
# load the contents of each sheet in the excel file and assign to a key in the deg dictionary
degDict = {}
for cellType in sheetNames.sheet_names:
    degDict[cellType] = pd.read_excel(os.path.join(derivatives, 'degs_per_cell-type_and_regions_male_SD_BH.xlsx'), sheet_name=cellType)
# remove unlabeled cells from data to be plotted
del degDict['sparse']
#%% look for overlapping genes
# first create a list of all DEGs across all clusters
allDegs = np.empty(0)
for cellType in degDict.keys():
    allDegs = np.append(allDegs, np.array(degDict[cellType]['Gene_ID']))
# reduce the list to only the unique DEGs, removing any repeated genes
allDegs = np.unique(allDegs)

# create a boolean array of which DEGs are present in which clusters
# start with an array that is entirely "False"
overlappingBool = np.full([len(allDegs), len(degDict)], False)
for cellType in enumerate(degDict):
    for deg in enumerate(allDegs):
        # if a DEG is present in a cluster, make that cell "True"
        if deg[1] in np.array(degDict[cellType[1]]['Gene_ID']):
            overlappingBool[deg[0], cellType[0]] = True
# convert the boolean array into a Pandas DataFrame
# each row is labeled with the gene name (index=allDegs) and each column labeled with the cluster names (columns=clusterNames)
overlappingDF = pd.DataFrame(overlappingBool, index=allDegs, columns=clusterNames)
overlappingDF = overlappingDF.set_index(clusterNames)

#%% create dataframe of overlapping degs 
plt.close('all')

upset = upsetplot.UpSet(overlappingDF, show_counts='%d')

# set the colors of the bars for the regional association
# since we created a dictionary with this information, we can use a for loop
for i in range(len(clusterInfo['clusterName'])):
    # first option labels the bar graph, bar_facecolor assigns bar graph color, shading_facecolor assigns background color
    upset.style_categories(clusterInfo['clusterName'][i], 
                           bar_facecolor=clusterInfo['foreColor'][i], 
                           shading_facecolor=clusterInfo['backColor'][i])
### use dark grey for fill to replace with correct color scheme in editing tool
upset.style_subsets(min_subset_size=1, facecolor=np.squeeze(np.array([0.1,0.1,0.1,1])), edgecolor='black')
# change the label above the bar graphs
fig = upset.plot()
fig['totals'].set_title('Number of DEGs', fontweight='bold')
fig['intersections'].set_ylabel('Intersecting DEGs', fontweight='bold')
plt.show()

# output pdf and svg, however since we're editing the colors we don't need to output pdf
# plt.savefig(os.path.join(figureFolder, 'upset_plot_of_DEGS_color_underlay_black_lines.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(figureFolder, 'upset_plot_for_editing.svg'), bbox_inches='tight', dpi=300)

#%% the for loop is doing the exact same as if we ran the following
plt.close('all')

# color assignment for the main colors, i.e. bar plots
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
# color assignment for the opaque, used as the background for the rows
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

upset = upsetplot.UpSet(overlappingDF, show_counts='%d')

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

fig = upset.plot()
fig['totals'].set_title('Number of DEGs', fontweight='bold')
plt.show()
