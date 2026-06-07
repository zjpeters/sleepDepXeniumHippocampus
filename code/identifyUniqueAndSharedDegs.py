#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 06:45:27 2026

@author: zjpeters
"""
import os
import numpy as np
import pandas as pd
import upsetplot

derivatives = os.path.join('/','home','zjpeters','Documents','sleepDepXeniumHippocampus','derivatives')
figureFolder = os.path.join('/','home','zjpeters','Documents','sleepDepXeniumHippocampus','writing','figures')


sheetNames = pd.ExcelFile(os.path.join(derivatives, 'degs_per_cell-type_and_regions_male_SD_BH.xlsx'))
degDict = {}
for cellType in sheetNames.sheet_names:
    degDict[cellType] = pd.read_excel(os.path.join(derivatives, 'degs_per_cell-type_and_regions_male_SD_BH.xlsx'), sheet_name=cellType)

