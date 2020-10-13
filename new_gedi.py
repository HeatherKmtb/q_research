#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 15:03:19 2020

@author: heatherkay
"""

import pandas as pd
import numpy as np
from pandas import DataFrame
import glob
import os.path
import geopandas as gpd
import os, ogr, osr
import pickle
import rsgislib.zonalstats

def lc_join(folderin, filein, folderout):
    """
    Function to join gedi files with ccilc tif

    Parameters
    ----------
    folderin : string
        filepath for directory with gedi shp files
        
    filein : string
        filepath for CCI LC tif file
        
    folderout : string
        filepath for output files directory

    """
    split_files = glob.glob(folderin + '{}.gpkg')

    layers = {'BEAM0000','BEAM0001'}
    for filename in split_files:
        basename = os.path.splitext(os.path.basename(filename))[0]
        inputimage = filein
        inputgpkg = filename
        for l in layers:
            inputvector = (inputgpkg + '|layername=' + l)
            outputvector = os.path.join(folderout, "{}_{}_join.shp".format(basename, l))
            removeExistingVector = True
            useBandNames = True
            print(inputimage)
            print(inputvector)
            print(outputvector)
            rsgislib.zonalstats.pointValue2SHP(inputimage, inputvector, outputvector, removeExistingVector, useBandNames)
