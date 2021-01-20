#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 15:17:15 2020

@author: heatherkay
"""

import rsgislib.vectorutils
import glob
import os.path
from multiprocessing import Pool
import geopandas as gpd
import numpy as np
import pandas as pd
import rsgislib.zonalstats

def shp_join(filein, folderout, folderno):
    """
    Function to join a shapefile to the gla14 data. Must be run for folderno from 1-10
    
    Parameters
    ----------
    filein: string
          Filepath for shp file to join with gla14 data
          
    folderout: string
             Filepath for folder to contain joined files
             
    folderno: string
            Number - needs to be changed from 1 to 10 and run due to file quantity and size         
    """
    def performSpatialJoin(base_vec, base_lyr, join_vec, join_lyr, output_vec, output_lyr):
        import geopandas
        # Must have rtree installed - otherwise error "geopandas/tools/sjoin.py"
        # AttributeError: 'NoneType' object has no attribute 'intersection'
        base_gpd_df = geopandas.read_file(base_vec)
        join_gpg_df = geopandas.read_file(join_vec)
    
        join_gpg_df = geopandas.sjoin(base_gpd_df, join_gpg_df, how="inner", op="within")
        join_gpg_df.to_file(output_vec)

    def run_join(params):
        base_vec = params[0]
        join_vec = params[1]
        output_vec = params[2]
        performSpatialJoin(base_vec, '', join_vec, '', output_vec, '')
    
    split_files = glob.glob('./gla14/split_files/folder_{}/*.shp'.format(folderno))


    params = []
    for filename in split_files:
        basename = os.path.splitext(os.path.basename(filename))[0]
        output_file = os.path.join(folderout, "{}_join.shp".format(basename))
        params.append([filename, filein, output_file])


    ncores = 50
    p = Pool(ncores)
    p.map(run_join, params)

    #joined_files = glob.glob('./intersect_koppen_split/*.shp')
    #rsgislib.vectorutils.mergeShapefiles(joined_files, './gla14/gla14_koppen.shp')

def split_per(folderin, folderout, split_col='ECO_ID', colNms=['i_h100','i_cd',
    'doy','i_wflen','i_acqdate','b1','vcf','ECO_NAME','ECO_ID','BIOME','geometry']):
    """
    Function which will divide shapefiles by individual elements in one column 
    to generate new shapefiles with filename referring to element in column 
    (e.g split data by ecoregion and give each new file ecoregion number)
    
    Parameters
    ----------
    folderin: string
          filepath for folder containing shapefiles
          
    folderout: string
             filepath for folder where new files will be saved
             
    split_col: string
             name of column in files to use for split
             
    colNms: list of strings
          names of columns to be retained in output shapefile.
          Default = ['i_h100','i_cd','doy','i_wflen','i_acqdate','b1','vcf','ECO_NAME','ECO_ID','BIOME','geometry']                
    """

    split_files = glob.glob(folderin + '*.shp')

    for filename in split_files:
        print(filename)
        basename = os.path.splitext(os.path.basename(filename))[0]
        dfa = gpd.read_file(filename)
        df = dfa.astype({split_col: 'int32'}) 
        ecoNames = list(np.unique(df[split_col]))#get list of unique ecoregions    
        
        for eco in ecoNames:
            #create new df with just columns I want
            df2 = gpd.GeoDataFrame(df, columns=colNms)
            ID = str(eco)
            df_eco = df.loc[df2[split_col]==eco, colNms]
            df_eco.to_file(folderout + '/{}_eco_{}.shp'.format(basename, ID))    

def join_per(folderin, folderout, IDfile='./eco/final_ID.csv', column='ECO_ID', naming='*_eco_{}.shp'):
    """
    Function to regroup files that have been split with spilt_per function on elements of split
    
    Parameters
    ----------
    folderin: string
            filepath for folder containing shapefiles to be joined
            
    folderout: string
             filepath for folder where output shapefiles will be saved 
             
    IDfile: string
          filepath for csv with column containing list of elements for the join.
          Default = './eco/final_ID.csv'
          
    column: string
          column name from IDfile containing elements for the join.   
          Default = 'ECO_ID'
          
    naming: string
          filename with {} to select part of filename which matches naming of element of join
          Default = '*_eco_{}.shp'
    """
    #import csv with IDs to obtain list for merge
    df = pd.read_csv(IDfile)
    ecoNms = list(np.unique(df[column]))#get list of unique ecoregions     

    for ecoNm in ecoNms:
        fileList = glob.glob(folderin + naming.format(ecoNm))#here also need dict ref
        rsgislib.vectorutils.mergeShapefiles(fileList, folderout + 'gla14_eco_{}.shp'.format(ecoNm))#use dict to get ecoNm, create new folder too?
 
    #mkdir is make new folder
    
def lc_join(folderin, filein, folderout):
    """
    Function to join glas files with ccilc tif

    Parameters
    ----------
    folderin : string
        filepath for directory with gedi shp files
        
    filein : string
        filepath for CCI LC tif file
        
    folderout : string
        filepath for output files directory

    """
    split_files = glob.glob(folderin + '*.shp')

    for filename in split_files:
        basename = os.path.splitext(os.path.basename(filename))[0]
        inputimage = filein
        inputvector = filename
        outputvector = os.path.join(folderout, "{}_join.shp".format(basename))
        removeExistingVector = True
        useBandNames = True
        print(inputimage)
        print(inputvector)
        print(outputvector)
        rsgislib.zonalstats.pointValue2SHP(inputimage, inputvector, outputvector, removeExistingVector, useBandNames)
 
def rmv_cat(folderin, folderout, column='b1', cat=['0.0', '190.0','200.0','202.0', '210.0', '220.0']):
    """
    Function to remove categories e.g. land cover classifications, vcf categories, ecoregions
    
    Parameters
    ----------
    folderin: string
            filepath for folder containing shapefiles to be processed
    
    folderout: string
             filepath for folder where output shapefiles will be saved
             
    column: string
          column from shapefile with categories for removal.
          Default = 'b1'
          
    cat: list of strings
       names of categories to be dropped      
    """
    fileList = glob.glob(folderin + '*.shp')

    for filename in fileList:
        basename = os.path.splitext(os.path.basename(filename))[0]
        df = gpd.read_file(filename) 
        new = df[np.logical_not(df[column].isin(cat))]
        if new.empty:
            continue
        new.to_file(folderout + '{}.shp'.format(basename))
        
        
        