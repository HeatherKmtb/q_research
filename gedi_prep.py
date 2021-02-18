#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 12:41:46 2021

@author: heatherkay
"""
import geopandas
import os.path
import glob
from rasterstats import zonal_stats
import numpy as np

def tif_join(folderin, rasterin, folderout):
    """
    Function to join GEDI gpkg files to tif
    
    Parameters
    ----------
    folderin: string
          Filepath for folder contain GEDI files
          
    rasterin: string
          Filepath for tif file
          
    folderout: string
             Filepath for folder to contain joined files
             
     
    """
    gedifiles = glob.glob(folderin + '*.gpkg')
    raster = rasterin
    beams = ['BEAM0000','BEAM0001','BEAM0010','BEAM0011','BEAM0101','BEAM0110',
                 'BEAM1000','BEAM1011']
    
    stats = 'median'
    
    for file in gedifiles:
        name = os.path.splitext(os.path.basename(file))[0]
        for beam in beams:
            vector = geopandas.read_file(file, layer=beam)
            result = zonal_stats(vector, raster, stats=stats, geojson_out=True)
            geostats = geopandas.GeoDataFrame.from_features(result)
    
            geostats.to_file(folderout + name + ".gpkg", layer = beam, driver='GPKG')
  
def rmv_cat(folderin, folderout):
    """
    Function to remove unvegetated GEDI footprints based on CCILC, remove 
    unneccesary columns, remove footprints with 'surface flag' equal to 1,
    remove any footprints with rh100 less than 0, and add a column with the
    acquisition date extracted from the filename
    
    Parameters
    ----------
    folderin: string
          Filepath for folder contain GEDI files
                    
    folderout: string
             Filepath for folder to contain processed files
    """   
    gedifiles = glob.glob(folderin + '*.gpkg')

    beams = ['BEAM0000','BEAM0001','BEAM0010','BEAM0011','BEAM0101','BEAM0110',
                 'BEAM1000','BEAM1011']
    colNms =['elevation_bin0','elevation_lastbin','height_bin0','height_lastbin',
                 'shot_number','solar_azimuth','solar_elevation','latitude_bin0',
                 'latitude_lastbin','longitude_bin0','longitude_lastbin',
                 'master_frac','master_int','omega','pai','pgap_theta',
                 'pgap_theta_error','rg','rhog','rhog_error','rhov','rhov_error',
                 'rossg','rv']
    cat=['0.0', '190.0','200.0','202.0', '210.0', '220.0']
    column='median'
    for file in gedifiles:
        name = os.path.splitext(os.path.basename(file))[0]
        name_comp = name.split('_')
        date = name_comp[2]
        
        for beam in beams:
            df = geopandas.read_file(file, layer=beam)
            new = df[np.logical_not(df[column].isin(cat))]
            if new.empty:
                continue
            df2 = new[new['surface_flag']==1] 
            df3 = df2[df2['rh100']>=0]
            df3.drop(colNms, 1, inplace = True)
            df3['date']= date
            df3.to_file(folderout + name + '.gpkg', layer = beam, driver='GPKG')
            
def shp_join(folderin, shapein, folderout):
    """
    Function to join GEDI gpkg files to shp (here wwf ecoregions)
    
    Parameters
    ----------
    folderin: string
          Filepath for folder contain GEDI files
          
    wwfin: string
          Filepath for shape file
          
    folderout: string
             Filepath for folder to contain joined files
             
     
    """    
    gedifiles = glob.glob(folderin + '*.gpkg')

    beams = ['BEAM0000','BEAM0001','BEAM0010','BEAM0011','BEAM0101','BEAM0110',
                 'BEAM1000','BEAM1011']
    colNms = ['AREA', 'ECO_NUM', 'ECO_SYM','G200_BIOME', 'G200_NUM', 
                  'G200_REGIO', 'G200_STAT', 'GBL_STAT', 'OBJECTID', 'PERIMETER', 
                  'PER_area', 'PER_area_1', 'PER_area_2','REALM', 'Shape_Area', 
                  'Shape_Leng', 'area_km2']
    wwf_layer = geopandas.read_file(shapein)
    
    for file in gedifiles:
        name = os.path.splitext(os.path.basename(file))[0]
        for beam in beams:
            vector = geopandas.read_file(file, layer=beam)
            result = geopandas.sjoin(vector, wwf_layer, how="inner", op="within")
            geostats = geopandas.GeoDataFrame.from_features(result)
            geostats.drop(colNms, 1, inplace = True)
            geostats.to_file(folderout + name + ".gpkg", layer = beam, driver='GPKG')

def split_per_eco(folderin, folderout):
    """
    Function to split GEDI files per ecoregion
    
    Parameters
    ----------
    folderin: string
          Filepath for folder contain GEDI files
                    
    folderout: string
             Filepath for folder to contain processed files
    """ 
    gedifiles = glob.glob(folderin + '*.gpkg')

    beams = ['BEAM0000','BEAM0001','BEAM0010','BEAM0011','BEAM0101','BEAM0110',
                 'BEAM1000','BEAM1011']
    split_col='ECO_ID'
    
    for file in gedifiles:
        name = os.path.splitext(os.path.basename(file))[0]
        for beam in beams:
            dfa = geopandas.read_file(file, layer=beam)
            df = dfa.astype({split_col: 'int32'})
            ecoNames = list(np.unique(df[split_col]))#get list of unique ecoregions    
        
            for eco in ecoNames:
                #create new df 
                df2 = geopandas.GeoDataFrame(df)
                ID = str(eco)
                df_eco = df2.loc[df2[split_col]==eco]
                df_eco.to_file(folderout + name + "_" + ID + ".gpkg", layer = beam, driver='GPKG')    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    