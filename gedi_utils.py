#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 12:28:32 2020

@author: ciaran
"""
import pandas as pd
import numpy as np
from pandas import DataFrame
import glob
import os.path
import geopandas as gpd
import os, ogr, osr, gdal
import pickle
from tqdm import tqdm
gdal.UseExceptions()
ogr.UseExceptions()





def lc_eliminate(inShp, inRas, band=1, nodata_value=0):
    
    """ 
    Delete a feature if it has an undesirable landcover class underneath
    
    Parameters
    ----------
    
    inShp: string
                  input shapefile
        
    inRas: string
                  input raster

    band: int
           an integer val eg - 2
                            
    nodata_value: numerical
                   If used the no data val of the raster
        
    """    
    
    # possible to delete features with 
    # layer.DeleteFeature(feat.GetFID())
    

    rds = gdal.Open(inRas, gdal.GA_ReadOnly)
    rb = rds.GetRasterBand(band)
    rgt = rds.GetGeoTransform()

    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)

    vds = ogr.Open(inShp, 1)  # TODO maybe open update if we want to write stats
    
    beams = ['BEAM0000','BEAM0001','BEAM0010','BEAM0011','BEAM0101','BEAM0110',
                 'BEAM1000','BEAM1011']
    
    for beam in beams:
         vlyr = vds.GetLayerByName(beam)
         
         feat = vlyr.GetNextFeature()
         features = np.arange(vlyr.GetFeatureCount())
         print(beam+" processing")
         featList = []
         for label in tqdm(features):
        
                if feat is None:
                    continue
                
                # the vector geom
                geom = feat.geometry()
          
                mx, my = geom.GetX(), geom.GetY()  #coord in map units

                # Convert from map to pixel coordinates.
                # No rotation but for this that should not matter
                px = int((mx - rgt[0]) / rgt[1])
                py = int((my - rgt[3]) / rgt[5])
                
                
                src_array = rb.ReadAsArray(px, py, 1, 1)

                if src_array is None:
                    # unlikely but if none will have no data in the attribute table
                    continue
                outval =  int(src_array.max())
                # unacceptable vals
                if outval in {0,190,200,202,210,220}:
                    featList.append(feat.GetFID())
                    
                    

        
         [vlyr.DeleteFeature(f) for f in featList]    
    
    vlyr.SyncToDisk()



    vds = None
    rds = None



