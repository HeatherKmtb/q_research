#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:53:26 2020

@author: heatherkay
"""

import pandas as pd
import numpy as np
from pandas import DataFrame
import glob
import os.path
#import geopandas as gpd
import os, ogr, osr
import pickle
#import rsgislib.zonalstats
from rasterstats import zonal_stats

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
    split_files = glob.glob(folderin + '*.gpkg')

    for filename in split_files:
        basename = os.path.splitext(os.path.basename(filename))[0]
        inputimage = filein
        inputvector = filename
        outputvector = os.path.join(folderout, "{}_lcjoin.gpkg".format(basename))
        removeExistingVector = True
        useBandNames = True
        print(inputimage)
        print(inputvector)
        print(outputvector)
        rsgislib.zonalstats.pointValue2SHP(inputimage, inputvector, outputvector, removeExistingVector, useBandNames)


def lcjoin_gpkg(folderin, tiffile, folderout):
    """
    Function to join gedi (gpkg) files with ccilc tif

    Parameters
    ----------
    folderin : string
        filepath for directory with GEDI gpkg files
    
    folderout: string
        filepath for directory for new 
    
    beam : string, optional
        layer definition in gpkg. The default is 'BEAM0000'.

    Returns
    -------
    None.

    """   
    
    
    filelist = glob.glob(folderin + '*.gpkg')
    for file in filelist:
        name = os.path.splitext(os.path.basename(file))[0]
        infile = file
        openfile = ogr.Open(infile)
        #inlayer = openfile.GetLayerByName(beam)
        #inLayerDefn = inlayer.GetLayerDefn()
        #inLayerRef = inlayer.GetSpatialRef()
 
        beams = ['BEAM0000','BEAM0001','BEAM0010','BEAM0011','BEAM0101','BEAM0110',
                 'BEAM1000','BEAM1011']
        
        for beam in beams:
            inlayer = openfile.GetLayerByName(beam)
            inLayerDefn = inlayer.GetLayerDefn()
            inLayerRef = inlayer.GetSpatialRef()
                        
            outpath = folderout + '{}_{}.gpkg'.format(name, beam)
            out_driver = ogr.GetDriverByName("gpkg")
            out_ds = out_driver.CreateDataSource(outpath)
            outlayer = out_ds.CreateLayer('outlayer', geom_type=ogr.wkbPoint, srs=inLayerRef)
            deflay = outlayer.GetLayerDefn()
            #outSpatialRef = osr.SpatialReference()
            #outSpatialRef.ImportFromEPSG(4326)
           
            for i in range(0, inLayerDefn.GetFieldCount()):
            
                fieldDefn = inLayerDefn.GetFieldDefn(i)
                #fieldName = fieldDefn.GetName()

                outlayer.CreateField(fieldDefn)

            for inFeature in inlayer:
                # Create output Feature
                #buffer point
                #pt = ogr.CreateGeometry(inFeature) #geometry of the point of inFeature 
                pt = inFeature.geometry()
                bufferDistance = 10
                poly = pt.Buffer(bufferDistance)
                
                #add tif data to buffered point
                stats = zonal_stats(poly, tiffile)
                stats[0].keys()
                #dict_keys(['min', 'max', 'mode', 'count'])
                [f['mode'] for f in stats]
               
               
                outFeature = ogr.Feature(deflay)

                # Add field values from input Layer
                for i in range(0, deflay.GetFieldCount()):
                    fieldDefn = deflay.GetFieldDefn(i)
                    #fieldName = fieldDefn.GetName()
                    
                    outFeature.SetField(deflay.GetFieldDefn(i).GetNameRef(), 
                                    inFeature.GetField(i))
                    
                    #add field values from tif?
                    
    
            
        out_ds.SyncToDisk()
        out_ds=None



#for reference
countries_gdf = geopandas.read_file("package.gpkg", layer='countries')

#pointless function that is incredibly slow and extracts each layer of the gpkg to it's own individual gpkg
def extract_gpkg(folderin, folderout):
    """
    Function to join gedi (gpkg) files with ccilc tif

    Parameters
    ----------
    folderin : string
        filepath for directory with GEDI gpkg files
    
    folderout: string
        filepath for directory for new 
    
    beam : string, optional
        layer definition in gpkg. The default is 'BEAM0000'.

    Returns
    -------
    None.

    """   
    
    
    filelist = glob.glob(folderin + '*.gpkg')
    for file in filelist:
        name = os.path.splitext(os.path.basename(file))[0]
        infile = file
        openfile = ogr.Open(infile)
        #inlayer = openfile.GetLayerByName(beam)
        #inLayerDefn = inlayer.GetLayerDefn()
        #inLayerRef = inlayer.GetSpatialRef()
 
        beams = ['BEAM0000','BEAM0001','BEAM0010','BEAM0011','BEAM0101','BEAM0110',
                 'BEAM1000','BEAM1011']
        
        for beam in beams:
            inlayer = openfile.GetLayerByName(beam)
            inLayerDefn = inlayer.GetLayerDefn()
            inLayerRef = inlayer.GetSpatialRef()
                        
            outpath = folderout + '{}_{}.gpkg'.format(name, beam)
            out_driver = ogr.GetDriverByName("gpkg")
            out_ds = out_driver.CreateDataSource(outpath)
            outlayer = out_ds.CreateLayer('outlayer', geom_type=ogr.wkbPoint, srs=inLayerRef)
            deflay = outlayer.GetLayerDefn()
            #outSpatialRef = osr.SpatialReference()
            #outSpatialRef.ImportFromEPSG(4326)
           
            for i in range(0, inLayerDefn.GetFieldCount()):
            
                fieldDefn = inLayerDefn.GetFieldDefn(i)
                #fieldName = fieldDefn.GetName()

                outlayer.CreateField(fieldDefn)

            for inFeature in inlayer:
                # Create output Feature
                outFeature = ogr.Feature(deflay)

                # Add field values from input Layer
                for i in range(0, deflay.GetFieldCount()):
                    fieldDefn = deflay.GetFieldDefn(i)
                    #fieldName = fieldDefn.GetName()
                    
                    outFeature.SetField(deflay.GetFieldDefn(i).GetNameRef(), 
                                    inFeature.GetField(i))
                
                # Set geometry as centroid
                geom = inFeature.GetGeometryRef()
                outFeature.SetGeometry(geom.Clone())
                # Add new feature to output Layer
                outlayer.CreateFeature(outFeature)
                outFeature = None    
            
            out_ds.SyncToDisk()
            out_ds=None

