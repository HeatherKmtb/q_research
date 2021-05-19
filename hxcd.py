#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:09:51 2021

@author: heatherkay
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.optimize import curve_fit
import geopandas as gpd
import glob
from os import path
from scipy.stats import gaussian_kde

def grid_hxcd(folderin, fileout, naming=4, eco_loc=2):
    """
    Function to calculate mean of top 10% of canopy height and canopy density
    for each polygon (intersection of 1 degree grid and wwf ecoregions)
    
    Parameters
    ----------
    
    folderin: string
            Filepath for folder with files ready for analysis
                
    naming: int
          Section of filename to obtain ID (here grid number). Obtained
          by splitting filename by '_' and indexing
          Default = 3

    eco_loc: int
          Section of filename to obtain ecoregion (if applicable). 
          Obtained as with naming
          Default = 2
          
    fileout: string
           Filepath for results file ending '.csv'
           
    folderout: string
             Filepath for folder to save the figures            
    """

    #using 'file' to title plot  
    fileList = glob.glob(folderin + '*.shp')

    #create df for results
    results = pd.DataFrame(columns = ['eco', 'ID', 'height', 'cd', 'result', 'deg_free'])
    #resultsb = pd.DataFrame(columns = ['eco', 'ID', 'qout', 'r_sq', 'deg_free', 'rmse'])

    for file in fileList:
        df = gpd.read_file(file)
        if df.empty:
            continue 
        hd, tl = path.split(file)
        shp_lyr_name = path.splitext(tl)[0]
        name_comp = shp_lyr_name.split('_')
        name = name_comp[naming] 
        eco = name_comp[eco_loc]
        print(name)
        print(eco)
        #remove data with H_100 >= 0 prior to logging
        test2 = df[df['i_h100']>=0] 
        
        #means x is just the h100 data - needs logging to normalise (not skewed) 
        x = test2['i_h100']
        
        #create new column in df with log of H_100 
        y = np.log(x)
        test2a = test2.assign(log_i_h100 = y)
        
        if test2a.empty:
            continue

        #get quantiles
        a = np.quantile(test2a['log_i_h100'],0.95)
        b = np.quantile(test2a['log_i_h100'],0.05)

        #remove data outside of 5% quantiles
        test3 = test2a[test2a.log_i_h100 >b]
        final = test3[test3.log_i_h100 <a]

        if final.empty:
            continue
        del a, b, x, y, test2, test2a, test3

        #get 10% quantiles
        height = np.quantile(final['i_h100'], 0.10)
        cd = np.quantile(final['i_cd'], 0.10)
        
        mean_h = height.mean()
        mean_cd = cd.mean()
        result = mean_h * mean_cd


        footprints = len(final['i_h100'])
        
        if footprints < 100:
            print(name +'_' + eco)
            continue
        
                
        results = results.append({'eco': eco, 'ID': name, 'height': mean_h,
                                  'cd': mean_cd, 'result': result, 
                                  'deg_free': footprints}, ignore_index=True)
        
        results.to_csv(fileout)
        
def grid_mean_hxcd(folderin, fileout, naming=4, eco_loc=2):
    """
    Function to calculate mean canopy height and canopy density
    for each polygon (intersection of 1 degree grid and wwf ecoregions)
    
    Parameters
    ----------
    
    folderin: string
            Filepath for folder with files ready for analysis
                
    naming: int
          Section of filename to obtain ID (here grid number). Obtained
          by splitting filename by '_' and indexing
          Default = 3

    eco_loc: int
          Section of filename to obtain ecoregion (if applicable). 
          Obtained as with naming
          Default = 2
          
    fileout: string
           Filepath for results file ending '.csv'
           
    folderout: string
             Filepath for folder to save the figures            
    """

    #using 'file' to title plot  
    fileList = glob.glob(folderin + '*.shp')

    #create df for results
    results = pd.DataFrame(columns = ['eco', 'ID', 'height', 'cd', 'result', 'deg_free'])
    #resultsb = pd.DataFrame(columns = ['eco', 'ID', 'qout', 'r_sq', 'deg_free', 'rmse'])

    for file in fileList:
        df = gpd.read_file(file)
        if df.empty:
            continue 
        hd, tl = path.split(file)
        shp_lyr_name = path.splitext(tl)[0]
        name_comp = shp_lyr_name.split('_')
        name = name_comp[naming] 
        eco = name_comp[eco_loc]
        print(name)
        print(eco)
        #remove data with H_100 >= 0 prior to logging
        test2 = df[df['i_h100']>=0] 
        
        #means x is just the h100 data - needs logging to normalise (not skewed) 
        x = test2['i_h100']
        
        #create new column in df with log of H_100 
        y = np.log(x)
        test2a = test2.assign(log_i_h100 = y)
        
        if test2a.empty:
            continue

        #get quantiles
        a = np.quantile(test2a['log_i_h100'],0.95)
        b = np.quantile(test2a['log_i_h100'],0.05)

        #remove data outside of 5% quantiles
        test3 = test2a[test2a.log_i_h100 >b]
        final = test3[test3.log_i_h100 <a]

        if final.empty:
            continue
        del a, b, x, y, test2, test2a, test3

        footprints = len(final['i_h100'])
        
        if footprints < 100:
            print(name +'_' + eco)
            continue
        
        height = final['i_h100']
        cd = final['i_cd']
        
        mean_h = height.mean()
        mean_cd = cd.mean()
        result = mean_h * mean_cd

        results = results.append({'eco': eco, 'ID': name, 'height': mean_h,
                                  'cd': mean_cd, 'result': result, 
                                  'deg_free': footprints}, ignore_index=True)
        
        results.to_csv(fileout)      
        
        
