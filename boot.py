
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:33:56 2019

@author: heatherkay
"""


import pandas as pd
import numpy as np
from pandas import DataFrame
from scipy.optimize import curve_fit
import geopandas as gpd
import glob
from os import path
from sklearn.utils import resample as smpl


def glas_ecoregions(folderin, fileout, naming='gla14_eco_'):
    """
    Function to run bootstrpping with replacement over 100 iterations, for ICESat GLAS
    data grouped per ecoregion
    
    Parameters
    ----------
    
    folderin: string
            Filepath for folder with files ready for analysis
                
    naming: string
          Section of filename to remove to obtain eco ID.
          Default = 'gla14_eco_'
          
    fileout: string
           Filepath for results file ending '.csv'
           
    """


    fileList = glob.glob(folderin + '*.shp')

    q_samples = pd.DataFrame(columns=['ID', 'mean_q', 'sd_q', 'SE'])

    for file in fileList:
        df2 = gpd.read_file(file)
        hd, tl = path.split(file)
        name = tl.replace(naming, "")
        name = name.replace('.shp', "")
        print(name)
        q_values = pd.DataFrame(columns=['q'])
    
        for i in range(100):
            df = smpl(df2)
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
        
            def f(x,q):
                return 1- np.exp(-q * x)
    
            x = final['i_h100'].to_numpy()
            y = final['i_cd'].to_numpy() 
    
            qout, qcov = curve_fit(f, x, y, 0.04)
            qout = qout.round(decimals=4)
        
            q_values = q_values.append({'q':qout}, ignore_index=True)
            del df, final, x, y, qout
       
        mean_q = q_values['q'].mean()
        sd_q = q_values['q'].std()
        SE = sd_q/10
        q_samples = q_samples.append({'ID':name, 'mean_q':mean_q, 'sd_q':sd_q, 'SE':SE}, ignore_index=True)
        del mean_q, sd_q, q_values   
            #df2 = smpl(df, stratify=df['h_100'])
    q_samples.to_csv(fileout)   
            #means x is just the h100 data - needs logging to normalise (not skewed) 

def glas_grid_eco(folderin, fileout, naming=4, eco_loc=2):
    """
    Function to run bootstrpping with replacement over 100 iterations, for ICESat GLAS
    data grouped per ecoregion and 1 degree grid
    
    Parameters
    ----------
    
    folderin: string
            Filepath for folder with files ready for analysis
                
    naming: int
          Section of filename to obtain ID (here grid number). Obtained
          by splitting filename by '_' and indexing
          Default = 4

    eco_loc: int
          Section of filename to obtain ecoregion (if applicable). 
          Obtained as with naming
          Default = 2
          
    fileout: string
           Filepath for results file ending '.csv'
           
    """

    fileList = glob.glob(folderin + '*.shp')

    q_samples = pd.DataFrame(columns=['ID', 'eco', 'mean_q', 'sd_q', 'SE'])

    for file in fileList:
        df2 = gpd.read_file(file)
        hd, tl = path.split(file)
        shp_lyr_name = path.splitext(tl)[0]
        name_comp = shp_lyr_name.split('_')
        name = name_comp[naming] 
        eco = name_comp[eco_loc]
        print(name)
        print(eco)
        q_values = pd.DataFrame(columns=['q'])
    
        for i in range(100):
            df = smpl(df2)
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
        
            def f(x,q):
                return 1- np.exp(-q * x)
    
            x = final['i_h100'].to_numpy()
            y = final['i_cd'].to_numpy() 
    
            qout, qcov = curve_fit(f, x, y, 0.04)
            qout = qout.round(decimals=4)
        
            q_values = q_values.append({'q':qout}, ignore_index=True)
            del df, final, x, y, qout
       
        mean_q = q_values['q'].mean()
        sd_q = q_values['q'].std()
        SE = sd_q/10
        q_samples = q_samples.append({'ID':name, 'eco':eco, 'mean_q':mean_q, 'sd_q':sd_q, 'SE':SE}, ignore_index=True)
        del mean_q, sd_q, q_values   
            #df2 = smpl(df, stratify=df['h_100'])
    q_samples.to_csv(fileout)   
            #means x is just the h100 data - needs logging to normalise (not skewed) 




