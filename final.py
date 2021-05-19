#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:03:00 2020

@author: heatherkay
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:49:01 2019

@author: heatherkay
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from pandas import DataFrame
from scipy.optimize import curve_fit
import geopandas as gpd
import glob
from os import path
from scipy.stats import gaussian_kde

def glas_eco(folderin, fileout, folderout, naming='gla14_eco_'):
    """
    Function to compute q and provide results (csv) and figures (pdf). 
    Using ICESat-GLAS data.
    
    Parameters
    ----------
    
    folderin: string
            Filepath for folder with files ready for analysis
                
    naming: string
          Section of filename to remove to obtain eco ID.
          Default = 'gla14_eco_'
          
    fileout: string
           Filepath for results file ending '.csv'
           
    folderout: string
             Filepath for folder to save the figures            
    """
    #import csv with IDs and convert to dict
    df_id2 = pd.read_csv('./eco/final_ID.csv')
    df_id = df_id2.astype({'ECO_ID': 'str'})
    eco_ID = df_id.set_index('ECO_ID')['ECO_NAME'].to_dict()

    #using 'file' to title plot  
    fileList = glob.glob(folderin + '*.shp')

    #create df for results
    resultsa = pd.DataFrame(columns = ['eco', 'ID', 'qout', 'r_sq', 'deg_free', 'mse'])
    #resultsb = pd.DataFrame(columns = ['eco', 'ID', 'qout', 'r_sq', 'deg_free', 'rmse'])

    for file in fileList:
        df = gpd.read_file(file)
        hd, tl = path.split(file)
        name = tl.replace(naming, "")
        name = name.replace('.shp', "")
        name2 = eco_ID[name] 
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
            continue    
                   
        #regression 
        def f(x,q):
            return 1- np.exp(-q * x)
    
        x = final['i_h100'].to_numpy()
        y = final['i_cd'].to_numpy() 
        x = np.append(x, [0])
        y = np.append(y, [0])
    
        qout, qcov = curve_fit(f, x, y, 0.04)
        qout = qout.round(decimals=4)

        y_predict = f(x, qout)
        
        #calculating r2
        residuals2 = y - f(x, qout)
        res_ss2 = np.sum(residuals2**2)
        tot_ss2 = np.sum((y-np.mean(y))**2)
        r_sq = 1- (res_ss2/tot_ss2)
        r_sq = round(r_sq, 2)
            
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y, y_predict)
        mse = round(mse, 3)

        resultsa = resultsa.append({'eco':name, 'ID': name2, 'qout': qout, 
                                    'r_sq': r_sq, 'deg_free': footprints, 
                                    'mse': mse}, ignore_index=True)

        resultsa.to_csv(fileout)

        #plot the result
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)

        fig, ax = plt.subplots()
        ax.scatter(x, y, c=z, s=10, edgecolor='')
        plt.rcParams.update({'font.size':12}) 
        #sets title and axis labels
        ax.set_title(name2)
        ax.set_ylabel('Canopy Density')
        ax.set_xlabel('Height - h100 (m)')
        ax.set_xlim([0, 60])
        ax.set_ylim([0,1])
        #plotting regression
        #putting x data in an order, cause that's what the code needs
        xdata = np.linspace(0, 60)
        #for each value of x calculating the corresponding y value
        ycurve = [f(t, qout) for t in xdata]
        #plotting the curve
        ax.plot(xdata, ycurve, linestyle='-')
        #adding qout, r_sq and deg_free to plot
        ax.annotate('q = ' + str(qout[0]), xy=(0.975,0.15), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        #ax.annotate('r2 = ' + str(r_sq), xy=(0.975,0.15), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        ax.annotate('MSE = ' + str(mse),xy=(0.975,0.10), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')   
        ax.annotate('No of footprints = ' + str(footprints),xy=(0.975,0.05), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        plt.savefig(folderout + 'fig{}.pdf'.format(name))
        plt.close



def glas_grid_eco(folderin, fileout, folderout, naming=4, eco_loc=2):
    """
    Function to compute q and provide results (csv) and figures (pdf).
    Using ICESat-GLAS data.
    
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
           
    folderout: string
             Filepath for folder to save the figures            
    """
    #import csv with IDs and convert to dict
    df_id2 = pd.read_csv('./eco/final_ID.csv')
    df_id = df_id2.astype({'ECO_ID': 'str'})
    eco_ID = df_id.set_index('ECO_ID')['ECO_NAME'].to_dict()
    
    #using 'file' to title plot  
    fileList = glob.glob(folderin + '*.shp')

    #create df for results
    resultsa = pd.DataFrame(columns = ['eco', 'ID', 'qout', 'r_sq', 'deg_free', 'mse'])
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
        eco_name = eco_ID[eco]
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
         
        #regression 
        def f(x,q):
            return 1- np.exp(-q * x)
    
        x = final['i_h100'].to_numpy()
        y = final['i_cd'].to_numpy() 
        x = np.append(x, [0])
        y = np.append(y, [0])
    
        qout, qcov = curve_fit(f, x, y, 0.04)
        qout = qout.round(decimals=4)
        
        y_predict = f(x, qout)
        
        #calculating r2
        residuals2 = y - f(x, qout)
        res_ss2 = np.sum(residuals2**2)
        tot_ss2 = np.sum((y-np.mean(y))**2)
        r_sq = 1- (res_ss2/tot_ss2)
        r_sq = round(r_sq, 2)
                   
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y, y_predict)
        mse = round(mse, 3)
        
        #fig1 = plt.figure(); ax =fig1.add_subplot(1,1,1)
        #ax.scatter(plot['y'],plot['y_predict'])
        #plt.savefig1('./eco/results/figs/values{}.pdf'.format(name))
        #plt.close
        
        #extract info: eco, qout, r_sq, deg_free (only gets one eco in data)
        resultsa = resultsa.append({'eco': eco, 'ID': name, 'qout': qout, 
                                    'r_sq': r_sq, 'deg_free': footprints, 
                                    'mse': mse}, ignore_index=True)

        resultsa.to_csv(fileout)

        #plot the result
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)

        fig, ax = plt.subplots()
        ax.scatter(x, y, c=z, s=10, edgecolor='')
        plt.rcParams.update({'font.size':12}) 
        #sets title and axis labels
        ax.set_title('Grid square in ' + eco_name)
        ax.set_ylabel('Canopy Density')
        ax.set_xlabel('Height - h100 (m)')
        ax.set_xlim([0, 60])
        ax.set_ylim([0,1])
        #plotting regression
        #putting x data in an order, cause that's what the code needs
        xdata = np.linspace(0, 60)
        #for each value of x calculating the corresponding y value
        ycurve = [f(t, qout) for t in xdata]
        #plotting the curve
        ax.plot(xdata, ycurve, linestyle='-')
        #adding qout, r_sq and deg_free to plot
        #ax.annotate('adj_r2 = ' + str(adj_r2[0]), xy=(0.975,0.10), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        ax.annotate('q = ' + str(qout[0]), xy=(0.975,0.15), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        #ax.annotate(u'R\u0305'u'\u00b2 = ' + str(adj_r2), xy=(0.975,0.15), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        ax.annotate('mean squared error = ' + str(mse),xy=(0.975,0.10), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')   
        ax.annotate('No of footprints = ' + str(footprints),xy=(0.975,0.05), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        plt.savefig(folderout + 'fig{}_{}.pdf'.format(eco, name))
        plt.close
        
        
def glas_grid_only(folderin, fileout, folderout, naming=2):
    """
    Function to compute q and provide results (csv) and figures (pdf).
    Using ICESat-GLAS data.
    
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
    resultsa = pd.DataFrame(columns = ['ID', 'qout', 'r_sq', 'deg_free', 'mse'])
    #resultsb = pd.DataFrame(columns = ['eco', 'ID', 'qout', 'r_sq', 'deg_free', 'rmse'])

    for file in fileList:
        df = gpd.read_file(file)
        if df.empty:
            continue 
        hd, tl = path.split(file)
        shp_lyr_name = path.splitext(tl)[0]
        name_comp = shp_lyr_name.split('_')
        name = name_comp[naming] 
        
        print(name)
        
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
        
        if footprints < 50:
            continue
             
        #regression 
        def f(x,q):
            return 1- np.exp(-q * x)
    
        x = final['i_h100'].to_numpy()
        y = final['i_cd'].to_numpy() 
        x = np.append(x, [0])
        y = np.append(y, [0])
    
        qout, qcov = curve_fit(f, x, y, 0.04)
        qout = qout.round(decimals=4)
        
        y_predict = f(x, qout)
        
        #calculating r2
        residuals2 = y - f(x, qout)
        res_ss2 = np.sum(residuals2**2)
        tot_ss2 = np.sum((y-np.mean(y))**2)
        r_sq = 1- (res_ss2/tot_ss2)
        r_sq = round(r_sq, 2)
            
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y, y_predict)
        mse = round(mse, 3)
                
        #extract info: eco, qout, r_sq, deg_free (only gets one eco in data)
        resultsa = resultsa.append({'ID': name, 'qout': qout, 
                                    'r_sq': r_sq, 'deg_free': footprints, 
                                    'mse': mse}, ignore_index=True)

        resultsa.to_csv(fileout)

        #plot the result
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)

        fig, ax = plt.subplots()
        ax.scatter(x, y, c=z, s=10, edgecolor='')
        plt.rcParams.update({'font.size':12}) 

        #sets title and axis labels
        ax.set_title('Grid ' + name)
        ax.set_ylabel('Canopy Density')
        ax.set_xlabel('Height - h100 (m)')
        ax.set_xlim([0, 60])
        ax.set_ylim([0,1])
        #plotting regression
        #putting x data in an order, cause that's what the code needs
        xdata = np.linspace(0, 60)
        #for each value of x calculating the corresponding y value
        ycurve = [f(t, qout) for t in xdata]
        #plotting the curve
        ax.plot(xdata, ycurve, linestyle='-')
        #adding qout, r_sq and deg_free to plot
        #ax.annotate('adj_r2 = ' + str(adj_r2[0]), xy=(0.975,0.10), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        ax.annotate('q = ' + str(qout[0]), xy=(0.975,0.15), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        ax.annotate('MSE = ' + str(mse),xy=(0.975,0.10), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')   
        ax.annotate('No of footprints = ' + str(footprints),xy=(0.975,0.05), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        plt.savefig(folderout + 'fig{}.pdf'.format(name))
        plt.close
        


def gedi_ecoregions(folderin, fileout, folderout, naming='gedi_'):
    """
    Function to compute q and provide results (csv) and figures (pdf)
    
    Parameters
    ----------
    
    folderin: string
            Filepath for folder with files ready for analysis
                
    naming: string
          Section of filename to remove to obtain eco ID.
          Default = 'gla14_eco_'
          
    fileout: string
           Filepath for results file ending '.csv'
           
    folderout: string
             Filepath for folder to save the figures            
    """
    #import csv with IDs and convert to dict
    df_id2 = pd.read_csv('/Users/heatherkay/q_res/ID.csv')
    df_id = df_id2.astype({'ID': 'str'})
    eco_ID = df_id.set_index('ID')['ECO_NAME'].to_dict()

    #using 'file' to title plot  
    fileList = glob.glob(folderin + '*.gpkg')

    #create df for results
    resultsa = pd.DataFrame(columns = ['eco', 'ID', 'qout', 'r_sq', 'deg_free', 'mse'])
    #resultsb = pd.DataFrame(columns = ['eco', 'ID', 'qout', 'r_sq', 'deg_free', 'rmse'])

    for file in fileList:
        df = gpd.read_file(file, layer ='join')
        hd, tl = path.split(file)
        name = tl.replace(naming, "")
        name = name.replace('.gpkg', "")
        name2 = eco_ID[name] 
        #remove data with H_100 >= 0 prior to logging
        test2 = df[df['rh100']>=0] 
        
        #calculate canopy density
        rv = test2['rv']
        rg = test2['rg']
        cd = rv/(rv + rg)
        test2['cd'] = cd
        t3 = test2.dropna(subset = ['cd'])
        
        #convert height to metres
        incm = t3['rh100']
        x = incm/100
        t3['h100']=x 
        
        #create new column in df with log of H_100 
        y = np.log(x)
        test2a = t3.assign(log_i_h100 = y)
        
        if test2a.empty:
            continue

        #get quantiles
        a = np.quantile(test2a['log_i_h100'],0.95)
        b = np.quantile(test2a['log_i_h100'],0.05)

        #remove data outside of 5% quantiles
        #test3 = t3[t3.log_i_h100 >b]
        final = test2a[test2a.log_i_h100 <a]

        if final.empty:
            continue
        del a, b, x, y, test2, test2a, t3

        footprints = len(final['rh100'])
        
        if footprints < 100:
            continue    
     
        #regression 
        def f(x,q):
            return 1- np.exp(-q * x)
    
        x = final['rh100'].to_numpy()
        y = final['cd'].to_numpy() 
        x = np.append(x, [0])
        y = np.append(y, [0])
    
        qout, qcov = curve_fit(f, x, y, 0.04)
        qout = qout.round(decimals=4)

        y_predict = f(x, qout)
        
        #calculating r2
        residuals2 = y - f(x, qout)
        res_ss2 = np.sum(residuals2**2)
        tot_ss2 = np.sum((y-np.mean(y))**2)
        r_sq = 1- (res_ss2/tot_ss2)
        r_sq = round(r_sq, 2)
            
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y, y_predict)
        mse = round(mse, 3)

        resultsa = resultsa.append({'ID': name, 'qout': qout, 'r_sq': r_sq, 'deg_free': footprints, 'mse': mse}, ignore_index=True)
        resultsa.to_csv(fileout)
   
        #plot the result
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)

        fig, ax = plt.subplots()
        ax.scatter(x, y, c=z, s=10, edgecolor='')
        plt.rcParams.update({'font.size':12}) 

        #sets title and axis labels
        ax.set_title(name2)
        ax.set_ylabel('Canopy Density')
        ax.set_xlabel('Height - h100 (m)')
        ax.set_xlim([0, 60])
        ax.set_ylim([0,1])
        #plotting regression
        #putting x data in an order, cause that's what the code needs
        xdata = np.linspace(0, 60)
        #for each value of x calculating the corresponding y value
        ycurve = [f(t, qout) for t in xdata]
        #plotting the curve
        ax.plot(xdata, ycurve, linestyle='-')
        #adding qout, r_sq and deg_free to plot
        ax.annotate('q = ' + str(qout[0]), xy=(0.975,0.15), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        #ax.annotate('r2 = ' + str(r_sq), xy=(0.975,0.15), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        ax.annotate('MSE = ' + str(mse),xy=(0.975,0.10), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')   
        ax.annotate('No of footprints = ' + str(footprints),xy=(0.975,0.05), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        plt.savefig(folderout + 'fig{}.pdf'.format(name))
        plt.close
       
def gedi_grid(folderin, fileout, folderout, naming=2, eco_loc=1):
    """
    Function to compute q and provide results (csv) and figures (pdf)
    
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
    df_id2 = pd.read_csv('./eco/final_ID.csv')
    df_id = df_id2.astype({'ECO_ID': 'str'})
    eco_ID = df_id.set_index('ECO_ID')['ECO_NAME'].to_dict()

    #using 'file' to title plot  
    fileList = glob.glob(folderin + '*.gpkg')

    #create df for results
    resultsa = pd.DataFrame(columns = ['eco', 'ID', 'qout', 'deg_free', 'mse', 'join'])
    #resultsb = pd.DataFrame(columns = ['eco', 'ID', 'qout', 'r_sq', 'deg_free', 'rmse'])

    for file in fileList:
        df = gpd.read_file(file, layer='join')
        if df.empty:
            continue 
        hd, tl = path.split(file)
        shp_lyr_name = path.splitext(tl)[0]
        name_comp = shp_lyr_name.split('_')
        name = name_comp[naming] 
        eco = name_comp[eco_loc]
        eco_name = eco_ID[eco]
        print(name)
        print(eco)
        #remove data with H_100 >= 0 prior to logging
        test2 = df[df['rh100']>=0] 
        
        #calculate canopy density
        rv = test2['rv']
        rg = test2['rg']
        cd = rv/(rv + rg)
        test2['cd'] = cd
        t3 = test2.dropna(subset = ['cd'])
        
        #convert height to metres
        incm = t3['rh100']
        x = incm/100
        t3['h100']=x 
        
        #create new column in df with log of H_100 
        y = np.log(x)
        test2a = t3.assign(log_i_h100 = y)
        
        if test2a.empty:
            continue

        #get quantiles
        a = np.quantile(test2a['log_i_h100'],0.95)
        b = np.quantile(test2a['log_i_h100'],0.05)

        #remove data outside of 5% quantiles
        #test3 = test2a[test2a.log_i_h100 >b]
        final = test2a[test2a.log_i_h100 <a]

        if final.empty:
            continue
        del a, b, x, y, test2, test2a, t3


        footprints = len(final['h100'])
        
        if footprints < 100:
            continue
            
        #regression 
        def f(x,q):
            return 1- np.exp(-q * x)
    
        x = final['rh100'].to_numpy()
        y = final['cd'].to_numpy() 
        x = np.append(x, [0])
        y = np.append(y, [0])
    
        qout, qcov = curve_fit(f, x, y, 0.04)
        qout = qout.round(decimals=4)
        
        y_predict = f(x, qout)
            
        from sklearn.metrics import mean_squared_error

        mse = mean_squared_error(y, y_predict)
        mse = round(mse, 3)
        
        #j_e = eco.astype(str)
        #j_g = name.astype(str)
        join = eco + '_' + name
        
       
        resultsa = resultsa.append({'eco': eco, 'ID': name, 'qout': qout, 
                                    'deg_free': footprints, 
                                    'mse': mse, 'join': join}, 
                                    ignore_index=True)

        resultsa.to_csv(fileout)

        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)

        fig, ax = plt.subplots()
        ax.scatter(x, y, c=z, s=10, edgecolor='')
        plt.rcParams.update({'font.size':12}) 

        ax.set_title('Grid square within ' + eco_name)
        ax.set_ylabel('Canopy Density')
        ax.set_xlabel('Height - h100 (m)')
        ax.set_xlim([0, 60])
        ax.set_ylim([0,1])
        #plotting regression
        #putting x data in an order, cause that's what the code needs
        xdata = np.linspace(0, 60)
        #for each value of x calculating the corresponding y value
        ycurve = [f(t, qout) for t in xdata]
        #plotting the curve
        ax.plot(xdata, ycurve, linestyle='-', color='red')
        #adding qout, mse and deg_free to plot
        #ax.annotate('adj_r2 = ' + str(adj_r2[0]), xy=(0.975,0.10), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        ax.annotate('q = ' + str(qout[0]), xy=(0.975,0.15), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        ax.annotate('MSE = ' + str(mse), xy=(0.975,0.10), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        ax.annotate('No of footprints = ' + str(footprints),xy=(0.975,0.05), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        plt.savefig(folderout + 'fig{}_{}.pdf'.format(eco, name))
        plt.close     
         