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
from pandas import DataFrame
from scipy.optimize import curve_fit
import geopandas as gpd
import glob
from os import path


def ecoregions(folderin, fileout, folderout, naming='gla14_eco_'):
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
    df_id2 = pd.read_csv('./eco/final_ID.csv')
    df_id = df_id2.astype({'ECO_ID': 'str'})
    eco_ID = df_id.set_index('ECO_ID')['ECO_NAME'].to_dict()

    #using 'file' to title plot  
    fileList = glob.glob(folderin + '*.shp')

    #create df for results
    resultsa = pd.DataFrame(columns = ['eco', 'ID', 'qout', 'r_sq', 'deg_free', 'rmse','r_sq_mean'])
    #resultsb = pd.DataFrame(columns = ['eco', 'ID', 'qout', 'r_sq', 'deg_free', 'rmse'])

    for file in fileList:
        df = gpd.read_file(file)
        hd, tl = path.split(file)
        name = tl.replace(naming, "")
        name = name.replace('.shp', "")
        name2 = eco_ID[name] 
        #remove data with H_100 >= 0 prior to logging
        test2 = df[df['i_h100']>=0] 
        footprints = len(df['i_h100'])
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
    
        #NEXT STEP. Bin remaining data in order to get mean and IQR of each bin

        #add column with bins 
        final['H_bins']=pd.cut(x=final['i_h100'], bins=np.arange(0, 120+2, 2))

        #now something along the lines of:
        #for bin in HBins find the mean and IQR...
        #first create lists to append mean and IQRs to
        cd_mean = []
        cd_iqr = []
        #Hbin = []
        print(name)
        HBins = list(np.unique(final['H_bins']))
        for bins in HBins:
            #for each one make a df with just that bin
            new = final.loc[final['H_bins']==bins]
            #get mean and IQR of each bin
            data = new['i_cd'].to_numpy()
            mean = data.mean()
            cd_mean.append(mean)
            q75, q25 = np.percentile (data, [75, 25])
            iqr = q75 - q25
            cd_iqr.append(iqr)
    
        #getting median of bins for mean r2 calculation
        greats = []
        for index,i in final.iterrows():
            great = [i['H_bins'].left + 1] 
            greats.append(great)

    
        final['H_bin'] = greats 
        new1 = final['H_bin'] = final.H_bin.astype(str)
        new2 = new1.str.strip('[]').astype(int)
        final['H_bin1'] = new2
        
        del new, data, q75, q25, new1 
    
        #get median of bins for plotting
        med = [binn.left + 1 for binn in HBins]
        plot = pd.DataFrame({'mean': cd_mean, 'iqr': iqr, 'bins': HBins, 'median': med})
        bin_dict = plot.set_index('median')['mean'].to_dict()
    
        plot_y = []
        for i in final['H_bin1']:
            y = bin_dict[i]
            plot_y.append(y)
            del y
        
        final['plot_y'] = plot_y
     
        #regression 
        def f(x,q):
            return 1- np.exp(-q * x)
    
        x = final['i_h100'].to_numpy()
        y = final['i_cd'].to_numpy() 
        x = np.append(x, [0])
        y = np.append(y, [0])
    
        qout, qcov = curve_fit(f, x, y, 0.04)
        qout = qout.round(decimals=4)
        #calculating mean r2
        residuals = plot_y - f(new2, qout)
        res_ss = np.sum(residuals**2)
        tot_ss = np.sum((plot_y-np.mean(plot_y))**2)
        r_sq_mean = 1 - (res_ss/tot_ss)
        #deg_free = (len(x)-1)
        r_sq_mean = round(r_sq_mean, 2)
        y_predict = f(x, qout)
        
        #calculating r2
        residuals2 = y - f(x, qout)
        res_ss2 = np.sum(residuals2**2)
        tot_ss2 = np.sum((y-np.mean(y))**2)
        r_sq = 1- (res_ss2/tot_ss2)
        r_sq = round(r_sq, 2)
            
        from sklearn.metrics import mean_squared_error
        from math import sqrt
        mse = mean_squared_error(y, y_predict)
        rms = sqrt(mse)
        rms = round(rms, 4)

        #fig1 = plt.figure(); ax =fig1.add_subplot(1,1,1)
        #ax.scatter(plot['y'],plot['y_predict'])
        #plt.savefig1('./eco/results/figs/values{}.pdf'.format(name))
        #plt.close
        
        #extract info: eco, qout, r_sq, deg_free (only gets one eco in data)
        resultsa = resultsa.append({'ID': name, 'qout': qout, 'r_sq': r_sq, 'deg_free': footprints, 'rmse': rms, 'r_sq_mean': r_sq_mean}, ignore_index=True)
        #if deg_free>=60:
            #resultsb = resultsb.append({'eco': name2, 'ID': name, 'qout': qout, 'r_sq': r_sq, 'deg_free': deg_free, 'rmse': rms}, ignore_index=True)        
            #export to excel
        resultsa.to_csv(fileout)
            #resultsb.to_csv('./eco/new/results/results_over60.csv')

        #plot the result
        fig = plt.figure(); ax = fig.add_subplot(1,1,1)
        plt.rcParams.update({'font.size':12})
        #plots H_100 on x with I_CD on y
        ax.scatter(plot['median'],plot['mean'])
        #plots IQR
        ax.bar(plot['median'],plot['mean'],width=0, yerr=plot['iqr'])
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
        ax.annotate('q = ' + str(qout[0]), xy=(0.975,0.20), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        ax.annotate('r2 = ' + str(r_sq), xy=(0.975,0.15), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        ax.annotate('RMSE = ' + str(rms),xy=(0.975,0.10), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')   
        ax.annotate('No of footprints = ' + str(footprints),xy=(0.975,0.05), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        plt.savefig(folderout + 'fig{}.pdf'.format(name))
        plt.close


def worldclim(folderin, fileout, folderout, ecosplit = 2, climsplit = 4):
    """
    Function to compute q and provide results (csv) and figures (pdf)
    
    Parameters
    ----------
    
    folderin: string
            Filepath for folder with files ready for analysis
            
    fileout: string
           Filepath for results file ending '.csv'
           
    folderout: string
             Filepath for folder to save the figures      
    """
    #import csv with IDs and convert to dict
    #df_id2 = pd.read_csv('./eco/final_ID.csv')
    #df_id = df_id2.astype({'ECO_ID': 'str'})
    #eco_ID = df_id.set_index('ECO_ID')['ECO_NAME'].to_dict()

    #using 'file' to title plot  
    fileList = glob.glob(folderin + '*.shp')

    #create df for results
    resultsa = pd.DataFrame(columns = ['ID', 'qout', 'r_sq', 'deg_free', 'rmse', 'r_sq_mean'])
    #resultsb = pd.DataFrame(columns = ['eco', 'ID', 'qout', 'r_sq', 'deg_free', 'rmse', 'r_sq_mean'])



    for i in range(10101,81334):
        ecofile = [f for f in fileList if 'eco_'+str(i)+'_eco' in f]
        for eco in ecofile:
            print(eco)
            hd, tl = path.split(eco)
            shp_lyr_name = path.splitext(tl)[0]
            name_comps = shp_lyr_name.split('_')
            print(name_comps)
            name = name_comps[ecosplit] # eco region
            print(name)
            clim = name_comps[climsplit]
            print(name, clim)
        
            df = gpd.read_file(eco)
            test1 = df[df['i_h100']>=0] 
            test2 = test1[test1['i_h100']<=122]
            #means x is just the h100 data - needs logging to normalise (not skewed) 
            x = test2['i_h100']
            footprints = len(test2['i_h100'])    
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
    
            #NEXT STEP. Bin remaining data in order to get mean and IQR of each bin

            #add column with bins 
            final['H_bins']=pd.cut(x=final['i_h100'], bins=np.arange(0, 120+2, 2))

            #now something along the lines of:
            #for bin in HBins find the mean and IQR...
            #first create lists to append mean and IQRs to
            cd_mean = []
            cd_iqr = []
            #Hbin = []
            print(name, clim)
            HBins = list(np.unique(final['H_bins']))
            for bins in HBins:
                #for each one make a df with just that bin
                new = final.loc[final['H_bins']==bins]
                #get mean and IQR of each bin
                data = new['i_cd'].to_numpy()
                mean = data.mean()
                cd_mean.append(mean)
                q75, q25 = np.percentile (data, [75, 25])
                iqr = q75 - q25
                cd_iqr.append(iqr)
    
            #getting median of bins for mean r2 calculation
            greats = []
            for index,i in final.iterrows():
                    great = [i['H_bins'].left + 1] 
                    greats.append(great)

    
            final['H_bin'] = greats 
            new1 = final['H_bin'] = final.H_bin.astype(str)
            new2 = new1.str.strip('[]').astype(int)
            final['H_bin1'] = new2
        
            del new, data, q75, q25, new1 
    
            #get median of bins for plotting
            med = [binn.left + 1 for binn in HBins]
            plot = pd.DataFrame({'mean': cd_mean, 'iqr': iqr, 'bins': HBins, 'median': med})
            bin_dict = plot.set_index('median')['mean'].to_dict()
    
            plot_y = []
            for i in final['H_bin1']:
                y = bin_dict[i]
                plot_y.append(y)
                del y
        
            final['plot_y'] = plot_y
     
            #regression 
            def f(x,q):
                return 1- np.exp(-q * x)
    
            x = final['i_h100'].to_numpy()
            y = final['i_cd'].to_numpy() 
            x = np.append(x, [0])
            y = np.append(y, [0])
    
            qout, qcov = curve_fit(f, x, y, 0.04)
            qout = qout.round(decimals=4)
            #calculating mean r2
            residuals = plot_y - f(new2, qout)
            res_ss = np.sum(residuals**2)
            tot_ss = np.sum((plot_y-np.mean(plot_y))**2)
            r_sq_mean = 1 - (res_ss/tot_ss)
            #deg_free = (len(x)-1)
            r_sq_mean = round(r_sq_mean, 2)
            y_predict = f(x, qout)

            #calculating r2
            residuals2 = y - f(x, qout)
            res_ss2 = np.sum(residuals2**2)
            tot_ss2 = np.sum((y-np.mean(y))**2)
            r_sq = 1- (res_ss2/tot_ss2)
            r_sq = round(r_sq, 2)
        
            from sklearn.metrics import mean_squared_error
            from math import sqrt
            mse = mean_squared_error(y, y_predict)
            rms = sqrt(mse)
            rms = round(rms, 4)

            #extract info: eco, qout, r_sq, deg_free (only gets one eco in data)
            resultsa = resultsa.append({'ID': name, 'clim': clim, 'qout': qout, 'r_sq': r_sq, 'deg_free': footprints, 'rmse': rms, 'r_sq_mean': r_sq_mean}, ignore_index=True)
            #if deg_free>=60:
                #resultsb = resultsb.append({'eco': name2, 'ID':name, 'clim':clim, 'qout': qout, 'r_sq': r_sq, 'deg_free': deg_free, 'rmse': rms}, ignore_index=True)        
                #export to excel
            resultsa.to_csv(fileout)
            #resultsb.to_csv('./worldclim/worldclim01/results/results_over60footprints.csv')

            #plot the result
            fig = plt.figure(); ax = fig.add_subplot(1,1,1)
            plt.rcParams.update({'font.size':12})
            #plots H_100 on x with I_CD on y
            ax.scatter(plot['median'],plot['mean'])
            #plots IQR
            ax.bar(plot['median'],plot['mean'],width=0, yerr=plot['iqr'])
            #sets title and axis labels
            ax.set_title(name + 'with climate' + clim)
            ax.set_ylabel('Canopy Density')
            ax.set_xlabel('Height - h100')
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
            ax.annotate('q = ' + str(qout[0]), xy=(0.975,0.20), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
            ax.annotate('r2 = ' + str(r_sq), xy=(0.975,0.15), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
            ax.annotate('RMSE = ' + str(rms),xy=(0.975,0.10), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')   
            ax.annotate('No of footprints = ' + str(footprints),xy=(0.975,0.05), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
            plt.savefig(folderout + 'fig{}_{}.pdf'.format(name, clim))
            plt.close
            #ax.annotate('r2 mean = ' + str(r_sq_mean), xy=(0.975,0.20), xycoords='axes fraction', fontsize=9, horizontalalignment='right', verticalalignment='bottom')


def biomes(folderin, fileout, folderout, naming='gla14_eco_'):
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
    #df_id2 = pd.read_csv('./eco/final_ID.csv')
    #df_id = df_id2.astype({'BIOME': 'str'})
    #eco_ID = df_id.set_index('ECO_ID')['ECO_NAME'].to_dict()
    eco_ID = {'1':'Tropical & Subtropical Moist Broadleaf','2':'Tropical & Subtropical Dry Broadleaf',
              '3':'Tropical & Subtropical Coniferous Forests','4':'Temperate Broadleaf & Mixed Forests'
              ,'5':'Temperate Conifer','7':'Tropical & Subtropical Grasslands, Savanna and Shrub',
              '9':'Flooded Grass & Shrub','10':'Montane Grass & Shrub','12':'Mediterranean',
              '13':'Deserts & Xeric Shrublands','14':'Mangroves'}

    #using 'file' to title plot  
    fileList = glob.glob(folderin + '*.shp')

    #create df for results
    resultsa = pd.DataFrame(columns = ['eco', 'ID', 'qout', 'r_sq', 'deg_free', 'rmse','r_sq_mean'])
    #resultsb = pd.DataFrame(columns = ['eco', 'ID', 'qout', 'r_sq', 'deg_free', 'rmse'])

    for file in fileList:
        df = gpd.read_file(file)
        hd, tl = path.split(file)
        name = tl.replace(naming, "")
        name = name.replace('.shp', "")
        name2 = eco_ID[name] 
        #remove data with H_100 >= 0 prior to logging
        test2 = df[df['i_h100']>=0] 
        footprints = len(df['i_h100'])
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
        
    
        #NEXT STEP. Bin remaining data in order to get mean and IQR of each bin

        #add column with bins 
        final['H_bins']=pd.cut(x=final['i_h100'], bins=np.arange(0, 120+2, 2))

        #now something along the lines of:
        #for bin in HBins find the mean and IQR...
        #first create lists to append mean and IQRs to
        cd_mean = []
        cd_iqr = []
        #Hbin = []
        print(name)
        HBins = list(np.unique(final['H_bins']))
        for bins in HBins:
            #for each one make a df with just that bin
            new = final.loc[final['H_bins']==bins]
            #get mean and IQR of each bin
            data = new['i_cd'].to_numpy()
            mean = data.mean()
            cd_mean.append(mean)
            q75, q25 = np.percentile (data, [75, 25])
            iqr = q75 - q25
            cd_iqr.append(iqr)
    
        #getting median of bins for mean r2 calculation
        greats = []
        for index,i in final.iterrows():
            great = [i['H_bins'].left + 1] 
            greats.append(great)

    
        final['H_bin'] = greats 
        new1 = final['H_bin'] = final.H_bin.astype(str)
        new2 = new1.str.strip('[]').astype(int)
        final['H_bin1'] = new2
        
        del new, data, q75, q25, new1 
    
        #get median of bins for plotting
        med = [binn.left + 1 for binn in HBins]
        plot = pd.DataFrame({'mean': cd_mean, 'iqr': iqr, 'bins': HBins, 'median': med})
        bin_dict = plot.set_index('median')['mean'].to_dict()
    
        plot_y = []
        for i in final['H_bin1']:
            y = bin_dict[i]
            plot_y.append(y)
            del y
        
        final['plot_y'] = plot_y
     
        #regression 
        def f(x,q):
            return 1- np.exp(-q * x)
    
        x = final['i_h100'].to_numpy()
        y = final['i_cd'].to_numpy() 
        x = np.append(x, [0])
        y = np.append(y, [0])
    
        qout, qcov = curve_fit(f, x, y, 0.04)
        qout = qout.round(decimals=4)
        #calculating mean r2
        residuals = plot_y - f(new2, qout)
        res_ss = np.sum(residuals**2)
        tot_ss = np.sum((plot_y-np.mean(plot_y))**2)
        r_sq_mean = 1 - (res_ss/tot_ss)
        #deg_free = (len(x)-1)
        r_sq_mean = round(r_sq_mean, 2)
        y_predict = f(x, qout)
        
        #calculating r2
        residuals2 = y - f(x, qout)
        res_ss2 = np.sum(residuals2**2)
        tot_ss2 = np.sum((y-np.mean(y))**2)
        r_sq = 1- (res_ss2/tot_ss2)
        r_sq = round(r_sq, 2)
            
        from sklearn.metrics import mean_squared_error
        from math import sqrt
        mse = mean_squared_error(y, y_predict)
        rms = sqrt(mse)
        rms = round(rms, 4)

        #fig1 = plt.figure(); ax =fig1.add_subplot(1,1,1)
        #ax.scatter(plot['y'],plot['y_predict'])
        #plt.savefig1('./eco/results/figs/values{}.pdf'.format(name))
        #plt.close
        
        #extract info: eco, qout, r_sq, deg_free (only gets one eco in data)
        resultsa = resultsa.append({'ID': name, 'qout': qout, 'r_sq': r_sq, 'deg_free': footprints, 'rmse': rms, 'r_sq_mean': r_sq_mean}, ignore_index=True)
        #if deg_free>=60:
            #resultsb = resultsb.append({'eco': name2, 'ID': name, 'qout': qout, 'r_sq': r_sq, 'deg_free': deg_free, 'rmse': rms}, ignore_index=True)        
            #export to excel
        resultsa.to_csv(fileout)
            #resultsb.to_csv('./eco/new/results/results_over60.csv')

        #plot the result
        fig = plt.figure(); ax = fig.add_subplot(1,1,1)
        plt.rcParams.update({'font.size':12})
        #plots H_100 on x with I_CD on y
        ax.scatter(plot['median'],plot['mean'])
        #plots IQR
        ax.bar(plot['median'],plot['mean'],width=0, yerr=plot['iqr'])
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
        ax.annotate('q = ' + str(qout[0]), xy=(0.975,0.20), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        ax.annotate('r2 = ' + str(r_sq), xy=(0.975,0.15), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        ax.annotate('RMSE = ' + str(rms),xy=(0.975,0.10), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')   
        ax.annotate('No of footprints = ' + str(footprints),xy=(0.975,0.05), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        plt.savefig(folderout + 'fig{}.pdf'.format(name))
        plt.close


def grid(folderin, fileout, folderout, naming=3, eco_loc=2):
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

    #using 'file' to title plot  
    fileList = glob.glob(folderin + '*.shp')

    #create df for results
    resultsa = pd.DataFrame(columns = ['eco', 'ID', 'qout', 'r_sq', 'deg_free', 'rmse','r_sq_mean'])
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
        footprints = len(df['i_h100'])
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

        if footprints < 50:
            continue
        
        #NEXT STEP. Bin remaining data in order to get mean and IQR of each bin

        #add column with bins 
        final['H_bins']=pd.cut(x=final['i_h100'], bins=np.arange(0, 120+2, 2))

        #now something along the lines of:
        #for bin in HBins find the mean and IQR...
        #first create lists to append mean and IQRs to
        cd_mean = []
        cd_iqr = []
        #Hbin = []
        print(name)
        print(eco)
        HBins = list(np.unique(final['H_bins']))
        for bins in HBins:
            #for each one make a df with just that bin
            new = final.loc[final['H_bins']==bins]
            #get mean and IQR of each bin
            data = new['i_cd'].to_numpy()
            mean = data.mean()
            cd_mean.append(mean)
            q75, q25 = np.percentile (data, [75, 25])
            iqr = q75 - q25
            cd_iqr.append(iqr)
    
        #getting median of bins for mean r2 calculation
        greats = []
        for index,i in final.iterrows():
            great = [i['H_bins'].left + 1] 
            greats.append(great)

    
        final['H_bin'] = greats 
        new1 = final['H_bin'] = final.H_bin.astype(str)
        new2 = new1.str.strip('[]').astype(int)
        final['H_bin1'] = new2
        
        del new, data, q75, q25, new1 
    
        #get median of bins for plotting
        med = [binn.left + 1 for binn in HBins]
        plot = pd.DataFrame({'mean': cd_mean, 'iqr': iqr, 'bins': HBins, 'median': med})
        bin_dict = plot.set_index('median')['mean'].to_dict()
    
        plot_y = []
        for i in final['H_bin1']:
            y = bin_dict[i]
            plot_y.append(y)
            del y
        
        final['plot_y'] = plot_y
     
        #regression 
        def f(x,q):
            return 1- np.exp(-q * x)
    
        x = final['i_h100'].to_numpy()
        y = final['i_cd'].to_numpy() 
        x = np.append(x, [0])
        y = np.append(y, [0])
    
        qout, qcov = curve_fit(f, x, y, 0.04)
        qout = qout.round(decimals=4)
        #calculating mean r2
        residuals = plot_y - f(new2, qout)
        res_ss = np.sum(residuals**2)
        tot_ss = np.sum((plot_y-np.mean(plot_y))**2)
        r_sq_mean = 1 - (res_ss/tot_ss)
        #deg_free = (len(x)-1)
        r_sq_mean = round(r_sq_mean, 2)
        y_predict = f(x, qout)
        
        #calculating r2
        residuals2 = y - f(x, qout)
        res_ss2 = np.sum(residuals2**2)
        tot_ss2 = np.sum((y-np.mean(y))**2)
        r_sq = 1- (res_ss2/tot_ss2)
        r_sq = round(r_sq, 2)
            
        from sklearn.metrics import mean_squared_error
        from math import sqrt
        mse = mean_squared_error(y, y_predict)
        rms = sqrt(mse)
        rms = round(rms, 4)

        #fig1 = plt.figure(); ax =fig1.add_subplot(1,1,1)
        #ax.scatter(plot['y'],plot['y_predict'])
        #plt.savefig1('./eco/results/figs/values{}.pdf'.format(name))
        #plt.close
        
        #extract info: eco, qout, r_sq, deg_free (only gets one eco in data)
        resultsa = resultsa.append({'eco': eco, 'ID': name, 'qout': qout, 'r_sq': r_sq, 'deg_free': footprints, 'rmse': rms, 'r_sq_mean': r_sq_mean}, ignore_index=True)
        #if deg_free>=60:
            #resultsb = resultsb.append({'eco': name2, 'ID': name, 'qout': qout, 'r_sq': r_sq, 'deg_free': deg_free, 'rmse': rms}, ignore_index=True)        
            #export to excel
        resultsa.to_csv(fileout)
            #resultsb.to_csv('./eco/new/results/results_over60.csv')

        #plot the result
        fig = plt.figure(); ax = fig.add_subplot(1,1,1)
        plt.rcParams.update({'font.size':12})
        #plots H_100 on x with I_CD on y
        ax.scatter(plot['median'],plot['mean'])
        #plots IQR
        ax.bar(plot['median'],plot['mean'],width=0, yerr=plot['iqr'])
        #sets title and axis labels
        ax.set_title('ecoregion' + eco + 'in grid no.' + name)
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
        ax.annotate('q = ' + str(qout[0]), xy=(0.975,0.20), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        ax.annotate('r2 = ' + str(r_sq), xy=(0.975,0.15), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        ax.annotate('RMSE = ' + str(rms),xy=(0.975,0.10), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')   
        ax.annotate('No of footprints = ' + str(footprints),xy=(0.975,0.05), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        plt.savefig(folderout + 'fig{}_{}.pdf'.format(eco, name))
        plt.close
        

def grid_date(folderin, fileout, folderout, grid=4, eco_loc=2, datnum=6):
    """
    Function to compute q and provide results (csv) and figures (pdf)
    
    Parameters
    ----------
    
    folderin: string
            Filepath for folder with files ready for analysis
                
    grid: int
          Section of filename to obtain ID (here grid number). Obtained
          by splitting filename by '_' and indexing
          Default = 4

    eco_loc: int
          Section of filename to obtain ecoregion (if applicable). 
          Obtained as with naming
          Default = 2
          
    datnum: int
          Section of filename to obtain date (if applicable). 
          Obtained as with naming
          Default = 6      
          
    fileout: string
           Filepath for results file ending '.csv'
           
    folderout: string
             Filepath for folder to save the figures            
    """

    #using 'file' to title plot  
    fileList = glob.glob(folderin + '*.shp')

    #create df for results
    resultsa = pd.DataFrame(columns = ['eco', 'ID', 'date', 'qout', 'r_sq', 'deg_free', 'rmse','r_sq_mean'])
    #resultsb = pd.DataFrame(columns = ['eco', 'ID', 'qout', 'r_sq', 'deg_free', 'rmse'])

    for file in fileList:
        df = gpd.read_file(file)
        if df.empty:
            continue 
        hd, tl = path.split(file)
        shp_lyr_name = path.splitext(tl)[0]
        name_comp = shp_lyr_name.split('_')
        name = name_comp[grid] 
        eco = name_comp[eco_loc]
        date = name_comp[datnum]
        print('Grid ' + name, ' Eco ' + eco, ' Date ' + date)
        #remove data with H_100 >= 0 prior to logging
        test2 = df[df['i_h100']>=0] 
        footprints = len(df['i_h100'])
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

        if footprints < 50:
            continue
        
        #NEXT STEP. Bin remaining data in order to get mean and IQR of each bin

        #add column with bins 
        final['H_bins']=pd.cut(x=final['i_h100'], bins=np.arange(0, 120+2, 2))

        #now something along the lines of:
        #for bin in HBins find the mean and IQR...
        #first create lists to append mean and IQRs to
        cd_mean = []
        cd_iqr = []
        #Hbin = []
        print(name)
        print(eco)
        HBins = list(np.unique(final['H_bins']))
        for bins in HBins:
            #for each one make a df with just that bin
            new = final.loc[final['H_bins']==bins]
            #get mean and IQR of each bin
            data = new['i_cd'].to_numpy()
            mean = data.mean()
            cd_mean.append(mean)
            q75, q25 = np.percentile (data, [75, 25])
            iqr = q75 - q25
            cd_iqr.append(iqr)
    
        #getting median of bins for mean r2 calculation
        greats = []
        for index,i in final.iterrows():
            great = [i['H_bins'].left + 1] 
            greats.append(great)

    
        final['H_bin'] = greats 
        new1 = final['H_bin'] = final.H_bin.astype(str)
        new2 = new1.str.strip('[]').astype(int)
        final['H_bin1'] = new2
        
        del new, data, q75, q25, new1 
    
        #get median of bins for plotting
        med = [binn.left + 1 for binn in HBins]
        plot = pd.DataFrame({'mean': cd_mean, 'iqr': iqr, 'bins': HBins, 'median': med})
        bin_dict = plot.set_index('median')['mean'].to_dict()
    
        plot_y = []
        for i in final['H_bin1']:
            y = bin_dict[i]
            plot_y.append(y)
            del y
        
        final['plot_y'] = plot_y
     
        #regression 
        def f(x,q):
            return 1- np.exp(-q * x)
    
        x = final['i_h100'].to_numpy()
        y = final['i_cd'].to_numpy() 
        x = np.append(x, [0])
        y = np.append(y, [0])
    
        qout, qcov = curve_fit(f, x, y, 0.04)
        qout = qout.round(decimals=4)
        #calculating mean r2
        residuals = plot_y - f(new2, qout)
        res_ss = np.sum(residuals**2)
        tot_ss = np.sum((plot_y-np.mean(plot_y))**2)
        r_sq_mean = 1 - (res_ss/tot_ss)
        #deg_free = (len(x)-1)
        r_sq_mean = round(r_sq_mean, 2)
        y_predict = f(x, qout)
        
        #calculating r2
        residuals2 = y - f(x, qout)
        res_ss2 = np.sum(residuals2**2)
        tot_ss2 = np.sum((y-np.mean(y))**2)
        r_sq = 1- (res_ss2/tot_ss2)
        r_sq = round(r_sq, 2)
            
        from sklearn.metrics import mean_squared_error
        from math import sqrt
        mse = mean_squared_error(y, y_predict)
        rms = sqrt(mse)
        rms = round(rms, 4)

        #fig1 = plt.figure(); ax =fig1.add_subplot(1,1,1)
        #ax.scatter(plot['y'],plot['y_predict'])
        #plt.savefig1('./eco/results/figs/values{}.pdf'.format(name))
        #plt.close
        
        #extract info: eco, qout, r_sq, deg_free (only gets one eco in data)
        resultsa = resultsa.append({'eco': eco, 'ID': name, 'date': date, 'qout': qout, 'r_sq': r_sq, 'deg_free': footprints, 'rmse': rms, 'r_sq_mean': r_sq_mean}, ignore_index=True)
        #if deg_free>=60:
            #resultsb = resultsb.append({'eco': name2, 'ID': name, 'qout': qout, 'r_sq': r_sq, 'deg_free': deg_free, 'rmse': rms}, ignore_index=True)        
            #export to excel
        resultsa.to_csv(fileout)
            #resultsb.to_csv('./eco/new/results/results_over60.csv')

        #plot the result
        fig = plt.figure(); ax = fig.add_subplot(1,1,1)
        plt.rcParams.update({'font.size':12})
        #plots H_100 on x with I_CD on y
        ax.scatter(plot['median'],plot['mean'])
        #plots IQR
        ax.bar(plot['median'],plot['mean'],width=0, yerr=plot['iqr'])
        #sets title and axis labels
        ax.set_title('ecoregion' + eco + 'in grid no.' + name)
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
        ax.annotate('q = ' + str(qout[0]), xy=(0.975,0.20), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        ax.annotate('r2 = ' + str(r_sq), xy=(0.975,0.15), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        ax.annotate('RMSE = ' + str(rms),xy=(0.975,0.10), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')   
        ax.annotate('No of footprints = ' + str(footprints),xy=(0.975,0.05), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        plt.savefig(folderout + 'fig{}_{}_{}.pdf'.format(eco, name, date))
        plt.close        