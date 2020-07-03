#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 17:27:38 2020

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

ogr.UseExceptions()

def extract_gpkg_layers(folderin, folderout):
    """
    Function to extract layers from geopackage for each of the GEDI beams and create
    a shp file

    Parameters
    ----------
    folderin : string
        filepath for directory with GEDI gpkg files
    
    folderout: string
        filepath for directory for new shapefiles
    
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
        inlayer = openfile.GetLayerByName(beam)
        inLayerDefn = inlayer.GetLayerDefn()
        inLayerRef = inlayer.GetSpatialRef()
 
        beams = ['BEAM0000','BEAM0001','BEAM0010','BEAM0011','BEAM0101','BEAM0110',
                 'BEAM1000','BEAM1011']
        
        for beam in beams:
            inlayer = openfile.GetLayerByName(beam)
            inLayerDefn = inlayer.GetLayerDefn()
            inLayerRef = inlayer.GetSpatialRef()
        
            outpath = folderout + '{}_{}.shp'.format(name, beam)
            out_driver = ogr.GetDriverByName("ESRI Shapefile")
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
    split_files = glob.glob(folderin)

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

    colNms =['elevation_bin0','elevation_lastbin','height_bin0','height_lastbin',
                 'shot_number','solar_azimuth','solar_elevation','latitude_bin0',
                 'latitude_lastbin','longitude_bin0','longitude_lastbin',
                 'master_frac','master_int','omega','pai','pgap_theta',
                 'pgap_theta_error','rg','rhog','rhog_error','rhov','rhov_error',
                 'rossg','rv']
    
    for filename in fileList:
        basename = os.path.splitext(os.path.basename(filename))[0]
        df = gpd.read_file(filename) 
        new = df[np.logical_not(df[column].isin(cat))]
        if new.empty:
            continue
        df2 = new[new['surface_flag']==1] 
        df3 = df2[df2['rh100']>=0]
        df3.drop(colNms, 1, inplace = True)
        df3.to_file(folderout + '{}.shp'.format(basename))
        

def grid_join(base_vec, folderin, folderout):
    """
    Function to join a shapefile to the gedi data. 
    
    Parameters
    ----------
    base_vec: string
          Filepath for shp file to join with gedi data
          
    folderin: string
            Filepath for folder containing gedi files      
          
    folderout: string
             Filepath for folder to contain joined files
             
    """
    filelist = glob.glob(folderin + '*.shp')
    
    for join_vec in filelist:
        name = os.path.splitext(os.path.basename(join_vec))[0]
        # Must have rtree installed - otherwise error "geopandas/tools/sjoin.py"
        # AttributeError: 'NoneType' object has no attribute 'intersection'
        base_gpd_df = geopandas.read_file(base_vec)
        join_gpg_df = geopandas.read_file(join_vec)
    
        join_gpg_df = geopandas.sjoin(base_gpd_df, join_gpg_df, how="inner", op="within")
        join_gpg_df.to_file(folderout + '{}.shp'.format(name))


        
def split_per(folderin, folderout, split_col='join', colNms=['degrade_fl','digital_el',
    'landsat_tr','modis_nonv','modis_no_1','modis_tree','modis_tr_1','beam','cover',
    'num_detect','rh100','sensitivit','stale_retu','surface_fl','l2a_qualit',
    'l2b_qualit','b1','ECO_NAME','Id','join','geometry']):
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
        basename =  os.path.splitext(os.path.basename(filename))[0] 
        name_comp = basename.split('_')
        name = name_comp[2]
        
        df3 = gpd.read_file(filename)
        dfa = df3[df3['surface_flag']==1] 
        df = dfa[dfa['rh100']>=0]
        #df = dfa.astype({split_col: 'int32'}) 
        ecoNames = list(np.unique(df[split_col]))#get list of unique ecoregions    
        
        for eco in ecoNames:
            #create new df with just columns I want
            df2 = gpd.GeoDataFrame(df, columns=colNms)
            ID = str(eco)
            df_eco = df.loc[df2[split_col]==eco, colNms]
            df_eco.to_file(folderout + '/{}_join_{}.shp'.format(name, ID)) 

def sort_files(folderin, folderout, fileout = '/scratch/a.hek4/joinlist.txt'):
    """
    Function to rename shapefiles by 'join' add date column from filename, and 
    create txt file with list of each 'join'
    
    Parameters
    ----------          
    folderin: string
             Filepath for shp files to join based on column
           
    folderout: string
             Filepath for folder to contain joined files 
        
    fileout: string
           Filepath for joinlist text file (which provides a list for next stage)
           Default = '/scratch/a.hek4/joinlist.txt'               
    """     
    filelist = glob.glob(folderin + '*.shp')
    joinlist = []
    for file in filelist:
        basename =  os.path.splitext(os.path.basename(file))[0] 
        name_comp = basename.split('_')
        date = name_comp[0]
        join = name_comp[2] + '_' + name_comp[3]
        df = gpd.read_file(file)
        df['date'] = date
        joinlist.append(join)
        df.to_file(folderout + '{}.shp'.format(basename))
    
    
    with open(fileout, 'wb') as fp:
        pickle.dump(joinlist, fp)
            
    
def join_gedi_files(folderin, folderout, filein = '/scratch/a.hek4/joinlist.txt'):   
    
    
    with open (filein, 'rb') as fp:
        joinlist = pickle.load(fp)
    
    for i in joinlist:
        outputMergefn = folderout + i + '.shp'
        driverName = 'ESRI Shapefile'
        geometryType = ogr.wkbPoint
    
        out_driver = ogr.GetDriverByName(driverName)
        out_ds = out_driver.CreateDataSource(outputMergefn)
        out_layer = out_ds.CreateLayer(outputMergefn, geom_type = geometryType)      
        
        files = glob.glob(folderin + '*_join_' + i + '.shp')
        for file in files:
            ds = ogr.Open(file)
            lyr = ds.GetLayer()
            for feat in lyr:
                out_feat = ogr.Feature(out_layer.GetLayerDefn())
                out_feat.SetGeometry(feat.GetGeometryRef().Clone())
                out_layer.CreateFeature(out_feat)
                #out_feat = None
        out_layer.SyncToDisk()
                

def grid(folderin, fileout, folderout, naming=1, eco_loc=0):
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
        test4 = df[df['rh100']>=0] 
        test2 = test4[test4['rh100']<=120]
        
        #means x is just the h100 data - needs logging to normalise (not skewed) 
        x = test2['rh100']
        
        #create new column in df with log of H_100 
        y = np.log(x)
        test2a = test2.assign(log_rh100 = y)
        
        if test2a.empty:
            continue

        #get quantiles
        a = np.quantile(test2a['log_rh100'],0.95)
        b = np.quantile(test2a['log_rh100'],0.05)

        #remove data outside of 5% quantiles
        test3 = test2a[test2a.log_rh100 >b]
        final = test3[test3.log_rh100 <a]

        if final.empty:
            continue
        del a, b, x, y, test2, test2a, test3, test4
        
        footprints = len(final['rh100'])
        if footprints < 100:
            continue
        
        #NEXT STEP. Bin remaining data in order to get mean and IQR of each bin

        #add column with bins 
        final['H_bins']=pd.cut(x=final['rh100'], bins=np.arange(0, 120+2, 2))

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
            data = new['cover'].to_numpy()
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
    
        x = final['rh100'].to_numpy()
        y = final['cover'].to_numpy() 
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
        ax.set_title('GEDI ecoregion' + eco + 'in grid no.' + name)
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
        