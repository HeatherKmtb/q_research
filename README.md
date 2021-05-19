# q_research
For processing of ICESat GLAS, GEDI and ICESat-2 LiDAR data, to derive q parameter (from simple allometric model) for canopy height to canopy density relationship

This project investigates the relationship between forest canopy height and canopy density for regions across the globe, using a simple allometric model. 
Each module deals with an aspect of the processing.

glas_prep: contains functions for the preparation of the ICESat GLAS data. For merging with .shp and .tif files, and for filtering and sorting the data

gedi_prep: contains functions for the preparation of the GEDI data. For merging with .shp and .tif files, and for filtering and sorting the data

final.py: contains functions for the regression with both ICESat GLAS and GEDI data that has been prepared

boot.py: contains functions for bootstrapping to obtain a standard error of the regression with the ICESat GLAS data that has been prepared

hxcd.py: contains functions for calculating a 'maximum' (mean of the top 10%) canopy desnity and height value for each polygon of ICESat GLAS data
