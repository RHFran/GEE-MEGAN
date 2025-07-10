# ------------------------------------------------------------------------------
# File: main_all_type.py
# ------------------------------------------------------------------------------
# Description:
#     End-to-end GEE-MEGAN BVOC emission processing pipeline. Reads model settings
#     from a namelist file, generates a series of hourly timestamps, and dispatches
#     to the appropriate scale- and data-specific emission routines (global 5 km air
#     temperature, global 5 km skin temperature, global 500 m, or local 10 m/30 m).
#
# Inputs:
#     - namelist_path (str): path to a plain-text configuration file containing keys
#         such as model_type, chem_names, PFTS_year, DEFMAP flags, roi, scale,
#         output_scale, mini_batch, out_dir, out_dir_filename, TemporaryDirectory, etc.
#     - start_date (str): simulation start in 'YYYY-MM-DD HH:MM:SS' format
#     - end_date   (str): simulation end   in 'YYYY-MM-DD HH:MM:SS' format
#     - perturbation_dict (dict, optional): user-defined perturbations for sensitivity runs
#
# Functions:
#     - read_parameters_from_file(file_path) -> dict
#         Parses key = value lines (with optional inline comments) into Python types.
#     - Emporc_global_5km_without_roi(...)
#     - Emporc_global_5km_with_roi(...)
#         Core emission rate calculator for a given ROI, date, hour batch and canopy layers.
#     - Emporc_global_5km_with_roi_ST(...)
#         Variants of the global 5 km model: full-domain, regional ROI, and skin-temperature input.
#     - Emporc_global_500m(...)
#         500 m global emission workflow.
#     - Emporc_local(...)
#         Local 10 m/30 m workflow with fused LAI, dynamic LUCC, and optional ML evaluation.
#     - process_date_range_global_5km[_ST|_500m](date_range, input_variables, year)
#     - process_date_range_local(date_range, input_variables, year)
#     - main(namelist_path, start_date, end_date, perturbation_dict=None)
#         CLI entry point: initializes Earth Engine, reads parameters, builds datetime list,
#         and invokes the correct processing pipeline based on model_type.
#
# Outputs:
#     - Multi-band GeoTIFF files of hourly emission rates for each chemical species,
#       written to <out_dir>/<out_dir_filename>/<TemporaryDirectory>.
#
# Main Functionality:
#     1. Initialize Earth Engine API (authenticate if necessary).
#     2. Read and validate configuration parameters.
#     3. Generate a list of hourly timestamps over the simulation window.
#     4. For each hour (and mini-batch of species), compute emission rate maps using
#        canopy layers, LAI, meteorology, and emission factors.
#     5. Export emission rate images as GeoTIFFs at the specified output resolution.
#
# Dependencies:
#     - ee
#     - geemap
#     - EMPORC, Echo_parameter, canopy_lib, ECMWF_ERA5_Preprocess_lib,
#       gamma_lib, get_DEFMAP, get_local_lucc
#     - os, math, argparse, ast, datetime, functools
# ------------------------------------------------------------------------------

import os
import ee
import geemap
import math
import argparse
import ast
from datetime import datetime, timedelta
from functools import partial
try:
        # ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com') 
        ee.Initialize()  
except Exception as e:
        ee.Authenticate()  
        ee.Initialize()  
# os.chdir(r'/data/Model/M_GEE_MEGAN/GEE_MEGAN_releasev1')

import EMPORC as EMPORC
import Echo_parameter as Echo_parameter
import canopy_lib as canopy
import ECMWF_ERA5_Preprocess_lib as METPreprocess
import gamma_lib as gamma
import get_DEFMAP as get_DEFMAP
import get_local_lucc as get_local_lucc
def Emporc(date,Hour,chem_names,PFTS_ContainPlant,
           TotalPTFS,EFMAPS,roi,scale,outputscale,Layers,mini_batch,
           out_dir,out_dir_filename):
    """
    --------------------------------------------------------------------------
    Function: Emporc
    --------------------------------------------------------------------------
    Parameters:
        - date (str): date in "YY-MM-DD" format
        - Hour (list): list of hour strings based on the zero meridian timezone
        - chem_names (ee.List): Earth Engine list of chemical species names to calculate
        - PFTS_ContainPlant (ee.Image): image containing vegetation-only PFT bands
        - TotalPTFS (ee.Image): image of total PFT proportions
        - EFMAPS (ee.Image): emission factor maps for each species
        - roi (ee.Geometry): region of interest for clipping images
        - scale (int): processing resolution in meters 
        - outputscale (int): final output image resolution in meters
        - Layers (int): number of canopy layers for ER calculation
        - mini_batch (int): batch size of species per calculation to avoid server limits
        - out_dir (str): base directory path for output files
        - out_dir_filename (str): name of subdirectory for storing outputs

    Returns:
        - None: saves emission rate (ER) images to disk
            - Results are exported as GeoTIFFs in the specified output directory
           
    Description:
        This function computes emission rate (ER) images for specified chemicals by
        combining PFT distributions, emission factors, meteorological inputs, and canopy data.
        It processes data hourly in batches and exports the results as TIFF files.
        The function creates temporary directories as needed and writes outputs to disk.

    Example:
        ```python
        - example call
            - result = Emporc("2022-06-06", ["0"], chem_names,
                              PFTS_ContainPlant, TotalPTFS, EFMAPS, roi,
                              500, 1000, 3, 2, "/data", "output_folder")
        ```
    """
    # Prepare parameters
    import ee
    import Echo_parameter as Echo_parameter
    COMPOUND_EMISSION_FACTORS =Echo_parameter.COMPOUND_EMISSION_FACTORS
    PDT_LOS_CP = Echo_parameter.PDT_LOS_CP
    LD_FCT = Echo_parameter.LD_FCT
    REA_SPC = Echo_parameter.REA_SPC
    REL_EM_ACT = Echo_parameter.REL_EM_ACT
    RHO_FCT =  ee.List(PDT_LOS_CP.get('RHO_FCT'))
    LDF_FCT = ee.List(LD_FCT.get('LDF_FCT'))
    # Convert EE List to Python list for chemical names
    chem_names_list = list(chem_names.getInfo())
    METbandNameList = ["surface_pressure","dewpoint_temperature_2m","temperature_2m",
                "u_component_of_wind_10m","v_component_of_wind_10m"]
    METcollectionId = "ECMWF/ERA5_LAND/HOURLY"
    date_str = date 
    # LAIcollectionName = "MODIS/061/MCD15A3H" #间隔周期为4天
    LAIcollectionName = 'projects/sat-io/open-datasets/BU_LAI_FPAR/wgs_5km_8d'
    full_out_dir = os.path.join(out_dir,out_dir_filename)
    print(out_dir)
    print(full_out_dir)
    PFTSNamesList = PFTS_ContainPlant.bandNames()
    
    # Get LAI
    
    lai_p, lai_c = EMPORC.find_LAI_new(date_str, LAIcollectionName)

    # Get original LAI image resolution
    original_scale = lai_p.projection().nominalScale().getInfo()
    if scale <=500:
        pass
    else:
        # Compute multiplier for resolution adjustment
        multiplier = math.ceil(scale/original_scale)
        resolution = multiplier*original_scale
        print(f'final resolution is {multiplier*original_scale}')
        lai_p = EMPORC.reduce_mean_image_resolution(lai_p,multiplier)
        lai_c = EMPORC.reduce_mean_image_resolution(lai_c,multiplier)

    lai_p = lai_p.clip(roi).multiply(0.1) # User's guide lai * 0.1
    lai_c = lai_c.clip(roi).multiply(0.1)
    # Retrieve meteorological datasets
    METDict = EMPORC.get_MET(date,METcollectionId,METbandNameList)
    surface_pressured_Dataset = METDict[METbandNameList[0]]
    dewpoint_temperature_2m_Dataset = METDict[METbandNameList[1]]
    temperature_2m_Dataset = METDict[METbandNameList[2]]
    u_wind_10m_Dataset  = METDict[METbandNameList[3]]
    v_wind_10m_Dataset = METDict[METbandNameList[4]] 
    relativeHumidityDataset = METPreprocess.calculateRelativeHumidity(dewpoint_temperature_2m_Dataset, temperature_2m_Dataset)
    windSpeed_Dataset = METPreprocess.calculateWindSpeed(u_wind_10m_Dataset, v_wind_10m_Dataset)

    surface_pressured_size = surface_pressured_Dataset.size()
    surface_pressured_List = surface_pressured_Dataset.toList(surface_pressured_size)
    relativeHumidityDataset_Size =  relativeHumidityDataset.size()
    relativeHumidityDataset_List = relativeHumidityDataset.toList(relativeHumidityDataset_Size)
    windSpeed_Dataset_Size = windSpeed_Dataset.size()
    windSpeed_Dataset_List =  windSpeed_Dataset.toList(windSpeed_Dataset_Size)
    temperature_2m_Dataset_size = temperature_2m_Dataset.size()
    temperature_2m_Dataset_List =  temperature_2m_Dataset.toList(temperature_2m_Dataset_size)
    # Process solar radiation data
    collectionId = "ECMWF/ERA5_LAND/HOURLY"
    bandName = 'surface_solar_radiation_downwards'
    currentSolarDataset = METPreprocess.filterByDate(date, METcollectionId, bandName)
    previousDay = ee.Date(date).advance(-1, 'day').format('YYYY-MM-dd')
    previousDaySolarDataset =  METPreprocess.filterByDate(previousDay, METcollectionId, bandName)
    instantaneousRadiationCollection = METPreprocess.calcInstantaneousSolarRadiation(currentSolarDataset, previousDaySolarDataset)
    D_solar_radition = instantaneousRadiationCollection.reduce(ee.Reducer.mean())
    solarDataSetSize = instantaneousRadiationCollection.size()
    solardatDataList = instantaneousRadiationCollection.toList(solarDataSetSize)

    lon_lat_image = METPreprocess.create_lon_lat_image(lai_p, roi)
    lat = lon_lat_image.select('latitude')
    TSTLEN = 30
    # Loop over each specified hour
    for i in range(len(Hour)):
        utcTimeHours = int(Hour[i])
        localTimeImage = METPreprocess.computeLocalTime(lon_lat_image, utcTimeHours,date)
        Day_Image = localTimeImage.select('julian_day')
        Hour_Image = localTimeImage.select('local_time')
        Tc = ee.Image(temperature_2m_Dataset_List.get(utcTimeHours)).clip(roi)
        T24 = ee.Image(temperature_2m_Dataset.reduce(ee.Reducer.mean())).clip(roi)
        T240 = T24
        Wind = ee.Image(windSpeed_Dataset_List.get(utcTimeHours)).clip(roi)
        Humidity = ee.Image(relativeHumidityDataset_List.get(utcTimeHours)).clip(roi)
        Pres = ee.Image(surface_pressured_List.get(utcTimeHours)).clip(roi)
        solar_ = ee.Image(solardatDataList.get(utcTimeHours)).clip(roi)
        meanSolar = solar_.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi).values().get(0) 
        solar = solar_.unmask(ee.Image.constant(meanSolar))
        DI = ee.Image.constant(0)
        PPFD = solar.where(solar.lte(0), 0)
        PPFD24 = solar.where(D_solar_radition.lte(0), 0).clip(roi)
        PPFD240 = PPFD24 
        TemporaryDirectory = os.path.join(full_out_dir,'TemporaryDirectory')
        if os.path.exists(TemporaryDirectory):
            print('The path [{}] exists'.format(TemporaryDirectory))
            
        else:
            os.makedirs(TemporaryDirectory, exist_ok=True)

        for j in range(0, len(chem_names_list), mini_batch):
                
            mini_batch_chem_names = ee.List(chem_names_list[j:j+mini_batch])
            ER_i = EMPORC.ER_calculate(TSTLEN ,Day_Image, Hour_Image, lat,
                    Tc, T24, T240,PPFD ,PPFD24,PPFD240,
                    Wind, Humidity, Pres, DI, lai_c,lai_p,Layers,
                    TotalPTFS.clip(roi),EFMAPS.clip(roi),mini_batch_chem_names,PFTS_ContainPlant,PFTSNamesList)
            size_mini_batch = len(chem_names_list[j:j+mini_batch])
            num_lens = len(str(len(chem_names_list)))
            print(f'save {mini_batch_chem_names.getInfo()}' )
            # Temporaryfilename = 'ER_{}.tif'.format(j,'0{}'.format(num_lens))
            if len(chem_names_list)<=1:
                Temporaryfilename = f'GEE_MEGAN_ER_{mini_batch_chem_names.getInfo()[0]}_{date_str}_{utcTimeHours:02}.tif'
            else:
                Temporaryfilename = f'GEE_MEGAN_ER_{j:02}_{mini_batch:02}_{date_str}_{utcTimeHours:02}.tif'
            filename = os.path.join(TemporaryDirectory , Temporaryfilename)
            ER = {}
            for k in range(size_mini_batch):
                ER[chem_names_list[j:j+mini_batch][k]] = ee.Image(ER_i.get(k))
            ER_images = list(ER.values())
            ER_image = ee.Image.cat(ER_images)
            # ER_image_int = ER_image.multiply(1000).toInt()
            # ER_scale = ER_image.projection().nominalScale().getInfo()
            # print(f'ER scale is {ER_scale}')
            geemap.ee_export_image(ee.Image(ER_image), filename=filename, scale=outputscale, region=roi, file_per_band=False)


def Emporc_global_5km_without_roi(date,Hour,chem_names,PFTS_ContainPlant,
           TotalPTFS,EFMAPS,scale,outputscale,Layers,mini_batch,
           out_dir,out_dir_filename,TemporaryDirectory_in,input_variables):
    """
    --------------------------------------------------------------------------
    Function: Emporc_global_5km_without_roi
    --------------------------------------------------------------------------
    Parameters:
        - date (str): date in "YY-MM-DD" format
        - Hour (list): list of hour strings based on the zero meridian timezone
        - chem_names (ee.List): Earth Engine list of chemical species names to calculate
        - PFTS_ContainPlant (ee.Image): image containing vegetation-only PFT bands
        - TotalPTFS (ee.Image): image of total PFT proportions
        - EFMAPS (ee.Image): emission factor maps for each species
        - roi (ee.Geometry): region of interest for clipping images
        - scale (int): processing resolution in meters 
        - outputscale (int): final output image resolution in meters
        - Layers (int): number of canopy layers for ER calculation
        - mini_batch (int): batch size of species per calculation to avoid server limits
        - out_dir (str): base directory path for output files
        - out_dir_filename (str): name of subdirectory for storing outputs


    Returns:
        - None: saves emission rate images to disk.
            - Outputs are exported as GeoTIFF files in specified directory.

    Description:
        This function computes global 5km emission rate (ER) data without using a ROI.

    Example:
        ```python
        - example call
            - result = Emporc_global_5km_without_roi(
                  "2022-06-06", ["0"], chem_names,
                  PFTS_ContainPlant, TotalPTFS, EFMAPS,
                  500, 1000, 3, 1,
                  "/data", "output_folder", "TempDir")
        ```
    """

    import ee
    import Echo_parameter as Echo_parameter
    COMPOUND_EMISSION_FACTORS =Echo_parameter.COMPOUND_EMISSION_FACTORS
    PDT_LOS_CP = Echo_parameter.PDT_LOS_CP
    LD_FCT = Echo_parameter.LD_FCT
    REA_SPC = Echo_parameter.REA_SPC
    REL_EM_ACT = Echo_parameter.REL_EM_ACT
    RHO_FCT =  ee.List(PDT_LOS_CP.get('RHO_FCT'))
    LDF_FCT = ee.List(LD_FCT.get('LDF_FCT'))
    chem_names_list = list(chem_names.getInfo())
    METbandNameList = ["surface_pressure","dewpoint_temperature_2m","temperature_2m",
                "u_component_of_wind_10m","v_component_of_wind_10m"]
    METcollectionId = "ECMWF/ERA5_LAND/HOURLY"
    date_str = date 

    LAIcollectionName = 'projects/sat-io/open-datasets/BU_LAI_FPAR/wgs_5km_8d'
    full_out_dir = os.path.join(out_dir,out_dir_filename)
    # print(out_dir)
    # print(full_out_dir)
    PFTSNamesList = PFTS_ContainPlant.bandNames()
    # print(date_str)
    

    year = int(date_str[:4])
    # 'projects/sat-io/open-datasets/BU_LAI_FPAR/wgs_5km_8d'  only to 2022

    if year > 2022:
        new_date_str = '2022' + date_str[4:]
    else:
        new_date_str = date_str

    print(new_date_str)
    lai_p, lai_c = EMPORC.find_LAI_new(new_date_str, LAIcollectionName)

    # Get original LAI resolution and downscale if needed
    original_scale = lai_p.projection().nominalScale().getInfo()
    if scale <=500:
        pass
    else:
        multiplier = math.ceil(scale/original_scale)
        resolution = multiplier*original_scale
        print(f'final resolution is {multiplier*original_scale}')
        lai_p = EMPORC.reduce_mean_image_resolution(lai_p,multiplier)
        lai_c = EMPORC.reduce_mean_image_resolution(lai_c,multiplier)

    lai_p = lai_p.multiply(0.1) # User's guide lai * 0.1
    lai_c = lai_c.multiply(0.1)

    lAI_v = input_variables['lAI_v']
    if lAI_v == "True":
        print("Using LAI_v adjustment (LAI / VCF)")
        VCF_image = EMPORC.find_VCF_image(date_str, dataset="MODIS/061/MOD13C1")
        lai_p  = EMPORC.adjust_LAI_by_VCF(lai_p , VCF_image)
        lai_c  = EMPORC.adjust_LAI_by_VCF(lai_c , VCF_image)

    # Fetch meteorological datasets
    METDict = EMPORC.get_MET(date,METcollectionId,METbandNameList)
    surface_pressured_Dataset = METDict[METbandNameList[0]]
    dewpoint_temperature_2m_Dataset = METDict[METbandNameList[1]]
    temperature_2m_Dataset = METDict[METbandNameList[2]]
    u_wind_10m_Dataset  = METDict[METbandNameList[3]]
    v_wind_10m_Dataset = METDict[METbandNameList[4]]
    relativeHumidityDataset = METPreprocess.calculate_density(dewpoint_temperature_2m_Dataset, temperature_2m_Dataset,surface_pressured_Dataset)
    windSpeed_Dataset = METPreprocess.calculateWindSpeed(u_wind_10m_Dataset, v_wind_10m_Dataset)
    surface_pressured_size = surface_pressured_Dataset.size()
    surface_pressured_List = surface_pressured_Dataset.toList(surface_pressured_size)
    relativeHumidityDataset_Size =  relativeHumidityDataset.size()
    relativeHumidityDataset_List = relativeHumidityDataset.toList(relativeHumidityDataset_Size)
    windSpeed_Dataset_Size = windSpeed_Dataset.size()
    windSpeed_Dataset_List =  windSpeed_Dataset.toList(windSpeed_Dataset_Size)
    temperature_2m_Dataset_size = temperature_2m_Dataset.size()
    temperature_2m_Dataset_List =  temperature_2m_Dataset.toList(temperature_2m_Dataset_size)
    # Process solar radiation data
    collectionId = "ECMWF/ERA5_LAND/HOURLY"
    bandName = 'surface_solar_radiation_downwards'
    currentSolarDataset = METPreprocess.filterByDate(date, METcollectionId, bandName)
    previousDay = ee.Date(date).advance(-1, 'day').format('YYYY-MM-dd')
    previousDaySolarDataset =  METPreprocess.filterByDate(previousDay, METcollectionId, bandName)
    instantaneousRadiationCollection = METPreprocess.calcInstantaneousSolarRadiation(currentSolarDataset, previousDaySolarDataset)
    D_solar_radition = instantaneousRadiationCollection.reduce(ee.Reducer.mean())
    solarDataSetSize = instantaneousRadiationCollection.size()
    solardatDataList = instantaneousRadiationCollection.toList(solarDataSetSize)

    lon_lat_image = METPreprocess.create_lon_lat_image_without_roi(lai_p)
    lat = lon_lat_image.select('latitude')
    TSTLEN = 30
    # Loop over each specified hour
    for i in range(len(Hour)):
        utcTimeHours = int(Hour[i])
        localTimeImage = METPreprocess.computeLocalTime(lon_lat_image, utcTimeHours,date)
        Day_Image = localTimeImage.select('julian_day')
        Hour_Image = localTimeImage.select('local_time')
        Tc = ee.Image(temperature_2m_Dataset_List.get(utcTimeHours))
        T24 = ee.Image(temperature_2m_Dataset.reduce(ee.Reducer.mean()))
        T240 = T24
        Wind = ee.Image(windSpeed_Dataset_List.get(utcTimeHours))
        Humidity = ee.Image(relativeHumidityDataset_List.get(utcTimeHours))
        Pres = ee.Image(surface_pressured_List.get(utcTimeHours))
        solar = ee.Image(solardatDataList.get(utcTimeHours))
        DI = ee.Image.constant(0)
        PPFD = solar.where(solar.lte(0), 0)
        PPFD24 = solar.where(D_solar_radition.lte(0), 0)
        PPFD240 = PPFD24 
        TemporaryDirectory = os.path.join(full_out_dir,TemporaryDirectory_in)
        if os.path.exists(TemporaryDirectory):
            print('The path [{}] exists'.format(TemporaryDirectory))
            
        else:
            os.makedirs(TemporaryDirectory, exist_ok=True)

        for j in range(0, len(chem_names_list), mini_batch):
                
            mini_batch_chem_names = ee.List(chem_names_list[j:j+mini_batch])
            ER_i = EMPORC.ER_calculate(TSTLEN ,Day_Image, Hour_Image, lat,
                    Tc, T24, T240,PPFD ,PPFD24,PPFD240,
                    Wind, Humidity, Pres, DI, lai_c,lai_p,Layers,
                    TotalPTFS,EFMAPS,mini_batch_chem_names,PFTS_ContainPlant,PFTSNamesList)
            size_mini_batch = len(chem_names_list[j:j+mini_batch])
            num_lens = len(str(len(chem_names_list)))
            print(f'save {mini_batch_chem_names.getInfo()}' )
            if len(chem_names_list)<=1:
                Temporaryfilename = f'GEE_MEGAN_ER_{mini_batch_chem_names.getInfo()[0]}_{date_str}_{utcTimeHours:02}.tif'
            else:
                Temporaryfilename = f'GEE_MEGAN_ER_{j:02}_{mini_batch:02}_{date_str}_{utcTimeHours:02}.tif'
            filename = os.path.join(TemporaryDirectory , Temporaryfilename)
            ER = {}
            for k in range(size_mini_batch):
                ER[chem_names_list[j:j+mini_batch][k]] = ee.Image(ER_i.get(k))

            ER_images = list(ER.values())
            ER_image = ee.Image.cat(ER_images)
            geemap.ee_export_image(ee.Image(ER_image), filename=filename, scale=outputscale,file_per_band=False)


def Emporc_global_5km_with_roi(date,Hour,chem_names,PFTS_ContainPlant,
           TotalPTFS,EFMAPS,roi,scale,outputscale,Layers,mini_batch,
           out_dir,out_dir_filename,TemporaryDirectory_in,input_variables):
    """
    --------------------------------------------------------------------------
    Function: Emporc_global_5km_with_roi
    --------------------------------------------------------------------------
    Parameters:
        - date (str): date in "YYYY-MM-DD" format.
        - Hour (list): list of hour strings based on zero meridian.
        - chem_names (ee.List): list of chemical species to calculate.
        - PFTS_ContainPlant (ee.Image): vegetation-only PFT bands image.
        - TotalPTFS (ee.Image): composite PFT proportions image.
        - EFMAPS (ee.Image): emission factor maps image.
        - roi (ee.Geometry): region of interest for clipping images.
        - scale (int): processing resolution in meters.
        - outputscale (int): final output image resolution in meters.
        - Layers (int): number of canopy layers for ER calculation.
        - mini_batch (int): batch size of species per GEE calculation.
        - out_dir (str): base directory path for outputs.
        - out_dir_filename (str): subdirectory name for outputs.
        - TemporaryDirectory_in (str): name of temporary directory inside output folder.

    Returns:
        - None: saves emission rate images to disk.
            - Outputs are exported as GeoTIFF files in specified directory.

    Description:
        This function computes global 5km emission rate (ER) data using a specified ROI.
        It retrieves LAI and meteorological data, processes hourly species batches within the region,
        and exports the resulting ER images as GeoTIFFs. Temporary directories are managed automatically.

    Example:
        ```python
        - example call
            - result = Emporc_global_5km_with_roi(
                  "2022-06-06", ["0"], chem_names,
                  PFTS_ContainPlant, TotalPTFS, EFMAPS, roi,
                  500, 1000, 3, 2,
                  "/data", "output_folder", "TempDir")
        ```
    """
    # parameter definitions
    import ee
    import Echo_parameter as Echo_parameter
    COMPOUND_EMISSION_FACTORS =Echo_parameter.COMPOUND_EMISSION_FACTORS
    PDT_LOS_CP = Echo_parameter.PDT_LOS_CP
    LD_FCT = Echo_parameter.LD_FCT
    REA_SPC = Echo_parameter.REA_SPC
    REL_EM_ACT = Echo_parameter.REL_EM_ACT
    RHO_FCT =  ee.List(PDT_LOS_CP.get('RHO_FCT'))
    LDF_FCT = ee.List(LD_FCT.get('LDF_FCT'))
    chem_names_list = list(chem_names.getInfo())
    METbandNameList = ["surface_pressure","dewpoint_temperature_2m","temperature_2m",
                "u_component_of_wind_10m","v_component_of_wind_10m"]
    METcollectionId = "ECMWF/ERA5_LAND/HOURLY"
    date_str = date 
    lAI_v  = input_variables['lAI_v']
    LAIcollectionName = 'projects/sat-io/open-datasets/BU_LAI_FPAR/wgs_5km_8d'
    full_out_dir = os.path.join(out_dir,out_dir_filename)
    print(out_dir)
    print(full_out_dir)
    PFTSNamesList = PFTS_ContainPlant.bandNames()
    
    lai_p, lai_c = EMPORC.find_LAI_new(date_str, LAIcollectionName)

    original_scale = lai_p.projection().nominalScale().getInfo()
    if scale <=500:
        pass
    else:
        multiplier = math.ceil(scale/original_scale)
        resolution = multiplier*original_scale
        print(f'final resolution is {multiplier*original_scale}')
        lai_p = EMPORC.reduce_mean_image_resolution(lai_p,multiplier)
        lai_c = EMPORC.reduce_mean_image_resolution(lai_c,multiplier)
    lai_p = lai_p.clip(roi).multiply(0.1) # User's guide lai * 0.1
    lai_c = lai_c.clip(roi).multiply(0.1)

    lAI_v = input_variables['lAI_v']
    if lAI_v == "True":
        print("Using LAI_v adjustment (LAI / VCF)")
        VCF_image = EMPORC.find_VCF_image(date_str, dataset="MODIS/061/MOD13C1")
        lai_p  = EMPORC.adjust_LAI_by_VCF(lai_p , VCF_image)
        lai_c  = EMPORC.adjust_LAI_by_VCF(lai_c , VCF_image)

    METDict = EMPORC.get_MET(date,METcollectionId,METbandNameList)
    surface_pressured_Dataset = METDict[METbandNameList[0]]
    dewpoint_temperature_2m_Dataset = METDict[METbandNameList[1]]
    temperature_2m_Dataset = METDict[METbandNameList[2]]
    u_wind_10m_Dataset  = METDict[METbandNameList[3]]
    v_wind_10m_Dataset = METDict[METbandNameList[4]]

    relativeHumidityDataset = METPreprocess.calculateRelativeHumidity(dewpoint_temperature_2m_Dataset, temperature_2m_Dataset)
    windSpeed_Dataset = METPreprocess.calculateWindSpeed(u_wind_10m_Dataset, v_wind_10m_Dataset)
  
    surface_pressured_size = surface_pressured_Dataset.size()
    surface_pressured_List = surface_pressured_Dataset.toList(surface_pressured_size)
    relativeHumidityDataset_Size =  relativeHumidityDataset.size()
    relativeHumidityDataset_List = relativeHumidityDataset.toList(relativeHumidityDataset_Size)
    windSpeed_Dataset_Size = windSpeed_Dataset.size()
    windSpeed_Dataset_List =  windSpeed_Dataset.toList(windSpeed_Dataset_Size)
    temperature_2m_Dataset_size = temperature_2m_Dataset.size()
    temperature_2m_Dataset_List =  temperature_2m_Dataset.toList(temperature_2m_Dataset_size)

    collectionId = "ECMWF/ERA5_LAND/HOURLY"
    bandName = 'surface_solar_radiation_downwards'
    currentSolarDataset = METPreprocess.filterByDate(date, METcollectionId, bandName)
    previousDay = ee.Date(date).advance(-1, 'day').format('YYYY-MM-dd')
    previousDaySolarDataset =  METPreprocess.filterByDate(previousDay, METcollectionId, bandName)
    instantaneousRadiationCollection = METPreprocess.calcInstantaneousSolarRadiation(currentSolarDataset, previousDaySolarDataset)
    D_solar_radition = instantaneousRadiationCollection.reduce(ee.Reducer.mean())
    solarDataSetSize = instantaneousRadiationCollection.size()
    solardatDataList = instantaneousRadiationCollection.toList(solarDataSetSize)

    lon_lat_image = METPreprocess.create_lon_lat_image(lai_p, roi)
    lat = lon_lat_image.select('latitude')
    TSTLEN = 30

    for i in range(len(Hour)):
        utcTimeHours = int(Hour[i])
        localTimeImage = METPreprocess.computeLocalTime(lon_lat_image, utcTimeHours,date)
        Day_Image = localTimeImage.select('julian_day')
        Hour_Image = localTimeImage.select('local_time')
        Tc = ee.Image(temperature_2m_Dataset_List.get(utcTimeHours)).clip(roi)
        T24 = ee.Image(temperature_2m_Dataset.reduce(ee.Reducer.mean())).clip(roi)
        T240 = T24
        Wind = ee.Image(windSpeed_Dataset_List.get(utcTimeHours)).clip(roi)
        Humidity = ee.Image(relativeHumidityDataset_List.get(utcTimeHours)).clip(roi)
        Pres = ee.Image(surface_pressured_List.get(utcTimeHours)).clip(roi)
        solar_ = ee.Image(solardatDataList.get(utcTimeHours)).clip(roi)

        meanSolar = solar_.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi).values().get(0) 
        solar = solar_.unmask(ee.Image.constant(meanSolar))
        DI = ee.Image.constant(0)
        PPFD = solar.where(solar.lte(0), 0)
        PPFD24 = solar.where(D_solar_radition.lte(0), 0).clip(roi)
        PPFD240 = PPFD24 
        TemporaryDirectory = os.path.join(full_out_dir,TemporaryDirectory_in)
        if os.path.exists(TemporaryDirectory):
            print('The path [{}] exists'.format(TemporaryDirectory))
            
        else:
            os.makedirs(TemporaryDirectory, exist_ok=True)

        for j in range(0, len(chem_names_list), mini_batch):
                
            mini_batch_chem_names = ee.List(chem_names_list[j:j+mini_batch])
            ER_i = EMPORC.ER_calculate(TSTLEN ,Day_Image, Hour_Image, lat,
                    Tc, T24, T240,PPFD ,PPFD24,PPFD240,
                    Wind, Humidity, Pres, DI, lai_c,lai_p,Layers,
                    TotalPTFS.clip(roi),EFMAPS.clip(roi),mini_batch_chem_names,PFTS_ContainPlant,PFTSNamesList)
            size_mini_batch = len(chem_names_list[j:j+mini_batch])

            num_lens = len(str(len(chem_names_list)))
            print(f'save {mini_batch_chem_names.getInfo()}' )
  
            if len(chem_names_list)<=1:
                Temporaryfilename = f'GEE_MEGAN_ER_{mini_batch_chem_names.getInfo()[0]}_{date_str}_{utcTimeHours:02}.tif'
            else:
                Temporaryfilename = f'GEE_MEGAN_ER_{j:02}_{mini_batch:02}_{date_str}_{utcTimeHours:02}.tif'
            filename = os.path.join(TemporaryDirectory , Temporaryfilename)
            ER = {}
            for k in range(size_mini_batch):
                ER[chem_names_list[j:j+mini_batch][k]] = ee.Image(ER_i.get(k))

            ER_images = list(ER.values())
            ER_image = ee.Image.cat(ER_images)
            ER_image = ee.Image.cat(ER_images).unmask(0)
 
            geemap.ee_export_image(ee.Image(ER_image), filename=filename, scale=outputscale, region=roi,crs='EPSG:4326', file_per_band=False)

def Emporc_global_5km_with_roi_ST(date,Hour,chem_names,PFTS_ContainPlant,
           TotalPTFS,EFMAPS,roi,scale,outputscale,Layers,mini_batch,
           out_dir,out_dir_filename,TemporaryDirectory_in,input_variables):
    """
    --------------------------------------------------------------------------
    Function: Emporc_global_5km_with_roi_ST
    --------------------------------------------------------------------------
    Parameters:
        - date (str): date in "YYYY-MM-DD" format.
        - Hour (list): list of hour strings based on zero meridian.
        - chem_names (ee.List): list of chemical species to calculate.
        - PFTS_ContainPlant (ee.Image): vegetation-only PFT bands image.
        - TotalPTFS (ee.Image): composite PFT proportions image.
        - EFMAPS (ee.Image): emission factor maps image.
        - roi (ee.Geometry): region of interest for clipping images.
        - scale (int): processing resolution in meters.
        - outputscale (int): final output image resolution in meters.
        - Layers (int): number of canopy layers for ER calculation.
        - mini_batch (int): batch size of species per GEE calculation.
        - out_dir (str): base directory path for outputs.
        - out_dir_filename (str): subdirectory name for outputs.
        - TemporaryDirectory_in (str): name of temporary directory inside output folder.

    Returns:
        - None: saves emission rate images to disk.
            - Outputs are exported as GeoTIFF files in specified directory.

    Description:
        This function computes global 5km emission rate (ER) data using a specified ROI.
        temperature (skin temperature) instead of 2-meter air temperature.

    Example:
        ```python
        - example call
            - result = Emporc_global_5km_with_roi_ST(
                  "2022-06-06", ["0"], chem_names,
                  PFTS_ContainPlant, TotalPTFS, EFMAPS, roi,
                  5000, 10000, 5, 1,
                  "/data", "output_folder", "TempDir")
        ```
    """
    # parameter definitions
    import ee
    import Echo_parameter as Echo_parameter
    COMPOUND_EMISSION_FACTORS =Echo_parameter.COMPOUND_EMISSION_FACTORS
    PDT_LOS_CP = Echo_parameter.PDT_LOS_CP
    LD_FCT = Echo_parameter.LD_FCT
    REA_SPC = Echo_parameter.REA_SPC
    REL_EM_ACT = Echo_parameter.REL_EM_ACT
    RHO_FCT =  ee.List(PDT_LOS_CP.get('RHO_FCT'))
    LDF_FCT = ee.List(LD_FCT.get('LDF_FCT'))
    chem_names_list = list(chem_names.getInfo())
    
    METbandNameList = ["surface_pressure","dewpoint_temperature_2m","temperature_2m",
                "u_component_of_wind_10m","v_component_of_wind_10m","skin_temperature"]
    METcollectionId = "ECMWF/ERA5_LAND/HOURLY"
    date_str = date 

    LAIcollectionName = 'projects/sat-io/open-datasets/BU_LAI_FPAR/wgs_5km_8d'
    full_out_dir = os.path.join(out_dir,out_dir_filename)
    print(out_dir)
    print(full_out_dir)
    PFTSNamesList = PFTS_ContainPlant.bandNames()
    
    print(date_str)
    
    year = int(date_str[:4])

    if year > 2022:
        new_date_str = '2022' + date_str[4:]
    else:
        new_date_str = date_str
    

    print(new_date_str)
    lai_p, lai_c = EMPORC.find_LAI_new(new_date_str, LAIcollectionName)

    original_scale = lai_p.projection().nominalScale().getInfo()
    if scale <=500:
        pass
    else:
        multiplier = math.ceil(scale/original_scale)
        resolution = multiplier*original_scale
        print(f'final resolution is {multiplier*original_scale}')
        lai_p = EMPORC.reduce_mean_image_resolution(lai_p,multiplier)
        lai_c = EMPORC.reduce_mean_image_resolution(lai_c,multiplier)

    lai_p = lai_p.multiply(0.1) # User's guide lai * 0.1
    lai_c = lai_c.multiply(0.1)

    lAI_v = input_variables['lAI_v']
    if lAI_v == "True":
        print("Using LAI_v adjustment (LAI / VCF)")
        VCF_image = EMPORC.find_VCF_image(date_str, dataset="MODIS/061/MOD13C1")
        lai_p  = EMPORC.adjust_LAI_by_VCF(lai_p , VCF_image)
        lai_c  = EMPORC.adjust_LAI_by_VCF(lai_c , VCF_image)

    METDict = EMPORC.get_MET(date,METcollectionId,METbandNameList)
    surface_pressured_Dataset = METDict[METbandNameList[0]]
    dewpoint_temperature_2m_Dataset = METDict[METbandNameList[1]]
    temperature_2m_Dataset = METDict[METbandNameList[2]]
    u_wind_10m_Dataset  = METDict[METbandNameList[3]]
    v_wind_10m_Dataset = METDict[METbandNameList[4]]
    temperature_Skin_Dataset = METDict[METbandNameList[5]]

    relativeHumidityDataset = METPreprocess.calculate_density(dewpoint_temperature_2m_Dataset, temperature_2m_Dataset,surface_pressured_Dataset)
    windSpeed_Dataset = METPreprocess.calculateWindSpeed(u_wind_10m_Dataset, v_wind_10m_Dataset)

    surface_pressured_size = surface_pressured_Dataset.size()
    surface_pressured_List = surface_pressured_Dataset.toList(surface_pressured_size)
    relativeHumidityDataset_Size =  relativeHumidityDataset.size()
    relativeHumidityDataset_List = relativeHumidityDataset.toList(relativeHumidityDataset_Size)
    windSpeed_Dataset_Size = windSpeed_Dataset.size()
    windSpeed_Dataset_List =  windSpeed_Dataset.toList(windSpeed_Dataset_Size)
    ## replaced here with skin temperature
    temperature_2m_Dataset_size =  temperature_Skin_Dataset.size() 
    temperature_2m_Dataset_List =  temperature_Skin_Dataset.toList(temperature_2m_Dataset_size)

    collectionId = "ECMWF/ERA5_LAND/HOURLY"
    bandName = 'surface_solar_radiation_downwards'
    currentSolarDataset = METPreprocess.filterByDate(date, METcollectionId, bandName)
    previousDay = ee.Date(date).advance(-1, 'day').format('YYYY-MM-dd')
    previousDaySolarDataset =  METPreprocess.filterByDate(previousDay, METcollectionId, bandName)
    instantaneousRadiationCollection = METPreprocess.calcInstantaneousSolarRadiation(currentSolarDataset, previousDaySolarDataset)
    D_solar_radition = instantaneousRadiationCollection.reduce(ee.Reducer.mean())
    solarDataSetSize = instantaneousRadiationCollection.size()
    solardatDataList = instantaneousRadiationCollection.toList(solarDataSetSize)

    lon_lat_image = METPreprocess.create_lon_lat_image(lai_p, roi)
    lat = lon_lat_image.select('latitude')
    TSTLEN = 30

    for i in range(len(Hour)):
        utcTimeHours = int(Hour[i])
        localTimeImage = METPreprocess.computeLocalTime(lon_lat_image, utcTimeHours,date)
        Day_Image = localTimeImage.select('julian_day')
        Hour_Image = localTimeImage.select('local_time')
        Tc = ee.Image(temperature_2m_Dataset_List.get(utcTimeHours)).clip(roi)
        T24 = ee.Image(temperature_2m_Dataset.reduce(ee.Reducer.mean())).clip(roi)
        T240 = T24
        Wind = ee.Image(windSpeed_Dataset_List.get(utcTimeHours)).clip(roi)
        Humidity = ee.Image(relativeHumidityDataset_List.get(utcTimeHours)).clip(roi)
        Pres = ee.Image(surface_pressured_List.get(utcTimeHours)).clip(roi)
        solar_ = ee.Image(solardatDataList.get(utcTimeHours)).clip(roi)

        meanSolar = solar_.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi).values().get(0) 
        solar = solar_.unmask(ee.Image.constant(meanSolar))
        DI = ee.Image.constant(0)
        PPFD = solar.where(solar.lte(0), 0)
        PPFD24 = solar.where(D_solar_radition.lte(0), 0).clip(roi)
        PPFD240 = PPFD24 
        TemporaryDirectory = os.path.join(full_out_dir,TemporaryDirectory_in)
        if os.path.exists(TemporaryDirectory):
            print('The path [{}] exists'.format(TemporaryDirectory))
            
        else:
            os.makedirs(TemporaryDirectory, exist_ok=True)

        for j in range(0, len(chem_names_list), mini_batch):
                
            mini_batch_chem_names = ee.List(chem_names_list[j:j+mini_batch])
            ER_i = EMPORC.ER_calculate(TSTLEN ,Day_Image, Hour_Image, lat,
                    Tc, T24, T240,PPFD ,PPFD24,PPFD240,
                    Wind, Humidity, Pres, DI, lai_c,lai_p,Layers,
                    TotalPTFS.clip(roi),EFMAPS.clip(roi),mini_batch_chem_names,PFTS_ContainPlant,PFTSNamesList)
            size_mini_batch = len(chem_names_list[j:j+mini_batch])

            num_lens = len(str(len(chem_names_list)))
            print(f'save {mini_batch_chem_names.getInfo()}' )
  
            if len(chem_names_list)<=1:
                Temporaryfilename = f'GEE_MEGAN_ER_{mini_batch_chem_names.getInfo()[0]}_{date_str}_{utcTimeHours:02}.tif'
            else:
                Temporaryfilename = f'GEE_MEGAN_ER_{j:02}_{mini_batch:02}_{date_str}_{utcTimeHours:02}.tif'
            filename = os.path.join(TemporaryDirectory , Temporaryfilename)
            ER = {}
            for k in range(size_mini_batch):
                ER[chem_names_list[j:j+mini_batch][k]] = ee.Image(ER_i.get(k))

            ER_images = list(ER.values())
            ER_image = ee.Image.cat(ER_images)
            ER_image = ee.Image.cat(ER_images).unmask(0)
 
            geemap.ee_export_image(ee.Image(ER_image), filename=filename, scale=outputscale, region=roi,crs='EPSG:4326', file_per_band=False)

def Emporc_global_500m(date,Hour,chem_names,PFTS_ContainPlant,
           TotalPTFS,EFMAPS,roi,scale,outputscale,Layers,mini_batch,
           out_dir,out_dir_filename,TemporaryDirectory_in,input_variables):
    """
    --------------------------------------------------------------------------
    Function: Emporc_global_500m
    --------------------------------------------------------------------------
    Parameters:
        - date (str): date in "YYYY-MM-DD" format.
        - Hour (list): list of hour strings based on zero meridian.
        - chem_names (ee.List): list of chemical species to calculate.
        - PFTS_ContainPlant (ee.Image): vegetation-only PFT bands image.
        - TotalPTFS (ee.Image): composite PFT proportions image.
        - EFMAPS (ee.Image): emission factor maps image.
        - roi (ee.Geometry): region of interest for clipping images.
        - scale (int): processing resolution in meters.
        - outputscale (int): final output image resolution in meters.
        - Layers (int): number of canopy layers for ER calculation.
        - mini_batch (int): batch size of species per GEE calculation.
        - out_dir (str): base directory path for outputs.
        - out_dir_filename (str): subdirectory name for outputs.
        - TemporaryDirectory_in (str): name of temporary directory inside output folder.

    Returns:
        - None: saves emission rate images to disk.
            - Outputs are exported as GeoTIFF files in specified directory.

    Description:
        This function computes global 500m emission rate (ER) data for specified dates and hours.
    Example:
        ```python
        result = Emporc_global_500m(
            "2022-06-06", ["06"], chem_names,
            PFTS_ContainPlant, TotalPTFS, EFMAPS, roi,
            500, 500, 3, 2,
            "/data", "output_500m", "TempDir")
        ```
    """

    import ee
    import Echo_parameter as Echo_parameter
    COMPOUND_EMISSION_FACTORS =Echo_parameter.COMPOUND_EMISSION_FACTORS
    PDT_LOS_CP = Echo_parameter.PDT_LOS_CP
    LD_FCT = Echo_parameter.LD_FCT
    REA_SPC = Echo_parameter.REA_SPC
    REL_EM_ACT = Echo_parameter.REL_EM_ACT
    RHO_FCT =  ee.List(PDT_LOS_CP.get('RHO_FCT'))
    LDF_FCT = ee.List(LD_FCT.get('LDF_FCT'))
    chem_names_list = list(chem_names.getInfo())
    METbandNameList = ["surface_pressure","dewpoint_temperature_2m","temperature_2m",
                "u_component_of_wind_10m","v_component_of_wind_10m"]
    METcollectionId = "ECMWF/ERA5_LAND/HOURLY"
    date_str = date 
    
    LAIcollectionName = "MODIS/061/MOD15A2H" 
    # 5km 
    # LAIcollectionName = 'projects/sat-io/open-datasets/BU_LAI_FPAR/wgs_5km_8d'
    full_out_dir = os.path.join(out_dir,out_dir_filename)
    print(out_dir)
    print(full_out_dir)
    PFTSNamesList = PFTS_ContainPlant.bandNames()
    

    lai_p, lai_c = EMPORC.find_LAI_new(date_str, LAIcollectionName)

 
    original_scale = lai_p.projection().nominalScale().getInfo()
    if scale <=500:
        pass
    else:
        multiplier = math.ceil(scale/original_scale)
        resolution = multiplier*original_scale
        print(f'final resolution is {multiplier*original_scale}')
        lai_p = EMPORC.reduce_mean_image_resolution(lai_p,multiplier)
        lai_c = EMPORC.reduce_mean_image_resolution(lai_c,multiplier)

    def fill_nodata_with_mean(image, roi):

        mean = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=500,
            maxPixels=1e13
        ).values().get(0) 
        # print(mean.getInfo(),2)

        mean_value = mean.getInfo()
        if mean_value is not None:

            return image.unmask(round(mean_value, 2)).clip(roi)
        else:

            # print("Warning: Mean calculation failed or returned None")
            return image.unmask(round(0, 2)).clip(roi)
            
    lai_p = fill_nodata_with_mean(lai_p, roi)
    lai_c = fill_nodata_with_mean(lai_c, roi)

    lai_p = lai_p.clip(roi).multiply(0.1) # User's guide lai * 0.1
    lai_c = lai_c.clip(roi).multiply(0.1)
    lAI_v = input_variables['lAI_v']
    if lAI_v == "True":
         print("Using LAI_v adjustment (500m)")
         VCF_image = EMPORC.find_VCF_image(date_str, dataset="MODIS/006/MOD44B")
         lai_p = EMPORC.adjust_LAI_by_VCF(lai_p, VCF_image)
         lai_c = EMPORC.adjust_LAI_by_VCF(lai_c, VCF_image)

    METDict = EMPORC.get_MET(date,METcollectionId,METbandNameList)
    surface_pressured_Dataset = METDict[METbandNameList[0]]
    dewpoint_temperature_2m_Dataset = METDict[METbandNameList[1]]
    temperature_2m_Dataset = METDict[METbandNameList[2]]
    u_wind_10m_Dataset  = METDict[METbandNameList[3]]
    v_wind_10m_Dataset = METDict[METbandNameList[4]]

  
    relativeHumidityDataset = METPreprocess.calculate_specific_humidity(dewpoint_temperature_2m_Dataset, temperature_2m_Dataset,surface_pressured_Dataset) ## org--

    windSpeed_Dataset = METPreprocess.calculateWindSpeed(u_wind_10m_Dataset, v_wind_10m_Dataset)

    surface_pressured_size = surface_pressured_Dataset.size()
    surface_pressured_List = surface_pressured_Dataset.toList(surface_pressured_size)
    relativeHumidityDataset_Size =  relativeHumidityDataset.size()
    relativeHumidityDataset_List = relativeHumidityDataset.toList(relativeHumidityDataset_Size)
    windSpeed_Dataset_Size = windSpeed_Dataset.size()
    windSpeed_Dataset_List =  windSpeed_Dataset.toList(windSpeed_Dataset_Size)
    temperature_2m_Dataset_size = temperature_2m_Dataset.size()
    temperature_2m_Dataset_List =  temperature_2m_Dataset.toList(temperature_2m_Dataset_size)
    collectionId = "ECMWF/ERA5_LAND/HOURLY"
    bandName = 'surface_solar_radiation_downwards'
    currentSolarDataset = METPreprocess.filterByDate(date, METcollectionId, bandName)
    previousDay = ee.Date(date).advance(-1, 'day').format('YYYY-MM-dd')
    previousDaySolarDataset =  METPreprocess.filterByDate(previousDay, METcollectionId, bandName)
    instantaneousRadiationCollection = METPreprocess.calcInstantaneousSolarRadiation(currentSolarDataset, previousDaySolarDataset)
    D_solar_radition = instantaneousRadiationCollection.reduce(ee.Reducer.mean())
    solarDataSetSize = instantaneousRadiationCollection.size()
    solardatDataList = instantaneousRadiationCollection.toList(solarDataSetSize)

    lon_lat_image = METPreprocess.create_lon_lat_image(lai_p, roi)
    lat = lon_lat_image.select('latitude')
    TSTLEN = 30

    for i in range(len(Hour)):
   
        utcTimeHours = int(Hour[i])
        localTimeImage = METPreprocess.computeLocalTime(lon_lat_image, utcTimeHours,date)
        Day_Image = localTimeImage.select('julian_day')
        Hour_Image = localTimeImage.select('local_time')
        Tc = ee.Image(temperature_2m_Dataset_List.get(utcTimeHours)).clip(roi)
        T24 = ee.Image(temperature_2m_Dataset.reduce(ee.Reducer.mean())).clip(roi)
        T240 = T24
        Wind = ee.Image(windSpeed_Dataset_List.get(utcTimeHours)).clip(roi)
        Humidity = ee.Image(relativeHumidityDataset_List.get(utcTimeHours)).clip(roi)
        Pres = ee.Image(surface_pressured_List.get(utcTimeHours)).clip(roi)
        solar_ = ee.Image(solardatDataList.get(utcTimeHours)).clip(roi)
        meanSolar = solar_.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi).values().get(0) 
        solar = solar_.unmask(ee.Image.constant(meanSolar))
        DI = ee.Image.constant(0)
        PPFD = solar.where(solar.lte(0), 0)
        PPFD24 = solar.where(D_solar_radition.lte(0), 0).clip(roi)
        PPFD240 = PPFD24 
        TemporaryDirectory = os.path.join(full_out_dir,TemporaryDirectory_in)
        if os.path.exists(TemporaryDirectory):
            print('The path [{}] exists'.format(TemporaryDirectory))
            
        else:
            os.makedirs(TemporaryDirectory, exist_ok=True)

        for j in range(0, len(chem_names_list), mini_batch):
                
            mini_batch_chem_names = ee.List(chem_names_list[j:j+mini_batch])
            ER_i = EMPORC.ER_calculate(TSTLEN ,Day_Image, Hour_Image, lat,
                    Tc, T24, T240,PPFD ,PPFD24,PPFD240,
                    Wind, Humidity, Pres, DI, lai_c,lai_p,Layers,
                    TotalPTFS.clip(roi),EFMAPS.clip(roi),mini_batch_chem_names,PFTS_ContainPlant,PFTSNamesList)
            size_mini_batch = len(chem_names_list[j:j+mini_batch])
            num_lens = len(str(len(chem_names_list)))
            print(f'save {mini_batch_chem_names.getInfo()}' )
            # Temporaryfilename = 'ER_{}.tif'.format(j,'0{}'.format(num_lens))
            if len(chem_names_list)<=1:
                Temporaryfilename = f'GEE_MEGAN_ER_{mini_batch_chem_names.getInfo()[0]}_{date_str}_{utcTimeHours:02}.tif'
            else:
                Temporaryfilename = f'GEE_MEGAN_ER_{j:02}_{mini_batch:02}_{date_str}_{utcTimeHours:02}.tif'
            filename = os.path.join(TemporaryDirectory , Temporaryfilename)
            ER = {}
            for k in range(size_mini_batch):
                ER[chem_names_list[j:j+mini_batch][k]] = ee.Image(ER_i.get(k))
            ER_images = list(ER.values())
            ER_image = ee.Image.cat(ER_images)
            geemap.ee_export_image(ee.Image(ER_image), filename=filename, scale=outputscale, region=roi,crs='EPSG:4326',file_per_band=False)

def Emporc_local(date,Hour,chem_names,PFTS_ContainPlant,
           TotalPTFS,EFMAPS,roi,scale,outputscale,Layers,mini_batch,
           out_dir,out_dir_filename,TemporaryDirectory_in):
    """
    --------------------------------------------------------------------------
    Function: Emporc_local
    --------------------------------------------------------------------------
    Parameters:
        - date (str): date in "YYYY-MM-DD" format.
        - Hour (list): list of hour strings based on zero meridian.
        - chem_names (ee.List): list of chemical species to calculate.
        - PFTS_ContainPlant (ee.Image): vegetation-only PFT bands image.
        - TotalPTFS (ee.Image): composite PFT proportions image.
        - EFMAPS (ee.Image): emission factor maps image.
        - roi (ee.Geometry): region of interest for local calculation.
        - scale (int): processing resolution in meters.
        - outputscale (int): final output image resolution in meters.
        - Layers (int): number of canopy layers for ER calculation.
        - mini_batch (int): batch size of species per GEE calculation.
        - out_dir (str): base directory path for outputs.
        - out_dir_filename (str): subdirectory name for outputs.
        - TemporaryDirectory_in (str): name of temporary directory inside output folder.

    Returns:
        - None: saves local emission rate images to disk.
            - Outputs are exported as GeoTIFF files in the specified directory.

    Description:
        This function performs local-scale EMPORC emission calculations using fused MODIS-Landsat LAI, local-lucc
        and ERA5-Land meteorological datasets. For each selected UTC hour, it prepares biophysical and 
        meteorological parameters (e.g., temperature, humidity, radiation), computes emission rates 
        per mini-batch of chemical species, and exports the results as multi-band GeoTIFF files.

    Example:
        ```python
        result = Emporc_local(
            "2022-06-06", ["06", "12"], chem_names,
            PFTS_ContainPlant, TotalPTFS, EFMAPS, roi,
            500, 1000, 3, 2,
            "/data", "output_local", "TempDir")
        ```
    """

    import ee
    import Echo_parameter as Echo_parameter
    COMPOUND_EMISSION_FACTORS =Echo_parameter.COMPOUND_EMISSION_FACTORS
    PDT_LOS_CP = Echo_parameter.PDT_LOS_CP
    LD_FCT = Echo_parameter.LD_FCT
    REA_SPC = Echo_parameter.REA_SPC
    REL_EM_ACT = Echo_parameter.REL_EM_ACT
    RHO_FCT =  ee.List(PDT_LOS_CP.get('RHO_FCT'))
    LDF_FCT = ee.List(LD_FCT.get('LDF_FCT'))
    chem_names_list = list(chem_names.getInfo())
    METbandNameList = ["surface_pressure","dewpoint_temperature_2m","temperature_2m",
                "u_component_of_wind_10m","v_component_of_wind_10m"]
    METcollectionId = "ECMWF/ERA5_LAND/HOURLY"
    date_str = date 
    LAIcollectionName = "MODIS/061/MOD15A2H" 
 
    full_out_dir = os.path.join(out_dir,out_dir_filename)
    print(out_dir)
    print(full_out_dir)
    PFTSNamesList = PFTS_ContainPlant.bandNames()
    
    lai_p, lai_c = EMPORC.get_local_LAI(date_str, LAIcollectionName,roi)
    

    def fill_nodata_with_mean(image, roi):

        mean = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=500,
            maxPixels=1e13
        ).values().get(0) 

        mean_value = mean.getInfo()
        if mean_value is not None:

            return image.unmask(round(mean_value, 2)).clip(roi)
        else:
            # print("Warning: Mean calculation failed or returned None")
            return image.unmask(round(0, 2)).clip(roi)
        
    lai_p = fill_nodata_with_mean(lai_p, roi)
    lai_c = fill_nodata_with_mean(lai_c, roi)

    METDict = EMPORC.get_MET(date,METcollectionId,METbandNameList)
    surface_pressured_Dataset = METDict[METbandNameList[0]]
    dewpoint_temperature_2m_Dataset = METDict[METbandNameList[1]]
    temperature_2m_Dataset = METDict[METbandNameList[2]]
    u_wind_10m_Dataset  = METDict[METbandNameList[3]]
    v_wind_10m_Dataset = METDict[METbandNameList[4]]

    relativeHumidityDataset = METPreprocess.calculate_specific_humidity(dewpoint_temperature_2m_Dataset, temperature_2m_Dataset,surface_pressured_Dataset)  ## org
    # relativeHumidityDataset = METPreprocess.calculate_density(dewpoint_temperature_2m_Dataset, temperature_2m_Dataset,surface_pressured_Dataset)
    windSpeed_Dataset = METPreprocess.calculateWindSpeed(u_wind_10m_Dataset, v_wind_10m_Dataset)

    surface_pressured_size = surface_pressured_Dataset.size()
    surface_pressured_List = surface_pressured_Dataset.toList(surface_pressured_size)
    relativeHumidityDataset_Size =  relativeHumidityDataset.size()
    relativeHumidityDataset_List = relativeHumidityDataset.toList(relativeHumidityDataset_Size)
    windSpeed_Dataset_Size = windSpeed_Dataset.size()
    windSpeed_Dataset_List =  windSpeed_Dataset.toList(windSpeed_Dataset_Size)
    temperature_2m_Dataset_size = temperature_2m_Dataset.size()
    temperature_2m_Dataset_List =  temperature_2m_Dataset.toList(temperature_2m_Dataset_size)

    collectionId = "ECMWF/ERA5_LAND/HOURLY"
    bandName = 'surface_solar_radiation_downwards'
    currentSolarDataset = METPreprocess.filterByDate(date, METcollectionId, bandName)
    previousDay = ee.Date(date).advance(-1, 'day').format('YYYY-MM-dd')
    previousDaySolarDataset =  METPreprocess.filterByDate(previousDay, METcollectionId, bandName)
    instantaneousRadiationCollection = METPreprocess.calcInstantaneousSolarRadiation(currentSolarDataset, previousDaySolarDataset)
    D_solar_radition = instantaneousRadiationCollection.reduce(ee.Reducer.mean())
    solarDataSetSize = instantaneousRadiationCollection.size()
    solardatDataList = instantaneousRadiationCollection.toList(solarDataSetSize)

    lon_lat_image = METPreprocess.create_lon_lat_image(lai_p, roi)
    lat = lon_lat_image.select('latitude')
    TSTLEN = 30

    for i in range(len(Hour)):
        utcTimeHours = int(Hour[i])
        localTimeImage = METPreprocess.computeLocalTime(lon_lat_image, utcTimeHours,date)
        Day_Image = localTimeImage.select('julian_day')
        Hour_Image = localTimeImage.select('local_time')
        
        Tc_ = ee.Image(temperature_2m_Dataset_List.get(utcTimeHours)).clip(roi)
        meanTc = Tc_.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi).values().get(0)
        Tc = Tc_.unmask(ee.Image.constant(meanTc))
        
        T24_ = ee.Image(temperature_2m_Dataset.reduce(ee.Reducer.mean())).clip(roi)
        meanT24 = T24_.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi).values().get(0)
        T24 = T24_.unmask(ee.Image.constant(meanT24))
        T240 = T24
        
        Wind_ = ee.Image(windSpeed_Dataset_List.get(utcTimeHours)).clip(roi)
        meanWind = Wind_.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi).values().get(0)
        Wind = Wind_.unmask(ee.Image.constant(meanWind))
        
        Humidity_ = ee.Image(relativeHumidityDataset_List.get(utcTimeHours)).clip(roi)
        meanHumidity = Humidity_.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi).values().get(0)
        Humidity = Humidity_.unmask(ee.Image.constant(meanHumidity))
        
        Pres_ = ee.Image(surface_pressured_List.get(utcTimeHours)).clip(roi)
        meanPres = Pres_.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi).values().get(0)
        Pres = Pres_.unmask(ee.Image.constant(meanPres))
        
        solar_ = ee.Image(solardatDataList.get(utcTimeHours)).clip(roi)

        meanSolar = solar_.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi).values().get(0) 
        solar = solar_.unmask(ee.Image.constant(meanSolar))
        
        DI = ee.Image.constant(0)
        PPFD = solar.where(solar.lte(0), 0)
        PPFD24 = solar.where(D_solar_radition.lte(0), 0).clip(roi)
        PPFD240 = PPFD24 
        TemporaryDirectory = os.path.join(full_out_dir,TemporaryDirectory_in)
        if os.path.exists(TemporaryDirectory):
            print('The path [{}] exists'.format(TemporaryDirectory))
            
        else:
            os.makedirs(TemporaryDirectory, exist_ok=True)

        for j in range(0, len(chem_names_list), mini_batch):
                
            mini_batch_chem_names = ee.List(chem_names_list[j:j+mini_batch])
            ER_i = EMPORC.ER_calculate(TSTLEN ,Day_Image, Hour_Image, lat,
                    Tc, T24, T240,PPFD ,PPFD24,PPFD240,
                    Wind, Humidity, Pres, DI, lai_c,lai_p,Layers,
                    TotalPTFS.clip(roi),EFMAPS.clip(roi),mini_batch_chem_names,PFTS_ContainPlant,PFTSNamesList)
            size_mini_batch = len(chem_names_list[j:j+mini_batch])
   
            num_lens = len(str(len(chem_names_list)))
            print(f'save {mini_batch_chem_names.getInfo()}' )
            if len(chem_names_list)<=1:
                Temporaryfilename = f'GEE_MEGAN_ER_{mini_batch_chem_names.getInfo()[0]}_{date_str}_{utcTimeHours:02}.tif'
            else:
                Temporaryfilename = f'GEE_MEGAN_ER_{j:02}_{mini_batch:02}_{date_str}_{utcTimeHours:02}.tif'
            filename = os.path.join(TemporaryDirectory , Temporaryfilename)
            ER = {}
            for k in range(size_mini_batch):
                ER[chem_names_list[j:j+mini_batch][k]] = ee.Image(ER_i.get(k))

            ER_images = list(ER.values())
            ER_image = ee.Image.cat(ER_images).unmask(0)

            geemap.ee_export_image(ee.Image(ER_image), filename=filename, scale=outputscale, region=roi,crs='EPSG:4326', file_per_band=False)

def generate_datetime_list(start_str: str, end_str: str, step: int) -> list:
    """
    --------------------------------------------------------------------------
    Function: generate_datetime_list
    --------------------------------------------------------------------------
    Parameters:
        - start_str (str): Start datetime as a string in the format 'YYYY-MM-DD HH:MM:SS'.
        - end_str (str): End datetime as a string in the format 'YYYY-MM-DD HH:MM:SS'.
        - step (int): Step interval in hours.

    Returns:
        - list: A list of datetime strings (format 'YYYY-MM-DD HH:MM:SS') spaced by the given step.

    Description:
        This function generates a list of datetime strings starting from `start_str` to `end_str`
        using a specified hourly step. It validates input format and ensures chronological order.
        The output list can be used for time-stepped simulations or GEE input time sequences.

    Example:
        ```python
        generate_datetime_list("2023-06-18 00:00:00", "2023-06-18 12:00:00", 3)
        ```
    --------------------------------------------------------------------------
    """
    
    
    def validate_datetime(datetime_str: str) -> datetime:
        """
        Validate a datetime string and return its datetime representation.

        Args:
        - datetime_str (str): Datetime as a string in the format 'YYYY-MM-DD HH:MM:SS'.

        Returns:
        - datetime: DateTime representation of the given datetime string.
        """
        try:
            valid_datetime = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
            return valid_datetime
        except ValueError:
            raise ValueError(f"The provided datetime '{datetime_str}' is not in the correct format 'YYYY-MM-DD HH:MM:SS' or is not a valid datetime.")

    start_datetime = validate_datetime(start_str)
    end_datetime = validate_datetime(end_str)

    # Check if end_datetime is earlier than start_datetime
    if end_datetime < start_datetime:
        raise ValueError("The end datetime must be after the start datetime.")

    # Calculate total hours difference
    total_hours_diff = int((end_datetime - start_datetime).total_seconds() // 3600) + 1

    return [(start_datetime + timedelta(hours=x)).strftime('%Y-%m-%d %H:%M:%S') for x in range(0, total_hours_diff, step)]

def read_parameters_from_file(file_path):
    """
    --------------------------------------------------------------------------
    Function: read_parameters_from_file
    --------------------------------------------------------------------------
    Parameters:
        - file_path (str): Path to the parameter configuration file. 
          Each line should follow the format 'key = value' and support
          optional inline comments prefixed by '#'.

    Returns:
        - dict: Dictionary of parameter names and their corresponding values.
          The values are automatically converted to int, float, list, or str
          depending on their format.

    Description:
        Reads a configuration file and parses each line to extract parameter key-value pairs.
    Example:
        ```
        # content of config.nml
        start_day = '2022-06-06 00:00:00'
            ...
        ```
    --------------------------------------------------------------------------
    """
    parameters = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Skip lines starting with '#' (comments)
            if not line.strip().startswith('#'):
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    # Remove inline comments from the value
                    value = value.split('#', 1)[0].strip()  

                    # Remove surrounding single or double quotes from strings
                    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
                        value = value[1:-1]

                   #  Convert value to appropriate Python type
                    if value.isdigit():  
                        parameters[key.strip()] = int(value)
                    elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:  
                        parameters[key.strip()] = float(value)
                    elif value.startswith('[') and value.endswith(']'):  
                        parameters[key.strip()] = eval(value)
                    else:
                        parameters[key.strip()] = str(value)  
    return parameters

def get_model_year(start_str: str, end_str: str) -> list:
    """
    --------------------------------------------------------------------------
    Function: get_model_year
    --------------------------------------------------------------------------
    Parameters:
        - start_str (str): Start datetime string in 'YYYY-MM-DD HH:MM:SS' format.
        - end_str (str): End datetime string in 'YYYY-MM-DD HH:MM:SS' format.

    Returns:
        - list: A list of integer years from the start year to the end year (inclusive).

    Description:
        Extracts the start and end years from the provided datetime strings and returns
        a list containing all years within that range. This is useful for determining
        which years are involved in a multi-year simulation.

    Example:
        ```python
        get_model_year("2020-06-01 00:00:00", "2022-08-01 00:00:00")
        ```
    --------------------------------------------------------------------------
    """
    start_date_obj = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
    end_date_obj = datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S')
    start_year = start_date_obj.year
    end_year = end_date_obj.year
    # print(start_year)
    # print(end_year)
    year_list = [year for year in range(start_year, end_year + 1)]
    return year_list
def split_dates_by_year(date_list):
    """
    --------------------------------------------------------------------------
    Function: split_dates_by_year
    --------------------------------------------------------------------------
    Parameters:
        - date_list (list): A list of datetime strings in 'YYYY-MM-DD HH:MM:SS' format.

    Returns:
        - dict: A dictionary where keys are years (int), and values are lists of
                datetime strings belonging to that year.

    Description:
        Groups a list of datetime strings by their corresponding calendar year.

    Example:
        ```python
        split_dates_by_year(["2020-01-01 00:00:00", "2021-05-01 00:00:00"])
        # Output: {2020: [...], 2021: [...]}
        ```
    --------------------------------------------------------------------------
    """    
    years = {}
    for date in date_list:
        date_obj = datetime.strptime(date , '%Y-%m-%d %H:%M:%S')
        year = date_obj.year
        if year not in years:
            years[year] = []
        years[year].append(date)
    return years



def process_date_range_global_5km(date_range,input_variables,year):
    """
    --------------------------------------------------------------------------
    Function: process_date_range_global_5km
    --------------------------------------------------------------------------
    Parameters:
        - date_range (list): List of datetime strings in 'YYYY-MM-DD HH:MM:SS' format.
        - input_variables (dict): Dictionary of input parameters including
        - year (int or str): The simulation year being processed.

    Returns:
        - None: Processes emissions for each date in the list and exports results as GeoTIFF files.

    Description:
        This function prepares all required environmental inputs and runs EMPORC
        emission calculations over a series of datetimes for the global 5 km model.
        It handles land cover reclassification, emission factor mapping (EFMAP),
        optional DEFMAP corrections based on fire data, dynamic lucc, and dynamically switches
        between full global or ROI-based processing. Results are exported hourly
        and grouped by date.

    Example:
        ```python
        date_range = ['2022-06-18 02:00:00']
        input_variables = {
            'model_type': 'global_5000m',
             ...
        }
        process_date_range_global_5km(date_range, input_variables, '2025')
        ```
    --------------------------------------------------------------------------
    """
    
    model_type = input_variables['model_type']
    DEFMAP_variables = input_variables['DEFMAP']
    chem_names = ee.List(input_variables['chem_names'])
    DEFMAP_type = input_variables['DEFMAP_type']
    COMPOUND_EMISSION_FACTORS = Echo_parameter.COMPOUND_EMISSION_FACTORS

    LUCC_collection_name = 'MODIS/061/MCD12C1'
    PFTS_year  = input_variables['PFTS_year']
    

    if model_type == 'global_5000m':
        Collection_name = "projects/gee-megan/assets/MCD64A1_5km"
        burnedAreaFraction,burnedDate = get_DEFMAP.get_burn_data_5000m(int(year))
 

    if input_variables['PFTS_year'] == '':
        print('dynamic PFTS year')
        PFTS = METPreprocess.ReClassPFTS_all_type(int(year),LUCC_collection_name)
        year_p = int(year)-1
        if year_p < 2001:
            year_p = 2001

        PFTS_p = METPreprocess.ReClassPFTS_all_type(int(year_p),LUCC_collection_name)
        EFMAPS = METPreprocess.calculateChem(chem_names, COMPOUND_EMISSION_FACTORS, PFTS)
        EFMAPS_p = METPreprocess.calculateChem(chem_names, COMPOUND_EMISSION_FACTORS, PFTS_p)
        EFMAPS = ee.Image(ee.List(EFMAPS).slice(1).iterate(METPreprocess.add_bands, ee.Image(ee.List(EFMAPS).get(0))))
        EFMAPS_p = ee.Image(ee.List(EFMAPS_p).slice(1).iterate(METPreprocess.add_bands, ee.Image(ee.List(EFMAPS_p).get(0))))
        F_EFMAPs = EFMAPS
    else:
        print('constant PFTS year')
        PFTS = METPreprocess.ReClassPFTS_all_type(PFTS_year,LUCC_collection_name)
        EFMAPS = METPreprocess.calculateChem(chem_names, COMPOUND_EMISSION_FACTORS, PFTS)
        EFMAPS = ee.Image(ee.List(EFMAPS).slice(1).iterate(METPreprocess.add_bands, ee.Image(ee.List(EFMAPS).get(0))))
        F_EFMAPs = EFMAPS
    # Total plant percentage
    ContainPlantbandNames = PFTS.bandNames().slice(0, 15)
    PFTS_ContainPlant = PFTS.select(ContainPlantbandNames)
    TotalPTFS = PFTS_ContainPlant.reduce(ee.Reducer.sum())
    for date in date_range:
        date_Day, date_time = date.split(' ')
        Hour = [date_time.split(':')[0]]
        if model_type == 'global_5000m' and DEFMAP_variables == 'T':
            print(f'Dynamic EFMAPS is {date_Day}')
            DEFMAPS = get_DEFMAP.calculate_defmap(EFMAPS_p,EFMAPS, date_Day,burnedDate,burnedAreaFraction)
            F_EFMAPs = DEFMAPS 
            print(f'DEFMAPS is True {date_Day}')
            if DEFMAP_type == 'DEFMAP_only_fire':
                print(f'DEFMAPS is only fire')
                DEFMAPS_only_fire = get_DEFMAP.calculate_defmap_only_fire(EFMAPS, date_Day,burnedDate,burnedAreaFraction)
                F_EFMAPs = DEFMAPS_only_fire
            elif DEFMAP_type == 'DEFMAP__not_only_fire':
                print(f'DEFMAPS is couple lucc with fire')
                DEFMAPS = get_DEFMAP.calculate_defmap(EFMAPS_p,EFMAPS, date_Day,burnedDate,burnedAreaFraction)
                F_EFMAPs = DEFMAPS
        elif DEFMAP_variables == 'F':
            F_EFMAPs = EFMAPS
            print('Not use DEFMAPs, just EFMAPs')
           
            original_scale = F_EFMAPs.projection().nominalScale().getInfo()
            print(original_scale)
        chem_names = ee.List(input_variables['chem_names'])
        if input_variables['roi'] =='':
            Layers = input_variables['Layers']
            scale = input_variables['scale']
            output_scale = input_variables['output_scale']
            mini_batch = input_variables['mini_batch']
            out_dir = input_variables['out_dir']
            out_dir_filename = input_variables['out_dir_filename']
            TemporaryDirectory = input_variables['TemporaryDirectory']
            print('global')
            print(f'-------------------------------strat--{date}-------------------------------')
            Emporc_global_5km_without_roi(date_Day, Hour,chem_names,PFTS_ContainPlant,
                TotalPTFS,F_EFMAPs,scale,output_scale,Layers,mini_batch,
            out_dir,out_dir_filename,TemporaryDirectory,input_variables)
            print(f'--------------------------------end--{date}--------------------------------')
        else:
            print('use roi {}'.format(input_variables['roi']))
            roi_1 = ee.Geometry.Rectangle(input_variables['roi'])

            coords = roi_1.coordinates()
        
      
            roi = ee.Geometry.Polygon(coords, None, False)
            # print(input_variables['roi'])
            # print(roi.getInfo())
            Layers = input_variables['Layers']
            scale = input_variables['scale']
            output_scale = input_variables['output_scale']
            mini_batch = input_variables['mini_batch']
            out_dir = input_variables['out_dir']
            out_dir_filename = input_variables['out_dir_filename']
            TemporaryDirectory = input_variables['TemporaryDirectory']
            # print(date)
          
            print(f'-------------------------------strat--{date}-------------------------------')
            Emporc_global_5km_with_roi(date_Day, Hour,chem_names,PFTS_ContainPlant,
                TotalPTFS,F_EFMAPs,roi,scale,output_scale,Layers,mini_batch,
            out_dir,out_dir_filename,TemporaryDirectory,input_variables)
            print(f'--------------------------------end--{date}--------------------------------')

def Emporc_global_5km_without_roi_ST(date,Hour,chem_names,PFTS_ContainPlant,
           TotalPTFS,EFMAPS,scale,outputscale,Layers,mini_batch,
           out_dir,out_dir_filename,TemporaryDirectory_in,input_variables):
    """
    --------------------------------------------------------------------------
    Function: Emporc_global_5km_without_roi_ST
    --------------------------------------------------------------------------
    Parameters:
        - date (str): Target date in "YYYY-MM-DD" format.
        - Hour (list): List of hour strings (e.g., ["00"]).
        - chem_names (ee.List): List of chemical species to calculate emission rates for.
        - PFTS_ContainPlant (ee.Image): Image containing only vegetated PFT bands.
        - TotalPTFS (ee.Image): Total PFT proportion image.
        - EFMAPS (ee.Image): Emission factor maps for different species.
        - scale (int): Processing resolution in meters (e.g., 5000).
        - outputscale (int): Export resolution in meters for output GeoTIFF.
        - Layers (int): Number of canopy layers used in EMPORC emission calculation.
        - mini_batch (int): Batch size for species processed per call to avoid GEE limits.
        - out_dir (str): Directory path for storing output files.
        - out_dir_filename (str): Subfolder name within output directory.
        - TemporaryDirectory_in (str): Subfolder name for storing intermediate hourly results.

    Returns:
        - None: Saves ER (emission rate) GeoTIFF images to the specified output directory.

    Description:
        This function computes global 5 km gridded biogenic emission rates using surface 
        temperature (skin temperature) instead of 2-meter air temperature. 

    Example:
        ```python
        Emporc_global_5km_without_roi_ST(
            date="2020-07-01",
            Hour=["06"],
            chem_names=ee.List(['ISOP']),
            PFTS_ContainPlant=pft_image,
            TotalPTFS=total_pft,
            EFMAPS=ef_map,
            scale=5000,
            outputscale=25000,
            Layers=5,
            mini_batch=2,
            out_dir="/data/emissions",
            out_dir_filename="global_5km_skin",
            TemporaryDirectory_in="temp_2020"
        )
        ```
    --------------------------------------------------------------------------
    """

    import ee
    import Echo_parameter as Echo_parameter
    COMPOUND_EMISSION_FACTORS =Echo_parameter.COMPOUND_EMISSION_FACTORS
    PDT_LOS_CP = Echo_parameter.PDT_LOS_CP
    LD_FCT = Echo_parameter.LD_FCT
    REA_SPC = Echo_parameter.REA_SPC
    REL_EM_ACT = Echo_parameter.REL_EM_ACT
    RHO_FCT =  ee.List(PDT_LOS_CP.get('RHO_FCT'))
    LDF_FCT = ee.List(LD_FCT.get('LDF_FCT'))
    chem_names_list = list(chem_names.getInfo())
    METbandNameList = ["surface_pressure","dewpoint_temperature_2m","temperature_2m",
                "u_component_of_wind_10m","v_component_of_wind_10m","skin_temperature"]
    METcollectionId = "ECMWF/ERA5_LAND/HOURLY"
    date_str = date 

    LAIcollectionName = 'projects/sat-io/open-datasets/BU_LAI_FPAR/wgs_5km_8d'
    full_out_dir = os.path.join(out_dir,out_dir_filename)
    print(out_dir)
    print(full_out_dir)
    PFTSNamesList = PFTS_ContainPlant.bandNames()
    

    print(date_str)
    

    year = int(date_str[:4])

    if year > 2022:
        new_date_str = '2022' + date_str[4:]
    else:
        new_date_str = date_str
    

    print(new_date_str)
    lai_p, lai_c = EMPORC.find_LAI_new(new_date_str, LAIcollectionName)


    original_scale = lai_p.projection().nominalScale().getInfo()
    if scale <=500:
        pass
    else:
        multiplier = math.ceil(scale/original_scale)
        resolution = multiplier*original_scale
        print(f'final resolution is {multiplier*original_scale}')
        lai_p = EMPORC.reduce_mean_image_resolution(lai_p,multiplier)
        lai_c = EMPORC.reduce_mean_image_resolution(lai_c,multiplier)

    lai_p = lai_p.multiply(0.1) # User's guide lai * 0.1
    lai_c = lai_c.multiply(0.1)

    lAI_v = input_variables['lAI_v']
    if lAI_v == "True":
        print("Using LAI_v adjustment (LAI / VCF)")
        VCF_image = EMPORC.find_VCF_image(date_str, dataset="MODIS/061/MOD13C1")
        lai_p  = EMPORC.adjust_LAI_by_VCF(lai_p , VCF_image)
        lai_c  = EMPORC.adjust_LAI_by_VCF(lai_c , VCF_image)

    METDict = EMPORC.get_MET(date,METcollectionId,METbandNameList)
    surface_pressured_Dataset = METDict[METbandNameList[0]]
    dewpoint_temperature_2m_Dataset = METDict[METbandNameList[1]]
    temperature_2m_Dataset = METDict[METbandNameList[2]]
    u_wind_10m_Dataset  = METDict[METbandNameList[3]]
    v_wind_10m_Dataset = METDict[METbandNameList[4]]
    temperature_Skin_Dataset = METDict[METbandNameList[5]]

    relativeHumidityDataset = METPreprocess.calculate_density(dewpoint_temperature_2m_Dataset, temperature_2m_Dataset,surface_pressured_Dataset)
    windSpeed_Dataset = METPreprocess.calculateWindSpeed(u_wind_10m_Dataset, v_wind_10m_Dataset)

    surface_pressured_size = surface_pressured_Dataset.size()
    surface_pressured_List = surface_pressured_Dataset.toList(surface_pressured_size)
    relativeHumidityDataset_Size =  relativeHumidityDataset.size()
    relativeHumidityDataset_List = relativeHumidityDataset.toList(relativeHumidityDataset_Size)
    windSpeed_Dataset_Size = windSpeed_Dataset.size()
    windSpeed_Dataset_List =  windSpeed_Dataset.toList(windSpeed_Dataset_Size)
    ## replaced here with skin temperature
    temperature_2m_Dataset_size =  temperature_Skin_Dataset.size() 
    temperature_2m_Dataset_List =  temperature_Skin_Dataset.toList(temperature_2m_Dataset_size)

    collectionId = "ECMWF/ERA5_LAND/HOURLY"
    bandName = 'surface_solar_radiation_downwards'
    currentSolarDataset = METPreprocess.filterByDate(date, METcollectionId, bandName)
    previousDay = ee.Date(date).advance(-1, 'day').format('YYYY-MM-dd')
    previousDaySolarDataset =  METPreprocess.filterByDate(previousDay, METcollectionId, bandName)
    instantaneousRadiationCollection = METPreprocess.calcInstantaneousSolarRadiation(currentSolarDataset, previousDaySolarDataset)
    D_solar_radition = instantaneousRadiationCollection.reduce(ee.Reducer.mean())
    solarDataSetSize = instantaneousRadiationCollection.size()
    solardatDataList = instantaneousRadiationCollection.toList(solarDataSetSize)
 
    lon_lat_image = METPreprocess.create_lon_lat_image_without_roi(lai_p)
    lat = lon_lat_image.select('latitude')
    TSTLEN = 30

    for i in range(len(Hour)):
        utcTimeHours = int(Hour[i])
        localTimeImage = METPreprocess.computeLocalTime(lon_lat_image, utcTimeHours,date)
        Day_Image = localTimeImage.select('julian_day')
        Hour_Image = localTimeImage.select('local_time')
        Tc = ee.Image(temperature_2m_Dataset_List.get(utcTimeHours))
        T24 = ee.Image(temperature_2m_Dataset.reduce(ee.Reducer.mean()))
        T240 = T24
        Wind = ee.Image(windSpeed_Dataset_List.get(utcTimeHours))
        Humidity = ee.Image(relativeHumidityDataset_List.get(utcTimeHours))
        Pres = ee.Image(surface_pressured_List.get(utcTimeHours))
        solar = ee.Image(solardatDataList.get(utcTimeHours))
        DI = ee.Image.constant(0)
        PPFD = solar.where(solar.lte(0), 0)
        PPFD24 = solar.where(D_solar_radition.lte(0), 0)
        PPFD240 = PPFD24 
        TemporaryDirectory = os.path.join(full_out_dir,TemporaryDirectory_in)
        if os.path.exists(TemporaryDirectory):
            print('The path [{}] exists'.format(TemporaryDirectory))
            
        else:
            os.makedirs(TemporaryDirectory, exist_ok=True)

        for j in range(0, len(chem_names_list), mini_batch):
                
            mini_batch_chem_names = ee.List(chem_names_list[j:j+mini_batch])
            ER_i = EMPORC.ER_calculate(TSTLEN ,Day_Image, Hour_Image, lat,
                    Tc, T24, T240,PPFD ,PPFD24,PPFD240,
                    Wind, Humidity, Pres, DI, lai_c,lai_p,Layers,
                    TotalPTFS,EFMAPS,mini_batch_chem_names,PFTS_ContainPlant,PFTSNamesList)
            size_mini_batch = len(chem_names_list[j:j+mini_batch])

            num_lens = len(str(len(chem_names_list)))
            print(f'save {mini_batch_chem_names.getInfo()}' )

            if len(chem_names_list)<=1:
                Temporaryfilename = f'GEE_MEGAN_ER_{mini_batch_chem_names.getInfo()[0]}_{date_str}_{utcTimeHours:02}.tif'
            else:
                Temporaryfilename = f'GEE_MEGAN_ER_{j:02}_{mini_batch:02}_{date_str}_{utcTimeHours:02}.tif'
            filename = os.path.join(TemporaryDirectory , Temporaryfilename)
            ER = {}
            for k in range(size_mini_batch):
                ER[chem_names_list[j:j+mini_batch][k]] = ee.Image(ER_i.get(k))

            ER_images = list(ER.values())
            ER_image = ee.Image.cat(ER_images)
            geemap.ee_export_image(ee.Image(ER_image), filename=filename, scale=outputscale,file_per_band=False)

def process_date_range_global_5km_ST(date_range,input_variables,year):
    """
    --------------------------------------------------------------------------
    Function: process_date_range_global_5km_ST
    --------------------------------------------------------------------------
    Parameters:
        - date_range (list): A list of datetime strings in 'YYYY-MM-DD HH:MM:SS' format.
        - input_variables (dict): Dictionary containing model settings and I/O paths
        - year (int or str): Simulation year (used for land cover and DEFMAP retrieval).

    Returns:
        - None: Calls appropriate EMPORC emission functions for each hour in the date range
                and saves results as GeoTIFFs to disk.

    Description:
        This function automates global 5km EMPORC emission calculations using hourly time steps.
        It supports both static and dynamic PFT maps, with optional DEFMAP integration for fire-induced
        vegetation change. Unlike standard EMPORC workflows, this version uses **skin temperature**
        instead of 2-meter air temperature as the temperature input for emission response.
        Depending on the presence of ROI, either full-global or regional processing is triggered.

    Example:
        ```python
        process_date_range_global_5km_ST(
            date_range=["2020-07-01 00:00:00", "2020-07-01 03:00:00"],
            input_variables={
                "model_type": 'global_5000m_skin_temperature',
            },
            year="2020"
        )
        ```
    --------------------------------------------------------------------------
    """
    model_type = input_variables['model_type']
    DEFMAP_variables = input_variables['DEFMAP']
    chem_names = ee.List(input_variables['chem_names'])
    DEFMAP_type = input_variables['DEFMAP_type']
    COMPOUND_EMISSION_FACTORS = Echo_parameter.COMPOUND_EMISSION_FACTORS
        # 对每个年份的日期列表进行处理
    LUCC_collection_name = 'MODIS/061/MCD12C1'
    PFTS_year  = input_variables['PFTS_year']
    if model_type == 'global_5000m':
        Collection_name = "projects/gee-megan/assets/MCD64A1_5km"
        burnedAreaFraction,burnedDate = get_DEFMAP.get_burn_data_5000m(int(year))
    # elif model_type == 'global_500m':
    #     # Collection_name = "MODIS/061/MCD64A1"
    #     burnedAreaFraction,burnedDate = get_DEFMAP.get_burn_data_500m(int(year))

    if input_variables['PFTS_year'] == '':
        print('dynamic PFTS year')
        PFTS = METPreprocess.ReClassPFTS_all_type(int(year),LUCC_collection_name)
        year_p = int(year)-1
        if year_p < 2001:
            year_p = 2001

        PFTS_p = METPreprocess.ReClassPFTS_all_type(int(year_p),LUCC_collection_name)
        EFMAPS = METPreprocess.calculateChem(chem_names, COMPOUND_EMISSION_FACTORS, PFTS)
        EFMAPS_p = METPreprocess.calculateChem(chem_names, COMPOUND_EMISSION_FACTORS, PFTS_p)
        EFMAPS = ee.Image(ee.List(EFMAPS).slice(1).iterate(METPreprocess.add_bands, ee.Image(ee.List(EFMAPS).get(0))))
        EFMAPS_p = ee.Image(ee.List(EFMAPS_p).slice(1).iterate(METPreprocess.add_bands, ee.Image(ee.List(EFMAPS_p).get(0))))
        F_EFMAPs = EFMAPS
    else:
        print('constant PFTS year')
        PFTS = METPreprocess.ReClassPFTS_all_type(PFTS_year,LUCC_collection_name)
        EFMAPS = METPreprocess.calculateChem(chem_names, COMPOUND_EMISSION_FACTORS, PFTS)
        EFMAPS = ee.Image(ee.List(EFMAPS).slice(1).iterate(METPreprocess.add_bands, ee.Image(ee.List(EFMAPS).get(0))))
        F_EFMAPs = EFMAPS
    # Total plant percentage
    ContainPlantbandNames = PFTS.bandNames().slice(0, 15)
    PFTS_ContainPlant = PFTS.select(ContainPlantbandNames)
    TotalPTFS = PFTS_ContainPlant.reduce(ee.Reducer.sum())
    for date in date_range:
        date_Day, date_time = date.split(' ')
        Hour = [date_time.split(':')[0]]
        if model_type == 'global_5000m' and DEFMAP_variables == 'T':
            print(f'Dynamic EFMAPS is {date_Day}')
            DEFMAPS = get_DEFMAP.calculate_defmap(EFMAPS_p,EFMAPS, date_Day,burnedDate,burnedAreaFraction)
            F_EFMAPs = DEFMAPS 
            print(f'DEFMAPS is True {date_Day}')
            if DEFMAP_type == 'DEFMAP_only_fire':
                print(f'DEFMAPS is only fire')
                DEFMAPS_only_fire = get_DEFMAP.calculate_defmap_only_fire(EFMAPS, date_Day,burnedDate,burnedAreaFraction)
                F_EFMAPs = DEFMAPS_only_fire
            elif DEFMAP_type == 'DEFMAP__not_only_fire':
                print(f'DEFMAPS is couple lucc with fire')
                DEFMAPS = get_DEFMAP.calculate_defmap(EFMAPS_p,EFMAPS, date_Day,burnedDate,burnedAreaFraction)
                F_EFMAPs = DEFMAPS
        elif DEFMAP_variables == 'F':
            F_EFMAPs = EFMAPS
            print('Not use DEFMAPs, just EFMAPs')
           
            original_scale = F_EFMAPs.projection().nominalScale().getInfo()
            print(original_scale)
        chem_names = ee.List(input_variables['chem_names'])
        if input_variables['roi'] =='':
            Layers = input_variables['Layers']
            scale = input_variables['scale']
            output_scale = input_variables['output_scale']
            mini_batch = input_variables['mini_batch']
            out_dir = input_variables['out_dir']
            out_dir_filename = input_variables['out_dir_filename']
            TemporaryDirectory = input_variables['TemporaryDirectory']
            print('global')
            print(f'-------------------------------strat--{date}-------------------------------')
            Emporc_global_5km_without_roi_ST(date_Day, Hour,chem_names,PFTS_ContainPlant,
                TotalPTFS,F_EFMAPs,scale,output_scale,Layers,mini_batch,
            out_dir,out_dir_filename,TemporaryDirectory,input_variables)
            print(f'--------------------------------end--{date}--------------------------------')
        else:
            print('use roi {}'.format(input_variables['roi']))
            roi_1 = ee.Geometry.Rectangle(input_variables['roi'])

            coords = roi_1.coordinates()
        

            roi = ee.Geometry.Polygon(coords, None, False)
            # print(input_variables['roi'])
            # print(roi.getInfo())
            Layers = input_variables['Layers']
            scale = input_variables['scale']
            output_scale = input_variables['output_scale']
            mini_batch = input_variables['mini_batch']
            out_dir = input_variables['out_dir']
            out_dir_filename = input_variables['out_dir_filename']
            TemporaryDirectory = input_variables['TemporaryDirectory']
            # print(date)
            print(f'-------------------------------strat--{date}-------------------------------')
            Emporc_global_5km_with_roi_ST(date_Day, Hour,chem_names,PFTS_ContainPlant,
                TotalPTFS,F_EFMAPs,roi,scale,output_scale,Layers,mini_batch,
            out_dir,out_dir_filename,TemporaryDirectory,input_variables)
            print(f'--------------------------------end--{date}--------------------------------')

def process_date_range_global_500m(date_range,input_variables,year):
    """
    --------------------------------------------------------------------------
    Function: process_date_range_global_500m
    --------------------------------------------------------------------------
    Parameters:
        - date_range (list): List of datetime strings in 'YYYY-MM-DD HH:MM:SS' format.
        - input_variables (dict): Dictionary containing configuration and runtime parameters
        - year (int or str): Simulation year; overrides PFTS_year if model is dynamic.

    Returns:
        - None: Runs hourly emission rate calculations and saves GeoTIFF outputs to disk.

    Description:
        This function processes a list of hourly timestamps to compute BVOC emissions 
        at global 500m resolution using the EMPORC framework.

    Example:
        ```python
        process_date_range_global_500m(
            date_range=["2020-07-01 00:00:00", "2020-07-01 03:00:00"],
            input_variables={
                "model_type": "global_500m",
                 ...
            },
            year="2020"
        )
        ```
    --------------------------------------------------------------------------
    """
    DEFMAP_type = input_variables['DEFMAP_type']
    model_type = input_variables['model_type']
    DEFMAP_variables = input_variables['DEFMAP']
    chem_names = ee.List(input_variables['chem_names'])
    COMPOUND_EMISSION_FACTORS = Echo_parameter.COMPOUND_EMISSION_FACTORS
    LUCC_collection_name = "MODIS/061/MCD12Q1"
    # LUCC_collection_name = 'MODIS/061/MCD12C1'
    PFTS_year  = input_variables['PFTS_year']
    
    if model_type == 'global_500m':
        if input_variables['PFTS_year'] != '':
           year = PFTS_year
        # Collection_name = "MODIS/061/MCD64A1"
        print(f'Model type is global_500m for {year}')
        burnedAreaFraction,burnedDate = get_DEFMAP.get_burn_data_500m(int(year))
        



    if input_variables['PFTS_year'] == '':
        print(f'dynamic PFTS year for {year}')
        PFTS = METPreprocess.ReClassPFTS_all_type(int(year),LUCC_collection_name)
        year_p = int(year)-1
        if year_p < 2001:
            year_p = 2001

        PFTS_p = METPreprocess.ReClassPFTS_all_type(int(year_p),LUCC_collection_name)
        EFMAPS = METPreprocess.calculateChem(chem_names, COMPOUND_EMISSION_FACTORS, PFTS)
        EFMAPS_p = METPreprocess.calculateChem(chem_names, COMPOUND_EMISSION_FACTORS, PFTS_p)
        EFMAPS = ee.Image(ee.List(EFMAPS).slice(1).iterate(METPreprocess.add_bands, ee.Image(ee.List(EFMAPS).get(0))))
        EFMAPS_p = ee.Image(ee.List(EFMAPS_p).slice(1).iterate(METPreprocess.add_bands, ee.Image(ee.List(EFMAPS_p).get(0))))
        F_EFMAPs = EFMAPS
    else:
        print(f'constant PFTS year for {PFTS_year}')
        PFTS = METPreprocess.ReClassPFTS_all_type(int(PFTS_year),LUCC_collection_name)
        EFMAPS = METPreprocess.calculateChem(chem_names, COMPOUND_EMISSION_FACTORS, PFTS)
        EFMAPS = ee.Image(ee.List(EFMAPS).slice(1).iterate(METPreprocess.add_bands, ee.Image(ee.List(EFMAPS).get(0))))
        F_EFMAPs = EFMAPS
    ContainPlantbandNames = PFTS.bandNames().slice(0, 15)
    PFTS_ContainPlant = PFTS.select(ContainPlantbandNames)
    TotalPTFS = PFTS_ContainPlant.reduce(ee.Reducer.sum())
    for date in date_range:
        date_Day, date_time = date.split(' ')
        Hour = [date_time.split(':')[0]]
        if model_type == 'global_500m' and DEFMAP_variables == 'T':
            print(f'Dynamic EFMAPS is {date_Day}')
            if DEFMAP_type == 'DEFMAP_only_fire':
                print(f'DEFMAPS is only fire')
                DEFMAPS_only_fire = get_DEFMAP.calculate_defmap_only_fire(EFMAPS, date_Day,burnedDate,burnedAreaFraction)
                F_EFMAPs = DEFMAPS_only_fire
            elif DEFMAP_type == 'DEFMAP__not_only_fire':
                print(f'DEFMAPS is couple lucc with fire')
                DEFMAPS = get_DEFMAP.calculate_defmap(EFMAPS_p,EFMAPS, date_Day,burnedDate,burnedAreaFraction)
                F_EFMAPs = DEFMAPS

        chem_names = ee.List(input_variables['chem_names'])
        roi_1 = ee.Geometry.Rectangle(input_variables['roi'])
        coords = roi_1.coordinates()
        

        roi = ee.Geometry.Polygon(coords, None, False)
        # print(input_variables['roi'])
        # print(roi.getInfo())
        Layers = input_variables['Layers']
        scale = input_variables['scale']
        output_scale = input_variables['output_scale']
        mini_batch = input_variables['mini_batch']
        out_dir = input_variables['out_dir']
        out_dir_filename = input_variables['out_dir_filename']

        TemporaryDirectory = input_variables['TemporaryDirectory']
        # print(date)

        print(f'-------------------------------strat--{date}-------------------------------')
        Emporc_global_500m(date_Day, Hour,chem_names,PFTS_ContainPlant,
                TotalPTFS,F_EFMAPs,roi,scale,output_scale,Layers,mini_batch,
            out_dir,out_dir_filename,TemporaryDirectory,input_variables)
        print(f'--------------------------------end--{date}--------------------------------')

def check_roi_size(roi):
    """
    --------------------------------------------------------------------------
    Function: check_roi_size
    --------------------------------------------------------------------------
    Parameters:
        - roi (list): Region of interest represented as a list of four coordinates:
                      [min_lon, min_lat, max_lon, max_lat].

    Returns:
        - str: Message indicating whether the ROI is within the acceptable size limit.
               Returns an error message if the ROI exceeds 1 degree in either direction.

    Description:
        This function checks whether the provided region of interest (ROI)
        exceeds 1 degree in longitude or latitude. This is useful for ensuring
        the selected region is small enough for certain localized computations.

    Example:
        ```python
        check_roi_size([105.0, 30.0, 106.5, 31.0])
        # Output: "Error: The ROI exceeds the maximum allowed range of 1 degree in longitude or latitude."
        ```
    --------------------------------------------------------------------------
    """

    min_lon, min_lat, max_lon, max_lat = roi


    lon_diff = abs(max_lon - min_lon)
    lat_diff = abs(max_lat - min_lat)


    if lon_diff > 1 or lat_diff > 1:
        return "Error: The ROI exceeds the maximum allowed range of 1 degree in longitude or latitude."
    else:
        return "ROI is within the acceptable range."
    

def process_date_range_local(date_range,input_variables,year):
    """
    --------------------------------------------------------------------------
    Function: process_date_range_local
    --------------------------------------------------------------------------
    Parameters:
        - date_range (list): A list of datetime strings in 'YYYY-MM-DD HH:MM:SS' format.
        - input_variables (dict): Dictionary containing model settings and configurations:
            - 'model_type' (str): 'local'.
            - 'chem_names' (list): List of chemical species (e.g., ['ISOP').
            - 'DEFMAP' (str): 'T' or 'F' to control use of DEFMAP correction.
            - 'DEFMAP_type' (str): Either 'DEFMAP_only_fire' or 'DEFMAP__not_only_fire'.
            - 'PFTS_year' (str or int): Year of PFT classification, or '' for dynamic mode.
            - 'roi' (list): List representing bounding box: [min_lon, min_lat, max_lon, max_lat].
            - 'scale' (int): Processing resolution in meters (e.g., 10, 30).
            - 'output_scale' (int): Output resolution for GeoTIFFs.
            - 'Layers' (int): Number of canopy layers.
            - 'mini_batch' (int): Number of species per batch.
            - 'local_resolution' (int or str): Local resolution (10 or 30 meters).
            - 'evaluate' (bool): Whether to perform model evaluation using ML.
            - 'train_ratio' (float): Training ratio for machine learning split.
            - 'cv_folds' (int): Number of cross-validation folds.
            - 'output_dir_ML' (str): Directory to store ML evaluation results.
            - 'out_dir' (str): Output root directory.
            - 'out_dir_filename' (str): Subdirectory name for outputs.
            - 'TemporaryDirectory' (str): Subdirectory for intermediate data.

        - year (int or str): Simulation year, used for LUCC, fire, and PFT lookup.

    Returns:
        - None: This function executes EMPORC emission calculations for each timestamp
                in the input date range and saves the result to disk.

    Description:
        This function runs local or global EMPORC BVOC emission simulations for a series of timestamps
        within a specified ROI. It dynamically selects LUCC maps and optionally applies DEFMAP corrections.
        In local mode, it supports Sentinel-2 (10 m) or Landsat (30 m) LUCC classification, and optionally
        runs a machine learning evaluation pipeline if `evaluate` is set to True.

        For each datetime in the range, this function constructs vegetation maps (PFTS), emission factors
        (EFMAPS), and handles DEFMAP adjustment (if enabled), before calling the `Emporc_local` core function
        to compute and export hourly emission rates.

    Example:
        ```python
        process_date_range_local(
            date_range=["2020-07-01 00:00:00", "2020-07-01 03:00:00"],
            input_variables={
                "model_type": "local",
                ...
            },
            year="2020"
        )
        ```
    --------------------------------------------------------------------------
    """
    roi_size = input_variables['roi'] 
    date_Day, date_time = date_range[0].split(' ')

    check_roi_size(roi_size)
    roi_1 = ee.Geometry.Rectangle(input_variables['roi'])
    coords = roi_1.coordinates()
        

    roi = ee.Geometry.Polygon(coords, None, False)
  
    DEFMAP_type = input_variables['DEFMAP_type']
    model_type = input_variables['model_type']
    DEFMAP_variables = input_variables['DEFMAP']
    chem_names = ee.List(input_variables['chem_names'])
    local_resolution = input_variables['local_resolution']

    evaluate = input_variables["evaluate"]                  # bool
    train_ratio = float(input_variables["train_ratio"])     # float
    cv_folds = int(input_variables["cv_folds"])             # int
    output_dir_ML= str(input_variables["output_dir_ML"])    # str
    eval_config = None
    if evaluate == "Ture":
        eval_config = {
            "evaluate": True,
            "train_ratio": train_ratio,
            "cv_folds": cv_folds,
            "output_dir": output_dir_ML,
            "date_Day": date_Day
        }

    print(type(local_resolution))
    print(local_resolution)
    COMPOUND_EMISSION_FACTORS = Echo_parameter.COMPOUND_EMISSION_FACTORS
    LUCC_collection_name = 'MODIS/061/MCD12C1'
    PFTS_year  = input_variables['PFTS_year']
    if model_type == 'global_5000m':
        print('model type is global_5000m')
        LUCC_collection_name = 'MODIS/061/MCD12C1'
        Collection_name = "projects/gee-megan/assets/MCD64A1_5km"
        burnedAreaFraction,burnedDate = get_DEFMAP.get_burn_data_5000m(int(year))
    elif model_type == 'global_500m':
        print('model type is global_500m')
        LUCC_collection_name = 'MODIS/061/MCD12Q1'
        Collection_name = "MODIS/061/MCD64A1"
        burnedAreaFraction,burnedDate = get_DEFMAP.get_burn_data_500m(int(year))
    elif model_type == 'local':
        # print('aaaa')
        LUCC_collection_name = 'local'
        Collection_name = "MODIS/061/MCD64A1"
        burnedAreaFraction,burnedDate = get_DEFMAP.get_burn_data_500m(int(year))
        if local_resolution == 10:
            print( 'run in res 10m')
            sentinel_class_10m =  get_local_lucc.classify_sentinel2_using_training(year, roi)
            LUCC_image = sentinel_class_10m
        elif local_resolution == 30:
            print( 'run in res 30m')
            date_Day_0, date_time_0  = date_range[0].split(' ')
            # Modis_class_30m =  get_local_lucc.classify_landsat_using_modis(year, roi)
            # LUCC_image = Modis_class_30m
            # glc_class_30m =  get_local_lucc.classify_landsat_using_glc(date_Day_0, roi)
            glc_class_30m =  get_local_lucc.classify_landsat_using_glc_ESA(date_Day_0, roi, eval_config = eval_config )
            LUCC_image = glc_class_30m
        elif local_resolution == '':
            print('GEE-MEGAN run in local must be set resolution')
            return 
    if input_variables['PFTS_year'] == '':
        year_p = int(year)-1
        if year_p < 2001:
            year_p = 2001
        print('dynamic PFTS year')
        if model_type == 'local':
            print('model type is local')
            PFTS = METPreprocess.ReClassPFTS_all_type(int(year),LUCC_collection_name,MODIS_Image=LUCC_image,roi=roi)
            PFTS_p = METPreprocess.ReClassPFTS_all_type(int(year_p),LUCC_collection_name,MODIS_Image=LUCC_image,roi=roi)
        elif model_type != 'local':
            print('model type is not local')
            PFTS = METPreprocess.ReClassPFTS_all_type(int(year),LUCC_collection_name)
            PFTS_p = METPreprocess.ReClassPFTS_all_type(int(year_p),LUCC_collection_name)
        print(year_p)
        
        EFMAPS = METPreprocess.calculateChem(chem_names, COMPOUND_EMISSION_FACTORS, PFTS)
        EFMAPS_p = METPreprocess.calculateChem(chem_names, COMPOUND_EMISSION_FACTORS,PFTS_p)
        EFMAPS = ee.Image(ee.List(EFMAPS).slice(1).iterate(METPreprocess.add_bands, ee.Image(ee.List(EFMAPS).get(0))))
        EFMAPS_p = ee.Image(ee.List(EFMAPS_p).slice(1).iterate(METPreprocess.add_bands, ee.Image(ee.List(EFMAPS_p).get(0))))
        F_EFMAPs = EFMAPS
    else:
        print('constant PFTS year')
        PFTS = METPreprocess.ReClassPFTS_all_type(int(PFTS_year),LUCC_collection_name)
        EFMAPS = METPreprocess.calculateChem(chem_names, COMPOUND_EMISSION_FACTORS, PFTS)
        EFMAPS = ee.Image(ee.List(EFMAPS).slice(1).iterate(METPreprocess.add_bands, ee.Image(ee.List(EFMAPS).get(0))))
        F_EFMAPs = EFMAPS
    # Total plant percentage
    ContainPlantbandNames = PFTS.bandNames().slice(0, 15)
    PFTS_ContainPlant = PFTS.select(ContainPlantbandNames)
    TotalPTFS = PFTS_ContainPlant.reduce(ee.Reducer.sum())
    for date in date_range:
        date_Day, date_time = date.split(' ')
        Hour = [date_time.split(':')[0]]
        if model_type == 'global_5000m' and DEFMAP_variables == 'T':
            print(f'Dynamic EFMAPS is {date_Day} 1')
            DEFMAPS = get_DEFMAP.calculate_defmap(EFMAPS_p,EFMAPS, date_Day,burnedDate,burnedAreaFraction)
            F_EFMAPs = DEFMAPS
        elif model_type == 'global_500m' and DEFMAP_variables == 'T':
            print(f'Dynamic EFMAPS is {date_Day} 2')
            DEFMAPS = get_DEFMAP.calculate_defmap(EFMAPS_p,EFMAPS, date_Day,burnedDate,burnedAreaFraction)
            F_EFMAPs = DEFMAPS
        elif model_type == 'local' and DEFMAP_variables == 'T':
            print(f'Dynamic EFMAPS is {date_Day} 3')
            if DEFMAP_type == 'DEFMAP_only_fire':
                print(f'DEFMAPS is only fire')
                DEFMAPS_only_fire = get_DEFMAP.calculate_defmap_only_fire(EFMAPS, date_Day,burnedDate,burnedAreaFraction)
                F_EFMAPs = DEFMAPS_only_fire
            elif DEFMAP_type == 'DEFMAP__not_only_fire':
                print(f'DEFMAPS is couple lucc with fire')
                DEFMAPS = get_DEFMAP.calculate_defmap(EFMAPS_p,EFMAPS, date_Day,burnedDate,burnedAreaFraction)
                F_EFMAPs = DEFMAPS
        chem_names = ee.List(input_variables['chem_names'])
        Layers = input_variables['Layers']
        scale = input_variables['scale']
        output_scale = input_variables['output_scale']
        mini_batch = input_variables['mini_batch']
        out_dir = input_variables['out_dir']
        out_dir_filename = input_variables['out_dir_filename']
        TemporaryDirectory = input_variables['TemporaryDirectory']
            
        print(f'-------------------------------strat--{date}-------------------------------')

        Emporc_local(date_Day, Hour,chem_names,PFTS_ContainPlant,
                TotalPTFS,F_EFMAPs,roi,scale,output_scale,Layers,mini_batch,
            out_dir,out_dir_filename,TemporaryDirectory)
        print(f'--------------------------------end--{date}--------------------------------')        

def parse_dict(arg_value):
    """
    --------------------------------------------------------------------------
    Function: parse_dict
    --------------------------------------------------------------------------
    Parameters:
        - arg_value (str): string representation of a Python dict

    Returns:
        - dict: parsed dictionary

    Description:
        Safely parses a string literal into a Python dict using ast.literal_eval.
        Raises argparse.ArgumentTypeError if parsing fails or result is not a dict.

    Mathematical Details:
        - None

    Example:
        ```python
        d = parse_dict("{'a':1, 'b':2}")
        ```
    """
    try:
        result = ast.literal_eval(str(arg_value))
        if not isinstance(result, dict):
            raise ValueError("Parsed value is not a dictionary")
    except (SyntaxError, ValueError) as e:
        raise argparse.ArgumentTypeError(f"Invalid dictionary format: {e}")
    return result

def main(namelist_path  ,start_date, end_date,perturbation_dict=None):
    """
    --------------------------------------------------------------------------
    Function: main
    --------------------------------------------------------------------------
    Parameters:
        - namelist_path (str): Path to the configuration (namelist) file containing model parameters.
        - start_date (str): Start date for the simulation in 'YYYY-MM-DD HH:MM:SS' format.
        - end_date (str): End date for the simulation in 'YYYY-MM-DD HH:MM:SS' format.

    Returns:
        - None: Executes the full GEE-MEGAN emission simulation pipeline for the specified date range.

    Description:
        The main driver function for GEE-MEGAN emission processing. It reads model settings from 
        a namelist file, determines the model type, generates a list of datetime strings based on
        the specified timestep, and delegates control to the appropriate processing function:
        - `process_date_range_global_5km` for global 5 km simulations (air temperature)
        - `process_date_range_global_5km_ST` for global 5 km simulations using skin temperature
        - `process_date_range_global_500m` for high-resolution global (500m) simulation
        - `process_date_range_local` for fine-scale local simulations (10m / 30m)

    Example:
        ```bash
        python main_all_type.py ./namelist '2012-04-01 00:00:00' '2012-04-01 00:00:00'
        ```

    Notes:
        - The function assumes the presence of a valid input namelist in a readable format.
        - It requires Earth Engine (ee) and other dependencies to be initialized prior to execution.
    --------------------------------------------------------------------------
    """
    print(f"Running script from {start_date} to {end_date}")
    
    #
    print('main')
    # namelist_path = r'./input/GEE_MEGAN_namelist'
    input_variables = read_parameters_from_file(namelist_path)
    start_day = start_date
    end_day = end_date
    PFTsyear = start_day[:4]
    time_step = input_variables['time_step']
    day_list = generate_datetime_list(start_day, end_day, time_step)
    model_type = input_variables['model_type']
    if model_type == 'global_5000m' :
        print(f'Model type is {model_type}')
        process_date_range_global_5km(day_list,input_variables,int(PFTsyear))
    elif model_type == 'global_5000m_skin_temperature' :
        print(f'Model type is {model_type}')
        process_date_range_global_5km_ST(day_list,input_variables,int(PFTsyear))
    elif model_type == 'global_500m' :
        print(f'Model type is {model_type}')
        process_date_range_global_500m(day_list,input_variables,int(PFTsyear))
    elif model_type == 'local' :
        print(f'Model type is {model_type}')
        process_date_range_local(day_list,input_variables,int(PFTsyear))   

if __name__ == "__main__":
    '''
    Example:
        python main_all_type.py ./namelist '2012-04-01 00:00:00' '2012-04-01 00:00:00'
    
    '''
    parser = argparse.ArgumentParser(description="Process a range of dates.")
    parser.add_argument('name_list', type=str, help='Model parameters path')
    parser.add_argument('start_date', type=str, help='The start date to process (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('end_date', type=str, help='The end date to process (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--perturbation_dict', type=parse_dict, nargs='?', help='Directory for perturbation', default=None)
    
    args = parser.parse_args()
    
    main(args.name_list,args.start_date, args.end_date,args.perturbation_dict)
