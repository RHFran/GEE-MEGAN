# ------------------------------------------------------------------------------
# Module: EMPORC.py
# ------------------------------------------------------------------------------
# Description:
#     Main workflow for retrieving Leaf Area Index (LAI), vegetation cover,
#     processing Earth Engine images, and computing emission rates (ER)
#     for a list of chemical species using the EMPORC methodology.
#
# Inputs:
#     - chem_names (ee.List): list of chemical species strings to process.
#     - date_str (str): target dates for LAI and Landsat retrieval in 'YYYY-MM-DD' format.
#     - roi (ee.Geometry): region of interest for image clipping.
#     - PFT fraction image and PFTSNamesList (ee.List) for land‐cover weighting.
#     - Emission factor maps (EFMAPS) as an ee.Image.
#
# Outputs:
#     - A list of EE Images (one per chemical species) containing the computed
#       emission rates as per-species bands.
#
# Functions:
#     - reduce_image_resolution(image, multiplier) -> ee.Image
#     - reduce_mean_image_resolution(image, multiplier) -> ee.Image
#     - find_LAI_new(date_str, collectionName) -> (ee.Image, ee.Image)
#     - find_VCF_image(date_str, dataset) -> ee.Image
#     - adjust_LAI_by_VCF(LAI_image, VCF_image) -> ee.Image
#     - find_clear_landsat_image(before_date, after_date, roi) -> ee.Image or None
#     - calculate_lai_from_landsat(image) -> ee.Image
#     - calculate_time_intervals(before_date_str, after_date_str, landsat_date_str) -> (int, int)
#     - get_image_date(image) -> str
#     - correct_lai_factor(lai_start, lai_end, lai_landsat, time_interval, total_time) -> ee.Image
#     - get_local_LAI(date_str, collectionName, roi=None) -> (ee.Image, ee.Image)
#     - get_MET(date, collectionId, bandNameList) -> dict of ee.ImageCollections
#     - ER_calculate(TSTLEN, Day, Hour, lat, Tc, T24, T240, PPFD, PPFD24, PPFD240,
#                    Wind, Humidity, Pres, DI, lai_c, lai_p, Layers,
#                    TotalPTFS, EFMAPS, chem_names, PFTS, PFTSNamesList) -> ee.List(ee.Image)
#
# Main Functionality:
#     1. Define species list (chem_names) and import required libraries.
#     2. Provide helper functions for image‐resolution reduction.
#     3. Retrieve and adjust MODIS and Landsat LAI with vegetation cover.
#     4. Filter meteorological data via ERA5 preprocess library.
#     5. Compute emission rates for each species in ER_calculate.
#
# Dependencies:
#     - math
#     - ee
#     - os
#     - geemap
#     - datetime, timedelta
#     - Echo_parameter
#     - canopy_lib
#     - ECMWF_ERA5_Preprocess_lib
#     - gamma_lib
#     - functools.partial
# ------------------------------------------------------------------------------
import math
import ee
import os
import geemap
from datetime import datetime, timedelta
import Echo_parameter as Echo_parameter
import canopy_lib as canopy
import ECMWF_ERA5_Preprocess_lib as METPreprocess
import gamma_lib as gamma
from functools import partial


# Define list of chemical species for EMPORC calculation
chem_names = ee.List(['ISOP', 'MYRC', 'SABI', 'LIMO', 
                  'A_3CAR', 'OCIM', 'BPIN', 'APIN', 
                  'OMTP', 'FARN', 'BCAR', 'OSQT', 
                  'MBO', 'MEOH', 'ACTO', 'CO', 
                  'NO', 'BIDER', 'STRESS', 'OTHER'])


def reduce_image_resolution(image, multiplier):
    """
    --------------------------------------------------------------------------
    Function: reduce_image_resolution
    --------------------------------------------------------------------------
    Parameters:
        - image (ee.Image): input image to be downscaled
        - multiplier (int): number of pixels per row to combine, total of multiplier^2 pixels aggregated

    Returns:
        - ee.Image: output low resolution image
            - pixel values equal sum of original pixels in each block

    Description:
        Aggregates blocks of pixels in the input image to create a lower-resolution output image.
        Uses mean reduction followed by scaling to preserve total pixel values.
        Handles large pixel blocks by specifying maxPixels parameter.

    Mathematical Details:
        - mean_block: computes mean of pixels over multiplier x multiplier window.
        - scale_back: multiplies mean by multiplier^2 to approximate sum of pixels.

    Example:
        ```python
        - result = reduce_image_resolution(input_img, 4)
        ```
    """
    # Get original image projection
    original_projection = image.projection()

    # Get original scale (meters per pixel)
    original_scale = image.projection().nominalScale()

    # Compute new scale by multiplying original scale by multiplier
    new_scale = original_scale.multiply(multiplier)

     # Define mean reducer for pixel aggregation
    reducer = ee.Reducer.mean()

    # Reduce resolution using the reducer and reproject to new scale
    low_res_image = image.reduceResolution(
        reducer=reducer,
        maxPixels=1024
    ).reproject(
        crs=original_projection,
        scale=new_scale
    )

    # Multiply pixel values to preserve sum after mean reduction
    low_res_image = low_res_image.multiply(multiplier * multiplier)

    return low_res_image

def reduce_mean_image_resolution(image, multiplier):
    """
    --------------------------------------------------------------------------
    Function: reduce_mean_image_resolution
    --------------------------------------------------------------------------
    Parameters:
        - image (ee.Image): input image to be downscaled
        - multiplier (int): number of pixels per row to combine, total of multiplier^2 pixels aggregated

    Returns:
        - ee.Image: output low resolution image
            - pixel values represent mean of original pixels in each block

    Description:
        Aggregates blocks of pixels in the input image by computing the mean value
        over multiplier x multiplier windows, then reprojects to the new scale.
        Useful for generating coarser-resolution images with averaged pixel values.

    Mathematical Details:
        - mean_block: computes mean of pixels over multiplier x multiplier window.

    Example:
        ```python
        - result = reduce_mean_image_resolution(input_img, 4)
        ```
    """
    # Get original image projection
    original_projection = image.projection()

    # Get original scale (meters per pixel)
    original_scale = image.projection().nominalScale()

    # Compute new scale by multiplying original scale by multiplier
    new_scale = original_scale.multiply(multiplier)

    # Define mean reducer for pixel aggregation
    reducer = ee.Reducer.mean()

    # Reduce resolution using the mean reducer and reproject to new scale
    low_res_image = image.reduceResolution(
        reducer=reducer,
        maxPixels=1024
    ).reproject(
        crs=original_projection,
        scale=new_scale
    )



    return low_res_image

def find_LAI(date_str, collectionName):
    """
    --------------------------------------------------------------------------
    Function: find_LAI
    --------------------------------------------------------------------------
    Parameters:
        - date_str (str): target date in 'YYYY-MM-DD' format
        - collectionName (str): name of the LAI image collection to query

    Returns:
        - ee.Image: latest_image (LAI image for most recent period)
        - ee.Image: previous_image (LAI image for prior period)

    Description:
        Finds the image immediately before the target date and the image at or
        after the target date.

    Example:
        ```python
        - latest, previous = find_LAI('2025-06-17', 'MODIS/061/MCD15A3H')
        ```
    """
    
    # target_date = ee.Date(date_str+'T'+hour_str+':00:00')
    # Shift target_date forward one day to include images from the same day
    target_date = ee.Date(date_str).advance(1, 'day')
    # Compute date one year before target_date
    one_year_ago = target_date.advance(-1, 'year')
    
   
    imageCollection = ee.ImageCollection(collectionName).sort('system:time_start', True)
    
    
    first_image_date = ee.Image(imageCollection.first()).date().format()
    last_image_date = ee.Image(imageCollection.sort('system:time_start', False).first()).date().format()
    
    # Check if target_date is within the collection date range
    if target_date.millis().getInfo() < ee.Date(first_image_date).millis().getInfo() or target_date.millis().getInfo() > ee.Date(last_image_date).millis().getInfo():
        raise ValueError('The target date is out of range. Please choose a date between {} and {}.'.format(first_image_date.getInfo(), last_image_date.getInfo()))
    
    # Filter images between one year ago and target_date, sort descending
    imageCollection = imageCollection.filterDate(one_year_ago, target_date).sort('system:time_start', False)

     # Select the most recent image
    latest_image = ee.Image(imageCollection.first())

    # Select the previous image if available, else reuse latest_image
    previous_image = ee.Image(imageCollection.toList(2).get(1)) if imageCollection.size().getInfo() > 1 else latest_image
    
    return latest_image.select('Lai'), previous_image.select('Lai')

def find_LAI_new(date_str, collectionName):
    """
    --------------------------------------------------------------------------
    Function: find_LAI_new
    --------------------------------------------------------------------------
    Parameters:
        - date_str (str): target date in 'YYYY-MM-DD' format
        - collectionName (str): name of the LAI image collection to query

    Returns:
        - previous_image (ee.Image): LAI image for period before target date
        - after_image (ee.Image): LAI image for period after (or within) target date

    Description:
        Finds the image immediately before the target date and the image at or
        after the target date.

    Example:
        ```python
        - prev, aft = find_LAI_new('2025-06-17', 'MODIS/061/MCD15A3H')
        ```
    """
    # Initialize LAI band name
    LAI_str = ''
    selected_date = ee.Date(date_str)
    one_month_before = ee.Date(date_str).advance(-1, 'month')
    one_month_after = ee.Date(date_str).advance(1, 'month')
  
    # Determine LAI band name based on collection
    if collectionName == 'projects/sat-io/open-datasets/BU_LAI_FPAR/wgs_5km_8d':
        LAI_str = 'LAI'
    elif collectionName == "MODIS/061/MCD15A3H":
        LAI_str = 'Lai'
    elif collectionName == "MODIS/061/MOD15A2H": # 8
        # print('ok')
        LAI_str = 'Lai_500m'

    # Load and sort the image collection by time ascending
    imageCollection = ee.ImageCollection(collectionName).sort('system:time_start', True)
    previous_Collection = imageCollection.filterDate(one_month_before, selected_date)

    previous_image = imageCollection.filterDate(one_month_before, selected_date).sort('system:time_start', False).first().select(LAI_str)
    previous_image_date = previous_image.date().format()
    after_image = imageCollection.filterDate(selected_date, one_month_after).sort('system:time_start', True).first().select(LAI_str)
    after_image_date = after_image.date().format()

    # if selected_date.millis().getInfo() < ee.Date(previous_image_date).millis().getInfo() or selected_date.millis().getInfo() > ee.Date(after_image_date).millis().getInfo():
    #     raise ValueError('The target date is out of range. Please choose a date between {} and {}.'.format(previous_image_date.getInfo(), after_image_date.getInfo()))
    # previous_image = ee.Image(previous_Collection.toList(2).get(1)) if previous_Collection.size().getInfo() > 1 else after_image
    
    return previous_image, after_image

def find_VCF_image(date_str, dataset='MODIS/061/MOD13C1'):
    """
    ------------------------------------------------------------------------------
    Function: find_VCF_image
    ------------------------------------------------------------------------------
    Parameters:
        - date_str (str): Target date in 'YYYY-MM-DD' format.
        - dataset (str): GEE dataset ID. Supports:
            - 'MODIS/061/MOD13C1'  # NDVI 0.05 deg (~5km), 16-day composite
            - 'MODIS/006/MOD44B'  # Annual VCF  250m


    Returns:
        - vcf_image (ee.Image): Vegetation Cover Fraction (0-1) image.

    Description:
        Loads and processes a specified VCF or NDVI dataset, returning a VCF-like
        image (fractional vegetation cover) using the normalized NDVI formula or
        built-in percent vegetation layers.
        
        NDVI to VCF: Carlson, T. N., & Ripley, D. A. (1997). On the relation between NDVI, fractional 
        vegetation cover, and leaf area index. Remote sensing of Environment, 62(3), 241-252.

    Example:
        ```python
        vcf_img = find_VCF_image('2020-07-15', 'MODIS/061/MOD13C1')
        ```
    """

    date = ee.Date(date_str)
    ndvi_soil=0.2
    ndvi_veg=0.86
    if dataset == 'MODIS/061/MOD13C1':
        # 16-day composite, 5km NDVI
        image = ee.ImageCollection(dataset) \
                    .filterDate(date, date.advance(16, 'day')) \
                    .select('NDVI') \
                    .mean() \
                    .multiply(0.0001)

        vcf = image.subtract(ndvi_soil).divide(ndvi_veg - ndvi_soil).clamp(0, 1)

    elif dataset == 'MODIS/006/MOD44B':
        # Annual VCF: Percent_Tree_Cover + Percent_NonTree_Vegetation
        image = ee.ImageCollection(dataset) \
                    .filterDate(date.get('year').format().cat('-01-01'),
                                date.get('year').format().cat('-12-31')) \
                    .first()

        vcf = image.select('Percent_Tree_Cover') \
                   .add(image.select('Percent_NonTree_Vegetation')) \
                   .divide(100) \
                   .clamp(0, 1)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return vcf.rename('VCF')

def adjust_LAI_by_VCF(LAI_image, VCF_image):
    """
    ------------------------------------------------------------------------------
    Function: adjust_LAI_by_VCFne
    ------------------------------------------------------------------------------
    Parameters:
        - LAI_image (ee.Image): Original LAI image
        - VCF_image (ee.Image): Vegetation Cover Fraction image (0-1 range).

    Returns:
        - adjusted_LAI (ee.Image): Adjusted LAI (LAI_v = LAI / VCF), 
                                   where VCF < 0.05 is replaced with 1 to preserve original LAI.
                                   Final result is capped at LAI_v <= 10.

    Description:
        For vegetated areas (VCF >= 0.05), LAI is divided by VCF to reflect per-vegetation-area LAI.
        For sparse/non-vegetated areas (VCF < 0.05), LAI is kept unchanged by setting VCF = 1.
        Final LAI_v is capped to a maximum value of 10 to avoid unrealistic outputs.

    Example:
        ```python
        laiv_img = adjust_LAI_by_VCFne(lai_img, vcf_img)
        ```
    """
   
    safe_vcf = VCF_image.where(VCF_image.lt(0.05), 1.0)
    laiv = LAI_image.divide(safe_vcf)
    laiv_capped = laiv.clamp(0, 10)

    return laiv_capped

def get_image_dates(lai_image_before, lai_image_after):
    """
    --------------------------------------------------------------------------
    Function: get_image_dates
    --------------------------------------------------------------------------
    Parameters:
        - lai_image_before (ee.Image): LAI image acquired before target date
        - lai_image_after (ee.Image): LAI image acquired after target date

    Returns:
        - before_date (str): formatted date 'YYYY-MM-DD' of the before image
        - after_date (str): formatted date 'YYYY-MM-DD' of the after image

    Description:
        Extracts the acquisition dates from two Earth Engine images and returns
        them as formatted strings.

    Example:
        ```python
        - before, after = get_image_dates(img_before, img_after)
        ```
    """
    def get_image_date_(image):
        date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
        return date.getInfo()

    before_date = get_image_date_(lai_image_before)
    after_date = get_image_date_(lai_image_after)

    return  before_date, after_date

def find_clear_landsat_image(before_date, after_date,roi):
    """
    --------------------------------------------------------------------------
    Function: find_clear_landsat_image
    --------------------------------------------------------------------------
    Parameters:
        - before_date (str): start date in 'YYYY-MM-DD' format
        - after_date (str): end date in 'YYYY-MM-DD' format
        - roi (ee.Geometry): region of interest for image filtering and clipping

    Returns:
        - ee.Image or None: least cloudy, masked Landsat 7 image clipped to ROI,
          or None if no images are available

    Description:
        Retrieves Landsat 7 Level 2 surface reflectance images within the given
        date range over the specified ROI, applies cloud and saturation masks,
        and returns the image with the lowest cloud cover.

    Example:
        ```python
        - clear_img = find_clear_landsat_image('2025-06-01', '2025-06-30', my_roi)
        ```
    """
    point = roi
    start_date = ee.Date(before_date)
    end_date = ee.Date(after_date)

    def mask_landsat7_sr(image):
        qa_mask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
        saturation_mask = image.select('QA_RADSAT').eq(0)

        optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermal_band = image.select('ST_B6').multiply(0.00341802).add(149.0)

        masked_image = image.addBands(optical_bands, None, True) \
                            .addBands(thermal_band, None, True) \
                            .updateMask(qa_mask) \
                            .updateMask(saturation_mask)

        return masked_image


    landsat_collection = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2") \
        .filterDate(start_date, end_date) \
        .filterBounds(point)


    if landsat_collection.size().getInfo() > 0:

        landsat_collection = landsat_collection.map(mask_landsat7_sr)

        clear_landsat_image = landsat_collection.sort('CLOUD_COVER').first()
        return clear_landsat_image.clip(roi)
    else:
        return None
    
def calculate_lai_from_landsat(image):
    """
    --------------------------------------------------------------------------
    Function: calculate_lai_from_landsat
    --------------------------------------------------------------------------
    Parameters:
        - image (ee.Image): input Landsat surface reflectance image

    Returns:
        - ee.Image: concatenated bands of LAI estimates (NDVI_LAI, OSAVI_LAI,
          EVI2_LAI, MTVI2_LAI)

    Description:
        Calculates Leaf Area Index from a Landsat image using empirical
        relationships with NDVI, OSAVI, EVI2, and MTVI2 indices. 

    Mathematical Details:
        - NDVI = (B4 - B3) / (B4 + B3)
        - OSAVI = 1.16 * (B4 - B3) / (B4 + B3 + 0.16)
        - EVI2 = 2.5 * (B4 - B3) / (B4 + 2.4*B3 + 1)
        - MTVI2 = 1.5 * [(1.2*(B4 - B2) - 2.5*(B3 - B2))] / sqrt((2*B4 + 1)^2 - (6*B4 - 5*sqrt(B3) - 0.5))
        - LAI = -1/coef * log((index * -coef + 1) * mult)
        - Liu, J., Pattey, E., & Jégo, G. (2012). Assessment of vegetation indices for regional crop green LAI estimation from Landsat images over multiple growing seasons.
          Remote Sensing of Environment, 123, 347-358.

    Example:
        ```python
        - lai_img = calculate_lai_from_landsat(my_landsat_image)
        ```
    """
    # Calculate NDVI
    ndvi = image.normalizedDifference(['SR_B4', 'SR_B3']).rename('NDVI')
    B2 = image.select('SR_B2')
    B3 = image.select('SR_B3')
    B4 = image.select('SR_B4')
    OSAVI = ((B4.subtract(B3)).multiply(1.16)).divide(B4.add(B3).add(0.16))
    EVI2 = ((B4.subtract(B3)).multiply(2.5)).divide(B4.add(B3.multiply(2.4)).add(1))
    MTVI2 = (((B4.subtract(B2)).multiply(1.2)).subtract((B3.subtract(B2)).multiply(2.5))).multiply(1.5).divide(
            ((((B4.multiply(2)).add(1)).pow(2)).subtract(
                (B4.multiply(6).subtract((B3.sqrt()).multiply(5))).subtract(0.5))).sqrt())
    # Define coefficients for the empirical LAI-NDVI relationship
    # These coefficients need to be defined based on empirical research or literature

    # Calculate LAI using the empirical relationship
    # LAI = a * NDVI + b
    NDVI_LAI = ((((ndvi.multiply(1.017).multiply(-1)).add(1)).multiply(1.081)).log()).multiply(-1).divide(0.647)
    OSAVI_LAI = ((((OSAVI.multiply(1.058).multiply(-1)).add(1)).multiply(1.128)).log()).multiply(-1).divide(0.459)
    EVI2_LAI = ((((EVI2.multiply(0.910).multiply(-1)).add(1)).multiply(1.102)).log()).multiply(-1).divide(0.273)
    MTVI2_LAI = ((((MTVI2.multiply(0.819).multiply(-1)).add(1)).multiply(1.027)).log()).multiply(-1).divide(0.245)
    outputImage = ee.Image.cat([NDVI_LAI,OSAVI_LAI,EVI2_LAI,MTVI2_LAI])\
                          .rename([
                            ee.String('NDVI_LAI'),
                            ee.String('OSAVI_LAI'),
                            ee.String('EVI2_LAI'),
                            ee.String('MTVI2_LAI')])
    return outputImage

def calculate_time_intervals(before_date_str, after_date_str, landsat_date_str):
    """
    ---------------------------------------------------------------------------
    Function: calculate_time_intervals
    ---------------------------------------------------------------------------
    Parameters:
        - before_date_str (str): The date string for the before date ('YYYY-MM-DD' format).
        - after_date_str (str): The date string for the after date ('YYYY-MM-DD' format).
        - landsat_date_str (str): The date string for the Landsat image date ('YYYY-MM-DD' format).

    Returns:
        - total_time (int): Total days between the before and after dates.
        - time_interval (int): Days between the before date and the Landsat image date.

    Description:
        Computes the number of days between two dates (before and after) and
        the number of days from the before date to the Landsat image date.

    Mathematical Details:
        - total_time = (after_date - before_date).days
        - time_interval = (landsat_date - before_date).days

    Example:
        ```python
        - total, interval = calculate_time_intervals('2025-06-01', '2025-06-30', '2025-06-15')
        ```
    """
    before_date = datetime.strptime(before_date_str, '%Y-%m-%d')
    after_date = datetime.strptime(after_date_str, '%Y-%m-%d')
    landsat_date = datetime.strptime(landsat_date_str, '%Y-%m-%d')

    total_time = (after_date - before_date).days
    time_interval = (landsat_date - before_date).days

    return total_time, time_interval


def get_image_date(image):
        date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
        return date.getInfo()

def correct_lai_factor(lai_start, lai_end, lai_landsat, time_interval, total_time):
    """
    --------------------------------------------------------------------------
    Function: correct_lai_factor
    --------------------------------------------------------------------------
    Parameters:
        - lai_start (ee.Image): LAI value at the start
        - lai_end (ee.Image): LAI value at the end 
        - lai_landsat (ee.Image): Landsat-derived LAI value on day x
        - time_interval (int): days from the start to the Landsat LAI day
        - total_time (int): total days between start and end LAI values

    Returns:
        - corrected_lai_f (ee.Image): corrected LAI value for the Landsat data

    Description:
        Performs linear interpolation between two MODIS LAI values to estimate
        the expected LAI at the Landsat acquisition date, then blends this
        estimate with the actual Landsat LAI using a 0.5/0.5 weighting and applies
        masking rules for extreme values.

    Example:
        ```python
        - corrected = correct_lai_factor(start_img, end_img, landsat_img, 2, 8)
        ```
    """
    lai_landsat = lai_landsat.reduce(ee.Reducer.mean()).rename('lai')
    lai_expected = lai_start.add(lai_end.subtract(lai_start).multiply(time_interval / total_time))
    lai_expected = lai_expected
    # Calculate the correction factor
    correction_lai =  (lai_expected.multiply(0.5)).add(lai_landsat.multiply(0.5))
    # print(lai_expected)
    # print(correction_factor )
    # Apply the correction factor to the Landsat LAI value
    # Apply the correction factor to the Landsat LAI value
    correction_lai = correction_lai.unmask(lai_expected)
    

    corrected_lai_f = correction_lai.where(lai_landsat.gte(4), lai_expected)

    return corrected_lai_f 

def get_local_LAI(date_str, collectionName,roi=None):
    """
    --------------------------------------------------------------------------
    Function: get_local_LAI
    --------------------------------------------------------------------------
    Parameters:
        - date_str (str): target date in 'YYYY-MM-DD' format
        - collectionName (str): name of the MODIS LAI image collection
        - roi (ee.Geometry or None): region of interest for clipping images

    Returns:
        - previous_image (ee.Image): LAI image before the target date
        - after_image (ee.Image): corrected LAI image after the target date

    Description:
        Retrieves MODIS LAI images for one month before and after the given date,
        clips and scales them, then attempts to correct the after-date LAI using
        a clear-sky Landsat image and interpolation.

    Example:
        ```python
        - prev, aft = get_local_LAI('2025-06-17', 'MODIS/061/MCD15A3H', my_roi)
        ```
    """
    LAIcollectionName = collectionName 
    lai_p, lai_c = find_LAI_new(date_str, LAIcollectionName)
    before_date, after_date = get_image_dates(lai_p, lai_c )
    lai_p = lai_p.clip(roi)
    lai_c =lai_c.clip(roi)
    lai_p = lai_p.multiply(0.1)
    lai_c = lai_c.multiply(0.1)
    
    landsat_image = find_clear_landsat_image(before_date, after_date,roi)
    if landsat_image==None:
        previous_image = lai_p
        after_image = lai_c
        print('no Landsat image could use for lai')
    elif landsat_image!=None:
        landsat_image_date = get_image_date(landsat_image)
        # print()
        # print(landsat_image_date)
        landsat_lai = calculate_lai_from_landsat(landsat_image)
        total_time, time_interval = calculate_time_intervals(before_date, after_date,landsat_image_date)
        lai_corret =  correct_lai_factor(lai_p, lai_c, landsat_lai , time_interval, total_time)
        previous_image = lai_p
        after_image = lai_corret
    
    return previous_image, after_image

def get_MET(date,collectionId,bandNameList):
    """
    --------------------------------------------------------------------------
    Function: get_MET
    --------------------------------------------------------------------------
    Parameters:
        - date (str or ee.Date): target date for filtering the collection
        - collectionId (str): identifier of the meteorological data collection
        - bandNameList (list of str): list of band names to retrieve

    Returns:
        - collectionDict (dict): mapping of band names to filtered EE image collections

    Description:
        Filters a meteorological collection by date for each specified band
        and returns a dictionary of the resulting image collections.

    Example:
        ```python
        - mets = get_MET('2025-06-17', 'ECMWF/ERA5_LAND/HOURLY', ['TEMP', 'RH'])
        ```
    """
    collectionDict = {}
    for bandName in bandNameList:
        collectionDict[bandName] = METPreprocess.filterByDate(date, collectionId, bandName)
    return collectionDict

def ER_calculate(TSTLEN, Day, Hour, lat,
                 Tc, T24, T240,PPFD ,PPFD24,PPFD240,
                 Wind, Humidity, Pres, DI, lai_c,lai_p,Layers,
                 TotalPTFS,EFMAPS,chem_names,PFTS,PFTSNamesList):
    """
    --------------------------------------------------------------------------
    Function: ER_calculate
    --------------------------------------------------------------------------
    Parameters:
        - TSTLEN (int): test length or time step length
        - Day (ee.Image): image of date values for each grid cell
        - Hour (ee.Image): image of hour values for each grid cell
        - lat (ee.Image): image of latitude values for each grid cell
        - Tc, T24, T240 (ee.Image): temperature, 24h avg temp, 240h avg temp
        - PPFD, PPFD24, PPFD240 (ee.Image): PPFD and its 24h, 240h averages
        - Wind (ee.Image): wind speed
        - Humidity (ee.Image): relative humidity
        - Pres (ee.Image): atmospheric pressure
        - DI (ee.Image): drought index (stomatal)
        - lai_c, lai_p (ee.Image): current and previous LAI
        - Layers (int): canopy layer count
        - TotalPTFS (ee.Image): total PFT fraction sum
        - EFMAPS (ee.Image): emission factor maps per species
        - chem_names (ee.List): list of chemical species names
        - PFTS (ee.Image): PFT fraction image
        - PFTSNamesList (ee.List): list of PFT band names

    Returns:
        - ER (ee.List): list of EE Images with ER (emission result) per species

    Mathematical Details:
        - TotalPTFS_masked: masks zero PFT fractions to avoid division errors.
        - ER_i = GAMMA_all_i * EFMAPS_i
    Description:
        Computes emission rates for specified chemical species using
        interpolation of MODIS LAI/local LAI, GAMMA canopy functions, PFT weights,
        and standard emission factors.

    Example:
        ```python
        - ER_list = ER_calculate(30, date_image, hour_image, lat_image,
                            Tc, T24, T240, PPFD, PPFD24, PPFD240,
                            Wind, Humidity, Pres, DI,
                            lai_c, lai_p, 5,
                            TotalPTFS, EFMAPS, chem_names, PFTS, PFTSNamesList)
        ```
    """
    # Load emission calculation parameters from configuration
    PDT_LOS_CP = Echo_parameter.PDT_LOS_CP
    LD_FCT = Echo_parameter.LD_FCT
    REA_SPC = Echo_parameter.REA_SPC
    REL_EM_ACT = Echo_parameter.REL_EM_ACT
    RHO_FCT =  ee.List(PDT_LOS_CP.get('RHO_FCT'))
    LDF_FCT = ee.List(LD_FCT.get('LDF_FCT'))
    LDF_SPC = ee.List(LD_FCT.get('LDF_SPC'))
    #
    chem_names_size = chem_names.size() 
    # Handle zero total PFT fraction to avoid division by zero
    TotalPTFS_lte_0 = TotalPTFS.lte(0)
    TotalPTFS_masked = TotalPTFS.updateMask(TotalPTFS_lte_0)
    #
    GAM_LHT = gamma.gamma_lai(lai_c)
    GAM_S = gamma.GAMMA_S(lai_c)

    def adjust_factors(index,SPC_NAME):
        Cantype = index
        PFTSNames_i = PFTSNamesList.get(index)
        PFTS_i = PFTS.select(ee.String(PFTSNames_i))
        GAMMA_CE_i = canopy.GAMMA_CE(Day, Hour, lat,
                        Tc, T24, T240,PPFD ,PPFD24,PPFD240,
                        Wind, Humidity, Pres, DI, lai_c,
                        Layers,ee.String(SPC_NAME),Cantype)
        GAMMA_TD = GAMMA_CE_i.select("Ea1CanopyFinal")
        GAMMA_TI = GAMMA_CE_i.select('EatiCanopySUM')
        ADJUST_FACTOR_LD = GAMMA_TD.multiply(PFTS_i) 
        ADJUST_FACTOR_LI = GAMMA_TI.multiply(PFTS_i) 
        outputImage =  ee.Image.cat([ADJUST_FACTOR_LD,ADJUST_FACTOR_LI])\
                                .rename(['ADJUST_FACTOR_LD', 'ADJUST_FACTOR_LI'])
        return outputImage

    def ER_func(index_SPC):
        SPC_NAME = ee.List(chem_names).get(index_SPC)
        SPC_NAME_n = ee.List(LDF_SPC).indexOf(SPC_NAME)

        GAM_AGE_i = gamma.GAMMA_A(Day, Hour, lai_c, lai_p, 
                                  TSTLEN, T24, SPC_NAME, REA_SPC, REL_EM_ACT)
        PFTSNames_i = PFTSNamesList.get(index_SPC)
        # PFTS_i = PFTS.select(ee.String(PFTSNames_i))

        gamma_CE = ee.List.sequence(0, ee.Number(15).subtract(1)).map(partial(adjust_factors, SPC_NAME=SPC_NAME))

        GAMMA_CE_BandName = ['ADJUST_FACTOR_LD','ADJUST_FACTOR_LI']
        GAMMA_CE_collection = ee.ImageCollection.fromImages(ee.Image(gamma_CE))

        GAMMA_CE_Update = canopy.updateDictList(GAMMA_CE_collection,GAMMA_CE_BandName,15)
        ADJUST_FACTOR_LD = ee.ImageCollection.fromImages(GAMMA_CE_Update.get("ADJUST_FACTOR_LD"))
        ADJUST_FACTOR_LI = ee.ImageCollection.fromImages(GAMMA_CE_Update.get("ADJUST_FACTOR_LI"))
        Total_ADJUST_FACTOR_LD = ADJUST_FACTOR_LD.reduce(ee.Reducer.sum())
        Total_ADJUST_FACTOR_LI = ADJUST_FACTOR_LI.reduce(ee.Reducer.sum())

        GAM_TLD = Total_ADJUST_FACTOR_LD.where(TotalPTFS.gt(0),Total_ADJUST_FACTOR_LD.divide(TotalPTFS_masked))\
                                        .where(TotalPTFS.lte(0),1)
        GAM_TLI = Total_ADJUST_FACTOR_LI.where(TotalPTFS.gt(0),Total_ADJUST_FACTOR_LI.divide(TotalPTFS_masked))\
                                        .where(TotalPTFS.lte(0),1)

        RHO_i = ee.Number(RHO_FCT.get(SPC_NAME_n))
        LDF_i = ee.Number(LDF_FCT.get(SPC_NAME_n))
        EFMAPS_i = EFMAPS.select(ee.String(SPC_NAME))
        ER_i = (GAM_AGE_i.multiply(GAM_S).multiply(RHO_i)\
                .multiply(((ee.Image(1).subtract(LDF_i)).multiply(GAM_TLI).multiply(GAM_LHT)).add(GAM_TLD.multiply(LDF_i))))\
                .multiply(EFMAPS_i)

        outputImage =  ee.Image.cat([ER_i]).rename([SPC_NAME])
        return outputImage
    ER = ee.List.sequence(0, ee.Number(chem_names_size).subtract(1)).map(ER_func)
    return ER
