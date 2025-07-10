# ------------------------------------------------------------------------------
# File: get_local_lucc.py
# ------------------------------------------------------------------------------
# Description:
#     Retrieve high-resolution local land use/land cover change (LUCC) and
#     classify Landsat imagery using combined GLC-FCS30D, ESA WorldCover, and
#     MODIS reference data with canopy-height and urban adjustments.
#
# Inputs:
#     - year (int): Target year for LUCC retrieval and classification.
#     - date (str): Date string in "YYYY-MM-DD" format for composite generation.
#     - roi (ee.Geometry or ee.Feature): Region of interest for clipping and processing.
#     - eval_config (dict, optional): Evaluation configuration for classification.
#
# Outputs:
#     - ee.Image: Unified LUCC map and final land cover classification
#       clipped to the ROI.
#
# Functions:
#     - classify_landsat_using_glc_ESA(date: str, roi: ee.Geometry, eval_config: dict=None) -> ee.Image
#         Generate Landsat composites around the date, sample and classify land cover
#         using GLC, ESA, and MODIS datasets, apply canopy-height and urban masks,
#         and return final classification.
#
# Main Functionality:
#     1. classify_landsat_using_glc_ESA:
#        a) Create Landsat composites.
#        b) Retrieve GLC, ESA, and MODIS reference maps.
#        c) Sample and train Random Forest classifiers with canopy height.
#
# Dependencies:
#     - ee
#     - datetime
#     - json
#     - os
# ------------------------------------------------------------------------------

import ee
import datetime 
import json
import os


def get_modis_landcover(year, roi):
    """
    --------------------------------------------------------------------------
    Function: get_modis_landcover
    --------------------------------------------------------------------------

    Parameters:
        - year (str): Year in 'YYYY' format
        - roi (ee.Geometry): region of interest for clipping

    Returns:
        - ee.Image: clipped land cover classification image (LC_Type1)

    Description:
        Retrieves the MODIS MCD12Q1 land cover dataset for the specified year,
        selects the LC_Type1 classification band, and clips it to the provided ROI.

    Mathematical Details:
        None (direct retrieval of precomputed classification).

    Example:
        ```python
        - result = get_modis_landcover('2020', my_roi)
        ```
    """

    dataset_id = "MODIS/061/MCD12Q1"

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    landcover = ee.ImageCollection(dataset_id) \
        .filterDate(start_date, end_date) \
        .first()  

    lc_type1 = landcover.select('LC_Type1')

    clipped_lc = lc_type1.clip(roi)
    
    return clipped_lc



# -----------------------------------------------------------------------------
# Mapping from GLC_FCS30D classes to MODIS IGBP classification codes
# -----------------------------------------------------------------------------
GLC_FCS30D_to_modis_mapping = {
    1:   [71,  72],                  # Evergreen needleleaf forests
    2:   [51,  52],                  # Evergreen broadleaf forests
    3:   [81,  82],                  # Deciduous broadleaf forests
    4:   [61,  62],                  # Mixed forests
    5:   [91,  92, 185],             # Closed shrublands
    6:   [120],                      # Woody savannas
    7:   [121, 122],                 # Savannas
    8:   [150],                      # Grasslands
    9:   [160],                      # Permanent wetlands
    10:  [130],                      # Croplands
    11:  [181, 182, 183, 184, 186, 187],  # Urban and built-up
    12:  [10,  11,  12,  20],         # Snow and ice
    13:  [190, 200],                 # Barren or sparsely vegetated
    14:  [152, 153],                 # Water bodies
    15:  [220],                      # Herbaceous wetlands
    16:  [201, 202],                 # Moss and lichen
    17:  [210]                       # Unclassified
}


def map_glc_to_modis(image):
    """
    --------------------------------------------------------------------------
    Function: map_glc_to_modis
    --------------------------------------------------------------------------
    Parameters:
        - image (ee.Image): input GLC_FCS30D classification image

    Returns:
        - ee.Image: remapped image with MODIS IGBP class codes

    Description:
        Remaps pixel values from GLC_FCS30D classes to MODIS IGBP classification
        codes using the GLC_FCS30D_to_modis_mapping dictionary.

    Example:
        ```python
        - result = map_glc_to_modis(glc_image)
        ```
    """
    remapped_image = image
    for modis_class, glc_classes in GLC_FCS30D_to_modis_mapping.items():
        for glc_class in glc_classes:
            remapped_image = remapped_image.where(image.eq(glc_class), modis_class)
    return remapped_image


def get_glc_fcs30d(year, roi):
    """
    --------------------------------------------------------------------------
    Function: get_glc_fcs30d
    --------------------------------------------------------------------------
    Parameters:
        - year (int): Year of interest (e.g., 2020).
        - roi (ee.Geometry or ee.Feature): Region of interest to clip the image.

    Returns:
        - ee.Image: Remapped and clipped land cover image named 'LC_Type1'.
            - Contains MODIS classification values.

    Description:
        This function retrieves the annual GLC-FCS30D land cover image for the specified year,
        remaps classification values to the MODIS scheme, and clips the result to the ROI.
        - Liangyun Liu, Xiao Zhang, & Tingting Zhao. (2023).GLC_FCS30D: the first global 30-m 
            land-cover dynamic monitoring product with fine classification system from 1985 to 2022 
            [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8239305

    Mathematical Details:
        None.

    Example:
        ```python
        result = get_glc_fcs30d(2020, my_roi)
        ```
    """
    # Load the annual GLC-FCS30D ImageCollection
    dataset = ee.ImageCollection("projects/sat-io/open-datasets/GLC-FCS30D/annual")
    
    # Select the band corresponding to the specified year
    band_name = f'b{year - 1999}'
    image = dataset.select(band_name).mosaic()
    
    # Remap classification values from GLC-FCS30D to MODIS classes
    remapped_image = map_glc_to_modis(image)
    
    clipped_image = remapped_image.clip(roi)
    
    return clipped_image.rename('LC_Type1')



def maskLandsat8(image):
    """
    --------------------------------------------------------------------------
    Function: maskLandsat8
    --------------------------------------------------------------------------
    Parameters:
        - image (ee.Image): input Landsat 8 image.

    Returns:
        - ee.Image: cloud-masked image with surface reflectance bands.
            - Bands SR_B1 through SR_B7 retained; original acquisition time preserved.

    Description:
        This function masks out clouds and shadows from a Landsat 8 image using QA_PIXEL flags,
        selects the surface reflectance bands, and copies over the original time_start property.

    Mathematical Details:
        None.

    Example:
        ```python
        result = maskLandsat8(image)
        ```
    """
    qa = image.select('QA_PIXEL')
    cloud_mask = qa.bitwiseAnd(1 << 5).eq(0).And(qa.bitwiseAnd(1 << 3).eq(0))
    return image.updateMask(cloud_mask).select(["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]).copyProperties(image, ["system:time_start"])


def get_landsat8_year_composite(year, roi):
    """
    --------------------------------------------------------------------------
    Function: get_landsat8_year_composite
    --------------------------------------------------------------------------
    Parameters:
        - year (int): Year for which to generate the composite (2013-2024).
        - roi (ee.Geometry or ee.Feature): Region of interest to filter and clip images.

    Returns:
        - ee.Image: Median composite of cloud-masked Landsat 8 images clipped to ROI.
            - Derived from images with <10% cloud cover, ensuring at least 6 images.

    Description:
        This function filters the Landsat 8 L2 collection by date, bounds, and cloud cover,
        retries with the previous year if too few images are found, applies a cloud mask,
        computes the median composite, and clips it to the ROI.

    Mathematical Details:
        None.

    Example:
        ```python
        result = get_landsat8_year_composite(2020, my_roi)
        ```
    """
    if year <= 2013:
        print('Landsat data can be used from 2013 to 2024.')

    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    print(f'testing {start_date} to {end_date}')
    while True:
        landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
            .filterBounds(roi) \
            .filterDate(start_date, end_date) \
            .sort('CLOUD_COVER')

        filtered_landsat8 = landsat8.filter(ee.Filter.lt('CLOUD_COVER', 10))

        count = filtered_landsat8.size().getInfo()
        if count > 5:
            break

        previous_year = year - 1
        start_date = f'{previous_year}-01-01'
        print(f'Insufficient data for year {year}, trying from year {previous_year}')

    masked_landsat8 = filtered_landsat8.map(maskLandsat8)
    composite = masked_landsat8.median()

    return composite.clip(roi)

def maskLandsat(image):
    """
    --------------------------------------------------------------------------
    Function: maskLandsat
    --------------------------------------------------------------------------
    Parameters:
        - image (ee.Image): input Landsat image.

    Returns:
        - ee.Image: cloud-masked image with selected surface reflectance bands.
            - Bands SR_B1, SR_B2, SR_B3, SR_B4, SR_B5, SR_B7 retained.

    Description:
        Masks out clouds and shadows using QA_PIXEL flags,
        selects specific surface reflectance bands, and preserves acquisition time.

    Mathematical Details:
        None.

    Example:
        ```python
        result = maskLandsat(image)
        ```
    """
    qa = image.select('QA_PIXEL')
    # Bit 5 and bit 3 are used for cloud and cloud shadow masks
    cloud_mask = qa.bitwiseAnd(1 << 5).eq(0).And(qa.bitwiseAnd(1 << 3).eq(0))
    return image.updateMask(cloud_mask).select(["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5",  "SR_B7"]).copyProperties(image, ["system:time_start"])


def get_landsat8_composite(start_date, end_date, roi):
    """
    --------------------------------------------------------------------------
    Function: get_landsat8_composite
    --------------------------------------------------------------------------
    Parameters:
        - start_date (str): start date in format 'YYYY-MM-DD'.
        - end_date (str): end date in format 'YYYY-MM-DD'.
        - roi (ee.Geometry or ee.Feature): region of interest to filter and clip images.

    Returns:
        - ee.Image or None: median composite of cloud-masked Landsat 8 images clipped to ROI.

    Description:
        This function filters the Landsat 8 L2 collection by date, bounds, and cloud cover,
        checks for sufficient data count, applies a cloud mask, computes the median composite,
        and clips the result to the ROI.

    Mathematical Details:
        None.

    Example:
        ```python
        result = get_landsat8_composite("2020-06-01", "2020-08-31", my_roi)
        ```
    """
    landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .sort('CLOUD_COVER')

    filtered_landsat8 = landsat8.filter(ee.Filter.lt('CLOUD_COVER', 10))
    
    count = filtered_landsat8.size().getInfo()
    print(count)
    if count < 10:
        print(f"No sufficient data available from {start_date} to {end_date}")
        return None

    masked_landsat8 = filtered_landsat8.map(maskLandsat8)
    composite = masked_landsat8.median()
    
    return composite.clip(roi)

def get_landsat_composite_by_dates(start_date, end_date, roi):
    """
    --------------------------------------------------------------------------
    Function: get_landsat_composite_by_dates
    --------------------------------------------------------------------------
    Parameters:
        - start_date (str): start date in 'YYYY-MM-DD'.
        - end_date (str): end date in 'YYYY-MM-DD'.
        - roi (ee.Geometry or ee.Feature): region of interest to filter and clip.

    Returns:
        - ee.Image or None: median composite of cloud-masked Landsat images clipped to ROI.

    Description:
        This function selects Landsat 5 or 8 collection based on year of start_date,
        filters by bounds, date, and cloud cover, applies cloud mask, computes median composite.

    Mathematical Details:
        None.

    Example:
        ```python
        result = get_landsat_composite_by_dates("2010-06-01", "2010-08-31", my_roi)
        ```
    """
    year = int(start_date[:4])
    if year <= 2012:  
        dataset_id = 'LANDSAT/LT05/C02/T1_L2'
    else:  
        dataset_id = 'LANDSAT/LC08/C02/T1_L2'
    
    landsat_collection = ee.ImageCollection(dataset_id) \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .sort('CLOUD_COVER')
    

    filtered_landsat = landsat_collection.filter(ee.Filter.lt('CLOUD_COVER', 10))
    
    count = filtered_landsat.size().getInfo()
    print(count)
    if count < 10:
        print(f"No sufficient data available from {start_date} to {end_date}")
        return None

    masked_landsat = filtered_landsat.map(maskLandsat)
    composite = masked_landsat.median()
    
    return composite.clip(roi)


def get_landsat_composite(year, roi):
    """
    --------------------------------------------------------------------------
    Function: get_landsat_composite
    --------------------------------------------------------------------------
    Parameters:
        - year (int): Year used to select Landsat collection (e.g., 2010).
        - roi (ee.Geometry or ee.Feature): Region of interest to filter and clip images.

    Returns:
        - ee.Image: Median composite of cloud-masked Landsat images clipped to ROI.
            - Computed from images with <10% cloud cover, backtracking years if needed.

    Description:
        This function chooses Landsat 5 or 8 based on the input year,
        filters images by date, bounds, and cloud cover, optionally backtracks
        to earlier years until sufficient images are found, then applies a cloud mask
        and returns the median composite clipped to the ROI.

    Mathematical Details:
        None.

    Example:
        result = get_landsat_composite(2020, my_roi)
    """
    if year <= 2012:
        dataset_id = 'LANDSAT/LT05/C02/T1_L2'
    else:
        dataset_id = 'LANDSAT/LC08/C02/T1_L2'

    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    print(f'Trying to get data from {start_date} to {end_date}')

    data_found = False
    current_year = year
    while not data_found:

        landsat_collection = ee.ImageCollection(dataset_id) \
            .filterBounds(roi) \
            .filterDate(start_date, end_date) \
            .sort('CLOUD_COVER')


        filtered_landsat = landsat_collection.filter(ee.Filter.lt('CLOUD_COVER', 10))
        count = filtered_landsat.size().getInfo()
        print(count)

        if count > 5:
            data_found = True
        else:

            print(f'Insufficient data for year {current_year}, trying from year {current_year - 1}')
            current_year -= 1
            start_date = f'{current_year}-01-01'
            end_date = f'{year}-12-31'
        
 
            if current_year < 1984:
                print("No sufficient data found even after going back to 1984.")
                break
    

            if current_year <= 2012 and dataset_id == 'LANDSAT/LC08/C02/T1_L2':
                print(f"Switching to Landsat 5 as current year is {current_year}")
                # dataset_id = 'LANDSAT/LT05/C02/T1_L2'
                break
    
    # Apply cloud mask and compute median composite
    masked_landsat = filtered_landsat.map(maskLandsat)
    composite = masked_landsat.median()

    return composite.clip(roi)

def add_indices(image):
    """
    --------------------------------------------------------------------------
    Function: add_indices
    --------------------------------------------------------------------------
    Parameters:
        - image (ee.Image): Input surface reflectance image with bands SR_B2, SR_B4, SR_B5.

    Returns:
        - ee.Image: The original image with added bands:
            - NDVI: Normalized Difference Vegetation Index.
            - SAVI: Soil Adjusted Vegetation Index.
            - EVI: Enhanced Vegetation Index.

    Description:
        Computes three vegetation indices (NDVI, SAVI, EVI) from a Landsat 8 surface reflectance image
        and appends them as new bands.

    Mathematical Details:
        - NDVI: (NIR - RED) / (NIR + RED)
        - SAVI: ((NIR - RED) / (NIR + RED + L)) * (1 + L), where L = 0.5.
        - EVI: 2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)

    Example:
        ```python
        result = add_indices(image)
        ```
    """
    ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    savi = image.expression(
        '((NIR - RED) / (NIR + RED + L)) * (1 + L)', {
            'NIR': image.select('SR_B5'),
            'RED': image.select('SR_B4'),
            'L': 0.5
        }).rename('SAVI')
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
            'NIR': image.select('SR_B5'),
            'RED': image.select('SR_B4'),
            'BLUE': image.select('SR_B2')
        }).rename('EVI')
    return image.addBands([ndvi, savi, evi])


def derive_canopy_height(composite_L8, composite_L8_sample, roi):
    """
    --------------------------------------------------------------------------
    Function: derive_canopy_height
    --------------------------------------------------------------------------
    Parameters:
        - composite_L8 (ee.Image): multi-band composite image used for training.
        - composite_L8_sample (ee.Image): multi-band composite image to predict canopy height.
        - roi (ee.Geometry or ee.Feature): region of interest for sampling and clipping.

    Returns:
        - ee.Image: composite_L8_sample with an added 'canopy_height' band.
            - Contains predicted canopy height values, zeroed where no vegetation.

    Description:
        This function samples canopy height and vegetation presence from a high-resolution canopy
        height dataset, trains Random Forest models (regressor for height, classifier for vegetation),
        then applies these models to composite_L8_sample to derive a canopy height.

    Mathematical Details:
        - Random Forest regression: 50 trees for canopy height prediction.
        - Random Forest classification: 50 trees for vegetation mask prediction.

    Example:
        ```python
        result = derive_canopy_height(composite_L8, composite_L8_sample, roi)
        ```
    """
    # Load canopy height image, clip to ROI, and rename band
    canopy_height = ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1").clip(roi).rename('height')
    # Fill no-data with zero
    canopy_height = canopy_height.unmask(0)
    
    # Create binary mask for vegetation presence
    vegetation_mask = canopy_height.gt(0).rename('vegetation')
    
    # Sample canopy height and spectral bands for training
    height_samples = canopy_height.addBands(composite_L8).sample(
        region=roi,
        scale=30,
        numPixels=10000,
        geometries=True
    ).filter(ee.Filter.notNull(['height']))
    
    # Sample vegetation mask and spectral bands for training
    vegetation_samples = vegetation_mask.addBands(composite_L8).sample(
        region=roi,
        scale=30,
        numPixels=10000,
        geometries=True
    ).filter(ee.Filter.notNull(['vegetation']))
    
    # Define bands used for training
    height_bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']
    height_regressor = ee.Classifier.smileRandomForest(50).setOutputMode('REGRESSION').train(
        features=height_samples,
        classProperty='height',
        inputProperties=height_bands
    )
    
    # Train Random Forest classifier for vegetation presence
    vegetation_classifier = ee.Classifier.smileRandomForest(50).train(
        features=vegetation_samples,
        classProperty='vegetation',
        inputProperties=height_bands
    )
    

    derived_canopy_height = composite_L8_sample.select(height_bands).classify(height_regressor)

    derived_vegetation = composite_L8_sample.select(height_bands).classify(vegetation_classifier)
    

    adjusted_canopy_height = derived_canopy_height.where(derived_vegetation.eq(0), 0)
    

    return composite_L8_sample.addBands(adjusted_canopy_height.rename('canopy_height'))


def get_modis_landcover(year, roi):
    """
    --------------------------------------------------------------------------
    Function: get_modis_landcover
    --------------------------------------------------------------------------
    Parameters:
        - year (int): Year to retrieve MODIS land cover for.
        - roi (ee.Geometry or ee.Feature): Region of interest for clipping.

    Returns:
        - ee.Image: Land cover image of LC_Type1 clipped to ROI.

    Description:
        Retrieves the yearly MODIS MCD12Q1 land cover image and clips it to the ROI.

    Mathematical Details:
        None.

    Example:
        result = get_modis_landcover(2020, my_roi)
    """
    dataset_id = "MODIS/061/MCD12Q1"
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    landcover = ee.ImageCollection(dataset_id) \
        .filterDate(start_date, end_date) \
        .first()  # select first image

    lc_type1 = landcover.select('LC_Type1')
    clipped_lc = lc_type1.clip(roi)
    
    return clipped_lc

def get_urban_area_from_gisa(year, roi):
    """
    --------------------------------------------------------------------------
    Function: get_urban_area_from_gisa
    --------------------------------------------------------------------------
    Parameters:
        - year (int): The year to retrieve urban area for.
        - roi (ee.Geometry): The region of interest to clip the urban area image.
        
    Returns:
        - ee.Image: An image representing urban areas within the specified region.
        - pixel value 1 indicates urban, 0 indicates non-urban.
        
    Description:
        Retrieve urban area mask from the GISA dataset based on the given year
        and clip it to the specified region. Masked areas are set to zero.
        data source: Xin Huang, Jiayi Li, Jie Yang, Zhen Zhang, Dongrui Li, & Xiaoping Liu. (2021).
                    30 m global impervious surface area dynamics and urban expansion pattern observed
                    by Landsat satellites: from 1972 to 2019 (Version 1.0.0)
                    [Data set]. http://doi.org/10.1007/s11430-020-9797-9
        
    Mathematical Details:
        - updateMask and unmask: ensures masked pixels are filled with zero.
        
    Example:
        ```python
        - result = get_urban_area_from_gisa(2010, some_roi)
        ```
    """
    # Load the GISA ImageCollection
    gisa = ee.ImageCollection("projects/sat-io/open-datasets/GISA_1972_2019").mosaic()
    
    # Define pixel value thresholds by year
    year_thresholds = {
        1972: 1,
        1978: 2,
        1985: 3,
        1986: 4,
        1987: 5,
        1988: 6,
        1989: 7,
        1990: 8,
        1991: 9,
        1992: 10,
        1993: 11,
        1994: 12,
        1995: 13,
        1996: 14,
        1997: 15,
        1998: 16,
        1999: 17,
        2000: 18,
        2001: 19,
        2002: 20,
        2003: 21,
        2004: 22,
        2005: 23,
        2006: 24,
        2007: 25,
        2008: 26,
        2009: 27,
        2010: 28,
        2011: 29,
        2012: 30,
        2013: 31,
        2014: 32,
        2015: 33,
        2016: 34,
        2017: 35,
        2018: 36,
        2019: 37
    }
    
    threshold = year_thresholds.get(year, 37)  # Default value is 37
    urban_area = gisa.lte(threshold).rename('urban')
    
   # Set masked areas to 0
    urban_area = urban_area.updateMask(urban_area).unmask(0)
    
    return urban_area.clip(roi)

def adjust_final_classification_No_urban(pre_GLC, Pre_height, Pre_ESA, urban_class):
    """
    --------------------------------------------------------------------------
    Function: adjust_final_classification_No_urban
    --------------------------------------------------------------------------
    Parameters:
        - pre_GLC (ee.Image): GLC pre-classification image.
        - Pre_height (ee.Image): Canopy height image in meters.
        - Pre_ESA (ee.Image): ESA land cover classification image.
        - urban_class (ee.Image): Urban mask image.
        
    Returns:
        - ee.Image: Adjusted classification image.
        - Pixel values updated based on height thresholds and ESA classes.
        
    Description:
        Refine the initial GLC classification by applying height-based thresholds
        and ESA-based adjustments, without incorporating urban-specific changes.
        
    Mathematical Details:
        - .gt, .lte, .And: generate boolean masks for height comparisons.
        - .where(condition, value): replace pixel values where condition is true.
        
    Example:
        ```python
        - result = adjust_final_classification_No_urban(pre_GLC, Pre_height, Pre_ESA, urban_class)
        - result = ee.Image(...)
        ```
    """
    ## Create boolean masks for canopy height ranges
    # low_height = Pre_height.gt(0).And(Pre_height.lte(2))
    # low_height_1 = Pre_height.gt(0).And(Pre_height.lte(5))
    # mid_height = Pre_height.gt(2).And(Pre_height.lte(10))
    # mid_height_1 = Pre_height.gt(5).And(Pre_height.lte(10))
    # mid_height_2 = Pre_height.gt(5).And(Pre_height.lte(10))
    # high_height = Pre_height.gt(10)
    # high_height_1 = Pre_height.gt(12)
    very_high_height = Pre_height.gt(15)

    # 1. Keep classes 1–5 unchanged

    # 2. For classes 8–10 with canopy >15m, set to class 5
    pre_GLC = pre_GLC.where(pre_GLC.gte(8).And(pre_GLC.lte(10)).And(very_high_height), 5)

    # 3. For class 11 with very high canopy height, set to class 5
    pre_GLC = pre_GLC.where(pre_GLC.eq(11).And(very_high_height), 5)

    # 4. For class 12, if canopy >5m set to 5
    pre_GLC = pre_GLC.where(pre_GLC.eq(12).And(very_high_height), 5)

    pre_GLC = pre_GLC.where(pre_GLC.eq(13).And(very_high_height), 5)
    pre_GLC = pre_GLC.where(pre_GLC.eq(14).And(very_high_height), 5)

    pre_GLC = pre_GLC.where(pre_GLC.eq(15), 15)
    pre_GLC = pre_GLC.where(pre_GLC.eq(16), 16)
    pre_GLC = pre_GLC.where(pre_GLC.eq(17), 17)

    # ESA-based adjustments
    tree_cover_mask = Pre_ESA.eq(10)
   
    
    shrubland_mask = Pre_ESA.eq(20)

    grassland_mask = Pre_ESA.eq(30)

    pre_GLC = pre_GLC.where(grassland_mask.And(pre_GLC.neq(10)), 10)


    cropland_mask = Pre_ESA.eq(40)
    pre_GLC = pre_GLC.where(cropland_mask.And(pre_GLC.neq(12)), 12)

    
    built_up_mask = Pre_ESA.eq(50)
    mask_ve = pre_GLC.eq(13).And(pre_GLC.eq(16))
    pre_GLC = pre_GLC.where(built_up_mask.And(mask_ve), 13)
    pre_GLC = pre_GLC.where(built_up_mask.Or(pre_GLC.eq(13)), 13)

    
    bare_mask = Pre_ESA.eq(60)

    pre_GLC = pre_GLC.where(bare_mask.And(pre_GLC.eq(13).Or(pre_GLC.eq(16))), pre_GLC)

    
    snow_mask = Pre_ESA.eq(70)
    pre_GLC = pre_GLC.where(snow_mask, 15)
    
    water_mask = Pre_ESA.eq(80)
    pre_GLC = pre_GLC.where(water_mask, 17)
    
    wetland_mask = Pre_ESA.eq(90)
    pre_GLC = pre_GLC.where(wetland_mask, 11)
    
    mangroves_mask = Pre_ESA.eq(95)
    pre_GLC = pre_GLC.where(mangroves_mask, 5)
    
    moss_lichen_mask = Pre_ESA.eq(100)
    pre_GLC = pre_GLC.where(moss_lichen_mask, 10)

    
    return pre_GLC

def adjust_final_classification(pre_GLC, Pre_height, Pre_ESA, urban_class):
    """
    --------------------------------------------------------------------------
    Function: adjust_final_classification
    --------------------------------------------------------------------------
    Parameters:
        - pre_GLC (ee.Image): GLC pre-classification image.
        - Pre_height (ee.Image): Canopy height image in meters.
        - Pre_ESA (ee.Image): ESA land cover classification image.
        - urban_class (ee.Image): Urban mask image.

    Returns:
        - ee.Image: Adjusted classification image with urban adjustments.
        - Pixel values updated based on height thresholds, ESA classes, and urban mask.

    Description:
        Apply height-based and ESA-based rules to refine GLC pre-classification,
        then incorporate urban mask adjustments.

    Example:
        ```python
        - result = adjust_final_classification(pre_GLC, Pre_height, Pre_ESA, urban_class)
        ```
    """
    # Create boolean masks for canopy height thresholds
    low_height = Pre_height.gt(0).And(Pre_height.lte(5))
    mid_height = Pre_height.gt(5).And(Pre_height.lte(10))
    mid_height_1 = Pre_height.gt(5).And(Pre_height.lte(10))
    high_height = Pre_height.gt(10)
    very_high_height = Pre_height.gt(15)

    # 1. Keep classes 1–5 unchanged

    # For classes 8–10 with canopy height >15m, set to class 5
    pre_GLC = pre_GLC.where(pre_GLC.gte(8).And(pre_GLC.lte(10)).And(very_high_height), 5)

    pre_GLC = pre_GLC.where(pre_GLC.eq(11).And(low_height), 10)
    pre_GLC = pre_GLC.where(pre_GLC.eq(11).And(mid_height), 10)
    pre_GLC = pre_GLC.where(pre_GLC.eq(11).And(high_height), 5)
    pre_GLC = pre_GLC.where(pre_GLC.eq(11).And(very_high_height), 5)

    pre_GLC = pre_GLC.where(pre_GLC.eq(12).And(very_high_height), 5)
    pre_GLC = pre_GLC.where(pre_GLC.eq(12).And(high_height), 6)
    pre_GLC = pre_GLC.where(pre_GLC.eq(12).And(mid_height_1), 6)

    pre_GLC = pre_GLC.where(pre_GLC.eq(13).And(very_high_height), 5)

    pre_GLC = pre_GLC.where(pre_GLC.eq(14).And(very_high_height), 5)
    pre_GLC = pre_GLC.where(pre_GLC.eq(14).And(high_height), 6)
    pre_GLC = pre_GLC.where(pre_GLC.eq(14).And(mid_height_1), 6)

    pre_GLC = pre_GLC.where(pre_GLC.eq(15), 15)
    pre_GLC = pre_GLC.where(pre_GLC.eq(16), 16)
    pre_GLC = pre_GLC.where(pre_GLC.eq(17), 17)

    # Create masks for ESA classes
    tree_cover_mask = Pre_ESA.eq(10)
    pre_GLC = pre_GLC.where(tree_cover_mask.And(pre_GLC.neq(1)).And(pre_GLC.neq(2)).And(pre_GLC.neq(3)).And(pre_GLC.neq(4)).And(pre_GLC.neq(5)), 5)
    
    shrubland_mask = Pre_ESA.eq(20)

    pre_GLC = pre_GLC.where(shrubland_mask.And(pre_GLC.neq(6)).And(pre_GLC.neq(7)).And(pre_GLC.neq(8)).And(pre_GLC.neq(9)), 
                            pre_GLC.where(high_height, 8).where(mid_height, 6))
    grassland_mask = Pre_ESA.eq(30)
   
    pre_GLC = pre_GLC.where(grassland_mask.And(pre_GLC.neq(10)), 10)
    

    cropland_mask = Pre_ESA.eq(40)
    pre_GLC = pre_GLC.where(cropland_mask.And(pre_GLC.neq(12)), 12)
    pre_GLC = pre_GLC.where(cropland_mask.And(high_height), 5)
    pre_GLC = pre_GLC.where(cropland_mask.And(very_high_height), 5)
    
    built_up_mask = Pre_ESA.eq(50)
    mask_ve = pre_GLC.eq(13).And(pre_GLC.eq(16))
    pre_GLC = pre_GLC.where(built_up_mask.And(mask_ve), 13)
    pre_GLC = pre_GLC.where(built_up_mask.Or(pre_GLC.eq(13)), 13)
    pre_GLC = pre_GLC.where(built_up_mask.And(high_height), 6)
    pre_GLC = pre_GLC.where(built_up_mask.And(very_high_height), 5)
    
    bare_mask = Pre_ESA.eq(60)
    pre_GLC = pre_GLC.where(bare_mask.And(pre_GLC.eq(13).Or(pre_GLC.eq(16))), pre_GLC)
    pre_GLC = pre_GLC.where(bare_mask.And(very_high_height), 5)
    
    snow_mask = Pre_ESA.eq(70)
    pre_GLC = pre_GLC.where(snow_mask, 15)
    
    water_mask = Pre_ESA.eq(80)
    pre_GLC = pre_GLC.where(water_mask, 17)
    
    wetland_mask = Pre_ESA.eq(90)
    pre_GLC = pre_GLC.where(wetland_mask, 11)
    
    mangroves_mask = Pre_ESA.eq(95)
    pre_GLC = pre_GLC.where(mangroves_mask, 5)
    
    moss_lichen_mask = Pre_ESA.eq(100)
    pre_GLC = pre_GLC.where(moss_lichen_mask, 10)
 
    pre_GLC = pre_GLC.where(urban_class.And(pre_GLC.eq(10).And(high_height)), 5)
    
    pre_GLC = pre_GLC.where(urban_class.And(pre_GLC.eq(11).And(high_height)), 5)

    pre_GLC = pre_GLC.where(urban_class.And(pre_GLC.eq(12).And(high_height)), 5)
    
    pre_GLC = pre_GLC.where(urban_class.And(pre_GLC.eq(14).And(high_height)), 5)
    
    
    return pre_GLC


def get_esa_worldcover(year, roi):
    """
    --------------------------------------------------------------------------
    Function: get_esa_worldcover
    --------------------------------------------------------------------------
    Parameters:
        - roi (ee.Geometry): Region of interest to clip the WorldCover image.
        
    Returns:
        - ee.Image: Clipped WorldCover classification image.
            - Renamed band 'LC_Type1' for consistency.
        
    Description:
        Load the ESA WorldCover v100 classification image, clip it to the given
        region of interest, and rename its band for consistency.
       
    Example:
        ```python
        - result = get_esa_worldcover(2020, some_roi)
        ```
    """
    dataset = ee.ImageCollection("ESA/WorldCover/v100").first()
    return dataset.clip(roi).rename('LC_Type1')

def get_samples(image, class_band, num_points, scale, region):
    """
    --------------------------------------------------------------------------
    Function: get_samples
    --------------------------------------------------------------------------
    Parameters:
        - image (ee.Image): The input image to sample from.
        - class_band (list): List of the band containing class labels.
        - num_points (int): Number of sample points per class.
        - scale (float): Spatial resolution in meters for sampling.
        - region (ee.Geometry): Geographic region to constrain sampling.
        
    Returns:
        - ee.FeatureCollection: Stratified sample points with class labels.
        
    Description:
        Generate a stratified random sample of points from the input image
        based on the specified class band, number of points, scale, and region.
        Each sample includes its geographic coordinates and class label.
        
    Mathematical Details:
        - None
        
    Example:
        ```python
        - result = get_samples(img, ['SR_B1', 'SR_B2'...], 100, 30, roi)
        ```
    """
    return image.stratifiedSample(
        numPoints=num_points,
        classBand=class_band,
        scale=scale,
        region=region,
        geometries=True
    )

def get_samples_roi(image, class_band, num_points, scale, region):
    """
    --------------------------------------------------------------------------
    Function: get_samples_roi
    --------------------------------------------------------------------------
    Parameters:
        - image (ee.Image): The input image to sample from.
        - class_band (list): List of the band containing class labels.
        - num_points (int): Number of sample points to extract.
        - scale (float): Spatial resolution in meters for sampling.
        - region (ee.Geometry): Geographic region to constrain sampling.

    Returns:
        - ee.FeatureCollection: Random sample points with geometries.

    Description:
        Perform random sampling on the input image within the specified region
        at the given scale, returning the requested number of points with their
        geometries.

    Example:
        ```python
        - result = get_samples_roi(img, ['SR_B1', 'SR_B2'...], 100, 30, roi)
        ```
    """
    return image.sample(
        region=region,
        scale=scale,
        numPixels= num_points,
        seed=1,
        geometries=True
    )

def get_valid_classes(task_name):
    """
    --------------------------------------------------------------------------
    Function: get_valid_classes
    --------------------------------------------------------------------------
    Parameters:
        - task_name (str): Name of the classification task.
        
    Returns:
        - ee.List or None: List of valid class values or None if unrestricted.
            - ESA: [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
            - GLC and MODIS: sequence from 1 to 17.
        
    Description:
        Generate a list of valid class values based on the provided task name.
        Supports ESA, GLC, and MODIS classification schemes.
        
    Example:
        ```python
        - classes = get_valid_classes("ESA")
    ```
    """
    # task_name = task_name.lower()
    if task_name == "ESA":
        return ee.List([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
    elif task_name == "GLC":
        return ee.List.sequence(1, 17)
    elif task_name == "MODIS":
        return ee.List.sequence(1, 17)
    else:
        return None  # no limited

def sample_and_classify(composite, composite_pre, samples, bands, label, eval_config=None, task_name="classification"):
    """
    --------------------------------------------------------------------------
    Function: sample_and_classify
    --------------------------------------------------------------------------
    Parameters:
        - composite (ee.Image): Feature image for sampling and training.
        - composite_pre (ee.Image): Image to classify using the trained model.
        - samples (ee.FeatureCollection): Sample points with labels.
        - bands (list of str): List of band names to use as features.
        - label (str): Name of the property containing the sample label.
        - eval_config (dict, optional): Configuration for evaluation, including:
            - evaluate (bool): Whether to perform cross-validation.
            - train_ratio (float): Ratio of training samples.
            - cv_folds (int): Number of cross-validation folds.
            - date_Day (str): Date string for output naming.
            - output_dir (str): Directory to save evaluation results.
        - task_name (str): Task name for valid class restriction and output naming.

    Returns:
        - ee.Image: Classified image (with invalid classes cleaned).
            - Pixel values correspond to predicted classes.

    Description:
        Train a Random Forest classifier on provided samples and classify the
        composite_pre image. Optionally perform cross-validation evaluation and
        save metrics to a JSON file.

    Mathematical Details:
        - None

    Example:
        ```python
        - result = sample_and_classify(comp, comp_pre, samples, ['B1','B2'], 'landcover',
        {'evaluate':True, 'train_ratio':0.7, 'cv_folds':5, 'output_dir':'./out'}, 'GLC')
        - result = ee.Image(...)
        ```
    """
    valid_classes = get_valid_classes(task_name)
    print(f"[Task] {task_name}")
    
    if eval_config is None:
        eval_config = {}
        evaluate = 'False'
        print(f"evaluate {evaluate}")
    else:
        evaluate = eval_config["evaluate"]

    if evaluate == 'True':
        # Extract evaluation parameters
        train_ratio = float(eval_config["train_ratio"])
        n_folds = int(eval_config["cv_folds"])
        output_dir = str(eval_config["output_dir"])
        date_Day = str(eval_config.get("date_Day", "unknown_date"))

        os.makedirs(output_dir, exist_ok=True)

        # Add random column for cross-validation splits
        samples = samples.randomColumn("random", seed=1234)
        fold_size = 1.0 / n_folds
        all_metrics = []

        for fold in range(n_folds):
            start = fold * fold_size
            end = (fold + 1) * fold_size

            testing = samples.filter(ee.Filter.gte("random", start)).filter(ee.Filter.lt("random", end))
            training = samples.filter(ee.Filter.Or(
                ee.Filter.lt("random", start),
                ee.Filter.gte("random", end)
            ))

            # Train classifier on training samples
            trained = composite.select(bands).sampleRegions(
                collection=training,
                properties=[label],
                scale=30
            )
            classifier = ee.Classifier.smileRandomForest(50).train(
                trained, label, bands
            )

            # Classify testing set
            validated = composite.select(bands).sampleRegions(
                collection=testing,
                properties=[label],
                scale=30
            ).classify(classifier)

            if valid_classes:
                def clean_classification(feat):
                    cls = feat.get("classification")
                    is_valid = valid_classes.contains(cls)
                    cls_clean = ee.Algorithms.If(is_valid, cls, 0)
                    return feat.set("classification", cls_clean)
                validated = validated.map(clean_classification)

             # Compute evaluation metrics
            error_matrix = validated.errorMatrix(label, 'classification')
            accuracy = error_matrix.accuracy().getInfo()
            matrix = error_matrix.getInfo()

            fold_result = {
                "fold": fold + 1,
                "accuracy": accuracy,
                "confusion_matrix": matrix
            }
            all_metrics.append(fold_result)

        # Save cross-validation results to JSON
        output_path = os.path.join(output_dir, f"{task_name}_{date_Day}.json")
        with open(output_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"[Eval] Cross-validation results saved to {output_path}")

    # Train on full sample set and classify composite_pre
    full_training = composite.select(bands).sampleRegions(
        collection=samples,
        properties=[label],
        scale=30
    )
    final_classifier = ee.Classifier.smileRandomForest(50).train(
        full_training, label, bands
    )
    result = composite_pre.select(bands).classify(final_classifier)

    # if valid_classes:
    #     result = result.where(valid_classes.contains(result).Not(), 0)
    return result

def clean_image_with_valid_classes(image, task_name):
    """
    --------------------------------------------------------------------------
    Function: clean_image_with_valid_classes
    --------------------------------------------------------------------------
    Parameters:
        - image (ee.Image): Input classification image.
        - task_name (str): Name of the classification task to determine valid classes.
        
    Returns:
        - ee.Image: Cleaned classification image with invalid values set to 0.
        
    Description:
        Retain only pixels with valid class values based on task_name. All other
        pixels are set to zero and the output band is renamed.
        
    Example:
        ```python
        - result = clean_image_with_valid_classes(img, "ESA")
        ```
    """
    valid_classes = get_valid_classes(task_name)
    return image.remap(valid_classes, valid_classes, 0).toInt().rename('LC_Type1')

def classify_glc_with_height(GLC_class_2020, composite_L8_2020,composite_L8, roi, eval_config=None, task_name="GLC"):
    """
    --------------------------------------------------------------------------
    Function: classify_glc_with_height
    --------------------------------------------------------------------------
    Parameters:
        - GLC_class_2020 (ee.Image): Initial  GLC classification image.
        - composite_L8_2020 (ee.Image): Landsat 8 composite for (training/test).
        - composite_L8 (ee.Image): Landsat 8 composite for target classification.
        - roi (ee.Geometry): Region of interest for sampling.
        - eval_config (dict, optional): Evaluation configuration (evaluate, train_ratio, cv_folds, date_Day, output_dir).
        - task_name (str): Task name for valid class restriction and naming.

    Returns:
        - ee.Image: Classified image of composite_L8.
            - Predictions based on trained Random Forest.

    Description:
        Clean the classification image, generate stratified and random samples,
        optionally perform cross-validation on data, and then classify the
        current composite using canopy height and spectral bands.

    Mathematical Details:
        - None

    Example:
        ```python
        - result = classify_glc_with_height(GLC2020, comp2020, compCurrent, roi,
        {'evaluate':True, 'train_ratio':0.7, 'cv_folds':5, 'output_dir':'./out'}, 'GLC')
        - result = ee.Image(...)
        ```
    """

    GLC_class_2020 = clean_image_with_valid_classes(GLC_class_2020, task_name)
    stratified_samples = get_samples(GLC_class_2020, 'LC_Type1', 3000, 30, roi)
    # stratified_samples = get_samples(GLC_class_2020, 'LC_Type1', 1000, 30, roi)
    roi_samples = get_samples_roi(GLC_class_2020, 'LC_Type1', 5000, 30, roi)
    combined_samples = roi_samples.merge(stratified_samples)
    bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'NDVI', 'SAVI', 'EVI','canopy_height']
    # Validation workflow
    if eval_config and eval_config.get("evaluate", False):
        _ = sample_and_classify(
            composite_L8_2020,    # train on 2020
            composite_L8_2020,    # test on 2020
            combined_samples,
            bands,
            'LC_Type1',
            eval_config=eval_config,
            task_name=task_name
        )
    return sample_and_classify(composite_L8_2020,composite_L8, combined_samples, bands, 'LC_Type1', eval_config=None,task_name=task_name)

def classify_ESA_with_height(ESA_class_2020,composite_L8_2020,composite_L8, roi, eval_config=None, task_name="ESA"):
    """
    --------------------------------------------------------------------------
    Function: classify_ESA_with_height
    --------------------------------------------------------------------------
    Parameters:
        - ESA_class_2020 (ee.Image): Initial ESA classification image.
        - composite_L8_2020 (ee.Image): Landsat 8 composite for  (training/test).
        - composite_L8 (ee.Image): Landsat 8 composite for current classification.
        - roi (ee.Geometry): Region of interest for sampling.
        - eval_config (dict, optional): Configuration for evaluation, including:
            evaluate/train_ratio/cv_folds/date_Day/output_dir.
        - task_name (str): Task name for valid class restriction and naming.

    Returns:
        - ee.Image: Classified image of composite_L8.
            - Pixel values correspond to predicted ESA classes.

    Description:
        Clean the classification, generate stratified and random samples,
        optionally perform cross-validation on data, and then classify the
        current composite using spectral bands.

    Mathematical Details:
        - None

    Example:
        ```python
        - result = classify_ESA_with_height(ESA2020, comp2020, compCurrent, roi,
        {'evaluate':True, 'train_ratio':0.7, 'cv_folds':5, 'output_dir':'./out'}, 'ESA')
        ```
    """
    ESA_class_2020 = clean_image_with_valid_classes(ESA_class_2020, task_name)
    stratified_samples = get_samples(ESA_class_2020, 'LC_Type1', 3000, 10, roi)
    # stratified_samples = get_samples(ESA_class_2020, 'LC_Type1', 1000, 10, roi)
    roi_samples = get_samples_roi(ESA_class_2020, 'LC_Type1', 5000, 10, roi)
    combined_samples = roi_samples.merge(stratified_samples)
    bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'NDVI', 'SAVI', 'EVI']
    # Validation workflow
    if eval_config and eval_config.get("evaluate", False):
        print('Validation')
        _ = sample_and_classify(
            composite_L8_2020,    # train on 2020
            composite_L8_2020,    # test on 2020
            combined_samples,
            bands,
            'LC_Type1',
            eval_config=eval_config,
            task_name=task_name
        )
    # print('Validation')
    return sample_and_classify(composite_L8_2020,composite_L8, combined_samples, bands, 'LC_Type1', eval_config=None,task_name=task_name)

def classify_MODIS_with_height( MODIS_class_2020,composite_L8_2020,composite_L8, roi, eval_config=None, task_name="MODIS"):
    """
    --------------------------------------------------------------------------
    Function: classify_MODIS_with_height
    --------------------------------------------------------------------------
    Parameters:
        - MODIS_class_2020 (ee.Image): Initial MODIS classification image.
        - composite_L8_2020 (ee.Image): Landsat 8 composite for (training/test).
        - composite_L8 (ee.Image): Landsat 8 composite for current classification.
        - roi (ee.Geometry): Region of interest for sampling.
        - eval_config (dict, optional): Evaluation settings (evaluate, train_ratio, cv_folds, date_Day, output_dir).
        - task_name (str): Task name for valid class restriction and naming.

    Returns:
        - ee.Image: Classified image of composite_L8.
            - Pixel values correspond to predicted MODIS classes.

    Description:
        Clean the MODIS classification, generate stratified and random samples,
        optionally perform cross-validation on data, and then classify the
        current composite using spectral bands.

    Mathematical Details:
        - None

    Example:
        ```python
        - result = classify_MODIS_with_height(MODIS2020, comp2020, compCurrent, roi,
        {'evaluate':True, 'train_ratio':0.7, 'cv_folds':5, 'output_dir':'./out'}, 'MODIS')
        ```
    """
    MODIS_class_2020 = clean_image_with_valid_classes(MODIS_class_2020, task_name)
    stratified_samples = get_samples( MODIS_class_2020, 'LC_Type1', 3000, 500, roi)
    # stratified_samples = get_samples( MODIS_class_2020, 'LC_Type1', 1000, 500, roi)
    roi_samples = get_samples_roi( MODIS_class_2020, 'LC_Type1', 5000, 500, roi)
    combined_samples = roi_samples.merge(stratified_samples)
    bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'NDVI', 'SAVI', 'EVI']
    # Validation workflow
    if eval_config and eval_config.get("evaluate", False):
        _ = sample_and_classify(
            composite_L8_2020,    # train on 2020
            composite_L8_2020,    # test on 2020
            combined_samples,
            bands,
            'LC_Type1',
            eval_config=eval_config,
            task_name=task_name
        )
    return sample_and_classify(composite_L8,composite_L8, combined_samples, bands, 'LC_Type1', eval_config=None,task_name=task_name)




def classify_landsat_using_glc_ESA(date, roi,eval_config=None):
    """
    --------------------------------------------------------------------------
    Function: classify_landsat_using_glc_ESA
    --------------------------------------------------------------------------
    Parameters:
        - date (str): Date string in "YYYY-MM-DD" format for composite generation.
        - roi (ee.Geometry): Region of interest for imagery processing.
        - eval_config (dict, optional): Evaluation configuration dictionary.

    Returns:
        - ee.Image: Final land cover classification image clipped to ROI.
            - Combines GLC, ESA, and MODIS adjustments and fills NoData with MODIS values.

    Description:
        Generate Landsat composites image around the given date, sample and classify land cover
        using GLC, ESA, and MODIS reference data, adjust classifications based on canopy height,
        and produce a final clean classification image. Handles Earth Engine exceptions by returning
        the IGBP classification.

    Mathematical Details:
        - None

    Example:
        ```python
        - result = classify_landsat_using_glc_ESA("2021-06-15", some_roi, {'evaluate': True})
        - result = ee.Image(...)
        ```
    """
    try:
        # Check if ROI dimensions are smaller than 1° x 1°
        roi_org = roi
        bounds = roi.bounds()
        coords = bounds.coordinates().get(0).getInfo()
        
        # Get ROI boundaries
        lon_min, lat_min = coords[0]
        lon_max, lat_max = coords[2]
        
        # Calculate width and height in degrees
        width = abs(lon_max - lon_min)
        height = abs(lat_max - lat_min)
        roi_area = width *height

        if roi_area < 0.5:
            lon_center = (lon_min + lon_max) / 2
            lat_center = (lat_min + lat_max) / 2
            
   
            lon_min = lon_center - 0.5
            lon_max = lon_center + 0.5
            lat_min = lat_center - 0.5
            lat_max = lat_center + 0.5
            

            roi = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
            print(f'ROI was expanded to 1° x 1° around [{lon_center}, {lat_center}]')
        # year = int(date[:4])
        date = datetime.datetime.strptime(date, "%Y-%m-%d")
        year = date.year
        start_date = (date - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = (date + datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        
        # Retrieve Landsat composite by date range

        composite_L8_org = get_landsat_composite_by_dates(start_date, end_date, roi_org)
        if composite_L8_org is None:
            composite_L8_org = get_landsat_composite(year, roi_org)
        composite_L8 = get_landsat_composite_by_dates(start_date, end_date, roi)
        if composite_L8 is None:
            composite_L8 = get_landsat_composite(year, roi)
        # Retrieve 2020 composite for training
        composite_L8_2020 = get_landsat_composite(2020, roi)
        GLC_class_2020 = get_glc_fcs30d(2020, roi)
        ESA_class_2020 = get_esa_worldcover(2020, roi)
        canopy_height_2020 = ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1").clip(roi)
        canopy_height = canopy_height_2020.rename('canopy_height')
        composite_L8_2020 = add_indices(composite_L8_2020).addBands(canopy_height)
        # GLC_class_adj_with_height = adjust_using_canopy_height(GLC_class_2020, roi, 2020, composite_L8_2020)
        
        composite_L8_with_height = derive_canopy_height(composite_L8_2020, composite_L8, roi)
        composite_L8_with_height = add_indices(composite_L8_with_height)
   
        pre_GLC = classify_glc_with_height(GLC_class_2020, composite_L8_2020,composite_L8_with_height, roi,eval_config=eval_config)
        Pre_ESA = classify_ESA_with_height(ESA_class_2020,composite_L8_2020,composite_L8_with_height, roi,eval_config=eval_config)
        pre_canopy = composite_L8_with_height.select('canopy_height').clip(roi)
        urban_class_gisa = get_urban_area_from_gisa(year, roi)
        # Check that pre_GLC contains valid data
        pre_GLC_info = pre_GLC.reduceRegion(
            reducer=ee.Reducer.count(), 
            geometry=roi, 
            scale=500, 
            maxPixels=1e12
        ).getInfo()
        
        MODIS_class = get_modis_landcover(year, roi)
        # urban_class = urban_class_gisa.neq(1).And(MODIS_class.eq(13))
        Pre_MODIS_class = classify_MODIS_with_height(MODIS_class,composite_L8_2020,composite_L8_with_height, roi,eval_config=eval_config)
        urban_class = MODIS_class.eq(13)
        # final_classification = adjust_final_classification(pre_GLC, pre_canopy ,Pre_ESA, urban_class)
        # final_classification = adjust_final_classification(Pre_MODIS_class, pre_canopy ,Pre_ESA, urban_class)
        # intersection_check = urban_class.reduceRegion(
        #     reducer=ee.Reducer.anyNonZero(),
        #     geometry=roi_org,
        #     scale=5000,
        #     maxPixels=1e12 
        # ).getInfo()
        
        # if intersection_check.get('LC_Type1', False): 
        #     print('urban')
        #     final_classification = adjust_final_classification(Pre_MODIS_class, pre_canopy, Pre_ESA, urban_class)
        # else:
        #     print("No urban area within ROI, returning non-urban classification.")
        #     final_classification =adjust_final_classification_No_urban(Pre_MODIS_class, pre_canopy, Pre_ESA, urban_class)
        # return final_classification.rename('LC_Type1')
        final_classification = adjust_final_classification(Pre_MODIS_class, pre_canopy, Pre_ESA, urban_class)
        
        # Apply final classification adjustments and mask NoData
        final_classification = final_classification.unmask(MODIS_class)
        return final_classification.rename('LC_Type1')
    
    except ee.EEException as e:
        # Handle Earth Engine exception by returning MODIS classification
        print(f"EEException encountered: {e}")
        # date = datetime.datetime.strptime(date, "%Y-%m-%d")
        year = date.year
        MODIS_class = get_modis_landcover(year, roi_org)
        return MODIS_class.rename('LC_Type1')
    
def mask_sentinel2(image, use_alternate=False):
    """
    --------------------------------------------------------------------------
    Function: mask_sentinel2
    --------------------------------------------------------------------------
    Parameters:
        - image (ee.Image): Input Sentinel-2 image.
        - use_alternate (bool): Whether to use alternate classification bands.
        
    Returns:
        - ee.Image: Image with cloud mask applied.
        
    Description:
        Apply a cloud and cirrus mask to a Sentinel-2 image using either the
        QA60 band or alternate MSK_CLASSI bands. Pixels flagged as cloudy or
        cirrus are masked out.
        
    Mathematical Details:
        - bitwiseAnd: tests specific QA bits for clouds (bit 10) and cirrus (bit 11).
        
    Example:
        ```python
        - result = mask_sentinel2(img, use_alternate=True)
        ```
    """
    if use_alternate:
        opaque_mask = image.select('MSK_CLASSI_OPAQUE')
        cirrus_mask = image.select('MSK_CLASSI_CIRRUS')
        mask = opaque_mask.eq(0).And(cirrus_mask.eq(0))
    else:
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask)


def has_qa60_band(image):
    """
    --------------------------------------------------------------------------
    Function: has_qa60_band
    --------------------------------------------------------------------------
    Parameters:
        - image (ee.Image): Input Sentinel-2 image.
        
    Returns:
        - bool: True if 'QA60' band is present, False otherwise.
        
    Description:
        Check whether the given image contains the QA60 band, which is used
        for cloud masking on Sentinel-2 surface reflectance data.
        
    Mathematical Details:
        - None
        
    Example:
        ```python
        - has_band = has_qa60_band(img)
        ```
    """
    return image.bandNames().contains('QA60')
    

def get_sentinel2_year_composite(start_day,end_day, roi):
    """
    --------------------------------------------------------------------------
    Function: get_sentinel2_year_composite
    --------------------------------------------------------------------------
    Parameters:
        - start_day (str): Start date in "YYYY-MM-DD" format.
        - end_day (str): End date in "YYYY-MM-DD" format.
        - roi (ee.Geometry): Region of interest for composite.
        
    Returns:
        - ee.Image or None: Median composite image with clouds removed, or None if no data.
        
    Description:
        Retrieve Sentinel-2 imagery over the specified date range and ROI,
        filter for low cloud cover, apply cloud masking, and produce a median
        composite image.
        
    Example:
        ```python
        - comp = get_sentinel2_year_composite("2021-01-01", "2021-12-31", roi)
        ```
    """
    start_date = f'{start_day}'
    end_date = f'{end_day}'
    # Load Sentinel-2 collection and sort by cloudiness
    sentinel2 = ee.ImageCollection("COPERNICUS/S2") \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .sort('CLOUDY_PIXEL_PERCENTAGE')   
    # Filter images with cloud cover below threshold
    filtered_sentinel2 = sentinel2.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',1))
    # Mark images that have QA60 band
    filtered_sentinel2 = filtered_sentinel2.map(lambda image: ee.Image(image.set('has_QA60', has_qa60_band(image))))
    filtered_sentinel2 = filtered_sentinel2.filter(ee.Filter.eq('has_QA60', True))
    # Check if there is any image left
    count = filtered_sentinel2.size().getInfo()
    if count == 0:
        print("No sufficient data available for year:", start_date,end_date)
        return None
    
    # Apply cloud mask to each image
    masked_sentinel2 = filtered_sentinel2.map(mask_sentinel2)
    
    # Compute median composite and clip to ROI
    composite = masked_sentinel2.median()
    
    return composite.clip(roi)

def create_mask(modis_img, esa_img, mapping):
    """
    --------------------------------------------------------------------------
    Function: create_mask
    --------------------------------------------------------------------------
    Parameters:
        - modis_img (ee.Image): MODIS classification image.
        - esa_img (ee.Image): ESA classification image.
        - mapping (dict): Mapping of ESA classes to lists of MODIS classes.
        
    Returns:
        - ee.Image: Binary mask image.
            - 1 where ESA and MODIS class conditions match, else 0.
        
    Description:
        Build a mask by iterating over ESA classes and marking pixels where
        the corresponding MODIS classes are present.
        
    Mathematical Details:
        - And, Or: combine boolean conditions for multi-class matching.
        
    Example:
        ```python
        - mask = create_mask(modis_img, esa_img, {10:[1,2],20:[3,4]})
        ```
    """
    mask = ee.Image.constant(0)
    for esa_class, modis_classes in mapping.items():
        esa_condition = esa_img.eq(esa_class)
        modis_condition = modis_img.eq(modis_classes[0])
        for modis_class in modis_classes[1:]:
            modis_condition = modis_condition.Or(modis_img.eq(modis_class))
        combined_condition = esa_condition.And(modis_condition)
        mask = mask.Or(combined_condition)
    return mask


def get_training_samples(year, extended_roi):
    """
    --------------------------------------------------------------------------
    Function: get_training_samples
    --------------------------------------------------------------------------
    Parameters:
        - year (int): Year for which to extract training samples.
        - extended_roi (ee.Geometry): Region of interest, possibly buffered.
        
    Returns:
        - ee.FeatureCollection: Training samples with labels.
        - ee.List: List of feature band names.
        
    Description:
        Generate training data by masking MODIS classes with ESA conditions,
        perform random and stratified sampling, and extract Sentinel-2 features.
        
    Mathematical Details:
        - None
        
    Example:
        ```python
        - training, bands = get_training_samples(2020, extended_roi)
        ```
    """
    # Create annual Sentinel-2 composite for the specified year
    sentinel_composite = get_sentinel2_year_composite(f'{year}-01-01', f'{year}-12-31', extended_roi)
    esa_world_cover = ee.Image('ESA/WorldCover/v100/2020').clip(extended_roi)
    modis_igbp = ee.ImageCollection('MODIS/061/MCD12Q1').filter(ee.Filter.date('2020-01-01', '2020-12-31')).first().select('LC_Type1').clip(extended_roi)
    # Mapping from ESA classes to MODIS classes
    esa_to_modis_mapping = {
        10: [1, 2, 3, 4, 5, 6, 7, 8, 9], 
        20: [1, 2, 3, 4, 5, 6, 7, 8, 9], 
        30: [6, 7, 8, 9, 10],  
        40: [12, 14], 
        50: [13],  
        60: [16],  
        70: [15],  
        80: [17],  
        90: [11],  
        95: [1, 2, 3, 4, 5],  
        100: [10]  
    }

    mask = create_mask(modis_igbp, esa_world_cover, esa_to_modis_mapping)
    filtered_modis = modis_igbp.updateMask(mask)

    samples = filtered_modis.sample(**{
        'region': extended_roi,
        'scale': 10,
        'numPixels': 5000,
        'geometries': True
    })

    valid_mask = filtered_modis.mask()
    stratified_samples = filtered_modis.updateMask(valid_mask).stratifiedSample(**{
        'numPoints': 500,
        'classBand': 'LC_Type1',
        'scale': 10,
        'region': extended_roi,
        'geometries': True
    })

    combined_samples = samples.merge(stratified_samples)

    bands = sentinel_composite.bandNames()
    label = 'LC_Type1'

    training = sentinel_composite.select(bands).sampleRegions(**{
        'collection': combined_samples,
        'properties': [label],
        'scale': 10
    })

    return training, bands

def classify_sentinel2_using_training(year, roi):
    """
    --------------------------------------------------------------------------
    Function: classify_sentinel2_using_training
    --------------------------------------------------------------------------
    Parameters:
        - year (int): Year for classification.
        - roi (ee.Geometry): Region of interest for composite and sampling.
        
    Returns:
        - ee.Image or None: Classified image or None if no data.
        
    Description:
        Train a Random Forest classifier using sampled Sentinel-2 training data
        and classify the annual composite image.
        
    Mathematical Details:
        - None
        
    Example:
        ```python
        - result = classify_sentinel2_using_training(2020, roi)
        ```
    """
    sentinel_composite = get_sentinel2_year_composite(f'{year}-01-01', f'{year}-12-31', roi)
    extended_roi = roi.buffer(10000) 

    training, bands = get_training_samples(year, extended_roi)
    if sentinel_composite is None:
        return None

    classifier = ee.Classifier.smileRandomForest(50).train(training, 'LC_Type1', bands)
    classified = sentinel_composite.classify(classifier)

    return classified