# ------------------------------------------------------------------------------
# File: get_DEFMAP.py
# ------------------------------------------------------------------------------
# Description:
#     Utilities to retrieve and process MODIS burn data at 500 m and 5 km resolutions,
#     and to compute dynamic emission factor maps based on burn occurrences
#     and land use/land cover changes across previous and current periods.
#
# Inputs:
#     - year (int): target year for annual MODIS burn data retrieval.
#     - EFMAP_p (ee.Image): emission factor map for the previous period (pre-LUCC).
#     - EFMAP_c (ee.Image): emission factor map for the current period (post-LUCC).
#     - date_str (str): target date in 'YYYY-MM-DD' format.
#     - burn_date_image (ee.Image): per-pixel first burn day-of-year image.
#     - burn_fraction_image (ee.Image): per-pixel burned area fraction image.
#
# Functions:
#     - accumulate_first_burn(image, previous) -> ee.Image
#         Updates accumulated burn mask to record first burn occurrence.
#     - get_burn_data_500m(year) -> (ee.Image, ee.Image)
#         Generates binary burned area mask and first burn date image at 500 m.
#     - get_burn_data_5000m(year) -> (ee.Image, ee.Image)
#         Retrieves annual burned area fraction and first burn date at 5 km.
#     - calculate_defmap(EFMAP_p, EFMAP_c, date_str, burn_date_image, burn_fraction_image) -> ee.Image
#         Linearly interpolates emission factor maps and applies dynamic reduction
#         based on burn events and LUCC-driven map changes.
#     - calculate_defmap_only_fire(EFMAP_c, date_str, burn_date_image, burn_fraction_image) -> ee.Image
#         Applies burn-based reduction to the current emission factor map for fire effects.
#
# Outputs:
#     - ee.Image objects representing burned area masks, burn dates, and dynamically adjusted emission maps.
#
# Main Functionality:
#     1. Load and filter MODIS MCD64A1 collections by year.
#     2. Accumulate first burn dates and generate binary masks.
#     3. Retrieve summary burn data at 5 km resolution.
#     4. Compute daily emission factor adjustments based on burn timing, severity,
#        and land use/land cover changes between pre- and post-period maps.
#
# Dependencies:
#     - ee
#     - datetime
# ------------------------------------------------------------------------------

import ee
from datetime import datetime


def accumulate_first_burn(image, previous):
    """
    --------------------------------------------------------------------------
    Function: accumulate_first_burn
    --------------------------------------------------------------------------
    Parameters:
        - image (ee.Image): current burn mask image for this time step
        - previous (ee.Image): accumulated burn mask from prior time steps

    Returns:
        - ee.Image: updated burn mask retaining first-burn occurrence

    Description:
        Updates an accumulated burn mask to record the first occurrence of burn.
        Pixels retain their original non-zero value once burned, and new burns
        are recorded where previous mask is zero.

    Mathematical Details:
        - updated(x) = previous(x) if previous(x) ≠ 0 else image(x)

    Example:
        ```python
        - result = accumulate_first_burn(current_burn, accumulated_burn)
        ```
    """
    # convert previous to an Earth Engine Image
    previous = ee.Image(previous)
    
    # update pixels that have not burned before: keep non-zero values, else use current image
    updated = previous.where(previous.eq(0), image)
    return updated

def get_burn_data_500m(year):
    """
    --------------------------------------------------------------------------
    Function: get_burn_data_500m
    --------------------------------------------------------------------------
    Parameters:
        - year (int): target year for burn data aggregation

    Returns:
        - burned_areas (ee.Image): binary mask of burned areas for the year
          - 1 indicates at least one burn event occurred
          - 0 indicates no burn event
        - sum_burn_date (ee.Image): first burn date per pixel aggregated over year
          - pixel values represent day-of-year of first burn
          - masked where no burn occurred

    Description:
        Retrieves the MODIS MCD64A1 burn date collection for a given year,
        accumulates the first burn occurrence per pixel, masks out unburned
        pixels, and generates a binary burned area mask.

    Mathematical Details:
        - sum_burn_date = first non-zero BurnDate value across all images.
        - burned_areas(x) = 1 if sum_burn_date(x) > 0, else 0.

    Example:
        ```python
        - burned_mask, burn_dates = get_burn_data_500m(2020)
        ```
    """
    # Load MODIS MCD64A1 burn date image collection
    dataset = ee.ImageCollection('MODIS/061/MCD64A1')

    start_date = ee.Date.fromYMD(year, 1, 1)
    end_date = ee.Date.fromYMD(year + 1, 1, 1)
    filtered = dataset.filterDate(start_date, end_date)
    
    # Select the 'BurnDate' band from each image
    burn_date = filtered.select('BurnDate')
    # Accumulate first burn date per pixel using iterate, starting from zero image
    sum_burn_date = burn_date.iterate(accumulate_first_burn, ee.Image(0))
    # Mask out pixels with no burn occurrence (value zero)
    sum_burn_date = ee.Image(sum_burn_date).updateMask(ee.Image(sum_burn_date).neq(0))
    # Create binary burned area mask: 1 where burn occurred, else 0
    burned_areas = sum_burn_date.gt(0).unmask(0)
    return burned_areas,sum_burn_date

def get_burn_data_5000m(year):
    """
    --------------------------------------------------------------------------
    Function: get_burn_data_5000m
    --------------------------------------------------------------------------
    Parameters:
        - year (int): target year for 5 km burn data retrieval

    Returns:
        - burnedAreaFraction (ee.Image): fraction of area burned per pixel
        - burnedDate (ee.Image): day-of-year image of first burn date

    Description:
        Retrieves annual burned area fraction and first burn date from a 5 km
        MODIS-derived dataset for the specified year. Uses the first image in
        the filtered collection as the annual summary.

    Example:
        ```python
        - fraction, date = get_burn_data_5000m(2020)
        ```
    """
    startDate = ee.Date.fromYMD(year, 1, 1)
    endDate = ee.Date.fromYMD(year, 12, 31)
    dataset = ee.ImageCollection('projects/gee-megan/assets/MCD64A1_5km').filterDate(startDate, endDate)
    burnedAreaFraction = dataset.first().select('b1')
    burnedDate = dataset.first().select('b2')
    return burnedAreaFraction,burnedDate

def calculate_defmap(EFMAP_p, EFMAP_c, date_str, burn_date_image, burn_fraction_image):
    """
    --------------------------------------------------------------------------
    Function: calculate_defmap
    --------------------------------------------------------------------------
    Parameters:
        - EFMAP_p (ee.Image or image-like): EFMAP image for the previous period
        - EFMAP_c (ee.Image or image-like): EFMAP image for the current period
        - date_str (str): target date in 'YYYY-MM-DD' format
        - burn_date_image (ee.Image or image-like): per-pixel first burn day-of-year
        - burn_fraction_image (ee.Image or image-like): per-pixel burned area fraction
    
    Returns:
        - Dynamic_EFMAPday (ee.Image): daily EFMAP adjusted for burns
    
    Description:
        Computes a daily EFMAP by linearly interpolating between previous and
        current period maps based on the day-of-year, then reduces EFMAP where
        burning has occurred using the burn fraction.
    
    Mathematical Details:
        - DOY = day of year of date_str.
        - TDY = total days in year (365 or 366).
        - EFMAPday = EFMAP_p + (EFMAP_c − EFMAP_p) × DOY / TDY.
        - Dynamic_EFMAPday(x) = EFMAPday(x) × (1 − burn_fraction(x)) if burn_date(x) ≤ DOY, else EFMAPday(x).
    
    Example:
        ```python
        - result = calculate_defmap(prev_map, curr_map, '2025-06-17',
                                   burn_date_img, burn_frac_img)
        ```
    """
    date = datetime.strptime(date_str, '%Y-%m-%d')
    DOY = date.timetuple().tm_yday
    year = date.year
    # determine if current year is leap year to set total days
    is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    TDY = 366 if is_leap_year else 365


    EFMAP_p = ee.Image(EFMAP_p)
    EFMAP_c = ee.Image(EFMAP_c)
    burn_date = ee.Image(burn_date_image)
    burn_fraction = ee.Image(burn_fraction_image)


    EFMAPday = EFMAP_p.add((EFMAP_c.subtract(EFMAP_c)).multiply(DOY).divide(TDY))  # Assuming non-leap year for simplicity


    condition_burn = burn_date.gt(0).And(burn_date.lte(DOY))
    EFMAPday_fra = EFMAPday.subtract(EFMAPday.multiply( burn_fraction))
    Dynamic_EFMAPday = EFMAPday.where(condition_burn.gt(0),EFMAPday_fra)

    return Dynamic_EFMAPday

def calculate_defmap_only_fire(EFMAP_c, date_str, burn_date_image, burn_fraction_image):
    """
    --------------------------------------------------------------------------
    Function: calculate_defmap_only_fire
    --------------------------------------------------------------------------
    Parameters:
        - EFMAP_c (ee.Image): EFMAP image for the current period
        - date_str (str): target date in 'YYYY-MM-DD' format
        - burn_date_image (ee.Image): per-pixel first burn day-of-year
        - burn_fraction_image (ee.Image): per-pixel burned area fraction

    Returns:
        - Dynamic_EFMAPday (ee.Image): daily EFMAP adjusted for fire effects
            - pixels reduced by burn fraction where burn occurred

    Description:
        Uses the current EFMAP map directly for daily values, then applies
        a reduction where burning has occurred. Burned pixels are scaled by
        (1 - burn_fraction) for days on or after fire occurrence.

    Mathematical Details:
        - DOY = day of year of date_str.
        - EFMAPday = EFMAP_c.
        - Dynamic_EFMAPday(x) = EFMAPday(x) * (1 - burn_fraction(x)) if burn_date(x) ≤ DOY, else EFMAPday(x).

    Example:
        ```python
        - result = calculate_defmap_only_fire(EFMAP_current, '2025-06-17',
                                             burn_date_img, burn_frac_img)
        ```
    """
    date = datetime.strptime(date_str, '%Y-%m-%d')
    DOY = date.timetuple().tm_yday
    year = date.year

    is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    TDY = 366 if is_leap_year else 365

    EFMAP_c = ee.Image(EFMAP_c)
    burn_date = ee.Image(burn_date_image)
    burn_fraction = ee.Image(burn_fraction_image)


    EFMAPday = EFMAP_c 

    condition_burn = burn_date.gt(0).And(burn_date.lte(DOY))
    EFMAPday_fra = EFMAPday.subtract(EFMAPday.multiply( burn_fraction))
    Dynamic_EFMAPday = EFMAPday.where(condition_burn.gt(0),EFMAPday_fra)

    return Dynamic_EFMAPday