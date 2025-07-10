# ------------------------------------------------------------------------------
# ECMWF_ERA5_Preprocess_lib.py
# ------------------------------------------------------------------------------
# Module for preprocessing and extracting meteorological variables from the
# ECMWF/ERA5_LAND/HOURLY dataset in Google Earth Engine.
#
# Description:
#   This library provides functions to:
#      - Filter hourly ERA5-LAND images by date and select specific bands.
#      - Compute hourly relative humidity, specific humidity (g/kg), and mixing ratio (kg/kg).
#      - Calculate 10 m wind speed and instantaneous solar radiation rates.
#      - Generate per-pixel longitude/latitude rasters and compute local time and Julian day.
#      - Reclassify MODIS land-cover into 16 Plant Functional Types (PFTs) using Köppen–Geiger zones.
#      - Combine MODIS IGBP land-cover classification with climate zones to derive PFT fractions.
#      - Build chemical emission rasters by applying emission factors to PFT bands.
#
# Dependencies:
#   - ee
# ------------------------------------------------------------------------------

import ee


def filterByDate(dateStr, collectionId, bandName):
    """
    --------------------------------------------------------------------------
    Function: filterByDate
    --------------------------------------------------------------------------
    Parameters:
        - dateStr (str): Date string in 'YYYY-MM-DD' format to filter the collection.
        - collectionId (str): ID of the Earth Engine ImageCollection to filter.
        - bandName (str or list): Name or list of names of band(s) to select.

    Returns:
        - ee.ImageCollection: Images from the specified date containing only the selected band(s).

    Description:
        Converts the input date string to an ee.Date and defines a 24-hour window.
        Primarily intended for the 'ECMWF/ERA5_LAND/HOURLY' dataset to extract hourly fields.

    Mathematical Details:
        - nextDay = inputDate.advance(1, 'day'): computes the exclusive end date for filtering.

    Example:
        ```python
        - example call
            - result = filterByDate('2025-06-17', 'ECMWF/ERA5_LAND/HOURLY', 'surface_pressure')
        ```
    """
    # Convert the input date string to an Earth Engine Date object
    inputDate = ee.Date(dateStr)
    
    # Compute the next day for exclusive end of filter range
    nextDay = inputDate.advance(1, 'day')
    
    # Filter the 'ECMWF/ERA5_LAND/HOURLY' collection for the given date and select the band(s)
    filter_dataset = ee.ImageCollection(collectionId)\
        .filterDate(inputDate, nextDay)\
        .select(bandName)
    
    return filter_dataset

def calculateRelativeHumidity(dewpointTemperatureDataset, temperatureDataset):
    """
    --------------------------------------------------------------------------
    Function: calculateRelativeHumidity
    --------------------------------------------------------------------------
    Parameters:
        - dewpointTemperatureDataset (ee.ImageCollection): hourly dewpoint temperature in Kelvin.
        - temperatureDataset (ee.ImageCollection): hourly surface temperature in Kelvin.

    Returns:
        - ee.ImageCollection: ImageCollection of hourly relative humidity (%).
          - Band name: 'relativeHumidity'.

    Description:
        Converts dewpoint and temperature from Kelvin to Celsius, computes saturation
        Base on Clausius-Clapeyron
        vapor pressures at both, and calculates relative humidity percentage.
        Processes each image in the collections sequentially.

    Mathematical Details:
        - es = 6.112 * exp((17.67 * T) / (T + 243.5))
        - RH = (es_dew / es) * 100

    Example:
        ```python
        - example call
            - result = calculateRelativeHumidity(dewDataset, tempDataset)
        ```
    """
    sizes = ee.Number(dewpointTemperatureDataset.size())

    def calc_relative_humidity(index):
        dewpointTemperatureC = ee.Image(dewpointTemperatureDataset.toList(sizes).get(index)).subtract(273.15)
        temperatureC = ee.Image(temperatureDataset.toList(sizes).get(index)).subtract(273.15)

        saturationVaporPressure = ee.Image(6.112).multiply(temperatureC.divide(temperatureC.add(243.5)).multiply(17.67).exp())
        saturationVaporPressureDewpoint = ee.Image(6.112).multiply(dewpointTemperatureC.divide(dewpointTemperatureC.add(243.5)).multiply(17.67).exp())

        relativeHumidity = saturationVaporPressureDewpoint.divide(saturationVaporPressure).multiply(100)
        relativeHumidity = relativeHumidity.rename('relativeHumidity')
        
        return relativeHumidity

    results = ee.List.sequence(0, ee.Number(sizes).subtract(1)).map(calc_relative_humidity)

    return ee.ImageCollection(results)

def calculate_specific_humidity(temperatureDataset, dewpointTemperatureDataset, pressureDataset):
    """
    --------------------------------------------------------------------------
    Function: calculate_specific_humidity
    --------------------------------------------------------------------------
    Parameters:
        - temperatureDataset (ee.ImageCollection): Hourly air temperature in Kelvin.
        - dewpointTemperatureDataset (ee.ImageCollection): Hourly dewpoint temperature in Kelvin.
        - pressureDataset (ee.ImageCollection): Hourly atmospheric pressure in Pascal (Pa).
        
    Returns:
        - ee.ImageCollection: Specific humidity in g/kg for each hourly image.
            - Band name: 'specific_humidity'.

    Description:
        Computes specific humidity using dewpoint temperature and pressure.
        Temperature is only used for consistency in the function signature.
        Returns an image collection with a single band representing specific humidity.

    Mathematical Details:
        - Td(°C) = Td(K) - 273.15: converts dewpoint from Kelvin to Celsius.
        - e = 6.112 * exp[(17.67 * Td) / (Td + 243.5)]: calculates actual vapor pressure in hPa.
        - q = (0.622 * e / P) * 1000: specific humidity in g/kg, where P is pressure in hPa.

    Example:
        ```python
        - result = calculate_specific_humidity(tempIC, dewIC, pressureIC)
        - result.first().select('specific_humidity').getInfo()
        ```
    """
    sizes = ee.Number(temperatureDataset.size())

    def calc_specific_humidity(index):
        temperatureC = ee.Image(temperatureDataset.toList(sizes).get(index)).subtract(273.15)
        dewpointTemperatureC = ee.Image(dewpointTemperatureDataset.toList(sizes).get(index)).subtract(273.15)
        pressure_hPa = ee.Image(pressureDataset.toList(sizes).get(index)).divide(100)
        
       
        # Compute actual vapor pressure e (hPa)
        e = ee.Image(6.112).multiply(
            dewpointTemperatureC.multiply(17.67).divide(dewpointTemperatureC.add(243.5)).exp()
        )
        # Compute specific humidity q (g/kg)
        q = e.multiply(0.622).divide(pressure_hPa).multiply(1000)
        
        # 将比湿添加为新的波段
        specific_humidity = q.rename('specific_humidity')
        
        return specific_humidity

    results = ee.List.sequence(0, ee.Number(sizes).subtract(1)).map(calc_specific_humidity)

    return ee.ImageCollection(results)

def calculate_water_vapor_mixing_ratio(temperatureDataset, dewpointTemperatureDataset, pressureDataset):
    """
    --------------------------------------------------------------------------
    Function: calculate_water_vapor_mixing_ratio
    --------------------------------------------------------------------------
    Parameters:
        - temperatureDataset (ee.ImageCollection): Hourly air temperature in Kelvin.
        - dewpointTemperatureDataset (ee.ImageCollection): Hourly dewpoint temperature in Kelvin.
        - pressureDataset (ee.ImageCollection): Hourly atmospheric pressure in Pascal (Pa).
        
    Returns:
        - ee.ImageCollection: Mixing ratio in kg/kg for each hourly image.
            - Band name: 'mixing_ratio'.

    Description:
        Computes the mixing ratio of water vapor to dry air using dewpoint temperature and total pressure.
        Temperature is not used in the actual computation but retained for interface consistency.

    Mathematical Details:
        - Td(°C) = Td(K) - 273.15: converts dewpoint from Kelvin to Celsius.
        - e = 6.112 * exp[(17.67 * Td) / (Td + 243.5)]: actual vapor pressure (hPa).
        - P = pressure in hPa = pressure / 100.
        - r = (0.622 * e) / (P - e): mixing ratio (kg/kg)

    Example:
        ```python
        - result = calculate_mixing_ratio(tempIC, dewIC, pressureIC)
        ```
    """

    sizes = ee.Number(temperatureDataset.size())

    def calc_mixing_ratio(index):
        # Convert temperatures from Kelvin to Celsius
        temperatureC = ee.Image(temperatureDataset.toList(sizes).get(index)).subtract(273.15)
        dewpointTemperatureC = ee.Image(dewpointTemperatureDataset.toList(sizes).get(index)).subtract(273.15)
        
        # Convert pressure from Pa to hPa
        pressure_hPa = ee.Image(pressureDataset.toList(sizes).get(index)).divide(100)
        
        # Compute actual vapor pressure e (hPa) from dewpoint temperature
        e = ee.Image(6.112).multiply(
            dewpointTemperatureC.multiply(17.67).divide(dewpointTemperatureC.add(243.5)).exp()
        )
        
        # Compute mixing ratio r (kg/kg)
        r = e.multiply(0.622).divide(pressure_hPa.subtract(e))
        
        # Rename output band
        mixing_ratio = r.rename('mixing_ratio')
        
        return mixing_ratio

    results = ee.List.sequence(0, ee.Number(sizes).subtract(1)).map(calc_mixing_ratio)

    return ee.ImageCollection(results)


# 风速数据集计算
def calculateWindSpeed(u_wind_10m_Dataset, v_wind_10m_Dataset):
    """
    --------------------------------------------------------------------------
    Function: calculateWindSpeed
    --------------------------------------------------------------------------
    Parameters:
        - u_wind_10m_Dataset (ee.ImageCollection): collection of u-component wind at 10m.
        - v_wind_10m_Dataset (ee.ImageCollection): collection of v-component wind at 10m.

    Returns:
        - ee.ImageCollection: collection of wind speed images at 10m.
            - each image has band 'wind_speed_10m' and carries system:time_start property.

    Description:
        This function computes the wind speed at 10 meters height from u and v wind components.
        It iterates through the image collections, computes the magnitude of wind vectors,
        and preserves the timestamp of each image.

    Mathematical Details:
        - windSpeed = sqrt(u^2 + v^2): computes wind speed magnitude from u and v components.

    Example:
        ```python
        result = calculateWindSpeed(u_collection, v_collection)
        ```
    """
    # Get the size of the input datasets
    u_wind_10m_size = u_wind_10m_Dataset.size()
    v_wind_10m_size = v_wind_10m_Dataset.size()
    sizes = u_wind_10m_size

    # Define a function to compute wind speed for each image
    def calc_windSpeed(index):
        u_wind_10m = ee.Image(u_wind_10m_Dataset.toList(sizes).get(index))
        v_wind_10m = ee.Image(v_wind_10m_Dataset.toList(sizes).get(index))

        # Retrieve u and v wind components for the given index
        windSpeed = (u_wind_10m.pow(2).add(v_wind_10m.pow(2))).sqrt()
        windSpeed = windSpeed.rename('wind_speed_10m')

        # Compute wind speed magnitude from u and v components
        time = u_wind_10m.get('system:time_start')
        windSpeed = windSpeed.set('system:time_start', time)

        return windSpeed

    results = ee.List.sequence(0, ee.Number(sizes).subtract(1)).map(calc_windSpeed)

    return ee.ImageCollection(results)



def calcInstantaneousSolarRadiation(filteredDataset, previousDayDataset):
    """
    --------------------------------------------------------------------------
    Function: calcInstantaneousSolarRadiation
    --------------------------------------------------------------------------
    Parameters:
        - filteredDataset (ee.ImageCollection): collection of cumulative solar radiation for the current day.
        - previousDayDataset (ee.ImageCollection): collection of cumulative solar radiation for the previous day.

    Returns:
        - ee.ImageCollection: collection of instantaneous solar radiation rate images.
            - values represent radiation difference per second.

    Description:
        Computes instantaneous solar radiation by differencing cumulative values.
        For the first image, uses the last image of the previous day.
        Divides differences by 3600 to convert from per-hour accumulation to per-second rate.

    Mathematical Details:
        - instantaneousRadiation = (current - previous) / 3600: converts hourly accumulation to per-second rate.

    Example:
        ```python
        result = calcInstantaneousSolarRadiation(todayCumulative, yesterdayCumulative)
        ```
    """
    # Get the number of images in current day's dataset
    currentDaysize = filteredDataset.size()
    currentDayDatasetList = filteredDataset.toList(currentDaysize)
    
    # Get the number of images in previous day's dataset
    previous_size = previousDayDataset.size()
    previousDayDatasetList = previousDayDataset.toList(previous_size)
     # Retrieve last image from previous day for initial difference
    previousDayLastImage = ee.Image(previousDayDatasetList.get(ee.Number(previous_size).subtract(1)))
    
    def calc_instantaneousRadiation(index):
        index = ee.Number(index)
        currentImage = ee.Image(currentDayDatasetList.get(index))


        instantaneousRadiation = ee.Image(ee.Algorithms.If(
            index.eq(0),
            currentImage.subtract(previousDayLastImage).divide(3600),
            ee.Algorithms.If(
                index.eq(1),
                currentImage.divide(3600),
                currentImage.subtract(ee.Image(currentDayDatasetList.get(index.subtract(1)))).divide(3600)
            )
        ))
        
        return instantaneousRadiation

    instantaneousRadiationList = ee.List.sequence(0,ee.Number(currentDaysize).subtract(1)).map(calc_instantaneousRadiation)
    instantaneousRadiationCollection = ee.ImageCollection(instantaneousRadiationList)
    
    return instantaneousRadiationCollection

def create_lon_lat_image(input_image, roi):
    """
    --------------------------------------------------------------------------
    Function: create_lon_lat_image
    --------------------------------------------------------------------------
    Parameters:
        - input_image (ee.Image): the image defining desired resolution and mask.
        - roi (ee.Geometry): region of interest to clip the grids.

    Returns:
        - ee.Image: image containing two bands, 'longitude' and 'latitude', matching the input resolution.

    Description:
        Generates rasters of longitude and latitude values at the center of each pixel.
        Clips and masks the grids to the specified region of interest.

    Mathematical Details:
        None

    Example:
        ```python
        grid = create_lon_lat_image(myImage, myRegion)
        ```
    """
    lon = ee.Image.pixelLonLat().select('longitude').clip(roi)
    lon_new = input_image.mask()
    lon_new = lon_new.clip(roi).multiply(lon)
    lon_new = lon_new.updateMask(input_image.mask()).rename('longitude')
    
    lat = ee.Image.pixelLonLat().select('latitude').clip(roi)
    lat_new = input_image.mask()
    lat_new = lat_new.clip(roi).multiply(lat)
    lat_new = lat_new.updateMask(input_image.mask()).rename('latitude')
    
     # Create a new image concatenating longitude and latitude bands
    lon_lat_image = ee.Image.cat([lon_new, lat_new])
    
    return lon_lat_image

def create_lon_lat_image_without_roi(input_image):
    """
    --------------------------------------------------------------------------
    Function: create_lon_lat_image_without_roi
    --------------------------------------------------------------------------
    Parameters:
        - input_image (ee.Image): the image defining desired resolution and mask.

    Returns:
        - ee.Image: image containing two bands, 'longitude' and 'latitude', for each pixel center.

    Description:
        Generates rasters of longitude and latitude values at the center of each pixel
        matching the spatial resolution and mask of the input image.
        Does not apply any region clipping.

    Mathematical Details:
        None

    Example:
        ```python
        grid = create_lon_lat_image_without_roi(myImage)
        ```
    """
    lon = ee.Image.pixelLonLat().select('longitude')
    lon_new = input_image.mask()
    lon_new = lon_new.multiply(lon)
    lon_new = lon_new.updateMask(input_image.mask()).rename('longitude')
    
    lat = ee.Image.pixelLonLat().select('latitude')
    lat_new = input_image.mask()
    lat_new = lat_new.multiply(lat)
    lat_new = lat_new.updateMask(input_image.mask()).rename('latitude')
    

    lon_lat_image = ee.Image.cat([lon_new, lat_new])
    
    return lon_lat_image

def computeLocalTime(lon_lat_image, utcTimeHours, dateString):
    """
    --------------------------------------------------------------------------
    Function: computeLocalTime
    --------------------------------------------------------------------------
    Parameters:
        - lon_lat_image (ee.Image): image with 'longitude' band for each pixel center.
        - utcTimeHours (numeric): UTC time in hours.
        - dateString (string): date in ISO format (e.g., '2025-06-17').

    Returns:
        - ee.Image: original image plus two bands:
            - 'local_time': per-pixel local hour (0–23).
            - 'julian_day': per-pixel Julian day of local date.

    Description:
        Calculates per-pixel local solar time and Julian day based on longitude.
        Converts longitude to time-zone offset, adjusts UTC hours, and handles day shifts.
        Accounts for cross-midnight and leap-year cases when computing Julian day.

    Mathematical Details:
        - timeZone = floor((longitude + 7.5) / 15): converts longitude to integer time-zone.
        - localTimeSeconds = utcTimeHours*3600 + timeZone*3600: computes local time in seconds.
        - localTimeHours = (localTimeSeconds/3600) mod 24: wraps hours into 0–23.
        - dayOffset = floor(localTimeHours_temp): shifts Julian day by ±1 as needed.

    Example:
        ```python
        output = computeLocalTime(gridImage, 8.0, '2025-06-17')
        ```
    """
    # Extract longitude band
    longitude = lon_lat_image.select('longitude')
    # Convert longitude to time-zone (one zone per 15°)
    timeZone = longitude.add(7.5).divide(15).floor()

    utcTimeSeconds = utcTimeHours * 3600

    timeZoneOffset = timeZone.multiply(3600)

    localTimeSeconds = timeZoneOffset.add(utcTimeSeconds)
    # Convert to local hours and wrap into 0–23
    localTimeHours = localTimeSeconds.divide(3600).mod(24)
    localTimeHours_temp = localTimeSeconds.divide(3600).divide(24)
    # Julian-day
    inputDate = ee.Date(dateString)
    startOfYear = ee.Date.fromYMD(inputDate.get('year'), 1, 1)
    dayOfYear = inputDate.difference(startOfYear, 'day').add(1)

    julianDayImage = ee.Image(dayOfYear).rename('julian_day')

    adjustedJulianDayImage = julianDayImage.subtract(localTimeHours.lt(0))
    adjustedJulianDayImage = adjustedJulianDayImage.where(localTimeHours_temp.gte(1), adjustedJulianDayImage.add(1))
    # Handle leap-year for cross-year adjustment
    yearImage = ee.Image(inputDate.get('year')).rename('year')
    isLeapYearImage = yearImage.mod(4).eq(0).And(yearImage.mod(100).neq(0)).Or(yearImage.mod(400).eq(0))
    daysInPrevYear = isLeapYearImage.multiply(366).add(isLeapYearImage.Not().multiply(365))

    finalJulianDayImage = adjustedJulianDayImage.add(daysInPrevYear.multiply(adjustedJulianDayImage.lte(0)))
    localTimeHours = localTimeHours.abs()
    localTimeHours_ = longitude.divide(15).add(utcTimeHours)
    localTimeHours = localTimeHours_.where(localTimeHours_.lt(0),localTimeHours_.add(24))
    outputImage = lon_lat_image.addBands(localTimeHours.rename('local_time')).addBands(finalJulianDayImage.rename('julian_day'))
    
    return outputImage



def ReClassPFTS(year):
    """
    --------------------------------------------------------------------------
    Function: ReClassPFTS
    --------------------------------------------------------------------------
    Parameters:
        - year (int): Year for which to reclassify PFT  (e.g., 2001).

    Returns:
        - ee.Image: image with bands 'C1' through 'C16' representing global PFT fractions for the given year.

    Description:
        Loads MODIS land cover and Köppen-Geiger climate classification for the specified year,
        reclassifies climate zones, and computes weighted cover fractions for 16 plant functional types.
        Combines MODIS and climate data according to predefined conversion rules.
        Uses climate classification data from Beck et al. (2018) Sci Data 5:180214.
        https://doi.org/10.1038/sdata.2018.214

    Mathematical Details:
        - result = Σ(mask * factor) over all MODIS and climate class combinations: builds PFT fractional cover.

    Example:
        ```python
        pftImage = ReClassPFTS(2020)
        ```
    """

    startDate = ee.Date.fromYMD(year, 1, 1)
    endDate = ee.Date.fromYMD(year, 12, 31)
    # Modis classes
    modisClasses = {
        'M1': 1,
        'M2': 2,
        'M3': 3,
        'M4': 4,
        'M5': 5,
        'M6': 6,
        'M7': 7,
        'M8': 8,
        'M9': 9,
        'M10': 10,
        'M11': 11,
        'M12': 12,
        'M13': 13,
        'M14': 14,
        'M15': 15,
        'M16': 16
    }
    # Koppen classes
    koppenClasses = {
        'A': 1,
        'Bh': 2,
        'Bk': 3,
        'Ca': 4,
        'Cb': 5,
        'Da': 6,
        'Db': 7,
        'E' : 8    }


    dataset = ee.ImageCollection('MODIS/061/MCD12Q1').filterDate(startDate, endDate)

    # Beck, H., Zimmermann, N., McVicar, T. et al. 
    # Present and future Köppen-Geiger climate classification maps at 1-km resolution. 
    # Sci Data 5, 180214 (2018). https://doi.org/10.1038/sdata.2018.214

    koppen = ee.Image('projects/gee-megan/assets/Beck_KG_V1_present_0p0083')

    MODISImage = dataset.select('LC_Type1').first()

    zero_map = MODISImage.multiply(0)
    koppen_reclass = zero_map.where(koppen.gte(1).And(koppen.lte(3)),1)                             # A
    koppen_reclass = koppen_reclass.where(koppen.eq(4).Or(koppen.eq(6)),2)                          # Bh
    koppen_reclass = koppen_reclass.where(koppen.eq(5).Or(koppen.eq(7)),3)                          # Bk
    koppen_reclass = koppen_reclass.where(koppen.eq(8).Or(koppen.eq(11)).Or(koppen.eq(14)),4)       # Ca
    koppen_reclass = koppen_reclass.where(koppen.eq(9).Or(koppen.eq(10)).Or(koppen.eq(12))
                                    .Or(koppen.eq(13)).Or(koppen.eq(15)).Or(koppen.eq(16)),5)       # Cb
    koppen_reclass = koppen_reclass.where(koppen.eq(17).Or(koppen.eq(18)).Or(koppen.eq(21))
                                    .Or(koppen.eq(22)).Or(koppen.eq(25)).Or(koppen.eq(26)),6)       # Da
    koppen_reclass = koppen_reclass.where(koppen.eq(19).Or(koppen.eq(20)).Or(koppen.eq(23))
                                    .Or(koppen.eq(24)).Or(koppen.eq(27)).Or(koppen.eq(28)),7)       # Db
    koppen_reclass = koppen_reclass.where(koppen.eq(29).Or(koppen.eq(30)),8)                        # E




    # Create an image of zeroes
    def perform_conversion(conversion_list,koppen,modis,modisClasses,koppenClasses):
        result = ee.Image(0)
        for conv in conversion_list:
            band, factor = conv.split('*')
            modiss_key,koppen_key = band.split('_')
            temp  = modis.multiply(0)
            temp = temp.where(modis.eq(modisClasses[modiss_key]).And(koppen.eq(koppenClasses[koppen_key])),1)\
                    .multiply(ee.Number(float(factor)))
                # print(band,factor)
            result = result.add(temp)
        return result
    class_dict = {'C1': ['M1_A*1', 'M1_Bh*1','M1_Ca*1','M1_Cb*1','M1_Da*0.5',
                    'M5_A*0.33','M5_Bh*0.33','M5_Ca*0.3','M5_Cb*0.33','M5_Da*0.2',
                        'M8_Bh*0.5','M8_Ca*0.3','M8_Cb*0.4','M8_Da*0.4',
                        'M9_Bh*0.33','M9_Ca*0.3','M9_Cb*0.3','M9_Da*0.3'],
                'C2': ['M1_Da*0.5','M1_Db*1', 'M1_Bk*1','M1_E*1','M2_Bh*0.5','M4_Bk*0.33',
                       'M5_Bk*0.33','M5_Db*0.33','M5_E*0.33',
                        'M8_Bk*0.8','M8_Db*0.8','M8_E*0.8',
                        'M9_Bk*0.208','M9_Db*0.208','M9_E*0.508'],
                'C3': ['M4_Bh*0.33','M4_Bk*0.33','M3_A*1', 'M3_Bh*1','M3_Bk*1','M3_Ca*1','M3_Cb*1', 'M3_Da*1','M3_Db*1', 'M3_E*1',
                        'M5_Bk*0.33','M5_Db*0.33','M5_E*0.33'],
                'C4': ['M2_A*1','M2_Bh*0.5','M2_Ca*0.5',
                        'M5_A*0.33','M5_Ca*0.1',
                        'M8_A*0.4','M8_Ca*0.1',
                        'M9_A*0.4',],
                'C5': ['M2_Bk*1','M2_Ca*0.5', 'M2_Cb*1','M2_Da*1','M2_Db*1','M2_E*1',
                        'M5_Bh*0.33','M5_Ca*0.2','M5_Cb*0.33','M5_Da*0.2',
                        'M8_Ca*0.2','M8_Cb*0.2','M8_Da*0.2',
                        'M9_Bh*0.33','M9_Ca*0.15','M9_Cb*0.15','M9_Da*0.15'],
                'C6': ['M4_A*1','M4_Bh*0.33','M4_Ca*0.5',
                        'M5_A*0.33','M5_Ca*0.1',
                        'M8_A*0.6','M8_Ca*0.1',
                        'M9_A*0.6',],
                'C7': ['M4_Bh*0.33','M4_Ca*0.5', 'M4_Cb*1','M4_Da*1',
                        'M5_Bh*0.33','M5_Ca*0.3','M5_Cb*0.33','M5_Da*0.6',
                        'M8_Bh*0.5','M8_Ca*0.3','M8_Cb*0.4','M8_Da*0.4',
                        'M9_Bh*0.33','M9_Ca*0.3','M9_Cb*0.3','M9_Da*0.3'],
                'C8': ['M4_Bk*0.33','M4_Db*1', 'M4_E*1',
                        'M5_Bk*0.33','M5_Db*0.33','M5_E*0.33',
                        'M8_Bk*0.2','M8_Db*0.2','M8_E*0.2',
                        'M9_Bk*0.132','M9_Db*0.132','M9_E*0.132'],
                'C9': ['M6_A*1',
                        'M7_A*1'],
                'C10':['M6_Bh*0.5','M6_Bk*0.5','M6_Ca*1','M6_Cb*1','M6_Da*1',
                        'M7_Bh*0.5','M7_Bk*0.5','M7_Ca*1','M7_Cb*1','M7_Da*1',],
                'C11':['M6_Bh*0.5','M6_Bk*0.5','M6_Db*1','M6_E*1',
                        'M7_Bh*0.5','M7_Bk*0.5','M7_Db*1','M7_E*1',],
                'C12': ['M10_E*1',
                        'M9_Bk*0.33','M9_Db*0.33','M9_E*0.33'],
                'C13': ['M10_Cb*1','M10_Bk*1','M10_Da*1',"M10_Db*1",
                        'M9_Bk*0.33','M9_Cb*0.25','M9_Da*0.25','M9_Db*0.25'],
                'C14': ['M10_A*1','M10_Bh*1','M10_Ca*1',
                        'M9_Ca*0.25']  ,     
                'C15': ['M12_A*1', 'M12_Bh*1','M12_Bk*1','M12_Ca*1','M12_Cb*1', 'M12_Da*1','M12_Db*1','M12_E*1',
                        'M14_A*1', 'M14_Bh*1','M14_Bk*1','M14_Ca*1','M14_Cb*1', 'M14_Da*1','M14_Db*1','M14_E*1',
                        ],
                'C16': ['M11_A*1', 'M11_Bh*1','M11_Bk*1','M11_Ca*1','M11_Cb*1', 'M11_Da*1','M11_Db*1','M11_E*1',
                        'M13_A*1', 'M13_Bh*1','M13_Bk*1','M13_Ca*1','M13_Cb*1', 'M13_Da*1','M13_Db*1','M13_E*1',
                        'M15_A*1', 'M15_Bh*1','M15_Bk*1','M15_Ca*1','M15_Cb*1', 'M15_Da*1','M15_Db*1','M15_E*1',
                        'M16_A*1', 'M16_Bh*1','M16_Bk*1','M16_Ca*1','M16_Cb*1', 'M16_Da*1','M16_Db*1','M16_E*1',]
    }
    converted_images = {key: perform_conversion(value,koppen_reclass,MODISImage,modisClasses,koppenClasses) for key, value in class_dict.items()}
    for i in converted_images:

        if i == 'C1':
            outputImage = ee.Image.cat([converted_images[i]]).rename(ee.String(i))
        else:
            outputImage = outputImage.addBands(ee.Image(converted_images[i]).rename(ee.String(i)))
    return outputImage
 


def ReClassPFTS_all_type(year,Collection_name,MODIS_Image =None,roi =None):
    """
    --------------------------------------------------------------------------
    Function: ReClassPFTS_all_type
    --------------------------------------------------------------------------
    Parameters:
        - year (int): Year for which to reclassify PFT  (e.g., 2001).
        - Collection_name (str): Earth Engine collection name for land-cover data.
        - MODIS_Image (ee.Image, optional): custom land-cover image when using a local collection.
        - roi (ee.Geometry, optional): region to clip local Köppen data.

    Returns:
        - ee.Image: image with bands 'C1' through 'C16' representing global PFT fractions for the given year.

    Description:
        Loads MODIS land cover and Köppen-Geiger climate classification for the specified year,
        reclassifies climate zones, and computes weighted cover fractions for 16 plant functional types.
        Combines MODIS and climate data according to predefined conversion rules.
        Uses climate classification data from Beck et al. (2018) Sci Data 5:180214.
        Sci Data 5, 180214 (2018). https://doi.org/10.1038/sdata.2018.214

    Mathematical Details:
        - result = Σ(mask * factor) over all MODIS and climate class combinations: builds PFT fractional cover.

    Example:
        ```python
        pft_all = ReClassPFTS_all_type(2020, 'MODIS/061/MCD12Q1')
        ```
    """

    startDate = ee.Date.fromYMD(year, 1, 1)
    endDate = ee.Date.fromYMD(year, 12, 31)
    # Modis classes
    modisClasses = {
        'M1': 1,
        'M2': 2,
        'M3': 3,
        'M4': 4,
        'M5': 5,
        'M6': 6,
        'M7': 7,
        'M8': 8,
        'M9': 9,
        'M10': 10,
        'M11': 11,
        'M12': 12,
        'M13': 13,
        'M14': 14,
        'M15': 15,
        'M16': 16
    }
    # Koppen classes
    koppenClasses = {
        'A': 1,
        'Bh': 2,
        'Bk': 3,
        'Ca': 4,
        'Cb': 5,
        'Da': 6,
        'Db': 7,
        'E' : 8    }

     # Load the dataset name for the year and select land-cover band
    dataset = ee.ImageCollection(Collection_name).filterDate(startDate, endDate)
   
    if Collection_name == 'MODIS/061/MCD12Q1':
        # 500m
        MODISImage = dataset.select('LC_Type1').first()
        # koppen = ee.Image('projects/gee-megan/assets/Beck_KG_V1_present_0p0083')
        koppen = ee.Image('projects/gee-megan/assets/Beck_KG_V1_present_0p0083')
    elif Collection_name == 'MODIS/061/MCD12C1':
        # 5000m
        MODISImage = dataset.select('Majority_Land_Cover_Type_1').first()
        koppen = ee.Image('projects/gee-megan/assets/Beck_KG_V1_present_0p083')
    elif Collection_name == 'local':
        # local
        MODISImage = MODIS_Image
        koppen = ee.Image('projects/gee-megan/assets/Beck_KG_V1_present_0p0083').clip(roi)

    # Load Köppen-Geiger climate classification image
    zero_map = MODISImage.multiply(0)
    koppen_reclass = zero_map.where(koppen.gte(1).And(koppen.lte(3)),1)                             # A
    koppen_reclass = koppen_reclass.where(koppen.eq(4).Or(koppen.eq(6)),2)                          # Bh
    koppen_reclass = koppen_reclass.where(koppen.eq(5).Or(koppen.eq(7)),3)                          # Bk
    koppen_reclass = koppen_reclass.where(koppen.eq(8).Or(koppen.eq(11)).Or(koppen.eq(14)),4)       # Ca
    koppen_reclass = koppen_reclass.where(koppen.eq(9).Or(koppen.eq(10)).Or(koppen.eq(12))
                                    .Or(koppen.eq(13)).Or(koppen.eq(15)).Or(koppen.eq(16)),5)       # Cb
    koppen_reclass = koppen_reclass.where(koppen.eq(17).Or(koppen.eq(18)).Or(koppen.eq(21))
                                    .Or(koppen.eq(22)).Or(koppen.eq(25)).Or(koppen.eq(26)),6)       # Da
    koppen_reclass = koppen_reclass.where(koppen.eq(19).Or(koppen.eq(20)).Or(koppen.eq(23))
                                    .Or(koppen.eq(24)).Or(koppen.eq(27)).Or(koppen.eq(28)),7)       # Db
    koppen_reclass = koppen_reclass.where(koppen.eq(29).Or(koppen.eq(30)),8)                        # E



    # convert MODIS x climate combinations into PFT fractions
    def perform_conversion(conversion_list,koppen,modis,modisClasses,koppenClasses):
        result = ee.Image(0)
        for conv in conversion_list:
            band, factor = conv.split('*')
            modiss_key,koppen_key = band.split('_')
            temp  = modis.multiply(0)
            temp = temp.where(modis.eq(modisClasses[modiss_key]).And(koppen.eq(koppenClasses[koppen_key])),1)\
                    .multiply(ee.Number(float(factor)))
                # print(band,factor)
            result = result.add(temp)
        return result
    class_dict = {'C1': ['M1_A*1', 'M1_Bh*1','M1_Ca*1','M1_Cb*1', 'M1_Da*0.5',
                        'M5_A*0.33','M5_Bh*0.33','M5_Ca*0.3','M5_Cb*0.33','M5_Da*0.2',
                        'M8_Bh*0.5','M8_Ca*0.3','M8_Cb*0.4','M8_Da*0.4',
                        'M9_Bh*0.33','M9_Ca*0.3','M9_Cb*0.3','M9_Da*0.3'],
                'C2': ['M1_Da*0.5','M1_Db*1', 'M1_Bk*1','M1_E*1','M2_Bh*0.5','M4_Bk*0.33',
                       'M5_Bk*0.33','M5_Db*0.33','M5_E*0.33',
                        'M8_Bk*0.8','M8_Db*0.8','M8_E*0.8',
                        'M9_Bk*0.208','M9_Db*0.208','M9_E*0.508'],
                'C3': ['M4_Bh*0.33','M4_Bk*0.33','M3_A*1', 'M3_Bh*1','M3_Bk*1','M3_Ca*1','M3_Cb*1', 'M3_Da*1','M3_Db*1', 'M3_E*1',
                        'M5_Bk*0.33','M5_Db*0.33','M5_E*0.33'],
                'C4': ['M2_A*1','M2_Bh*0.5','M2_Ca*0.5',
                        'M5_A*0.33','M5_Ca*0.1',
                        'M8_A*0.4','M8_Ca*0.1',
                        'M9_A*0.4',],
                'C5': ['M2_Bk*1','M2_Ca*0.5', 'M2_Cb*1','M2_Da*1','M2_Db*1','M2_E*1',
                        'M5_Bh*0.33','M5_Ca*0.2','M5_Cb*0.33','M5_Da*0.2',
                        'M8_Ca*0.2','M8_Cb*0.2','M8_Da*0.2',
                        'M9_Bh*0.33','M9_Ca*0.15','M9_Cb*0.15','M9_Da*0.15'],
                'C6': ['M4_A*1','M4_Bh*0.33','M4_Ca*0.5',
                        'M5_A*0.33','M5_Ca*0.1',
                        'M8_A*0.6','M8_Ca*0.1',
                        'M9_A*0.6',],
                'C7': ['M4_Bh*0.33','M4_Ca*0.5', 'M4_Cb*1','M4_Da*1',
                        'M5_Bh*0.33','M5_Ca*0.3','M5_Cb*0.33','M5_Da*0.6',
                        'M8_Bh*0.5','M8_Ca*0.3','M8_Cb*0.4','M8_Da*0.4',
                        'M9_Bh*0.33','M9_Ca*0.3','M9_Cb*0.3','M9_Da*0.3'],
                'C8': ['M4_Bk*0.33','M4_Db*1', 'M4_E*1',
                        'M5_Bk*0.33','M5_Db*0.33','M5_E*0.33',
                        'M8_Bk*0.2','M8_Db*0.2','M8_E*0.2',
                        'M9_Bk*0.132','M9_Db*0.132','M9_E*0.132'],
                'C9': ['M6_A*1',
                        'M7_A*1'],
                'C10':['M6_Bh*0.5','M6_Bk*0.5','M6_Ca*1','M6_Cb*1','M6_Da*1',
                        'M7_Bh*0.5','M7_Bk*0.5','M7_Ca*1','M7_Cb*1','M7_Da*1',],
                'C11':['M6_Bh*0.5','M6_Bk*0.5','M6_Db*1','M6_E*1',
                        'M7_Bh*0.5','M7_Bk*0.5','M7_Db*1','M7_E*1',],
                'C12': ['M10_E*1',
                        'M9_Bk*0.33','M9_Db*0.33','M9_E*0.33'],
                'C13': ['M10_Cb*1','M10_Bk*1','M10_Da*1',"M10_Db*1",
                        'M9_Bk*0.33','M9_Cb*0.25','M9_Da*0.25','M9_Db*0.25'],
                'C14': ['M10_A*1','M10_Bh*1','M10_Ca*1',
                        'M9_Ca*0.25']  ,     
                'C15': ['M12_A*1', 'M12_Bh*1','M12_Bk*1','M12_Ca*1','M12_Cb*1', 'M12_Da*1','M12_Db*1','M12_E*1',
                        'M14_A*1', 'M14_Bh*1','M14_Bk*1','M14_Ca*1','M14_Cb*1', 'M14_Da*1','M14_Db*1','M14_E*1',
                        ],
                'C16': ['M11_A*1', 'M11_Bh*1','M11_Bk*1','M11_Ca*1','M11_Cb*1', 'M11_Da*1','M11_Db*1','M11_E*1',
                        'M13_A*1', 'M13_Bh*1','M13_Bk*1','M13_Ca*1','M13_Cb*1', 'M13_Da*1','M13_Db*1','M13_E*1',
                        'M15_A*1', 'M15_Bh*1','M15_Bk*1','M15_Ca*1','M15_Cb*1', 'M15_Da*1','M15_Db*1','M15_E*1',
                        'M16_A*1', 'M16_Bh*1','M16_Bk*1','M16_Ca*1','M16_Cb*1', 'M16_Da*1','M16_Db*1','M16_E*1',]
    }
    converted_images = {key: perform_conversion(value,koppen_reclass,MODISImage,modisClasses,koppenClasses) for key, value in class_dict.items()}
    
    # Apply conversion for all PFT classes
    for i in converted_images:

        if i == 'C1':
            outputImage = ee.Image.cat([converted_images[i]]).rename(ee.String(i))
        else:
            outputImage = outputImage.addBands(ee.Image(converted_images[i]).rename(ee.String(i)))
    return outputImage
  



def calculateSingleChemImage(chem_name, COMPOUND_EMISSION_FACTORS, PFTS_image):
    """
    --------------------------------------------------------------------------
    Function: calculateSingleChemImage
    --------------------------------------------------------------------------
    Parameters:
        - chem_name (str): the name of the chemical to process
        - COMPOUND_EMISSION_FACTORS (dict-like): mapping of chemical names to emission factor lists
        - PFTS_image (ee.Image): image containing plant functional type bands

    Returns:
        - ee.Image: an image with calculated chemical emissions named by chem_name
            - pixel values represent weighted emissions for the chemical

    Description:
        Calculates a single chemical emission image by multiplying each PFT band by its emission factor
        and summing the results to produce a final image renamed for the chemical.
        Uses Earth Engine iterate to accumulate per-band products.

    Mathematical Details:
        - sum_over_bands: computes sum_i(PFT_band_i * emission_factor_i)

    Example:
        ```python
        result = calculateSingleChemImage('isoprene', emission_factors_dict, pfts_image)
        ```
    """
    # Get the list of emission factors for the given chemical name
    chem_list = ee.List(COMPOUND_EMISSION_FACTORS.get(ee.String(chem_name)))
    
    # Use the provided PFTS_image
    PFT = PFTS_image
    bandNames = PFT.bandNames()
    
    # Initialize an accumulator as an all zero image
    initial = ee.Image(0)

    # Create an iterative function to multiply each band by its corresponding value and then add to the accumulator
    def accumulate(bandName, previous):
        bandName = ee.String(bandName)
        index = bandNames.indexOf(bandName)
  
        factor = ee.Number(chem_list.get(index))

        result = PFT.select(bandName).multiply(factor)
        return ee.Image(previous).add(result)

    # Use the iterate function to create the final image with the initial accumulator image
    final_image = ee.Image(ee.List(bandNames).iterate(accumulate, initial))
    
    # Rename the final image with the chemical name to differentiate images of different chemicals
    final_image = final_image.rename(chem_name)
  
    return ee.Image(final_image)

def add_bands(previous_image, current_image):
    """
    --------------------------------------------------------------------------
    Function: add_bands
    --------------------------------------------------------------------------
    Parameters:
        - previous_image (ee.Image or convertible): base image to which bands will be added
        - current_image (ee.Image): image whose bands are to be appended

    Returns:
        - ee.Image: an image containing all bands from previous_image followed by bands from current_image

    Description:
        Combines two Earth Engine images by appending the bands of current_image onto previous_image.
        Preserves the order of existing bands and adds new bands at the end.

    Example:
        ```python
        combined = add_bands(base_img, additional_bands_img)
        ```
    """
    return ee.Image(previous_image).addBands(current_image)


# The calculateChem function
def calculateChem(chem_names, COMPOUND_EMISSION_FACTORS, PFT):
    """
    --------------------------------------------------------------------------
    Function: calculateChem
    --------------------------------------------------------------------------
    Parameters:
        - chem_names (ee.List): list of chemical names to process
        - COMPOUND_EMISSION_FACTORS (dict-like): mapping of chemical names to emission factor lists
        - PFT (ee.Image): PFTS image containing plant functional type bands

    Returns:
        - ee.List: list of ee.Image objects, each representing emissions for a chemical

    Description:
        Iterates over a list of chemical names, converting each to a string and computing
        its emission image using calculateSingleChemImage, returning a list of resulting images.

    Mathematical Details:
        - emission_mapping: applies calculateSingleChemImage to each chem_name in chem_names

    Example:
        ```python
        result_list = calculateChem(ee.List(['isoprene', 'monoterpene']), emission_factors_dict, pfts_image)
        ```
    """
    def map_func(chem_name):
        chem_images = calculateSingleChemImage(ee.String(chem_name), COMPOUND_EMISSION_FACTORS, PFT)
        return chem_images
    # loop_size = chem_names.size()
    # print(loop_size)
    outputImage = chem_names.map(map_func)

    return outputImage





