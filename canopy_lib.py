# ------------------------------------------------------------------------------
# Module: canopy_emission.py
# ------------------------------------------------------------------------------
# Description:
#     Implements canopy emission calculations following the MEGAN2.1 GAMMA_CE algorithm.
#     Provides functions for:
#       - Solar geometry (Calcbeta)
#       - Orbital eccentricity adjustment (calcEccentricity)
#       - Daily radiation filtering (filterByDate) and rate calculation
#       - Radiation partitioning into beam/diffuse, visible/non-visible components (solarFractions)
#       - Gaussian integration over canopy depth (gaussianIntegration)
#       - Sunlit leaf weighting (weightSLW)
#       - Canopy radiation components per layer (calcExtCoeff, calcRadComponents, CanopyRad)
#       - Leaf energy balance for sunlit/shaded leaves (LeafEB and related helpers)
#       - Final canopy emission factor computation (GAMMA_CE)
#
# Inputs:
#     - Day, Hour, lat: solar geometry parameters
#     - Tc, T24, T240: canopy air temperature fields (K)
#     - PPFD, PPFD24, PPFD240: radiation (mu mol/m^2/s)
#     - Wind, Humidity, Pres, DI: meteorological and drought index inputs
#     - lai_c: current leaf area index 
#     - Layers: number of vertical canopy layers
#     - SPC_NAME, Cantype: species code and canopy type index for parameter lookup
#
# Outputs:
#     - Intermediate ee.ImageCollections and ee.Images for:
#         solar fractions, instantaneous solar rates, canopy radiation components,
#         layer temperatures, heat fluxes, stomatal responses, etc.
#     - Final ee.Image with two bands:
#         'Ea1CanopyFinal' : combined temperature+light emission factor scaled by Cce and LAI (or LAI_v)
#         'EatiCanopySUM'   : cumulative temperature‐only activity factor across canopy
#
# Dependencies:
#     earthengine-api, math, datetime, timedelta, Echo_parameter
# ------------------------------------------------------------------------------


import math
import ee
from datetime import datetime, timedelta
import Echo_parameter as Echo_parameter

# Calcbeta function
def Calcbeta(lat, Day, Hour):
    """
    ------------------------------------------------------------------------------
    Function: Calcbeta
    ------------------------------------------------------------------------------

    Parameters:
        - lat (ee.Image): Latitude in degrees for each pixel/location.
        - Day (ee.Image): Day of year (1–365) as integer image band.
        - Hour (ee.Image): Hour of day (0–23) as integer image band.

    Returns:
        ee.Image: Solar elevation angle β in degrees for each pixel.

    Description:
        Computes the per-pixel solar elevation angle based on latitude, day of year,
        and hour. Uses Earth Engine image operations to calculate solar declination,
        hour angle, and their projections, returning the elevation angle without side effects.

    Mathematical Details:
        - `sinDelta = -sin(0.40907) * cos(((Day + 10) / 365) * 2π)`:  
          Computes the sine of the solar declination δ, shifted by 10 days to approximate seasonal variation.
        - `cosDelta = sqrt(1 - sinDelta**2)`:  
          Derives the cosine of δ via the Pythagorean identity.
        - `A = sin(lat * π/180) * sinDelta`:  
          Projection of latitude sine onto δ sine.
        - `B = cos(lat * π/180) * cosDelta`:  
          Projection of latitude cosine onto δ cosine.
        - `sinBeta = A + B * cos(((Hour - 12) / 24) * 2π)`:  
          Combines declination and hour angle to yield sin(β).
        - `β = arcsin(sinBeta) * (180/π)`:  
          Converts the angle from radians to degrees.

    Example:
        ```python
        # Given an ee.Image 'image' with bands 'latitude', 'doy', and 'hour':
        beta = Calcbeta(
            image.select('latitude'),
            image.select('doy'),
            image.select('hour')
        )
        ```
    """
    Rpi180 = 57.29578
    sinDelta = ee.Image(math.sin(0.40907)).multiply(-1).multiply((((Day.add(10)).divide(365)).multiply(6.28)).cos())
    cosDelta = ee.Image(1).subtract(sinDelta.pow(2)).sqrt()
    A = lat.multiply(math.pi / 180).sin().multiply(sinDelta)
    B = lat.multiply(math.pi / 180).cos().multiply(cosDelta)
    sinBeta = A.add(B.multiply((Hour.subtract(12).divide(24)).multiply(2*math.pi).cos()))
    beta = sinBeta.asin().multiply(180 / math.pi)
    return beta

# calcEccentricity function
def calcEccentricity(Day):
    """
    ------------------------------------------------------------------------------
    Function: calcEccentricity
    ------------------------------------------------------------------------------
    Parameters:
        - Day (ee.Image): Day of year as an Earth Engine Image.

    Returns:
        ee.Image: Orbital eccentricity adjustment factor as an Earth Engine Image.

    Description:
        Computes the Earth's orbital eccentricity factor for a given day of year.
        This factor reflects the variation in Earth–Sun distance throughout the year.

    Mathematical Details:
        - e = 1 + 0.033 * cos(2π * (Day - 10) / 365): computes seasonal deviation due to orbital eccentricity.

    Example:
        ```python
            day_image = ee.Image.constant(100)  # Day-of-year = 100
            ecc_img = calcEccentricity(day_image)
        ```
    """
    return ee.Image(1).add(ee.Image(0.033)\
        .multiply((((Day.subtract(10)).divide(365)).multiply(math.pi).multiply(2)).cos()))

# filterByDate function
def filterByDate(dateStr, collectionId, bandName):
    """
    ------------------------------------------------------------------------------
    Function: filterByDate
    ------------------------------------------------------------------------------
    Parameters:
        - dateStr (str): Date string in 'YYYY-MM-DD' format to filter the collection.
        - collectionId (str): ID of the Earth Engine ImageCollection.
        - bandName (str): Name of the band to select from the images.

    Returns:
        ee.ImageCollection: Filtered ImageCollection containing only the specified band's images for the given date.

    Description:
        Filters an Earth Engine ImageCollection by the given date and selects a single band.
        Creates a range from the start of that day to the start of the next day.

    Mathematical Details:
        - nextDay = inputDate + timedelta(days=1): computes the end of the 24-hour interval.

    Example:
        ```python
            coll = filterByDate("2025-06-15", "xxx", "xx")
        ```
    """
    inputDate = datetime.strptime(dateStr, "%Y-%m-%d")
    nextDay = inputDate + timedelta(days=1)

    filter_dataset = (ee.ImageCollection(collectionId)
                     .filterDate(ee.Date(str(inputDate)), ee.Date(str(nextDay)))
                     .select(bandName))

    return filter_dataset


def calcInstantaneousSolarRadiation(filteredDataset, previousDayDataset):
    """
    ------------------------------------------------------------------------------
    Function: calcInstantaneousSolarRadiation
    ------------------------------------------------------------------------------
    Parameters:
        - filteredDataset (ee.ImageCollection): Images for the current day.
        - previousDayDataset (ee.ImageCollection): Images for the previous day.

    Returns:
        ee.ImageCollection: Instantaneous solar radiation images for each time step.

    Description:
        Calculates the instantaneous solar radiation rate from cumulative solar data.
        Uses the last image of the previous day to compute the first time step.

    Mathematical Details:
        - instantaneousRadiation = (currentImage - previousImage) / 3600: computes rate per second.
        - For index 1, uses currentImage / 3600 as difference from midnight.

    Example:
        ```python
            inst_rad = calcInstantaneousSolarRadiation(todayColl, yesterdayColl)
        ```
    """
    currentDayDatasetList = filteredDataset.toList(filteredDataset.size())
    size = currentDayDatasetList.size()
    previoussize = previousDayDataset.size()
    previousDayDatasetList = previousDayDataset.toList(previoussize)
    previousDayLastImage = ee.Image(previousDayDatasetList.get(previoussize.subtract(1)))
    
    def map_func(index):
        index = ee.Number(index)
        currentImage = ee.Image(currentDayDatasetList.get(index))
        instantaneousRadiation = ee.Algorithms.If(
            index.eq(0),
            currentImage.subtract(previousDayLastImage).divide(3600),
            ee.Algorithms.If(
                index.eq(1),
                currentImage.divide(3600),
                currentImage.subtract(ee.Image(currentDayDatasetList.get(index.subtract(1)))).divide(3600)
            )
        )
        return instantaneousRadiation
    
    instantaneousRadiationList = ee.List.sequence(0, size.subtract(1)).map(map_func)

    instantaneousRadiationCollection = ee.ImageCollection(instantaneousRadiationList)
    return instantaneousRadiationCollection




def solarFractions(Solar, Maxsolar):
    """
    ------------------------------------------------------------------------------
    Function: solarFractions
    ------------------------------------------------------------------------------
    Parameters:
        - Solar (ee.Image): Cumulative solar radiation image for the current period.
        - Maxsolar (ee.Image): Maximum possible solar radiation reference image.

    Returns:
        ee.Image: Four-band image containing:
            - Qdiffv: Diffuse visible radiation .
            - Qbeamv: Beam (direct) visible radiation.
            - Qdiffn: Diffuse non-visible radiation.
            - Qbeamn: Beam (direct) non-visible radiation.

    Description:
        Computes the fractions of visible and non-visible solar radiation into diffuse and beam components.
        If Maxsolar ≤ 0, uses a default transmission of 0.5; caps all fraction values at 1.0.
        Returns a concatenated image with four bands representing each radiation component.

    Mathematical Details:
        - Transmis = Solar / Maxsolar, clipped to [0.5, 1.0]: transmission factor.
        - FracDiff = 0.156 + 0.86 / (1 + exp(11.1 * (Transmis - 0.53))), capped at 1.0: non-visible diffuse fraction.
        - PPFDdifFrac = FracDiff * (1.06 + 0.4 * Transmis), capped at 1.0: visible diffuse fraction.

    Example:
        ```python
            result = solarFractions(Solar_img, Maxsolar_img)
        ```
    """
    Transmis = Solar.divide(Maxsolar)\
                    .where(Maxsolar.lte(0), 0.5)\
                    .where(Maxsolar.lt(Solar), 1.0)
                    
    FracDiff_ = ee.Image(0.156).add(ee.Image(0.86).divide(ee.Image(1).add(((Transmis.subtract(0.53)).multiply(11.1)).exp())))
    FracDiff =  FracDiff_.where(FracDiff_.gt(1.0), 1.0)
    
    PPFDfrac = ee.Image(0.55).subtract(Transmis.multiply(0.12))
    PPFDdifFrac_ = FracDiff.multiply(ee.Image(1.06).add(Transmis.multiply(0.4)))
    PPFDdifFrac =  PPFDdifFrac_.where(PPFDdifFrac_.gt(1.0), 1.0)

    Qv = PPFDfrac.multiply(Solar)
    Qdiffv = Qv.multiply(PPFDdifFrac)
    Qbeamv = Qv.subtract(Qdiffv)
    
    Qn = Solar.subtract(Qv)
    Qdiffn = Qn.multiply(FracDiff)
    Qbeamn = Qn.subtract(Qdiffn)
    
    outputImage = ee.Image.cat([Qdiffv, Qbeamv, Qdiffn, Qbeamn])\
                          .rename(['Qdiffv', 'Qbeamv', 'Qdiffn', 'Qbeamn'])
    return outputImage



def gaussianIntegration(layers):
    """
    ------------------------------------------------------------------------------
    Function: gaussianIntegration
    ------------------------------------------------------------------------------
    Parameters:
        - layers (int): Number of integration layers.

    Returns:
        dict: Dictionary containing:
            - weightgauss (ee.List): Weights for Gaussian integration summation.
            - distgauss (ee.List): Sampling point distances within each layer.
            - distgauss2 (ee.List): Upper boundary distances for each layer.

    Description:
        Generates integration parameters for Gaussian quadrature over the specified number of layers.
        Uses preset nodes and weights for 1, 3, and 5 layers; otherwise falls back to uniform sampling.
        Returns three lists for weights and two sets of distances needed for integration.

    Mathematical Details:
        - Preset cases use standard Gaussian quadrature coefficients.
        - For custom layers:
            • weight = 1 / layers
            • distgauss = (i - 0.5) / layers for i = 1…layers
            • distgauss2 = i / layers for i = 1…layers

    Example:
        ```python
            params = gaussianIntegration(3)
        ```
    """

    if layers == 1:
        weightgauss = ee.List([1])
        distgauss = ee.List([0.5])
        distgauss2 = ee.List([1])
    elif layers == 3:
        weightgauss = ee.List([0.277778, 0.444444, 0.277778])
        distgauss = ee.List([0.112702, 0.5, 0.887298])
        distgauss2 = ee.List([0.277778, 0.722222, 1])
    elif layers == 5:
        weightgauss = ee.List([0.1184635, 0.2393144, 0.284444444, 0.2393144, 0.1184635])
        distgauss = ee.List([0.0469101, 0.2307534, 0.5, 0.7692465, 0.9530899])
        distgauss2 = ee.List([0.1184635, 0.3577778, 0.6422222, 0.881536, 1.0])
    else:
        weightgauss = distgauss = distgauss2 = ee.List([])
        for i in range(1, layers + 1):
            weightgauss = weightgauss.add(1 / layers)
            distgauss = distgauss.add((i - 0.5) / layers)
            distgauss2 = distgauss2.add(i / layers)

    return {
        'weightgauss': weightgauss,
        'distgauss': distgauss,
        'distgauss2': distgauss2
    }


def weightSLW(distgauss, weightgauss, LAI, layers):
    """
    ------------------------------------------------------------------------------
    Function: weightSLW
    ------------------------------------------------------------------------------
    Parameters:
        - distgauss (ee.List): Sampling distances for each canopy layer.
        - weightgauss (ee.List): Gaussian integration weights for each layer.
        - LAI (ee.Image): Leaf area index image.
        - layers (int): Number of integration layers.

    Returns:
        ee.List: List of SLW (sunlit leaf weight) images, one per layer.

    Description:
        Computes the sunlit leaf weight factor for each canopy layer using LAI and distances.
        Applies an empirical formula to estimate sunlit leaf fraction per layer.
        Returns a list of ee.Image objects representing SLW for each layer.

    Mathematical Details:
        - SLW_i = 0.63 + 0.37 * exp(1 - LAI * distgauss_i): empirical sunlit leaf weight formula.

    Example:
        ```python
            slw_list = weightSLW(distgauss, weightgauss, LAI_img, 5)
        ```
    """
    def sequence_function(i):
        distgauss_i = ee.Number(distgauss.get(ee.Number(i).subtract(1)))
        SLW_i = ee.Image(0.63).add(ee.Image(0.37).multiply(ee.Image(-1).multiply(LAI.multiply(distgauss_i).subtract(1)).exp()))
        return SLW_i

    SLW = ee.List.sequence(1, layers).map(sequence_function)

    return SLW



def calcExtCoeff(Qbeam, scat, Kb, Kd):
    """
    --------------------------------------------------------------------------
    Function: calcExtCoeff
    --------------------------------------------------------------------------
    Parameters:
        - Qbeam (ee.Image): Incoming beam radiation image.
        - scat (float): Scattering coefficient (unitless).
        - Kb (ee.Image): Beam extinction coefficient image.
        - Kd (ee.Image): Diffuse extinction coefficient image.

    Returns:
        ee.Image: Four-band image containing:
            - Reflb: Reflectance factor for beam radiation.
            - Kbp: Beam extinction coefficient adjusted by P.
            - Kdp: Diffuse extinction coefficient adjusted by P.
            - QbeamAbsorb: Absorbed beam radiation image.

    Description:
        Computes extinction coefficients and absorbed beam radiation for a canopy layer.
        Uses scattering to adjust extinction and reflectance factors.
        Returns a concatenated image of four components.

    Mathematical Details:
        - P = sqrt(1 - scat): factor for extinction adjustment.
        - Reflb = 1 - exp( -2 * (1 - P)/(1 + P) * Kb / (1 + Kb) ): reflectance formula.
        - Kbp = Kb * P, Kdp = Kd * P: adjusted extinction coefficients.
        - QbeamAbsorb = Qbeam * Kb * (1 - scat): absorbed radiation.

    Example:
        ```python
        result = calcExtCoeff(Qbeam_img, 0.2, Kb_img, Kd_img)
        ```
    """
    # compute P factor from scattering
    P = ee.Number(1).subtract(scat).sqrt()

    # compute reflectance factor Reflb
    Reflb = ee.Image(1).subtract(((ee.Image(-2).multiply((ee.Image(1).subtract(P)).divide((ee.Image(1).add(P))).multiply(Kb)))\
         .divide(ee.Image(1).add(Kb))\
          ).exp())

    # Extinction coefficients

    Kbp = Kb.multiply(P)
    Kdp = Kd.multiply(P)

    # Absorbed beam radiation

    QbeamAbsorb = Kb.multiply(Qbeam).multiply(ee.Image(1).subtract(ee.Number(scat)))

    # Return an ee.Dictionary with the results

    outputImage = ee.Image.cat([Reflb, Kbp, Kdp, QbeamAbsorb]).rename(['Reflb', 'Kbp', 'Kdp', 'QbeamAbsorb'])
    return outputImage



def calcRadComponents(Qdiff, Qbeam, kdp, kbp, kb, scat, refld, reflb, LAIdepth):
    """
    --------------------------------------------------------------------------
    Function: calcRadComponents
    --------------------------------------------------------------------------
    Parameters:
        - Qdiff (ee.Image): Diffuse radiation image.
        - Qbeam (ee.Image): Beam radiation image.
        - kdp (ee.Image): Diffuse extinction coefficient image.
        - kbp (ee.Image): Beam extinction coefficient image.
        - kb (ee.Image): Beam extinction coefficient before adjustment.
        - scat (float): Scattering coefficient.
        - refld (ee.Image): Diffuse reflectance image.
        - reflb (ee.Image): Beam reflectance image.
        - LAIdepth (ee.Image): Leaf area index depth image.
    
    Returns:
        ee.Image: Image with two bands:
            - QdAbs: Absorbed diffuse radiation.
            - QsAbs: Absorbed beam radiation.
    
    Description:
        Calculates absorbed diffuse and beam radiation for each canopy layer.
        Uses extinction and reflectance parameters along with LAI depth.
        Returns a concatenated image with QdAbs and QsAbs bands.
    
    Mathematical Details:
        - QdAbs = Qdiff * kdp * (1 - refld) * exp(-kdp * LAIdepth): computes diffuse absorption.
        - QsAbs = Qbeam * [kbp * (1 - reflb) * exp(-kbp * LAIdepth) - kb * (1 - scat) * exp(-kb * LAIdepth)]: computes beam absorption.
    
    Example:
        ```python
        - example call
            - result = calcRadComponents(Qdiff_img, Qbeam_img, kdp_img, kbp_img, kb_img, scat_val, refld_img, reflb_img, LAIdepth_img)
        ```
    """
    QdAbs = Qdiff.multiply(kdp)\
                   .multiply(ee.Image(1).subtract(refld))\
                   .multiply((kdp.multiply(LAIdepth).multiply(-1)).exp())
    QsAbs = Qbeam.multiply((kbp.multiply(ee.Image(1).subtract(reflb))\
                                  .multiply((kbp.multiply(LAIdepth).multiply(-1)).exp()))\
                                  .subtract(kb.multiply(ee.Image(1).subtract(scat))\
                                            .multiply((kb.multiply(LAIdepth).multiply(-1)).exp())))

    return ee.Image.cat([QdAbs, QsAbs]).rename(['QdAbs', 'QsAbs'])

def CanopyRad(Distgauss, Layers, LAI, Sinbeta, Qbeamv, Qbeamn, Qdiffv, Qdiffn, Cantype, CanopyChar):
    """
    --------------------------------------------------------------------------
    Function: CanopyRad
    --------------------------------------------------------------------------
    Parameters:
        - Distgauss (ee.List): Sampling distances for each canopy layer.
        - Layers (int): Number of canopy layers.
        - LAI (ee.Image): Leaf Area Index image.
        - Sinbeta (ee.Image): Sine of solar zenith angle image.
        - Qbeamv (ee.Image): Visible beam radiation image.
        - Qbeamn (ee.Image): Non-visible beam radiation image.
        - Qdiffv (ee.Image): Visible diffuse radiation image.
        - Qdiffn (ee.Image): Non-visible diffuse radiation image.
        - Cantype (int): Index for canopy type.
        - CanopyChar (list): List of canopy characteristic lists.

    Returns:
        ee.ImageCollection: Collection of canopy radiation component images.
            - Each image has bands:
                'QbAbsV', 'QbAbsn', 'Sunfrac', 'ShadePPFD', 'SunPPFD',
                'QdAbsV', 'QsAbsV', 'QdAbsn', 'QsAbsn', 'ShadeQv',
                'SunQv', 'ShadeQn', 'SunQn'

    Description:
        Computes per-layer canopy radiation absorption and sunlit fractions.
        Applies extinction, scattering, and reflectance to beam and diffuse inputs.
        Filters out results where radiation or LAI are below thresholds.

    Mathematical Details:
        - Kb = 0.5 / Sinbeta * Cluster: beam extinction coefficient.
        - Kd = 0.8 * Cluster: diffuse extinction coefficient.
        - Sunfrac_i = exp(-Kb * LAI * distgauss_i): sunlit fraction.
        - QdAbs = Qdiff * kdp * (1 - refld) * exp(-kdp * LAIdepth_i): diffuse absorption.
        - QsAbs = Qbeam * [kbp * (1 - reflb) * exp(-kbp * LAIdepth_i)
                  - kb * (1 - scat) * exp(-kb * LAIdepth_i)]: beam absorption.

    Example:
        ```python
        - example call
            - result = CanopyRad(Distgauss, Layers, LAI, Sinbeta,
                                 Qbeamv, Qbeamn, Qdiffv, Qdiffn,
                                 Cantype, CanopyChar)
        ```
    """
    # get image geometry for mask application
    imageGeometry = LAI.geometry()

    # Stefan-boltzman constant W m-2 K-4
    Sb = 5.67e-8

    # ConvertShadePPFD and ConvertSunPPFD as constants
    ConvertShadePPFD = 4.6
    ConvertSunPPFD = 4.0
    # Ensure CanopyChar is an ee.List
    CanopyChar = ee.List(CanopyChar)
    # create mask for valid radiation and LAI
    condition = Qbeamv.add(Qdiffv).gt(0.001)\
                .And(Sinbeta.gt(0.00002))\
                .And(LAI.gt(0.001))

    ScatV = ee.Number(ee.List(CanopyChar.get(4)).get(Cantype))
    ScatN = ee.Number(ee.List(CanopyChar.get(5)).get(Cantype))
    RefldV = ee.Number(ee.List(CanopyChar.get(6)).get(Cantype))
    RefldN = ee.Number(ee.List(CanopyChar.get(7)).get(Cantype))
    Cluster = ee.Number(ee.List(CanopyChar.get(8)).get(Cantype))

    Kb = ee.Image(0.5).divide(Sinbeta).multiply(Cluster)
    # Kd = ee.Image(Cluster).multiply(0.8).clip(imageGeometry)
    Kd = ee.Image(Cluster).multiply(0.8)

    
    calcExtCoeff_1 = calcExtCoeff(Qbeamv, ScatV, Kb, Kd)
    ReflbV = calcExtCoeff_1.select('Reflb')
    KbpV = calcExtCoeff_1.select('Kbp')
    KdpV = calcExtCoeff_1.select('Kdp')
    QbAbsV = calcExtCoeff_1.select('QbeamAbsorb')
    calcExtCoeff_2 = calcExtCoeff(Qbeamn, ScatN, Kb, Kd)
    ReflbN = calcExtCoeff_2.select('Reflb')
    KbpN = calcExtCoeff_2.select('Kbp')
    KdpN = calcExtCoeff_2.select('Kdp')
    QbAbsn = calcExtCoeff_2.select('QbeamAbsorb')

    # laiMask = LAI.mask()
    # laiProjection = LAI.projection()

    def function1(index):
        index = ee.Number(index)
        index_i_str = ee.String(index.toInt())
        distgauss_i = ee.Number(Distgauss.get(index))
        LAIdepth_i = LAI.multiply(distgauss_i)
        Sunfrac_i = (LAIdepth_i.multiply(Kb).multiply(-1)).exp()

        calcRadComponents_1 = calcRadComponents(Qdiffv, Qbeamv, KdpV, KbpV, Kb, ScatV, RefldV, ReflbV, LAIdepth_i)
        QdAbsVL = calcRadComponents_1.select('QdAbs')
        QsAbsVL = calcRadComponents_1.select('QsAbs')
        calcRadComponents_2 = calcRadComponents(Qdiffn, Qbeamn, KdpN, KbpN, Kb, ScatN, RefldN, ReflbN, LAIdepth_i)
        QdAbsNL = calcRadComponents_2.select('QdAbs')
        QsAbsNL = calcRadComponents_2.select('QsAbs')

        ShadePPFD = QdAbsVL.add(QsAbsVL).multiply(ConvertShadePPFD).divide(ee.Image(1).subtract(ScatV))
        SunPPFD = ShadePPFD.add(QbAbsV.multiply(ConvertSunPPFD).divide(ee.Image(1).subtract(ScatV)))

        ShadeQv = QdAbsVL.add(QsAbsVL)
        SunQv = ShadeQv.add(QbAbsV)

        ShadeQn = QdAbsNL.add(QsAbsNL)
        SunQn = ShadeQn.add(QbAbsn)

        outputImage = ee.Image.cat([QbAbsV, QbAbsn, Sunfrac_i, ShadePPFD, SunPPFD, QdAbsVL, QsAbsVL, QdAbsNL, QsAbsNL, ShadeQv, SunQv, ShadeQn, SunQn]).rename([
        'QbAbsV', 
        'QbAbsn',
        'Sunfrac',
        'ShadePPFD', 
        'SunPPFD', 
        'QdAbsV', 
        'QsAbsV', 
        'QdAbsn', 
        'QsAbsn', 
        'ShadeQv', 
        'SunQv', 
        'ShadeQn', 
        'SunQn'
        ])

        resultDict = ee.Dictionary({
            'index': index_i_str,
            'outputImage':outputImage
        })

        return resultDict

    results = ee.List.sequence(0, ee.Number(Layers).subtract(1)).map(function1)

    def function2(dict):
        outputImage = ee.Image(ee.Dictionary(dict).get('outputImage'))

        updatedOutputImage = outputImage\
                            .select(['QbAbsV', 'QbAbsn'])\
                            .where(condition.eq(0), 0)\
                            .addBands(outputImage.select('Sunfrac')\
                            .where(condition.eq(0), 0.2))\
                            .addBands(outputImage.select(['SunQn', 'ShadeQn', 'SunQv', 'ShadeQv', 'SunPPFD', 'ShadePPFD', 'QdAbsV', 'QsAbsV', 'QdAbsn', 'QsAbsn'])\
                            .where(condition.eq(0), 0))

        return ee.Dictionary(dict).set('outputImage', updatedOutputImage)

    updatedResults = results.map(function2)

    def function3(dict):
        outputImage = ee.Dictionary(dict).get('outputImage')
        return ee.Image(outputImage)

    updatedImageCollection = ee.ImageCollection.fromImages(updatedResults.map(function3))

    return updatedImageCollection


def updateCanopyRadList(CanopyRad_res, Layers):
    """
    --------------------------------------------------------------------------
    Function: updateCanopyRadList
    --------------------------------------------------------------------------
    Parameters:
        - CanopyRad_res (ee.ImageCollection): CanopyRad results per layer.
        - Layers (int): Number of canopy layers.
    
    Returns:
        ee.Dictionary: Dictionary mapping band names to lists of images.
            - Each key is a band name, value is ee.List of ee.Image for each layer.
    
    Description:
        Converts a canopy radiation ImageCollection into a dictionary of band-specific image lists.
        Each band is extracted per layer from the collection.
        Returns an ee.Dictionary of lists for further analysis.
    
    Mathematical Details:
        - N/A
    
    Example:
        ```python
        - example call
            - result = updateCanopyRadList(CanopyRad_res, Layers)
            - result = ...
        ```
    """
    CanopyRad_res_list = CanopyRad_res.toList(Layers)
    bands = ["QbAbsV", "QbAbsn", "Sunfrac", "SunQn", "ShadeQn", "SunQv", "QbAbsn", "ShadeQv", "SunPPFD", "ShadePPFD", "QdAbsV", "QsAbsV", "QdAbsn", "QsAbsn"]
    results = {}

    for band in bands:
        bandImages = ee.List.sequence(1, Layers).map(lambda i: ee.Image(CanopyRad_res_list.get(ee.Number(i).subtract(1))).select(band))
        results[band] = bandImages

    return ee.Dictionary(results)



def waterVapPres(RH, Pres, waterAirRatio):
    """
    ------------------------------------------------------------------------------
    Function: waterVapPres
    ------------------------------------------------------------------------------

    Parameters:
        - RH (ee.Image): Water vapor mixing ratio (kg/kg) 
        - Pres (ee.Image): Total atmospheric pressure (Pa)
        - waterAirRatio (ee.Number or ee.Image): Molecular weight ratio of water to dry air (≈0.622).

    Returns:
        ee.Image: Partial pressure of water vapor (Pa).
            - Band name: "waterVapPres"

    Description:
        Calculates the water vapor partial pressure from the mixing ratio and total pressure.
        Implements the standard thermodynamic relationship:
            e = (r / (r + ε)) * P

    Mathematical Details:
        - r: water vapor mixing ratio (kg H₂O / kg dry air)
        - ε: molecular weight ratio H₂O / dry air ≈ 18.016/28.97
        - P: total atmospheric pressure (Pa)
        - e = (r / (r + ε)) * P

    Example:
        ```python
        e_img = waterVapPres(RH_img, Pres_img, ε)
        ```
    """
    WaterVapPres = RH.divide(RH.add(waterAirRatio)).multiply(Pres)
    WaterVapPres = WaterVapPres.rename("waterVapPres")
    return WaterVapPres

def stability(CanopyChar, CanType, solar):
    """
    --------------------------------------------------------------------------
    Function: stability
    --------------------------------------------------------------------------
    Parameters:
        - CanopyChar (list): List of canopy characteristic lists.
        - CanType (int): Index specifying canopy type.
        - solar (ee.Image): Solar radiation image.

    Returns:
        ee.Image: Stability factor image named "Stability".
            - Outputs interpolated stability values per pixel.

    Description:
        Computes a stability factor for canopy based on solar radiation thresholds.
        Uses two reference stability values from CanopyChar at indices 11 and 12.
        Returns an ee.Image renamed "Stability".

    Mathematical Details:
        - If solar > trateBoundary: returns Stability_11.
        - If 0 < solar ≤ trateBoundary: returns Stability_11 - ((solar - trateBoundary)/trateBoundary * (Stability_11 - Stability_12)).
        - If solar ≤ 0: returns Stability_12.

    Example:
        ```python
        - example call
            - result = stability(CanopyChar, CanType, solar_img)
        ```
    """
    trateBoundary = 500
    Stability_11 = ee.Number(ee.List(CanopyChar.get(11)).get(CanType))
    Stability_12 = ee.Number(ee.List(CanopyChar.get(12)).get(CanType))
    # compute stability factor based on thresholds
    Stability = (solar.gt(trateBoundary).multiply(Stability_11)
                 .add((solar.gt(0).And(solar.lte(trateBoundary)).multiply(
                     ee.Image(Stability_11).subtract(
                         (solar.subtract(trateBoundary)).divide(trateBoundary).multiply(-1).multiply(
                             Stability_11.subtract(Stability_12))
                     )
                 )))
                 .add(solar.lte(0).multiply(Stability_12)))
    Stability = Stability.rename("Stability")
    return Stability

def deltah(warmHumidityChange, coolHumidityChange, TairK0, Cheight):
    """
    --------------------------------------------------------------------------
    Function: deltah
    --------------------------------------------------------------------------
    Parameters:
        - warmHumidityChange (float or ee.Image): Change rate for warm humidity.
        - coolHumidityChange (float or ee.Image): Change rate for cool humidity.
        - TairK0 (ee.Image): Air temperature in Kelvin.
        - Cheight (ee.Image): Characteristic height image.
    
    Returns:
        ee.Image: Deltah image renamed "Deltah".
            - Represents layer-specific enthalpy change factor.
    
    Description:
        Calculates the enthalpy change factor based on temperature thresholds.
        Uses different humidity change rates for warm (>288K), moderate (278-288K), and cool (<=278K).
        Returns an image of Deltah for each pixel.
    
    Mathematical Details:
        - For TairK0 > 288: Deltah = warmHumidityChangeImg.
        - For 278 < TairK0 ≤ 288: Deltah = [warmHumidityChange - ((TairK0 - 288)/10)*(warmHumidityChangeImg - coolHumidityChangeImg)] / Cheight.
        - For TairK0 ≤ 278: Deltah = coolHumidityChangeImg.
    
    Example:
        ```python
        - example call
            - result = deltah(whc, chc, Tair_img, Cheight_img)
        ```
    """
    warmHumidityChangeImg = ee.Image(warmHumidityChange).divide(Cheight)
    coolHumidityChangeImg = ee.Image(coolHumidityChange).divide(Cheight)
    # compute Deltah based on temperature thresholds
    Deltah = (TairK0.gt(288).multiply(warmHumidityChangeImg)
              .add((TairK0.lte(288).And(TairK0.gt(278)).multiply(
                  ee.Image(warmHumidityChange).subtract(
                      (TairK0.subtract(288).multiply(-1)).divide(10).multiply(
                          warmHumidityChangeImg.subtract(coolHumidityChangeImg))
                  ).divide(Cheight)
              )))
              .add(TairK0.lte(278).multiply(coolHumidityChangeImg)))
    Deltah = Deltah.rename("Deltah")
    return Deltah

def unexposedLeafIRin(Tk, Eps):
    """
    --------------------------------------------------------------------------
    Function: unexposedLeafIRin
    --------------------------------------------------------------------------
    Parameters:
        - Tk (ee.Image): Leaf temperature as Earth Engine Image.
        - Eps (float or ee.Number): Emissivity of the leaf surface.
    
    Returns:
        ee.Image: Unexposed leaf incoming infrared radiation.
            - Represents IR emission per unit area.
    
    Description:
        Calculates the infrared emission from unexposed leaf surfaces using the Stefan-Boltzmann law.
        Uses leaf temperature and emissivity to compute the outgoing infrared flux.
        Returns an Earth Engine Image of IR radiation.
    
    Mathematical Details:
        - IR = Eps * Sb * Tk^4: Stefan-Boltzmann law for blackbody radiation.
    
    Example:
        ```python
        - example call
            - result = unexposedLeafIRin(Tk_img, 0.98)
        ```
    """
    # Stefan-Boltzmann constant (W m^-2 K^-4)
    Sb = 0.0000000567
    Eps = ee.Number(Eps)
    UnexposedLeafIRin = Tk.pow(4).multiply(Eps).multiply(Sb)
    return UnexposedLeafIRin

def exposedLeafIRin(HumidPa, Tk):
    """
    --------------------------------------------------------------------------
    Function: exposedLeafIRin
    --------------------------------------------------------------------------
    Parameters:
        - HumidPa (ee.Image): Water vapor partial pressure image.
        - Tk (ee.Image): Leaf temperature as an Earth Engine Image.

    Returns:
        ee.Image: Incoming infrared radiation for exposed leaf surfaces.
            - Computed in W m^-2 using atmospheric emissivity and Stefan-Boltzmann law.

    Description:
        Calculates incoming infrared radiation for leaves exposed to the atmosphere.
        Uses humidity and temperature to estimate atmospheric emissivity.
        Applies the Stefan-Boltzmann law to compute IR flux.

    Mathematical Details:
        - EmissAtm = (HumidPa / Tk)^(1/7) * 0.642: empirical atmospheric emissivity formula.
        - IR = EmissAtm * Sb * Tk^4: Stefan-Boltzmann law for emission.

    Example:
        ```python
        - example call
            - result = exposedLeafIRin(HumidPa_img, Tk_img)
        ```
    """
    Sb = 0.0000000567
    EmissAtm = HumidPa.divide(Tk).pow(1.0/7.0).multiply(0.642)
    ExposedLeafIRin = EmissAtm.multiply(Sb).multiply(Tk.pow(4))
    return ExposedLeafIRin

def convertHumidityPa2kgm3(Pa, Tk):
    """
    --------------------------------------------------------------------------
    Function: convertHumidityPa2kgm3
    --------------------------------------------------------------------------
    Parameters:
        - Pa (ee.Image): Water vapor partial pressure image in Pascals.
        - Tk (ee.Image): Air temperature image in Kelvin.
    
    Returns:
        ee.Image: Absolute humidity image in kg/m3.
            - Calculated from partial pressure and temperature.
    
    Description:
        Computes the water vapor density from its partial pressure.
        Applies the ideal gas constant for water vapor.
        Returns an Earth Engine Image of absolute humidity.
    
    Mathematical Details:
        - humidity = Pa * 0.002165 / Tk: applies gas constant 0.002165 (kPa·m³·kg⁻¹·K⁻¹).
    
    Example:
        ```python
        - example call
            - result = convertHumidityPa2kgm3(Pa_img, Tk_img)
        ```
    """

    ConvertHumidityPa2kgm3 = Pa.multiply(0.002165).divide(Tk)
    return ConvertHumidityPa2kgm3



def DIstomata(DI):
    """
    --------------------------------------------------------------------------
    Function: DIstomata
    --------------------------------------------------------------------------
    Parameters:
        - DI (ee.Image): Drought index image.
    
    Returns:
        ee.Image: Stomatal response factor image.
            - Factor in [0,1] representing stomatal conductance adjustment.
    
    Description:
        Computes the stomatal conductance adjustment based on drought index thresholds.
    
    Mathematical Details:
        - For DI > DIhigh: response = 1.
        - For DIlow < DI ≤ DIhigh: response = 1 - 0.9 * (DI - DIhigh)/(DIlow - DIhigh).
        - For DI ≤ DIlow: response = 0.
    
    Example:
        ```python
        - example call
            - result = DIstomata(DI_img)
        ```
    """
    DIhigh = ee.Image(-0.5)
    DIlow = ee.Image(-5)
    # compute stomatal response based on DI thresholds
    dIstomata = DI.gt(DIhigh).multiply(ee.Image(1)).add(
        DI.gt(DIlow).And(DI.lte(DIhigh)).multiply(
            ee.Image(1).subtract(ee.Image(0.9).multiply((DI.subtract(DIhigh)).divide(DIlow.subtract(DIhigh))))
        )
    ).add(DI.lte(DIlow).multiply(ee.Image(0)))

    return dIstomata


def ResSC(Par, StomataDI):
    """
    --------------------------------------------------------------------------
    Function: ResSC
    --------------------------------------------------------------------------
    Parameters:
        - Par (ee.Image): Photosynthetic photon flux density (PPFD) image.
        - StomataDI (ee.Image): Stomatal response factor image (drought index).
    
    Returns:
        ee.Image: Stomatal resistance in s/m.
            - High default resistance (2000) when conductance adjustment is low.
            - Otherwise resistance = 200 / SCadj.
    
    Description:
        Computes stomatal resistance using PPFD and a stomatal drought index.
    
    Mathematical Details:
        - SCadj = StomataDI * [0.0027 * 1.066 * Par / sqrt(1 + (0.0027^2 * Par^2))]: adjustment factor.
        - resSC = 2000 if SCadj < 0.1 else 200 / SCadj: piecewise resistance formula.
    
    Example:
        ```python
        - example call
            - result = ResSC(PPFD_img, StomataDI_img)
        ```
    """
    SCadj = ee.Image(StomataDI).multiply((ee.Image(0.0027).multiply(ee.Image(1.066)).multiply(Par)).divide(
        (ee.Image(1).add(ee.Image(0.0027).multiply(ee.Image(0.0027)).multiply(Par.pow(2)))).sqrt()))

    resSC = SCadj.where(SCadj.lt(0.1), ee.Image(2000)).where(SCadj.gte(0.1), ee.Image(200).divide(SCadj))

    return resSC


def LeafIROut(Tleaf, Eps):
    """
    --------------------------------------------------------------------------
    Function: LeafIROut
    --------------------------------------------------------------------------
    Parameters:
        - Tleaf (ee.Image): Leaf temperature in Kelvin as an Earth Engine Image.
        - Eps (float or ee.Image or ee.Number): Leaf emissivity factor (0–1).

    Returns:
        ee.Image: Outgoing leaf infrared radiation image.
            - Band values represent IR flux in W·m⁻².

    Description:
        Calculates outgoing infrared radiation from leaves using the Stefan–Boltzmann law.

    Mathematical Details:
        - IRout = 2 * Sb * Eps * Tleaf⁴: computes leaf IR emission flux.

    Example:
        ```python
        - example call
            - result = LeafIROut(Tleaf_img, ee.Number(0.98))
        ```
    """
    Sb = ee.Image.constant(0.0000000567)
    leafIROut = Tleaf.pow(4).multiply(2).multiply(Sb).multiply(Eps)

    return leafIROut


def LHV(Tk):
    """
    --------------------------------------------------------------------------
    Function: LHV
    --------------------------------------------------------------------------
    Parameters:
        - Tk (ee.Image or ee.Number): Temperature in Kelvin.
    
    Returns:
        ee.Image or ee.Number: Latent heat of vaporization.
            - lHV: latent heat value for each pixel or scalar.
    
    Description:
        Calculates latent heat of vaporization from air temperature in Kelvin.
        Converts temperature from Kelvin to Celsius and applies a linear equation.
        Outputs the latent heat in joules per kilogram.
    
    Mathematical Details:
        - lHV = -2370 * (Tk - 273) + 2501000: linear estimate of latent heat of vaporization.
    
    Example:
        ```python
        - example call
            - result = LHV(Tk_img)
        ```
    """
    lHV = Tk.subtract(273).multiply(-2370).add(2501000)

    return lHV


def LeafLE(Tleaf, Ambvap, LatHv, GH, StomRes, TranspireType):
    """
    --------------------------------------------------------------------------
    Function: LeafLE
    --------------------------------------------------------------------------
    Parameters:
        - Tleaf (ee.Image): Leaf temperature in Kelvin.
        - Ambvap (ee.Image): Atmospheric vapor density.
        - LatHv (ee.Image): Latent heat of vaporization.
        - GH (ee.Image): Boundary layer conductance image.
        - StomRes (ee.Image): Stomatal resistance.
        - TranspireType (ee.Image): Transpiration type indicator.

    Returns:
        ee.Image: Latent heat flux image.
            - Represents energy loss due to transpiration per unit leaf area.

    Description:
        Calculates latent heat flux from leaves using resistance terms and vapor pressure gradient.
        Uses leaf and air temperatures, humidity, and conductance to estimate evaporative cooling.


    Mathematical Details:
        - LeafRes = 1/(1.075*(GH/1231)) + StomRes: total resistance combining boundary layer and stomatal.
        - Svp = 10^(-2937.4/Tleaf - 4.9283*log10(Tleaf) + 23.5518).
        - SvdTk = 0.2165 * Svp / Tleaf.
        - Vapdeficit = SvdTk - Ambvap: vapor pressure deficit (Pa).
        - LE = TranspireType/LeafRes * LatHv * Vapdeficit.

    Example:
        ```python
        - example call
            - result = LeafLE(Tleaf_img, Ambvap_img, LatHv_img, GH_img, StomRes_img, TranspireType_img)
        ```
    """
    LeafRes = (ee.Image(1).divide(ee.Image(1.075).multiply(GH.divide(1231)))).add(StomRes)
    Svp = ee.Image(10).pow(
        ee.Image(-2937.4).divide(Tleaf).subtract(Tleaf.log10().multiply(4.9283)).add(23.5518))
    SvdTk = ee.Image(0.2165).multiply(Svp).divide(Tleaf)
    Vapdeficit = SvdTk.subtract(Ambvap)

    LE = ee.Image(TranspireType).divide(LeafRes).multiply(LatHv).multiply(Vapdeficit)
    leafLE = LE.where(LE.lt(0), 0).where(LE.gte(0), LE)

    return leafLE


def LeafBLC(GHforced, Tdelta, Llength):
    """
    --------------------------------------------------------------------------
    Function: LeafBLC
    --------------------------------------------------------------------------
    Parameters:
        - GHforced (ee.Image): Forced convective conductance image.
        - Tdelta (ee.Image): Leaf-air temperature difference image in Kelvin.
        - Llength (ee.Image): Leaf length.

    Returns:
        ee.Image: Total boundary layer conductance image.
            - Sum of forced and free convection conductance.

    Description:
        Combines forced and free convection to compute boundary layer conductance.

    Mathematical Details:
        - GhFree = 0.5 * 0.00253 * ((1.6e8 * (Tdelta / Llength^3))^0.25 / Llength): free convection conductance.
        - leafBLC = GHforced + GhFree: total conductance.

    Example:
        ```python
        - example call
            - result = LeafBLC(GHforced_img, Tdelta_img, Llength_img)
        ```
    """
    Tdelta = Tdelta.where(Tdelta.lte(0),0)
    # Tdelta_masked = Tdelta.updateMask(Tdelta_lt_0)
    GhFree = ee.Image(0.5).multiply(0.00253).multiply(
        (ee.Image(160000000).multiply(ee.Image(Tdelta).divide(ee.Image(Llength).pow(3)))).pow(0.25)
        .divide(ee.Image(Llength)))
    GhFree.where(ee.Image(Tdelta).lt(0), 0)

    leafBLC = GHforced.add(GhFree)

    return leafBLC


def LeafH(Tdelta, GH):
    """
    --------------------------------------------------------------------------
    Function: LeafH
    --------------------------------------------------------------------------
    Parameters:
        - Tdelta (ee.Image): Leaf-air temperature difference image in Kelvin.
        - GH (ee.Image): Boundary layer conductance imag.
    
    Returns:
        ee.Image: Sensible heat flux image.
            - Computed as GH * Tdelta * 2 representing two-sided heat transfer.
    
    Description:
        Calculates sensible heat flux for leaves using temperature difference and conductance.
    
    Mathematical Details:
        - leafH = 2 * GH * Tdelta: computes two-sided sensible heat flux per unit area.
    
    Example:
        ```python
        - example call
            - result = LeafH(Tdelta_img, GH_img)
        ```
    """
    leafH = GH.multiply(Tdelta).multiply(2)
    return leafH


def LeafEB(PPFD, SunQv, SunQn, IRin, Eps, TranspireType, Lwidth, Llength, TairK, HumidairPa, Ws, StomataDI):
    """
    --------------------------------------------------------------------------
    Function: LeafEB
    --------------------------------------------------------------------------
    Parameters:
        - PPFD (ee.Image): Photosynthetic photon flux density image.
        - SunQv (ee.Image): Sunlit leaf visible radiation image.
        - SunQn (ee.Image): Sunlit leaf non-visible radiation image.
        - IRin (ee.Image): Incoming infrared radiation image.
        - Eps (float or ee.Number): Leaf emissivity.
        - TranspireType (ee.Image): Transpiration type indicator image.
        - Lwidth (ee.Image): Leaf width image in meters.
        - Llength (ee.Image): Leaf length image in meters.
        - TairK (ee.Image): Air temperature image in Kelvin.
        - HumidairPa (ee.Image): Water vapor partial pressure image.
        - Ws (ee.Image): Wind speed image in m/s.
        - StomataDI (ee.Image): Stomatal drought index image.

    Returns:
        - ee.Image: Energy balance output image with bands:
            - IRout: Outgoing infrared radiation.
            - Tleaf: Computed leaf temperature in Kelvin.
            - SensibleHeat: Sensible heat flux.
            - LatentHeat: Latent heat flux.
            - finalTdelt: Final leaf-air temperature difference.
            - finalBalance: Final energy balance residual.
            - E1: Net available energy.

    Description:
        Calculates leaf energy balance by combining radiation, convection, and transpiration terms.


    Mathematical Details:
        - Q = SunQv + SunQn: total absorbed shortwave radiation.
        - E1 = Q + IRin - IRoutairT - LHairT: net available energy.
        - Tleaf = TairK + finalTdelt: leaf temperature after balance.
    
    Example:
        ```python
        - example call
            - result = LeafEB(PPFD_img, SunQv_img, SunQn_img, IRin_img,
                              0.98, transpireType_img, Lwidth_img,
                              Llength_img, TairK_img, HumidairPa_img,
                              Ws_img, StomataDI_img)
        ```
    """
    Ws = Ws.where(Ws.lte(0), 0.001)
    Q = SunQv.add(SunQn)
    # Eps  TranspireType Lwidth Llength  to ee.Number() format.
    # Eps = ee.Image(Eps)
    TranspireType = ee.Image(TranspireType)
    Lwidth = ee.Image(Lwidth)
    Llength = ee.Image(Llength)
    HumidAirKgm3 = convertHumidityPa2kgm3(HumidairPa, TairK)
    GHforced = ee.Image(0.0259).divide(
        ee.Image(0.004).multiply((Llength.divide(Ws)).sqrt())
    )
    StomRes = ResSC(PPFD, StomataDI)

    IRoutairT = LeafIROut(TairK, Eps)
    LatHv = LHV(TairK)
    LHairT = LeafLE(TairK, HumidAirKgm3, LatHv, GHforced, StomRes, TranspireType)
    E1 = Q.add(IRin).subtract(IRoutairT).subtract(LHairT)
    E1 = E1.where(E1.eq(0), -1)

    Tdelt = ee.Image(1)
    Balance = ee.Image(10)

    def Balance_iterationFunction(i, inputs):
        inputs = ee.Dictionary(inputs)

        Tdelt = ee.Image(inputs.get('Tdelt'))
        Balance = ee.Image(inputs.get('Balance'))
        Tdelt = Tdelt.where(Tdelt.eq(0), 1)
        GH1 = LeafBLC(GHforced, Tdelt, Llength)
        SH1 = LeafH(Tdelt, GH1)
        TK_delt = TairK.add(Tdelt)
        LatHv = LHV(TK_delt)
        LH = LeafLE(TK_delt, HumidAirKgm3, LatHv, GH1, StomRes, TranspireType)
        LH1 = LH.subtract(LHairT)
        IRout = LeafIROut(TK_delt, Eps)
        IRout1 = IRout.subtract(IRoutairT)
        TdeltNew = E1.divide((SH1.add(LH1).add(IRout1)).divide(Tdelt))
        BalanceNew = Q.add(IRin).subtract(IRout).subtract(SH1).subtract(LH)
        Tdelt = Tdelt.where(Balance.abs().gt(2), TdeltNew)
        Balance = Balance.where(Balance.abs().gt(2), BalanceNew)

        return ee.Dictionary({
            'Tdelt': Tdelt,
            'Balance': Balance
        })

    initialInputs = ee.Dictionary({
        'Tdelt': Tdelt,
        'Balance': Balance
    })
    sequence = ee.List.sequence(1, 10)
    finalOutputs = sequence.iterate(Balance_iterationFunction, initialInputs)

    finalTdelt = ee.Image(ee.Dictionary(finalOutputs).get('Tdelt'))
    finalBalance = ee.Image(ee.Dictionary(finalOutputs).get('Balance'))
    finalTdelt = finalTdelt.where(finalTdelt.gt(10), 10).where(finalTdelt.lt(-10), -10)
    Tleaf = TairK.add(finalTdelt)
    GH = LeafBLC(GHforced, finalTdelt, Llength)
    SH = LeafH(finalTdelt, GH)
    LatHvNew = LHV(Tleaf)
    LH = LeafLE(Tleaf, HumidAirKgm3, LatHvNew, GH, StomRes, TranspireType)
    IRout = LeafIROut(Tleaf, Eps)
    OutputImage = ee.Image.cat([IRout, Tleaf, SH, LH, finalTdelt, finalBalance, E1]).rename(
        ["IRout", "Tleaf", "SensibleHeat", "LatentHeat", "finalTdelt", "finalBalance", "E1"]
    )

    return OutputImage


def updateSunShadeLeafIRList(SunShadeLeafIR, Layers):
    """
    --------------------------------------------------------------------------
    Function: updateSunShadeLeafIRList
    --------------------------------------------------------------------------
    Parameters:
        - SunShadeLeafIR (ee.ImageCollection): Collection of sunlit/shaded leaf energy balance images.
        - Layers (int): Number of canopy layers / images in the collection.

    Returns:
        ee.Dictionary: Dictionary mapping each band name to an ee.List of ee.Image objects.
            - Each key is a band name, the value is a list of that band's image for each layer.

    Description:
        Converts a canopy sunlit/shaded leaf energy balance image collection into a dictionary.
        Extracts each specified band across all layers into separate lists for downstream analysis.
        Returns an ee.Dictionary of image lists keyed by band names.

    Mathematical Details:
        - N/A

    Example:
        ```python
        - example call
            - result = updateSunShadeLeafIRList(SunShadeLeafIR, Layers)
        ```
    """
    SunShadeLeafIR_list = SunShadeLeafIR.toList(Layers)
    bands = ['SunLeafTk', 
             'SunLeafSensibleHeat', 
             'SunLeafLatentHeat', 
             'SunLeafIR', 
             'ShadeLeafTk', 
             'ShadeLeafSensibleHeat', 
             'ShadeLeafLatentHeat', 
             'ShadeLeafIR', 
             'Ws', 
             'HumidairPa', 
             'TairK',
             'SunfinalTdelt',
             'SunfinalBalance',
             'SunE1']
    results = {}
    for band in bands:
        bandImages = ee.List.sequence(1, Layers).map(lambda i: ee.Image(SunShadeLeafIR_list.get(ee.Number(i).subtract(1))).select(band))
        results[band] = bandImages
    return ee.Dictionary(results)

def CanopyEB(Trate,Layers,Distgauss,CanopyChar,
              Cantype,StomataDI,TairK0,HumidairPa0,Ws0,
              SunPPFD,ShadePPFD,SunQv,ShadeQv,SunQn,ShadeQn):
    """
    --------------------------------------------------------------------------
    Function: CanopyEB
    --------------------------------------------------------------------------
    Parameters:
        - Trate (ee.Number): Temperature increase rate per unit canopy depth.
        - Layers (int): Number of canopy layers.
        - Distgauss (ee.List): Fractional depths for each integration layer.
        - CanopyChar (list): Lists of canopy characteristic parameters.
        - Cantype (int): Index specifying the canopy type.
        - StomataDI (ee.Image): Stomatal drought index image.
        - TairK0 (ee.Image): Base air temperature in Kelvin.
        - HumidairPa0 (ee.Image): Base water vapor partial pressure.
        - Ws0 (ee.Image): Base wind speed in m/s.
        - SunPPFD (ee.List): Sunlit PPFD for each layer.
        - ShadePPFD (ee.List): Shaded PPFD for each layer.
        - SunQv (ee.List): Sunlit visible radiation per layer.
        - ShadeQv (ee.List): Shaded visible radiation per layer.
        - SunQn (ee.List): Sunlit non-visible radiation per layer.
        - ShadeQn (ee.List): Shaded non-visible radiation per layer.

    Returns:
        - ee.Dictionary: Dictionary mapping variable names to ee.List of ee.Image per layer.
            - Keys include:
              'SunLeafTk', 'SunLeafSensibleHeat', 'SunLeafLatentHeat', 'SunLeafIR',
              'ShadeLeafTk', 'ShadeLeafSensibleHeat', 'ShadeLeafLatentHeat', 'ShadeLeafIR',
              'Ws', 'HumidairPa', 'TairK', 'SunfinalTdelt', 'SunfinalBalance', 'SunE1'.

    Description:
        Performs energy balance calculations through canopy layers.
        Computes temperature, heat flux, and IR flux for sunlit and shaded leaves.
        Returns structured lists of images for each output variable.

    Mathematical Details:
        - Ldepth = Distgauss * Cdepth: physical depth of each layer.
        - TairK[layer] = TairK0 + Trate * Ldepth[layer]: air temperature profile.
        - HumidairPa[layer] = HumidairPa0 + Deltah * Ldepth[layer]: humidity profile.
        - Iterative solve for leaf temperature difference using energy balance equation.

    Example:
        ```python
        - example call
            - result = CanopyEB(Trate, Layers, Distgauss, CanopyChar,
                                Cantype, StomataDI, TairK0, HumidairPa0, Ws0,
                                SunPPFD, ShadePPFD, SunQv, ShadeQv, SunQn, ShadeQn)
        ```
    """
    Cdepth = ee.Number(ee.List(CanopyChar.get(0)).get(Cantype))
    Lwidth = ee.Number(ee.List(CanopyChar.get(1)).get(Cantype))
    Llength = ee.Number(ee.List(CanopyChar.get(2)).get(Cantype))
    Cheight = ee.Number(ee.List(CanopyChar.get(3)).get(Cantype))
    Eps = ee.Number(ee.List(CanopyChar.get(9)).get(Cantype))
    TranspireType = ee.Number(ee.List(CanopyChar.get(10)).get(Cantype))
    warmHumidityChange = ee.Number(ee.List(CanopyChar.get(13)).get(Cantype))
    coolHumidityChange = ee.Number(ee.List(CanopyChar.get(14)).get(Cantype))
    canopyNormalizedWind = ee.Number(ee.List(CanopyChar.get(15)).get(Cantype))
  
    Deltah = deltah(warmHumidityChange, coolHumidityChange, TairK0, Cheight)
    # compute physical layer depths
    def compute_Ldepth(num):
        return ee.Number(num).multiply(Cdepth)

    Ldepth = Distgauss.map(compute_Ldepth)
    
    # print(Ldepth)
    # compute air temperature for each layer
    def compute_TairK(i):
        Ldepth_i = ee.Number(Ldepth.get(ee.Number(i).subtract(1)))
        return TairK0.add(Trate.multiply(Ldepth_i))

    TairK = ee.List.sequence(1, Layers).map(compute_TairK)

    # compute humidity for each layer
    def compute_HumidairPa(i):
        Ldepth_i = ee.Number(Ldepth.get(ee.Number(i).subtract(1)))
        return HumidairPa0.add(Deltah.multiply(Ldepth_i))

    HumidairPa = ee.List.sequence(1, Layers).map(compute_HumidairPa)

    # compute wind shear height for each layer
    def compute_Wsh(num):
        return Cheight.subtract(ee.Number(num)).subtract(canopyNormalizedWind.multiply(Cheight))

    Wsh = Ldepth.map(compute_Wsh)  
    
    # compute layer-specific wind speed
    def Ws_compute(i):
        Wsh_i = ee.Number(Wsh.get(ee.Number(i).subtract(1)))
        Wsh_i_new = ee.Image(Wsh_i).where(Wsh_i.lte(0),0.00001)
        Ws_i = (Ws0.multiply(Wsh_i_new.log())).divide((Cheight.subtract(canopyNormalizedWind.multiply(Cheight))).log())
        Ws_i = Ws_i.where(ee.Image(Wsh_i).lt(0.001),0.05)
        return Ws_i
    Ws = ee.List.sequence(1, Layers).map(Ws_compute)
    
    # compute energy balance for sunlit and shaded leaves per layer
    def SunShadeLeafIR_compute(i):
        i = ee.Number(i)
        TairK_i = ee.Image(TairK.get(i))
        IRin_i = unexposedLeafIRin(TairK_i,Eps)
        ShadeleafIR_i = IRin_i.multiply(2.0)
        exposedLeafIRin_i = exposedLeafIRin(HumidairPa0,TairK0)
        SunleafIR_i = ee.Image(exposedLeafIRin_i).multiply(0.5).add(IRin_i.multiply(1.5))
      
        SunPPFD_i = ee.Image(SunPPFD.get(i))
        SunQv_i = ee.Image(SunQv.get(i))
        SunQn_i = ee.Image(SunQn.get(i))
        Q_i = SunQv_i.add(SunQn_i)
        SunIRin_i = SunleafIR_i
        HumidairPa_i = ee.Image(HumidairPa.get(i))
        Ws_i = ee.Image(Ws.get(i))

        SunLeafEBout = LeafEB(SunPPFD_i,SunQv_i,SunQn_i,SunIRin_i,Eps,TranspireType
                            ,Lwidth,Llength,TairK_i,HumidairPa_i,Ws_i,StomataDI)
        SunleafIR_i = SunleafIR_i.subtract(SunLeafEBout.select("IRout"))
      
        ShadePPFD_i = ee.Image(ShadePPFD.get(i))
        ShadeQv_i = ee.Image(ShadeQv.get(i))
        ShadeQn_i = ee.Image(ShadeQn.get(i))
        SahdeIRin_i = ShadeleafIR_i
        ShadeLeafEBout = LeafEB(ShadePPFD_i,ShadeQv_i,ShadeQn_i,SahdeIRin_i,Eps,TranspireType
                            ,Lwidth,Llength,TairK_i,HumidairPa_i,Ws_i,StomataDI)
        ShadeleafIR_i = ShadeleafIR_i.subtract(ShadeLeafEBout.select("IRout"))

        outputImage = ee.Image.cat([SunLeafEBout.select("Tleaf"),
                                    SunLeafEBout.select("SensibleHeat"),
                                    SunLeafEBout.select("LatentHeat"),
                                    SunleafIR_i,
                                    ShadeLeafEBout.select("Tleaf"),
                                    ShadeLeafEBout.select("SensibleHeat"),
                                    ShadeLeafEBout.select("LatentHeat"),
                                    ShadeleafIR_i,
                                    Ws_i,HumidairPa_i,TairK_i,
                                    SunLeafEBout.select("finalTdelt"),
                                    SunLeafEBout.select("finalBalance"),
                                    SahdeIRin_i]).rename([
                                  ee.String('SunLeafTk'),
                                  ee.String('SunLeafSensibleHeat'),
                                  ee.String('SunLeafLatentHeat'),
                                  ee.String('SunLeafIR'),
                                  ee.String('ShadeLeafTk'),
                                  ee.String('ShadeLeafSensibleHeat'),
                                  ee.String('ShadeLeafLatentHeat'),
                                  ee.String('ShadeLeafIR'),
                                  ee.String('Ws'),
                                  ee.String('HumidairPa'),
                                  ee.String('TairK'),
                                  ee.String('SunfinalTdelt'),
                                  ee.String('SunfinalBalance'),
                                  ee.String('SunE1')])
        return outputImage
    
    SunShadeLeafIR = ee.List.sequence(0, ee.Number(Layers).subtract(1)).map(SunShadeLeafIR_compute)
  
    updatedImageCollection = ee.ImageCollection(SunShadeLeafIR.map(lambda img: img))
  
    SunShadeLeafIRList = updateSunShadeLeafIRList(updatedImageCollection,Layers)
  
    return SunShadeLeafIRList


def Ea1t99(T1, T24, T240, CONST_CANOPY, SPC_NAME):
    """
    --------------------------------------------------------------------------
    Function: Ea1t99
    --------------------------------------------------------------------------
    Parameters:
        - T1 (ee.Image): Target temperature image in Kelvin.
        - T24 (ee.Image): 24-hour averaged temperature image in Kelvin.
        - T240 (ee.Image): 240-hour averaged temperature image in Kelvin.
        - CONST_CANOPY (dict): Dictionary containing canopy constants lists.
        - SPC_NAME (string): Species name key to select constants.

    Returns:
        ee.Image: gamma_t.

    Description:
        Calculates gamma_t (temperature activity factor)

    Mathematical Details:
        - Topt = 312.5 + 0.6*(T240 - 297): optimal temperature.
        - X = (1/Topt - 1/T1)/0.00831
        - Eopt = CLeo * exp(0.05*(T24 - 297)) * exp(0.05*(T240 - 297))
        - Ea1t99New = Eopt * [Ctm2 * exp(X*Ctm1) / (Ctm2 - Ctm1*(1 - exp(X*Ctm2)))].

    Example:
        ```python
        - example call
            - result = Ea1t99(T1_img, T24_img, T240_img, CONST_CANOPY, "speciesA")
        ```
    """
    Ctm2 = ee.Image(230)
    index = ee.List(CONST_CANOPY.get('MGN_SPC')).indexOf(SPC_NAME)
    CLeo = ee.Number(ee.List(CONST_CANOPY.get('CLeo')).get(index))
    Ctm1 = ee.Number(ee.List(CONST_CANOPY.get('Ctm1')).get(index))
    Ea1t99 = T1.multiply(ee.Image(0))
    Topt = ee.Image(312.5).add((T240.subtract(ee.Image(297))).multiply(ee.Image(0.6)))
    X = ((ee.Image(1).divide(Topt)).subtract(ee.Image(1).divide(T1))).divide(ee.Image(0.00831))
    Eopt = ee.Image(CLeo).multiply(((T24.subtract(ee.Image(297))).multiply(ee.Image(0.05))).exp())\
                                   .multiply(((T240.subtract(ee.Image(297))).multiply(ee.Image(0.05))).exp())
    Ea1t99New = Eopt.multiply(ee.Image(Ctm2).multiply((X.multiply(ee.Image(Ctm1))).exp())\
                    .divide(ee.Image(Ctm2)\
                            .subtract(ee.Image(Ctm1).multiply(ee.Image(1).subtract((X.multiply(ee.Image(Ctm2))).exp()))\
                            )))
    Ea1t99 = Ea1t99.where(T1.gte(260), Ea1t99New)
    return Ea1t99


def Ea1p99(PPFD1, PPFD24, PPFD240, PSTD):
    """
    --------------------------------------------------------------------------
    Function: Ea1p99
    --------------------------------------------------------------------------
    Parameters:
        - PPFD1 (ee.Image): Instantaneous photosynthetic photon flux density.
        - PPFD24 (ee.Image): 24-hour average PPFD.
        - PPFD240 (ee.Image): 240-hour average PPFD.
        - PSTD (float): Species-specific PPFD threshold parameter.
    
    Returns:
        ee.Image: light-dependent activity factor.
    
    Description:
        Computes the light-dependent activity factor
    
    Mathematical Details:
        - Alpha = 0.004 - 0.0005·ln(PPFD240)
        - C1 = 0.0468·exp[0.0005·(PPFD24 - PSTD)]·PPFD240^0.6
        - Ea1p99New = Alpha·C1·PPFD1 / sqrt((Alpha²·PPFD1²) + 1)
    
    Example:
        ```python
        - example call
            - result = Ea1p99(PPFD1_img, PPFD24_img, PPFD240_img, 200.0)
        ```
    """
    Ea1p99 = PPFD1.multiply(ee.Image(0))
    Alpha =  ee.Image(0.004).subtract(PPFD240.log().multiply(ee.Image(0.0005)))
    C1 = ee.Image(0.0468).multiply(((PPFD24.subtract(ee.Image(PSTD))).multiply(ee.Image(0.0005))).exp()).multiply(PPFD240.pow(0.6))
    Ea1p99New = Alpha.multiply(C1).multiply(PPFD1).divide(((Alpha.pow(2)).multiply(PPFD1.pow(2)).add(ee.Image(1))).sqrt())
    Ea1p99 = Ea1p99.where(PPFD240.gte(0.01), Ea1p99New)
    return Ea1p99


def Ealti99(T, TEMPD_PRM, SPC_NAME):
    """
    --------------------------------------------------------------------------
    Function: Ealti99
    --------------------------------------------------------------------------
    Parameters:
        - T (ee.Image): Temperature image in Kelvin.
        - TEMPD_PRM (dict): Dictionary containing:
            - 'TDF_SPC' (list): Species codes list.
            - 'TDF_PRM' (list): Corresponding temperature response parameters.
        - SPC_NAME (string): Species code to select parameter.
    
    Returns:
        - ee.Image:
            - ealti99 = exp(TDF_PRM * (T - Ts)), where Ts = 303.15 K.
    
    Description:
        Computes a species-specific  temperature response factor.

    
    Mathematical Details:
        - ealti99 = exp[TDF_PRM * (T - Ts)]
    
    Example:
        ```python
        - example call
            - result = Ealti99(T_img, TEMPD_PRM, "speciesA")
        ```
    """
    Ts = 303.15
    index = ee.List(TEMPD_PRM.get('TDF_SPC')).indexOf(SPC_NAME)
    TDF_PRM = ee.Number(ee.List(TEMPD_PRM.get('TDF_PRM')).get(index))
    ealti99 = (T.subtract(ee.Image(Ts)).multiply(ee.Image(TDF_PRM))).exp()
    return ealti99


def updateDictList(DataDictList, BandNameList, Layers):
    """
    --------------------------------------------------------------------------
    Function: updateDictList
    --------------------------------------------------------------------------
    Parameters:
        - DataDictList (ee.Dictionary): Dictionary of ImageCollections to convert.
        - BandNameList (list of str): List of band names to extract images for.
        - Layers (int): Number of layers/images in each collection.

    Returns:
        ee.Dictionary: A dictionary mapping each band name to an ee.List of ee.Image objects.
            - Each key is the band name, and the value is the list of images for that band.

    Description:
        Converts a dictionary of image collections into separate lists of images for each band.
        Iterates through collection entries by layer and selects the specified band from each image.
        Returns an Earth Engine Dictionary of per-band image lists.

    Mathematical Details:

    Example:
        ```python
        - example call
            - result = updateDictList(DataDictList, ['B1','B2'], 5)
        ```
    """
    DataDictList_list = DataDictList.toList(Layers)
    bands = BandNameList
    results = {}

    for band in bands:
        bandImages = ee.List.sequence(1, Layers).map(lambda i: ee.Image(DataDictList_list.get(ee.Number(i).subtract(1))).select(band))
        results[band] = bandImages
    return ee.Dictionary(results)

def GAMMA_CE( Day, Hour, lat, Tc, T24, T240, 
             PPFD ,PPFD24,PPFD240, Wind, Humidity, Pres, DI, lai_c, 
             Layers, SPC_NAME, Cantype):
    """
    --------------------------------------------------------------------------
    Function: GAMMA_CE
    --------------------------------------------------------------------------
    Parameters:
        - Day (ee.Image): image of date values for each grid cell
        - Hour (ee.Image): image of hour values for each grid cell
        - lat (ee.Image): image of latitude values for each grid cell
        - Tc (ee.Image): Canopy temperature image in Kelvin.
        - T24 (ee.Image): 24-hour mean temperature image in Kelvin.
        - T240 (ee.Image): 240-hour mean temperature image in Kelvin.
        - PPFD (ee.Image): Instantaneous photosynthetic photon flux density.
        - PPFD24 (ee.Image): 24-hour mean PPFD.
        - PPFD240 (ee.Image): 240-hour mean PPFD.
        - Wind (ee.Image): Wind speed image in m/s.
        - Humidity (ee.Image):  Specific humidity.
        - Pres (ee.Image): Atmospheric pressure image in Pa.
        - DI (ee.Image): Drought index image.
        - lai_c (ee.Image): Leaf area index image.
        - Layers (int): Number of canopy layers.
        - SPC_NAME (string): Species code key.
        - Cantype (int): Canopy type index.

    Returns:
        ee.Image: Image with bands:
            - 'Ea1CanopyFinal': Final canopy emission activity factor.
                - Sum over layers of (Ea1t × Ea1p) weighted by SLW and Gaussian weights,
                  then scaled by canopy coefficient Cce and LAI.
            - 'EatiCanopySUM': Sum of temperature activity factors across layers.

    Description:
        Computes canopy-level emission potentials by integrating layer-specific temperature
        and light response factors with sunlit leaf weights and Gaussian integration.
        Uses MEGAN2.1 

    Mathematical Details:
        - Ea1t = temperature response factor 
        - Ea1p = light response factor 

    Example:
        ```python
        - result = GAMMA_CE(Day_img, Hour_img, lat, Tc_img, T24_img, T240_img,
                            PPFD_img, PPFD24_img, PPFD240_img, Wind_img,
                            Humidity_img, Pres_img, DI_img, lai_img,
                            Layers, "speciesA", Cantype)
        ```
    """
    import Echo_parameter as Echo_parameter
    REA_SPC = Echo_parameter.REA_SPC
    canopyDict = Echo_parameter.canopyDict
   
    CONST_CANOPY = Echo_parameter.CONST_CANOPY
    TEMPD_PRM = Echo_parameter.TEMPD_PRM

    waterAirRatio = ee.Number(18.016).divide(28.97)

    beta = Calcbeta(lat, Day, Hour)
    Sinbeta = (beta.multiply(math.pi / 180)).sin()

    SolarConstant = 1367

    Maxsolar= (Sinbeta.multiply(SolarConstant)).multiply(calcEccentricity(Day))
    # 转换 solar 为太阳光中可视部分的辐射通量密度
    solar =  PPFD.divide(2.25)
    solarFractionImage = solarFractions(solar, Maxsolar)
    Qdiffv = solarFractionImage.select('Qdiffv')
    Qbeamv = solarFractionImage.select('Qbeamv')
    Qdiffn = solarFractionImage.select('Qdiffn')
    Qbeamn = solarFractionImage.select('Qbeamn')

    gaussian_integration = gaussianIntegration(Layers)
    weightgauss = gaussian_integration['weightgauss']
    distgauss = gaussian_integration['distgauss']

    SLW = weightSLW(distgauss,weightgauss,lai_c,Layers)


    LAI = lai_c
    CanopyChar = ee.List(canopyDict.get('CanopyChar'))
  
    CanopyRad_res = CanopyRad(distgauss, Layers, lai_c, Sinbeta, Qbeamv, Qbeamn, Qdiffv, Qdiffn, Cantype, CanopyChar)

    updateCanopyRadList_res = updateCanopyRadList(CanopyRad_res, Layers)
    
    QbAbsV = ee.List(updateCanopyRadList_res.get("QbAbsV"))
    QbAbsn = ee.List(updateCanopyRadList_res.get("QbAbsn"))
    Sunfrac = ee.List(updateCanopyRadList_res.get("Sunfrac"))
    SunQn = ee.List(updateCanopyRadList_res.get("SunQn"))
    ShadeQn = ee.List(updateCanopyRadList_res.get("ShadeQn"))
    SunQv = ee.List(updateCanopyRadList_res.get("SunQv"))
    ShadeQv = ee.List(updateCanopyRadList_res.get("ShadeQv"))
    SunPPFD = ee.List(updateCanopyRadList_res.get("SunPPFD"))
    ShadePPFD = ee.List(updateCanopyRadList_res.get("ShadePPFD"))
    QdAbsV = ee.List(updateCanopyRadList_res.get("QdAbsV"))
    QsAbsV = ee.List(updateCanopyRadList_res.get("QsAbsV"))
    QdAbsn = ee.List(updateCanopyRadList_res.get("QdAbsn"))
    QsAbsn = ee.List(updateCanopyRadList_res.get("QsAbsn"))

    HumidairPa0 = waterVapPres(Humidity, Pres, waterAirRatio)

    Trate  = stability(CanopyChar,Cantype,solar)

    TairK0 = Tc
    Ws0 = Wind
    StomataDI = DIstomata(DI)

    CanopyEBresult = CanopyEB(Trate,Layers,distgauss,CanopyChar,
                    Cantype,StomataDI,TairK0,HumidairPa0,Ws0,
                    SunPPFD,ShadePPFD,SunQv,ShadeQv,SunQn,ShadeQn)
    SunLeafTk = ee.List(CanopyEBresult.get("SunLeafTk"))
    SunLeafSH = ee.List(CanopyEBresult.get("SunLeafSensibleHeat"))
    SunLeafLH = ee.List(CanopyEBresult.get("SunLeafLatentHeat"))
    SunLeafIR = ee.List(CanopyEBresult.get("SunLeafIRout"))
    ShadeLeafTk = ee.List(CanopyEBresult.get("ShadeLeafTk"))
    ShadeLeafSH = ee.List(CanopyEBresult.get("ShadeLeafSensibleHeat"))
    ShadeLeafLH = ee.List(CanopyEBresult.get("ShadeLeafLatentHeat"))
    ShadeLeafIR = ee.List(CanopyEBresult.get("ShadeLeafIRout"))
    Ws = ee.List(CanopyEBresult.get("Ws"))
    HumidairPa = ee.List(CanopyEBresult.get("HumidairPa"))
    TairK = ee.List(CanopyEBresult.get("TairK"))

    pstd_Sun = 200.0
    pstd_Shade = 50.0
    def map_func(index):
        i = index
        Sunleaftk_i = ee.Image(SunLeafTk.get(i))
        Sunfrac_i = ee.Image(Sunfrac.get(i))
        Shadeleaftk_i = ee.Image(ShadeLeafTk.get(i))
        SunPPFD_i = ee.Image(SunPPFD.get(i))
        ShadePPFD_i = ee.Image(ShadePPFD.get(i))

        #Ea1tLayer
        Ea1tLayerSun_i   = Ea1t99(Sunleaftk_i,T24,T240,CONST_CANOPY,SPC_NAME)
        Ea1tLayerShade_i = Ea1t99(Shadeleaftk_i,T24,T240,CONST_CANOPY,SPC_NAME)
        Ea1tLayer_i      = (Ea1tLayerSun_i.multiply(Sunfrac_i))\
                          .add(Ea1tLayerShade_i.multiply(ee.Image(1).subtract(Sunfrac_i)))

        #Ea1pLayer
        Ea1pLayerSun_i   = Ea1p99(SunPPFD_i,PPFD24.multiply(0.5),PPFD240.multiply(0.5),pstd_Sun)
        Ea1pLayerShade_i = Ea1p99(ShadePPFD_i,PPFD24.multiply(0.16),PPFD240.multiply(0.16),pstd_Shade)
        Ea1pLayer_i      = (Ea1tLayerSun_i.multiply(Sunfrac_i))\
                          .add(Ea1tLayerShade_i.multiply(ee.Image(1).subtract(Sunfrac_i)))

        #Ea1Layer
        Ea1Layer_i       = (Ea1tLayerSun_i.multiply(Ea1pLayerSun_i).multiply(Sunfrac_i))\
                          .add(Ea1tLayerShade_i.multiply(Ea1pLayerShade_i).multiply(ee.Image(1).subtract(Sunfrac_i)))

        #EatiLayer
        EatiLayerSun_i   = Ealti99(Sunleaftk_i,TEMPD_PRM,SPC_NAME)
        EatiLayerShade_i = Ealti99(Shadeleaftk_i,TEMPD_PRM,SPC_NAME)
        EatiLayer_i      =(EatiLayerSun_i.multiply(Sunfrac_i))\
                          .add(EatiLayerShade_i.multiply(ee.Image(1).subtract(Sunfrac_i)))

        # Canopy calculations
        VPgausWt_i = ee.Number(weightgauss.get(i))
        VPslwWT = ee.Image(SLW.get(i))
        Ea1tCanopy_i = Ea1tLayer_i.multiply(VPslwWT).multiply(VPgausWt_i)
        Ea1pCanopy_i = Ea1pLayer_i.multiply(VPslwWT).multiply(VPgausWt_i)
        Ea1Canopy_i  = Ea1Layer_i.multiply(VPslwWT).multiply(VPgausWt_i)
        EatiCanopy_i = EatiLayer_i.multiply(VPslwWT).multiply(VPgausWt_i)
        #out Ea1Canopy,EatiCanopy
        # Ea1Canopy = 
        # Output image
        outputImage =  ee.Image.cat([Ea1tCanopy_i,Ea1pCanopy_i,Ea1Canopy_i,EatiCanopy_i]).rename(['Ea1tCanopy','Ea1pCanopy','Ea1Canopy','EatiCanopy'])
        resultDict = {'index': i, 'outputImage': outputImage}
        return resultDict


    CanopyEa1TPCI = ee.List.sequence(0, ee.Number(Layers).subtract(1)).map(map_func)

    def dict_to_image(dict):
        outputImage = ee.Dictionary(dict).get('outputImage')
        return ee.Image(outputImage)

    CanopyEa1TPCICollection = ee.ImageCollection(CanopyEa1TPCI.map(dict_to_image))
    CanopyEa1TPCI_BandName = ['Ea1tCanopy','Ea1pCanopy','Ea1Canopy','EatiCanopy']
    CanopyEa1TPCI_Update = updateDictList(CanopyEa1TPCICollection, CanopyEa1TPCI_BandName, Layers)
    Ea1Canopy = ee.ImageCollection.fromImages(CanopyEa1TPCI_Update.get("Ea1Canopy"))
    Ea1CanopySUM = Ea1Canopy.reduce(ee.Reducer.sum())
    EatiCanopy = ee.ImageCollection.fromImages(CanopyEa1TPCI_Update.get("EatiCanopy"))
    EatiCanopySUM = EatiCanopy.reduce(ee.Reducer.sum())
    Cce = 0.56
    Ea1CanopyFinal = Ea1CanopySUM.multiply(Cce).multiply(LAI)
    outputImage = ee.Image.cat([Ea1CanopyFinal,EatiCanopySUM])\
                          .rename([
                            ee.String('Ea1CanopyFinal'),
                            ee.String('EatiCanopySUM')])
    return outputImage 