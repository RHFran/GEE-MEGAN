# ------------------------------------------------------------------------------
# Module:  gamma_lib.py
# ------------------------------------------------------------------------------
# Description:
#     A collection of functions to compute gamma scaling factors for
#     emission processes based on leaf area index (LAI), leaf age,
#     and soil moisture.
#
# Inputs:
#     - ee.Image objects representing spatial data (LAI, date, hour,
#       temperature, soil moisture).
#     - Numeric parameters for time step length and species-specific
#       parameter mappings.
#
# Functions:
#     - gamma_lai(LAI: ee.Image) -> ee.Image
#         Computes an empirical LAI scaling factor.
#     - GAMMA_A(Day: ee.Image, Hour: ee.Image, lai_c: ee.Image,
#               lai_p: ee.Image, TSTLEN: int,
#               TempureMeanDaily: ee.Image,
#               SPC_NAME: str, REA_SPC: dict-like,
#               REL_EM_ACT: dict-like) -> ee.Image
#         Computes leaf-age-adjusted gamma factor using foliage age
#         fractions and species-specific coefficients.
#     - GAMMA_S(soil_moisture: ee.Image) -> ee.Image
#         Generates a masked constant image of ones based on soil moisture.
#
# Outputs:
#     - ee.Image objects representing the requested gamma factors.
#
# Main Functionality:
#     1. gamma_lai: applies a quadratic relationship to derive a canopy
#        scaling factor.
#     2. GAMMA_A: calculates temperature-dependent age fractions and
#        combines them with species-specific parameters.
#     3. GAMMA_S: creates a soil moisture mask for scaling.
#
# Dependencies:
#     - ee
#
# Licensed under the Apache License, Version 2.0 (see LICENSE)
# ------------------------------------------------------------------------------

import ee
def gamma_lai(LAI):
    """
    --------------------------------------------------------------------------
    Function: gamma_lai
    --------------------------------------------------------------------------
    Parameters:
        - LAI (ee.Image): Leaf Area Index image for canopy
    
    Returns:
        - gamma_lai (ee.Image):  Leaf Area Index scaling factor
    
    Description:
        Computes a scaling factor based on LAI using an empirical relationship.
        This factor modulates canopy processes in emission calculations.
    
    Mathematical Details:
        - intermediate = LAI^2 * 0.2
        - gamma_lai = 0.49 * LAI / sqrt(intermediate + 1).
    
    Example:
        ```python
        - result = gamma_lai(LAI_image)
        ```
    """
    intermediate = LAI.pow(2).multiply(0.2)
    gamma_lai = LAI.multiply(0.49).divide(intermediate.add(1).sqrt())
    return gamma_lai


def GAMMA_A(Day, Hour, lai_c, lai_p, TSTLEN, TempureMeanDaily, SPC_NAME, REA_SPC, REL_EM_ACT):
    """
    --------------------------------------------------------------------------
    Function: GAMMA_A
    --------------------------------------------------------------------------
    Parameters:
        - Day (ee.Image): image of date values for each grid cell
        - Hour (ee.Image): image of hour values for each grid cell
        - lai_c (ee.Image): current Leaf Area Index image
        - lai_p (ee.Image): previous Leaf Area Index image
        - TSTLEN (int): time step length
        - TempureMeanDaily (ee.Image): daily mean temperature image
        - SPC_NAME (str): species name key in parameter lists
        - REA_SPC (dict-like): mapping of species names to parameter indices
        - REL_EM_ACT (dict-like): mapping containing age parameter lists 'Anew', 'Agro', 'Amat', 'Aold'
    
    Returns:
        - GAM_A (ee.Image):  gamma value adjusted by leaf age influence coefficient
    
    Description:
        Computes the gamma adjustment factor for leaf age based on  LAI values
        from the earlier (lai_p) and later (lai_c) periods to derive fractional
        proportions of new, growth, mature, and old foliage classes, then
        multiplies these fractions by species-specific correction coefficients.
    
    Mathematical Details:
        - Fnew, Fmat, Fgro, Fold: fractional foliage age distributions based on lai_p vs lai_c.
        - GAM_A = Anew·Fnew + Agro·Fgro + Amat·Fmat + Aold·Fold.
    
    Example:
        ```python
        - result = GAMMA_A(Day_img, Hour_img, lai_current, lai_prev, 30,
                          TempMeanDaily, 'ISOP', REA_params, REL_params)
        ```
    """
    # get species-specific parameter index
    SPC_num = ee.Number(REA_SPC.get(SPC_NAME))
    Anew = ee.List(REL_EM_ACT.get('Anew')).get(SPC_num)
    Agro = ee.List(REL_EM_ACT.get('Agro')).get(SPC_num)
    Amat = ee.List(REL_EM_ACT.get('Amat')).get(SPC_num)
    Aold = ee.List(REL_EM_ACT.get('Aold')).get(SPC_num)
    
    # assign time step and daily mean temperature
    t = TSTLEN
    Tt = TempureMeanDaily

    # calculate ti and tm
    ti = Tt.where(Tt.lte(303.0), ((Tt.subtract(300)).multiply(-0.7)).add(5.0))\
               .where(Tt.gt(303.0), 2.9)
    tm = ti.multiply(2.3)
    # Fnew
    Fnew = lai_p.where(lai_p.lt(lai_c), \
                            lai_p.where(ti.gte(t), (lai_p.divide(lai_c).subtract(1.0)).multiply(-1.0))\
                              .where(ti.lt(t), ti.divide(t).multiply((lai_p.divide(lai_c).subtract(1.0)).multiply(-1.0))))\
                .where(lai_p.eq(lai_c), 0.0)\
                .where(lai_p.gt(lai_c), 0.0)
    # Fmat
    Fmat = lai_p.where(lai_p.lt(lai_c), \
                        lai_p.where(tm.gte(t), lai_p.divide(lai_c))\
                             .where(tm.lt(t), lai_p.divide(lai_c).add((ee.Image(t).subtract(tm).divide(t))\
                            .multiply(ee.Image(1.0).subtract(lai_p.divide(lai_c))))))\
                .where(lai_p.eq(lai_c), 0.8)\
                .where(lai_p.gt(lai_c), (ee.Image(1).subtract((lai_p.subtract(lai_c)).divide(lai_p))))
    # Fgro
    Fgro = lai_p.where(lai_p.lt(lai_c), ee.Image(1.0).subtract(Fnew).subtract(Fmat))\
                .where(lai_p.eq(lai_c), 0.1)\
                .where(lai_p.gt(lai_c), 0.0)
    
    # Fold
    Fold = lai_p.where(lai_p.lt(lai_c), 0.0)\
                .where(lai_p.eq(lai_c), 0.1)\
                .where(lai_p.gt(lai_c), (lai_p.subtract(lai_c)).divide(lai_p))
   

    # combine age fractions with species-specific parameters
    GAM_A = Fnew.multiply(ee.Number(Anew))\
            .add(Fgro.multiply(ee.Number(Agro)))\
            .add(Fmat.multiply(ee.Number(Amat)))\
            .add(Fold.multiply(ee.Number(Aold)))
    return GAM_A


def GAMMA_S(soil_moisture):
    """
    --------------------------------------------------------------------------
    Function: GAMMA_S
    --------------------------------------------------------------------------

    Parameters:
        - soil_moisture (ee.Image): soil moisture image used for mask

    Returns:
        - GAM_S (ee.Image): temporary constant image of ones masked by soil moisture

    Description:
        Temporary replacement using soil moisture image to generate a constant
        ones image for GAM_S, ensuring output is one within valid soil moisture pixels.

    Example:
        ```python
        - result = GAMMA_S(soil_moisture_image)
        ```
    """
    # Copy mask from soil moisture image to maintain valid grid
    imageMask = soil_moisture.mask()
    # Create constant ones image with soil moisture mask
    GAM_S = ee.Image(1).updateMask(imageMask)
    return GAM_S

