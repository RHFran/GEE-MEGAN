# ------------------------------------------------------------------------------
# Module: Image_Postprocessing.py
# ------------------------------------------------------------------------------
# Description:
#     Convert a series of hourly GeoTIFF flux outputs for specified species into
#     time-indexed NetCDF4 files, including spatial coordinates and metadata.
#
# Inputs:
#     - directory_path (str): Directory containing TIFF files named
#         GEE_MEGAN_ER_<species>_YYYY-MM-DD_HH.tif
#     - species_list (list of str): Species codes to process (e.g., ["ISOP", "MYRC"]).
#     - start_day (str): Simulation start timestamp, format "YYYY-MM-DD HH:MM:SS".
#     - end_day (str): Simulation end timestamp, format "YYYY-MM-DD HH:MM:SS".
#     - output_dir (str, optional): Directory to write NetCDF files; defaults to directory_path.
#
# Functions:
#     - convert_tifs_to_netcdf(directory_path, species_list, start_day, end_day, output_dir=None) -> None
#
# Outputs:
#     - NetCDF4 files named GEE-MEGAN_<species>_flux_<start>_to_<end>.nc
#       each containing variables:
#         * time (hours since 1970-01-01)
#         * lat (y, x)
#         * lon (y, x)
#         * <species>_flux (time, y, x)
#
# Main Functionality:
#     1. Compile sorted list of GeoTIFFs per species using regex matching.
#     2. Read raster dimensions, transform, and CRS metadata.
#     3. Load each TIFF into a NumPy data cube with associated datetime list.
#     4. Generate 2D latitude and longitude grids for cell centers.
#     5. Create NetCDF4 file with appropriate dimensions, variables, and attributes.
#
# Dependencies:
#     - os
#     - glob
#     - re
#     - datetime
#     - numpy
#     - rasterio
#     - netCDF4 
#
# Licensed under the Apache License, Version 2.0 (see LICENSE)
# ------------------------------------------------------------------------------
import os
import glob
import re
from datetime import datetime
import numpy as np
import rasterio
from netCDF4 import Dataset, date2num

def convert_tifs_to_netcdf(directory_path, species_list, start_day, end_day, output_dir=None):
    """
    --------------------------------------------------------------------------
    Function: convert_tifs_to_netcdf
    --------------------------------------------------------------------------
    Parameters:
        - directory_path (str): Path where the .tif files are stored.
        - species_list (list of str): List of species codes (e.g. ["ISOP", "MYRC"]).
        - start_day (str): Simulation start date, format "YYYY-MM-DD HH:MM:SS".
        - end_day (str): Simulation end date, format "YYYY-MM-DD HH:MM:SS".
        - output_dir (str, optional): Directory to write the .nc files; defaults to directory_path.
        
    Returns:
        - None: Writes one NetCDF file per species in output_dir.
        
    Description:
        Search for TIFF files matching a naming pattern for each species, load
        them in chronological order, assemble a time-series data cube, compute
        geographic coordinates, and write the data into NetCDF format.
                
    Example:
        ```python
        - convert_tifs_to_netcdf("/data/.../Beijing-local-flux-compare", ["ISOP","MYRC"], 
        "2022-06-01 00:00:00", "2022-06-30 23:00:00", "/output/dir")
        ```
    """
    if output_dir is None:
        output_dir = directory_path
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


    # expected filename pattern: GEE_MEGAN_ER_{species}_{YYYY-MM-DD}_{HH}.tif
    filename_re = re.compile(
        r"GEE_MEGAN_ER_(?P<sp>\w+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<hour>\d{2})\.tif$"
    )

    for sp in species_list:
        # collect and sort all tif paths for this species
        pattern = os.path.join(directory_path, f"GEE_MEGAN_ER_{sp}_*.tif")
        tif_paths = sorted(
            glob.glob(pattern),
            key=lambda p: datetime.strptime(
                filename_re.search(os.path.basename(p)).group("date") + " "
                + filename_re.search(os.path.basename(p)).group("hour") + ":00:00",
                "%Y-%m-%d %H:%M:%S"
            )
        )

        if not tif_paths:
            print(f"[WARN] No .tif files found for species {sp}")
            continue

        # read metadata from first tif
        with rasterio.open(tif_paths[0]) as src0:
            height = src0.height
            width  = src0.width
            transform = src0.transform
            crs = src0.crs

        # build time list and data cube
        times = []
        data_cube = np.empty((len(tif_paths), height, width), dtype=np.float32)

        for idx, tif in enumerate(tif_paths):
            m = filename_re.search(os.path.basename(tif))
            dt = datetime.strptime(f"{m.group('date')} {m.group('hour')}:00:00",
                                   "%Y-%m-%d %H:%M:%S")
            times.append(dt)

            with rasterio.open(tif) as src:
                arr = src.read(1)  # read single band
            data_cube[idx, :, :] = arr

        # compute lat/lon arrays for each grid cell center
        # use rasterio.transform.xy to build 2D lon/lat
        rows = np.arange(height)
        cols = np.arange(width)
        cols2d, rows2d = np.meshgrid(cols, rows)
        lon2d, lat2d = rasterio.transform.xy(
            transform,
            rows2d,
            cols2d,
            offset="center"
        )
        lon2d = np.array(lon2d)
        lat2d = np.array(lat2d)

        # prepare output NetCDF
        out_nc = os.path.join(
            output_dir,
            f"GEE-MEGAN_{sp}_flux_{start_day[:10]}_to_{end_day[:10]}.nc"
        )
        with Dataset(out_nc, "w", format="NETCDF4") as nc:
            # dimensions
            nc.createDimension("time", None)
            nc.createDimension("y", height)
            nc.createDimension("x", width)

            # variables
            time_var = nc.createVariable("time", "f8", ("time",))
            lat_var  = nc.createVariable("lat",  "f8", ("y", "x"))
            lon_var  = nc.createVariable("lon",  "f8", ("y", "x"))
            data_var = nc.createVariable(
                sp + "_flux", "f4", ("time", "y", "x"), zlib=True
            )

            # assign coordinate data
            units = "hours since 1970-01-01 00:00:00"
            time_var[:] = date2num(times, units=units, calendar="standard")
            time_var.units = units
            time_var.calendar = "standard"

            lat_var[:, :] = lat2d
            lon_var[:, :] = lon2d
            lat_var.units = "degrees_north"
            lon_var.units = "degrees_east"

            # assign flux data
            data_var[:, :, :] = data_cube
            data_var.units = "flux_units(μg m^{-2} h^{-1})" 

            # global attrs
            nc.description = f"{sp} flux from {start_day} to {end_day}"
            nc.source = "GEE-MEGAN processed outputs"

        print(f"[INFO] Wrote NetCDF: {out_nc}")