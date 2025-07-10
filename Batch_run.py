# ------------------------------------------------------------------------------
# Module: batch_processing.py
# ------------------------------------------------------------------------------
# Description:
#     Batch-run GEE-MEGAN over a date range, species list, and ROI,
#     check for missing TIFFs, and optionally convert to NetCDF.
#
# Inputs:
#     - start_day (str): start datetime in "%Y-%m-%d %H:%M:%S" format.
#     - end_day (str): end datetime in "%Y-%m-%d %H:%M:%S" format.
#     - directory_path (str): path to TIFF output directory for checks and processing.
#     - script_path (str): path to the main GEE-MEGAN processing Python script.
#     - species_list (list): list of chemical species strings to process.
#     - namelist (str): path to the configuration namelist file.
#     - nc_format (str): flag ("True"/"False") to enable NetCDF conversion.
#     - nc_path (str): output directory path for NetCDF files.
#
# Outputs:
#     - Prints counts of unmatched dates and status messages to stdout.
#     - Creates or updates directories for TIFF and NetCDF outputs.
#     - Writes NetCDF files if nc_format is "True".
#     - Launches subprocesses running the processing script for missing intervals.
#
# Functions:
#     - generate_month_ranges(start_day, end_day) -> list:
#         Splits the full period into month-long (or shorter) date ranges.
#     - generate_three_day_ranges_for_month(start_day, end_day) -> list:
#         Splits each month range into consecutive three-day intervals.
#     - generate_hourly_time_points(start_date, end_date) -> list:
#         Produces hourly datetime points between two datetimes.
#     - check_files_and_generate_unmatched_dates(directory_path, date_ranges, species) -> tuple:
#         Checks for existing TIFFs matching each hourly timestamp and species;
#         returns matched filenames and missing datetime strings.
#     - update_namelist(namelist_path, species=None, roi=None, temp_directory=None) -> None:
#         Updates lines in the namelist file for species, ROI, and temp directory.
#     - parse_dict(arg_value) -> dict:
#         Safely parses a string literal into a Python dict for argparse.
#     - read_parameters_from_file(file_path) -> dict:
#         Reads and converts key=value lines from a config file into a dict.
#     - batch_calculation(start_day, end_day, directory_path, script_path, species_list, namelist) -> None:
#         Coordinates monthly loops, updates namelist, checks missing TIFFs, and
#         launches parallel subprocesses for each missing timestamp.
#
# Main Functionality:
#     1. Parse command-line arguments for namelist path.
#     2. Read configuration parameters from the namelist file.
#     3. Build and ensure existence of output directories.
#     4. Run batch_calculation
#     5. If NetCDF conversion is enabled, convert all generated TIFFs to NetCDF.
#
# Dependencies:
#     - argparse
#     - ast
#     - os
#     - glob
#     - datetime
#     - subprocess
#     - re
#     - Image_Postprocessing
# ------------------------------------------------------------------------------

import argparse
import ast
import os
import glob
from datetime import datetime, timedelta
import subprocess
import re
import Image_Postprocessing as Postprocessing

def generate_month_ranges(start_day, end_day):
    """
    --------------------------------------------------------------------------
    Function: generate_month_ranges
    --------------------------------------------------------------------------
    Parameters:

        - start_day (str): start datetime string in "%Y-%m-%d %H:%M:%S" format
        - end_day (str): end datetime string in "%Y-%m-%d %H:%M:%S" format

    Returns:

        - list: list of tuples, each containing start and end datetime strings for each month period

    Description:

        Splits the period from start_day to end_day into monthly ranges.
        Each tuple represents a month-long period, ensuring the end does not exceed end_day.

    Example:

    ```python
    result = generate_month_ranges("2021-01-01 00:00:00", "2021-03-15 00:00:00")
    ```
    """
    start = datetime.strptime(start_day, "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(end_day, "%Y-%m-%d %H:%M:%S")
    current = start
    
    date_ranges_month = []
    while current < end:
        next_month = current.replace(day=28) + timedelta(days=4)  # this will never fail
        next_month_start = next_month - timedelta(days=next_month.day - 1)
        next_month_start = min(next_month_start, end)  # ensure not to exceed end_day
        date_ranges_month.append((current.strftime("%Y-%'-%d %H:%M:%S"), next_month_start.strftime("%Y-%m-%d %H:%M:%S")))
        current = next_month_start
    return date_ranges_month

def generate_three_day_ranges_for_month(start_day, end_day):
    """
    --------------------------------------------------------------------------
    Function: generate_three_day_ranges_for_month
    --------------------------------------------------------------------------
    Parameters:

        - start_day (str): start datetime string in "%Y-%m-%d %H:%M:%S" format
        - end_day (str): end datetime string in "%Y-%m-%d %H:%M:%S" format

    Returns:

        - list: list of tuples, each containing start and end datetime strings for each 3-day period

    Description:

        Splits the period from start_day to end_day into three-day intervals.
        Ensures the last interval ends exactly at end_day if fewer than three days remain.

    Example:

    ```python
    result = generate_three_day_ranges_for_month("2021-01-01 00:00:00", "2021-01-10 00:00:00")
    ```
    """
    start = datetime.strptime(start_day, "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(end_day, "%Y-%m-%d %H:%M:%S")
    date_ranges_day = []
    
    while start < end:
        next_three_day = start + timedelta(days=3)
        if next_three_day > end:
            next_three_day = end
        date_ranges_day.append((start.strftime("%Y-%m-%d %H:%M:%S"), next_three_day.strftime("%Y-%m-%d %H:%M:%S")))
        start = next_three_day
    return date_ranges_day


def generate_hourly_time_points(start_date, end_date):
    """
    --------------------------------------------------------------------------
    Function: generate_hourly_time_points
    --------------------------------------------------------------------------
    Parameters:

        - start_date (str): start datetime string in "%Y-%m-%d %H:%M:%S" format
        - end_date (str): end datetime string in "%Y-%m-%d %H:%M:%S" format

    Returns:

        - list: list of datetime objects at hourly intervals between start_date and end_date

    Description:

        Generates a list of datetime objects for every hour from start_date up to (but not including) end_date.

    Example:

    ```python
    points = generate_hourly_time_points("2021-01-01 00:00:00", "2021-01-01 05:00:00")
    ```
    """
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
    current_datetime = start_datetime
    time_points = []
    while current_datetime < end_datetime:
        time_points.append(current_datetime)
        current_datetime += timedelta(hours=1)
    return time_points

def check_files_and_generate_unmatched_dates(directory_path, date_ranges):
    """
    --------------------------------------------------------------------------
    Function: check_files_and_generate_unmatched_dates
    --------------------------------------------------------------------------
    Parameters:

        - directory_path (str): path to directory containing TIFF files
        - date_ranges (list): list of tuples with start and end datetime strings

    Returns:

        - tuple: (matched_files, unmatched_dates)
          - matched_files (list): filenames of TIFFs matching expected hourly patterns
          - unmatched_dates (list): datetime strings for which no matching file was found

    Description:

        Checks a directory for TIFF files matching each hourly time point within given date_ranges.
        Returns lists of matched filenames and missing datetime strings.

    Example:

    ```python
    matched, missing = check_files_and_generate_unmatched_dates("/data/tifs", [("2021-01-01 00:00:00","2021-01-01 03:00:00")])
    ```
    """
    tif_files = glob.glob(os.path.join(directory_path, '*.tif'))
    matched_files = []
    unmatched_dates = []
    
    for start_date, end_date in date_ranges:
        hourly_time_points = generate_hourly_time_points(start_date, end_date)
        for time_point in hourly_time_points:
            expected_pattern = time_point.strftime("%Y-%m-%d_%H")
            # Check if there is a file matching this time point
            found = False
            for file_path in tif_files:
                filename = os.path.basename(file_path)
                if expected_pattern in filename:
                    matched_files.append(filename)
                    found = True
                    break  # Stop searching once a match is found for this time point
            if not found:
                unmatched_dates.append(time_point.strftime("%Y-%m-%d %H:%M:%S"))

    return matched_files,unmatched_dates


def generate_month_ranges(start_day, end_day):
    """
    --------------------------------------------------------------------------
    Function: generate_month_ranges
    --------------------------------------------------------------------------
    Parameters:

        - start_day (str): start datetime string in "%Y-%m-%d %H:%M:%S" format
        - end_day (str): end datetime string in "%Y-%m-%d %H:%M:%S" format

    Returns:

        - list: list of tuples, each containing start and end datetime strings for each month period

    Description:

        Splits the period from start_day to end_day into monthly ranges.
        Each tuple represents a month-long period, ensuring the end does not exceed end_day.

    Mathematical Details:

        - None

    Example:

    ```python
    result = generate_month_ranges("2021-01-01 00:00:00", "2021-03-15 00:00:00")
    ```
    """
    start = datetime.strptime(start_day, "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(end_day, "%Y-%m-%d %H:%M:%S")
    current = start
    
    date_ranges_month = []
    while current < end:
        next_month = current.replace(day=28) + timedelta(days=4)  # this will never fail
        next_month_start = next_month - timedelta(days=next_month.day - 1)
        next_month_start = min(next_month_start, end)  # ensure not to exceed end_day
        date_ranges_month.append((current.strftime("%Y-%m-%d %H:%M:%S"), next_month_start.strftime("%Y-%m-%d %H:%M:%S")))
        current = next_month_start
    return date_ranges_month


def generate_three_day_ranges_for_month(start_day, end_day):
    """
    --------------------------------------------------------------------------
    Function: generate_three_day_ranges_for_month
    --------------------------------------------------------------------------
    Parameters:
        - start_day (str): start datetime in "%Y-%m-%d %H:%M:%S" format
        - end_day (str): end datetime in "%Y-%m-%d %H:%M:%S" format

    Returns:
        - list: list of tuples of start and end datetime strings for each 3-day period

    Description:
        Splits the interval from start_day to end_day into consecutive three-day ranges.
        If the remaining period is less than three days, the final range ends at end_day.

    Mathematical Details:
        - None

    Example:
        ```python
        ranges = generate_three_day_ranges_for_month("2021-01-01 00:00:00", "2021-01-10 00:00:00")
        ```
    """
    start = datetime.strptime(start_day, "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(end_day, "%Y-%m-%d %H:%M:%S")
    date_ranges_day = []
    
    while start < end:
        next_three_day = start + timedelta(days=3)
        if next_three_day > end:
            next_three_day = end
        date_ranges_day.append((start.strftime("%Y-%m-%d %H:%M:%S"), next_three_day.strftime("%Y-%m-%d %H:%M:%S")))
        start = next_three_day
    return date_ranges_day


def generate_hourly_time_points(start_date, end_date):
    """
    --------------------------------------------------------------------------
    Function: generate_hourly_time_points
    --------------------------------------------------------------------------
    Parameters:
        - start_date (str): start datetime in "%Y-%m-%d %H:%M:%S" format
        - end_date (str): end datetime in "%Y-%m-%d %H:%M:%S" format

    Returns:
        - list: list of datetime.datetime objects at hourly intervals

    Description:
        Produces a list of datetime objects for each hour between start_date and end_date.

    Example:
        ```python
        points = generate_hourly_time_points("2021-01-01 00:00:00", "2021-01-01 03:00:00")
        ```
    """
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
    current_datetime = start_datetime
    time_points = []
    while current_datetime < end_datetime:
        time_points.append(current_datetime)
        current_datetime += timedelta(hours=1)
    return time_points


def check_files_and_generate_unmatched_dates(directory_path, date_ranges, species):
    """
    --------------------------------------------------------------------------
    Function: check_files_and_generate_unmatched_dates
    --------------------------------------------------------------------------
    Parameters:
        - directory_path (str): path containing TIFF files
        - date_ranges (list): list of (start, end) datetime strings
        - species (str): species identifier to match in filenames

    Returns:
        - tuple:
            - matched_files (list): filenames matching expected patterns
            - unmatched_dates (list): datetime strings without matching files

    Description:
        Scans the directory for TIFF files that match each hourly timestamp and species.
        Returns lists of found filenames and missing timestamps.

    Mathematical Details:
        - None

    Example:
        ```python
        matched, missing = check_files_and_generate_unmatched_dates("/data", [("2021-01-01 00:00:00","2021-01-01 01:00:00")], "ISOP")
        ```
    """
    tif_files = glob.glob(os.path.join(directory_path, '*.tif'))
    matched_files = []
    unmatched_dates = []
    
    for start_date, end_date in date_ranges:
        hourly_time_points = generate_hourly_time_points(start_date, end_date)
        for time_point in hourly_time_points:
            expected_pattern = time_point.strftime("%Y-%m-%d_%H")
            # Check if there is a file matching this time point and species
            found = False
            for file_path in tif_files:
                filename = os.path.basename(file_path)
                if expected_pattern in filename and species in filename:
                    matched_files.append(filename)
                    found = True
                    break  # Stop searching once a match is found for this time point
            if not found:
                unmatched_dates.append(time_point.strftime("%Y-%m-%d %H:%M:%S"))

    return matched_files, unmatched_dates



def update_namelist(namelist_path, species=None, roi=None, temp_directory=None):
    """
    --------------------------------------------------------------------------
    Function: update_namelist
    --------------------------------------------------------------------------
    Parameters:
        - namelist_path (str): path to the namelist file
        - species (str or None): new species string to set; if None, leave unchanged
        - roi (list or None): new ROI coordinates list; if None, leave unchanged
        - temp_directory (str or None): new temporary directory path; if None, leave unchanged

    Returns:
        - None: updates the file in place

    Description:
        Reads the namelist file, replaces chem_names, roi, and TemporaryDirectory lines
        if corresponding arguments are provided, and writes back to the file.

    Example:
        ```python
        update_namelist("namelist", species="MYRC", roi=[0,0,10,10], temp_directory="/tmp")
        ```
    """
    with open(namelist_path, 'r') as file:
        lines = file.readlines()
    
    with open(namelist_path, 'w') as file:
        for line in lines:
           
            if line.strip().startswith("chem_names =") and species is not None:
                file.write(f"chem_names = ['{species}']\n")

            elif line.strip().startswith("roi =") and roi is not None:
                roi_str = f"roi = {roi}\n"
                file.write(roi_str)

            elif line.strip().startswith("TemporaryDirectory =") and temp_directory is not None:
                file.write(f"TemporaryDirectory = '{temp_directory}'\n")
            else:
                file.write(line)

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

def batch_calculation(start_day, end_day, directory_path, script_path, species_list,namelist):
    """
    --------------------------------------------------------------------------
    Function: batch_calculation
    --------------------------------------------------------------------------
    Parameters:
        - start_day (str): start datetime in "%Y-%m-%d %H:%M:%S" format
        - end_day (str): end datetime in "%Y-%m-%d %H:%M:%S" format
        - directory_path (str): path containing TIFF files
        - script_path (str): path to the processing script
        - species_list (list): list of species strings
        - namelist (str): path to the namelist file

    Description:
        For each month in the period and each species, updates the namelist,
        checks for unmatched dates, and runs the script in batches for missing timestamps.

    Example:
        ```python
        batch_calculation("2021-01-01 00:00:00", "2021-03-01 00:00:00", "/data", "run.py", ["ISOP"], "config.nml")
        ```
    """
    date_ranges_month = generate_month_ranges(start_day, end_day)
    for month_range in date_ranges_month:
        for species in species_list:
            update_namelist(namelist, species=species)
            matched_files, unmatched_dates = check_files_and_generate_unmatched_dates(directory_path, date_ranges_month, species)
            print(f'the number unmatched dates {len(unmatched_dates)}')
            while len(unmatched_dates) != 0:
                unmatched_date_ranges = [(date, date) for date in unmatched_dates]
                for i in range(0, len(unmatched_date_ranges), 40):
                    current_batch = unmatched_date_ranges[i:i+40]
                    processes = []
                    for start_date, _ in current_batch:
                        # use start_date for both start and end in subprocess call
                        process = subprocess.Popen(['python', script_path,namelist, start_date, start_date])
                        processes.append(process)
                    for process in processes:
                        process.wait()
                matched_files, unmatched_dates = check_files_and_generate_unmatched_dates(directory_path, date_ranges_month, species)
                print(f'Unmatched dates for species {species}:', len(unmatched_dates))

def read_parameters_from_file(file_path):
    """
    --------------------------------------------------------------------------
    Function: read_parameters_from_file
    --------------------------------------------------------------------------
    Parameters:
        - file_path (str): path to the namelist or config file

    Returns:
        - dict: mapping of parameter names to values (int, float, list, or str)

    Description:
        Reads a file line by line, ignores comments, splits on '=', strips inline comments,
        converts values to appropriate Python types, and returns a parameter dict.

    Mathematical Details:
        - None

    Example:
        ```python
        params = read_parameters_from_file("config.nml")
        ```
    """
    # initialize empty dict for parameters
    parameters = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # skip full-line comments
            if not line.strip().startswith('#'):
                if '=' in line:
                    # split into key and raw value parts
                    key, value = line.strip().split('=', 1)
                    # remove inline comments from the value
                    value = value.split('#', 1)[0].strip() 

                    # strip quotes around string literals
                    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
                        value = value[1:-1]

                    # convert to integer if digits only
                    if value.isdigit():  
                        parameters[key.strip()] = int(value)
                    elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:  # convert to float if numeric with one decimal point
                        parameters[key.strip()] = float(value)
                    elif value.startswith('[') and value.endswith(']'):  # evaluate list literals into Python lists
                        parameters[key.strip()] = eval(value)
                    else:
                        parameters[key.strip()] = str(value) 
    return parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch-run your GEE-MEGAN processing based solely on a namelist file."
    )
    parser.add_argument(
        "name_list_path",
        type=str,
        help="Path to your model parameters (namelist) file containing all settings"
    )
    args = parser.parse_args()

    # Read all required parameters from the namelist file
    params = read_parameters_from_file(args.name_list_path)
    start_day          = params["start_day"]          # e.g. "2017-05-14 00:00:00"
    end_day            = params["end_day"]            # e.g. "2017-06-26 00:00:00"
    species_list       = params["chem_names"]       # e.g. ["ISOP", "MYRC"]
    out_dir            = params["out_dir"]            # e.g. "/data/.../local-flux-compare"
    out_dir_filename   = params["out_dir_filename"]   # e.g. output_city_local
    TemporaryDirectory = params["TemporaryDirectory"] # e.g. 'Los_Angeles-final-ML'
    nc_format          = params["nc_format"]          # e.g. "Ture"
    nc_path            = params["nc_path"]            # e.g. "/data/.../output_for_nc_format"
    script_path        = params["main_py"]            # e.g. "/data/.../main_all_type.py"
    
    # Construct the working directory path by combining out_dir, out_dir_filename, and TemporaryDirectory
    directory_path = os.path.join(out_dir, out_dir_filename, TemporaryDirectory)
    os.makedirs(directory_path, exist_ok=True)  # ensure the directory exists
    os.makedirs(nc_path, exist_ok=True)  # ensure the directory exists
    
    # Launch the batch calculation
    batch_calculation(
        start_day      = start_day,
        end_day        = end_day,
        directory_path = directory_path,
        script_path    = script_path,
        species_list   = species_list,
        namelist       = args.name_list_path
    )

    if nc_format == 'Ture':
        # After batch processing, convert all TIFF outputs to NetCDF
        Postprocessing.convert_tifs_to_netcdf(
            directory_path=directory_path,
            species_list=species_list,
            start_day=start_day,
            end_day=end_day,
            output_dir=nc_path  
        )
