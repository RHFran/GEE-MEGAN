# GEE-MEGAN

GEE-MEGAN extends MEGAN2.1 biogenic VOC model to Google Earth Engine (GEE).

With dynamic, satellite-based land-cover and vegetation data, it generates near-real-time BVOC emissions at 10-30 m resolution, and can run continental-scale (500m) or global (5k m) simulations. 




## 1. Requirements

| Item        | Version                       |
| ----------- | ------------------------------|
| OS          | CentOS 7, Windows 10/11       |
| Python      | 3.8 - 3.9 (tested on 3.8.19)  |
| Key libs    | earthengine-api 0.1.409 · geemap 0.32.1 · numpy 1.24.3 · rasterio 1.3.10 · netCDF4 1.7.1 |

A complete list is in ‘requirements.txt’.



## 2. Installation

### 1. Install Miniconda (skip if already installed)

```
bash Miniconda3-latest-Linux-x86_64.sh
```

### 2. Create and activate an env

```
conda create -n tcg_env python=3.8.19 -y

conda activate tcg_env
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Authenticate Google Earth Engine

```
earthengine authenticate #follow the browser prompt
```

------------------------------------------------------------------------------------

Typical installation time **20–30 minutes** on a home connection.



## 3. Running the model (Demo)

### 3.1 Edit the configuration

Open the <namelist> file in the project root and adjust paths and options (each field is commented).

### 3.2 Single timestep

<BASH>

python main_all_type.py namelist '2018-06-01 03:00:00' '2018-06-01 03:00:00'

*#* *➜ output/GEE_MEGAN_ER_[SPECIES]_2018-06-01_03.tif*

### 3.3 Batch run

<BASH>

python Batch_run.py namelist

*#* *➜ output/*.tif  output/*.nc (if NetCDF enabled)

Processing 744 hourly steps (one month) took ~1 h in our tests; runtime is dominated by GEE, not local hardware.



## 4 Code Layout

```
GEE_MEGAN/
	├── EMPORC.py                        # Single-step emission core
	├── canopy_lib.py                    # Canopy energy balance
	├── gamma_lib.py                     # γ-factor calculations
	├── ECMWF_ERA5_Preprocess.py         # ERA5 & LAI preprocessing
	├── main_all_type.py                 # Driver for single / multi-scale runs
	├── Batch_run.py                     # Multi-timestep wrapper
	├── namelist                         # Example config (inline docs)
	└── requirements.txt
```

Each file begins with a doc-string describing inputs, outputs, and functions.



## 5. License

GEE-MEGAN is released under the Apache 2.0 License (see *LICENSE* file).

