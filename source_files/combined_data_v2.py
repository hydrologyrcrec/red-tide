# # Final script to merge MODIS Aqua NetCDF L2 OC files per day and clip to Florida

# from pathlib import Path
# from datetime import datetime, timezone
# from collections import defaultdict
# import numpy as np
# import xarray as xr
# from pyresample import geometry, kd_tree
# from tqdm import tqdm
# import re

# # ------------------------- USER CONFIGURABLE PARAMETERS -------------------------
# INPUT_DIR = Path("/Users/akhilreddy/Downloads/Red-Tide/Data/MODISA_L2_OC_2022.0-20250716_205129")  # Replace with your input directory
# OUTPUT_DIR = Path("/Users/akhilreddy/Downloads/Red-Tide/Data/MODISA_L2_OC_2022-2023-v2-2") # Replace with your desired output directory
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# PARAMS = ["chlor_a", "Kd_490", "nflh", "par", "Rrs_443", "Rrs_469", "Rrs_488"]
# FILL_VALUE = np.nan  # Internal placeholder for masked pixels

# # Florida bounding box (approximate, rectangular)
# FLORIDA_BBOX = dict(lat_min=24.523096, lat_max=31.000888,
#                     lon_min=-87.634938, lon_max=-80.031362)
# TARGET_RES = 0.01  # ~1km resolution in degrees

# # ------------------------- GRID PREPARATION -------------------------
# lat_grid = np.arange(FLORIDA_BBOX["lat_max"], FLORIDA_BBOX["lat_min"] - TARGET_RES, -TARGET_RES)
# lon_grid = np.arange(FLORIDA_BBOX["lon_min"], FLORIDA_BBOX["lon_max"] + TARGET_RES,  TARGET_RES)
# lon2d, lat2d = np.meshgrid(lon_grid, lat_grid)
# target_def = geometry.GridDefinition(lons=lon2d, lats=lat2d)

# # ------------------------- HELPER FUNCTIONS -------------------------
# def extract_date_from_filename(filename: str) -> str:
#     match = re.search(r'AQUA_MODIS\.(\d{8})T\d{6}', filename)
#     return match.group(1) if match else None

# def load_variable(file: Path, var: str):
#     try:
#         ds = xr.open_dataset(file, group="geophysical_data")
#         nav = xr.open_dataset(file, group="navigation_data")
#         if var not in ds:
#             return None
#         data = ds[var]
#         lat = nav['latitude']
#         lon = nav['longitude']
#         # Broadcast lat/lon if 1D
#         if lat.ndim == 1 and lon.ndim == 1:
#             lon, lat = np.meshgrid(lon, lat)
#         return data.values, lat, lon
#     except Exception as e:
#         print(f"Failed to load {var} from {file.name}: {e}")
#         return None

# # ------------------------- MAIN WORKFLOW -------------------------

# def process_day(files: list[Path], date_str: str):
#     print(f"[Processing] {date_str} with {len(files)} swaths")
#     daily_data = {}

#     for param in PARAMS:
#         resampled_stack = []

#         for file in files:
#             result = load_variable(file, param)
#             if result is None:
#                 continue
#             data, lat, lon = result

#             # Build swath definition and mask fill values
#             swath_def = geometry.SwathDefinition(lons=lon, lats=lat)
#             data = np.ma.masked_invalid(data)

#             try:
#                 resampled = kd_tree.resample_nearest(
#                     swath_def, data, target_def, radius_of_influence=5000, fill_value=np.nan
#                 )
#                 resampled_stack.append(resampled)
#             except Exception as e:
#                 print(f"Resampling failed for {file.name}: {e}")

#             if resampled_stack:
#                 # Filter valid (non-empty, partially valid) arrays
#                 valid_resampled = [
#                     arr for arr in resampled_stack
#                     if arr.size > 0 and not np.all(np.isnan(arr))
#                 ]
#                 if not valid_resampled:
#                     print(f"⚠️ Skipping {param} for {date_str} (no valid resampled data)")
#                     continue

#                 stacked = np.stack(valid_resampled)

#                 # Replace NaNs with 0 before computing mean
#                 stacked = np.nan_to_num(stacked, nan=0)
#                 daily_data[param] = np.mean(stacked, axis=0)



#     if not daily_data:
#         print(f"⚠️ No valid data for {date_str}")
#         return

#     # Build xarray dataset
#     ds_out = xr.Dataset({
#         param: (['lat', 'lon'], daily_data[param])
#         for param in daily_data
#     }, coords={
#         'lat': (['lat'], lat_grid),
#         'lon': (['lon'], lon_grid)
#     })

#     ds_out.attrs["title"] = f"Daily composite for {date_str} over Florida"
#     ds_out.attrs["history"] = f"Created {datetime.now(timezone.utc).isoformat()}"
#     output_file = OUTPUT_DIR / f"AQUA_MODIS_FL_{date_str}.L2.OC.nc"
#     ds_out.to_netcdf(output_file, format="NETCDF4")
#     print(f"✅ Saved: {output_file.name}")

# # ------------------------- GROUP FILES BY DAY -------------------------
# all_files = sorted(INPUT_DIR.glob("AQUA_MODIS.*.L2.OC.nc"))
# daily_files = defaultdict(list)

# for f in all_files:
#     date_str = extract_date_from_filename(f.name)
#     if date_str:
#         daily_files[date_str].append(f)

# print(f"📁 Found {len(daily_files)} days to process")

# for date_str, files in tqdm(daily_files.items(), desc="Processing days"):
#     process_day(files, date_str)

# print("\n🎉 All done. Each daily NetCDF file is clipped to Florida and saved.")

from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np
import xarray as xr
from pyresample import geometry, kd_tree
from tqdm import tqdm
import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------- USER CONFIGURABLE PARAMETERS -------------------------
INPUT_DIR = Path("/Users/akhilreddy/Downloads/Red-Tide/Data/MODISA_L2_OC_2022.0-20250716_205129")
OUTPUT_DIR = Path("/Users/akhilreddy/Downloads/Red-Tide/Data/MODISA_L2_OC_2022-2023-v2-2.3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARAMS = ["chlor_a", "Kd_490", "nflh", "par", "Rrs_443", "Rrs_469", "Rrs_488"]
FILL_VALUE = np.nan

FLORIDA_BBOX = dict(lat_min=24.523096, lat_max=31.000888,
                    lon_min=-87.634938, lon_max=-80.031362)
TARGET_RES = 0.01

# ------------------------- GRID PREPARATION -------------------------
lat_grid = np.arange(FLORIDA_BBOX["lat_max"], FLORIDA_BBOX["lat_min"] - TARGET_RES, -TARGET_RES)
lon_grid = np.arange(FLORIDA_BBOX["lon_min"], FLORIDA_BBOX["lon_max"] + TARGET_RES,  TARGET_RES)
lon2d, lat2d = np.meshgrid(lon_grid, lat_grid)
target_def = geometry.GridDefinition(lons=lon2d, lats=lat2d)

# ------------------------- HELPER FUNCTIONS -------------------------
def extract_date_from_filename(filename: str) -> str:
    match = re.search(r'AQUA_MODIS\.(\d{8})T\d{6}', filename)
    return match.group(1) if match else None

def load_variable(file: Path, var: str):
    try:
        ds = xr.open_dataset(file, group="geophysical_data")
        nav = xr.open_dataset(file, group="navigation_data")
        if var not in ds:
            return None
        data = ds[var]
        lat = nav['latitude']
        lon = nav['longitude']
        if lat.ndim == 1 and lon.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)
        return data.values, lat, lon
    except Exception as e:
        print(f"Failed to load {var} from {file.name}: {e}")
        return None

# ------------------------- MAIN WORKFLOW -------------------------
def process_day(files: list[Path], date_str: str):
    print(f"[Processing] {date_str} with {len(files)} swaths")
    daily_data = {}

    for param in PARAMS:
        resampled_stack = []

        for file in files:
            result = load_variable(file, param)
            if result is None:
                continue
            data, lat, lon = result

            swath_def = geometry.SwathDefinition(lons=lon, lats=lat)
            data = np.ma.masked_invalid(data)

            try:
                resampled = kd_tree.resample_gauss(
                    swath_def, data, target_def,
                    radius_of_influence=3000,  # Adjust as needed
                    sigmas=1500,                # Tighter kernel for sharper features
                    neighbours=32,
                    fill_value=np.nan
                )
                resampled_stack.append(resampled)
            except Exception as e:
                print(f"Resampling failed for {file.name}: {e}")

        if resampled_stack:
            valid_resampled = [
                arr for arr in resampled_stack
                if arr.size > 0 and not np.all(np.isnan(arr))
            ]
            if not valid_resampled:
                print(f"⚠️ Skipping {param} for {date_str} (no valid resampled data)")
                continue

            stacked = np.stack(valid_resampled)
            daily_data[param] = np.nanmean(stacked, axis=0)

    if not daily_data:
        print(f"⚠️ No valid data for {date_str}")
        return

    ds_out = xr.Dataset({
        param: (['lat', 'lon'], daily_data[param], {'_FillValue': np.nan})
        for param in daily_data
    }, coords={
        'lat': (['lat'], lat_grid),
        'lon': (['lon'], lon_grid)
    })
    ds_out.attrs["title"] = f"Daily composite for {date_str} over Florida"
    ds_out.attrs["history"] = f"Created {datetime.now(timezone.utc).isoformat()}"
    output_file = OUTPUT_DIR / f"AQUA_MODIS_FL_{date_str}.L2.OC.nc"
    ds_out.to_netcdf(output_file, format="NETCDF4")
    print(f"✅ Saved: {output_file.name}")

# ------------------------- GROUP FILES BY DAY -------------------------
all_files = sorted(INPUT_DIR.glob("AQUA_MODIS.*.L2.OC.nc"))
daily_files = defaultdict(list)

for f in all_files:
    date_str = extract_date_from_filename(f.name)
    if date_str:
        daily_files[date_str].append(f)

print(f"📁 Found {len(daily_files)} days to process")

for date_str, files in tqdm(daily_files.items(), desc="Processing days"):
    process_day(files, date_str)

print("\n🎉 All done. Each daily NetCDF file is clipped to Florida and saved.")
