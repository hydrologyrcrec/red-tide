from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re
import numpy as np
import h5py
import xarray as xr
from pyresample import geometry, kd_tree
import rasterio
from rasterio.transform import from_origin

# Input directory - contains all MODIS granules
IN_DIR  = Path("/Users/akhilreddy/Downloads/Red-Tide/Data/MODISA_L2_OC_2022.0-20250716_205129")    

# Output directory - Driectory where the combined modis granules will be added
OUT_DIR = Path("/Users/akhilreddy/Downloads/Red-Tide/Data/MODISA_L2_OC_2022-2023-Combined-Data")      
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Filtering only required parameters
REQUIRED_PARAMS  = ["nflh", "Rrs_488", "Kd_490", "par", "Rrs_469", "chlor_a", "Rrs_443"]

# Bounding box for florida region: https://observablehq.com/@rdmurphy/u-s-state-bounding-boxes
# West Longitude: -87.634938
# South Latitude: 24.523096
# East Longitude: -80.031362
# North Latitude: 31.000888
BBOX = dict(lat_min=24.523096, lat_max=31.000888, lon_min=-87.634938, lon_max=-80.031362)
TARGET_RES = 0.01        # degrees (≈1 km)
FILL_VALUE = np.nan       # use NaN internally; original _FillValue kept in attrs

# ---------- GRID PREPARATION -------------------------------------------------
lat_grid = np.arange(BBOX["lat_max"], BBOX["lat_min"] - TARGET_RES, -TARGET_RES)
lon_grid = np.arange(BBOX["lon_min"], BBOX["lon_max"] + TARGET_RES,  TARGET_RES)
lon2d, lat2d = np.meshgrid(lon_grid, lat_grid)            # 2‑D target grid

target_def = geometry.GridDefinition(lons=lon2d, lats=lat2d)
ny, nx = lat2d.shape
# -----------------------------------------------------------------------------

# holds latitude / longitude arrays
NAV_GROUP = "navigation_data"  

# holds science variables
GEO_GROUP = "geophysical_data"         

def parse_date(fname: str) -> datetime.date:
    """Extract YYYY-MM-DD from filenames like AQUA_MODIS.20220601T181500.L2.OC.nc"""
    m = re.search(r"AQUA_MODIS\.(\d{8})T\d{6}", fname)
    if not m:
        raise ValueError(f"Cannot parse date from {fname}")
    return datetime.strptime(m.group(1), "%Y%m%d").date()

def group_files_by_day(in_dir: Path) -> dict[datetime.date, list[Path]]:
    """Return {date: [granule1, granule2, …]}."""
    per_day = defaultdict(list)
    for fp in in_dir.glob("*.nc"):
        per_day[parse_date(fp.name)].append(fp)
    return per_day

def remap_swath(raw, lats, lons):
    """Nearest‑neighbour remap raw swath array onto target grid."""
    swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
    return kd_tree.resample_nearest(
        source_geo_def = swath_def,
        target_geo_def = target_def,
        data           = raw,
        radius_of_influence = 5000,
        fill_value     = np.nan,
        nprocs         = 1
    )

def clean_attrs(h5_attrs):
    """Return NetCDF‑safe attribute dict."""
    out = {}
    for k, v in h5_attrs.items():
        k = str(k)
        if isinstance(v, bytes):
            v = v.decode("utf-8")
        elif isinstance(v, np.ndarray):
            if v.size == 1:
                v = v.item()
            else:
                continue
        out[k] = v
    return out

def mosaic_day(files):
    # Initialise empty mosaic per param
    mosaics = {p: np.full((ny, nx), FILL_VALUE, dtype="float32") for p in REQUIRED_PARAMS}
    attrs   = {}   # store attrs from first granule

    for fp in files:
        with h5py.File(fp, "r") as f:
            lat = f["/navigation_data/latitude"][:]
            lon = f["/navigation_data/longitude"][:]

            # Clip early to bbox for speed
            mask_bbox = (
                (lat >= BBOX["lat_min"]) & (lat <= BBOX["lat_max"]) &
                (lon >= BBOX["lon_min"]) & (lon <= BBOX["lon_max"])
            )
            if not mask_bbox.any():
                continue  # granule outside bbox

            for p in REQUIRED_PARAMS:
                if p not in f["/geophysical_data"]:
                    continue
                raw = f[f"/geophysical_data/{p}"][:]
                att = f[f"/geophysical_data/{p}"].attrs
                if p not in attrs:
                    attrs[p] = clean_attrs(att)

                fill_val = att.get("_FillValue", None)
                if fill_val is not None:
                    raw = np.where(raw == fill_val, np.nan, raw)

                # Mask outside bbox
                raw = np.where(mask_bbox, raw, np.nan)

                # Remap
                remapped = remap_swath(raw, lat, lon)

                # Insert into mosaic: keep first valid encounter
                mosaic = mosaics[p]
                mask_new = np.isnan(mosaic) & ~np.isnan(remapped)
                mosaic[mask_new] = remapped[mask_new]

    # Build xarray Dataset
    data_vars = {}
    for p, arr in mosaics.items():
        da = xr.DataArray(arr, dims=("lat", "lon"),
                          coords={"lat": lat_grid, "lon": lon_grid},
                          attrs=attrs.get(p, {}))
        da.attrs["_FillValue"] = np.float32(np.nan)       # CF nan
        data_vars[p] = da

    ds = xr.Dataset(data_vars)
    ds.attrs["Conventions"] = "CF-1.6"
    ds.attrs["title"] = "MODIS Aqua L2 OC – Florida mosaic"
    # CF grid_mapping (WGS‑84)
    ds["crs"] = xr.DataArray(0, attrs={
        "grid_mapping_name": "latitude_longitude",
        "epsg_code": "EPSG:4326",
        "semi_major_axis": 6378137.0,
        "inverse_flattening": 298.257223563
    })
    for v in ds.data_vars:
        ds[v].attrs["grid_mapping"] = "crs"

    return ds

def save_geotiff(da: xr.DataArray, out_path: Path):
    """Save single‑band GeoTIFF for quick GIS preview."""
    arr = da.values.astype("float32")
    transform = from_origin(lon_grid.min(), lat_grid.max(), TARGET_RES, TARGET_RES)
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(arr, 1)

def main():
    for day, files in sorted(group_files_by_day(IN_DIR).items()):
        print(f"🟢 {day}: mosaicking {len(files)} granules…")
        ds = mosaic_day(files)

        nc_out = OUT_DIR / f"AQUA_MODIS_FL.{day}.L2.OC.nc" 
        comp   = {"zlib": True, "complevel": 4}
        enc    = {v: comp for v in ds.data_vars}
        ds.to_netcdf(nc_out, encoding=enc)
        print(f"   ✅ NetCDF written: {nc_out}")

        # Optional GeoTIFF preview for first parameter
        # first_param = next(iter(ds.data_vars))
        # tif_out = OUT_DIR / f"{first_param}_{day}.tif"
        # # save_geotiff(ds[first_param], tif_out)
        # print(f"   ✅ GeoTIFF preview: {tif_out}")

if __name__ == "__main__":
    main()