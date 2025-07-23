from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import re
import numpy as np
import xarray as xr
from pyresample import geometry, kd_tree
from typing import List, Dict

# ------------------------ Configuration ------------------------

# Input and output directories
INPUT_DIR = Path("/Users/akhilreddy/Downloads/Red-Tide/Data/MODISA_L2_OC_2022.0-20250716_205129")
OUTPUT_DIR = Path("/Users/akhilreddy/Downloads/Red-Tide/Data/MODISA_L2_OC_2022-2023-Combined-Data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameters to extract from L2 OC files
PARAMS_REQUIRED = ["chlor_a", "Kd_490", "nflh", "par", "Rrs_443", "Rrs_469", "Rrs_488"]
PARAMS_OPTIONAL = ["l2_flags"]  # Include if you want to preserve QA information
PARAMS = PARAMS_REQUIRED + PARAMS_OPTIONAL

# Fill value and dtype settings
FILL_VALUE = np.nan          # Standard fill value (NaN)
DTYPE = np.float32           # Use float32 for space-efficient storage

# Resampling and grid configuration
RADIUS_METERS = 5000         # Radius of influence for nearest-neighbor resampling (in meters)
RESOLUTION_DEG = 0.01        # Output grid resolution (in degrees ≈ 1 km)

# Florida bounding box for output grid
LAT_MIN, LAT_MAX = 24.52, 31.0
LON_MIN, LON_MAX = -87.63, -80.03

# Compositing strategy: mean, max, or latest
COMPOSITE_RULE = "mean"

# Toggle for applying l2_flags-based QA mask
APPLY_QA_MASK = False

# --------------------- Target Grid Setup ------------------------

def make_target_grid():
    """Return 2‑D lat/lon and pyresample definition of the ROI grid."""
    lat_vec = np.arange(LAT_MAX, LAT_MIN - RESOLUTION_DEG, -RESOLUTION_DEG)
    lon_vec = np.arange(LON_MIN, LON_MAX + RESOLUTION_DEG, RESOLUTION_DEG)
    lon2d, lat2d = np.meshgrid(lon_vec, lat_vec)
    grid_def = geometry.GridDefinition(lats=lat2d, lons=lon2d)
    return lat_vec, lon_vec, lat2d, lon2d, grid_def

LAT_VEC, LON_VEC, LAT2D, LON2D, TARGET_DEF = make_target_grid()
_RE_DATE = re.compile(r"AQUA_MODIS\.(\d{8})T\d{6}")

# ----------------------- Utilities -----------------------------

def extract_date(path: Path) -> str | None:
    """Extract YYYYMMDD from filename."""
    m = _RE_DATE.search(path.name)
    return m.group(1) if m else None

# ------------------ Granule Processing -------------------------

def resample_swath(nc_path: Path, grid_def: geometry.GridDefinition) -> Dict[str, np.ndarray]:
    """
    Read a single MODIS-Aqua L2 OC granule, remap selected parameters to the fixed output grid.
    Applies cloud/glint mask if APPLY_QA_MASK is True.
    """
    out = {}
    with xr.open_dataset(nc_path, group="geophysical_data", chunks={}) as ds:
        nav = xr.open_dataset(nc_path, group="navigation_data", chunks={})
        lats = nav["latitude"].values
        lons = nav["longitude"].values
        swath_def = geometry.SwathDefinition(lats=lats, lons=lons)

        if APPLY_QA_MASK and "l2_flags" in ds:
            flags = ds["l2_flags"].values
            good = (flags & 0b111) == 0
        else:
            good = np.ones_like(ds[PARAMS[0]].values, dtype=bool)

        for p in PARAMS:
            if p not in ds:
                raise KeyError(f"{p} missing in {nc_path}")

            data = ds[p].values.astype(DTYPE)
            data[~good] = np.nan  # apply QA mask only if enabled

            remapped = kd_tree.resample_nearest(
                source_geo_def=swath_def,
                data=data,
                target_geo_def=grid_def,
                radius_of_influence=RADIUS_METERS,
                fill_value=FILL_VALUE
            )
            out[p] = remapped

    return out

# ------------------ Daily Composite ----------------------------

def composite_one_day(nc_files: List[Path]) -> tuple[xr.Dataset, xr.Dataset]:
    nfiles = len(nc_files)
    ny, nx = LAT2D.shape
    stack = {p: np.full((nfiles, ny, nx), FILL_VALUE, dtype=DTYPE) for p in PARAMS}

    for idx, nc in enumerate(nc_files):
        remap = resample_swath(nc, TARGET_DEF)
        for p, arr in remap.items():
            stack[p][idx] = arr

    comp = {}
    for p, cube in stack.items():
        if COMPOSITE_RULE == "mean":
            valid_mask = ~np.isnan(cube)
            valid_count = np.sum(valid_mask, axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                summed = np.nansum(cube, axis=0)
                mean = np.where(valid_count > 0, summed / valid_count, FILL_VALUE)
            comp[p] = mean
        elif COMPOSITE_RULE == "max":
            comp[p] = np.nanmax(cube, axis=0)
        elif COMPOSITE_RULE == "latest":
            for slice_idx in reversed(range(nfiles)):
                layer = cube[slice_idx]
                mask = ~np.isnan(layer)
                if slice_idx == nfiles - 1:
                    latest = layer
                else:
                    latest[mask] = layer[mask]
            comp[p] = latest
        else:
            raise ValueError("Unknown COMPOSITE_RULE")

    geo_ds = xr.Dataset(
        {
            p: (("lat", "lon"), comp[p],
                {"_FillValue": FILL_VALUE, "units": getattr_unit(p)})
            for p in comp
        },
        coords={
            "lat": ("lat", LAT_VEC, {"units": "degrees_north"}),
            "lon": ("lon", LON_VEC, {"units": "degrees_east"})
        },
        attrs={
            "title": "MODIS‑Aqua daily OC composite",
            "institution": "Generated by composite_one_day()",
            "source": "NASA MODIS‑Aqua L2 OC granules",
            "history": f"Created {datetime.now(timezone.utc).isoformat()}",
            "composite_rule": COMPOSITE_RULE,
            "masking_applied": str(APPLY_QA_MASK)
        }
    )

    nav_ds = xr.Dataset(
        {
            "latitude": (("lat", "lon"), LAT2D),
            "longitude": (("lat", "lon"), LON2D)
        },
        coords=geo_ds.coords,
    )

    return geo_ds, nav_ds

def getattr_unit(varname: str) -> str:
    """Return scientific units for known ocean color parameters."""
    default = "1"
    units = {
        "chlor_a": "mg m^-3",
        "nflh": "W cm^-2 µm^-1 sr^-1",
        "par": "Einstein m^-2 day^-1",
        "Kd_490": "m^-1",
        "Rrs_443": "sr^-1",
        "Rrs_469": "sr^-1",
        "Rrs_488": "sr^-1",
    }
    return units.get(varname, default)

# --------------------- Main Driver ------------------------------

def main() -> None:
    groups: Dict[str, List[Path]] = defaultdict(list)
    for f in INPUT_DIR.rglob("*.L2.OC.nc"):
        d = extract_date(f)
        if d:
            groups[d].append(f)

    if not groups:
        raise RuntimeError(f"No granules found in {INPUT_DIR}")

    print(f"Found {len(groups)} days to process")
    for day, files in sorted(groups.items()):
        out_nc = OUTPUT_DIR / f"AQUA_MODIS_FL.{day}.L2.OC.nc"
        if out_nc.exists():
            print(f"[{day}] already done – skipping")
            continue

        print(f"[{day}] building composite from {len(files)} swaths")
        try:
            geo_ds, nav_ds = composite_one_day(sorted(files))

            geo_ds.to_netcdf(
                out_nc,
                group="geophysical_data",
                encoding={v: {"zlib": True, "complevel": 4} for v in geo_ds.data_vars}
            )
            nav_ds.to_netcdf(
                out_nc,
                mode="a",
                group="navigation_data",
                encoding={v: {"zlib": True, "complevel": 4} for v in nav_ds.data_vars}
            )
            print(f"[{day}] written → {out_nc}")
        except Exception as exc:
            print(f"[{day}] ERROR: {exc}")

if __name__ == "__main__":
    main()