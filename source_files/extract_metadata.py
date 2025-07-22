import os
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_origin

input_folder = "/Users/akhilreddy/Downloads/Red-Tide/Data/MODISA_L2_OC_2022.0-20250716_205129"
output_folder = "/Users/akhilreddy/Downloads/Red-Tide/Data/MODISA_L2_OC_2022.0-20250716_205129_GeoTIFF"
parameters = ["chlor_a", "Kd_490", "nflh", "Rrs_443", "Rrs_469", "Rrs_488", "par"]

# === Ensure output folder exists ===
os.makedirs(output_folder, exist_ok=True)

def process_file(nc_file):
    print(f"Processing {nc_file}")
    
    # Open NetCDF groups
    geo = xr.open_dataset(nc_file, group="geophysical_data")
    nav = xr.open_dataset(nc_file, group="navigation_data")
    
    # Read lat/lon
    lat = nav['latitude']
    lon = nav['longitude']

    # Handle 1D lat/lon
    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)

    lat_vals = lat.values
    lon_vals = lon.values


    # Loop over each parameter
    for param in parameters:
        if param not in geo:
            print(f"⚠️ Skipping missing parameter: {param}")
            continue

        if data.shape != lat_vals.shape:
            print(f"❌ Shape mismatch for {param} in {nc_file}. Skipping.")
            continue

        valid_ratio = np.sum(np.isfinite(data)) / data.size
        if valid_ratio < 0.01:
            print(f"⚠️ Only {valid_ratio*100:.2f}% valid data in {param}. Skipping.")
            continue


        # Get metadata from attributes
        attrs = geo[param].attrs
        long_name = attrs.get("long_name", "Unknown")
        units = attrs.get("units", "Unknown")
        valid_min = attrs.get("valid_min", "N/A")
        valid_max = attrs.get("valid_max", "N/A")

        print(f"📌 {param}: {long_name}, Units: {units}, Range: {valid_min} to {valid_max}")


        data = geo[param].values
        data = np.where(np.isfinite(data), data, np.nan)

        if np.isnan(data).all():
            print(f"❌ {param} in {nc_file} is entirely NaN. Skipping.")
            continue

        # Raster transform (affine)
        pixel_size_x = (lon_vals.max() - lon_vals.min()) / lon_vals.shape[1]
        pixel_size_y = (lat_vals.max() - lat_vals.min()) / lat_vals.shape[0]
        transform = from_origin(lon_vals.min(), lat_vals.max(), pixel_size_x, pixel_size_y)

        # Output filename
        base = os.path.basename(nc_file).replace(".nc", f"_{param}.tif")
        output_path = os.path.join(output_folder, base)

        # Write GeoTIFF
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs="EPSG:4326",
            transform=transform,
            nodata=np.nan,
        ) as dst:
            dst.write(data, 1)
        
        print(f"✅ Saved {output_path}")

# === Batch all files ===
for fname in os.listdir(input_folder):
    if fname.endswith(".nc"):
        process_file(os.path.join(input_folder, fname))
