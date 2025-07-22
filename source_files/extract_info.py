import h5py
import numpy as np
from pyproj import Geod

# File path
nc_file = "/Users/akhilreddy/Downloads/Red-Tide/Data/MODISA_L2_OC_2022.0-20250716_205129/AQUA_MODIS.20230202T180001.L2.OC.nc"

# Load data using h5py
with h5py.File(nc_file, 'r') as f:
    chlor = f['geophysical_data/chlor_a'][:]
    lat = f['navigation_data/latitude'][:]
    lon = f['navigation_data/longitude'][:]

# Check shape and dimensionality
assert lat.shape == lon.shape == chlor.shape, "Shape mismatch in lat/lon/data arrays"
assert lat.ndim == 2, "Expecting 2D latitude/longitude arrays"

# Get center pixel indices
center_y, center_x = lat.shape[0] // 2, lat.shape[1] // 2

# Extract 4 coordinates to compute pixel spacing
lat1 = lat[center_y, center_x]
lat2 = lat[center_y + 1, center_x]
lon1 = lon[center_y, center_x]
lon2 = lon[center_y, center_x + 1]

# Calculate spacing in degrees
dx_deg = abs(lon2 - lon1)
dy_deg = abs(lat2 - lat1)
print(f"Pixel size in degrees: dx = {dx_deg:.6f}, dy = {dy_deg:.6f}")

# Convert spacing to meters using WGS84 ellipsoid
geod = Geod(ellps="WGS84")
_, _, dx_m = geod.inv(lon1, lat1, lon2, lat1)  # horizontal
_, _, dy_m = geod.inv(lon1, lat1, lon1, lat2)  # vertical

print(f"Approximate pixel size: dx = {dx_m:.2f} meters, dy = {dy_m:.2f} meters")
