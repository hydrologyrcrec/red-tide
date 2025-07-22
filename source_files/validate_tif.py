import rasterio
import numpy as np
import os

input_folder = "/Users/akhilreddy/Downloads/Red-Tide/Data/MODISA_L2_OC_2022.0-20250716_205129_GeoTIFF"

for filename in os.listdir(input_folder):
    if filename.endswith(".tif"):
        file_path = os.path.join(input_folder, filename)
        try:
            with rasterio.open(file_path) as src:
                arr = src.read(1)
                if np.isnan(arr).all():
                    print(f"❌ {filename}: All NaNs, skipping.")
                    continue
                min_val = np.nanmin(arr)
                max_val = np.nanmax(arr)
                print(f"✅ {filename}: Min={min_val:.4f}, Max={max_val:.4f}")
        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")
