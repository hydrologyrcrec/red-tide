# Final Refined Code to Mosaic and Clip the Netcdf Files

import arcpy, os, re
from collections import defaultdict
from uuid import uuid4
from datetime import datetime

# CONFIG
input_root = r"C:\Users\Akhil\University of Florida\Hydrology_Lab_RCREC_AI&DSS - General\Red Tide\Data\Reuploaded_Data\MODISA_L2_OC_2022.0-20250716_205129"
output_root = r"C:\Users\Akhil\University of Florida\Hydrology_Lab_RCREC_AI&DSS - General\Red Tide\Data\Reuploaded_Data\MODISA_L2_OC_2022.0-20250716_205129_Fl_Mosaicked"
florida_mask = r"C:\Users\Akhil\Downloads\Red-Tide\Florida_Coordinates\Florida_Rectangle.shp"
gdb_path = r"C:\Users\Akhil\Downloads\Red-Tide\processing.gdb"
cellsize = "0.01"
pixel_type = "32_BIT_FLOAT"
sr_wgs84 = arcpy.SpatialReference(4326)
start_date = datetime.strptime("20221111", "%Y%m%d")
end_date = datetime.strptime("20221130", "%Y%m%d")
os.makedirs(output_root, exist_ok=True)
arcpy.env.overwriteOutput = True
arcpy.env.outputCoordinateSystem = sr_wgs84
arcpy.env.addOutputsToMap = False # Prevents automatic map addition (safety)

# Group files by date
daily = defaultdict(list)
pattern = re.compile(r"AQUA_MODIS\.(\d{8})T\d{6}\.L2\.OC\.nc$")
for fname in os.listdir(input_root):
    m = pattern.match(fname)
    if m:
        date = m.group(1)
        daily[date].append(os.path.join(input_root, fname))
print(f"🔎 Found {len(daily)} days to process")

# Create gdb file if it does not exist
if not arcpy.Exists(gdb_path):
    arcpy.management.CreateFileGDB(os.path.dirname(gdb_path), os.path.basename(gdb_path))

# Loop through each date
for date_str, swaths in sorted(daily.items()):
    if not swaths:
        continue

    current_date = datetime.strptime(date_str, "%Y%m%d")
    if not (start_date <= current_date <= end_date):
        print(f"Skipping date {current_date} (out of range)")
        continue

    print(f"\n📅 Processing date: {date_str} → {len(swaths)} swaths")

    # Create output folder for the day
    day_output_dir = os.path.join(output_root, date_str)
    os.makedirs(day_output_dir, exist_ok=True)
    variables = ['chlor_a', 'Kd_490', 'nflh', 'par', 'Rrs_443', 'Rrs_469', 'Rrs_488']

    # Loop through all the variables to be processed
    for variable in variables:
        print(f"   🔄 Processing variable: {variable}")
        layers = []

        raw_tif = os.path.join(day_output_dir, f"{variable}_{date_str}.tif")
        clipped_tif = os.path.join(day_output_dir, f"{variable}_{date_str}_FL.tif")

        if os.path.exists(raw_tif) and os.path.exists(clipped_tif):
            print(f"   ⏩ Skipping {variable} for {date_str} (One of raw_tif or clipped_tif already exists)")
            continue

        # Create MD raster layers
        for idx, nc in enumerate(swaths):
            lyr = f"{variable}_{date_str}_{idx}"
            arcpy.md.MakeMultidimensionalRasterLayer(
                in_multidimensional_raster=nc,
                out_multidimensional_raster_layer=lyr,
                variables=f"/geophysical_data/{variable}"
            )
            layers.append(lyr)

        # Create Mosaic Dataset
        mosaic_name = f"mosaic_{variable}_{date_str}_{uuid4().hex[:6]}"
        mosaic_path = os.path.join(gdb_path, mosaic_name)

        arcpy.management.CreateMosaicDataset(
            gdb_path, mosaic_name, sr_wgs84, 1, pixel_type)

        for lyr in layers:
            arcpy.management.AddRastersToMosaicDataset(
                in_mosaic_dataset=mosaic_path,
                raster_type="Raster Dataset",
                input_path=lyr,
                update_cellsize_ranges="UPDATE_CELL_SIZES",
                update_boundary="UPDATE_BOUNDARY")

        arcpy.management.SetMosaicDatasetProperties(
            in_mosaic_dataset=mosaic_path,
            mosaic_operator="MEAN",
            default_mosaic_method="Northwest")

        # Save and clip raster
        arcpy.management.CopyRaster(
            in_raster=mosaic_path,
            out_rasterdataset=raw_tif,
            pixel_type=pixel_type,
            format="TIFF")

        clipped_tif = os.path.join(day_output_dir, f"{variable}_{date_str}_FL.tif")
        arcpy.management.Clip(
            in_raster=raw_tif,
            rectangle="#",
            out_raster=clipped_tif,
            in_template_dataset=florida_mask,
            clipping_geometry="ClippingGeometry",
            maintain_clipping_extent="NO_MAINTAIN_EXTENT")

        for lyr in layers:
            arcpy.management.Delete(lyr)

        print(f"✔ Saved: {raw_tif} and \n {clipped_tif}")

print("\n✅ All done.")