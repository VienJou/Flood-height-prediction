import geopandas as gpd
import requests
import os
from urllib.parse import urlencode

# Configuration
input_shapefile = "/u/wz53/Flooding/Flooding_codes/Data_augmentation/flooding_points_with_12hour_S1.shp"
output_folder = "/u/wz53/Flooding/Flooding_codes/Data_augmentation/USGS_3DEP_1m"
log_file = "usgs_3dep_1m_download_log.txt"
fail_log_file = "usgs_3dep_1m_download_fail_log.txt"
wcs_endpoint = "https://elevation.nationalmap.gov/arcgis/services/3DEPElevation/ImageServer/WCSServer"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load processed log
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        processed_ids = set(line.strip() for line in f)
else:
    processed_ids = set()

# Load shapefile
gdf = gpd.read_file(input_shapefile)
print(f"Loaded {len(gdf)} features from shapefile.")

# Iterate over each point
for idx, row in gdf.iterrows():
    try:
        point_id = str(row['ID'])
        lat = float(row['latitude_d'])
        lon = float(row['longitude_'])
        log_key = point_id  # Only use ID for logging and filename

        if log_key in processed_ids:
            print(f"⏩ Already processed {log_key}, skipping.")
            continue

        # Define bounding box (approx. 300m x 300m area)
        buffer_deg = 0.0015  # ~150m in degrees
        minx = lon - buffer_deg
        miny = lat - buffer_deg
        maxx = lon + buffer_deg
        maxy = lat + buffer_deg

        bbox = f"{minx},{miny},{maxx},{maxy}"
        filename = f"{point_id}_USGS_3DEP_1m.tif"
        output_path = os.path.join(output_folder, filename)

        # Construct WCS GetCoverage request
        params = {
            "SERVICE": "WCS",
            "VERSION": "1.0.0",
            "REQUEST": "GetCoverage",
            "COVERAGE": "3DEPElevation",
            "CRS": "EPSG:4326",
            "BBOX": bbox,
            "WIDTH": 512,
            "HEIGHT": 512,
            "FORMAT": "GeoTIFF"
        }

        url = f"{wcs_endpoint}?{urlencode(params)}"
        print(f"⬇️ Downloading {filename} ...")

        response = requests.get(url, timeout=60)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            f.write(response.content)

        print(f"✅ Saved to {output_path}")

        # Log success
        with open(log_file, 'a') as f:
            f.write(f"{log_key}\n")

    except Exception as e:
        print(f"❌ Failed to process ID {row.get('ID')}: {e}")
        with open(fail_log_file, 'a') as f:
            f.write(f"{point_id}: {e}\n")
        print(f"URL attempted: {url}")

print("All downloads attempted. Check logs for details.")

