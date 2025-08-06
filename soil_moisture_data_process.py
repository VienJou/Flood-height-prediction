import os
import json
import numpy as np
import geopandas as gpd
import rasterio

# Set working directory
os.chdir('/anvil/projects/x-cis250634/team5')
print("Current Working Directory:", os.getcwd())

# Load shapefile
orig_points = gpd.read_file('./shp/filtered_flooding_points_over_one_day_nopr.shp')
points = orig_points.copy()

# ✅ Initialize new column to avoid KeyError
points['soil_moisture'] = np.nan

# Soil moisture raster directories
soil_moisture_tifs = [
    'Flooding_SMAP9KM_3hr_before',
    'Flooding_SMAP9KM_6hr_before',
    'Flooding_SMAP9KM_9hr_before',
    'Flooding_SMAP9KM_12hr_before',
]

def soil_moisture():
    # Load JSON data
    with open('./data_0718/json/flooding_dataset_with_precipitation_0801_with_height_above.json') as f:
        aligned_data = json.load(f)

    # Loop through each entry in JSON
    for data in aligned_data:
        # Extract matching point
        point = points[points['ID'] == str(data['id'])]
        if point.empty:
            continue  # Skip if no matching point

        soil_moi_values = []

        for tif_folder in soil_moisture_tifs:
            json_key = tif_folder[9:]  # e.g., '3hr_before'
            if json_key not in data:
                continue  # Skip if key is missing

            tif_filename = data[json_key]
            tif_path = f'./data_0718/Flooding_SMAP9KM/{tif_folder}/{tif_filename}'

            if not os.path.exists(tif_path):
                continue  # Skip if file is missing

            with rasterio.open(tif_path) as src:
                # Reproject point to match raster CRS if needed
                if src.crs != point.crs:
                    point_proj = point.to_crs(src.crs)
                else:
                    point_proj = point

                coords = [(x, y) for x, y in zip(point_proj.geometry.x, point_proj.geometry.y)]
                nodata = src.nodata

                # Sample raster values at point location
                values = list(src.sample(coords))
                moisture_values = [v[0] for v in values if v[0] != nodata and not np.isnan(v[0])]

                if moisture_values:
                    soil_moi_values.extend(moisture_values)

        # Assign the max value or NaN to the soil_moisture column
        if soil_moi_values:
            max_value = max(soil_moi_values)
        else:
            max_value = np.nan

        points.loc[points['ID'] == str(data['id']), 'soil_moisture'] = max_value

    # Output result
    print("soil_moisture:\n", points['soil_moisture'])
    points[['ID', 'S1_Date', 'soil_moisture']].to_csv('./xchen/soil_moisture.csv', index=False)

def precipitation():
    # Initialize column to avoid KeyError
    points['precipitation'] = np.nan

    with open('./data_0718/json/flooding_dataset_with_precipitation_0801_with_height_above.json') as f:
        aligned_data = json.load(f)

    for data in aligned_data:
        point = points[points['ID'] == str(data['id'])]
        if point.empty:
            continue

        prec = data['CHIRPS_1day_ahead']
        tif_path = f'./data_0718/CHIRPS/CHIRPS_1day_ahead/{prec}'

        if not os.path.exists(tif_path):
            continue

        with rasterio.open(tif_path) as src:
            point_proj = point.to_crs(src.crs) if src.crs != point.crs else point
            coords = [(x, y) for x, y in zip(point_proj.geometry.x, point_proj.geometry.y)]

            nodata = src.nodata
            values = list(src.sample(coords))
            precipitation = [v[0] for v in values if v[0] != nodata and not np.isnan(v[0])]

            # Assign first valid value or NaN
            points.loc[points['ID'] == str(data['id']), 'precipitation'] = precipitation[0] if precipitation else np.nan

    points[['ID', 'S1_Date', 'precipitation']].to_csv('./xchen/precipitation2.csv', index=False)
    print("Precipitation added!")


if __name__ == "__main__":
    precipitation()
