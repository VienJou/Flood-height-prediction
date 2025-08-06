import rasterio
import pandas as pd
import numpy as np
import os
import geopandas as gpd
import pylandstats as pls
from rasterio.mask import mask
from shapely.geometry import box
import warnings

# File paths
Flood_Points = r"C:\Users\alekb\Downloads\filtered_flooding_points_over_one_day_nopr.shp"
#Flood_Points = "/anvil/projects/x-cis250634/team5/shp/filtered_flooding_points_over_one_day_nopr.shp"
# Define NLCD file paths for each year
"""
nlcd_files = {
    2016: "/anvil/projects/x-cis250634/team5/data_0718/Annual_NLCD_ImpDsc_2016_CU_C1V1.tif",
    2017: "/anvil/projects/x-cis250634/team5/data_0718/Annual_NLCD_ImpDsc_2017_CU_C1V1.tif", 
    2018: "/anvil/projects/x-cis250634/team5/data_0718/Annual_NLCD_ImpDsc_2018_CU_C1V1.tif",
    2019: "/anvil/projects/x-cis250634/team5/data_0718/Annual_NLCD_ImpDsc_2019_CU_C1V1.tif",
    2020: "/anvil/projects/x-cis250634/team5/data_0718/Annual_NLCD_ImpDsc_2020_CU_C1V1.tif",
    2021: "/anvil/projects/x-cis250634/team5/data_0718/Annual_NLCD_ImpDsc_2021_CU_C1V1.tif",
    2022: "/anvil/projects/x-cis250634/team5/data_0718/Annual_NLCD_ImpDsc_2022_CU_C1V1.tif",
    2023: "/anvil/projects/x-cis250634/team5/data_0718/Annual_NLCD_ImpDsc_2023_CU_C1V1.tif",
    2024: "/anvil/projects/x-cis250634/team5/data_0718/Annual_NLCD_ImpDsc_2024_CU_C1V1.tif"
}
"""
# Windows paths for impervious surface data
nlcd_imp_files = {
     2016: r"C:\Users\alekb\Downloads\Annual_NLCD_ImpDsc_2016_CU_C1V1.tif",
     2017: r"C:\Users\alekb\Downloads\Annual_NLCD_ImpDsc_2017_CU_C1V1.tif",
     2018: r"C:\Users\alekb\Downloads\Annual_NLCD_ImpDsc_2018_CU_C1V1.tif",
     2019: r"C:\Users\alekb\Downloads\Annual_NLCD_ImpDsc_2019_CU_C1V1.tif",
     2020: r"C:\Users\alekb\Downloads\Annual_NLCD_ImpDsc_2020_CU_C1V1.tif",
     2021: r"C:\Users\alekb\Downloads\Annual_NLCD_ImpDsc_2021_CU_C1V1.tif",
     2022: r"C:\Users\alekb\Downloads\Annual_NLCD_ImpDsc_2022_CU_C1V1.tif",
     2023: r"C:\Users\alekb\Downloads\Annual_NLCD_ImpDsc_2023_CU_C1V1.tif",
     2024: r"C:\Users\alekb\Downloads\Annual_NLCD_ImpDsc_2024_CU_C1V1.tif"}

# Windows paths for vegetation/land cover data
nlcd_veg_files = {
     2016: r"C:\Users\alekb\Downloads\Annual_NLCD_LndCov_2016_CU_C1V1.tif",
     2017: r"C:\Users\alekb\Downloads\Annual_NLCD_LndCov_2017_CU_C1V1.tif",
     2018: r"C:\Users\alekb\Downloads\Annual_NLCD_LndCov_2018_CU_C1V1.tif",
     2019: r"C:\Users\alekb\Downloads\Annual_NLCD_LndCov_2019_CU_C1V1.tif",
     2020: r"C:\Users\alekb\Downloads\Annual_NLCD_LndCov_2020_CU_C1V1.tif",
     2021: r"C:\Users\alekb\Downloads\Annual_NLCD_LndCov_2021_CU_C1V1.tif",
     2022: r"C:\Users\alekb\Downloads\Annual_NLCD_LndCov_2022_CU_C1V1.tif",
     2023: r"C:\Users\alekb\Downloads\Annual_NLCD_LndCov_2023_CU_C1V1.tif",
     2024: r"C:\Users\alekb\Downloads\Annual_NLCD_LndCov_2024_CU_C1V1.tif"}


# Check which NLCD files exist
print("Checking NLCD file availability...")
available_years = []
for year in nlcd_imp_files.keys():
    imp_exists = os.path.exists(nlcd_imp_files[year])
    veg_exists = os.path.exists(nlcd_veg_files[year])
    
    if imp_exists and veg_exists:
        available_years.append(year)
        print(f"[OK] Found both NLCD files for {year}")
    elif imp_exists:
        print(f"[WARN] Found only impervious surface file for {year}")
    elif veg_exists:
        print(f"[WARN] Found only vegetation file for {year}")
    else:
        print(f"[MISSING] Missing both NLCD files for {year}")

print(f"\nAvailable years (with both files): {available_years}")

# Load flood points
print("\nLoading flood points...")
flood_points = gpd.read_file(Flood_Points)
print(f"Number of flood points: {len(flood_points)}")
print(f"CRS: {flood_points.crs}")
print(f"Columns: {flood_points.columns.tolist()}")

def extract_year_from_date(date_str):
    """Extract year from various date formats using pandas datetime parsing"""
    try:
        if pd.isna(date_str):
            return None
        parsed_date = pd.to_datetime(date_str, errors='coerce')
        if pd.isna(parsed_date):
            return None
        return parsed_date.year
    except:
        return None

# Extract years from peak_date
flood_points['year'] = flood_points['peak_date'].apply(extract_year_from_date)
print(f"\nExtracted years: {flood_points['year'].value_counts().sort_index()}")

def create_square_buffer(point, distance):
    """Create a square buffer around a point"""
    x, y = point.x, point.y
    return box(x - distance, y - distance, x + distance, y + distance)

def recode_nlcd_data(data):
    """Recode NLCD data: 1->1, 2->2, everything else->0"""
    return np.where((data == 1) | (data == 2), data, 0)

def group_vegetation_classes(data):
    """Group NLCD vegetation classes into parent categories"""
    # Forest: 41, 42, 43
    forest = np.isin(data, [41, 42, 43])
    # Shrubland: 51, 52  
    shrubland = np.isin(data, [51, 52])
    # Herbaceous: 71, 72, 73, 74
    herbaceous = np.isin(data, [71, 72, 73, 74])
    # Planted/Cultivated: 81, 82
    planted = np.isin(data, [81, 82])
    # Wetlands: 90, 95
    wetlands = np.isin(data, [90, 95])
    
    return {
        'forest': forest,
        'shrubland': shrubland, 
        'herbaceous': herbaceous,
        'planted': planted,
        'wetlands': wetlands
    }

# Create 1.5 km square buffers around each flood point
print(f"\nOriginal CRS: {flood_points.crs}")

# IMPORTANT: Always reproject to a projected CRS for accurate distance-based buffers
if flood_points.crs.is_geographic:
    flood_points_proj = flood_points.to_crs('EPSG:5070')
    print(f"Reprojected from geographic to projected CRS: {flood_points_proj.crs}")
    print("Buffer will be created in METERS")
else:
    flood_points_proj = flood_points.copy()
    print(f"Already in projected CRS: {flood_points_proj.crs}")
    print("Assuming CRS units are meters - verify this is correct!")

buffer_distance = 1500  # meters (1.5 km)
print(f"Buffer distance: {buffer_distance} meters ({buffer_distance/1000} km)")

print("Creating square buffers...")
flood_points_proj['square_buffer'] = flood_points_proj.geometry.apply(
    lambda point: create_square_buffer(point, buffer_distance)
)

buffered_points = flood_points_proj.copy()
buffered_points.geometry = buffered_points['square_buffer']

# Convert back to original CRS for consistency with NLCD data
if flood_points.crs.is_geographic:
    buffered_points = buffered_points.to_crs(flood_points.crs)
    print("Buffers converted back to original CRS for NLCD matching")

print(f"Created {len(buffered_points)} square buffers")

# Initialize results
buffer_metrics = []

print("\nCalculating multi-year landscape metrics...")

# Get reference CRS from first available NLCD file
reference_nlcd_file = nlcd_imp_files[available_years[0]]
with rasterio.open(reference_nlcd_file) as src:
    raster_crs = src.crs
    raster_transform = src.transform
    print(f"NLCD CRS: {raster_crs}")

# Ensure buffers are in the same CRS as raster
if buffered_points.crs != raster_crs:
    buffered_points_raster_crs = buffered_points.to_crs(raster_crs)
    print(f"Reprojected buffers to match NLCD CRS")
else:
    buffered_points_raster_crs = buffered_points.copy()

# Process each buffer with its corresponding year's NLCD data
for idx, buffer_geom in enumerate(buffered_points_raster_crs.geometry):
    try:
        flood_year = flood_points.iloc[idx]['year']
        original_id = flood_points.iloc[idx]['ID']
        peak_date = flood_points.iloc[idx]['peak_date']

        # Handle None flood_year or unavailable years
        if flood_year is None or flood_year not in available_years:
            if flood_year is None:
                closest_year = available_years[0]  # Use first available year as default
            else:
                # Find closest available year
                closest_year = min(available_years, key=lambda x: abs(x - flood_year))
            flood_year = closest_year

        nlcd_imp_file = nlcd_imp_files[flood_year]
        nlcd_veg_file = nlcd_veg_files[flood_year]

        # Process impervious surface data
        with rasterio.open(nlcd_imp_file) as src:
            # Mask the raster data with the buffer geometry
            masked_data, masked_transform = mask(src, [buffer_geom], crop=True, filled=False)
            masked_array = masked_data[0]

            # Handle masked values and recode data
            masked_array = np.where(masked_array.mask, -1, masked_array.data)
            recoded_masked = recode_nlcd_data(masked_array)

            # Create Landscape object
            ls = pls.Landscape(recoded_masked, res=(30, 30))

            # Calculate basic metrics
            metrics = {
                'ID': original_id,
                'peak_date': peak_date,
                'nlcd_year': flood_year,
                'total_area_km2': (recoded_masked.shape[0] * recoded_masked.shape[1] * 30 * 30) / 1000000,
            }

            # Calculate area coverage for each class
            class_1_pixels = np.sum(recoded_masked == 1)
            class_2_pixels = np.sum(recoded_masked == 2)
            total_pixels = recoded_masked.shape[0] * recoded_masked.shape[1]

            metrics['pct_area_1'] = (class_1_pixels / total_pixels) * 100
            metrics['pct_area_2'] = (class_2_pixels / total_pixels) * 100
            metrics['area_km_1'] = (class_1_pixels * 30 * 30) / 1000000
            metrics['area_km_2'] = (class_2_pixels * 30 * 30) / 1000000

            # Calculate core area index for impervious surface
            try:
                if class_1_pixels > 0:
                    cai_class_1 = ls.core_area_index(class_val=1, edge_depth=1, percent=True)
                    metrics['cai_1'] = cai_class_1.iloc[0] if isinstance(cai_class_1, pd.Series) else cai_class_1
                else:
                    metrics['cai_1'] = 0

                if class_2_pixels > 0:
                    cai_class_2 = ls.core_area_index(class_val=2, edge_depth=1, percent=True)
                    metrics['cai_2'] = cai_class_2.iloc[0] if isinstance(cai_class_2, pd.Series) else cai_class_2
                else:
                    metrics['cai_2'] = 0
            except Exception as e:
                print(f"Warning: Could not calculate CAI for ID {original_id}: {e}")
                metrics['cai_1'] = np.nan
                metrics['cai_2'] = np.nan

        # Process vegetation data
        with rasterio.open(nlcd_veg_file) as src:
            # Mask the vegetation raster data with the buffer geometry
            veg_masked_data, veg_masked_transform = mask(src, [buffer_geom], crop=True, filled=False)
            veg_masked_array = veg_masked_data[0]

            # Handle masked values
            veg_masked_array = np.where(veg_masked_array.mask, -1, veg_masked_array.data)
            
            # Group vegetation classes
            veg_groups = group_vegetation_classes(veg_masked_array)
            
            # Calculate vegetation metrics and CAI
            for veg_type, veg_mask in veg_groups.items():
                veg_pixels = np.sum(veg_mask)
                metrics[f'pct_area_{veg_type}'] = (veg_pixels / total_pixels) * 100
                metrics[f'area_km_{veg_type}'] = (veg_pixels * 30 * 30) / 1000000
                
                # Calculate CAI for vegetation groups
                try:
                    if veg_pixels > 0:
                        # Create binary raster for this vegetation type (1 where vegetation type exists, 0 elsewhere)
                        veg_binary = veg_mask.astype(int)
                        veg_ls = pls.Landscape(veg_binary, res=(30, 30))
                        cai_veg = veg_ls.core_area_index(class_val=1, edge_depth=1, percent=True)
                        metrics[f'cai_{veg_type}'] = cai_veg.iloc[0] if isinstance(cai_veg, pd.Series) else cai_veg
                    else:
                        metrics[f'cai_{veg_type}'] = 0
                except Exception as e:
                    print(f"Warning: Could not calculate CAI for {veg_type} in ID {original_id}: {e}")
                    metrics[f'cai_{veg_type}'] = np.nan

            buffer_metrics.append(metrics)

            # Progress reporting
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(buffered_points_raster_crs)} buffers")

    except Exception as e:
        print(f"Error processing buffer {idx} (ID: {original_id}): {e}")
        buffer_metrics.append({
            'ID': original_id,
            'peak_date': peak_date,
            'nlcd_year': np.nan,
            'total_area_km2': np.nan,
            'pct_area_1': np.nan,
            'pct_area_2': np.nan,
            'area_km_1': np.nan,
            'area_km_2': np.nan,
            'cai_1': np.nan,
            'cai_2': np.nan,
            'pct_area_forest': np.nan,
            'pct_area_shrubland': np.nan,
            'pct_area_herbaceous': np.nan,
            'pct_area_planted': np.nan,
            'pct_area_wetlands': np.nan,
            'area_km_forest': np.nan,
            'area_km_shrubland': np.nan,
            'area_km_herbaceous': np.nan,
            'area_km_planted': np.nan,
            'area_km_wetlands': np.nan,
            'cai_forest': np.nan,
            'cai_shrubland': np.nan,
            'cai_herbaceous': np.nan,
            'cai_planted': np.nan,
            'cai_wetlands': np.nan
        })

# Create final DataFrame and save results
multi_year_metrics_df = pd.DataFrame(buffer_metrics)
print(f"\nCompleted processing {len(buffer_metrics)} buffers")
print(f"Years used: {multi_year_metrics_df['nlcd_year'].value_counts().sort_index()}")
print("\nFirst few rows:")
print(multi_year_metrics_df.head())

# Save results
output_file = r"C:\Users\alekb\OneDrive - UCB-O365\Research\Flood-height-prediction\data\combined_nlcd_features.csv"
multi_year_metrics_df.to_csv(output_file, index=False)
print(f"Multi-year results with vegetation features saved to {output_file}")