# Flood Height Prediction Research Project

This repository contains code and data for analyzing flood height prediction using National Land Cover Database (NLCD) impervious surface data and landscape metrics around flood gauge locations.

## Project Overview

The project extracts landscape metrics from NLCD impervious surface layers within 1.5km square buffers around flood gauge points. It performs multi-temporal analysis by matching flood events to their corresponding year's NLCD data (2016-2024).

## File Structure

### Input Data Paths

#### Flood Point Data
- **Local (Windows)**: `C:\Users\alekb\Downloads\filtered_flooding_points_over_one_day_nopr.shp`
- **Anvil Cluster**: `/anvil/projects/x-cis250634/team5/shp/filtered_flooding_points_over_one_day_nopr.shp`

#### NLCD Impervious Surface Data
- **Local (Windows)**: `C:\Users\alekb\Downloads\Annual_NLCD_ImpDsc_[YEAR]_CU_C1V1.tif`
- **Anvil Cluster**: `/anvil/projects/x-cis250634/team5/data_0718/Annual_NLCD_ImpDsc_[YEAR]_CU_C1V1.tif`

Available years: 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024

### Output Data
- **Results**: `data/imp_surface_features.csv`
- **Contains**: Landscape metrics for each flood point buffer

## File Contents

### Input Files

#### `filtered_flooding_points_over_one_day_nopr.shp`
Shapefile containing flood gauge points with attributes:
- `ID`: Unique identifier for each flood point
- `peak_date`: Date/time of peak flood event  
- `geometry`: Point geometry (lat/lon coordinates)
- Additional flood event metadata (height, quality flags, etc.)

#### `Annual_NLCD_ImpDsc_[YEAR]_CU_C1V1.tif`
Annual NLCD impervious surface raster files:
- **Resolution**: 30m pixels
- **Coverage**: Continental United States
- **Values**: 
  - 1 = Developed, Open Space (20-49% impervious)
  - 2 = Developed, Low Intensity (50-79% impervious)
  - Other values = Non-impervious surfaces
- **CRS**: Typically Albers Equal Area Conic

### Output Files

#### `imp_surface_features.csv`
Processed landscape metrics with columns:
- `ID`: Flood point identifier
- `peak_date`: Original flood event date
- `nlcd_year`: NLCD data year used for analysis
- `total_area_km2`: Total buffer area in km²
- `pct_area_1`: Percentage of buffer with impervious class 1
- `pct_area_2`: Percentage of buffer with impervious class 2
- `area_km_1`: Absolute area (km²) of impervious class 1
- `area_km_2`: Absolute area (km²) of impervious class 2
- `cai_1`: Core Area Index for class 1 (fragmentation metric)
- `cai_2`: Core Area Index for class 2 (fragmentation metric)

## Code Versions and Changes

| Version | Date | Major Changes | Notes |
|---------|------|---------------|-------|
| v0.1.0 | 2025-01-XX | Initial implementation | - Multi-year NLCD processing<br>- 1.5km square buffer creation<br>- Landscape metrics calculation<br>- Year-based file matching |
| v0.1.0-fix1 | 2025-01-XX | Fixed date parsing error | - Fixed `extract_year_from_date()` to handle "M/D/YYYY H:MM:SS AM/PM" format<br>- Fixed None value handling in year selection<br>- Eliminated "unsupported operand type" errors |
| v0.1.0-fix2 | 2025-01-XX | Improved CRS handling | - Explicit reprojection to EPSG:5070 for accurate meter-based buffers<br>- Enhanced CRS transformation logging<br>- Verified buffer creation in meters not degrees |
| v0.1.0-clean | 2025-01-XX | Code cleanup and path management | - Removed unnecessary imports<br>- Added dual path support (Windows/Anvil)<br>- Improved error handling and progress reporting<br>- Consolidated helper functions |

## Technical Details

### Processing Workflow
1. **Load flood points** from shapefile
2. **Extract years** from peak_date column using pandas datetime parsing
3. **Create 1.5km square buffers** around each point in projected coordinates (EPSG:5070)
4. **Match flood events** to corresponding year's NLCD data
5. **Extract raster data** within each buffer using spatial masking
6. **Recode NLCD values** (1→1, 2→2, others→0)
7. **Calculate landscape metrics** using pylandstats
8. **Export results** to CSV

### Key Features
- **Memory efficient**: Uses windowed reading to load only buffer-intersecting pixels
- **Multi-temporal**: Matches flood events to appropriate NLCD year
- **Accurate buffering**: Ensures buffers are created in meters using projected CRS
- **Robust error handling**: Gracefully handles missing data and processing errors
- **Cross-platform**: Supports both Windows and Linux/Anvil environments

### Dependencies
- `rasterio`: Geospatial raster data I/O
- `geopandas`: Vector geospatial data handling
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical operations
- `pylandstats`: Landscape metrics calculation
- `shapely`: Geometric operations

## Usage

1. **Configure file paths** for your environment (Windows vs Anvil)
2. **Ensure NLCD files are available** for the years present in your flood data
3. **Run the script**: `python NLCD_Extract_v0.1.0.py`
4. **Check output**: Results saved to `data/imp_surface_features.csv`

## Notes

- Processing time scales with number of flood points (~1200 points may take several hours)
- NLCD files are large (~4GB each) but only small portions are loaded per buffer
- Script automatically handles coordinate system transformations
- Missing NLCD years will use the closest available year
- Core Area Index represents landscape fragmentation (higher = less fragmented)