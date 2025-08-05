# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a flood height prediction research project that processes geospatial data for flood analysis. The project combines National Land Cover Database (NLCD) data with flood point data to extract landscape metrics and features for predictive modeling. It includes data processing pipelines, routing algorithms, and feature extraction capabilities.

## Code Architecture

### Main Scripts
- **NLCD_Extract_v0.1.0.py**: Main script for extracting landscape metrics from NLCD impervious surface data around flood points. Creates square buffers, processes multi-year NLCD data, and calculates area percentages and core area indices.
- **routing/gs_routes.py**: Graph search routing algorithms
- **routing/routing.py**: Core routing functionality 
- **routing/training.py**: Training pipeline for routing models

### Jupyter Notebooks
- **NLCD_Extract_v0.1.0_test.ipynb**: Testing notebook for NLCD extraction workflow
- **dem_features.ipynb**: Digital elevation model feature extraction

### Data Files
- **data/imp_surface_features.csv**: Extracted impervious surface features from NLCD analysis
- **data/dem_features.csv**: Digital elevation model derived features
- **NLCD_MultiYear_Stats.csv**: Multi-year NLCD statistics output

## Dependencies

The project requires the following Python libraries:
- `rasterio` - For reading and processing geospatial raster data
- `geopandas` - For vector geospatial data handling
- `pandas` - For data manipulation and analysis
- `numpy` - For numerical computations
- `pylandstats` - For landscape metrics calculation
- `shapely` - For geometric operations
- `os` - For file system operations
- `warnings` - For warning management

## Key Features

### NLCD Processing Pipeline
- Loads flood point shapefiles with temporal data
- Creates 1.5km square buffers around flood points
- Processes multi-year NLCD impervious surface data (2016-2024)
- Calculates landscape metrics including area percentages and core area index
- Handles CRS transformations and geographic projections
- Outputs structured CSV files with extracted features

### Data Processing Capabilities
- Automated file existence checking for NLCD datasets
- Robust date parsing and year extraction from flood event data
- Error handling for missing data and processing failures
- Progress tracking for large dataset processing

## File Paths and Configuration

The project uses Windows file paths and is configured for local development:
- Flood points: `C:\Users\alekb\Downloads\filtered_flooding_points_over_one_day_nopr.shp`
- NLCD data: `C:\Users\alekb\Downloads\Annual_NLCD_ImpDsc_[YEAR]_CU_C1V1.tif`
- Output: `C:\Users\alekb\OneDrive - UCB-O365\Research\Flood-height-prediction\data\`

Alternative Linux paths are commented out for cluster computing environments.

## Running the Code

### Main NLCD Extraction
```bash
python NLCD_Extract_v0.1.0.py
```

### Jupyter Notebooks
For interactive analysis and testing:
```bash
jupyter notebook NLCD_Extract_v0.1.0_test.ipynb
jupyter notebook dem_features.ipynb
```

### Prerequisites
- Ensure all required Python dependencies are installed
- Verify NLCD raster files exist in the specified download directory
- Confirm flood point shapefile is available
- Create output directory structure if it doesn't exist

## Development Notes

- Project uses EPSG:5070 (Albers Equal Area) for accurate distance-based calculations
- Buffer creation uses projected coordinates to ensure proper metric distances
- NLCD data is recoded (1->1, 2->2, everything else->0) for analysis
- Core area index calculations use 1-pixel edge depth
- Error handling preserves processing continuity for large datasets