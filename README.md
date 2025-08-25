# Flood Height Prediction

Machine learning pipeline for predicting flood depths using remote sensing and environmental data.

## Overview

This project predicts flood heights by combining:
- Landscape and vegetation metrics from satellite data
- Topographic features from elevation models  
- Weather and soil moisture data
- SAR imagery from Sentinel-1

The pipeline extracts features around flood locations, processes multiple data sources, and trains ML models to predict flood depths.

## Repository Structure

- **`data/`** - Processed datasets and model outputs
- **`routing/`** - Neural network routing algorithms  
- **`notebooks/`** - Feature extraction scripts
- **`figures/`** - Visualizations and analysis results

## Main Files

- **`data_exploration.ipynb`** - Data merging, imputation, and UMAP clustering
- **`modelling.ipynb`** - Random Forest training and evaluation
- **`utils.py`** - Data processing and visualization utilities
- **`notebooks/NLCD_Extract_v0.1.0.py`** - Extract landscape features from satellite data
- **`notebooks/Calculate_HWM_Depth_v1.0.ipynb`** - Calculate flood depth targets
- **`notebooks/Sentinel_Features.ipynb`** - Extract SAR imagery features
- **`notebooks/dem_features.ipynb`** - Extract topographic features

## Usage

Run the main pipeline:
```bash
python notebooks/NLCD_Extract_v0.1.0.py  # Extract landscape features
jupyter notebook data_exploration.ipynb  # Merge and explore data  
jupyter notebook modelling.ipynb         # Train models
```

## Data

The `data/` folder contains processed datasets:
- `combined_features.csv` - Main dataset (1147 samples, 60+ features)
- `combined_nlcd_features.csv` - Landscape and vegetation metrics
- `HWM_Depth_m.csv` - Flood depth targets
- `dem_features.csv` - Topographic features
- `sentinel1_combined_features.csv` - SAR imagery features
