# Flood Height Prediction Research Project

This repository contains a comprehensive machine learning pipeline for predicting flood heights using multi-sensor remote sensing data, environmental variables, and advanced spatial analysis techniques.

## Project Overview

The project is a sophisticated flood height prediction system that integrates multiple remote sensing datasets, environmental variables, and advanced machine learning techniques. It consists of five main components:

1. **Enhanced Feature Extraction**: Extracts comprehensive landscape metrics (including vegetation analysis), topographic features, SAR signatures, precipitation data, and soil moisture characteristics around flood locations
2. **Target Generation**: Calculates precise flood depths (height_above) from USGS High Water Mark elevations and DEM data
3. **Advanced Data Processing**: Handles temporal alignment, coordinate system transformations, multi-sensor data fusion, and robust missing data imputation
4. **Multi-Modal Machine Learning**: Implements both traditional ensemble methods (Random Forest) and novel multi-route neural networks with attention-based fusion for flood depth prediction
5. **Comprehensive Analysis**: Provides extensive model diagnostics, residual analysis, and feature importance evaluation

## Repository Structure

### Folders
- **`data/`** - Contains processed feature datasets and analysis outputs
- **`routing/`** - Machine learning routing algorithms and training pipelines  
- **`notebooks/`** - Jupyter notebooks for data processing and feature extraction
- **`figures/`** - Output visualizations and clustering analysis results
- **`Mapping/`** - GIS project files and geospatial data

## Code Files Overview

| File | Last Updated | Description |
|------|-------------|-------------|
| **notebooks/NLCD_Extract_v0.1.0.py** | 2025-08-08 | **ENHANCED**: Main script for extracting comprehensive NLCD features including both impervious surface AND vegetation analysis. Creates 1.5km square buffers, processes multi-year data (2016-2024), calculates landscape metrics for 5 vegetation categories (forest, shrubland, herbaceous, planted, wetlands), and computes Core Area Index for all land cover types. Outputs 25 comprehensive features. |
| **data_exploration.ipynb** | 2025-08-08 | **NEW**: Renamed from Data_Joining.ipynb. Critical data pipeline component that merges all feature datasets with robust KNN imputation (k=3), performs UMAP dimensionality reduction and clustering analysis. Includes exploratory data analysis with comprehensive visualizations and outputs final combined dataset with 1147 samples and 60+ features. |
| **modelling.ipynb** | 2025-08-08 | **NEW**: Renamed from RF_Train_v0.1.0.ipynb. Production-ready Random Forest training pipeline for flood depth prediction with comprehensive hyperparameter tuning, cross-validation, feature importance analysis, residual analysis, and model diagnostics. Includes overfitting detection and extensive performance metrics. |
| **utils.py** | 2025-08-08 | **NEW**: Comprehensive utility functions for data processing, modeling, and visualization. Includes functions for dataset merging, cyclical datetime transformation, missing value imputation, train/test splitting, z-score normalization, regression metrics calculation, plotting utilities, and feature importance analysis. |
| **notebooks/soil_moisture_data_process.py** | 2025-08-08 | Extracts multi-temporal soil moisture and precipitation data from SMAP and CHIRPS datasets. Processes soil moisture at multiple time intervals (3hr, 6hr, 9hr, 12hr before flood events). |
| **notebooks/NLCD_Extract_v0.1.0_test.ipynb** | 2025-08-08 | Jupyter notebook for testing and prototyping the NLCD extraction workflow. Contains experimental code and visualizations for buffer creation and metrics calculation. |
| **notebooks/dem_features.ipynb** | 2025-08-08 | Jupyter notebook for extracting digital elevation model (DEM) features around flood points. Processes topographic characteristics and terrain metrics. |
| **notebooks/Calculate_HWM_Depth_v1.0.ipynb** | 2025-08-08 | Jupyter notebook for calculating High Water Mark (HWM) depth measurements. Processes flood depth data and generates target variable for machine learning models. |
| **notebooks/Sentinel_Features.ipynb** | 2025-08-08 | Jupyter notebook for extracting Sentinel-1 SAR features around flood points. Creates 1.5km square buffers, clips VV and VH polarization bands, and calculates backscatter statistics and ratios. |
| **notebooks/UMAP_Clustering_Inputs.ipynb** | 2025-08-08 | **NEW**: Specialized notebook for UMAP dimensionality reduction and clustering analysis of input features. Generates silhouette analysis, cluster visualizations, and outputs clustering results to figures/Input_Clustering_Results/. |
| **routing/routing.py** | 2025-08-08 | Core routing functionality for multi-route neural network architecture. Defines feature routing mappings and data preparation pipelines for ML models. |
| **routing/gs_routes.py** | 2025-08-08 | Grid search implementation for testing all possible feature-route combinations in the routing classifier. Generates and evaluates different routing strategies. |
| **routing/training.py** | 2025-08-08 | Training pipeline for routing-based neural network models. Includes model training, validation, and dummy data generation functions. |
| **notebooks/data_process.ipynb** | 2025-08-08 | Data processing notebook for extracting precipitation and soil moisture features from CHIRPS and SMAP datasets. Handles reprojection, temporal alignment with flood events, and spatial sampling at flood point locations. |
| **CLAUDE.md** | 2025-08-08 | Project documentation and guidance for Claude Code AI assistant. Contains architecture overview, dependencies, and development notes. |
| **README.md** | 2025-08-08 | This file - comprehensive project documentation including file descriptions, usage instructions, and technical details. |

### Major Changes Summary
- **v0.1.0 (Initial)**: Multi-year NLCD processing, buffer creation, landscape metrics
- **v0.2.0 (Enhanced Features)**: Added comprehensive vegetation analysis with 5 parent classes, Core Area Index calculations for all land cover types, expanded from 9 to 25 NLCD features
- **v0.3.0 (Data Pipeline)**: Implemented robust data joining pipeline with KNN imputation, column name preservation, temporal feature engineering, and multi-dataset fusion
- **v0.4.0 (ML Pipeline)**: Added multi-temporal soil moisture processing, weather data integration, Random Forest with hyperparameter tuning, and comprehensive model diagnostics
- **v0.5.0 (August 2025 - Code Organization)**: Major code restructuring with notebooks moved to notebooks/ directory, file renaming for clarity (data_exploration.ipynb, modelling.ipynb), creation of comprehensive utils.py module, addition of UMAP clustering analysis, and enhanced visualization capabilities
- **Latest (August 2025)**: Complete end-to-end pipeline with 1147 samples, 60+ features, advanced imputation, residual analysis, clustering analysis, and production-ready ML models with comprehensive utilities

## Input and Output Files

### Input Data Sources

#### External Input Files (Not in Repository)
| File | Size | Used By | Source | Description |
|------|------|---------|--------|-------------|
| **filtered_flooding_points_over_one_day*.shp** | N/A | NLCD_Extract_v0.1.0.py, Calculate_HWM_Depth_v1.0.ipynb, Sentinel_Features.ipynb | USGS | High water mark points with ID, peak_date, elev_ft, and geometry attributes |
| **Annual_NLCD_ImpDsc_{YEAR}_CU_C1V1.tif** | ~4GB each | NLCD_Extract_v0.1.0.py | USGS National Land Cover Database | Annual impervious surface data (2016-2024), 30m resolution |
| **Annual_NLCD_LndCov_{YEAR}_CU_C1V1.tif** | ~4GB each | NLCD_Extract_v0.1.0.py | USGS National Land Cover Database | **NEW**: Annual land cover/vegetation data (2016-2024), 30m resolution, used for vegetation analysis |
| **{ID}_USGS_3DEP_{YEAR}.tif** | Varies | Calculate_HWM_Depth_v1.0.ipynb, dem_features.ipynb | USGS 3DEP | Individual DEM raster files named by flood point ID and year, 10m resolution |
| **{ID}_S1_*.tif** | Varies | Sentinel_Features.ipynb | Google Earth Engine/ESA | Sentinel-1 SAR imagery for flood points with VV and VH polarizations, processed from GEE |
| **CHIRPS_1day_ahead/*.tif** | Varies | data_process.ipynb | CHIRPS/UCSB | Climate Hazards Group InfraRed Precipitation with Station data - 1-day ahead precipitation forecasts, 0.05° resolution |
| **Flooding_SMAP9KM_{TIME}/*.tif** | Varies | data_process.ipynb | NASA SMAP | Soil Moisture Active Passive (SMAP) 9km soil moisture data at various time intervals before flood events (3hr, 6hr, 9hr, 12hr) |
| **flooding_dataset_with_precipitation_0801_with_height_above.json** | N/A | dem_features.ipynb, data_process.ipynb | Project metadata | JSON file containing flood point metadata including Sentinel-1 data availability flags and temporal alignment information |

#### File Path Configurations
The project supports both Windows and Linux/cluster environments:
- **Windows paths**: `C:\Users\alekb\Downloads\`
- **Linux/Anvil cluster paths**: `/anvil/projects/x-cis250634/team5/`

### Output Data Files (In Repository)

#### `data/` Directory
| File | Size | Generated By | Description |
|------|------|-------------|-------------|
| **combined_nlcd_features.csv** | 1.2 MB | NLCD_Extract_v0.1.0.py | **ENHANCED**: Comprehensive NLCD features including impervious surface AND vegetation analysis (25 features total) with Core Area Index calculations for all land cover types |
| **combined_features.csv** | 1.5 MB | data_exploration.ipynb | **MAIN DATASET**: Final merged dataset with KNN imputation, 1147 samples, 60+ features including all spatial, temporal, and environmental variables |
| **imp_surface_features.csv** | 127 KB | NLCD_Extract_v0.1.0.py | **LEGACY**: Original NLCD impervious surface metrics (replaced by combined_nlcd_features.csv) |
| **dem_features.csv** | 89 KB | dem_features.ipynb | DEM-derived topographic features including elevation statistics and terrain metrics at 10m resolution |
| **dem_1m_features.csv** | 89 KB | dem_features.ipynb | **NEW**: High-resolution DEM features at 1m resolution for detailed topographic analysis |
| **HWM_Depth_m.csv** | 2 KB | Calculate_HWM_Depth_v1.0.ipynb | **Target variable**: High water mark depth measurements (height_above) in meters for ML models |
| **USGS_HWM_Height.csv** | 15 KB | External/Manual processing | **NEW**: Raw USGS High Water Mark elevation data used for target variable calculation |
| **soil_moisture.csv** | 126 KB | soil_moisture_data_process.py | **NEW**: Multi-temporal SMAP soil moisture data (3hr, 6hr, 9hr, 12hr before flood events) |
| **weather_data.csv** | 185 KB | External processing | **NEW**: Comprehensive weather features (temperature, humidity, wind, pressure, heat index) |
| **Coordinates.csv** | 89 KB | dem_features.ipynb | **NEW**: Geographic coordinates for all flood points with CRS information |
| **precipitation.csv** | 6 KB | External source | CHIRPS precipitation data with 1-day ahead forecasts |
| **sentinel1_combined_features.csv** | 192 KB | Sentinel_Features.ipynb | Sentinel-1 SAR backscatter features (VV/VH statistics and ratios) |
| **rf_feature_importance.csv** | 5 KB | modelling.ipynb | **NEW**: Random Forest feature importance rankings for model interpretability |
| **rf_residuals.csv** | 125 KB | modelling.ipynb | **NEW**: Comprehensive residual analysis with point IDs for error diagnosis |
| **rf_feature_importance_wcat.csv** | 5 KB | modelling.ipynb | **NEW**: Feature importance categorized by data modality (Land Cover, Weather, Sentinel, DEM, etc.) |
| **rf_residuals.xlsx** | 125 KB | modelling.ipynb | **NEW**: Excel version of comprehensive residual analysis for easy viewing and filtering |

#### `figures/` Directory
| File | Size | Generated By | Description |
|------|------|-------------|-------------|
| **Input_Clustering_Results/silhouette_scores.csv** | 1 KB | data_exploration.ipynb | Silhouette scores for clustering analysis (k=2 to k=10) |
| **Input_Clustering_Results/silhouette_scores.png** | 50 KB | data_exploration.ipynb | Silhouette score visualization showing optimal cluster count |
| **Input_Clustering_Results/umap_k{2-10}.png** | 200 KB each | data_exploration.ipynb | UMAP scatter plots colored by cluster assignments for each k value |
| **Input_Clustering_Results/umap_embedding_and_labels.csv** | 150 KB | data_exploration.ipynb | UMAP 2D embeddings with cluster labels for all k values and original IDs |
| **3DVIS_1.jpg, 3DVIS_2.jpg, 3DVIS_3.jpg** | Various | Mapping analysis | 3D visualization exports from GIS analysis


### Feature Columns

#### combined_nlcd_features.csv (Enhanced NLCD Dataset)
**Impervious Surface Features:**
- `ID`: Flood point identifier
- `peak_date`: Original flood event date  
- `nlcd_year`: NLCD data year used for analysis
- `total_area_km2`: Total buffer area in km² (typically 9.1809 km²)
- `pct_area_1`, `pct_area_2`: % of buffer with impervious surface (low/medium intensity)
- `area_km_1`, `area_km_2`: Absolute areas (km²) for each impervious class
- `cai_1`, `cai_2`: Core Area Index for impervious surface fragmentation

**Vegetation Features (NEW):**
- `pct_area_forest`, `pct_area_shrubland`, `pct_area_herbaceous`, `pct_area_planted`, `pct_area_wetlands`: Percentage coverage of each vegetation type
- `area_km_forest`, `area_km_shrubland`, `area_km_herbaceous`, `area_km_planted`, `area_km_wetlands`: Absolute areas (km²) for each vegetation type  
- `cai_forest`, `cai_shrubland`, `cai_herbaceous`, `cai_planted`, `cai_wetlands`: Core Area Index for each vegetation type (landscape fragmentation metrics)

#### combined_features.csv (Main ML Dataset)
**Contains all features from above datasets plus:**
- Temporal features: `year`, `month_sin`, `day_sin`, `hour_sin` (cyclical encoding)
- DEM features: `dem_min`, `dem_max`, `dem_mean`, `dem_iqr` (topographic characteristics)
- Weather features: Temperature, humidity, wind speed, pressure, heat index, feels-like temperature
- Soil moisture: Multi-temporal SMAP data at various time intervals
- SAR features: Sentinel-1 VV/VH backscatter statistics and ratios
- **Target variable**: `height_above` (flood depth in meters)

#### dem_features.csv
- `file_id`: Flood point identifier matching ID in other datasets
- `year`: Year of DEM data used (includes '.tif' extension)
- `dem_min`: Minimum elevation value (meters) within the analysis area
- `dem_max`: Maximum elevation value (meters) within the analysis area
- `dem_mean`: Mean elevation value (meters) within the analysis area
- `dem_iqr`: Interquartile range of elevation values (meters) - measure of topographic variability
- `projection`: Coordinate reference system used (EPSG:4326 - WGS84 geographic)

#### HWM_Depth_m.csv
- `ID`: Flood point identifier matching other datasets
- `Year`: Year of the flood event (integer format)
- `height_above`: **UPDATED**: High water mark depth in meters above ground surface, calculated as (flood elevation - ground elevation from DEM). This is the target variable for ML models.

#### precipitation.csv
- `ID`: Flood point identifier matching other datasets
- `S1_Date`: Sentinel-1 acquisition date for the flood event
- `precipitation`: Precipitation value (units to be confirmed)

#### sentinel1_combined_features.csv
- `ID`: Flood point identifier matching other datasets
- `peak_date`: Original flood event date and time
- `VV_Min`: Minimum VV polarization backscatter value (dB) within the 1.5km buffer
- `VV_Max`: Maximum VV polarization backscatter value (dB) within the buffer
- `VV_Mean`: Mean VV polarization backscatter value (dB) within the buffer
- `VV_IQR`: Interquartile range of VV backscatter values (dB)
- `VV_SD`: Standard deviation of VV backscatter values (dB)
- `VH_Min`: Minimum VH polarization backscatter value (dB) within the buffer
- `VH_Max`: Maximum VH polarization backscatter value (dB) within the buffer
- `VH_Mean`: Mean VH polarization backscatter value (dB) within the buffer
- `VH_IQR`: Interquartile range of VH backscatter values (dB)
- `VH_SD`: Standard deviation of VH backscatter values (dB)
- `VH_VV_Ratio`: Ratio of VH mean to VV mean backscatter values, useful for surface roughness and flood detection


## Dependencies

### Core Geospatial Libraries
- `rasterio` - Geospatial raster data I/O and processing, windowed reading, CRS transformations
- `geopandas` - Vector geospatial data handling, shapefile I/O, geometry operations
- `pylandstats` - Landscape ecology metrics calculation (Core Area Index, area percentages)
- `shapely` - Geometric operations, buffer creation, spatial analysis
- `pathlib` - Cross-platform file path handling (Python 3.4+)

### Data Science & ML Libraries  
- `pandas` - Data manipulation, CSV I/O, temporal data parsing
- `numpy` - Numerical computations, array operations, statistical functions
- `scipy` - Statistical functions (interquartile range calculation)
- `scikit-learn` - **ENHANCED**: Machine learning library with RandomForestRegressor, train_test_split, RandomizedSearchCV, cross-validation, and regression metrics
- `sklearn.impute` - **NEW**: Missing data imputation (KNNImputer and SimpleImputer for robust data preprocessing)
- `sklearn.model_selection` - Model selection tools for hyperparameter tuning and data splitting
- `sklearn.ensemble` - Ensemble methods including Random Forest algorithms
- `sklearn.metrics` - Performance evaluation metrics (MSE, MAE, R², silhouette_score, etc.)
- `sklearn.preprocessing` - **NEW**: Data preprocessing utilities (StandardScaler for feature normalization)
- `sklearn.cluster` - **NEW**: Clustering algorithms (AgglomerativeClustering for hierarchical clustering)
- `umap` - **NEW**: Uniform Manifold Approximation and Projection for dimensionality reduction and visualization
- `torch` - PyTorch deep learning framework for neural networks
- `torch.nn` - Neural network layers and architectures
- `torch.optim` - Optimization algorithms for model training
- `torch.utils.data` - Data loading utilities (DataLoader, TensorDataset)

### Visualization & Progress Libraries
- `matplotlib` - Data visualization, plotting, histograms
- `matplotlib.pyplot` - Plotting interface for visualizations
- `tqdm` - Progress bar displays for long-running processes

### Standard Python Libraries
- `os` - File system operations, path manipulation
- `sys` - System-specific parameters and functions
- `glob` - Unix-style pathname pattern expansion
- `json` - JSON data parsing and manipulation
- `warnings` - Warning control and management
- `datetime` - Date and time manipulation for temporal data


## Complete Machine Learning Pipeline

### Data Processing Pipeline
1. **Enhanced Feature Extraction**: NLCD vegetation analysis (25 features), multi-resolution DEM processing, SAR analysis, multi-temporal soil moisture, comprehensive weather data
2. **Robust Data Integration**: KNN imputation with column name preservation, temporal feature engineering, multi-dataset fusion in data_exploration.ipynb
3. **Exploratory Data Analysis**: UMAP dimensionality reduction, clustering analysis, and comprehensive visualizations
4. **Quality Control**: Missing data handling, coordinate system transformations, temporal alignment across all datasets

### Random Forest Implementation (modelling.ipynb)
**Production-ready ML pipeline** with comprehensive flood depth prediction capabilities:

#### Enhanced Model Features:
- **Advanced multi-sensor fusion**: Integrates 60+ features from NLCD (vegetation + impervious), DEM (10m + 1m), Sentinel-1 SAR, multi-temporal soil moisture, weather, and precipitation data
- **Robust preprocessing**: KNN imputation, z-score standardization, proper data splitting (70%/15%/15%)
- **Comprehensive hyperparameter optimization**: RandomizedSearchCV with 5-fold cross-validation and 100 iterations
- **Advanced diagnostics**: Overfitting detection, residual analysis with point IDs, feature importance rankings
- **Target variable**: Uses `height_above` (flood depth in meters) consistently across the pipeline

#### Model Performance & Insights:
- **Comprehensive metrics**: MSE, RMSE, MAE, R² score across train/validation/test splits
- **Feature importance**: Top features include geographic coordinates, weather variables (feels-like temp, heat index), and landscape metrics
- **Model diagnostics**: Identifies overfitting tendencies, provides feature selection guidance (80%/90% importance thresholds)
- **Residual analysis**: Comprehensive error analysis with point-level diagnostics for model improvement

#### Advanced Capabilities:
- **Multi-route neural networks**: Alternative architecture with attention-based feature fusion (routing/ modules)
- **Extensive hyperparameter search**: Grid search across all feature-route combinations
- **Publication-ready outputs**: Feature importance rankings, residual analysis, model performance visualizations
- **Reproducible results**: Comprehensive logging, version control, and standardized evaluation metrics

### Clustering and Dimensionality Reduction (data_exploration.ipynb)
**Advanced exploratory data analysis** with unsupervised learning techniques:

#### UMAP Implementation:
- **Dimensionality Reduction**: Reduces 60+ features to 2D embeddings for visualization using UMAP (Uniform Manifold Approximation and Projection)
- **Parameter Configuration**: Uses n_neighbors=15, min_dist=0.1, euclidean distance metric for optimal flood data representation
- **Preprocessing**: Includes median imputation, standard scaling, and automatic removal of non-numeric and all-NaN columns

#### Hierarchical Clustering:
- **Agglomerative Clustering**: Tests cluster counts from k=2 to k=10 using Ward linkage on UMAP embeddings
- **Silhouette Analysis**: Comprehensive evaluation of cluster quality with silhouette scores for optimal k selection
- **Visualization Suite**: Generates scatter plots for each k value with color-coded cluster assignments

#### Outputs and Analysis:
- **Comprehensive Results**: Saves silhouette scores, UMAP embeddings, cluster labels, and visualizations to figures/Input_Clustering_Results/
- **Data Structure Discovery**: Identifies natural groupings in flood event characteristics for better understanding of data heterogeneity
- **Feature Space Exploration**: Provides insights into the multivariate relationships between environmental, topographic, and weather features

## Notes

### Performance Considerations


### Data Quality & Processing Notes
- **Enhanced Imputation**: KNN imputation (k=3) handles missing data while preserving column names and data structure
- **Temporal Alignment**: All datasets temporally aligned to flood events with proper date handling and year matching
- **Coordinate Systems**: Robust CRS transformations between geographic (EPSG:4326) and projected (EPSG:5070) coordinates
- **Vegetation Analysis**: 5 parent vegetation classes with Core Area Index calculations for landscape fragmentation assessment
- **Multi-resolution Processing**: Both 10m and 1m DEM analysis for comprehensive topographic characterization
- **Target Variable Consistency**: Standardized `height_above` naming throughout the entire pipeline

### Advanced Capabilities
- **Comprehensive Feature Set**: 60+ features from multiple remote sensing sources and environmental datasets
- **Production Pipeline**: End-to-end processing from raw data to model predictions with comprehensive diagnostics
- **Model Interpretability**: Feature importance analysis, residual diagnostics, and performance visualization
- **Scalable Architecture**: Modular design supports both traditional ML (Random Forest) and advanced neural networks
- **Research Quality**: Publication-ready outputs with comprehensive evaluation metrics and statistical analysis

### Development & Deployment Notes
- **Cross-platform Support**: Configured for both Windows and Linux/cluster environments  
- **Reproducible Research**: Comprehensive logging, version control, and standardized evaluation protocols
- **Error Resilience**: Robust error handling maintains processing continuity across large datasets
- **Extensible Design**: Modular architecture allows easy integration of additional data sources and ML models
