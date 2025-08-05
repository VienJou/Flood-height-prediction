# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a flood height prediction research project that uses National Land Cover Database (NLCD) data for analysis. The project appears to be in early development stages with geospatial data processing capabilities.

## Code Architecture

- **NLCD_Extract_v0.1.0.py**: Main script for extracting and processing NLCD (National Land Cover Database) data. Uses rasterio for geospatial raster data handling, pandas for data manipulation, and numpy for numerical operations.

## Dependencies

The project requires the following Python libraries:
- `rasterio` - For reading and processing geospatial raster data
- `pandas` - For data manipulation and analysis
- `numpy` - For numerical computations
- `os` - For file system operations

## Development Notes

- This is a research project focused on flood height prediction using geospatial data
- The main script is currently incomplete with placeholder variables (`Impervious` and `Flood_Points`)
- No formal dependency management file (requirements.txt, pyproject.toml) exists yet
- The project structure is minimal with a single Python script

## Running the Code

Since there are no configuration files or build scripts, run the main script directly:
```bash
python NLCD_Extract_v0.1.0.py
```

Note: Ensure all required dependencies are installed in your Python environment before running.