
Station Mapping Library Documentation
=====================================

The `station_mapping` library is designed for mapping the location of hydrological station data onto the optimum grid cell location of a hydrological simulation result (netcdf).

The optimum grid cell location is searched through optimising the upstream area error and the cell distance(s) between that of the station and the grid cell. In this tool, user can define their acceptable area difference/ error using the parameter `max_area_difference` (%) and the maximum cell radius to search for this optimum grid using the parameter `max_neighboring_cell` (number of cells). The tool can also be parameterised to ignore further searching cells with insignificant upstream area difference, i.e. when the uspteam area of the nearest cell to the station is already deemed acceptable by defining `min_area_diff`(%). 

In conclusion, the tool only searches for grid cell with optimal upstream area between the user defined `min_area_diff` and `max_area_diff` that are within the `max_neighboring_cell` radius from the station location.


This library utilizes Python packages such as Pandas, GeoPandas, NumPy, and others to handle spatial data and perform geospatial calculations.

Installation
------------

Before using the `station_mapping` library, ensure you have all the required dependencies of HAT, see [installation.md](installation.md) for installation instruction.


Usage
-----
#### Station Mapping with Command Line
To use the `station_mapping` as command line, follow these steps:

1.  Prepare your data input: station data and grid data in the appropriate format. Station data should be in a CSV file, and grid data should be in a NetCDF file.
    
2.  Create a [JSON configuration](notebooks/examples/station_mapping_config_example.json) file specifying the paths to your data files, column names, and other relevant parameters.
    
3.  Run the `station_mapping.py` script with the path to your configuration file:
    
`./station_mapping.py path/to/your/config.json`


#### Station Mapping as python script, e.g. called within jupyter notebook or python file
1. Prepare your data input: station data and grid data in the appropriate format. Station data should be in a CSV file, and grid data should be in a NetCDF file.

2. Create a configuration dictionary

```
config = {
    # Netcdf information
    "upstream_area_file": "upArea.nc", #file path to netcdf of upstream area

    # Station Metadata CSV information
    "csv_file": "outlets.csv", #file path to csv station metadata
    "csv_lat_col": "StationLat", # column name for latitude (string)
    "csv_lon_col": "StationLon", # column name for longitude (string)
    "csv_station_name_col": "StationName", # column name for station  (string)
    "csv_ups_col": "DrainingArea.km2.Provider", # column name for metadata of upstream  (string)

    # Mapping parameters (3x)
    "max_neighboring_cells": 5, # Parameter 1: maximum radius to search for best cells (no. of cells)  
    "max_area_diff": 20, # Parameter 2: acceptable/ optimum upstream area difference (%)
    "min_area_diff": 0, # Parameter 3: minimum upstream area difference (%) between nearest grid and the station metadata

    # manual mapping as reference for evaluation (optional)
    "manual_lat_col": "LisfloodY",  # column name for latitude of manually mapped station (string)
    "manual_lon_col": "LisfloodX", # column name for longitude of manually mapped station (string)
    "manual_area": "DrainingArea.km2.LDD", # column name for area of manually mapped station (string)

    # if Output directory is provided, it will save the geodataframe outputs to geojson and csv readable by GIS or jupyter interactive
    # "out_directory": None # put none if you don't want to save the output
    "out_directory": "/home/dadiyorto/freelance/02_ort_ecmwf/dev/hat/hat/mapping/output"    
}
```
3. Run the `station_mapping` function with the config dictionary input and store result as dataframe (df)
Since in the above example, the out_directory is not empty/ None, i.e. hence geojson and csv output of the station mapping tool will be saved in the specified directory.

```
# import station mapping 
from hat.mapping.station_mapping import station_mapping 
# call station_mapping function and apply on the created config dictionary
df = station_mapping(config)
```

Process Overview
----------------

*   **Read and Validate Input Data**: Loads station metadata from the specified CSV file and grid data from the NetCDF file, applying any filters defined in the configuration.
*   **Nearest Grid Cell Search**: For each station, calculates the nearest grid cell based on latitude and longitude. 
*   **Optimum Grid Cell Search**: Searches each neighboring cell radius at a time until a specified maximum radius `max_neighboring_cells`,  for a grid cell that offers a closer match based on the upstream area difference until it reaches desired optimum of `max_area_diff`. When the optimum value of  `max_area_diff` is reached, the search will be stopped and the particular optimum cell location will be stored. It is also possible to ignore searching for optimum grid when the upstream area of nearest grid cell is already below or equal to `min_area_diff`.
*   **Upstream Area and Distance Calculation**: For both nearest and optimum grids found for each station, upstream area is calculated. Cell distance(s) from optimum grid to the stations grid (same as nearest grid) is calculated.
*   **Manual Mapping Output** (Optional): If manual mapping data is provided, it is also stored in to the station mapping result dataframe and later could be used as reference to compare automated mapping results to evaluate mapping performance. This can be done through `evaluation` module.
*   **Save Results**: If an output directory is specified, saves the processed data as GeoJSON and CSV files for further analysis or visualization. Otherwise it only returns result as dataframe.


Output
------

The following elements (column) will be written as dataframe as the expected `station_mapping` output.
```
return {
        # Station data
        'station_name': station[station_name_col],
        'station_lat': lat,
        'station_lon': lon,
        'station_area': station_area,

        # Near grid data
        'near_grid_lat_idx': lat_idx,
        'near_grid_lon_idx': lon_idx,
        'near_grid_lat': latitudes[lat_idx],
        'near_grid_lon': longitudes[lon_idx],
        'near_grid_area': near_grid_area,
        'near_grid_polygon': near_grid_polygon,

        # Optimum grid from search
        'optimum_grid_lat_idx': optimum_lat_idx,
        'optimum_grid_lon_idx': optimum_lon_idx,
        'optimum_grid_lat': latitudes[optimum_lat_idx],
        'optimum_grid_lon': longitudes[optimum_lon_idx],
        'optimum_grid_area': optimum_grid_area,
        'optimum_area_diff': optimum_area_diff,
        'optimum_distance_km': optimum_distance_km,
        'optimum_grid_polygon': optimum_grid_polygon,
    
        # Manually mapped variable
        'manual_lat' : manual_lat,
        'manual_lon' : manual_lon,
        'manual_lat_idx': manual_lat_idx,
        'manual_lon_idx': manual_lon_idx,
        'manual_area' : manual_area,
        }
```

if the "out_directory" in the `configuration` is specified, then the following files will be written in the directory:

1. `stations.geojson`: stations point vector in geojson (readable in GIS)
2. `near_grid.geojson`: nearest grid vector (readable in GIS)
3. `optimum_grid.geojson`: optimum grid vector (readable in GIS)
4. `stations2grid_optimum`: polyline connecting each station location to the optimum grid's centroid (readable in GIS)
5. `stations.csv`: the dataframe containing all the data column mentioned above in csv format. (readable in GIS)


Example in Jupyter notebook
---------------------------

For the implementation example of station mapping in Jupyter notebook, an example is created in [here](notebooks/examples/5a_station_mapping_evaluate.ipynb)
This configuration is based on DESTINE project, and you shall modify your netcdf and csv input file location accordingly.


