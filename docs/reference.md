## Reference

### Command Line Tools

#### `extract_timeseries`

Extract timeseries from a collection of gridded raster files. Timeseries extraction requires two variables 

1. simulation data directory 
2. station metadata filepath

They can be defined at the command line using `--simulation-datadir` and `--station-metadata`

    $ extract_timeseries --simulation-datadir $GRIB_DATADIR --station-metadata $STATION_METADATA_FILEPATH

This will use the default configuration, which you can show using `--show-default-config`

    $ extract_timeseries --show-default-config

To use a custom configuration there is `--config`

    $ extract_timeseries --config $CUSTOM_CONFIG

💡 PRO TIP

To create your own configuration json file you could start with the default configuration as a template. If your custom figuration file is missing any values then default values will be used in their place. For example, you could create a configuration file to specify netcdf, rather than grib, as the input file format.
    
    # example of a short custom configuration .json file
    {
        "simulation_datadir": "/path/to/simulation_datadir",
        "station_metadata": "path/to/Qgis_World_outlet_20221102.csv",
        "input_file_extension": ".nc"
    }

#### `hydrostats`

Calculate hydrological statistics on timeseries. To run this analysis the following are required:

- `--functions` = names of statistical function(s)
- `--sims` = filepath to simulation file
- `--obs` = filepath to observation file

For example

`hydrostats --functions kge --sims $SIMS --obs $OBS`

These are the currently supported functions:

- apb
- apb2
- bias
- br
- correlation
- kge
- index_agreement
- mae
- ns
- nslog
- pc_bias
- pc_bias2
- rmse
- rsr
- vr

You can calculate more than one function at once using commas with the `--functions` option

`hydrostats --functions kge, rmse, mae, correlation --sims $SIMS --obs $OBS`

(Optionally) define the minimum percentage of observations required for timeseries to be valid using the `--obs_threshold` option (default is 80%)

`hydrostats --functions kge --sims $SIMS --obs $OBS --obs_threshold 70`
