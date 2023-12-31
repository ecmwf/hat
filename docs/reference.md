## Reference

### Command Line Tools

#### `hat-extract-timeseries`

Extract timeseries from a collection of simulation raster files. Timeseries extraction requires a json configuration file, which can be provided on the command line using `--config`

    $ hat-extract-timeseries --config timeseries.json

You can show an example of config file using `--show-default-config`

    $ extract_simulation_timeseries --show-default-config

To create your own configuration json file you might want to start with the default configuration as a template. Default values will be used where possible if not defined in the custom figuration file. Here is an example custom configuration file.

    # example custom configuration .json file
    {
        "station_metadata_filepath": "/path/to/stations.csv",
        "simulation": {
            "type": "file",
            "files": "*.nc"
        },
        "simulation_output_filepath": "./simulation_timeseries.nc",
        "station_epsg": 4326,
        "station_id_column_name": "obsid",
        "station_filters":"",
        "station_coordinates": ["Lon", "Lat"]
    }

#### `hat-hydrostats`

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

`hat-hydrostats --functions kge, rmse, mae, correlation --sims $SIMS --obs $OBS`

(Optionally) define the minimum percentage of observations required for timeseries to be valid using the `--obs_threshold` option (default is 80%)

`hat-hydrostats --functions kge --sims $SIMS --obs $OBS --obs_threshold 70`
