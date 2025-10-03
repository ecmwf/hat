`hat-extract-timeseries` documentation
===============================

Extract timeseries from a collection of simulation raster files.

How to use
-----
Timeseries extraction requires a json configuration file, which can be provided on the command line using `--config`

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
