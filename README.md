# Hydrological Analysis Tools (HAT)

HAT is a suite of tools to perform data analysis on hydrological datasets.
    
### Installation

Clone source code repository

    $ git clone git@github.com:ecmwf-projects/hat.git

Create conda python environment

    $ cd hat
    $ conda env create -f environment.yml

### Usage

Start a hat environment

    $ conda activate hat
    
Run a command line tool

    $ extract_timeseries --help

![extract_timeseries_v0 2 0_help](https://github.com/ecmwf-projects/hat/assets/16657983/73b7b481-8280-4ad1-85b3-76ef31813786)

Timeseries extraction requires two variables 

1) root directory of input files 
2) station metadata filepath

They can be defined at the command line using `--simulation-datadir` and `--station-metadata` (all other variables are automatically defined using the default configuration)

    $ extract_timeseries --simulation-datadir $GRIB_DATADIR --station-metadata $STATION_METADATA_FILEPATH

To inspect the default configuration use `--show-default-config`

    $ extract_timeseries --show-default-config
    
![Screenshot 2023-06-07 at 11 03 08](https://github.com/ecmwf-projects/hat/assets/16657983/2494ff99-bc44-46fe-86ee-8e90732e57b3)

To define a custom configuration use the `--config` option

    $ extract_timeseries --config $CUSTOM_CONFIG
    
To create your own configuration json file it is recommended to start with the default configuration as a template (see above). If your custom figuration file is missing any values then default values will be used in their place. For example, you could create a configuration file to specify netcdf, rather than grib, as the input file format.
    
    # example of a short custom configuration .json file
    {
        "simulation_datadir": "/path/to/simulation_datadir",
        "station_metadata": "path/to/Qgis_World_outlet_20221102.csv",
        "input_file_extension": ".nc"
    }
