# HAT - Hydrological Analysis Toolkit

HAT is a suite of tools to perform data analysis on hydrological datasets.

The documentation can be found at https://hydro-analysis-toolkit.readthedocs.io


**DISCLAIMER**
This project is **BETA** and will be **Experimental** for the foreseeable future.
Interfaces and functionality are likely to change, and the project itself may be scrapped.
**DO NOT** use this software in any project/software that is operational.

### Installation

Clone source code repository

    $ git clone git@github.com:ecmwf-projects/hat.git

Create conda python environment

    $ cd hat
    $ conda env create -f environment.yml
    $ pip install .

### Usage

Start a hat environment

    $ conda activate hat
    
Run a command line tool

    $ hat-extract-timeseries --help

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

### Contributing

The main repository is hosted on GitHub, testing, bug reports and contributions are highly welcomed and appreciated:

https://github.com/ecmwf/hat

Please report [bug](https://github.com/ecmwf/hat/issues) reports or [pull-requests](https://github.com/ecmwf/hat/pulls) on [GitHub](https://github.com/ecmwf/hat)

We want your feedback, please e-mail: user-services@ecmwf.int

### License

Copyright 2023 European Centre for Medium-Range Weather Forecasts (ECMWF)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

In applying this licence, ECMWF does not waive the privileges and immunities
granted to it by virtue of its status as an intergovernmental organisation nor
does it submit to any jurisdiction.


### Citing

In publications, please use a link to this repository (https://github.com/ecmwf/hat) and its documentation (https://hydro-analysis-toolkit.readthedocs.io)