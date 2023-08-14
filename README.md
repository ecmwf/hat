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

### Contributing

The main repository is hosted on [GitHub](https://github.com/ecmwf/hat), testing, bug reports and contributions are highly welcomed and appreciated.

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