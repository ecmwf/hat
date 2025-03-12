# HAT - Hydrological Analysis Toolkit

The Hydrological Analysis Toolkit (HAT) is a software suite for hydrologists working with simulated and observed river discharge. HAT performs data analysis on hydrological datasets, with its main features being:
- mapping station locations into hydrological model grids
- extraction of timeseries
- statistical analysis of hydrological timeseries

The documentation can be found at https://hydro-analysis-toolkit.readthedocs.io

**DISCLAIMER**
This project is **BETA** and will be **Experimental** for the foreseeable future.
Interfaces and functionality are likely to change, and the project itself may be scrapped.
**DO NOT** use this software in any project/software that is operational.

### Installation

Clone source code repository

    $ git clone https://github.com/ecmwf/hat.git
    $ cd hat

Create and activate conda environment

    $ conda create -n hat python=3.10
    $ conda activate hat

For default installation, run

    $ pip install .

For a developer installation (includes linting and test libraries), run

    $ pip install -e .[dev]
    $ pre-commit install

If you only plan to run the tests, instead run

    $ pip install -e .[test]

If you plan to build a source and a wheel distribution, it is additionally required to run

    $ pip install build

### Usage

Run a command line tool

    $ hat-extract-timeseries --help

### Running the tests

Tests are stored in the `tests/` folder and can be run with

    $ pytest

### Deployment

To build a source and a wheel distribution, run

    $ python build

### Contributing

The main repository is hosted on [GitHub](https://github.com/ecmwf/hat). Testing, bug reports and contributions are highly welcomed and appreciated.

Please report [bug](https://github.com/ecmwf/hat/issues) reports or [pull-requests](https://github.com/ecmwf/hat/pulls) on [GitHub](https://github.com/ecmwf/hat).

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
