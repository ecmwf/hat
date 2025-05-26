# Hydrological Analysis Toolkit (HAT)

<p align="center">
  <a href="https://github.com/ecmwf/codex/raw/refs/heads/main/ESEE">
    <img src="https://github.com/ecmwf/codex/raw/refs/heads/main/ESEE/foundation_badge.svg" alt="ECMWF Software EnginE">
  </a>
  <a href="https://github.com/ecmwf/codex/raw/refs/heads/main/Project Maturity">
    <img src="https://github.com/ecmwf/codex/raw/refs/heads/main/Project Maturity/emerging_badge.svg" alt="Maturity Level">
  </a>
  <a href="https://opensource.org/licenses/apache-2-0">
    <img src="https://img.shields.io/badge/Licence-Apache 2.0-blue.svg" alt="Licence">
  </a>
  <a href="https://github.com/ecmwf/hat/releases">
    <img src="https://img.shields.io/github/v/release/ecmwf/hat?color=purple&label=Release" alt="Latest Release">
  </a>
</p>

<p align="center">
  <!-- <a href="#quick-start">Quick Start</a>
  • -->
  <a href="#installation">Installation</a>
  •
  <a href="https://hydro-analysis-toolkit.readthedocs.io">Documentation</a>
</p>

> \[!IMPORTANT\]
> This software is **Emerging** and subject to ECMWF's guidelines on [Software Maturity](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity).

The Hydrological Analysis Toolkit (HAT) is a software suite for hydrologists working with simulated and observed river discharge. HAT performs data analysis on hydrological datasets, with its main features being:
- mapping station locations into hydrological model grids
- extraction of timeseries
- statistical analysis of hydrological timeseries

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

## Licence

```
Copyright 2023, European Centre for Medium Range Weather Forecasts.

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
granted to it by virtue of its status as an intergovernmental organisation
nor does it submit to any jurisdiction.
```
