import os
import json
import pkg_resources
from pathlib import Path
import copy

from hat.cli import warning


def load_package_config(fname):
    "load configuration json files included in the pacakge through setup.py"
    resource_package = "hat"
    resource_path = os.path.join("config_json", fname)

    config_string = pkg_resources.resource_string(resource_package,
                                                  resource_path)
    config = json.loads(config_string.decode())

    return config


DEFAULT_CONFIG = load_package_config("timeseries.json")


def overwrite_config(config, key, value, required=True):
    "overwrite the configutation directionary if user provides keyvalue pair"

    # no config
    if not config:
        return

    # set key if value is given
    if value:
        config[key] = value

    # if key still not in config or value is none-like then warn user
    if required:
        if key not in config or not config[key]:
            warning(f"{key} required")
            return

    # valid config
    return config


def booleanify(config, key):
    "Parse string booleans into real booleans"

    if not config:
        return

    if config[key].lower() not in ["true", "false"]:
        warning(f'"{key}" configuration variable must be "True" or "False"')
        return

    if config[key].lower() == "true":
        config[key] = True
    else:
        config[key] = False

    return config


def valid_custom_config(custom_config: dict = {}):
    """Create a valid custom configutation dictionary,
    i.e. add missing key-value pairs using default values and
    remove key-value pairs that shouldn't be there"""

    if not custom_config:
        custom_config = DEFAULT_CONFIG

    # only keep keys that are in DEFAULT_CONFIG
    filtered_custom_config = {
        k: v
        for k, v in custom_config.items() if k in DEFAULT_CONFIG
    }

    # overwrite default_config values with custom values
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(filtered_custom_config)

    return config


def read_config(custom_config_filepath: str):
    """(optional) read config from a .json file.
    Will otherwise use default config and/or any user defined"""

    # check file exists
    if not os.path.exists(custom_config_filepath):
        print("Custom config file not found. Returning default configuration")
        return DEFAULT_CONFIG

    # check for .json extention
    if Path(custom_config_filepath).suffix.lower() != ".json":
        raise ValueError("Custom configuration file must be .json")

    # read config from json
    with open(custom_config_filepath) as file:
        custom_config = json.load(file)

    # parse boolean strings into real booleans
    for key in custom_config:
        if str(custom_config[key]).lower() in ["true", "false"]:
            config = booleanify(custom_config, key)

    # include custom config filepath to config dict
    custom_config["config_fpath"] = custom_config_filepath

    # valid configurations are complete
    config = valid_custom_config(custom_config)

    return config


def timeseries_config(simulation_datadir: str, station_metadata: str,
                      config_filepath: str):
    """Manage configuration settings for timeseries extraction.
    Priority order:
      1) command line arguments
      2) custom config file
      3) default config"""

    # custom or default configutation file
    if not config_filepath:
        config = DEFAULT_CONFIG
    else:
        config = read_config(config_filepath)

    # root directory for search of input files
    if not simulation_datadir and not config["simulation_datadir"]:
        raise ValueError(
            """Please provide a data directory of grib or netcdf
            simulation files for timeseries extraction"""
        )
    elif simulation_datadir:
        config["simulation_datadir"] = simulation_datadir

    # station metadata filepath
    if not station_metadata and not config["station_metadata_filepath"]:
        raise ValueError("Please provide a station metadata filepath")
    elif station_metadata:
        config["station_metadata_filepath"] = station_metadata

    # check output filepath is valid before running timeseries extraction
    config["simulation_output_file_extension"] = config[
        "simulation_output_filepath"].split(".")[-1]
    if config["simulation_output_file_extension"] not in ["nc", "csv"]:
        raise ValueError(
            "output_filepath must be .nc or .csv, please update config",
            config_filepath)

    return config
