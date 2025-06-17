import yaml
import argparse

from .compute_hydrostats.stat_calc import stat_calc
from .extract_timeseries.extractor import extractor
from .station_mapping.mapper import mapper

def commandlineify(func):
    def wrapper():
        parser = argparse.ArgumentParser(description="Run tool with YAML config")
        parser.add_argument("config", help="Path to the YAML config file")
        args = parser.parse_args()
        confpath = args.config
        with open(confpath, "r") as file:
            config = yaml.safe_load(file)
        func(config)
    return wrapper

mapper_cli = commandlineify(mapper)
extractor_cli = commandlineify(extractor)
stat_calc_cli = commandlineify(stat_calc)