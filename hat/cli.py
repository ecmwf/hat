import yaml
import argparse
import sys

from hat import _LOGGER as logger
from hat.compute_hydrostats.stat_calc import stat_calc
from hat.extract_timeseries.extractor import extractor
from hat.station_mapping.mapper import mapper

def commandlineify(func):
    def wrapper(args=None):
        if args is None:
            args = sys.argv[1:]
        parser = argparse.ArgumentParser(description="Run tool with YAML config")
        parser.add_argument("config", help="Path to the YAML config file")
        args = parser.parse_args(args)
        confpath = args.config
        with open(confpath, "r") as file:
            config = yaml.safe_load(file)
        func(config)
    return wrapper

mapper_cli = commandlineify(mapper)
extractor_cli = commandlineify(extractor)
stat_calc_cli = commandlineify(stat_calc)


if __name__ == "__main__":
    from importlib.metadata import entry_points
    eps = entry_points().select(group="console_scripts")
    tools = {ep.name: ep.load() for ep in eps if ep.module.startswith("hat.")}
    tool_name = sys.argv[1]
    if tool_name in tools:
        tools[tool_name](sys.argv[2:])
    else:
        logger.error(f"Tool '{tool_name}' not found. Available tools: {', '.join(tools.keys())}")
        sys.exit(1)
