from setuptools import find_packages, setup

setup(
    name="hat",
    version="0.3.0",
    author="ECMWF",
    description="ECMWF Hydrological Analysis Tools",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={'hat': ['config_json/*.json']},
    include_package_data=True,
    install_requires=[],
    tests_require=[],
    # command line tool name to function mapping
    entry_points={
        "console_scripts": [
            "extract_simulations = "
            "command_line_tools.extract_simulation_timeseries_cli:main",
            "river_networks = "
            "command_line_tools.river_network_coordinate_names:main",
            "hydrostats = "
            "command_line_tools.hydrostats_cli:main",
        ],
    },
)
