from setuptools import find_packages, setup

setup(
    name="hat",
    version="0.4.0",
    author="ECMWF",
    description="ECMWF Hydrological Analysis Tools",
    packages=find_packages(where="hat"),
    package_dir={"": "hat"},
    package_data={"hat": ["config_json/*.json"]},
    include_package_data=True,
    install_requires=[],
    tests_require=[],
    # command line tool name to function mapping
    entry_points={
        "console_scripts": [
            "hat-extract-timeseries=tools.extract_simulation_timeseries_cli:main",
            "hat-hydrostats=tools.hydrostats_cli:main",
        ],
    },
)
