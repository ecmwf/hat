import glob
import json
import os
import pathlib
import sys
from tempfile import TemporaryDirectory
from typing import List, Union

import earthkit.data
import geopandas as gpd
import humanize
import pandas as pd
import xarray as xr

"""filepaths"""


def valid_filepath(filepaths):
    """returns first valid filepath in list of filepaths"""

    for filepath in filepaths:
        if os.path.exists(filepath):
            return filepath


def get_tmpdir():
    """HPC friendly temporary directory
    defined by environment variable called TMPDIR"""

    # HPC has an environmented variable for the preferred
    # (i.e. safer) temporary directory
    if "TMPDIR" in os.environ:
        tmpdir = os.environ["TMPDIR"]

    # Otherwise return operating system default
    # (e.g. /tmp on a linux machine, etc.)
    else:
        tmpdir = TemporaryDirectory().name

    # Ensure the directory exists
    os.makedirs(tmpdir, exist_ok=True)

    return tmpdir


def get_tmp_filepath(
    filename_components: Union[List[str], str] = "file", extension=".txt"
) -> str:
    """HPC friendly temporary file path

    usage:

    # filepath with default extension
    fpath = get_tmp_filepath('filename')

    # filepath with specific extension
    fpath = get_tmp_filepath('filename', extension= '.png')

    # filepath from list of strings
    fpath = get_tmp_filepath(['a','list','of','strings'])

    """

    tmpdir = get_tmpdir()

    if isinstance(filename_components, str):
        fname = filename_components
    if isinstance(filename_components, list):
        fname = "_".join(filename_components)

    return os.path.join(tmpdir, f"{fname}{extension}")


def find_files(simulation_files):
    """Find files matching regex"""

    fpaths = glob.glob(simulation_files)

    if not fpaths:
        raise Exception(f"Could not find any file from regex {simulation_files}")
    else:
        print("Found following simulation files:")
        print(*fpaths, sep="\n")

    return fpaths


"""file information"""


def filesize(fpath, bytesize=False):
    """Given a filepath return the size of the file in human readable format"""

    # check exists
    if not os.path.exists(fpath):
        print("Does not exist: ", fpath)
        return

    # check is file
    if not os.path.isfile(fpath):
        print("Not a file", fpath)
        return

    size_in_bytes = os.path.getsize(fpath)

    if bytesize:
        return size_in_bytes
    else:
        return humanize.naturalsize(size_in_bytes)


def dirsize(simulation_datadir, bytesize=False):
    """given a root directory return total size of all files
    in a directory in human readable format"""

    if not os.path.exists(simulation_datadir) or not os.path.isdir(simulation_datadir):
        print("Not a directory", simulation_datadir)
        return

    size_in_bytes = 0

    for dirpath, __builtins__, filenames in os.walk(simulation_datadir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            size_in_bytes += os.path.getsize(fp)

    if bytesize:
        return size_in_bytes
    else:
        return humanize.naturalsize(size_in_bytes)


"""rasters"""


def raster_with_explicit_epsg(xarr, epsg: int = 4326):
    # define crs
    if not xarr.rio.crs:
        xarr.rio.write_crs(f"epsg:{epsg}", inplace=True)

    # reproject crs
    if xarr.rio.crs.to_epsg() != epsg:
        xarr = xarr.rio.reproject(epsg)

    return xarr


def river_network_from_filename(fname):
    """hack around lack of data management standards"""

    if fname == "fc_dis_hw4s_20191125_120h_flx.nc":
        return "Cama_6min"
    if fname == "fc_dis_hw4v_20191125_120h_flx.nc":
        return "Cama_3min"
    else:
        print(
            """Critical failure. Must resolve data management.
            River network must be known"""
        )
        sys.exit(1)


def raster_loader(fpath: str, epsg: int = 4326, engine=None):
    """load raster data into xarray with explicity CRS and fpath attribute"""

    # load raster as xarray
    xarr = xr.open_dataset(fpath, engine=engine)

    # explicit CRS
    xarr = raster_with_explicit_epsg(xarr, epsg)

    # rename 'x' and 'y' to 'lon' and 'lat' if epsg:4326
    if "x" in xarr.dims and epsg == 4326:
        xarr = xarr.rename({"x": "lon", "y": "lat"})

    # update attributes
    xarr.attrs["fpath"] = fpath
    xarr.attrs["fname"] = os.path.basename(fpath)
    xarr.attrs["experiment_name"] = pathlib.Path(fpath).stem
    xarr.attrs["file_extension"] = pathlib.Path(fpath).suffix
    xarr.attrs["river_network"] = river_network_from_filename(xarr.attrs["fname"])

    return xarr


def raster_loader_with_crs_fix(fpath: str, epsg: int = 4326):
    """sample data is missing metadata. GRIB has crs but no time,
    NETCDF has time but no crs.
    Is a very smelly solution. need to ensure INPUT data quality is better
    (i.e. has the required metadata: CRS, time)"""

    # crs from grib
    grb = xr.open_dataset(fpath.replace(".nc", ".grb"), engine="rasterio")
    crs = grb.rio.crs

    # discharge from netcdf (has time index)
    net = xr.open_dataset(fpath, engine="rasterio")
    dis = net.dis
    dis.rio.set_crs(crs)

    # reproject to 4326
    dis = dis.rio.reproject(epsg)
    dis = dis.rename({"x": "lon", "y": "lat"})

    # add attributes
    dis.attrs.update(net.attrs)
    dis.attrs["fpath"] = fpath
    dis.attrs["fname"] = os.path.basename(fpath)

    return dis


"""vectors"""


def is_csv(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower() == ".csv"


def read_csv_and_cache(fpath: str) -> gpd.GeoDataFrame:
    """read .csv file and cache to pickle"""

    # cache filepath
    cache_fname = os.path.splitext(os.path.basename(fpath))[0]
    cache_fpath = get_tmp_filepath(cache_fname, extension=".pickle")

    # use cache if it exists
    if os.path.exists(cache_fpath):
        gdf = pd.read_pickle(cache_fpath)

    # otherwise load from user defined filepath (and then cache)
    else:
        gdf = gpd.read_file(fpath)
        gdf.to_pickle(cache_fpath)

    return gdf


""" other data (e.g. non geospatial)"""


def read_json_to_dict(fpath: str) -> dict:
    """read json to dict"""

    if os.path.exists(fpath):
        with open(fpath) as file:
            data = json.load(file)
        return data
    else:
        return {}


def save_dataset_to_netcdf(ds: xr.Dataset, fpath: str):
    """jupyter notebook safe way of saving xarray datasets to netcdf"""

    if os.path.exists(fpath):
        os.remove(fpath)

    ds.to_netcdf(fpath)


def find_main_var(ds, min_dim=3):
    variable_names = [k for k in ds.variables if len(ds.variables[k].dims) >= min_dim]
    if len(variable_names) > 1:
        raise Exception("More than one variable in dataset")
    elif len(variable_names) == 0:
        raise Exception("Could not find a valid variable in dataset")
    else:
        var_name = variable_names[0]
    return var_name


def read_simulation_as_xarray(options):
    if options["type"] == "file":
        src_type = "file"
        # find station files
        files = find_files(
            options["files"],
        )
        args = [files]
    elif options["type"] in ["mars", "fdb"]:
        src_type = options["type"]
        args = [options["request"]]
    else:
        raise Exception(
            f"Simulation type {options['type']} not supported. Currently supported: file, mars, fdb"  # noqa: E501
        )

    # earthkit data file source
    fs = earthkit.data.from_source(src_type, *args)
    print(str(type(fs)))

    xarray_kwargs = {}
    if isinstance(fs, earthkit.data.readers.netcdf.fieldlist.NetCDFMultiFieldList):
        xarray_kwargs["xarray_open_mfdataset_kwargs"] = {"chunks": {"time": "auto"}}
    else:
        xarray_kwargs["xarray_open_dataset_kwargs"] = {"chunks": {"time": "auto"}}

    # xarray dataset
    print(xarray_kwargs)
    ds = fs.to_xarray(**xarray_kwargs)

    var = find_main_var(ds)

    da = ds[var]
    return da
