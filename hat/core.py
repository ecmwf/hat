import earthkit.data as ekd
from earthkit.hydro._readers import find_main_var


def load_da(ds_config, n_dims):
    src_name = list(ds_config["source"].keys())[0]
    ds = (
        ekd.from_source(*ds_config["source"])
        .from_source(src_name, **ds_config["source"][src_name])
        .to_xarray(**ds_config.get("to_xarray_options", {}))
    )
    var_name = find_main_var(ds, n_dims)
    da = ds[var_name]
    return da, var_name
