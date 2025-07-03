import abc
import yaml
import datetime

import pandas as pd
import xarray as xr

import earthkit.data as ekd

from hat.core.time import Time, ForecastFromBaseTimeAndStep


class Source(abc.ABC):

    @abc.abstractmethod
    def load(self, time: Time) -> xr.Dataset:
        """Load the dataset for the given time."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class MarsSource(Source):
    def __init__(self, request: dict):
        self._request = request

    def format_mars_request(self, time: ForecastFromBaseTimeAndStep) -> dict:
        for key, val in self._request.items():
            if isinstance(val, str):
                context = time.mars_keys()

                for context_key, context_val in context.items():
                    if context_key in val:
                        if isinstance(context_val, (list, pd.Index)):
                            self._request[key] = list(
                                set(
                                    [
                                        val.format(**{context_key: v})
                                        for v in context_val
                                    ]
                                )
                            )
                        else:
                            self._request[key] = val.format(
                                **{context_key: context_val}
                            )
        print(self._request)
        return self._request

    def load(self, time: ForecastFromBaseTimeAndStep) -> xr.Dataset:
        mars_request = self.format_mars_request(time)
        source = ekd.from_source("mars", mars_request)
        return source.to_xarray(
            time_dim_mode="valid_time", rename_dims={"valid_time": "time"}
        )


if __name__ == "__main__":
    with open("hat/core/test.yml", "r") as file:
        request = yaml.safe_load(file)

    base_time = datetime.datetime.strptime(
        str(request["time"]["base_time"]), "%Y%m%d%H"
    )
    base_time = pd.Timestamp(base_time)
    steps = request["time"]["steps"]
    steps = pd.to_timedelta(steps, unit="h")
    time = ForecastFromBaseTimeAndStep(base_time=base_time, steps=steps)

    mars_request = request["datasets"]["fc"]["request"]
    source = MarsSource(mars_request)
    ds_fc = source.load(time)

    mars_request = request["datasets"]["reanalysis"]["request"]
    source = MarsSource(mars_request)
    ds_re = source.load(time)

    print(ds_fc)
    print(ds_re)
