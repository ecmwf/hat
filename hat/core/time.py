import numpy as np
import pandas as pd

from pandas import DatetimeIndex


class Time:
    def time(self):
        """Return the time of the forecast."""
        raise NotImplementedError

    def dayofyear(self, calendar: str = "standard") -> int:
        """Return the day of the year of the forecast."""
        raise NotImplementedError


class Reanalysis(Time):

    def __init__(self, time: DatetimeIndex) -> None:
        """Initialize Reanalysis with a specific time.

        Args
        ----
        time : DatetimeIndex
            The time of the reanalysis.
        """
        self._time = time

    def time(self):
        """Return the time of the forecast."""
        return self._time


class Forecast(Time):
    """Represents a forecast time."""

    # def __init__(self, time: DatetimeIndex) -> None:
    #     """Initialize Forecast with a specific time.
    #     """
    #     super().__init__(time)

    #     self.time = time
    #     self.forecast_date = time[0]
    #     self.steps = time - self.forecast_date

    def time(self):
        """Return the time of the forecast."""
        raise NotImplementedError

    def steps(self):
        """Return the steps of the forecast."""
        raise NotImplementedError

    def forecast_date(self):
        """Return the date of the forecast."""
        raise NotImplementedError

    def step_from_time(self, time):
        """Return the step from the time of the forecast."""
        raise NotImplementedError

    def time_from_step(self, step):
        """Return the time from the step of the forecast."""
        raise NotImplementedError


class ForecastFromBaseTimeAndStep(Forecast):
    """Represents a forecast time derived from base time and step."""

    def __init__(self, base_time: pd.Timestamp, steps: pd.TimedeltaIndex) -> None:
        """Initialize ForecastFromBaseTimeAndStep with base time and step.

        Args
        ----
        base_time : DatetimeIndex
            The base time of the forecast.
        step : int
            The step of the forecast.
        """
        self._base_time = base_time
        self._steps = steps

    def time(self):
        """Return the time of the forecast."""
        return self._base_time + self._steps

    def steps(self):
        """Return the steps of the forecast."""
        return self._steps

    def forecast_date(self):
        """Return the date of the forecast."""
        return self._base_time

    def step_from_time(self, time):
        """Return the step from the time of the forecast."""
        if time < self._base_time:
            raise ValueError("Time is before the base time.")
        return time - self._base_time

    def time_from_step(self, step):
        """Return the time from the step of the forecast."""
        raise NotImplementedError

    def mars_keys(self) -> dict:
        """Return the keys for the Mars request."""
        # step_array = self._steps.astype('timedelta64[h]').astype(int).tolist()
        step_array = "/".join(
            map(str, (self._steps.total_seconds() // 3600).astype(int).tolist())
        )
        return {
            "base_time": self._base_time,
            "steps": step_array,
            "valid_time": self.time(),
        }


def construct_time(options):

    if options.get("base_time") and options.get("step"):
        return ForecastFromBaseTimeAndStep(
            base_time=options["base_time"], steps=options["step"]
        )
    else:
        raise ValueError(
            "Invalid options for constructing time. "
            "Must provide 'base_time' and 'step'."
        )


# class ForecastFromValidTimeAndStep(Forecast):
#     """Represents a forecast time derived from valid time and step."""

#     def __init__(
#         self, time_coordinate: Coordinate, step_coordinate: Coordinate) -> None:
#         """Initialize ForecastFromValidTimeAndStep with time, step, and optional date coordinates.

#         Args
#         ----
#         time_coordinate : Coordinate
#             The time coordinate.
#         step_coordinate : Coordinate
#             The step coordinate.
#         date_coordinate : Optional[Coordinate]
#             The date coordinate.
#         """
#         self.time_coordinate_name = time_coordinate.variable.name
#         self.step_coordinate_name = step_coordinate.variable.name
#         self.date_coordinate_name = date_coordinate.variable.name if date_coordinate else None


# class ForecastFromValidTimeAndBaseTime(Time):
#     """Represents a forecast time derived from valid time and base time."""

#     def __init__(self, date_coordinate: Coordinate, time_coordinate: Coordinate) -> None:
#         """Initialize ForecastFromValidTimeAndBaseTime with date and time coordinates.

#         Args
#         ----
#         date_coordinate : Coordinate
#             The date coordinate.
#         time_coordinate : Coordinate
#             The time coordinate.
#         """
#         self.date_coordinate_name = date_coordinate.name
#         self.time_coordinate_name = time_coordinate.name


# class ForecastFromBaseTimeAndDate(Time):
#     """Represents a forecast time derived from base time and date."""

#     def __init__(self, date_coordinate: Coordinate, step_coordinate: Coordinate) -> None:
#         """Initialize ForecastFromBaseTimeAndDate with date and step coordinates.

#         Args
#         ----
#         date_coordinate : Coordinate
#             The date coordinate.
#         step_coordinate : Coordinate
#             The step coordinate.
#         """
#         self.date_coordinate_name = date_coordinate.name
#         self.step_coordinate_name = step_coordinate.name
