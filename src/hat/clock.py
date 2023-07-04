import datetime
import functools
import time

import humanize


def digital_clock(func):
    "A quiet decorator for timing functions"

    @functools.wraps(func)
    def clocked(*args, **kwargs):
        # time
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        human_readable_time = humanize.naturaldelta(
            datetime.timedelta(seconds=elapsed))

        # name
        name = func.__name__

        print(f"{name}() took {human_readable_time}")
        return result

    # return function with timing decorator
    return clocked


def clock(func):
    "A verbose decorator for timing functions"

    @functools.wraps(func)
    def clocked(*args, **kwargs):
        # time
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        human_readable_time = humanize.naturaldelta(
            datetime.timedelta(seconds=elapsed))

        # name
        name = func.__name__

        # arguments
        arg_str = ", ".join(repr(arg) for arg in args)
        if arg_str == "":
            arg_str = "no arguments"

        # keywords
        pairs = [f"{k}={w}" for k, w in sorted(kwargs.items())]
        key_str = ", ".join(pairs)
        if key_str == "":
            key_str = "no keywords"

        print(
            f"""{name}() took {human_readable_time} to run
            with following inputs {arg_str} and {key_str}"""
        )
        return result

    # return function with timing decorator
    return clocked


if __name__ == "__main__":

    @clock
    def test(arg1, keyword=False):
        time.sleep(0.1)
        pass

    test(1, keyword="hello")
