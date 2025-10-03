import logging
import os

_LOGGER_NAME = "H.A.T."
_LOGGER_FMT = "[%(asctime)s] %(name)s %(levelname)s: %(message)s"

logging.basicConfig(level=os.getenv("HAT_LOGLEVEL", logging.INFO), format=_LOGGER_FMT, datefmt="%d %b %H:%M:%S")
_LOGGER = logging.getLogger(_LOGGER_NAME)

try:
    from ._version import __version__
except ImportError:
    __version__ = "no-local-version"
