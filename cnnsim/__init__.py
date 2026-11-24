from importlib.metadata import PackageNotFoundError, version as _version

from .simulator import Boundary, CNNSimulator, Template, correlate3x3, output, run_template
from . import templates

try:
    __version__ = _version("cnnsim")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0+unknown"

__all__ = ["__version__", "Boundary", "CNNSimulator", "Template", "correlate3x3", "output", "run_template", "templates"]
