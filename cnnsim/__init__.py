from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from . import templates
from .boundary import Boundary
from .dynamics import bias_map, correlate, derivative, make_steps
from .outputs import Output
from .simulator import CNNSimulator, Prepared, Template, correlate3x3, output, run_template

try:
    __version__ = _version("cnnsim")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0+unknown"

__all__ = [
    "Boundary",
    "CNNSimulator",
    "Output",
    "Prepared",
    "Template",
    "__version__",
    "bias_map",
    "correlate",
    "correlate3x3",
    "derivative",
    "make_steps",
    "output",
    "run_template",
    "templates",
]
