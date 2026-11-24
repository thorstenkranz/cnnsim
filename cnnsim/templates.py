"""A few classic CNN templates.

Values follow the CNN template library (SZTAKI / Chua & Roska) and PyCNN. Convention:
black = +1, white = -1. Run conventions (initial state, boundary) are stored with each
template. These are checked by tests/test_templates.py on synthetic images.
"""

from .boundary import Boundary
from .simulator import Template

_Z = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

EDGE = Template(
    name="edge",
    A=[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    B=[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
    z=-1,
    x0="zero",
    boundary=Boundary("fixed", -1.0),  # white outside
    description="Binary edge detection: black pixels with at least one white neighbour stay black.",
)

EDGE_GRAY = Template(
    name="edge_gray",
    A=[[0, 0, 0], [0, 2, 0], [0, 0, 0]],
    B=[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
    z=-0.5,
    x0="zero",
    boundary=Boundary("zeroflux"),
    description="Edge detection on grayscale input (bipolar output).",
)

CORNER = Template(
    name="corner",
    A=[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    B=[[-1, -1, -1], [-1, 4, -1], [-1, -1, -1]],
    z=-5,
    x0="zero",
    boundary=Boundary("fixed", -1.0),
    description="Convex corner detection: black pixels with >= 5 white neighbours.",
)

INVERSION = Template(
    name="inversion",
    A=[[0, 0, 0], [0, 2, 0], [0, 0, 0]],
    B=[[0, 0, 0], [0, -2, 0], [0, 0, 0]],
    z=0,
    x0="zero",
    boundary=Boundary("fixed", 0.0),
    description="Logic NOT: black <-> white.",
)

THRESHOLD = Template(
    name="threshold",
    A=[[0, 0, 0], [0, 2, 0], [0, 0, 0]],
    B=_Z,
    z=0.0,  # threshold level is -z
    x0="input",
    boundary=Boundary("zeroflux"),
    description="Grayscale -> binary: black where x0 > -z.",
)

HOLE_FILL = Template(
    name="hole_fill",
    A=[[0, 1, 0], [1, 3, 1], [0, 1, 0]],
    B=[[0, 0, 0], [0, 4, 0], [0, 0, 0]],
    z=-1,
    x0="ones",
    boundary=Boundary("fixed", -1.0),
    description=(
        "Fills enclosed holes. Starts all black; white 'erodes' in from the (white) "
        "boundary through white input pixels only, so enclosed regions remain black."
    ),
)

SHADOW_LEFT = Template(
    name="shadow_left",
    A=[[0, 0, 0], [0, 2, 2], [0, 0, 0]],
    B=[[0, 0, 0], [0, 2, 0], [0, 0, 0]],
    z=2,
    x0="input",
    boundary=Boundary("fixed", -1.0),
    description="Every pixel to the left of a black pixel turns black (propagating wave).",
)

CCD_HORIZONTAL = Template(
    name="ccd_horizontal",
    A=[[0, 0, 0], [1, 2, -1], [0, 0, 0]],
    B=_Z,
    z=0,
    x0="input",
    # white boundary: with 0, a black pixel in the first column is marginally stable
    # (dx/dt = 0) and keeps feeding black into the row
    boundary=Boundary("fixed", -1.0),
    description=(
        "Connected component detection: each horizontal run of black pixels is "
        "compressed to a single black pixel; these stack up at the right edge, "
        "separated by single white pixels."
    ),
)

AVERAGE = Template(
    name="average",
    A=[[0, 1, 0], [1, 2, 1], [0, 1, 0]],
    B=_Z,
    z=0,
    x0="input",
    boundary=Boundary("zeroflux"),
    description="Smoothing / majority vote; noise removal on binary images.",
)

ALL = {t.name: t for t in [EDGE, EDGE_GRAY, CORNER, INVERSION, THRESHOLD, HOLE_FILL,
                           SHADOW_LEFT, CCD_HORIZONTAL, AVERAGE]}
