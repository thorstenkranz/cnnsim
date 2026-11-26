"""Boundary conditions: how the virtual cells around the grid are filled.

All boundaries pad a (N, C, H, W) tensor by a halo of radius r (r = k // 2 for a k×k
template). Padding is differentiable and compiles into fused kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

import torch
import torch.nn.functional as F

BoundaryKind = Literal["fixed", "zeroflux", "periodic", "spiral"]
BOUNDARY_KINDS: tuple[str, ...] = get_args(BoundaryKind)


@dataclass(frozen=True)
class Boundary:
    """Boundary condition for the virtual cells.

    kind:
      "fixed"    Dirichlet: virtual cells have the constant `value` (0 = "neutral").
      "zeroflux" Neumann: virtual cells copy the nearest edge cell.
      "periodic" torus: rows and columns wrap around independently.
      "spiral"   closed spiral: the grid is one sequence in row-major order, closed into a
                 ring. The neighbour at offset (di, dj) of the cell with flat index p is the
                 cell p + di*W + dj (mod H*W): the right neighbour of the last cell in a row is
                 the first cell of the next row. Natural for a time series written row-wise.
    """

    kind: BoundaryKind = "fixed"
    value: float = 0.0

    def __post_init__(self):
        if self.kind not in BOUNDARY_KINDS:
            raise ValueError(f"unknown boundary kind {self.kind!r}; choose from {list(BOUNDARY_KINDS)}")

    def pad(self, t: torch.Tensor, r: int = 1) -> torch.Tensor:
        if r == 0:
            return t
        if self.kind == "fixed":
            return F.pad(t, (r, r, r, r), mode="constant", value=self.value)
        if self.kind == "zeroflux":
            return F.pad(t, (r, r, r, r), mode="replicate")
        if self.kind == "periodic":
            return F.pad(t, (r, r, r, r), mode="circular")
        if self.kind == "spiral":
            return _pad_spiral(t, r)
        raise ValueError(f"unknown boundary kind {self.kind!r}")


def _pad_spiral(t: torch.Tensor, r: int) -> torch.Tensor:
    n, c, h, w = t.shape
    flat = t.reshape(n, c, h * w)
    # ext[k] = flat[(k - r) mod HW]; row i of the horizontally padded grid is
    # flat[i*W - r : i*W + W + r], i.e. a window of length W + 2r starting at ext[i*W].
    ext = torch.cat([flat[..., h * w - r :], flat, flat[..., :r]], dim=-1)
    rows = ext.unfold(-1, w + 2 * r, w)  # (n, c, h, w + 2r)
    # vertical offsets are multiples of W in the flat index -> wrap rows periodically
    return torch.cat([rows[..., h - r :, :], rows, rows[..., :r, :]], dim=-2)
