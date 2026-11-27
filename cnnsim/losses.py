"""Loss building blocks. Outputs y have shape (P, D, 1, H, W): P candidates, D samples.

"Local" losses compare each output image with its reference -> (P, D);
"global" aggregations combine them over the samples -> (P,) (the SCNN terminology:
local error per training element, combined by an aggregation function).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch


def _pix(y: torch.Tensor) -> torch.Tensor:
    return y.flatten(2)  # (P, D, H*W)


def mse(y: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Mean squared error per image. ref (D,1,H,W) broadcasts over candidates."""
    return (_pix(y) - _pix(ref.unsqueeze(0))).pow(2).mean(-1)


def msen(y: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """SCNN 'normalised MSE': 1/(4 Mx My) sum (y - y_ref)^2, i.e. MSE/4, in [0, 1] for outputs in [-1, 1]."""
    return mse(y, ref) / 4.0


def mae(y: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return (_pix(y) - _pix(ref.unsqueeze(0))).abs().mean(-1)


def pixel_error_rate(y: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Fraction of pixels on the wrong side of 0 (binary tasks; not differentiable)."""
    return ((_pix(y) > 0) != (_pix(ref.unsqueeze(0)) > 0)).float().mean(-1)


def mean_output(y: torch.Tensor) -> torch.Tensor:
    """Global average of the output image (P, D): a scalar read-out per sample."""
    return _pix(y).mean(-1)


Aggregation = Literal["mean", "max", "harmonic", "rms"]


def aggregate(local: torch.Tensor, how: Aggregation = "mean", weights: torch.Tensor | None = None) -> torch.Tensor:
    """(P, D) -> (P,)."""
    if weights is not None:
        w = weights.to(local) / weights.sum()
        if how == "mean":
            return (local * w).sum(1)
        raise ValueError("weights are only supported for 'mean'")
    if how == "mean":
        return local.mean(1)
    if how == "max":
        return local.max(1).values
    if how == "rms":
        return local.pow(2).mean(1).sqrt()
    if how == "harmonic":
        return local.shape[1] / (1.0 / local.clamp_min(1e-12)).sum(1)
    raise ValueError(f"unknown aggregation {how!r}")


LOCAL: dict[str, Callable] = {"mse": mse, "msen": msen, "mae": mae}


def image_loss(local: str = "mse", how: Aggregation = "mean", key: str = "target"):
    """Loss function for `cnnsim.train.fit`: compare y with batch[key] image by image."""
    f = LOCAL[local]

    def loss(y: torch.Tensor, batch: dict) -> torch.Tensor:
        return aggregate(f(y, batch[key]), how)

    return loss
