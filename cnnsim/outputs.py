"""Output functions y = f(x) of the cells.

All are bounded to [-1, 1] except "identity". "sign" has a surrogate gradient (the
derivative of clip(beta * x, -1, 1)) so that networks using it can still be trained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

import torch

OutputKind = Literal["pwl", "sigmoid", "tanh", "sign", "identity"]
OUTPUT_KINDS: tuple[str, ...] = get_args(OutputKind)


@dataclass(frozen=True)
class Output:
    """Cell output function.

    kind:
      "pwl"      0.5 (|x+1| - |x-1|) = clip(x, -1, 1)      (Chua–Yang standard)
      "sigmoid"  2 / (1 + exp(-beta x)) - 1 = tanh(beta x / 2)   (SCNN definition)
      "tanh"     tanh(beta x)
      "sign"     +1 for x > 0, -1 otherwise; surrogate gradient of clip(beta x, -1, 1)
      "identity" x
    """

    kind: OutputKind = "pwl"
    beta: float = 1.0

    def __post_init__(self):
        if self.kind not in OUTPUT_KINDS:
            raise ValueError(f"unknown output kind {self.kind!r}; choose from {list(OUTPUT_KINDS)}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        k = self.kind
        if k == "pwl":
            return x.clamp(-1.0, 1.0)
        if k == "sigmoid":
            return torch.tanh(0.5 * self.beta * x)
        if k == "tanh":
            return torch.tanh(self.beta * x)
        if k == "sign":
            hard = torch.where(x > 0, 1.0, -1.0).to(x.dtype)
            if not x.requires_grad:
                return hard
            soft = (self.beta * x).clamp(-1.0, 1.0)
            return soft + (hard - soft).detach()
        if k == "identity":
            return x
        raise ValueError(f"unknown output kind {k!r}")


PWL = Output("pwl")
