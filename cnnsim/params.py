"""Learnable templates with symmetry constraints, masks and an optional population dimension.

A template under a symmetry group G has one free parameter per *orbit* of template
cells under G (e.g. a 3×3 isotropic template has 3: centre, edge, corner). The template
is built by gathering the orbit values, so constraints hold exactly and the parameter
count is minimal. The same parameters are exposed as a flat vector (P, n) for
gradient-free optimisers.

A leading population dimension P holds independent candidates (multi-start training,
CMA-ES populations, parameter sweeps); the simulator handles per-sample templates.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

# symmetry groups as subsets of D4 (the symmetries of the square)
_OPS = {
    "id": lambda t: t,
    "rot90": lambda t: torch.rot90(t, 1, (-2, -1)),
    "rot180": lambda t: torch.rot90(t, 2, (-2, -1)),
    "rot270": lambda t: torch.rot90(t, 3, (-2, -1)),
    "fliplr": lambda t: t.flip(-1),
    "flipud": lambda t: t.flip(-2),
    "transpose": lambda t: t.transpose(-1, -2),
    "antitranspose": lambda t: torch.rot90(t, 2, (-2, -1)).transpose(-1, -2),
}
SYMMETRIES: dict[str, tuple[str, ...]] = {
    "none": ("id",),
    "central": ("id", "rot180"),  # point symmetric: a(k,l) = a(-k,-l)
    "mirror_lr": ("id", "fliplr"),
    "mirror_ud": ("id", "flipud"),
    "mirror": ("id", "fliplr", "flipud", "rot180"),
    "rot90": ("id", "rot90", "rot180", "rot270"),
    "isotropic": tuple(_OPS),  # full D4: value depends only on the distance class
}


def orbit_map(k: int, symmetry: str = "none", mask: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
    """Map each cell of a k×k template to its orbit index (or -1 if masked out).

    Returns (index (k,k) long, number of orbits).
    """
    if symmetry not in SYMMETRIES:
        raise ValueError(f"unknown symmetry {symmetry!r}; choose from {list(SYMMETRIES)}")
    idx = torch.arange(k * k).reshape(k, k)
    images = torch.stack([_OPS[o](idx) for o in SYMMETRIES[symmetry]])  # (|G|, k, k)
    rep = images.min(0).values  # orbit representative: smallest cell index in the orbit
    if mask is not None:
        rep = torch.where(torch.as_tensor(mask, dtype=torch.bool), rep, torch.full_like(rep, -1))
    uniq = rep[rep >= 0].unique()
    out = torch.full_like(rep, -1)
    for n, u in enumerate(uniq.tolist()):
        out[rep == u] = n
    return out, len(uniq)


@dataclass
class TemplateSpec:
    """Which parts of (A, B, z) are learnable and how they are constrained.

    size_*: odd template size; sym_*: key of SYMMETRIES; mask_*: bool (k,k), False = fixed 0.
    learn_*: if False, the part is fixed to its initial value (A0/B0/z0 or zeros).
    """

    size_A: int = 3
    size_B: int = 3
    sym_A: str = "none"
    sym_B: str = "none"
    mask_A: torch.Tensor | None = None
    mask_B: torch.Tensor | None = None
    learn_A: bool = True
    learn_B: bool = True
    learn_z: bool = True

    def __post_init__(self):
        for part in ("A", "B"):
            k, sym, mask = getattr(self, f"size_{part}"), getattr(self, f"sym_{part}"), getattr(self, f"mask_{part}")
            if not (isinstance(k, int) and k >= 1 and k % 2 == 1):
                raise ValueError(f"size_{part} must be a positive odd integer, got {k!r}")
            if sym not in SYMMETRIES:
                raise ValueError(f"unknown sym_{part} {sym!r}; choose from {list(SYMMETRIES)}")
            if mask is not None and tuple(torch.as_tensor(mask).shape) != (k, k):
                raise ValueError(f"mask_{part} must have shape ({k}, {k}), got {tuple(torch.as_tensor(mask).shape)}")

    def maps(self):
        """((index_A, n_A), (index_B, n_B)): orbit maps of A and B, computed from the current fields."""
        return orbit_map(self.size_A, self.sym_A, self.mask_A), orbit_map(self.size_B, self.sym_B, self.mask_B)


def _project(t: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Least-squares orbit values of templates t (P,k,k): the mean over each orbit."""
    P = t.shape[0]
    flat, ix = t.reshape(P, -1), index.reshape(-1)
    keep = ix >= 0
    s = torch.zeros(P, n, dtype=t.dtype, device=t.device).index_add_(1, ix[keep], flat[:, keep])
    c = torch.zeros(n, dtype=t.dtype, device=t.device).index_add_(0, ix[keep], torch.ones_like(ix[keep], dtype=t.dtype))
    return s / c


def _gather(theta: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Orbit values theta (P,n) -> templates (P,k,k); masked cells are 0."""
    P = theta.shape[0]
    padded = torch.cat([theta, torch.zeros(P, 1, dtype=theta.dtype, device=theta.device)], 1)
    ix = torch.where(index >= 0, index, theta.shape[1]).to(theta.device)
    return padded[:, ix.reshape(-1)].reshape(P, *index.shape)


class TemplateParams(nn.Module):
    """Learnable (A, B, z) for a population of P candidates.

    init: initial templates A0 (k,k) or (P,k,k), B0, z0 (float or (P,)); entries not given are
    drawn from N(0, scale²) (z: N(0, scale²) as well). Initial values are projected onto the
    constraint set.
    """

    def __init__(
        self,
        spec: TemplateSpec | None = None,
        population: int = 1,
        *,
        A0=None,
        B0=None,
        z0=None,
        scale: float = 0.5,
        generator: torch.Generator | None = None,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.spec = spec = spec or TemplateSpec()
        (iA, nA), (iB, nB) = spec.maps()
        self.register_buffer("index_A", iA, persistent=False)
        self.register_buffer("index_B", iB, persistent=False)
        self.P = population
        kw = dict(device=device, dtype=dtype)

        def init_t(t0, k):
            if t0 is None:
                return torch.randn(population, k, k, generator=generator, dtype=dtype).to(device) * scale
            t0 = torch.as_tensor(t0, **kw)
            return (t0[None] if t0.dim() == 2 else t0).expand(population, k, k).clone()

        tA = _project(init_t(A0, spec.size_A), iA.to(device), nA)
        tB = _project(init_t(B0, spec.size_B), iB.to(device), nB)
        if z0 is None:
            tz = torch.randn(population, generator=generator, dtype=dtype).to(device) * scale
        else:
            tz = torch.as_tensor(z0, **kw).expand(population).clone()

        for name, t, learn in (
            ("theta_A", tA, spec.learn_A),
            ("theta_B", tB, spec.learn_B),
            ("theta_z", tz, spec.learn_z),
        ):
            if learn:
                setattr(self, name, nn.Parameter(t))
            else:
                self.register_buffer(name, t)

    # ------------------------------------------------------------- templates

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(A (P,kA,kA), B (P,kB,kB), z (P,))."""
        return _gather(self.theta_A, self.index_A), _gather(self.theta_B, self.index_B), self.theta_z

    # ------------------------------------------------------------- flat vector (learnable parts only)

    def _parts(self):
        return [
            n
            for n, learn in (
                ("theta_A", self.spec.learn_A),
                ("theta_B", self.spec.learn_B),
                ("theta_z", self.spec.learn_z),
            )
            if learn
        ]

    @property
    def n_free(self) -> int:
        return sum(getattr(self, n).reshape(self.P, -1).shape[1] for n in self._parts())

    def to_vector(self) -> torch.Tensor:
        """(P, n_free) learnable parameters."""
        return torch.cat([getattr(self, n).detach().reshape(self.P, -1) for n in self._parts()], 1)

    def templates_from_vector(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Templates for an arbitrary batch of vectors v (Q, n_free); fixed parts are broadcast."""
        Q, parts, i = v.shape[0], {}, 0
        for n in ("theta_A", "theta_B", "theta_z"):
            cur = getattr(self, n)
            width = cur.reshape(self.P, -1).shape[1]
            if n in self._parts():
                parts[n] = v[:, i : i + width].reshape(Q, *cur.shape[1:])
                i += width
            else:
                parts[n] = cur[:1].expand(Q, *cur.shape[1:])
        return _gather(parts["theta_A"], self.index_A), _gather(parts["theta_B"], self.index_B), parts["theta_z"]

    @torch.no_grad()
    def load_vector(self, v: torch.Tensor) -> None:
        i = 0
        for n in self._parts():
            cur = getattr(self, n)
            width = cur.reshape(self.P, -1).shape[1]
            cur.copy_(v[:, i : i + width].reshape(cur.shape))
            i += width

    def select(self, members) -> TemplateParams:
        """A new TemplateParams containing only the given population members."""
        members = torch.as_tensor(members).reshape(-1)
        A, B, z = (t.detach()[members] for t in self())
        return TemplateParams(self.spec, len(members), A0=A, B0=B, z0=z, device=A.device, dtype=A.dtype)
