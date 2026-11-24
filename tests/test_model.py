"""Generalised model: boundaries (incl. spiral), k×k templates, output functions, bias maps, gradients."""

import pytest
import torch

from cnnsim import Boundary, CNNSimulator, Output, Template, correlate, make_steps
from cnnsim.dynamics import bias_map

DEV = "cuda" if torch.cuda.is_available() else "cpu"
KINDS = ["fixed", "zeroflux", "periodic", "spiral"]


def correlate_reference(t, w, boundary):
    """Direct per-cell evaluation of the correlation with explicit boundary rules."""
    n, _, h, wd = t.shape
    k = w.shape[-1]
    r = k // 2
    out = torch.zeros_like(t)
    for b in range(n):
        wb = w[b if w.shape[0] > 1 else 0]
        for i in range(h):
            for j in range(wd):
                acc = 0.0
                for di in range(-r, r + 1):
                    for dj in range(-r, r + 1):
                        ii, jj = i + di, j + dj
                        if boundary.kind == "spiral":
                            p = (i * wd + j + di * wd + dj) % (h * wd)
                            v = t[b, 0, p // wd, p % wd]
                        elif 0 <= ii < h and 0 <= jj < wd:
                            v = t[b, 0, ii, jj]
                        elif boundary.kind == "fixed":
                            v = boundary.value
                        elif boundary.kind == "zeroflux":
                            v = t[b, 0, min(max(ii, 0), h - 1), min(max(jj, 0), wd - 1)]
                        else:  # periodic
                            v = t[b, 0, ii % h, jj % wd]
                        acc = acc + wb[di + r, dj + r] * v
                out[b, 0, i, j] = acc
    return out


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("k", [3, 5])
def test_correlate_matches_reference(kind, k):
    g = torch.Generator().manual_seed(1)
    t = torch.randn(2, 1, 6, 7, generator=g, dtype=torch.float64)
    w = torch.randn(2, k, k, generator=g, dtype=torch.float64)
    b = Boundary(kind, 0.3)
    ref = correlate_reference(t, w, b)
    for backend in ("stencil", "conv"):
        for ww, rr in ((w, ref), (w[:1], correlate_reference(t, w[:1], b))):
            torch.testing.assert_close(correlate(t, ww, b, backend), rr)


def test_spiral_neighbours_of_row_ends():
    t = torch.arange(12.0).reshape(1, 1, 3, 4)  # flat index = value
    right = torch.zeros(1, 3, 3)
    right[0, 1, 2] = 1.0  # picks the right neighbour
    out = correlate(t, right, Boundary("spiral"))
    assert out[0, 0, 0, 3] == 4.0  # end of row 0 -> start of row 1
    assert out[0, 0, 2, 3] == 0.0  # last cell wraps to the first
    left = torch.zeros(1, 3, 3)
    left[0, 1, 0] = 1.0
    assert correlate(t, left, Boundary("spiral"))[0, 0, 1, 0] == 3.0


def test_output_functions():
    x = torch.tensor([-3.0, -0.5, 0.0, 0.5, 3.0])
    torch.testing.assert_close(Output("pwl")(x), torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]))
    torch.testing.assert_close(Output("sigmoid", beta=2.0)(x), 2 / (1 + torch.exp(-2 * x)) - 1)
    torch.testing.assert_close(Output("tanh", beta=0.5)(x), torch.tanh(0.5 * x))
    torch.testing.assert_close(Output("sign")(x), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0]))
    torch.testing.assert_close(Output("identity")(x), x)


def test_sign_has_surrogate_gradient():
    x = torch.tensor([-2.0, -0.2, 0.3, 2.0], requires_grad=True)
    y = Output("sign", beta=1.0)(x)
    torch.testing.assert_close(y.detach(), torch.tensor([-1.0, -1.0, 1.0, 1.0]))
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([0.0, 1.0, 1.0, 0.0]))


def test_bias_map_forms():
    sim = CNNSimulator(device="cpu")
    u = torch.zeros(2, 4, 5)
    zmap = torch.arange(20.0).reshape(4, 5)
    p = sim.prepare(None, u, A=torch.zeros(3, 3), z=zmap)
    torch.testing.assert_close(p.bias[1, 0], zmap)
    p = sim.prepare(None, u, A=torch.zeros(3, 3), z=torch.tensor([1.0, 2.0]))
    assert p.bias[0].unique().item() == 1.0 and p.bias[1].unique().item() == 2.0


def test_template_output_and_input_boundary_are_used():
    u = torch.ones(1, 5, 5)
    # B sums the 4-neighbourhood of u; with a white (-1) input boundary the corners see 2 virtual cells
    t = Template(A=[[0.0]], B=[[0, 1, 0], [1, 0, 1], [0, 1, 0]], z=0.0,
                 boundary=Boundary("zeroflux"), boundary_u=Boundary("fixed", -1.0), output=Output("identity"))
    p = CNNSimulator(device="cpu").prepare(t, u)
    assert p.bias[0, 0, 0, 0] == 0.0 and p.bias[0, 0, 2, 2] == 4.0
    assert p.output.kind == "identity"


@pytest.mark.parametrize("kind", ["sigmoid", "tanh"])
def test_gradients_through_dynamics(kind):
    """gradcheck of the final state w.r.t. A, B, z through 10 Euler steps on the spiral boundary."""
    g = torch.Generator().manual_seed(0)
    u = torch.rand(2, 1, 4, 5, generator=g, dtype=torch.float64) * 2 - 1
    bnd, out = Boundary("spiral"), Output(kind, beta=1.5)
    steps = make_steps(bnd, out, 10)

    def f(A, B, z):
        bias = bias_map(u, B, z, bnd)
        return steps(u.clone(), bias, A, 0.2)

    A = (torch.randn(1, 3, 3, generator=g, dtype=torch.float64) * 0.5).requires_grad_()
    B = (torch.randn(1, 3, 3, generator=g, dtype=torch.float64) * 0.5).requires_grad_()
    z = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (A, B, z))


def test_k5_template_runs_in_all_modes():
    g = torch.Generator().manual_seed(0)
    u = (torch.rand(3, 16, 16, generator=g) * 2 - 1).to(DEV)
    A = torch.randn(5, 5, generator=g) * 0.2
    B = torch.randn(3, 3, generator=g) * 0.3
    kw = dict(A=A, B=B, z=0.1, x0="input", boundary=Boundary("spiral"), output=Output("sigmoid"))
    ref = CNNSimulator(device=DEV).run(None, u, steps=100, dt=0.2, **kw)["x"]
    for mode in (dict(backend="conv"), dict(compile=True, cuda_graph=True)):
        x = CNNSimulator(device=DEV, **mode).run(None, u, steps=100, dt=0.2, **kw)["x"]
        torch.testing.assert_close(x, ref, atol=1e-4, rtol=1e-4)
