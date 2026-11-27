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
    t = Template(
        A=[[0.0]],
        B=[[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        z=0.0,
        boundary=Boundary("zeroflux"),
        boundary_u=Boundary("fixed", -1.0),
        output=Output("identity"),
    )
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


def test_invalid_kinds_are_rejected_at_construction():
    with pytest.raises(ValueError, match="boundary kind"):
        Boundary("fxed")
    with pytest.raises(ValueError, match="output kind"):
        Output("relu")


def test_templates_are_immutable_and_do_not_share_state():
    import dataclasses

    from cnnsim import templates as T

    with pytest.raises(dataclasses.FrozenInstanceError):
        T.THRESHOLD.z = 1.0
    with pytest.raises(TypeError):
        T.THRESHOLD.B[0][0] = 99.0  # tuples
    t = dataclasses.replace(T.THRESHOLD, B=[[0, 0, 0], [0, 99, 0], [0, 0, 0]])
    assert t.B[1][1] == 99.0 and T.THRESHOLD.B[1][1] == 0.0 and T.AVERAGE.B[1][1] == 0.0
    assert Template(A=torch.eye(3), B=[[1.0]], z=0).A[0] == (1.0, 0.0, 0.0)  # tensors are converted


@pytest.mark.parametrize("mode", [dict(), dict(backend="conv"), dict(compile=True, cuda_graph=True)])
@pytest.mark.parametrize("steps", [0, 7, 20, 47])
def test_run_takes_exactly_the_requested_steps(mode, steps):
    g = torch.Generator().manual_seed(0)
    u = (torch.rand(2, 12, 13, generator=g) * 2 - 1).to(DEV)
    kw = dict(
        A=torch.randn(3, 3, generator=g) * 0.4,
        B=torch.randn(3, 3, generator=g) * 0.4,
        z=0.1,
        x0="input",
        boundary=Boundary("zeroflux"),
        output=Output("tanh"),
    )
    sim = CNNSimulator(device=DEV, chunk=20, **mode)
    r = sim.run(None, u, steps=steps, dt=0.2, **kw)
    assert r["steps"] == steps
    p = sim.prepare(None, u, **kw)
    ref = make_steps(p.boundary, p.output, steps)(p.x, p.bias, p.A, 0.2)
    torch.testing.assert_close(r["x"], ref, atol=1e-5, rtol=1e-5)


def test_import_leaves_torch_dynamo_config_alone():
    import subprocess
    import sys

    code = (
        "import torch._dynamo as d; before = d.config.cache_size_limit; import cnnsim; "
        "assert d.config.cache_size_limit == before, (before, d.config.cache_size_limit)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_pad_with_zero_radius_is_identity():
    t = torch.randn(1, 1, 3, 4)
    for kind in KINDS:
        assert Boundary(kind).pad(t, 0) is t


@pytest.mark.parametrize("shape", [(2, 2), (3, 4), (1, 3, 4), (2, 3, 5, 5)])
def test_even_or_non_square_templates_are_rejected(shape):
    with pytest.raises(ValueError, match="odd"):
        CNNSimulator(device="cpu").prepare(None, torch.zeros(1, 4, 4), A=torch.zeros(shape))


def test_negative_steps_are_rejected():
    with pytest.raises(ValueError, match="steps"):
        CNNSimulator(device="cpu").run(None, torch.zeros(1, 4, 4), A=torch.zeros(3, 3), steps=-1)


def test_prepare_needs_input_or_shape():
    sim = CNNSimulator(device="cpu")
    with pytest.raises(ValueError, match="shape"):
        sim.prepare(None, None, A=torch.zeros(3, 3))
    p = sim.prepare(None, None, A=torch.zeros(3, 3), shape=(2, 4, 5))
    assert p.x.shape == (2, 1, 4, 5)


@pytest.mark.parametrize("x0,value", [("zero", 0.0), ("ones", 1.0), ("minus_ones", -1.0), ("input", 0.5)])
def test_initial_state_kinds(x0, value):
    p = CNNSimulator(device="cpu").prepare(None, torch.full((1, 3, 3), 0.5), A=torch.zeros(3, 3), x0=x0)
    assert p.x.unique().tolist() == [value]


def test_initial_state_tensor_and_input_is_not_aliased():
    u = torch.full((1, 3, 3), 0.5)
    x0 = torch.arange(9.0).reshape(3, 3)
    p = CNNSimulator(device="cpu").prepare(None, u, A=torch.zeros(3, 3), x0=x0)
    torch.testing.assert_close(p.x[0, 0], x0)
    # x0="input" must copy u: integrating must not change the caller's input
    CNNSimulator(device="cpu").run(None, u, A=torch.eye(3), z=1.0, x0="input", steps=10)
    assert u.unique().item() == 0.5


def test_batched_templates_broadcast_against_one_input():
    u = torch.rand(1, 6, 6) * 2 - 1
    A = torch.stack([torch.zeros(3, 3), 2 * torch.eye(3)])
    r = CNNSimulator(device="cpu").run(None, u, A=A, z=torch.tensor([0.5, -0.5]), steps=40)
    assert r["y"].shape == (2, 1, 6, 6)
    for i in range(2):
        ri = CNNSimulator(device="cpu").run(None, u, A=A[i], z=float([0.5, -0.5][i]), steps=40)
        torch.testing.assert_close(r["y"][i], ri["y"][0])


def test_tol_stops_early_at_a_chunk_boundary_after_convergence():
    sim = CNNSimulator(device="cpu", chunk=10)
    # dx/dt = -x + 2: converges to x = 2 (|y| = 1) exponentially
    r = sim.run(None, torch.zeros(1, 4, 4), A=torch.zeros(3, 3), z=2.0, steps=10_000, dt=0.2, tol=1e-4, check_every=50)
    assert r["steps"] < 10_000 and r["steps"] % 50 == 0
    torch.testing.assert_close(r["x"], torch.full_like(r["x"], 2.0), atol=1e-3, rtol=0)
    # without tol the full number of steps is taken
    assert sim.run(None, torch.zeros(1, 4, 4), A=torch.zeros(3, 3), z=2.0, steps=95, dt=0.2)["steps"] == 95


def test_record_every_keeps_snapshots():
    sim = CNNSimulator(device="cpu", chunk=10)
    r = sim.run(None, torch.zeros(1, 4, 4), A=torch.zeros(3, 3), z=0.5, steps=100, dt=0.1, record_every=20)
    assert len(r["history"]) == 1 + 100 // 20  # initial output plus one per 20 steps
    means = [h.mean().item() for h in r["history"]]
    assert means == sorted(means)  # monotone approach to 0.5
    assert sim.run(None, torch.zeros(1, 4, 4), A=torch.zeros(3, 3), steps=20)["history"] == []


def test_run_template_passes_simulator_options():
    from cnnsim import run_template
    from cnnsim import templates as T

    u = torch.ones(1, 5, 5)
    y = run_template(T.INVERSION, u, device="cpu", chunk=5, steps=100)
    assert y.device.type == "cpu" and (y < 0).all()


@pytest.mark.parametrize("bkind", ["zeroflux", "periodic"])
def test_rk4_gradients_through_dynamics(bkind):
    g = torch.Generator().manual_seed(0)
    x0 = torch.rand(1, 1, 4, 4, generator=g, dtype=torch.float64) * 2 - 1
    bnd, out = Boundary(bkind), Output("tanh")
    steps = make_steps(bnd, out, 5, integrator="rk4")
    bias = torch.zeros_like(x0)
    A = (torch.randn(1, 3, 3, generator=g, dtype=torch.float64) * 0.5).requires_grad_()
    assert torch.autograd.gradcheck(lambda A: steps(x0, bias, A, 0.2), (A,))
