"""Fused Triton backend: values and gradients identical (to float32 rounding) to the stencil path."""

import zlib

import pytest
import torch

from cnnsim import Boundary, CNNSimulator, Output, make_steps
from cnnsim import fused
from cnnsim.losses import mean_output
from cnnsim.params import TemplateParams, TemplateSpec
from cnnsim.train import Dynamics, simulate

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and fused.HAVE_TRITON), reason="needs CUDA + Triton")
DEV = "cuda"


@pytest.mark.parametrize("bkind", ["fixed", "zeroflux", "periodic", "spiral"])
@pytest.mark.parametrize("okind", ["pwl", "sigmoid", "tanh", "identity", "sign"])
@pytest.mark.parametrize("k", [3, 5])
def test_values_and_gradients_match_stencil(bkind, okind, k):
    g = torch.Generator(device=DEV).manual_seed(zlib.crc32(f"{bkind}-{okind}-{k}".encode()))  # deterministic
    b = Boundary(bkind, -0.7 if bkind == "fixed" else 0.0)
    o = Output(okind, 1.7)
    N, H, W, steps, dt = 3, 7, 9, 25, 0.2
    x0 = (torch.rand(N, 1, H, W, device=DEV, generator=g) * 2 - 1).requires_grad_()
    bias = (torch.randn(N, 1, H, W, device=DEV, generator=g) * 0.3).requires_grad_()
    A = (torch.randn(N, k, k, device=DEV, generator=g) * (0.15 if okind == "identity" else 0.4)).requires_grad_()
    w = torch.randn(N, 1, H, W, device=DEV, generator=g)

    ref = make_steps(b, o, steps)(x0, bias, A, dt)
    torch.autograd.backward((ref * w).sum())
    g_ref = [t.grad.clone() for t in (x0, bias, A)]
    for t in (x0, bias, A):
        t.grad = None
    out = fused.integrate(x0, bias, A, b, o, steps, dt)
    torch.autograd.backward((out * w).sum())

    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)  # float32, different summation order
    for got, exp in zip((x0.grad, bias.grad, A.grad), g_ref):
        torch.testing.assert_close(got, exp, atol=1e-4 * exp.abs().max().item(), rtol=1e-4)


def test_shared_template_and_bias_broadcast():
    b, o = Boundary("spiral"), Output("tanh")
    x0 = torch.rand(4, 1, 8, 8, device=DEV) * 2 - 1
    bias = (torch.randn(1, 1, 8, 8, device=DEV) * 0.2).requires_grad_()
    A = (torch.randn(1, 3, 3, device=DEV) * 0.4).requires_grad_()
    ref = make_steps(b, o, 30)(x0, bias.expand(4, -1, -1, -1), A, 0.2)
    ga = torch.autograd.grad(ref.sum(), (bias, A))
    out = fused.integrate(x0, bias, A, b, o, 30, 0.2)
    gf = torch.autograd.grad(out.sum(), (bias, A))
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    for x, y in zip(gf, ga):
        torch.testing.assert_close(x, y, atol=1e-4, rtol=1e-4)


def test_simulator_backend_fused():
    u = torch.rand(5, 32, 40, device=DEV) * 2 - 1
    kw = dict(A=torch.randn(3, 3) * 0.3, B=torch.randn(3, 3) * 0.3, z=0.1, x0="input",
              boundary=Boundary("spiral"), output=Output("sigmoid"), steps=200, dt=0.2)
    ref = CNNSimulator(device=DEV).run(None, u, **kw)["x"]
    x = CNNSimulator(device=DEV, backend="fused").run(None, u, **kw)["x"]
    torch.testing.assert_close(x, ref, atol=1e-4, rtol=1e-4)


def test_train_simulate_backend_fused():
    u = (torch.rand(6, 1, 16, 16, device=DEV) * 2 - 1)
    grads, outs = [], []
    for backend in ("stencil", "fused"):
        dyn = Dynamics(boundary=Boundary("zeroflux"), output=Output("sigmoid"), steps=60, dt=0.2, backend=backend)
        p = TemplateParams(population=3, generator=torch.Generator().manual_seed(0), device=DEV)
        y = simulate(p(), u, dyn)
        mean_output(y).pow(2).sum().backward()
        outs.append(y.detach())
        grads.append(torch.cat([q.grad.flatten() for q in p.parameters()]))
    torch.testing.assert_close(outs[1], outs[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(grads[1], grads[0], atol=1e-4, rtol=1e-3)


def test_unsupported_configuration_falls_back():
    assert not fused.supported(Boundary("spiral"), Output("pwl"), 9, 64, 64)  # k > 7
    assert not fused.supported(Boundary("spiral"), Output("pwl"), 3, 256, 256)  # too many cells
    assert not fused.supported(Boundary("spiral"), Output("pwl"), 3, 64, 64, "cpu")


def test_training_policy_per_boundary():
    assert fused.prefer_for_training(Boundary("spiral"), 3)
    assert not fused.prefer_for_training(Boundary("spiral"), 5)
    for kind in ("fixed", "zeroflux", "periodic"):
        assert fused.prefer_for_training(Boundary(kind), 5)


@pytest.mark.parametrize("bkind", ["zeroflux", "fixed"])
def test_train_simulate_fused_k5(bkind):
    """k = 5 training through the fused backward matches the stencil path."""
    u = torch.rand(4, 1, 16, 16, device=DEV) * 2 - 1
    grads = []
    for backend in ("stencil", "fused"):
        dyn = Dynamics(boundary=Boundary(bkind), output=Output("sigmoid"), steps=40, dt=0.2, backend=backend)
        p = TemplateParams(TemplateSpec(size_A=5, size_B=5), population=2, scale=0.2,
                           generator=torch.Generator().manual_seed(1), device=DEV)
        mean_output(simulate(p(), u, dyn)).pow(2).sum().backward()
        grads.append(torch.cat([q.grad.flatten() for q in p.parameters()]))
    torch.testing.assert_close(grads[1], grads[0], atol=1e-4, rtol=1e-3)
