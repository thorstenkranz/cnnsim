"""Template parameterisation, differentiable simulation, training and gradient-free optimisers."""

import pytest
import torch

from cnnsim import Boundary, CNNSimulator, Output, templates as T
from cnnsim.losses import aggregate, image_loss, mean_output, msen, pixel_error_rate
from cnnsim.optim import cmaes, objective, simulated_annealing
from cnnsim.params import TemplateParams, TemplateSpec, orbit_map
from cnnsim.train import Dynamics, evaluate, fit, simulate

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize(
    "k,sym,n",
    [(3, "none", 9), (3, "central", 5), (3, "mirror", 4), (3, "rot90", 3), (3, "isotropic", 3), (5, "isotropic", 6), (3, "mirror_lr", 6)],
)
def test_orbit_counts(k, sym, n):
    assert orbit_map(k, sym)[1] == n


def test_mask_removes_cells():
    mask = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.bool)
    idx, n = orbit_map(3, "isotropic", mask)
    assert n == 2 and (idx[0, 0] == -1) and (idx[1, 1] != idx[0, 1])


def test_symmetry_holds_and_vector_roundtrip():
    spec = TemplateSpec(sym_A="isotropic", sym_B="central")
    p = TemplateParams(spec, population=4, generator=torch.Generator().manual_seed(0))
    A, B, z = p()
    torch.testing.assert_close(A, A.transpose(-1, -2))
    torch.testing.assert_close(A, A.flip(-1))
    torch.testing.assert_close(B, torch.rot90(B, 2, (-2, -1)))
    assert p.n_free == 3 + 5 + 1
    v = p.to_vector()
    A2, B2, z2 = p.templates_from_vector(v)
    torch.testing.assert_close((A2, B2, z2), (A, B, z))


def test_fixed_parts_and_initial_values():
    spec = TemplateSpec(learn_A=False)
    p = TemplateParams(spec, population=2, A0=T.EDGE.A, B0=T.EDGE.B, z0=T.EDGE.z)
    assert p.n_free == 9 + 1
    A, B, z = p()
    torch.testing.assert_close(A[1], torch.tensor(T.EDGE.A, dtype=torch.float32))
    assert z[0].item() == T.EDGE.z


def test_simulate_matches_simulator():
    g = torch.Generator().manual_seed(0)
    u = (torch.rand(3, 1, 12, 14, generator=g) * 2 - 1).to(DEV)
    dyn = Dynamics(boundary=Boundary("spiral"), output=Output("sigmoid"), steps=40, dt=0.2, x0="input")
    p = TemplateParams(population=2, generator=g, device=DEV)
    A, B, z = p()
    y = simulate((A, B, z), u, dyn)
    assert y.shape == (2, 3, 1, 12, 14)
    sim = CNNSimulator(device=DEV)
    for i in range(2):
        ref = sim.run(None, u, A=A[i].detach(), B=B[i].detach(), z=z[i].item(), x0="input",
                      boundary=Boundary("spiral"), output=Output("sigmoid"), steps=40, dt=0.2)["y"]
        torch.testing.assert_close(y[i].detach(), ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("variant", [dict(checkpoint_every=7), dict(checkpoint_every=10, compile=True)])
def test_gradients_unchanged_by_checkpointing_and_compile(variant):
    g = torch.Generator().manual_seed(0)
    u = (torch.rand(2, 1, 10, 10, generator=g) * 2 - 1).to(DEV)
    base = Dynamics(boundary=Boundary("zeroflux"), output=Output("tanh"), steps=30, dt=0.2, checkpoint_every=0)
    grads = []
    for dyn in (base, Dynamics(**{**base.__dict__, **variant})):
        p = TemplateParams(population=2, generator=torch.Generator().manual_seed(1), device=DEV)
        mean_output(simulate(p(), u, dyn)).pow(2).sum().backward()
        grads.append(torch.cat([q.grad.flatten() for q in p.parameters()]))
    torch.testing.assert_close(grads[0], grads[1], atol=1e-4, rtol=1e-3)


def test_fit_reduces_loss_and_tracks_best():
    g = torch.Generator().manual_seed(0)
    u = (torch.rand(8, 1, 12, 12, generator=g) * 2 - 1).to(DEV)
    data = {"u": u, "target": -u.sign()}  # learn inversion of the sign
    dyn = Dynamics(boundary=Boundary("fixed", 0.0), output=Output("tanh"), steps=20, dt=0.2, x0="zero")
    p = TemplateParams(population=4, scale=0.1, generator=g, device=DEV)
    res = fit(p, image_loss("msen"), data, dyn, epochs=60, lr=0.1)
    first, last = res.history["train"][0], res.history["train"][-1]
    assert (last < 0.5 * first).all()
    err = aggregate(pixel_error_rate(simulate(res.best(), u, dyn), data["target"]))
    assert err.min() < 0.02  # only pixels with |u| ~ 0 can stay wrong


def test_cmaes_on_quadratic():
    target = torch.tensor([1.0, -2.0, 0.5, 3.0])
    res = cmaes(lambda X: ((X - target) ** 2).sum(1), torch.zeros(4), sigma0=1.0, iters=150, seed=0)
    torch.testing.assert_close(res.best_x, target, atol=1e-3, rtol=0)


def test_simulated_annealing_on_quadratic():
    target = torch.tensor([1.0, -2.0])
    res = simulated_annealing(lambda X: ((X - target) ** 2).sum(1), torch.zeros(8, 2), iters=2000, step0=1.0, seed=0)
    assert res.best_f.min() < 1e-3


def test_cmaes_finds_template_via_objective():
    """End-to-end: CMA-ES over a small template parameterisation reproduces inversion."""
    g = torch.Generator().manual_seed(0)
    u = (torch.rand(4, 1, 10, 10, generator=g) * 2 - 1).to(DEV)
    data = {"u": u, "target": -u.sign()}
    dyn = Dynamics(boundary=Boundary("fixed", 0.0), output=Output("pwl"), steps=30, dt=0.2, x0="zero")
    spec = TemplateSpec(mask_A=torch.eye(3, dtype=torch.bool) & torch.eye(3, dtype=torch.bool).flip(0),
                        mask_B=torch.eye(3, dtype=torch.bool) & torch.eye(3, dtype=torch.bool).flip(0))
    p = TemplateParams(spec, population=1, device=DEV)
    assert p.n_free == 3  # centre of A, centre of B, z
    f = objective(p, lambda y, b: aggregate(msen(y, b["target"])), data, dyn)
    res = cmaes(f, torch.zeros(3, device=DEV), sigma0=1.0, iters=60, seed=0)
    assert res.best_f < 0.01


@pytest.mark.slow
def test_learn_edge_template_from_examples():
    g = torch.Generator().manual_seed(0)

    def blobs(n):
        u = (torch.rand(n, 1, 24, 24, generator=g) < 0.35).float()
        return ((torch.nn.functional.avg_pool2d(u, 3, 1, 1) > 0.3).float() * 2 - 1).to(DEV)

    sim = CNNSimulator(device=DEV)
    make = lambda n: (lambda u: {"u": u, "target": sim.run(T.EDGE, u, steps=200, dt=0.1)["y"].sign()})(blobs(n))  # noqa: E731
    train, val = make(24), make(48)
    dyn = Dynamics(boundary=Boundary("fixed", -1.0), output=Output("tanh"), steps=60, dt=0.2, x0="zero", compile=True)
    p = TemplateParams(population=8, generator=torch.Generator().manual_seed(1), device=DEV)
    res = fit(p, image_loss("mse"), train, dyn, val=val, epochs=200, lr=0.05)
    err = aggregate(pixel_error_rate(simulate(res.best(), val["u"], dyn), val["target"]))
    assert err.min() < 0.005


@pytest.mark.parametrize(
    "kw",
    [dict(size_A=4), dict(size_B=0), dict(sym_A="isotrpic"), dict(mask_A=torch.ones(5, 5, dtype=torch.bool))],
)
def test_template_spec_rejects_invalid_fields(kw):
    with pytest.raises(ValueError):
        TemplateSpec(**kw)


def test_template_spec_maps_follow_field_changes():
    spec = TemplateSpec()
    assert spec.maps()[0][1] == 9
    spec.sym_A = "isotropic"
    assert spec.maps()[0][1] == 3


def test_fit_tracks_best_on_val_only_with_eval_every():
    """With eval_every > 1, train losses must not enter the best-on-val tracking."""
    g = torch.Generator().manual_seed(0)
    u = (torch.rand(8, 1, 10, 10, generator=g) * 2 - 1).to(DEV)
    train = {"u": u, "target": -u.sign()}
    val = {"u": u, "target": u.sign()}  # opposite task: val loss stays high while train loss drops
    dyn = Dynamics(boundary=Boundary("fixed", 0.0), output=Output("tanh"), steps=20, dt=0.2, x0="zero")
    p = TemplateParams(population=3, scale=0.1, generator=g, device=DEV)
    loss = image_loss("msen")
    res = fit(p, loss, train, dyn, val=val, epochs=10, eval_every=3, lr=0.1)
    assert len(res.history["val"]) == 4  # epochs 0, 3, 6, 9
    torch.testing.assert_close(res.best_loss, torch.stack(res.history["val"]).min(0).values)
    torch.testing.assert_close(evaluate(res.best, loss, val, dyn).cpu(), res.best_loss, atol=1e-5, rtol=1e-5)
