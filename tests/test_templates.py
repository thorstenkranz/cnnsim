"""Sanity checks: templates produce the expected binary results, backends agree."""

import numpy as np
import pytest
import torch

from cnnsim import CNNSimulator
from cnnsim import templates as T
from cnnsim.simulator import Boundary, correlate3x3

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def bip(mask: np.ndarray) -> torch.Tensor:
    """bool mask (True = black) -> bipolar tensor."""
    return torch.tensor(np.where(mask, 1.0, -1.0), dtype=torch.float32)


def run(tmpl, u, steps=400, dt=0.1, **kw):
    sim = CNNSimulator(device=DEV, **kw)
    y = sim.run(tmpl, u, steps=steps, dt=dt)["y"][0, 0].cpu().numpy()
    return y > 0


def shapes(h=40, w=40):
    m = np.zeros((h, w), bool)
    m[5:15, 5:15] = True  # filled square
    m[20:32, 20:32] = True  # ring with hole
    m[23:29, 23:29] = False
    m[5:8, 25:35] = True  # bar
    return m


def neighbours_white(m):
    p = np.pad(m, 1, constant_values=False)
    h, w = m.shape
    cnt = np.zeros(m.shape, int)
    for i in range(3):
        for j in range(3):
            if (i, j) != (1, 1):
                cnt += ~p[i : i + h, j : j + w]
    return cnt


def test_edge():
    m = shapes()
    expect = m & (neighbours_white(m) >= 1)
    assert (run(T.EDGE, bip(m)) == expect).all()


def test_corner():
    m = shapes()
    expect = m & (neighbours_white(m) >= 5)
    assert (run(T.CORNER, bip(m)) == expect).all()


def test_inversion():
    m = shapes()
    assert (run(T.INVERSION, bip(m)) == ~m).all()


def test_threshold():
    g = torch.linspace(-1, 1, 41).repeat(10, 1)
    out = run(T.THRESHOLD, g)
    assert (out == (g.numpy() > 0)).all()


def test_hole_fill():
    m = shapes()
    expect = m.copy()
    expect[23:29, 23:29] = True
    assert (run(T.HOLE_FILL, bip(m), steps=1500) == expect).all()


def test_shadow_left():
    m = np.zeros((20, 30), bool)
    m[3, 20] = True
    m[10, 8] = True
    expect = np.zeros_like(m)
    expect[3, :21] = True
    expect[10, :9] = True
    assert (run(T.SHADOW_LEFT, bip(m), steps=1500) == expect).all()


def test_ccd_horizontal():
    m = np.zeros((6, 20), bool)
    m[1, 2:5] = True  # one run
    m[3, 1:3] = True  # two runs
    m[3, 8:12] = True
    m[5, 0] = True  # three runs, one touching the left edge, one the right edge
    m[5, 4:7] = True
    m[5, 19] = True
    out = run(T.CCD_HORIZONTAL, bip(m), steps=3000)
    assert out.sum(axis=1).tolist() == [0, 1, 0, 2, 0, 3]
    # runs collapse to single pixels stacked at the right edge, separated by one white pixel
    assert out[1, -1] and out[3, -1] and out[3, -3]
    assert out[5, -1] and out[5, -3] and out[5, -5]


def test_average_removes_salt_noise():
    m = shapes()
    noisy = m.copy()
    noisy[2, 2] = noisy[35, 10] = True  # isolated black pixels
    out = run(T.AVERAGE, bip(noisy))
    assert not out[2, 2] and not out[35, 10]


@pytest.mark.parametrize("bkind", ["fixed", "zeroflux", "periodic"])
def test_backends_agree(bkind):
    g = torch.Generator().manual_seed(0)
    t = torch.randn(3, 1, 33, 47, generator=g).to(DEV)
    w = torch.randn(3, 3, 3, generator=g).to(DEV)
    b = Boundary(bkind, 0.3)
    for ww in (w[:1], w):  # shared and per-sample templates
        a = correlate3x3(t, ww, b, "stencil")
        c = correlate3x3(t, ww, b, "conv")
        torch.testing.assert_close(a, c, atol=1e-5, rtol=1e-5)


def test_modes_agree():
    m = bip(shapes())
    ref = CNNSimulator(device=DEV).run(T.HOLE_FILL, m, steps=400)["x"]
    for kw in [dict(backend="conv"), dict(compile=True), dict(cuda_graph=True), dict(compile=True, cuda_graph=True)]:
        x = CNNSimulator(device=DEV, **kw).run(T.HOLE_FILL, m, steps=400)["x"]
        torch.testing.assert_close(x, ref, atol=1e-4, rtol=1e-4)


def test_rk4_matches_euler_small_dt():
    m = bip(shapes())
    a = CNNSimulator(device=DEV, integrator="rk4").run(T.EDGE, m, steps=100, dt=0.1)["y"]
    b = CNNSimulator(device=DEV).run(T.EDGE, m, steps=1000, dt=0.01)["y"]
    torch.testing.assert_close(a, b, atol=1e-2, rtol=0)
