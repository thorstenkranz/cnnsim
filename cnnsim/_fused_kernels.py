"""Triton kernels of the fused backend (imported lazily by cnnsim.fused)."""

import triton
import triton.language as tl

# --------------------------------------------------------------------------- device helpers


@triton.jit
def _tanh(a):
    return 1.0 - 2.0 / (tl.exp(2.0 * a) + 1.0)


@triton.jit
def _f(x, beta, OUT: tl.constexpr):
    if OUT == 0:
        y = tl.minimum(tl.maximum(x, -1.0), 1.0)
    elif OUT == 1:
        y = _tanh(0.5 * beta * x)
    elif OUT == 2:
        y = _tanh(beta * x)
    elif OUT == 3:
        y = x
    else:
        y = tl.where(x > 0, 1.0, -1.0)
    return y


@triton.jit
def _fprime(x, beta, OUT: tl.constexpr):
    if OUT == 0:
        d = tl.where((x >= -1.0) & (x <= 1.0), 1.0, 0.0)
    elif OUT == 1:
        t = _tanh(0.5 * beta * x)
        d = 0.5 * beta * (1.0 - t * t)
    elif OUT == 2:
        t = _tanh(beta * x)
        d = beta * (1.0 - t * t)
    elif OUT == 3:
        d = x * 0.0 + 1.0
    else:  # surrogate: derivative of clip(beta x)
        d = tl.where((beta * x >= -1.0) & (beta * x <= 1.0), beta, 0.0)
    return d


@triton.jit
def _gather(base, i, j, mask, H, W, bval, di: tl.constexpr, dj: tl.constexpr, BOUND: tl.constexpr):
    """Value of the neighbour at offset (di, dj) of cell (i, j) in the grid at `base`."""
    if BOUND == 0:  # fixed: virtual cells have value bval
        ii = i + di
        jj = j + dj
        ok = mask & (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W)
        v = tl.load(base + ii * W + jj, mask=ok, other=0.0)
        v = tl.where(ok, v, bval)
    elif BOUND == 1:  # zeroflux: clamp
        ii = tl.minimum(tl.maximum(i + di, 0), H - 1)
        jj = tl.minimum(tl.maximum(j + dj, 0), W - 1)
        v = tl.load(base + ii * W + jj, mask=mask, other=0.0)
    elif BOUND == 2:  # periodic
        ii = (i + di + 2 * H) % H
        jj = (j + dj + 2 * W) % W
        v = tl.load(base + ii * W + jj, mask=mask, other=0.0)
    else:  # spiral: flat row-major index, closed ring
        HW = H * W
        p = (i * W + j + di * W + dj + 2 * HW) % HW
        v = tl.load(base + p, mask=mask, other=0.0)
    return v


@triton.jit
def _cand(p, n, d: tl.constexpr, e: tl.constexpr):
    """Zeroflux transpose, 1D: e-th source q with clamp(q + d) == p (e = 0: q = p - d)."""
    if e == 0:
        q = p - d
        ok = (q >= 0) & (q < n)
    else:
        if d < 0:  # sources q in [0, -d-1] all clamp onto p == 0
            q = p * 0 + (e - 1)
            ok = (p == 0) & (e <= -d)
        elif d > 0:  # sources q in [n-d, n-1] all clamp onto p == n-1
            q = p * 0 + (n - e)
            ok = (p == n - 1) & (e <= d)
        else:
            q = p
            ok = p < 0  # never
    return q, ok


@triton.jit
def _gather_T(base, i, j, mask, H, W, di: tl.constexpr, dj: tl.constexpr, R: tl.constexpr, BOUND: tl.constexpr):
    """Σ over all cells q whose neighbour at offset (di, dj) is (i, j) of the grid value at q."""
    if BOUND == 0:  # fixed: virtual cells carry no gradient
        ii = i - di
        jj = j - dj
        ok = mask & (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W)
        v = tl.load(base + ii * W + jj, mask=ok, other=0.0)
    elif BOUND == 1:
        v = tl.zeros(i.shape, dtype=tl.float32) + 0.0 * j
        for ei in tl.static_range(R + 1):
            qi, oki = _cand(i, H, di, ei)
            for ej in tl.static_range(R + 1):
                qj, okj = _cand(j, W, dj, ej)
                ok = mask & oki & okj
                v += tl.load(base + qi * W + qj, mask=ok, other=0.0)
    elif BOUND == 2:
        ii = (i - di + 2 * H) % H
        jj = (j - dj + 2 * W) % W
        v = tl.load(base + ii * W + jj, mask=mask, other=0.0)
    else:
        HW = H * W
        p = (i * W + j - di * W - dj + 2 * HW) % HW
        v = tl.load(base + p, mask=mask, other=0.0)
    return v


# --------------------------------------------------------------------------- kernels


@triton.jit
def _fwd_kernel(
    x0_ptr, bias_ptr, A_ptr, a_stride, xout_ptr, hist_ptr, scratch_ptr,
    H, W, steps, dt, beta, bval,
    K: tl.constexpr, BOUND: tl.constexpr, OUT: tl.constexpr, STORE: tl.constexpr,
    BH: tl.constexpr, BW: tl.constexpr,
):
    n = tl.program_id(0)
    HW = H * W
    i = tl.arange(0, BH)[:, None]
    j = tl.arange(0, BW)[None, :]
    mask = (i < H) & (j < W)
    off = i * W + j
    x = tl.load(x0_ptr + n * HW + off, mask=mask, other=0.0)
    b = tl.load(bias_ptr + n * HW + off, mask=mask, other=0.0)
    scr = scratch_ptr + n * HW
    a_base = A_ptr + n * a_stride
    R: tl.constexpr = K // 2
    for t in range(steps):
        if STORE:
            tl.store(hist_ptr + (n * steps + t) * HW + off, x, mask=mask)
        tl.store(scr + off, _f(x, beta, OUT), mask=mask)
        tl.debug_barrier()
        acc = b
        for ki in tl.static_range(K):
            for kj in tl.static_range(K):
                a = tl.load(a_base + ki * K + kj)
                acc += a * _gather(scr, i, j, mask, H, W, bval, ki - R, kj - R, BOUND)
        tl.debug_barrier()
        x = x + dt * (acc - x)
    tl.store(xout_ptr + n * HW + off, x, mask=mask)


@triton.jit
def _bwd_kernel(
    gT_ptr, hist_ptr, A_ptr, a_stride, scr_y_ptr, scr_g_ptr, gb_ptr, gA_ptr, g0_ptr,
    H, W, steps, dt, beta, bval,
    K: tl.constexpr, BOUND: tl.constexpr, OUT: tl.constexpr, GRAD_A: tl.constexpr,
    BH: tl.constexpr, BW: tl.constexpr,
):
    n = tl.program_id(0)
    HW = H * W
    i = tl.arange(0, BH)[:, None]
    j = tl.arange(0, BW)[None, :]
    mask = (i < H) & (j < W)
    off = i * W + j
    g = tl.load(gT_ptr + n * HW + off, mask=mask, other=0.0)
    gb = tl.zeros((BH, BW), dtype=tl.float32)
    sy = scr_y_ptr + n * HW
    sg = scr_g_ptr + n * HW
    a_base = A_ptr + n * a_stride
    R: tl.constexpr = K // 2
    for s in range(steps):
        t = steps - 1 - s
        x = tl.load(hist_ptr + (n * steps + t) * HW + off, mask=mask, other=0.0)
        tl.store(sy + off, _f(x, beta, OUT), mask=mask)
        tl.store(sg + off, g, mask=mask)
        tl.debug_barrier()
        gb += dt * g
        at = tl.zeros((BH, BW), dtype=tl.float32)
        for ki in tl.static_range(K):
            for kj in tl.static_range(K):
                a = tl.load(a_base + ki * K + kj)
                if GRAD_A:
                    yn = _gather(sy, i, j, mask, H, W, bval, ki - R, kj - R, BOUND)
                    # row sums only (intra-warp); rows are summed once on the host
                    rs = tl.sum(tl.where(mask, g * yn, 0.0), axis=1, keep_dims=True)  # (BH, 1), parent layout
                    pa = gA_ptr + ((n * steps + t) * K * K + ki * K + kj) * BH + i
                    tl.store(pa, dt * rs)  # one slot per step: no read-modify-write
                at += a * _gather_T(sg, i, j, mask, H, W, ki - R, kj - R, R, BOUND)
        tl.debug_barrier()
        g = (1.0 - dt) * g + dt * _fprime(x, beta, OUT) * at
    tl.store(gb_ptr + n * HW + off, gb, mask=mask)
    tl.store(g0_ptr + n * HW + off, g, mask=mask)


