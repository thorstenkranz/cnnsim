# Simulation

## Running a template

[`CNNSimulator`][cnnsim.simulator.CNNSimulator] integrates one network or a batch of networks
without gradients. A [`Template`][cnnsim.simulator.Template] bundles everything that defines a
network: the templates A and B, the bias z, the initial state, the boundaries and the output
function. `cnnsim.templates` contains classic ones (`EDGE`, `EDGE_GRAY`, `CORNER`, `INVERSION`,
`THRESHOLD`, `HOLE_FILL`, `SHADOW_LEFT`, `CCD_HORIZONTAL`, `AVERAGE`).

```python
import torch
from cnnsim import CNNSimulator, templates as T

u = -torch.ones(32, 32)  # white image ...
u[8:24, 8:24] = 1.0      # ... with a black square

sim = CNNSimulator()
res = sim.run(T.EDGE, u, steps=200, dt=0.1)
edges = res["y"]          # black outline of the square
```

Inputs may have shape `(H, W)`, `(N, H, W)` or `(N, 1, H, W)` with values in [−1, 1]; the results
always have shape `(N, 1, H, W)`. `run` returns a dict:

| key | content |
|---|---|
| `"y"` | final output f(x) |
| `"x"` | final state |
| `"steps"` | number of steps taken |
| `"history"` | list of output snapshots, if `record_every` was given |

`tol` stops the integration early once max |dx/dt| falls below it (checked every `check_every`
steps); `record_every` keeps a snapshot of the output every that many steps.

## Your own templates

```python
from cnnsim import Boundary, Template

my = Template(
    A=[[0, 0, 0], [0, 2, 0], [0, 0, 0]],
    B=[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    z=0.5,
    x0="input",
    boundary=Boundary("zeroflux"),
)
res = sim.run(my, u, steps=100)
```

Templates are immutable (and hashable); keyword arguments of `run` override their fields, and
without a template everything can be given directly: `sim.run(None, u, A=..., B=..., z=...)`.

## Model options

| | options |
|---|---|
| boundary [`Boundary(kind, value)`][cnnsim.boundary.Boundary] | `fixed` (Dirichlet, virtual cells = `value`), `zeroflux` (Neumann), `periodic` (torus), `spiral` (closed spiral, see below) |
| output [`Output(kind, beta)`][cnnsim.outputs.Output] | `pwl` clip(x, −1, 1) · `sigmoid` 2/(1+e^(−βx))−1 · `tanh` tanh(βx) · `sign` (with a surrogate gradient of clip(βx)) · `identity` |
| templates | any odd k×k; A and B independent; shared, or one per sample |
| bias z | scalar, per sample `(N,)`, or a map `(H, W)` / `(N, 1, H, W)` |
| initial state `x0` | `zero`, `input`, `ones`, `minus_ones`, or a tensor |
| input boundary | `boundary_u`: the boundary for B ⊛ u, if it differs from the one for A ⊛ y |

The **closed spiral** boundary treats the grid as one sequence in row-major order, closed into a
ring: the neighbour at offset (di, dj) of the cell with flat index p is the cell
p + di·W + dj (mod H·W). The right neighbour of the last cell in a row is the first cell of the
next row. This is the natural boundary for a time series written row by row into the grid.

```python
from cnnsim import Output

g = torch.Generator().manual_seed(0)
A5 = torch.randn(5, 5, generator=g) * 0.1
B3 = torch.randn(3, 3, generator=g) * 0.3
zmap = torch.linspace(-0.2, 0.2, 32).expand(32, 32)
res = sim.run(
    None, u, A=A5, B=B3, z=zmap, x0="input",
    boundary=Boundary("spiral"), boundary_u=Boundary("fixed", 0.0),
    output=Output("sigmoid", beta=1.0),
)
```

## Batches and parameter sweeps

Templates with a leading batch dimension run one network per template, all in a single batched
simulation. A, B of shape `(N, k, k)` and z of shape `(N,)` broadcast against an input of batch
size 1 or N:

```python
zs = torch.linspace(-1, 1, 16)
A = torch.tensor(T.THRESHOLD.A).expand(16, 3, 3)
res = sim.run(None, u, A=A, B=torch.zeros(1, 1), z=zs, x0="input", steps=200)
res["y"].shape  # (16, 1, 32, 32): one output per bias value
```

## Lower-level access

`run` is `prepare` (turn template and input into tensors, compute the constant B ⊛ u + z once)
followed by `integrate`. Call them separately to integrate the same prepared network repeatedly,
or use `stepper` to advance it chunk by chunk.

The functions in [`cnnsim.dynamics`][cnnsim.dynamics] (`correlate`, `bias_map`, `derivative`,
`make_steps`) are the differentiable core that both the simulator and training build on: pure
tensor functions, and gradients flow through the whole integration.
