"""Run the bundled templates on a synthetic image and save a figure.

python demo.py                  # -> output/templates.png
python demo.py --image foo.png  # use your own image (converted to grayscale)
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cnnsim import CNNSimulator
from cnnsim import templates as T


def synthetic(size=128) -> np.ndarray:
    """Bipolar test image: shapes with holes, a bar, a diagonal, salt noise."""
    yy, xx = np.mgrid[:size, :size]
    m = np.zeros((size, size), bool)
    m |= (xx - 35) ** 2 + (yy - 35) ** 2 < 22**2  # disc ...
    m &= ~((xx - 35) ** 2 + (yy - 35) ** 2 < 9**2)  # ... with a hole (ring)
    m[70:110, 15:55] = True  # square ...
    m[80:90, 25:35] = False  # ... with two holes
    m[95:100, 40:47] = False
    m[20:28, 70:120] = True  # bar
    m |= (np.abs(xx - yy - 10) < 3) & (xx > 70) & (yy > 50)  # diagonal stripe
    rng = np.random.default_rng(1)
    m ^= rng.random(m.shape) < 0.01  # salt & pepper noise
    return np.where(m, 1.0, -1.0)


def load_image(path, size) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("L").resize((size, size))
    return 1.0 - 2.0 * np.asarray(img, dtype=np.float32) / 255.0  # dark -> +1 (black)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--out", type=Path, default=Path("output/templates.png"))
    args = p.parse_args()

    u_np = load_image(args.image, args.size) if args.image else synthetic(args.size)
    u = torch.tensor(u_np, dtype=torch.float32)
    sim = CNNSimulator(compile=False)

    panels = [("input", u_np)]
    for name, tmpl in T.ALL.items():
        r = sim.run(tmpl, u, steps=5000, dt=0.1, tol=1e-3, check_every=100)
        panels.append((f"{name}\n({r['steps']} steps)", r["y"][0, 0].cpu().numpy()))

    cols = 5
    rows = -(-len(panels) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.3 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, img) in zip(axes.flat, panels, strict=False):
        ax.imshow(img, cmap="gray_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
