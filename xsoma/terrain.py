"""Holosoma's `terrain_locomotion_mix` (upstream 79ba714, `simulator/shared/terrain.py`), ported draw for draw.

A grid of num_rows x num_cols square tiles (8 m, 0.1 m pixels, 5 mm height units) inside a flat 40 m border. Each tile, in
row-major order, draws its type (flat 0.2 / rough 0.6 / low_obstacles 0.2) and a difficulty in {0.5, 0.75, 0.9}:
  flat           height 0
  rough          every pixel U[-0.025 - 2 * 0.025 d / 0.9, -0.025) m, below zero
  low_obstacles  30 square pits of 0.8 m (8 x 8 pixels, may overlap), depth 0.03 d / 0.9 m
Heights are stored as int16 units (float heights / 5 mm, truncated toward zero, as numpy's int16 assignment does). A tile's
origin is its centre at the highest point of its central 1 x 1 m. With `np.random.RandomState(seed)` the draws are the ones
Holosoma makes from the global NumPy generator right after `np.random.seed(seed)`.

The MuJoCo heightfield built here spaces its samples exactly `horizontal_scale` apart (half-size (n - 1) h / 2); Holosoma's
MuJoCo backend uses half-size n h / 2, which stretches the physical ground by n / (n - 1) against the mesh its rewards read
(docs/holosoma_task.md, spec 11). Here physics and height lookups use the same surface: MuJoCo splits
each heightfield cell along the diagonal from its (low x, low y) to its (high x, high y) corner, as Holosoma's trimesh does,
and `height` interpolates on exactly those triangles.
"""

from __future__ import annotations

import dataclasses

import numpy as np

TYPES = ("flat", "rough", "low_obstacles")


@dataclasses.dataclass(frozen=True)
class TerrainCfg:
    num_rows: int = 10
    num_cols: int = 20
    tile: float = 8.0
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    border: float = 40.0
    proportions: tuple = (0.2, 0.6, 0.2)  # flat, rough, low_obstacles
    seed: int = 42

    @classmethod
    def scaled(cls, factor: float = 1.0, seed: int = 42) -> TerrainCfg:
        """Holosoma's `scale_factor` (README: 0.5 for PPO on MJWarp): fewer and smaller tiles, a narrower border and a finer
        height unit; heights in metres keep their distributions, pits shrink with the tile."""
        return cls(num_rows=int(max(1, 10 * factor)), num_cols=int(max(1, 20 * factor)), tile=float(max(1, int(8.0 * factor))),
                   vertical_scale=0.005 * factor, border=40.0 * factor, seed=seed)  # fmt: skip


@dataclasses.dataclass(frozen=True)
class Terrain:
    cfg: TerrainCfg
    heights: np.ndarray  # int16 [rows (x), cols (y)], units of vertical_scale
    origins: np.ndarray  # [num_rows, num_cols, 3] tile centres (m), z = max height of the central 1 x 1 m
    types: np.ndarray  # [num_rows, num_cols] index into TYPES
    difficulty: np.ndarray  # [num_rows, num_cols]

    @property
    def heights_m(self) -> np.ndarray:
        return self.heights.astype(np.float64) * self.cfg.vertical_scale

    @property
    def x0(self) -> float:
        """World x (and y) of pixel (0, 0): the border lies at negative coordinates."""
        return -self.cfg.border


def generate(cfg: TerrainCfg = TerrainCfg()) -> Terrain:
    rng = np.random.RandomState(cfg.seed)
    hs, vs = cfg.horizontal_scale, cfg.vertical_scale
    tp = int(cfg.tile / hs)  # pixels per tile side (80)
    border = max(1, int(cfg.border / hs))
    heights = np.zeros((cfg.num_rows * tp + 2 * border, cfg.num_cols * tp + 2 * border), np.int16)
    origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
    types = np.zeros((cfg.num_rows, cfg.num_cols), np.int32)
    difficulty = np.zeros((cfg.num_rows, cfg.num_cols))
    names = [t for t, q in zip(TYPES, cfg.proportions) if q > 0.0]
    p = np.asarray([q for q in cfg.proportions if q > 0.0], np.float64)
    p = p / p.sum()
    c1, c2 = int((cfg.tile / 2 - 0.5) / hs), int((cfg.tile / 2 + 0.5) / hs)
    for k in range(cfg.num_rows * cfg.num_cols):
        i, j = np.unravel_index(k, (cfg.num_rows, cfg.num_cols))
        kind = rng.choice(names, p=p)
        d = rng.choice([0.5, 0.75, 0.9])
        tile = np.zeros((tp, tp), np.int16)
        if kind == "rough":
            max_h = 0.025 * d / 0.9
            tile = rng.uniform(-max_h * 2 - 0.025, -0.025, (tp, tp)) / vs  # float; truncated toward zero below
        elif kind == "low_obstacles":
            max_h = 0.03 * d / 0.9
            size, num = tp // 10, 30
            xs = rng.randint(0, tp - size, (num,))
            ys = rng.randint(0, tp - size, (num,))
            for n in range(num):
                tile[xs[n] : xs[n] + size, ys[n] : ys[n] + size] = -max_h / vs  # int16 assignment truncates
        heights[border + i * tp : border + (i + 1) * tp, border + j * tp : border + (j + 1) * tp] = np.asarray(tile).astype(np.int16)
        # Holosoma takes the origin height from the tile as generated (rough tiles: before int16 truncation)
        origins[i, j] = ((i + 0.5) * cfg.tile, (j + 0.5) * cfg.tile, np.max(tile[c1:c2, c1:c2]) * vs)
        types[i, j], difficulty[i, j] = TYPES.index(kind), d
    return Terrain(cfg, heights, origins, types, difficulty)


def sample_env_origins(terrain: Terrain, num_envs: int, rng: np.random.RandomState) -> np.ndarray:
    """Holosoma's `sample_env_origins`: a random row (difficulty axis) per env, column by env index block."""
    rows = rng.randint(0, terrain.cfg.num_rows, (num_envs,))
    cols = np.floor_divide(np.arange(num_envs), num_envs / terrain.cfg.num_cols).astype(np.int32)
    return terrain.origins[rows, cols]


def mujoco_hfield(terrain: Terrain) -> dict:
    """MjSpec arguments for an hfield whose samples sit exactly `horizontal_scale` apart.

    MuJoCo hfield rows run along y and columns along x, so the (x, y) int16 map is transposed. Returns the hfield fields
    (nrow, ncol, size, userdata) and the geom position that puts pixel (0, 0) at world (x0, x0).
    """
    h = terrain.heights_m
    lo, hi = float(h.min()), float(h.max())
    span = max(hi - lo, 1e-6)
    nx, ny = h.shape
    hs = terrain.cfg.horizontal_scale
    rx, ry = (nx - 1) * hs / 2, (ny - 1) * hs / 2
    data = ((h - lo) / span).T.astype(np.float32)  # [ny (rows), nx (cols)]
    return {"nrow": ny, "ncol": nx, "size": [rx, ry, span, 0.1], "userdata": data.ravel().tolist(),
            "pos": [terrain.x0 + rx, terrain.x0 + ry, lo]}  # fmt: skip


def height(heights, x0: float, hs: float, x, y):
    """Terrain height at world (x, y) on MuJoCo's heightfield triangles; heights (nx, ny) in metres, pixel (0, 0) at (x0, x0).

    Differentiable in x and y (the slope of the triangle) for imagination. Outside the map the edge pixels extend flat."""
    import jax.numpy as jnp  # the generator and the MuJoCo export work without JAX

    nx, ny = heights.shape
    fx, fy = (x - x0) / hs, (y - x0) / hs
    i = jnp.clip(jnp.floor(fx), 0, nx - 2).astype(jnp.int32)
    j = jnp.clip(jnp.floor(fy), 0, ny - 2).astype(jnp.int32)
    u, v = jnp.clip(fx - i, 0.0, 1.0), jnp.clip(fy - j, 0.0, 1.0)
    h00, h10, h01, h11 = heights[i, j], heights[i + 1, j], heights[i, j + 1], heights[i + 1, j + 1]
    return jnp.where(u >= v, h00 + u * (h10 - h00) + v * (h11 - h10), h00 + v * (h01 - h00) + u * (h11 - h01))
