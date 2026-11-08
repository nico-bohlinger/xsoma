"""The README's figures and table, from the evaluation files in results/ (written by xsoma/evaluate.py).

    python tools/make_figures.py [--results results] [--figures figures]   # figures (light and dark), the table on stdout

results/<method>/<run>/eval.jsonl holds one record per checkpoint: Holosoma's tracking metrics, falls and tracking errors of 2048
deterministic 20-s episodes. Time is training time, as in XMimic: start-up and one-time compilation are excluded for every
method (see training_seconds).

The target is Holosoma's own bar for its FastSAC preset (its nightly test: Episode/rew_tracking_lin_vel >= 0.95 and
Episode/rew_tracking_ang_vel >= 0.8), here on deterministic evaluations, held for three consecutive checkpoints. Colors follow
XMimic's release: XSoma in teal, FastSAC and PPO in grays.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS = ("xsoma", "fast_sac", "ppo")
LABEL = {"xsoma": "XSoma", "fast_sac": "Holosoma FastSAC", "ppo": "Holosoma PPO"}
TARGET = {"rew_tracking_lin_vel": 0.95, "rew_tracking_ang_vel": 0.8}
STEPS_PER = {"xsoma": 1, "fast_sac": 4096, "ppo": 24 * 4096}  # env steps per unit of a record's "steps" (Holosoma: iterations)
HELD = 3
BUDGET = 15.0  # minutes shown: every method's comparison is decided within them
METRICS = (("rew_tracking_lin_vel", "Velocity tracking", "Holosoma's Episode/rew_tracking_lin_vel (max 2.0); dashed: the target", 0.95),
           ("err_v", "Velocity error", "m/s, mean |command - velocity| while upright", None),
           ("success", "Episodes without a fall", "share of 2048 episodes of 20 s", None))  # fmt: skip
THEMES = {  # XMimic's release palette: the method in teal, the baselines in grays, ink on white (dark: GitHub's background)
    "light": dict(surface="#ffffff", ink="#1d1d1f", ink2="#48484e", muted="#8e8e93", grid="#e5e5ea", axis="#c7c7cc",
                  xsoma="#00b0be", fast_sac="#6e6e73", ppo="#aeaeb2"),
    "dark": dict(surface="#0d1117", ink="#f0f6fc", ink2="#c9d1d9", muted="#8b949e", grid="#21262d", axis="#30363d",
                 xsoma="#19c3d0", fast_sac="#aeaeb2", ppo="#636366"),
}  # fmt: skip
FONT = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


# ---------------------------------------------------------------- data
def training_seconds(run, rows):
    """Each checkpoint's training time (s), start-up and one-time compilation excluded.

    XSoma compiles before its training clock starts: its log.jsonl records both clocks, so the training time is the
    checkpoint's time since launch minus the start-up. Holosoma compiles (torch.compile, CUDA graphs) inside its training
    loop: train_log.csv holds the time it logs since its loop started ("Total time") per logged iteration, and the first
    logged interval's excess over the steady pace, its one-time compilation, is removed."""
    if os.path.exists(os.path.join(run, "log.jsonl")):
        startup = float(np.median([r["wall_s"] - r["train_s"] for r in map(json.loads, open(os.path.join(run, "log.jsonl")))]))
        return [r["wall_s"] - startup for r in rows]
    if os.path.exists(os.path.join(run, "train_log.csv")):
        it, total = np.loadtxt(os.path.join(run, "train_log.csv"), delimiter=",", skiprows=1, unpack=True)
        pace = float(np.median(np.diff(total[:21]) / np.diff(it[:21])))  # seconds per iteration, steady
        excess = total[0] - pace * (it[0] + 1)  # the first logged interval covers iterations 0 to it[0]
        return [float(np.interp(r["steps"], it, total)) - excess for r in rows]  # Holosoma's steps are iterations
    return [r["wall_s"] for r in rows]  # no log: time since launch


def load_results(root=os.path.join(ROOT, "results")):
    """{method: [rows of each run, sorted by training time]}."""
    out = {}
    for m in METHODS:
        runs = []
        for path in sorted(glob.glob(os.path.join(root, m, "*", "eval.jsonl"))):
            rows = [json.loads(line) for line in open(path)]
            for r, t in zip(rows, training_seconds(os.path.dirname(path), rows)):
                r["train_s"], r["min"] = t, t / 60
            runs.append(sorted(rows, key=lambda r: r["min"]))
        if runs:
            out[m] = runs
    return out


def meets(r):
    return all(r[k] >= v for k, v in TARGET.items())


def target_row(rows, held=HELD):
    """The first checkpoint from which `held` consecutive checkpoints meet the target."""
    ok = [meets(r) for r in rows]
    return next((rows[i] for i in range(len(rows) - held + 1) if all(ok[i : i + held])), None)


def time_to_target(rows, held=HELD):
    """Training minutes of the first checkpoint from which `held` consecutive checkpoints meet the target."""
    r = target_row(rows, held)
    return None if r is None else r["min"]


def value_at(rows, key, minute):
    """A run's metric after some minutes of training: its last checkpoint at or before then."""
    before = [r for r in rows if r["min"] <= minute]
    return before[-1][key] if before else None


def summarize(results):
    out = {}
    for m, runs in results.items():
        held = [time_to_target(rows) for rows in runs]
        first = [time_to_target(rows, 1) for rows in runs]
        steps = [r["steps"] * STEPS_PER[m] for r in (target_row(rows) for rows in runs) if r is not None]
        out[m] = {"runs": len(runs), "held": [x for x in held if x is not None], "first": [x for x in first if x is not None],
                  "steps": steps, "end": float(np.mean([rows[-1]["min"] for rows in runs])),
                  **{f"{k}@end": float(np.mean([rows[-1][k] for rows in runs])) for k in ("err_v", "err_w", "success", "rew_tracking_lin_vel")},
                  **{f"{k}@{t}": float(np.mean([value_at(rows, k, t) for rows in runs])) for k in ("err_v", "err_w", "success", "rew_tracking_lin_vel")
                     for t in (10, BUDGET, 20, 60) if all(value_at(rows, k, t) is not None for rows in runs)}}  # fmt: skip
    return out


def curve(runs, key, grid):
    """Per-run values on the grid, linear between checkpoints (NaN outside a run's first and last checkpoint)."""
    out = []
    for rows in runs:
        t = np.asarray([r["min"] for r in rows])
        v = np.asarray([r[key] for r in rows])
        out.append(np.where((grid >= t[0]) & (grid <= t[-1]), np.interp(grid, t, v), np.nan))
    return np.asarray(out)


# ---------------------------------------------------------------- drawing
def setup(theme):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 11, "svg.fonttype": "none"})
    return THEMES[theme]


def style_axes(ax, t, grid_axis="y"):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["axis"])
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=t["ink2"], labelsize=10, length=0)
    ax.grid(axis=grid_axis, color=t["grid"], lw=1.0)
    ax.set_axisbelow(True)


def rounded_bar(ax, y, value, height, color, radius_px=4.0):
    """A horizontal bar from 0 with a 4 px rounded data end, square at the baseline."""
    x_per_px = (ax.get_xlim()[1] - ax.get_xlim()[0]) / ax.get_window_extent().width
    y_per_px = (ax.get_ylim()[0] - ax.get_ylim()[1]) / ax.get_window_extent().height
    rx, ry = min(radius_px * x_per_px, value / 2), radius_px * abs(y_per_px)
    y0, y1 = y - height / 2, y + height / 2
    verts = [(0, y0), (value - rx, y0), (value, y0), (value, y0 + ry), (value, y1 - ry), (value, y1), (value - rx, y1), (0, y1), (0, y0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", zorder=2))


def learning_curves(results, theme, out):
    t = setup(theme)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), facecolor=t["surface"])
    grid = np.linspace(0, BUDGET, 481)
    for ax, (key, title, sub, line) in zip(axes, METRICS):
        style_axes(ax, t)
        for m in METHODS:
            if m not in results:
                continue
            v = curve(results[m], key, grid)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # grid points no run covers
                mean, lo, hi = np.nanmean(v, 0), np.nanmin(v, 0), np.nanmax(v, 0)
            live = ~np.isnan(v).any(0)  # where every run has data, so the mean never mixes different sets of runs
            ax.fill_between(grid[live], lo[live], hi[live], color=t[m], alpha=0.12, lw=0, zorder=2)
            ax.plot(grid[live], mean[live], color=t[m], lw=2.0, solid_capstyle="round", solid_joinstyle="round", zorder=3, label=LABEL[m])
        if line is not None:
            ax.axhline(line, color=t["muted"], lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.set_xlim(0, BUDGET)
        ax.set_xticks(np.arange(0, BUDGET + 1, 5))
        ax.set_title(title, loc="left", color=t["ink"], fontsize=11.5, pad=18)
        ax.text(0, 1.03, sub, transform=ax.transAxes, color=t["muted"], fontsize=9, va="bottom")
        ax.set_xlabel("minutes of training", color=t["ink2"], fontsize=10)
        if key == "success":
            ax.set_ylim(0, 1.0)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        if key == "err_v":
            ax.set_ylim(0, 1.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False, fontsize=10, labelcolor=t["ink"], bbox_to_anchor=(0.995, 0.995))
    fig.text(0.008, 0.965, "Learning curves on one H100", ha="left", va="center", color=t["ink"], fontsize=13.5, fontweight="bold")
    fig.text(0.008, 0.905, "deterministic evaluation of every checkpoint; line: mean over runs, band: min to max", ha="left", va="center",
             color=t["ink2"], fontsize=10)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=2.5)
    fig.savefig(os.path.join(out, f"learning_curves{'_dark' if theme == 'dark' else ''}.png"), dpi=200, facecolor=t["surface"])
    plt.close(fig)


def time_to_target_figure(summary, theme, out):
    t = setup(theme)
    keys = [m for m in METHODS if m in summary]
    n = len(keys)
    times = [x for m in keys for x in summary[m]["held"]]
    top = max(BUDGET, 5 * np.ceil((max(times, default=10.0) + 3) / 5))  # room for the label right of the last run
    fig_h = 1.3 + 0.62 * n
    fig = plt.figure(figsize=(9.0, fig_h), facecolor=t["surface"])
    ax = fig.add_axes((0.22, 0.45 / fig_h, 0.62, (fig_h - 1.2) / fig_h))
    style_axes(ax, t, grid_axis="x")
    ax.set_xlim(0, top)
    ax.set_ylim(n - 0.4, -0.6)
    fig.canvas.draw()
    for i, m in enumerate(keys):
        s = summary[m]
        runs = f"{s['runs']} run" + ("s" if s["runs"] > 1 else "")
        if s["held"]:
            mean = float(np.mean(s["held"]))
            rounded_bar(ax, i, mean, 0.5, t[m])
            ax.scatter(s["held"], [i] * len(s["held"]), s=34, facecolor=t[m], edgecolors=t["surface"], linewidths=2.0, zorder=3)
            label = f"{mean:.1f} min" + (f"   ({len(s['held'])} of {runs})" if len(s["held"]) < s["runs"] else "")
            ax.text(max(s["held"]) + top * 0.02, i, label, va="center", ha="left", color=t["ink"], fontsize=10.5)
        else:
            ax.text(top * 0.01, i, f"not reached in {5 * round(s['end'] / 5):.0f} min ({runs})", va="center", ha="left", color=t["ink2"], fontsize=10.5)
    ax.set_yticks(range(n), [LABEL[m] for m in keys], color=t["ink"], fontsize=11)
    ax.set_xticks(np.arange(0, top + 1, 5))
    ax.set_xlabel("minutes of training (start-up and compilation excluded)", color=t["ink2"], fontsize=10)
    fig.text(0.012, 1 - 0.30 / fig_h, "Time to Holosoma's FastSAC bar on one H100", ha="left", va="center", color=t["ink"], fontsize=13.5,
             fontweight="bold")  # fmt: skip
    fig.text(0.012, 1 - 0.60 / fig_h, "velocity tracking ≥ 0.95 and yaw-rate tracking ≥ 0.8 (Holosoma's nightly test), held for 3 checkpoints; dots: runs",
             ha="left", va="center", color=t["ink2"], fontsize=10)  # fmt: skip
    fig.savefig(os.path.join(out, f"time_to_target{'_dark' if theme == 'dark' else ''}.png"), dpi=200, facecolor=t["surface"])
    plt.close(fig)


def table(summary):
    ref = float(np.mean(summary["fast_sac"]["held"])) if summary.get("fast_sac", {}).get("held") else None
    lines = [f"| method | runs | time to target (min) | speedup | env steps to target | after {BUDGET:.0f} min: velocity error, falls |",
             "|---|---|---|---|---|---|"]  # fmt: skip
    for m in METHODS:
        if m not in summary:
            continue
        s = summary[m]
        if s["held"]:
            tt = f"{np.mean(s['held']):.1f} ({min(s['held']):.1f}–{max(s['held']):.1f})" if len(s["held"]) > 1 else f"{s['held'][0]:.1f}"
            tt += f", {len(s['held'])} of {s['runs']} runs" if len(s["held"]) < s["runs"] else ""
        else:
            tt = f"not within {5 * round(s['end'] / 5):.0f}"  # the run length
        sp = f"{ref / np.mean(s['held']):.1f}×" if ref and s["held"] and len(s["held"]) == s["runs"] else "–"
        st = f"{np.mean(s['steps']) / 1e6:.0f} M" if s["steps"] else "–"
        at = f"{s[f'err_v@{BUDGET}']:.2f} m/s, {100 * (1 - s[f'success@{BUDGET}']):.1f} %" if f"err_v@{BUDGET}" in s else "–"
        lines.append(f"| {LABEL[m]} | {s['runs']} | {tt} | {sp} | {st} | {at} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="The README's figures and results table.")
    p.add_argument("--results", default=os.path.join(ROOT, "results"))
    p.add_argument("--figures", default=os.path.join(ROOT, "figures"))
    a = p.parse_args()
    os.makedirs(a.figures, exist_ok=True)
    results = load_results(a.results)
    summary = summarize(results)
    for theme in ("light", "dark"):
        learning_curves(results, theme, a.figures)
        time_to_target_figure(summary, theme, a.figures)
    print(table(summary))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
