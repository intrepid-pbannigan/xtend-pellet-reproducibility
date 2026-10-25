"""Observed optimization trajectories and the release–aspect Pareto front.

The three product objectives, plus the Chimera scalarizer that ranks them,
plotted against batch. Each panel is the campaign as it actually unfolded:
every feasible experiment, a running incumbent, and the current best marked.

This is not a surrogate diagnostic. It needs only the measured campaign table
and the vendored Chimera scalarizer, applied retrospectively. The bars (Table
S2) live in `objectives.py` so they can be edited without touching the rest of
the stack.

The Pareto panel is the pair the empty-corner claim is about: minimize
`release_area_diff` and minimize `granule_aspect_ratio` together. Drug mass
is not in that front because it is a design variable, not a measurement.

    python src/analysis/plot_optimization_trajectories.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg
import objectives as chimera_settings
from plotting import style

OBJECTIVES = (
    ("release_area_diff", "Release area difference",
     "fraction-hours, lower is better", "min"),
    ("granule_aspect_ratio", "Pellet aspect ratio",
     "1.0 is a sphere, lower is better", "min"),
    ("drug_mass", "Drug mass",
     "mg per 500 mg batch, higher is better", "max"),
    ("chimera_merit", "Chimera merit",
     "scalarized hierarchy, lower is better", "min"),
)


def load_chimera():
    path = cfg.ROOT / "src" / "vendor" / "chimera.py"
    spec = importlib.util.spec_from_file_location("vendored_chimera", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Chimera


def load_complete():
    """Feasible rows with a full objective vector."""
    df = pd.read_csv(cfg.CAMPAIGN_CSV)
    complete = df[df.feasible & df.release_area_diff.notna()
                  & df.granule_aspect_ratio.notna()
                  & df.drug_mass.notna()].copy()
    complete = complete.sort_values(["batch", "experiment_num"]).reset_index(drop=True)
    chimera = load_chimera()(
        tolerances=list(chimera_settings.TOLERANCES),
        absolutes=list(chimera_settings.ABSOLUTES),
        goals=list(chimera_settings.GOALS),
    )
    observed = complete[list(chimera_settings.OBJECTIVES)].to_numpy(float)
    complete["chimera_merit"] = chimera.scalarize(observed)
    return complete


def pareto_mask(cost):
    mask = np.ones(len(cost), dtype=bool)
    for i, point in enumerate(cost):
        if not mask[i]:
            continue
        dominated = np.all(cost <= point, axis=1) & np.any(cost < point, axis=1)
        if dominated.any():
            mask[i] = False
    return mask


def running_best(values, batches, goal):
    """Incumbent after each batch, aligned to the unique batch axis."""
    frame = pd.DataFrame({"batch": batches, "value": values})
    if goal == "min":
        frame["best"] = frame.value.cummin()
    else:
        frame["best"] = frame.value.cummax()
    return frame.groupby("batch", sort=True).best.last()


def incumbent_row(frame, column, goal):
    idx = frame[column].idxmin() if goal == "min" else frame[column].idxmax()
    return frame.loc[idx]


def point_color(batch):
    if batch == 13:
        return style.BATCH13
    if batch == 12:
        return style.BATCH12
    return style.FEASIBLE


def draw_trajectory(ax, frame, column, ylabel, goal, color=None):
    colors = color or [point_color(int(b)) for b in frame.batch]
    ax.scatter(frame.batch, frame[column], s=42, c=colors, edgecolor="k",
               linewidth=0.45, zorder=3)
    best = running_best(frame[column].to_numpy(), frame.batch.to_numpy(), goal)
    ax.step(best.index, best.to_numpy(), where="post", color="k", lw=1.2,
            ls="--", zorder=2)
    holder = incumbent_row(frame, column, goal)
    ax.scatter([holder.batch], [holder[column]], s=140, marker="*",
               c=style.TARGET, edgecolor="k", linewidth=0.5, zorder=4)
    ax.annotate(
        f"F{int(holder.experiment_num)}",
        (holder.batch, holder[column]),
        textcoords="offset points",
        xytext=(7, 6),
        fontsize=7.5,
        color=style.TARGET,
        fontweight="bold",
    )
    ax.set_xlabel("batch")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, cfg.N_BATCHES + 1))
    ax.set_xlim(0.4, cfg.N_BATCHES + 0.6)


def trajectory_figure(frame):
    style.set_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2))
    for ax, (column, title, units, goal) in zip(axes.ravel(), OBJECTIVES):
        draw_trajectory(ax, frame, column, units, goal)
        holder = incumbent_row(frame, column, goal)
        ax.set_title(f"{title}  ·  F{int(holder.experiment_num)} = {holder[column]:.2f}")

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=style.FEASIBLE,
                   markeredgecolor="k", markeredgewidth=0.4, label="batches 1–11"),
        plt.Line2D([], [], marker="o", ls="", color=style.BATCH12,
                   markeredgecolor="k", markeredgewidth=0.4, label="batch 12"),
        plt.Line2D([], [], marker="o", ls="", color=style.BATCH13,
                   markeredgecolor="k", markeredgewidth=0.4, label="batch 13"),
        plt.Line2D([], [], ls="--", color="k", label="best so far"),
        plt.Line2D([], [], marker="*", ls="", color=style.TARGET,
                   markeredgecolor="k", markeredgewidth=0.4, markersize=10,
                   label="campaign best"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.012))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.text(
        0.5, 0.002,
        chimera_settings.figure_note()
        + " Edit src/objectives.py. Drug mass is drug_ww_frac × 500 mg.",
        ha="center", fontsize=7.5, color="#444444",
    )
    style.save(fig, "optimization_trajectories")


def pareto_figure(frame):
    style.set_style()
    cost = np.column_stack([frame.release_area_diff, frame.granule_aspect_ratio])
    front = pareto_mask(cost)
    frame = frame.copy()
    frame["on_release_aspect_front"] = front

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    colors = [point_color(int(b)) for b in frame.batch]
    ax.scatter(frame.release_area_diff, frame.granule_aspect_ratio, s=48,
               c=colors, edgecolor="k", linewidth=0.45, zorder=3)
    edge = frame[front].sort_values("release_area_diff")
    ax.step(edge.release_area_diff, edge.granule_aspect_ratio, where="post",
            color="k", lw=1.2, ls="--", zorder=2)
    chimera_best = incumbent_row(frame, "chimera_merit", "min")
    ax.scatter([chimera_best.release_area_diff], [chimera_best.granule_aspect_ratio],
               s=150, marker="*", c=style.TARGET, edgecolor="k", linewidth=0.5, zorder=5)

    release_bar = chimera_settings.row_for("release_area_diff")
    aspect_bar = chimera_settings.row_for("granule_aspect_ratio")
    if release_bar[3] and release_bar[1] == "min":
        ax.axvline(release_bar[2], color=style.TARGET, lw=0.9, ls=":", zorder=1)
    if aspect_bar[3] and aspect_bar[1] == "min":
        ax.axhline(aspect_bar[2], color=style.TARGET, lw=0.9, ls=":", zorder=1)
    ax.set_xlabel("release area difference (lower is better)")
    ax.set_ylabel("median pellet aspect ratio (lower is rounder)")
    xpad = 0.06 * (frame.release_area_diff.max() - frame.release_area_diff.min())
    ypad = 0.06 * (frame.granule_aspect_ratio.max() - frame.granule_aspect_ratio.min())
    ax.set_xlim(frame.release_area_diff.min() - xpad, frame.release_area_diff.max() + xpad)
    ax.set_ylim(frame.granule_aspect_ratio.min() - ypad, frame.granule_aspect_ratio.max() + ypad)
    ax.set_title(
        f"Release vs aspect ratio  ·  {int(front.sum())} of {len(frame)} "
        "feasible runs are non-dominated"
    )
    items = []
    for _, row in frame.iterrows():
        if not bool(row.on_release_aspect_front):
            continue
        items.append((row.release_area_diff, row.granule_aspect_ratio,
                      f"F{int(row.experiment_num)}", "k"))
    avoid = list(zip(frame.release_area_diff, frame.granule_aspect_ratio))
    style.place_labels(ax, items, avoid=avoid, fontsize=7)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=style.FEASIBLE,
                   markeredgecolor="k", markeredgewidth=0.4, label="batches 1–11"),
        plt.Line2D([], [], marker="o", ls="", color=style.BATCH12,
                   markeredgecolor="k", markeredgewidth=0.4, label="batch 12"),
        plt.Line2D([], [], marker="o", ls="", color=style.BATCH13,
                   markeredgecolor="k", markeredgewidth=0.4, label="batch 13"),
        plt.Line2D([], [], ls="--", color="k", label="Pareto front"),
        plt.Line2D([], [], ls=":", color=style.TARGET, label="Chimera bars"),
        plt.Line2D([], [], marker="*", ls="", color=style.TARGET,
                   markeredgecolor="k", markeredgewidth=0.4, markersize=10,
                   label=f"Chimera best (F{int(chimera_best.experiment_num)})"),
    ]
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(
        0.5, 0.01,
        chimera_settings.figure_note()
        + " Dotted lines are those bars. The empty lower-left is the accepted corner.",
        ha="center", fontsize=7.5, color="#444444",
    )
    style.save(fig, "pareto_release_aspect")
    return frame


def write_tables(frame):
    cfg.TABLES.mkdir(parents=True, exist_ok=True)
    out = frame[[
        "experiment_num", "batch", "release_area_diff", "granule_aspect_ratio",
        "drug_mass", "chimera_merit", "on_release_aspect_front",
    ]].copy()
    for column, _, _, goal in OBJECTIVES:
        holder = incumbent_row(frame, column, goal)
        out[f"best_{column}"] = out.experiment_num.eq(holder.experiment_num)
    out.round(4).to_csv(cfg.TABLES / "optimization_trajectories.csv", index=False)
    front = out[out.on_release_aspect_front]
    front.to_csv(cfg.TABLES / "pareto_release_aspect.csv", index=False)
    return out


def passes_bar(series, goal, tolerance, absolute, frame=None):
    """Who clears one Chimera bar, using the same rule as an absolute gate.

    Relative bars still depend on the observed range of the current set, so
    this helper is only used to print the absolute gates. Relative membership
    is left to Chimera itself.
    """
    if not absolute:
        return pd.Series(False, index=series.index)
    if goal == "min":
        return series < tolerance
    return series > tolerance  # Chimera's max gate is strict, same as this


def report(frame):
    print(f"complete objective vectors: {len(frame)} "
          f"(feasible {int(pd.read_csv(cfg.CAMPAIGN_CSV).feasible.sum())} / "
          f"{len(pd.read_csv(cfg.CAMPAIGN_CSV))})")
    print()
    print("Chimera bars (edit src/objectives.py):")
    print(f"  {chimera_settings.describe()}")
    keep = pd.Series(True, index=frame.index)
    for name, goal, tolerance, absolute in chimera_settings.HIERARCHY:
        if absolute:
            ok = passes_bar(frame[name], goal, tolerance, absolute)
            print(f"  {chimera_settings.bar_text(name, goal, tolerance, absolute)}: "
                  f"{int(ok.sum())}/{len(frame)}  "
                  f"F{', F'.join(str(int(e)) for e in frame.loc[ok, 'experiment_num'])}")
            keep &= ok
        else:
            print(f"  {chimera_settings.bar_text(name, goal, tolerance, absolute)}: "
                  "relative, scored by Chimera on the observed range")
    if keep.any():
        inside = frame[keep]
        print(f"  inside every absolute bar: "
              f"F{', F'.join(str(int(e)) for e in inside.experiment_num)}")
    else:
        print("  inside every absolute bar: none")
    print()
    print("campaign incumbents under the retrospective Chimera hierarchy:")
    for column, title, _, goal in OBJECTIVES:
        holder = incumbent_row(frame, column, goal)
        print(f"  {title:<24}  experiment {int(holder.experiment_num):>2}  "
              f"batch {int(holder.batch):>2}  {column} = {holder[column]:.3f}")
    print()
    print("when each incumbent first appeared:")
    for column, title, _, goal in OBJECTIVES:
        best = running_best(frame[column].to_numpy(), frame.batch.to_numpy(), goal)
        holder = incumbent_row(frame, column, goal)
        first = next(int(b) for b, v in best.items()
                     if np.isclose(v, holder[column]))
        print(f"  {title:<24}  reached at batch {first}")
    print()
    front = frame[frame.on_release_aspect_front].sort_values("release_area_diff")
    print("release–aspect Pareto front:")
    print(front[["experiment_num", "batch", "release_area_diff",
                 "granule_aspect_ratio", "drug_mass", "chimera_merit"]]
          .round(3).to_string(index=False))


def main():
    frame = load_complete()
    trajectory_figure(frame)
    frame = pareto_figure(frame)
    write_tables(frame)
    report(frame)


if __name__ == "__main__":
    main()
