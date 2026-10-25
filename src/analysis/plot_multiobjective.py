"""Main-text Figure 5: progress on all three objectives, and the trade it made.

    (a) release deviation against batch        minimise
    (b) median granule aspect ratio            minimise, 1.0 is a sphere
    (c) drug mass per 500 mg                   maximise
    (d) release against aspect ratio, with the Pareto front, coloured by drug
        loading

The campaign is drawn as one series, without batch colours.

Release improved at the cost of morphology: the median aspect ratio rose from
1.18 before batch 10 to 1.46 in batches 10-11 (printed below). Morphology
entered the objective as a continuous aspect ratio at batch 12.

Release is drawn as `release_deviation`, not `release_area_diff`. The latter is
the column the campaign optimised, but it is hybrid -- opaque for the 2024 rows,
restated for 27 and 45-52 -- so a campaign-spanning axis on it mixes two
scales. For the same reason the release Chimera bar is
not drawn on panel (d): it is a threshold on the other column. The aspect bar is
left off too, so nothing competes with the drug-loading colour; the caption
gives it. The starred point is the Chimera incumbent, filled with its own
drug-loading colour; it is F49 on both release metrics.

The Chimera ranking is a retrospective rescoring with the hierarchy of
Table S2. Panel functions come from `plot_optimization_trajectories.py`.

    python src/analysis/plot_multiobjective.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg
from analysis.plot_optimization_trajectories import (
    draw_trajectory, incumbent_row, load_complete, pareto_mask)
from plotting import style

PANELS = (
    ("release_deviation", "(a) Release deviation from target",
     "release deviation (fraction-hours)", "min"),
    ("granule_aspect_ratio", "(b) Pellet aspect ratio",
     "median pellet aspect ratio", "min"),
    ("drug_mass", "(c) Drug loading", "drug loading (mg per 500 mg)", "max"),
)
DRUG_CMAP = "viridis"


def pareto_panel(ax, frame, cax):
    cost = np.column_stack([frame.release_deviation, frame.granule_aspect_ratio])
    front = pareto_mask(cost)
    points = ax.scatter(frame.release_deviation, frame.granule_aspect_ratio, s=44,
                        c=frame.drug_mass, cmap=DRUG_CMAP, vmin=50, vmax=200,
                        edgecolor="k", linewidth=0.45, zorder=3)
    bar = ax.figure.colorbar(points, cax=cax)
    bar.set_label("drug loading (mg per 500 mg)")
    edge = frame[front].sort_values("release_deviation")
    # The full staircase, from the top edge down to the first point and out
    # past the last, so the front visibly reaches F49 under its star.
    top = frame.granule_aspect_ratio.max() + 1.0
    far = frame.release_deviation.max() + 1.0
    xs = np.r_[edge.release_deviation.iloc[0], edge.release_deviation, far]
    ys = np.r_[top, edge.granule_aspect_ratio, edge.granule_aspect_ratio.iloc[-1]]
    ax.step(xs, ys, where="post", color="k", lw=1.2, ls="--", zorder=2)
    best = incumbent_row(frame, "chimera_merit", "min")
    ax.scatter([best.release_deviation], [best.granule_aspect_ratio], s=220, marker="*",
               c=[best.drug_mass], cmap=DRUG_CMAP, vmin=50, vmax=200, edgecolor="k",
               linewidth=1.0, zorder=5)
    span = frame.release_deviation.max() - frame.release_deviation.min()
    # Extra room on the left holds the F49 label clear of the front's vertical.
    ax.set_xlim(frame.release_deviation.min() - 0.19 * span,
                frame.release_deviation.max() + 0.05 * span)
    height = frame.granule_aspect_ratio.max() - frame.granule_aspect_ratio.min()
    ax.set_ylim(frame.granule_aspect_ratio.min() - 0.1 * height,
                frame.granule_aspect_ratio.max() + 0.05 * height)
    # Front IDs are data, kept in the clean build. Offsets are placed by hand
    # for this front, clear of the neighbouring points.
    offsets = {49: (-6, 6, "right", "bottom"), 50: (2, -4, "left", "top"),
               48: (5, -4, "left", "top"), 36: (-7, 0, "right", "center")}
    for r in edge.itertuples():
        dx, dy, ha, va = offsets.get(int(r.experiment_num), (6, 6, "left", "bottom"))
        ax.annotate(f"F{int(r.experiment_num)}", (r.release_deviation, r.granule_aspect_ratio),
                    xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=7.5,
                    fontweight="bold", zorder=6, gid=style.DATA_LABEL)
    ax.set_xlabel("release deviation (fraction-hours, lower is better)")
    ax.set_ylabel("median pellet aspect ratio")
    ax.set_title(f"(d) The trade: {int(front.sum())} of {len(frame)} feasible runs "
                 "are non-dominated", fontsize=9, loc="left")
    return edge


def main():
    style.set_style()
    frame = load_complete()

    # A dedicated colour-bar column: a colour bar attached with `ax=` breaks the
    # constrained-layout pass of build_clean_figures.
    fig = plt.figure(figsize=(10.4, 7.6))
    # The right column is split in both rows so (b) and (d) keep one width; only
    # the lower slot holds the bar.
    grid = fig.add_gridspec(2, 2, wspace=0.25)
    right = [grid[r, 1].subgridspec(1, 2, width_ratios=[1, 0.035], wspace=0.04) for r in range(2)]
    axes = np.array([[fig.add_subplot(grid[0, 0]), fig.add_subplot(right[0][0, 0])],
                     [fig.add_subplot(grid[1, 0]), fig.add_subplot(right[1][0, 0])]])
    for ax, (column, title, units, goal) in zip(axes.ravel()[:3], PANELS):
        draw_trajectory(ax, frame, column, units, goal, color=style.FEASIBLE)
        ax.set_title(title, fontsize=9, loc="left")
    edge = pareto_panel(axes.ravel()[3], frame, fig.add_subplot(right[1][0, 1]))

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=style.FEASIBLE,
                   markeredgecolor="k", markeredgewidth=0.4, label="formed formulation"),
        plt.Line2D([], [], ls="--", color="k", label="best so far / Pareto front"),
        plt.Line2D([], [], marker="*", ls="", color=style.TARGET, markeredgecolor="k",
                   markeredgewidth=0.4, markersize=10, label="best for the objective (a-c)"),
        plt.Line2D([], [], marker="*", ls="", markerfacecolor="w", markeredgecolor="k",
                   markeredgewidth=0.8, markersize=11,
                   label="Chimera best (d), filled by drug loading"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.005),
               fontsize=7.5)
    fig.subplots_adjust(left=0.07, right=0.95, top=0.96, bottom=0.12, hspace=0.32)
    style.save(fig, "multiobjective")

    late_batches = frame.batch.isin(cfg.LATE_BATCHES)
    early = frame[frame.batch < min(cfg.LATE_BATCHES)].granule_aspect_ratio.median()
    late = frame[late_batches].granule_aspect_ratio.median()
    print(f"aspect ratio median, batches before {min(cfg.LATE_BATCHES)}: {early:.3f}; "
          f"batches {cfg.LATE_BATCHES}: {late:.3f} ")
    print("release/aspect Pareto front (release_deviation):",
          ", ".join(f"F{int(e)}" for e in edge.experiment_num))


if __name__ == "__main__":
    main()
