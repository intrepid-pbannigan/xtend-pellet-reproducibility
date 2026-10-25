"""Main-text figure: the campaign map and the feasibility it had to learn.

Two panels sharing one experiment-number axis, because they are two readings of
the same 51 experiments and the paper argues they were optimised together:

    (a) release deviation for every feasible experiment, coloured by measured
        granule aspect ratio, with failures drawn along a dotted floor as
        unobserved rather than as poor-performing -- the distinction the whole
        argument rests on -- and the running best as a step.

    (b) how many of each batch formed pellets at all, with the logistic trend in
        P(pellets form) and its 95% band over the top. The bars and the curve are
        the same fact at two resolutions; the curve carries the uncertainty,
        which matters because the trend crosses p < 0.05 only once batches 12 and
        13 are included. Over the 2024 campaign alone it is OR 1.16, p = 0.15;
        over all 13 batches, OR 1.27, p = 0.007. Both belong in the paper, and
        the panel title is computed from the fit rather than asserted.

The logistic fit is imported from `analyze_feasibility` rather than refitted,
so the figure and the reported odds ratio cannot drift apart.

    python src/analysis/plot_campaign_and_feasibility.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg
from analysis.analyze_feasibility import batch_trend
from plotting import style


def batch_centre(batch):
    """Batch index on the shared experiment-number axis."""
    return (batch - 1) * cfg.BATCH_SIZE + 2.5


def campaign_panel(ax, df, feasible, failed):
    style.batch_bands(ax, df.batch)
    floor = feasible.release_deviation.max() * 1.12

    ax.scatter(feasible.experiment_num, feasible.release_deviation, s=48,
               c=[style.aspect_colour(a) for a in feasible.granule_aspect_ratio],
               edgecolor="k", linewidth=0.5, zorder=3)
    ax.scatter(failed.experiment_num, np.full(len(failed), floor), marker="x",
               s=38, c=style.FAILED, linewidth=1.5, zorder=3)
    ax.axhline(floor, color=style.FAILED, lw=0.8, ls=":")

    ordered = feasible.sort_values("experiment_num")
    ax.step(ordered.experiment_num, ordered.release_deviation.cummin(), where="post",
            color="k", lw=1.2, ls="--", zorder=2)

    handles = [plt.Line2D([], [], marker="x", ls="", color=style.FAILED,
                          markersize=7, label="failed to form")]
    if feasible.granule_aspect_ratio.isna().any():
        handles += style.aspect_handles()
    handles.append(plt.Line2D([], [], ls="--", color="k", label="best so far"))
    ax.legend(handles=handles, loc="lower left", ncol=2)

    ax.set_ylabel("release deviation from target\n(fraction-hours, lower is better)")
    ax.set_ylim(0, floor * 1.13)
    best = feasible.loc[feasible.release_deviation.idxmin()]
    first = feasible.sort_values("experiment_num").iloc[0]
    ax.set_title(
        f"Release improved {first.release_deviation / best.release_deviation:.1f}-fold over "
        f"{len(df)} experiments, of which {int((~df.feasible).sum())} never formed pellets",
        fontsize=9.5)
    return floor


def feasibility_panel(ax, df):
    """Counts per batch, with the logistic trend and its band over the top."""
    model, grid, band = batch_trend(df)
    per_batch = df.groupby("batch").feasible.agg(["sum", "size"])

    style.batch_bands(ax, df.batch)
    ax.bar([batch_centre(b) for b in per_batch.index], per_batch["sum"].values,
           width=3.4, color=style.FEASIBLE, alpha=0.55, zorder=2)
    for b, n in zip(per_batch.index, per_batch["sum"].values):
        ax.text(batch_centre(b), n + 0.12, str(int(n)), ha="center", fontsize=7.5, zorder=4)
    ax.set_ylim(0, cfg.BATCH_SIZE + 0.55)
    ax.set_yticks([0, 2, 4])
    ax.set_ylabel("feasible\nper batch")
    ax.set_xlabel("experiment number (batches of 4, run in order)")
    ax.set_xlim(0.2, cfg.N_EXPERIMENTS + 0.8)

    # Probability on its own axis rather than rescaled onto the count axis: one
    # batch holds three experiments, not four, because the repeated design point
    # is merged, so count/BATCH_SIZE is not the rate everywhere.
    rate_ax = ax.twinx()
    centres = [batch_centre(b) for b in grid]
    rate_ax.fill_between(centres, band.mean_ci_lower, band.mean_ci_upper,
                         color="k", alpha=0.10, lw=0, zorder=3)
    rate_ax.plot(centres, band["mean"], color="k", lw=1.4, zorder=4)
    rate_ax.scatter([batch_centre(b) for b in per_batch.index],
                    (per_batch["sum"] / per_batch["size"]).values,
                    s=26, color="k", zorder=5)
    rate_ax.set_ylim(-0.02, 1.16)
    rate_ax.set_yticks([0, 0.5, 1.0])
    rate_ax.set_ylabel("P(pellets form)")
    rate_ax.grid(visible=False)

    odds = np.exp(model.params.iloc[1])
    low, high = np.exp(model.conf_int().iloc[1])
    p_value = model.pvalues.iloc[1]
    # Read off the fit rather than asserted. Over the original 44 experiments this
    # trend was positive but not significant, and batches 12 and 13 are what
    # carried it past the conventional threshold; saying "not significant" here
    # would now be a stale claim rather than a cautious one.
    verdict = "improving" if p_value < 0.05 else "improving, not significantly"
    ax.set_title(
        f"Feasibility per batch: odds ratio {odds:.2f} per batch "
        f"(95% CI {low:.2f}-{high:.2f}, p = {p_value:.3f}) -- {verdict}",
        fontsize=9.5)
    return odds, low, high, p_value


def main():
    style.set_style()
    df = pd.read_csv(cfg.CAMPAIGN_CSV)
    feasible = df[df.feasible]
    failed = df[~df.feasible]

    # Drawn at page width with fixed positions, so the submission build keeps
    # this layout: the colour bar beside (a) and the twin axis of (b) share the
    # right margin, and both panels keep one x extent.
    fig = plt.figure(figsize=(style.DOUBLE_COLUMN, 5.3))
    left, width = 0.115, 0.735
    ax = fig.add_axes((left, 0.37, width, 0.58))
    ax2 = fig.add_axes((left, 0.08, width, 0.22), sharex=ax)
    cax = fig.add_axes((left + width + 0.02, 0.37, 0.018, 0.58))
    ax.tick_params(labelbottom=False)

    campaign_panel(ax, df, feasible, failed)
    style.aspect_colourbar(fig, ax, cax=cax)
    odds, low, high, p_value = feasibility_panel(ax2, df)

    style.save(fig, "campaign_and_feasibility")

    first = feasible.sort_values("experiment_num").iloc[0]
    best = feasible.loc[feasible.release_deviation.idxmin()]
    print(f"feasible {int(df.feasible.sum())}/{len(df)} over {cfg.N_BATCHES} batches")
    print(f"first feasible: experiment {int(first.experiment_num)} "
          f"release deviation {first.release_deviation:.3f}")
    print(f"best:           experiment {int(best.experiment_num)} "
          f"release deviation {best.release_deviation:.3f}")
    print(f"improvement factor {first.release_deviation / best.release_deviation:.2f}")
    print(f"feasibility OR {odds:.3f} per batch (95% CI {low:.3f}-{high:.3f}), p = {p_value:.3f}")


if __name__ == "__main__":
    main()
