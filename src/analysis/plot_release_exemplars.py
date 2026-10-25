"""Main-text Figure 4: release against the zero-order target, early and best.

Two panels, each comparing one experiment's mean profile with the target and shading the
area between them, which is what `release_deviation` measures:

    (a) experiment 43, the best of the 2024 campaign
    (b) experiment 49, the best on record, from batch 13

Experiment 4, the first feasible experiment, is drawn faded in both as the
starting point.

Profiles are the official ones: every capsule of an experiment, pooled across
assays, except experiment 27's superseded 2024 curve. Error bars are the
standard error over capsules, so their size reflects how many capsules there
are as well as how much they disagree; n is printed in each legend.

Fold-reductions are computed from `release_deviation` in `campaign.csv`.

Every profile here ends at exactly 100% with no error bar at 240 min, and that is
a property of the normalisation, not of the pellets. Each capsule is divided by
its own largest released mass (`frac_release` in the dataset), so any
profile still rising at 240 min finishes at 1.0 by construction, and incomplete
release cannot show up as a gap at the end. The deviation metric inherits the
same blind spot at the final timepoint.

    python src/analysis/plot_release_exemplars.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg
from plotting import style

BASELINE = 4
PANELS = ((43, "batch 11, best of the 2024 campaign"),
          (49, "batch 13, best on record"))
MEASURED = style.TARGET
BASE_COLOUR = "#E8B08A"


def capsule_profiles(experiment):
    """Percent released per capsule, rows capsules, columns the eight timepoints."""
    capsules = pd.read_csv(cfg.RELEASE_REPLICATES_CSV)
    superseded = capsules.apply(
        lambda r: cfg.RELEASE_SUPERSEDE_SERIES.get(int(r.experiment_num)) == r.series, axis=1)
    rows = capsules[~superseded & (capsules.experiment_num == experiment)]
    wide = rows.pivot_table(index=["series", "replicate"], columns="time_min",
                            values="frac_release")
    wide.insert(0, 0.0, 0.0)
    return 100 * wide.to_numpy(float)


def draw(ax, experiment, label, campaign):
    t = cfg.DISSOLUTION_TIMES_MIN
    target = 100 * np.asarray(cfg.TARGET_PROFILE)

    base = capsule_profiles(BASELINE)
    ax.errorbar(t, base.mean(0), yerr=base.std(0, ddof=1) / np.sqrt(len(base)),
                color=BASE_COLOUR, alpha=0.6, lw=1.2, ls="--", marker="o", ms=4,
                capsize=2, label=f"experiment {BASELINE}, first feasible (n = {len(base)})",
                zorder=2)

    profile = capsule_profiles(experiment)
    mean, sem = profile.mean(0), profile.std(0, ddof=1) / np.sqrt(len(profile))
    ax.fill_between(t, mean, target, color=MEASURED, alpha=0.15, lw=0, zorder=1)
    ax.errorbar(t, mean, yerr=sem, color=MEASURED, lw=1.6, ls="--", marker="o", ms=5,
                markeredgecolor="k", markeredgewidth=0.5, capsize=2.5,
                label=f"experiment {experiment} (n = {len(profile)})", zorder=4)
    ax.plot(t, target, color="#1B2A3A", lw=1.2, ls=":", marker="D", ms=5, zorder=3,
            label="target, zero order over 240 min")

    first = campaign.loc[BASELINE, "release_deviation"]
    this = campaign.loc[experiment, "release_deviation"]
    ax.annotate(f"{first / this:.1f}x lower release deviation\nthan experiment {BASELINE} "
                f"({this:.2f} vs {first:.2f})",
                xy=(150, np.interp(150, t, (mean + target) / 2)), xytext=(125, 22),
                fontsize=7.5, ha="left",
                bbox=dict(boxstyle="round,pad=0.35", fc="w", ec="#BBBBBB"),
                arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8,
                                connectionstyle="arc3,rad=0.25"))
    ax.set_title(f"({'ab'[[p[0] for p in PANELS].index(experiment)]}) experiment "
                 f"{experiment}: {label}", fontsize=9, loc="left")
    ax.set_ylabel("drug released (%)")
    ax.set_ylim(-2, 108)
    ax.set_xlim(-5, 250)
    ax.legend(loc="lower right", fontsize=7)
    return first / this


def main():
    style.set_style()
    campaign = pd.read_csv(cfg.CAMPAIGN_CSV).set_index("experiment_num")

    # Stacked, at page width, so each profile reads at full width in the text.
    fig, axes = plt.subplots(2, 1, figsize=(style.DOUBLE_COLUMN, 6.6), sharex=True)
    folds = [draw(ax, e, label, campaign) for ax, (e, label) in zip(axes, PANELS)]
    axes[1].set_xlabel("time (min)")
    fig.tight_layout(h_pad=1.5)
    style.save(fig, "release_exemplars")

    for (e, _), fold in zip(PANELS, folds):
        print(f"experiment {e}: {fold:.1f}x lower release_deviation than experiment {BASELINE}")


if __name__ == "__main__":
    main()
