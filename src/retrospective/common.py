"""Shared paths and data for the retrospective surrogate analysis."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as cfg  # noqa: E402

OUTPUTS = cfg.OUTPUTS / "retrospective"
CACHE = OUTPUTS / "cache"
FLUIDS_CSV = cfg.FLUID_DESCRIPTORS_CSV

RESPONSES = ("feasibility", "release deviation", "aspect ratio")
KIND = {"feasibility": "binary", "release deviation": "continuous",
        "aspect ratio": "continuous"}


def load():
    """The campaign, with the pooled measured morphology joined on.

    Aspect ratio comes from `particle_pooled.csv` rather than from the per-series
    summary, so an experiment photographed more than once contributes one number
    measured over all of its pellets. That is also what lets formulations 4 and 5
    have a morphology at all: the 2024 imaging never reached batches 1 and 2, and
    their only measurements are the 2026 remakes.
    """
    campaign = pd.read_csv(cfg.CAMPAIGN_CSV)
    pooled = pd.read_csv(cfg.PARTICLE_POOLED_CSV)[
        ["experiment_num", "pooled_aspect_ratio", "n_preparations", "n_particles"]]
    return campaign.merge(pooled.rename(columns={"n_particles": "n_particles_pooled"}),
                          on="experiment_num", how="left")


def campaign():
    """The campaign with the pooled aspect ratio joined, as the models are fitted."""
    return load().reset_index(drop=True)


def response(df, name):
    """Rows on which a response exists, and its values."""
    if name == "feasibility":
        return df, df.feasible.to_numpy(float)
    if name == "release deviation":
        rows = df[df.feasible]
        return rows, rows.release_deviation.to_numpy(float)
    if name == "aspect ratio":
        rows = df[df.pooled_aspect_ratio.notna()]
        return rows, rows.pooled_aspect_ratio.to_numpy(float)
    raise KeyError(name)


def release_sem(df):
    """Standard error of release deviation per experiment, over its capsules.

    Built the way the official profile is: superseded series dropped, every
    remaining capsule counted once.
    """
    capsules = pd.read_csv(cfg.RELEASE_REPLICATES_CSV)
    superseded = capsules.apply(
        lambda r: cfg.RELEASE_SUPERSEDE_SERIES.get(int(r.experiment_num)) == r.series, axis=1)
    capsules = capsules[~superseded]
    wide = capsules.pivot_table(index=["experiment_num", "series", "preparation", "replicate"],
                                columns="time_min", values="frac_release")
    times = cfg.DISSOLUTION_TIMES_MIN
    wide = wide.reindex(columns=times)
    wide[0.0] = wide[0.0].fillna(0.0)
    profiles = wide.to_numpy(float)
    keep = ~np.isnan(profiles).any(axis=1)
    deviation = np.trapezoid(np.abs(profiles[keep] - cfg.TARGET_PROFILE), times / 60.0, axis=1)
    per_capsule = pd.DataFrame({"experiment_num": wide.index.get_level_values(0)[keep],
                                "deviation": deviation})
    stats = per_capsule.groupby("experiment_num").deviation.agg(["std", "count"])
    stats["sem"] = stats["std"] / np.sqrt(stats["count"])
    return stats
