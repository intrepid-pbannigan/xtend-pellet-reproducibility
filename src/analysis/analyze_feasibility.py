"""Analysis B: did the campaign learn to avoid pelletization failure?

Feasibility is the one response observed for every experiment, so it is the
only place the campaign's learning can be assessed without conditioning on
success. Results are reported with uncertainty because the trend is weak.

    python src/analysis/analyze_feasibility.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg
from plotting import style


def batch_trend(df):
    """Logistic regression of feasibility on batch index."""
    X = sm.add_constant(df.batch.astype(float))
    model = sm.GLM(df.feasible.astype(int), X, family=sm.families.Binomial()).fit()

    grid = np.linspace(1, cfg.N_BATCHES, 100)
    pred = model.get_prediction(sm.add_constant(grid))
    frame = pred.summary_frame(alpha=0.05)
    return model, grid, frame


def univariate_odds(df):
    """Odds ratio of feasibility per standard deviation of each continuous variable."""
    rows = []
    for var in cfg.CONTINUOUS_VARS:
        x = df[var]
        z = (x - x.mean()) / x.std()
        model = sm.GLM(df.feasible.astype(int), sm.add_constant(z), family=sm.families.Binomial()).fit()
        lo, hi = model.conf_int().iloc[1]
        rows.append(
            dict(
                variable=var,
                odds_ratio=np.exp(model.params.iloc[1]),
                ci_low=np.exp(lo),
                ci_high=np.exp(hi),
                p_value=model.pvalues.iloc[1],
            )
        )
    return pd.DataFrame(rows).sort_values("odds_ratio")


def main():
    style.set_style()
    df = pd.read_csv(cfg.CAMPAIGN_CSV)

    model, grid, band = batch_trend(df)
    odds = univariate_odds(df)

    per_batch = df.groupby("batch").feasible.agg(["sum", "size"])
    per_batch["rate"] = per_batch["sum"] / per_batch["size"]

    # Split on the experiment number rather than on position, and count each half
    # rather than assume it: merging the duplicated design point leaves 43 rows
    # spanning a numbering that still runs to 44, so the halves are 22 and 21.
    # Split on the 2024 campaign's experiment number rather than on position.
    # Batch 12 is drawn on the by-batch curve but is not folded into this
    # half-and-half, which is a claim about the original 44.
    early = df[df.experiment_num <= cfg.N_CAMPAIGN_EXPERIMENTS // 2]
    late = df[(df.experiment_num > cfg.N_CAMPAIGN_EXPERIMENTS // 2)
              & (df.experiment_num <= cfg.N_CAMPAIGN_EXPERIMENTS)]
    first, second = int(early.feasible.sum()), int(late.feasible.sum())
    n_first, n_second = len(early), len(late)
    fisher_p = fisher_exact([[first, n_first - first], [second, n_second - second]])[1]
    or_batch = np.exp(model.params.iloc[1])
    or_ci = np.exp(model.conf_int().iloc[1])

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.7), gridspec_kw={"width_ratios": [1.15, 1, 1.25]})

    ax = axes[0]
    ax.fill_between(grid, band.mean_ci_lower, band.mean_ci_upper, color=style.FEASIBLE, alpha=0.15, lw=0)
    ax.plot(grid, band["mean"], color=style.FEASIBLE, label="logistic fit, 95% CI")
    ax.scatter(per_batch.index, per_batch.rate, s=44, color="k", zorder=3, label="observed rate")
    ax.axhline(df.feasible.mean(), color="k", ls=":", lw=0.9, label=f"overall {df.feasible.mean():.0%}")
    ax.set_xlabel("batch")
    ax.set_ylabel("P(pellets form)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(range(1, cfg.N_BATCHES + 1))
    ax.set_title(f"Feasibility by batch\nOR {or_batch:.2f} per batch (95% CI {or_ci[0]:.2f}-{or_ci[1]:.2f}), p = {model.pvalues.iloc[1]:.2f}")
    ax.legend(loc="upper left")

    # cumulative failures: the budget actually consumed by infeasible experiments
    ordered = df.sort_values("experiment_num")
    cum_failed = (~ordered.feasible).cumsum()
    ax = axes[1]
    ax.plot(ordered.experiment_num, cum_failed, color=style.FEASIBLE, label="observed")
    ax.plot(
        ordered.experiment_num,
        ordered.experiment_num * (1 - df.feasible.mean()),
        color="k",
        ls=":",
        lw=0.9,
        label="constant 50% failure rate",
    )
    ax.set_xlabel("experiment number")
    ax.set_ylabel("cumulative failed experiments")
    ax.legend(loc="upper left")
    ax.set_title(
        f"Experimental budget lost to failure\nfirst half {n_first - first}/{n_first} failed, "
        f"second half {n_second - second}/{n_second} (Fisher p = {fisher_p:.2f})"
    )

    ax = axes[2]
    y = np.arange(len(odds))
    ax.hlines(y, odds.ci_low, odds.ci_high, color="k", lw=1.1)
    colours = ["#B0B0B0" if p >= 0.05 else style.FEASIBLE for p in odds.p_value]
    ax.scatter(odds.odds_ratio, y, s=42, color=colours, zorder=3)
    ax.axvline(1, color="k", ls="--", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(odds.variable, fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("odds ratio for feasibility, per SD (log scale)")
    ax.set_title("Univariate associations\nexploratory, 9 uncorrected tests")

    fluid = pd.crosstab(df.fluid_type, df.feasible)
    fluid.columns = ["failed", "feasible"]
    fluid["rate"] = fluid.feasible / fluid.sum(axis=1)

    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.text(
        0.5,
        0.015,
        "Granulating fluid: "
        + ";  ".join(f"{k} {int(v.feasible)}/{int(v.feasible + v.failed)} feasible" for k, v in fluid.iterrows())
        + ".  Adaptively sampled, so associations are not independent effect estimates.",
        ha="center",
        fontsize=7.5,
        color="#444444",
    )
    style.save(fig, "feasibility_by_batch")

    cfg.TABLES.mkdir(parents=True, exist_ok=True)
    odds.to_csv(cfg.TABLES / "feasibility_univariate.csv", index=False)

    print(f"overall feasibility {df.feasible.sum()}/{len(df)} = {df.feasible.mean():.0%}")
    print(f"logistic slope per batch: OR {or_batch:.3f} (95% CI {or_ci[0]:.3f}-{or_ci[1]:.3f}), p = {model.pvalues.iloc[1]:.3f}")
    print(f"first half {first}/{n_first} vs second half {second}/{n_second}, "
          f"Fisher p = {fisher_p:.3f}")
    print()
    print("feasibility by granulating fluid:")
    print(fluid.to_string())
    print()
    print("univariate odds ratios per SD:")
    print(odds.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
