"""Analysis F, part 2: does the automated image analysis reproduce what was scored by eye?

Two validation questions, one figure each.

    1. Do the automated shape descriptors agree with the manual 1-5 visual score?
       If they do, morphology can enter the multi-objective analysis as a
       continuous distributional response instead of a four-level ordinal one.
    2. Do the automated size distributions agree with the manual ImageJ tables and
       the corrected workbook means, and what does a distribution buy over a mean?

The test that decides whether any of it is usable is neither of those. A method
that measures spheres well and dumbbells badly would manufacture the very
morphology trend the campaign is trying to read, so the error is compared between
morphology classes by permutation, against the fixed-distance watershed that this
separation rule replaced.

    python src/analysis/analyze_particle_morphology.py
"""

import sys
import textwrap
from collections import namedtuple
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg
from analysis import preparations
from plotting import style

SHAPE_LABELS = {
    1: "cylinder\n(1)",
    2: "rounded\ncylinder (2)",
    3: "dumbbell\n(3)",
    4: "ellipsoid\n(4)",
    5: "sphere\n(5)",
}

COLOUR_INFERRED = "#E69F00"

Bias = namedtuple("Bias", "gap p_value ci_low ci_high detectable")

def load_remakes():
    """The August remakes as measured, from the photographs the pipeline trusts.

    Restricted to those, rather than pooled over all six, because the descriptors
    are not equally robust to the new protocol. Aspect ratio barely moves between
    the two: it is a ratio of fitted axes, and a pellet showing three quarters of
    itself still fits the same ellipse. Circularity and solidity are computed from
    a perimeter, and a pellet cut out from behind another one has a boundary that
    was never its own, so both fall with the fraction of material the separation
    rule could account for -- circularity 0.79 down to 0.55 across these six
    frames, in step with accounted area. Across the 56 campaign photographs, none
    of them heaped,
    the same two quantities are unrelated (Spearman -0.09, p = 0.49), so this is a
    property of how the remakes were laid out and not a correction the campaign
    needs.
    """
    return preparations.load(cfg.REMAKE_SERIES)


def load():
    """The campaign series for every statistic, the remakes for drawing only.

    The August 2026 remakes carry no visual score, so they cannot enter a test of
    whether the automated descriptors agree with one, and pooling their particles
    would inflate every count here. They are still worth showing: their measured
    descriptors are the only thing that says where unscored material falls on the
    same axes, so they are returned separately and drawn outside every fit.
    """
    particles = pd.read_csv(cfg.PARTICLES_CSV).query("series == @cfg.CAMPAIGN_SERIES")
    summary = pd.read_csv(cfg.PARTICLE_SUMMARY_CSV).query("series == @cfg.CAMPAIGN_SERIES")
    remakes = load_remakes()
    auto = summary[summary.source == "segmentation"]
    manual = summary[(summary.source == "manual_imagej") & ~summary.provisional]
    # Shape descriptors are scale-free, so every segmented experiment carries
    # them; absolute size needs a calibration that three experiments never had.
    return particles, summary, auto, auto[auto.d50_um.notna()], manual, remakes


def permutation_gap(values, labels, group_a="spherical", group_b="dumbbell"):
    """Difference in median error between morphology classes: test and interval.

    With around a dozen experiments the normal-theory alternatives are not
    credible, so the p-value comes from permuting the labels and the interval from
    resampling within each class. Both are needed. A large p-value on six against
    seven experiments does not establish that the two classes are measured alike,
    it only fails to establish that they are not; the interval says how large a
    bias would have survived undetected.
    """
    values, labels = np.asarray(values, float), np.asarray(labels)
    a, b = values[labels == group_a], values[labels == group_b]
    if len(a) < 2 or len(b) < 2:
        return Bias(*([np.nan] * 5))
    observed = np.median(a) - np.median(b)

    # Thirteen experiments admit only 1716 distinct labellings, so the null is
    # enumerated rather than sampled: no Monte Carlo error, and no seed to report.
    pooled = np.concatenate([a, b])
    null = np.array([
        np.median(pooled[list(pick)]) - np.median(pooled[list(set(range(len(pooled))) - set(pick))])
        for pick in combinations(range(len(pooled)), len(a))
    ])
    p_value = float((np.abs(null) >= abs(observed) - 1e-12).mean())
    # The smallest gap this design could have called significant. Quoting it stops
    # a large p-value from being read as evidence that the two classes agree.
    detectable = float(np.percentile(np.abs(null), 95))

    rng = np.random.default_rng(cfg.RANDOM_SEED)
    draws = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b)))
             for _ in range(cfg.N_PERMUTATIONS)]
    low, high = np.percentile(draws, [2.5, 97.5])
    return Bias(float(observed), p_value, float(low), float(high), detectable)


def holm(p_values):
    """Holm-adjusted p-values, skipping the rows that report an effect size only.

    The suite asks a dozen questions of one small dataset, so an unadjusted 0.03
    somewhere in it is not the same evidence as an 0.03 asked alone. Holm rather
    than a false-discovery rule because these are the checks the method has to
    pass, not a screen for interesting leads.
    """
    p_values = pd.Series(p_values)
    tested = p_values.dropna()
    adjusted = pd.Series(multipletests(tested.values, method="holm")[1], index=tested.index)
    return p_values.where(p_values.isna(), adjusted)


def score_breakdown(per_particle, auto):
    """The numbers behind the violins: one row per visual score.

    Both counts are reported because they answer different questions. The particle
    count says how much material sits behind each violin; the experiment count says
    how much independent evidence there is, and for two of the five scores it is one.
    """
    rows = []
    for score, group in per_particle.groupby("particle_shape_score"):
        experiments = auto[auto.particle_shape_score == score]
        low, high = np.percentile(group.aspect_ratio, [25, 75])
        rows.append(dict(
            visual_score=int(score),
            description=SHAPE_LABELS[int(score)].replace("\n", " "),
            quality_class=cfg.shape_score_to_quality(score, feasible=True),
            n_experiments=len(experiments),
            n_particles=len(group),
            aspect_ratio_median=group.aspect_ratio.median(),
            aspect_ratio_q25=low,
            aspect_ratio_q75=high,
            circularity_median=group.circularity.median(),
            solidity_median=group.solidity.median(),
            experiments=" ".join(str(int(e)) for e in sorted(experiments.experiment_num)),
        ))
    return pd.DataFrame(rows)


def footnote(fig, text, y, colour="#444444", width=150):
    """A figure-level note, wrapped to the figure.

    Wrapped rather than left to the renderer because saving uses a tight bounding
    box: a note wider than the axes stretches the saved canvas instead of being
    clipped, which shrinks every panel in the figure and is easy to miss on
    screen. Returns the number of lines so the caller can stack notes.
    """
    lines = textwrap.wrap(text, width)
    fig.text(0.5, y, "\n".join(lines), ha="center", va="bottom", fontsize=7.5, color=colour)
    return len(lines)


def morphology_figure(particles, auto):
    """Automated shape descriptors against the visual score.

    Scored campaign experiments only. The 2026 remakes have no visual score, so
    they cannot sit in a figure that tests the descriptor against it; their
    morphology is in the size figure and the reproducibility figure.
    """
    style.set_style()
    # Stacked, at page width, so each panel reads at full width in the text.
    fig, axes = plt.subplots(3, 1, figsize=(style.DOUBLE_COLUMN, style.MAX_HEIGHT))

    segmented = particles[particles.source == "segmentation"].merge(
        auto[["experiment_num", "particle_shape_score", "pellet_quality"]],
        on="experiment_num", how="inner",
    )

    ax = axes[0]
    scores = sorted(segmented.particle_shape_score.dropna().unique())
    for position, score in enumerate(scores):
        values = segmented[segmented.particle_shape_score == score].aspect_ratio
        quality = cfg.shape_score_to_quality(score, feasible=True)
        parts = ax.violinplot([values], positions=[position], widths=0.75,
                              showextrema=False, showmedians=True)
        for body in parts["bodies"]:
            body.set_facecolor(style.QUALITY_COLOURS[quality])
            body.set_alpha(0.55)
        parts["cmedians"].set_color("k")
        ax.text(position, 1.02, f"n={len(values)}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(1.0, 2.8)
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels([SHAPE_LABELS[int(s)] for s in scores], fontsize=7.5)
    ax.set_xlabel("manual visual score")
    ax.set_ylabel("automated aspect ratio (per particle)")
    rho = spearmanr(segmented.particle_shape_score, segmented.aspect_ratio).statistic
    ax.set_title(f"Elongation tracks the visual score\nper-particle Spearman {rho:+.2f}")

    ax = axes[1]
    for quality, group in auto.groupby("pellet_quality"):
        ax.scatter(group.median_aspect_ratio, group.median_circularity, s=55,
                   color=style.QUALITY_COLOURS[int(quality)], edgecolor="k", linewidth=0.4,
                   label=style.QUALITY_LABELS[int(quality)], zorder=3)
    for row in auto.itertuples():
        ax.annotate(int(row.experiment_num), (row.median_aspect_ratio, row.median_circularity),
                    textcoords="offset points", xytext=(6, -1), fontsize=6.5, color="#555555")
    ax.set_xlabel("median aspect ratio")
    ax.set_ylabel("median circularity")
    rho_e = spearmanr(auto.particle_shape_score, auto.median_aspect_ratio).statistic
    ax.set_title(f"Two descriptors separate the classes\nexperiment-level Spearman {rho_e:+.2f}")
    ax.legend(loc="lower left", title="visual quality class", title_fontsize=7.5, fontsize=7.5)

    ax = axes[2]
    for row in auto.itertuples():
        subset = segmented[segmented.experiment_num == row.experiment_num]
        if subset.empty:
            continue
        values = np.sort(subset.aspect_ratio)
        ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post",
                color=style.QUALITY_COLOURS[int(row.pellet_quality)], alpha=0.85, lw=1.2)
    ax.set_xlim(1, 2.6)
    ax.set_xlabel("aspect ratio (per particle)")
    ax.set_ylabel("cumulative fraction of particles")
    ax.set_title("Whole distributions, not just medians\none curve per experiment")

    fig.tight_layout(h_pad=1.5)
    style.save(fig, "particle_morphology_validation")
    return rho, rho_e


def size_figure(sized, manual, scored, bias, remakes, original_sizes):
    """Automated size distributions against the manual tables and the workbook."""
    style.set_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.4))

    ax = axes[0]
    # The representative image is usually the JPEG original, whose scale comes
    # from its own TIFF re-save; only three experiments borrow from another day.
    own = sized[sized.calibration_source.isin(["tiff_tag", "inferred_same_experiment"])]
    borrowed = sized[sized.calibration_source == "inferred_same_session"]
    ax.scatter(own.workbook_size_um, own.d50_um, s=55, color=style.FEASIBLE,
               edgecolor="k", linewidth=0.4, label="scale from own photographs", zorder=3)
    ax.scatter(borrowed.workbook_size_um, borrowed.d50_um, s=55, color=COLOUR_INFERRED,
               edgecolor="k", linewidth=0.4, label="scale borrowed from same session", zorder=3)
    ax.scatter(manual.workbook_size_um, manual.d50_um, s=34, marker="s", color=style.FAILED,
               edgecolor="k", linewidth=0.3, label="pooled ImageJ tables", zorder=2)
    # The remakes against the recorded value for the material they replaced. Not a
    # measurement of the same pellets, which is why they are the only points here
    # that are allowed to sit far off the diagonal without being an error.
    remade = remakes.summary.assign(
        original=remakes.summary.experiment_num.map(original_sizes))
    ax.scatter(remade.original, remade.d50_um, s=80, marker=style.REMAKE_MARKER,
               facecolor="none", edgecolor=style.REMAKE, linewidth=1.5, zorder=4,
               label="remade 2026 vs the 2024 estimate")
    for row in remade.itertuples():
        ax.annotate(row.sample, (row.original, row.d50_um), textcoords="offset points",
                    xytext=(7, -3), fontsize=7, color=style.REMAKE)
    # Wide enough that the remakes, which sit at the two smallest recorded sizes in
    # the campaign, are not drawn on the spine.
    limits = np.array([900, 2200])
    ax.plot(limits, limits, color="k", ls="--", lw=0.9, zorder=1)
    ax.set_xlim(*limits)
    ax.set_xlabel("workbook size, corrected (um)")
    ax.set_ylabel("measured median size, D50 (um)")
    ax.set_title("Three measurements of the same pellets\n"
                 f"automated vs manual: {scored.size_error_pct.abs().median():.1f}% median")
    ax.legend(loc="lower right", fontsize=7)

    ax = axes[1]
    ordered = sized.sort_values("d50_um")
    positions = np.arange(len(ordered), dtype=float)
    ax.hlines(positions, ordered.d10_um, ordered.d90_um, color=style.FEASIBLE, lw=3, alpha=0.45)
    ax.scatter(ordered.d50_um, positions, s=28, color=style.FEASIBLE, zorder=3, label="D50")
    ax.scatter(ordered.workbook_size_um, positions, s=34, marker="|", color=style.TARGET,
               zorder=4, label="workbook mean, corrected")
    # Above the campaign block rather than sorted into it: these are a different
    # preparation, so their place in an ordering of campaign sizes would imply a
    # comparison that is not being made.
    remake_positions = len(ordered) + 0.6 + np.arange(len(remakes.summary))
    ax.hlines(remake_positions, remakes.summary.d10_um, remakes.summary.d90_um,
              color=style.REMAKE, lw=3, alpha=0.45)
    ax.scatter(remakes.summary.d50_um, remake_positions, s=28, color=style.REMAKE, zorder=3)
    if len(remake_positions):
        ax.axhline(len(ordered) - 0.2, color="k", lw=0.8, ls="--", alpha=0.45)
    ax.set_yticks(np.concatenate([positions, remake_positions]))
    ax.set_yticklabels([int(e) for e in ordered.experiment_num]
                       + list(remakes.summary["sample"]), fontsize=6.5)
    for tick in ax.get_yticklabels()[len(ordered):]:
        tick.set_color(style.REMAKE)
    ax.set_ylim(-1.0, remake_positions.max() + 1.0 if len(remake_positions) else len(ordered))
    ax.set_ylabel("experiment")
    ax.set_xlabel("particle size (um)")
    ax.set_title("What a mean throws away\nbars span D10 to D90")
    ax.legend(loc="lower right")

    ax = axes[2]
    ax.axhline(0, color="k", lw=0.8, ls="--")
    for column, colour, label in [("baseline_error_pct", style.FAILED, "fixed-distance watershed"),
                                  ("size_error_pct", style.FEASIBLE, "concavity-depth separation")]:
        for position, morphology in enumerate(["spherical", "dumbbell"]):
            values = scored[scored.morphology_class == morphology][column].dropna()
            offset = -0.16 if column == "baseline_error_pct" else 0.16
            ax.scatter(np.full(len(values), position + offset)
                       + np.random.default_rng(cfg.RANDOM_SEED).uniform(-0.05, 0.05, len(values)),
                       values, s=42, color=colour, edgecolor="k", linewidth=0.4, zorder=3,
                       label=label if position == 0 else None)
            ax.hlines(values.median(), position + offset - 0.11, position + offset + 0.11,
                      color="k", lw=1.6, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["spherical\n(score 4-5)", "dumbbell\n(score 1-3)"], fontsize=8)
    ax.set_ylabel("size error against manual tables (%)")
    chosen, baseline = bias["concavity"], bias["watershed"]
    ax.set_title("The test that decides usability\n"
                 f"gap {chosen.gap:+.1f} pp [{chosen.ci_low:+.1f}, {chosen.ci_high:+.1f}] vs "
                 f"{baseline.gap:+.1f} pp [{baseline.ci_low:+.1f}, {baseline.ci_high:+.1f}]")
    ax.legend(loc="lower left")

    fig.tight_layout(rect=(0, 0.19, 1, 1))
    gaps = ", ".join(f"{row.sample} {row.d50_um:.0f} against {row.original:.0f} um"
                     for row in remade.itertuples())
    lines = footnote(
        fig,
        f"The 2026 remakes are a different preparation, so the left panel compares them with the value recorded for "
        f"the material they replaced rather than with a measurement of the same pellets: {gaps}. Those two recorded "
        "values are the only ones in the campaign never backed by an ImageJ table, so the gap is as likely to be in "
        "them as in the remade pellets. The third panel needs a visual score, which the remakes do not have.",
        0.012, colour=style.REMAKE)
    footnote(fig,
             f"Brackets are bootstrap 95% intervals on the difference in median error between morphology classes; on "
             f"{len(scored)} experiments no gap below {bias['concavity'].detectable:.1f} points could have been called "
             "significant. Experiments 9 and 11 are excluded from the scoring: their sizes rest on the operator's own "
             "calibration line, so they cannot also be used to check it.",
             0.03 + 0.028 * lines)
    style.save(fig, "particle_size_distributions")


def main():
    particles, summary, auto, sized, manual, remakes = load()
    scored = summary[summary.in_calibration_set]

    bias = {
        "concavity": permutation_gap(scored.size_error_pct, scored.morphology_class),
        "watershed": permutation_gap(scored.baseline_error_pct, scored.morphology_class),
    }

    per_particle = particles[particles.source == "segmentation"].merge(
        auto[["experiment_num", "particle_shape_score"]], on="experiment_num"
    )
    by_score = score_breakdown(per_particle, auto)
    # The recorded size of the material the remakes replaced, read from the
    # workbook rather than from the summary, which deliberately withholds a
    # campaign response from a preparation that did not produce it.
    original_sizes = pd.read_csv(cfg.CAMPAIGN_CSV).set_index("experiment_num").size_um

    rho_particle, rho_experiment = morphology_figure(particles, auto)
    size_figure(sized, manual, scored, bias, remakes, original_sizes)

    particle_test = spearmanr(per_particle.particle_shape_score, per_particle.aspect_ratio)
    experiment_test = spearmanr(auto.particle_shape_score, auto.median_aspect_ratio)
    circularity_test = spearmanr(auto.particle_shape_score, auto.median_circularity)
    span_test = spearmanr(sized.particle_shape_score, sized.span)

    chosen, baseline_bias = bias["concavity"], bias["watershed"]

    # Width is the statistic the two instruments disagree on, so it is reported as
    # a comparison rather than as a measurement, and only on the experiments whose
    # manual table covers their photographs.
    width = scored.set_index("experiment_num")[["span", "iqr_width"]].join(
        manual.set_index("experiment_num")[["span", "iqr_width"]], rsuffix="_manual", how="inner"
    ).dropna()
    # A common-mode offset is not scatter, and reporting only the absolute median
    # would present it as though it were.
    offset = wilcoxon(scored.size_error_pct)
    clump = spearmanr(sized.clump_frac, sized.span)

    table = pd.DataFrame([
        # Descriptive only: those 1396 particles come from 20 experiments, so a
        # p-value computed as though they were independent is meaningless.
        dict(check="aspect ratio vs visual score, per particle", statistic=rho_particle,
             p_value=np.nan, n=len(per_particle)),
        dict(check="aspect ratio vs visual score, per experiment", statistic=rho_experiment,
             p_value=experiment_test.pvalue, n=len(auto)),
        dict(check="circularity vs visual score, per experiment", statistic=circularity_test.statistic,
             p_value=circularity_test.pvalue, n=len(auto)),
        dict(check="median size error vs manual tables, percent signed",
             statistic=float(scored.size_error_pct.median()), p_value=offset.pvalue, n=len(scored)),
        dict(check="median size error vs manual tables, percent absolute",
             statistic=float(scored.size_error_pct.abs().median()), p_value=np.nan, n=len(scored)),
        dict(check="top-decile error vs manual tables, percent signed",
             statistic=float(scored.d90_error_pct.median()), p_value=np.nan, n=len(scored)),
        dict(check="size error gap, spherical minus dumbbell, pp", statistic=chosen.gap,
             p_value=chosen.p_value, ci_low=chosen.ci_low, ci_high=chosen.ci_high,
             smallest_detectable=chosen.detectable, n=len(scored)),
        dict(check="same gap under fixed-distance watershed, pp", statistic=baseline_bias.gap,
             p_value=baseline_bias.p_value, ci_low=baseline_bias.ci_low,
             ci_high=baseline_bias.ci_high, smallest_detectable=baseline_bias.detectable,
             n=len(scored)),
        dict(check="size distribution span vs visual score", statistic=span_test.statistic,
             p_value=span_test.pvalue, n=len(sized)),
        dict(check="span vs fraction of oversized objects per image", statistic=clump.statistic,
             p_value=clump.pvalue, n=len(sized)),
        dict(check="span, automated over manual, median of paired ratios",
             statistic=float((width.span / width.span_manual).median()),
             p_value=np.nan, n=len(width)),
        dict(check="interquartile width, automated over manual, median of paired ratios",
             statistic=float((width.iqr_width / width.iqr_width_manual).median()),
             p_value=np.nan, n=len(width)),
    ])
    table["p_holm"] = holm(table.p_value)
    cfg.TABLES.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(cfg.TABLES / "particle_validation.csv", index=False)
    by_score.round(3).to_csv(cfg.TABLES / "morphology_by_score.csv", index=False)

    print(table.round(3).to_string(index=False))
    print()
    print(f"coverage: {len(auto)} experiments with shape descriptors, {len(sized)} with absolute "
          f"size, {len(scored)} scored against manual tables")
    print(f"span across experiments: {sized.span.min():.2f} to {sized.span.max():.2f} "
          f"(median {sized.span.median():.2f})")
    print(f"width is method-dependent: automated span is {(width.span / width.span_manual).median():.2f}x "
          f"the operator's on the same {len(width)} experiments, interquartile width "
          f"{(width.iqr_width / width.iqr_width_manual).median():.2f}x, while D50 agrees to "
          f"{scored.size_error_pct.abs().median():.1f}% and the top decile to "
          f"{scored.d90_error_pct.median():+.1f}%")
    print(f"between-image sampling CV: median {sized.image_median_cv_pct.median():.1f}%, "
          f"worst {sized.image_median_cv_pct.max():.1f}%")
    print()
    print(by_score.drop(columns=["quality_class", "solidity_median"]).round(2).to_string(index=False))
    for name in ("particle_validation.csv", "morphology_by_score.csv"):
        print(f"  table -> {(cfg.TABLES / name).relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
