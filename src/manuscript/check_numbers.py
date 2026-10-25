"""Every number quoted in the manuscript text, recomputed and checked.

Each row is one quoted value: where it appears, the value as written, the value
recomputed here from data/processed/ and outputs/, and whether the two agree to
the precision the manuscript states. The retrospective-model rows need
`retrospective/fit_models.py`, `ablation.py` and `importance.py` to have run
first; without them those rows are reported as missing rather than failing.

    outputs/tables/manuscript_numbers.csv

    python src/manuscript/check_numbers.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg  # noqa: E402
import design_space as ds  # noqa: E402
from analysis.analyze_feasibility import batch_trend  # noqa: E402
from analysis.plot_optimization_trajectories import (  # noqa: E402
    incumbent_row, load_complete, pareto_mask)

ROWS = []


def check(section, quantity, quoted, value, decimals=2):
    """Record one quoted number. `quoted` is the text value; `value` is recomputed."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        ROWS.append(dict(section=section, quantity=quantity, manuscript=quoted,
                         reproduced="missing", agrees="missing"))
        return
    if isinstance(quoted, str):
        shown = str(value)
        agrees = shown == quoted
    else:
        shown = f"{value:.{decimals}f}"
        agrees = abs(round(float(value), decimals) - quoted) < 0.5 * 10 ** -decimals + 1e-9
    ROWS.append(dict(section=section, quantity=quantity, manuscript=quoted,
                     reproduced=shown, agrees="yes" if agrees else "NO"))


def woolf(a, b, c, d):
    """Unadjusted odds ratio of a 2x2 table and its Woolf (logit) 95% interval."""
    ratio = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = norm.ppf(0.975)
    return ratio, ratio * np.exp(-z * se), ratio * np.exp(z * se)


def materials(df):
    frame = ds.prepare(df)
    out = []
    for row in frame.itertuples():
        names = {getattr(row, f"cntrl_polymer_{i}_material") for i in (0, 1)} - {ds.ABSENT}
        out.append(frozenset(names))
    return pd.Series(out, index=df.index)


def campaign_numbers(df):
    s = "2.2"
    n, formed = len(df), int(df.feasible.sum())
    check(s, "experiments", 51, n, 0)
    check(s, "formed pellets", 29, formed, 0)
    check(s, "failed", 22, n - formed, 0)
    check(s, "share infeasible (%)", 43, 100 * (n - formed) / n, 0)

    sets = materials(df)
    pairs = {m for m in sets if len(m) == 2}
    singles = {m for m in sets if len(m) == 1}
    n_materials = len(set().union(*sets))
    check(s, "binary combinations sampled", 17, len(pairs), 0)
    check(s, "possible binary combinations", 45, n_materials * (n_materials - 1) // 2, 0)
    check(s, "single-polymer formulations (materials)", 5, len(singles), 0)

    table = pd.read_csv(cfg.EXCIPIENT_DESCRIPTORS_CSV)
    table["material"] = table.excipient.replace(cfg.MATERIAL_ALIASES)
    barrier = set(table[table.functional_class == "barrier"].material)
    has = sets.map(lambda m: bool(m & barrier))
    a = int((has & df.feasible).sum())
    b = int((has & ~df.feasible).sum())
    c = int((~has & df.feasible).sum())
    d = int((~has & ~df.feasible).sum())
    check(s, "with hydrophobic/insoluble matrix material: formed", 26, a, 0)
    check(s, "with hydrophobic/insoluble matrix material: tried", 37, a + b, 0)
    check(s, "without: formed", 3, c, 0)
    check(s, "without: tried", 14, c + d, 0)
    ratio, low, high = woolf(a, b, c, d)
    check(s, "odds ratio", 8.7, ratio, 1)
    check(s, "odds ratio 95% CI low", 2.0, low, 1)
    check(s, "odds ratio 95% CI high", 37.3, high, 1)


def release_numbers(df):
    s = "2.3"
    formed = df[df.feasible].sort_values("experiment_num")
    first = formed.iloc[0]
    by = df.set_index("experiment_num")
    check(s, "first feasible experiment", 4, int(first.experiment_num), 0)
    check(s, "release deviation, experiment 4", 1.63, by.loc[4, "release_deviation"])
    check(s, "release deviation, experiment 43", 0.35, by.loc[43, "release_deviation"])
    check(s, "batch of experiment 43", 11, int(by.loc[43, "batch"]), 0)
    check(s, "release deviation, experiment 49", 0.17, by.loc[49, "release_deviation"])
    check(s, "batch of experiment 49", 13, int(by.loc[49, "batch"]), 0)
    check(s, "fold reduction, 4 to 49", 9.6,
          by.loc[4, "release_deviation"] / by.loc[49, "release_deviation"], 1)
    check(s, "fold reduction, 4 to 43", 4.7,
          by.loc[4, "release_deviation"] / by.loc[43, "release_deviation"], 1)
    late = df[df.batch >= 12]
    check(s, "batches 12-13 feasible / tried", "8/8",
          f"{int(late.feasible.sum())}/{len(late)}")

    model, _, _ = batch_trend(df)
    beta = model.params["batch"]
    low, high = model.conf_int().loc["batch"]
    check(s, "feasibility odds ratio per batch", 1.27, np.exp(beta))
    check(s, "  95% CI low", 1.07, np.exp(low))
    check(s, "  95% CI high", 1.51, np.exp(high))
    check(s, "  p", 0.007, model.pvalues["batch"], 3)

    sets = materials(formed)
    pair = frozenset({"carbopol_71", "compritol_hd5"})
    has_pair = sets == pair
    best8 = formed.nsmallest(8, "release_deviation")
    check(s, "eight lowest deviations all Carbopol 71G + Compritol", "True",
          str(bool((materials(best8) == pair).all())))
    check(s, "  their total polymer, min (% w/w)", 40,
          100 * best8.total_cntrl_polymer_ww_frac.min(), 0)
    check(s, "  their total polymer, max (% w/w)", 50,
          100 * best8.total_cntrl_polymer_ww_frac.max(), 0)
    check(s, "median deviation with the pair", 0.80,
          formed[has_pair].release_deviation.median())
    check(s, "median deviation without the pair", 1.57,
          formed[~has_pair].release_deviation.median())
    rho = spearmanr(formed.total_cntrl_polymer_ww_frac, formed.release_deviation)
    check(s, "Spearman rho, total polymer vs deviation", -0.69, rho.statistic)
    check(s, "  p < 0.001", "True", str(bool(rho.pvalue < 0.001)))
    check("2.5", "formed formulations with the pair", 15, int(has_pair.sum()), 0)


def objective_numbers(df):
    s = "2.4"
    formed = df[df.feasible]
    best = formed.groupby("batch").release_deviation.min().cummin()
    for batch, quoted in ((9, 1.14), (10, 0.72), (11, 0.35), (13, 0.17)):
        check(s, f"best release deviation through batch {batch}", quoted, best.loc[batch])
    check(s, "best release deviation through batch 8 above 1.4", "True",
          str(bool(best.loc[8] > 1.4)))
    top = formed.drug_mass.max()
    check(s, "maximum drug loading (mg per 500 mg)", 200, top, 0)
    check(s, "batch first reaching it", 5,
          int(formed[formed.drug_mass == top].batch.min()), 0)

    frame = load_complete()
    cost = np.column_stack([frame.release_deviation, frame.granule_aspect_ratio])
    front = frame[pareto_mask(cost)].sort_values("release_deviation")
    check(s, "Pareto front (release deviation, aspect ratio)", "F49, F50, F48, F36",
          ", ".join(f"F{int(e)}" for e in front.experiment_num))
    chosen = incumbent_row(frame, "chimera_merit", "min")
    check(s, "Chimera preferred formulation", "F49", f"F{int(chosen.experiment_num)}")
    roundest = incumbent_row(frame, "granule_aspect_ratio", "min")
    check(s, "lowest aspect ratio", "F36", f"F{int(roundest.experiment_num)}")


def table(name):
    path = cfg.TABLES / name
    return pd.read_csv(path) if path.exists() else None


def retrospective_numbers(df):
    sys.path.insert(0, str(cfg.ROOT / "src" / "retrospective"))
    from common import CACHE, campaign, response
    from cv import metric_ci, paired_bootstrap, score

    s = "2.5"
    frame = ds.prepare(df)
    slots = {}
    for i in (0, 1):
        for name in frame[f"cntrl_polymer_{i}_material"]:
            if name != ds.ABSENT:
                slots.setdefault(name, set()).add(i)
    check(s, "materials appearing in both slots", 5,
          sum(len(v) > 1 for v in slots.values()), 0)

    ablation = table("representation_ablation.csv")
    quoted = {
        ("release deviation", "gp", "loo"): {"A": 0.39, "B": 0.60, "C": 0.38,
                                             "B-A": (0.21, 0.01, 0.56),
                                             "C-B": (-0.22, -0.48, -0.02)},
        ("release deviation", "linear", "loo"): {"A": 0.47, "B": 0.61, "C": 0.61},
        ("aspect ratio", "gp", "loo"): {"A": 0.42, "B": 0.52, "C": 0.64,
                                        "B-A": (0.10, -0.06, 0.30),
                                        "C-B": (0.13, -0.00, 0.47)},
        ("aspect ratio", "linear", "loo"): {"A": 0.02, "B": 0.42, "C": 0.33,
                                            "B-A": (0.40, 0.13, 0.86)},
        ("feasibility", "gp", "loo"): {"A": 0.72, "B": 0.71, "C": 0.68},
        ("release deviation", "gp", "forward"): {"A": 0.50, "B": 0.48, "C": 0.28},
    }
    for (resp, model, scheme), values in quoted.items():
        for step, q in values.items():
            label = f"{resp}, {model}, {scheme}, {step}"
            if ablation is None:
                check(s, label, q if not isinstance(q, tuple) else q[-1], None)
                continue
            row = ablation[(ablation.response == resp) & (ablation.model == model)
                           & (ablation.scheme == scheme) & (ablation.step == step)].iloc[0]
            if isinstance(q, tuple):
                if q[0] is not None:
                    check(s, label, q[0], row.value)
                if len(q) == 3:
                    check(s, f"{label}, 95% low", q[1], row.ci_low)
                    check(s, f"{label}, 95% high", q[2], row.ci_high)
            else:
                check(s, label, q, row.value)

    data = campaign()
    fits = {}
    for resp, name, scheme in (("release deviation", "E1", "loo"), ("aspect ratio", "E5", "loo"),
                               ("feasibility", "E3", "loo"), ("feasibility", "E0", "lopo"),
                               ("feasibility", "E3", "lopo")):
        path = CACHE / f"{resp.replace(' ', '_')}__{scheme}__gp__{name}__plain.npz"
        fits[resp, name, scheme] = dict(np.load(path)) if path.exists() else None

    for resp, name, quoted_r2, quoted_rho in (("release deviation", "E1", 0.60, 0.80),
                                              ("aspect ratio", "E5", 0.64, 0.78)):
        fit = fits[resp, name, "loo"]
        _, y = response(data, resp)
        result = score(y, fit["mean"], False) if fit else {}
        check("Fig 7", f"{resp} R2", quoted_r2, result.get("r2"))
        check("Fig 7", f"{resp} Spearman rho", quoted_rho, result.get("rho"))
    _, y = response(data, "feasibility")
    fit = fits["feasibility", "E3", "loo"]
    check("Fig 7", "feasibility AUC (E3)", 0.80, score(y, fit["mean"], True)["auc"] if fit else None)

    old, new = fits["feasibility", "E0", "lopo"], fits["feasibility", "E3", "lopo"]
    if old is not None and new is not None:
        tested = old["tested"].astype(int)
        yt = y[tested]
        for label, fit, q in (("role-based", old, (0.36, 0.20, 0.51)),
                              ("physicochemical", new, (0.54, 0.37, 0.70))):
            low, high = metric_ci(yt, fit["mean"][tested], True)
            check(s, f"LOPO feasibility AUC, {label}", q[0], score(yt, fit["mean"][tested], True)["auc"])
            check(s, f"  95% CI low", q[1], low)
            check(s, f"  95% CI high", q[2], high)
        diff = paired_bootstrap(yt, new["mean"][tested], old["mean"][tested], True)
        check(s, "LOPO feasibility AUC difference", 0.18, diff["diff"])
        check(s, "  95% CI low", 0.02, diff["low"])
        check(s, "  95% CI high", 0.35, diff["high"])

    importance = table("input_ablation.csv")
    quoted_drop = (
        ("release_E1", "ww=pvp_30k", 0.10), ("release_E1", "fluid", 0.10),
        ("release_E1", "ww=carbopol_71", 0.09), ("release_E1", "gran_time", -0.10),
        ("aspect_E5", "gran_time", 0.34), ("aspect_E5", "ww=eudragit_sr", 0.33),
        ("aspect_E5", "ww=carbopol_940nf", 0.23), ("aspect_E5", "ww=hpmc", 0.17),
        ("feasibility_E3", "drug_ww_frac", 0.21), ("feasibility_E3", "fluid", 0.08),
        ("feasibility_E3", "hlb_load", 0.06), ("feasibility_E3", "anionic_ww", 0.06),
    )
    for tag, feature, q in quoted_drop:
        value = None
        if importance is not None:
            hit = importance[(importance.surrogate == tag) & (importance.feature == feature)]
            value = float(hit.importance.iloc[0]) if len(hit) else None
        check("Fig 7", f"{tag}: drop when {feature} removed", q, value)
    if importance is not None:
        hit = importance[(importance.surrogate == "release_E1") & (importance.feature == "gran_time")]
        check("Fig 7", "release_E1 gran_time interval below zero", "True",
              str(bool(hit.high.iloc[0] < 0)))
        top = importance[importance.surrogate == "release_E1"]
        check("Fig 7", "release_E1 inputs with an interval above zero", 0,
              int((top.low > 0).sum()), 0)

    explored = cfg.OUTPUTS / "retrospective" / "exploratory_scores.csv"
    e2_path = (cfg.OUTPUTS / "retrospective" / "exploratory_cache"
               / "release_deviation__loo__gp__E2__plain.npz")
    if explored.exists():
        check(s, "configurations in the exploratory search", 180, len(pd.read_csv(explored)), 0)
    if e2_path.exists():
        e2 = dict(np.load(e2_path))
        e1 = fits["release deviation", "E1", "loo"]
        _, y = response(data, "release deviation")
        check(s, "best alternative for release, GP E2, loo R2", 0.66, score(y, e2["mean"], False)["r2"])
        diff = paired_bootstrap(y, e2["mean"], e1["mean"], False)
        check(s, "  GP E2 - GP E1", 0.07, diff["diff"])
        check(s, "  95% low", -0.01, diff["low"])
        check(s, "  95% high", 0.16, diff["high"])


def methods_numbers(df):
    s = "4.4"
    manifest = pd.read_csv(cfg.IMAGE_MANIFEST_CSV)
    used = manifest[manifest.is_representative
                    & manifest.experiment_num.isin(df[df.feasible].experiment_num)]
    check(s, "unique photographs", 124, len(used), 0)
    check(s, "formulations photographed", 29, used.experiment_num.nunique(), 0)
    check(s, "photographs per formulation, median", 3,
          used.groupby("experiment_num").size().median(), 0)
    check(s, "scale, min (um/px)", 14, used.um_per_px.min(), 0)
    check(s, "scale, max (um/px)", 33, used.um_per_px.max(), 0)
    validation = table("particle_validation.csv")
    if validation is not None:
        v = validation.set_index("check")
        check(s, "Spearman rho, aspect ratio vs visual score, per experiment", -0.89,
              v.loc["aspect ratio vs visual score, per experiment", "statistic"])
        check(s, "  n", 19, v.loc["aspect ratio vs visual score, per experiment", "n"], 0)
        check(s, "median absolute size error vs ImageJ (%)", 5.7,
              v.loc["median size error vs manual tables, percent absolute", "statistic"], 1)
        check(s, "  n", 12, v.loc["median size error vs manual tables, percent absolute", "n"], 0)
        check("Fig S2", "per-granule Spearman rho", -0.45,
              v.loc["aspect ratio vs visual score, per particle", "statistic"])
    by_score = table("morphology_by_score.csv")
    if by_score is not None:
        check("Fig S2", "granules per visual score 2/3/4/5", "71/527/101/703",
              "/".join(str(int(n)) for n in by_score.sort_values("visual_score").n_particles))


def main():
    df = pd.read_csv(cfg.CAMPAIGN_CSV)
    campaign_numbers(df)
    release_numbers(df)
    objective_numbers(df)
    retrospective_numbers(df)
    methods_numbers(df)
    out = pd.DataFrame(ROWS)
    cfg.TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.TABLES / "manuscript_numbers.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 70,
                           "display.max_rows", 500):
        print(out.to_string(index=False))
    counts = out.agrees.value_counts()
    print(f"\n{counts.get('yes', 0)} agree, {counts.get('NO', 0)} disagree, "
          f"{counts.get('missing', 0)} missing")


if __name__ == "__main__":
    main()
