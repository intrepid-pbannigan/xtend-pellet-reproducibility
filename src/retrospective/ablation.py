"""Representation ablation: manuscript Figure 6 and Table 2.

The comparison of the campaign encoding with the final retrospective encoding
changes two things at once. This splits them:

    A  E0   slot one-hot and slot w/w, the campaign's encoding
    B  E1   role-independent material mass fractions, same other inputs
    C  E5   B plus the four wet-mass descriptors

Drug w/w, Mg stearate, the five process settings and the fluid one-hot are
shared, so A to B is the polymer encoding and B to C is the derived
descriptors. Predictions are the cached fits from `fit_models.py`. The paired
bootstrap is the one in cv.py (5000 resamples, seed 0); the interval on each
score is cv.metric_ci (1000 resamples, seed 0).

    outputs/tables/representation_ablation.csv   every score and contrast
    outputs/tables/table2_release_validation.csv  Table 2
    outputs/manuscript/fig06_representation_ablation.{pdf,png}

    python src/retrospective/ablation.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import CACHE, KIND, RESPONSES, campaign, cfg, response  # noqa: E402
from cv import PRIMARY, SCHEMES, metric_ci, paired_bootstrap, score  # noqa: E402
from features import build, material_fractions  # noqa: E402
from fit_models import key  # noqa: E402

import design_space as ds  # noqa: E402
from plotting import style  # noqa: E402

STEPS = (("A", "E0", "slot one-hot"),
         ("B", "E1", "material w/w"),
         ("C", "E5", "w/w + wet-mass"))
CONTRASTS = (("B-A", "E1", "E0"), ("C-B", "E5", "E1"), ("C-A", "E5", "E0"))


def load_fit(resp, scheme, model, name):
    path = CACHE / f"{key(resp, scheme, model, name)}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} -- run src/retrospective/fit_models.py first")
    return dict(np.load(path))


def shared_block():
    """Confirm A, B and C differ only in the polymer columns they claim to."""
    df = campaign()
    built = {name: build(df, name) for _, name, _ in STEPS}
    e0_cols = built["E0"][1]
    e1_cols = built["E1"][1]
    e5_cols = built["E5"][1]
    shared = [c for c in e1_cols if c in e0_cols]
    for column in shared:
        a = built["E0"][0][:, e0_cols.index(column)]
        b = built["E1"][0][:, e1_cols.index(column)]
        c = built["E5"][0][:, e5_cols.index(column)]
        if not (np.allclose(a, b) and np.allclose(b, c)):
            raise AssertionError(f"{column} is not shared")
    added = [c for c in e5_cols if c not in e1_cols]
    removed = [c for c in e1_cols if c not in e5_cols]
    if removed or added != ["fluid_per_mcc", "soluble_solids_frac",
                            "hydrophobic_frac", "binder_delivered"]:
        raise AssertionError(f"C minus B is {added}, and B has extras {removed}")
    only_a = [c for c in e0_cols if c not in e1_cols]
    only_b = [c for c in e1_cols if c not in e0_cols]
    print(f"shared non-polymer columns ({len(shared)}): {', '.join(shared)}")
    print(f"A only ({len(only_a)} polymer columns): {', '.join(only_a)}")
    print(f"B only ({len(only_b)} polymer columns): {', '.join(only_b)}")
    print(f"C adds to B: {', '.join(added)}")
    print(f"widths: A {len(e0_cols)}, B {len(e1_cols)}, C {len(e5_cols)}")
    return df


def slot_census(df):
    """How often the campaign itself used both slot orders of one material pair."""
    frame = ds.prepare(df)
    filled = ((frame.cntrl_polymer_0_material != ds.ABSENT)
              & (frame.cntrl_polymer_1_material != ds.ABSENT))
    orders = {}
    for row in frame[filled].itertuples():
        pair = tuple(sorted((row.cntrl_polymer_0_material, row.cntrl_polymer_1_material)))
        orders.setdefault(pair, set()).add(
            (row.cntrl_polymer_0_material, row.cntrl_polymer_1_material))
    both = sorted(pair for pair, seen in orders.items() if len(seen) > 1)
    either_slot = {}
    for index in (0, 1):
        for name in frame[f"cntrl_polymer_{index}_material"]:
            if name == ds.ABSENT:
                continue
            either_slot.setdefault(name, set()).add(index)
    swapped_materials = sorted(name for name, slots in either_slot.items() if len(slots) > 1)
    fractions = material_fractions(df)
    singles = int((fractions.gt(0).sum(axis=1) == 1).sum())
    print(f"two-polymer runs: {int(filled.sum())} of {len(df)}; "
          f"single-polymer runs: {singles}")
    print(f"unordered pairs recorded in both slot orders: {len(both)}"
          + (f" ({both})" if both else ""))
    print(f"materials placed in both slots, across runs ({len(swapped_materials)}): "
          f"{', '.join(swapped_materials)}")


def ablation_table(df):
    rows = []
    for resp in RESPONSES:
        _, y = response(df, resp)
        binary = KIND[resp] == "binary"
        metric = PRIMARY[binary]
        for scheme in SCHEMES:
            for model in ("gp", "linear"):
                fits = {name: load_fit(resp, scheme, model, name) for _, name, _ in STEPS}
                tested = fits["E0"]["tested"].astype(int)
                for name in ("E1", "E5"):
                    if not np.array_equal(fits[name]["tested"], tested):
                        raise AssertionError(f"{resp} {scheme} {model} {name} folds differ")
                yt = y[tested]
                for step, name, label in STEPS:
                    pred = fits[name]["mean"][tested]
                    low, high = metric_ci(yt, pred, binary)
                    point = score(yt, pred, binary)
                    rows.append(dict(response=resp, scheme=scheme, model=model, step=step,
                                     set=name, label=label, n=len(tested), metric=metric,
                                     value=point[metric], rho=point.get("rho", np.nan),
                                     ci_low=low, ci_high=high))
                for contrast, left, right in CONTRASTS:
                    result = paired_bootstrap(yt, fits[left]["mean"][tested],
                                              fits[right]["mean"][tested], binary)
                    rows.append(dict(response=resp, scheme=scheme, model=model, step=contrast,
                                     set=f"{left}-{right}", label=contrast, n=len(tested),
                                     metric=metric, value=result["diff"], rho=np.nan,
                                     ci_low=result["low"], ci_high=result["high"],
                                     p_better=result["p_better"]))
    table = pd.DataFrame(rows)
    cfg.TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.TABLES / "representation_ablation.csv", index=False)
    return table


def table_2(table):
    """Release deviation, GP: leave-one-out against forward-in-time R2."""
    gp = table[(table.response == "release deviation") & (table.model == "gp")
               & table.step.isin(list("ABC"))]
    out = (gp.pivot(index=["step", "label"], columns="scheme", values="value")
           .reset_index()[["step", "label", "loo", "forward"]]
           .rename(columns={"step": "representation", "label": "polymer representation",
                            "loo": "leave_one_out_r2", "forward": "forward_in_time_r2"}))
    out.round(4).to_csv(cfg.TABLES / "table2_release_validation.csv", index=False)
    return out


def figure_6(table):
    """Submission artwork. The caption carries A/B/C, the sample sizes and the
    paired differences; the axes carry only the scores and their intervals.
    """
    style.set_style()
    matplotlib.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7,
    })
    loo = table[(table.scheme == "loo") & table.step.isin(list("ABC"))]
    fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE_COLUMN, 2.85))
    x = np.arange(3)
    panels = (
        ("release deviation", "Release deviation\n$R^2$", (-0.35, 1.08)),
        ("aspect ratio", "Aspect ratio\n$R^2$", (-1.08, 1.08)),
        ("feasibility", "Feasibility\nAUC", (0.40, 1.03)),
    )
    series = (
        ("linear", "s", -0.08, style.FEASIBLE, "penalized linear"),
        ("gp", "o", 0.08, style.TARGET, "Gaussian process"),
    )
    for ax, (resp, ylabel, limits) in zip(axes, panels):
        for model, marker, offset, colour, label in series:
            part = (loo[(loo.response == resp) & (loo.model == model)]
                    .set_index("step").loc[list("ABC")])
            ax.errorbar(
                x + offset, part.value,
                yerr=[part.value - part.ci_low, part.ci_high - part.value],
                fmt=f"-{marker}", ms=4.6, lw=1.0, capsize=2.0, capthick=0.6,
                color=colour, ecolor=colour, elinewidth=0.7, mec="k", mew=0.35,
                zorder=3, label=label,
            )
        ax.axhline(0.5 if resp == "feasibility" else 0.0, color="0.45", lw=0.6, ls=":")
        ax.set_xlim(-0.45, 2.45)
        ax.set_ylim(limits)
        ax.set_xticks(x, list("ABC"))
        ax.set_ylabel(ylabel)
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left", fontsize=6.5, handlelength=1.6, borderpad=0.2)
    for ax, letter in zip(axes, "abc"):
        ax.set_title(f"({letter})", fontsize=9, fontweight="bold", loc="left", pad=2)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.13, top=0.90, wspace=0.48)
    cfg.MANUSCRIPT_FIGURES.mkdir(parents=True, exist_ok=True)
    path = cfg.MANUSCRIPT_FIGURES / "fig06_representation_ablation"
    with matplotlib.rc_context({"savefig.bbox": "standard", "savefig.dpi": 600}):
        fig.savefig(path.with_suffix(".pdf"))
        fig.savefig(path.with_suffix(".png"))
        cfg.FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.FIGURES / "representation_ablation.png")
    print(f"  figure -> {path.relative_to(cfg.ROOT)}.png")
    plt.close(fig)


def report(table):
    for model in ("gp", "linear"):
        for resp in RESPONSES:
            print(f"\n{resp}, {model}, leave-one-out")
            part = table[(table.response == resp) & (table.scheme == "loo")
                         & (table.model == model)]
            for _, row in part.iterrows():
                if row.step in ("A", "B", "C"):
                    rho = "" if not np.isfinite(row.rho) else f", rho {row.rho:+.2f}"
                    print(f"  {row.step} {row.set}: {row.value:.3f} "
                          f"[{row.ci_low:.2f}, {row.ci_high:.2f}]{rho}")
                else:
                    print(f"  {row.step}: {row.value:+.3f} "
                          f"[{row.ci_low:+.2f}, {row.ci_high:+.2f}], "
                          f"p_better {row.p_better:.2f}")


def main():
    df = shared_block()
    slot_census(df)
    table = ablation_table(df)
    report(table)
    print("\nTable 2, release deviation, GP R2:")
    print(table_2(table).round(2).to_string(index=False))
    figure_6(table)


if __name__ == "__main__":
    main()
