"""Manuscript Figure 7: the retrospectively selected surrogates and their inputs.

One surrogate per response, each the retrospective GP (Atlas-style classifier
for feasibility, mixed GP otherwise) with polymers described by what they are
rather than which role they were recorded in:

    feasibility    physicochemical descriptors: mass-weighted water-interaction
                   class, gel viscosity, HLB, thermal transition, ionic character
                   and granular grade                                    (E3)
    release        w/w of each material                                   (E1)
    aspect ratio   w/w of each material plus wet-mass descriptors: fluid per
                   MCC, soluble-solids fraction, hydrophobic fraction, PVP
                   binder delivered                                       (E5)

With the filler correctly all cellulose (config.FILLER_IS_CELLULOSE_ONLY) the
wet-mass descriptors no longer help release (leave-one-out R2 0.38 against 0.60
without them), so release uses material w/w alone.

All three share drug and Mg stearate w/w, the five process settings and the
granulating fluid. Rows: release, aspect ratio, feasibility. Left, leave-one-out
prediction against observation; right, leave-one-input-out ablation
(`importance.py`).

Reads the cached fits and outputs/tables/input_ablation.csv; nothing is refitted.

    python src/retrospective/figure7.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import CACHE, campaign, cfg, response  # noqa: E402
from cv import score  # noqa: E402

from plotting import style  # noqa: E402

UPDATED = {"feasibility": "E3", "release deviation": "E1", "aspect ratio": "E5"}
LETTER = dict(fontsize=9, fontweight="bold", loc="left")

LABELS = {
    "drug_ww_frac": "drug w/w", "mg_stearate_ww_frac": "Mg stearate w/w",
    "gran_time": "granulation time", "gran_rpm": "granulation speed",
    "spheronization_time": "spheronization time", "spheronization_rpm": "spheronization speed",
    "fluid_vol": "fluid volume", "fluid_type": "granulating fluid", "fluid": "granulating fluid",
    "control_polymer_0": "polymer A identity", "control_polymer_1": "polymer B identity",
    "cntrl_polymer_0_ww_frac": "polymer A w/w", "cntrl_polymer_1_ww_frac": "polymer B w/w",
    "swelling_insoluble_ww": "swelling-insoluble w/w", "soluble_gel_ww": "soluble gel-former w/w",
    "insoluble_lipid_ww": "lipid w/w", "insoluble_matrix_ww": "insoluble matrix w/w",
    "soluble_ww": "soluble polymer w/w", "gel_log_viscosity_load": "gel viscosity load",
    "hlb_load": "HLB load", "transition_load": "thermal-transition load",
    "anionic_ww": "anionic w/w", "cationic_ww": "cationic w/w", "granular_ww": "granular-grade w/w",
    "fluid_per_mcc": "fluid per MCC", "soluble_solids_frac": "soluble solids fraction",
    "hydrophobic_frac": "hydrophobic fraction", "binder_delivered": "PVP binder delivered",
    "ww=carbopol_71": "Carbopol 71G w/w", "ww=carbopol_940nf": "Carbopol 940NF w/w",
    "ww=carbopol_974p": "Carbopol 974P w/w", "ww=cmc": "CMC w/w",
    "ww=compritol_hd5": "Compritol w/w", "ww=eudragit_sr": "Eudragit w/w",
    "ww=hpmc": "HPMC w/w", "ww=kolidon_sr": "Kollidon SR w/w",
    "ww=koliphor_p188": "Kolliphor P188 w/w", "ww=pvp_30k": "PVP w/w",
}
PROCESS = {"gran_time", "gran_rpm", "spheronization_time", "spheronization_rpm", "fluid_vol",
           "fluid_type", "fluid", "fluid_per_mcc", "binder_delivered"}
ROWS = (("release deviation", "R$^2$"), ("aspect ratio", "R$^2$"), ("feasibility", "AUC"))


def fit(resp, name, scheme="loo"):
    path = CACHE / f"{resp.replace(' ', '_')}__{scheme}__gp__{name}__plain.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} -- run src/retrospective/fit_models.py first")
    return dict(np.load(path))


def parity(ax, df, resp, label):
    _, y = response(df, resp)
    model = fit(resp, UPDATED[resp])
    ax.errorbar(y, model["mean"], yerr=model["sigma"], fmt="o", ms=4, color=style.FEASIBLE,
                mec="k", mew=0.4, ecolor="#9DBBD6", elinewidth=0.7, capsize=0, zorder=3)
    values = np.concatenate([y, model["mean"] - model["sigma"], model["mean"] + model["sigma"]])
    pad = 0.05 * (values.max() - values.min())
    limits = (values.min() - pad, values.max() + pad)
    ax.plot(limits, limits, color="k", lw=0.7, ls="--", zorder=1)
    ax.set(xlim=limits, ylim=limits, aspect="equal",
           xlabel=f"observed {label}", ylabel=f"predicted {label}")


def feasibility(ax, df):
    _, y = response(df, "feasibility")
    p = fit("feasibility", UPDATED["feasibility"])["mean"]
    rng = np.random.default_rng(0)
    for outcome, colour in ((0, style.FAILED), (1, style.FEASIBLE)):
        values = p[y == outcome]
        ax.boxplot([values], positions=[outcome], widths=0.5, showfliers=False,
                   medianprops=dict(color="k"), boxprops=dict(color="#555555"),
                   whiskerprops=dict(color="#555555"), capprops=dict(color="#555555"))
        ax.scatter(outcome + rng.uniform(-0.15, 0.15, len(values)), values, s=14,
                   color=colour, edgecolor="k", linewidth=0.4, zorder=3)
    ax.set_xticks([0, 1], [f"failed (n = {int((y == 0).sum())})",
                           f"formed (n = {int((y == 1).sum())})"])
    ax.set(xlim=(-0.6, 1.6), ylim=(0, 1), ylabel="predicted P(pellets form)", xlabel="outcome")


def importance(ax, table, resp, metric, n_show=10):
    part = table[(table.response == resp) & (table.set == UPDATED[resp])]
    part = part.reindex(part.importance.abs().sort_values(ascending=False).index).head(n_show)
    part = part.sort_values("importance")
    y = np.arange(len(part))
    colours = ["#999999" if f in PROCESS else style.FEASIBLE for f in part.feature]
    ax.barh(y, part.importance, color=colours, edgecolor="k", linewidth=0.4, height=0.7, zorder=2)
    ax.errorbar(part.importance, y, xerr=[part.importance - part.low, part.high - part.importance],
                fmt="none", ecolor="k", elinewidth=0.6, capsize=1.2, zorder=3)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(y, [LABELS.get(f, f) for f in part.feature], fontsize=6.3)
    ax.set_ylim(-0.6, len(part) - 0.4)
    ax.set_xlabel(rf"Predictive contribution, $\Delta${metric}", labelpad=11)
    for x, ha, text in ((0.0, "left", "← improves when removed"),
                        (1.0, "right", "worsens when removed →")):
        ax.annotate(text, (x, 0), xycoords="axes fraction", xytext=(0, -14),
                    textcoords="offset points", ha=ha, va="top", fontsize=5.5, color="#555555")
    ax.grid(axis="y", visible=False)


def figure_7(df):
    table = pd.read_csv(cfg.TABLES / "input_ablation.csv")
    width, height = style.DOUBLE_COLUMN, style.MAX_HEIGHT
    fig = plt.figure(figsize=(width, height))
    side = 2.15
    bottoms = (6.6, 3.6, 0.6)
    letters = iter("abcdef")
    for (resp, metric), bottom in zip(ROWS, bottoms):
        left = fig.add_axes((0.6 / width, bottom / height, side / width, side / height))
        right = fig.add_axes((4.35 / width, bottom / height, 2.25 / width, side / height))
        if resp == "release deviation":
            parity(left, df, resp, "release deviation (fraction-hours)")
        elif resp == "aspect ratio":
            parity(left, df, resp, "median aspect ratio")
        else:
            feasibility(left, df)
        importance(right, table, resp, metric)
        for ax in (left, right):
            ax.set_title(f"({next(letters)})", **LETTER)
            ax.tick_params(labelsize=6.3)
            ax.xaxis.label.set_size(6.8)
            ax.yaxis.label.set_size(6.8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=style.FEASIBLE, label="formulation input"),
               plt.Rectangle((0, 0), 1, 1, color="#999999", label="process / fluid input")]
    right.legend(handles=handles, loc="lower right", fontsize=6.2, handlelength=1.2)
    cfg.MANUSCRIPT_FIGURES.mkdir(parents=True, exist_ok=True)
    path = cfg.MANUSCRIPT_FIGURES / "fig07_surrogate_importance"
    with matplotlib.rc_context({"savefig.bbox": "standard", "savefig.dpi": 600}):
        fig.savefig(path.with_suffix(".pdf"))
        fig.savefig(path.with_suffix(".png"))
        cfg.FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.FIGURES / "surrogate_importance.png")
    print(f"  figure -> {path.relative_to(cfg.ROOT)}.png")
    plt.close(fig)


def main():
    style.set_style()
    figure_7(campaign())


if __name__ == "__main__":
    main()
