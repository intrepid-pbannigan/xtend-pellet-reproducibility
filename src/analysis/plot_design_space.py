"""Main-text Figure 2: a large design space in which failure is common.

Three panels stacked for the page, and a reconciled Table 1 written beside them.

    (a) which control-polymer combinations were tried and how often each
        formed pellets, as a triangle over the materials actually in the pellet,
        grouped by polymer function: a role at zero w/w is not a partner, A + B
        is the same combination as B + A, and single-polymer formulations have
        their own "alone" column. Most combinations were tried once.

    (b) every experiment as a line across the nine continuous formulation and
        process variables, each axis scaled to the range the campaign used, with
        failures drawn beside successes. They overlap on every continuous axis,
        which is the point: no single setting separates them.

    (c) the formed pellets alone, coloured by release deviation: good release
        is not confined to a range of any continuous setting either.

Table 1 is written from the data. Filler is exactly 1 - drug - polymer -
stearate in every row, so 12 variables were varied
independently, and the process variables are recorded to several decimal
places, with 36 to 40 distinct values each, so they are continuous as run.

    python src/analysis/plot_design_space.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg
import design_space as ds
from plotting import style

PROCESS_VARS = ["gran_time", "gran_rpm", "spheronization_time",
                "spheronization_rpm", "fluid_vol", "fluid_type"]

AXIS_LABELS = {
    "drug_ww_frac": "drug\nw/w",
    "total_cntrl_polymer_ww_frac": "release-\nmodifying\nexcipient w/w",
    "cntrl_polymer_0_relative_ww_frac": "excipient\nratio",
    "mg_stearate_ww_frac": "Mg stearate\nw/w",
    "gran_time": "granulation\ntime",
    "gran_rpm": "granulation\nspeed",
    "spheronization_time": "spheron.\ntime",
    "spheronization_rpm": "spheron.\nspeed",
    "fluid_vol": "fluid\nvolume",
}

UNITS = {
    "drug_ww_frac": "w/w", "total_cntrl_polymer_ww_frac": "w/w",
    "cntrl_polymer_0_relative_ww_frac": "fraction of excipient",
    "mg_stearate_ww_frac": "w/w", "gran_time": "min", "gran_rpm": "rpm",
    "spheronization_time": "min", "spheronization_rpm": "rpm", "fluid_vol": "mL",
}


def design_table(df):
    """Table 1 as the campaign actually ran it."""
    rows = []
    for var in cfg.FORMULATION_VARS + PROCESS_VARS:
        values = df[var]
        role = "process" if var in PROCESS_VARS else "formulation"
        if values.dtype == object:
            levels = sorted(set(values))
            if var in ("cntrl_polymer_0_type", "cntrl_polymer_1_type"):
                levels = sorted(set(df.cntrl_polymer_0_type) | set(df.cntrl_polymer_1_type))
            rows.append(dict(variable=var, role=role, type="categorical",
                             levels_or_range=f"{len(levels)} options",
                             distinct_values=values.nunique(), units=""))
        else:
            kind = "discrete" if values.nunique() <= 12 else "continuous"
            rows.append(dict(variable=var, role=role, type=kind,
                             levels_or_range=f"{values.min():.4g} to {values.max():.4g}",
                             distinct_values=values.nunique(), units=UNITS.get(var, "")))
    rows.append(dict(variable="filler_ww_frac", role="formulation (derived)",
                     type="derived", levels_or_range="1 - drug - polymer - stearate",
                     distinct_values=df.filler_ww_frac.nunique()
                     if "filler_ww_frac" in df else np.nan, units="w/w"))
    return pd.DataFrame(rows)


RELEASE_CMAP = plt.get_cmap("viridis")


def parallel_panel(ax, df, by_release=False, show_failed=True, cax=None, letter="a",
                   legend=True):
    """Experiments across the continuous axes.

    `by_release` colours formed pellets by release deviation instead of one
    colour, drawn worst first so the best recipes sit on top; failures have no
    release, so they stay grey or, with `show_failed=False`, are left out. `cax`
    holds the colour bar where a stacked layout needs the panels aligned.
    """
    variables = list(cfg.CONTINUOUS_VARS)
    lo, hi = df[variables].min(), df[variables].max()
    scaled = (df[variables] - lo) / (hi - lo).replace(0, 1)
    x = np.arange(len(variables))

    for _, row in scaled[~df.feasible].iterrows() if show_failed else ():
        ax.plot(x, row.to_numpy(float), color=style.FAILED, lw=0.9,
                alpha=0.45 if by_release else 0.75, zorder=2)
    formed = df[df.feasible]
    if by_release:
        norm = plt.Normalize(formed.release_deviation.min(), formed.release_deviation.max())
        for index in formed.release_deviation.sort_values(ascending=False).index:
            ax.plot(x, scaled.loc[index].to_numpy(float), lw=1.4, alpha=0.9, zorder=3,
                    color=RELEASE_CMAP(norm(formed.release_deviation[index])))
        mappable = plt.cm.ScalarMappable(cmap=RELEASE_CMAP, norm=norm)
        bar = (ax.figure.colorbar(mappable, cax=cax) if cax is not None else
               ax.figure.colorbar(mappable, ax=ax, fraction=0.03, pad=0.015))
        bar.set_label("release deviation (fraction-hours)", fontsize=7)
        bar.ax.tick_params(labelsize=6.5)
    else:
        for index in formed.index:
            ax.plot(x, scaled.loc[index].to_numpy(float), color=style.FEASIBLE, lw=1.1,
                    alpha=0.6, zorder=3)

    split = len(cfg.FORMULATION_VARS) - 2  # continuous formulation axes come first
    ax.axvspan(split - 0.5, len(variables) - 0.5, color="k", alpha=0.04, lw=0, zorder=0)
    ax.text(split / 2 - 0.5, 1.07, "formulation", ha="center", fontsize=8, color="#444444")
    ax.text((split + len(variables)) / 2 - 0.5, 1.07, "process", ha="center",
            fontsize=8, color="#444444")
    ax.set_xticks(x)
    ax.set_xticklabels([AXIS_LABELS[v] for v in variables], fontsize=7)
    ax.set_xlim(-0.3, len(variables) - 0.7)
    ax.set_ylim(-0.04, 1.13)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["min used", "max used"], fontsize=7)
    ax.grid(axis="y", visible=False)
    for xi in x:
        ax.axvline(xi, color="#999999", lw=0.6, zorder=1)
    formed_handle = plt.Line2D(
        [], [], color=RELEASE_CMAP(0.5) if by_release else style.FEASIBLE, lw=1.4,
        label=f"formed pellets ({len(formed)})" + (", coloured by release" if by_release else ""))
    handles = [formed_handle]
    if show_failed:
        handles.append(plt.Line2D([], [], color=style.FAILED, lw=1.4,
                                  label=f"failed ({int((~df.feasible).sum())})"))
    if legend:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2,
                  frameon=False, fontsize=7.5)
    claim = ("Good release is not confined to a range of any continuous setting" if by_release
             else "No continuous setting separates success from failure")
    ax.set_title(f"({letter}) {claim}", fontsize=9, loc="left")


def polymer_combinations(df):
    """Formed and tried per unordered polymer combination.

    Built from the materials actually in the pellet, so a role recorded at zero
    w/w is not a partner, and A + B is the same combination as B + A. A
    single-polymer formulation has `first == second`; `blends` counts those made
    from two grades the merge treats as one material.
    """
    frame = ds.prepare(df)
    pairs = [tuple(sorted({a, b} - {ds.ABSENT})) for a, b in
             zip(frame.cntrl_polymer_0_material, frame.cntrl_polymer_1_material)]
    frame["first"] = [p[0] for p in pairs]
    frame["second"] = [p[-1] for p in pairs]
    return frame.groupby(["first", "second"]).agg(
        sum=("feasible", "sum"), size=("feasible", "size"),
        blends=("grade_blend", "sum")).reset_index()


ALONE_X = -1.5


CLASS_ORDER = ("gel", "barrier", "soluble_wetting")
CLASS_LABELS = {"gel": "gel former", "barrier": "barrier (lipid / insoluble matrix)",
                "soluble_wetting": "soluble / wetting"}
CLASS_COLOURS = {"gel": "#1B7F3B", "barrier": "#8C6D46", "soluble_wetting": "#7B5AA6"}


def material_classes():
    """Functional class of each canonical material, from the descriptor table.

    The table is one row per recorded grade; merged grades share a class, which
    is checked rather than assumed.
    """
    table = pd.read_csv(cfg.EXCIPIENT_DESCRIPTORS_CSV)
    classes = {}
    for grade, kind in zip(table.excipient, table.functional_class):
        material = cfg.MATERIAL_ALIASES.get(grade, grade)
        if classes.setdefault(material, kind) != kind:
            raise ValueError(f"grades of {material} disagree on functional class")
    return classes


def class_strips(ax, polymers, classes, n, legend=True):
    """Coloured strips along both axes marking each material's class."""
    for i, polymer in enumerate(polymers):
        colour = CLASS_COLOURS[classes[polymer]]
        ax.add_patch(plt.Rectangle((ALONE_X - 0.95, i - 0.5), 0.3, 1, facecolor=colour,
                                   edgecolor="w", lw=0.8, zorder=2))
        if i < n - 1:
            ax.add_patch(plt.Rectangle((i - 0.5, n - 0.4), 1, 0.3, facecolor=colour,
                                       edgecolor="w", lw=0.8, zorder=2))
    for tick, polymer in zip(ax.get_yticklabels(), polymers):
        tick.set_color(CLASS_COLOURS[classes[polymer]])
    for tick, polymer in zip(ax.get_xticklabels()[1:], polymers):
        tick.set_color(CLASS_COLOURS[classes[polymer]])
    if not legend:
        return
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CLASS_COLOURS[k], label=CLASS_LABELS[k])
               for k in CLASS_ORDER if k in {classes[p] for p in polymers}]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=6.5,
              title="polymer function", title_fontsize=7)


DISPLAY_NAMES = {
    "carbopol_71": "Carbopol 71G NF", "carbopol_940nf": "Carbopol 940NF",
    "carbopol_974p": "Carbopol 974P", "cmc": "CMC", "hpmc": "HPMC",
    "compritol_hd5": "Compritol HD5", "eudragit_sr": "Eudragit RS",
    "kolidon_sr": "Kollidon SR", "koliphor_p188": "Kolliphor P188", "pvp_30k": "PVP K30",
}


def pair_panel(ax, df, classes=None, letter="b", legend=True, cax=None):
    polymers = sorted(set(df.cntrl_polymer_0_type) | set(df.cntrl_polymer_1_type))
    if classes is not None:
        polymers.sort(key=lambda p: (CLASS_ORDER.index(classes[p]), p))
    index = {p: i for i, p in enumerate(polymers)}
    pairs = polymer_combinations(df)
    cmap = plt.get_cmap("RdYlBu")
    n = len(polymers)
    # Pairs fill the strict lower triangle; single-polymer formulations get
    # their own column, so no cell reads as a material combined with itself.
    for i in range(n):
        ax.add_patch(plt.Rectangle((ALONE_X - 0.5, i - 0.5), 1, 1, facecolor="#E9ECF2",
                                   edgecolor="w", lw=1, zorder=1))
        for j in range(i):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#F4F4F4",
                                       edgecolor="w", lw=1, zorder=1))
    for _, row in pairs.iterrows():
        formed, tried = int(row["sum"]), int(row["size"])
        if row["first"] == row["second"]:
            x, y = ALONE_X, index[row["first"]]
        else:
            y, x = sorted((index[row["first"]], index[row["second"]]), reverse=True)
        rate = formed / tried
        # A filled cell rather than a marker sized by count: the panel is drawn
        # to the same box as the others, so cells are wider than tall, and the
        # count is in the label.
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=cmap(rate),
                                   edgecolor="w", lw=1, zorder=3))
        ax.text(x, y, f"{formed}/{tried}", ha="center", va="center", fontsize=6,
                color="w" if rate < 0.2 or rate > 0.8 else "k", zorder=4,
                gid=style.DATA_LABEL)
    labels = [DISPLAY_NAMES.get(p, p.replace("_", " ")) for p in polymers]
    ax.set_xticks([ALONE_X] + list(range(n - 1)))
    ax.set_xticklabels(["alone"] + labels[:-1], rotation=35, ha="right",
                       rotation_mode="anchor", fontsize=6.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlim(ALONE_X - 0.6, n - 1.4)
    ax.set_ylim(n - 0.4, -0.6)
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if classes is not None:
        ax.set_xlim(ALONE_X - 1.0, n - 1.4)
        ax.set_ylim(n - 0.05, -0.6)
        ax.tick_params(length=0)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        class_strips(ax, polymers, classes, n, legend=legend)
    scale = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    bar = (ax.figure.colorbar(scale, cax=cax, ticks=[0, 0.5, 1]) if cax is not None else
           ax.figure.colorbar(scale, ax=ax, fraction=0.035, pad=0.02, ticks=[0, 0.5, 1]))
    bar.set_label("fraction formed", fontsize=7)
    bar.ax.tick_params(labelsize=6.5)
    two = pairs[pairs["first"] != pairs["second"]]
    alone = pairs[pairs["first"] == pairs["second"]]
    ax.set_title(f"({letter}) {len(two)} of {n * (n - 1) // 2} polymer pairs tried, "
                 f"{int((two['size'] == 1).sum())} once; {len(alone)} polymers also "
                 f"run alone\nlabel = formed / tried",
                 fontsize=9, loc="left")
    return pairs


def main():
    style.set_style()
    df = pd.read_csv(cfg.CAMPAIGN_CSV)

    # Stacked for the page: the polymer pairs, where feasibility is decided;
    # the continuous axes as formed against failed; then the formed pellets
    # alone, coloured by release. All three panels share one box size, with
    # colour bars in their own column, and colours are keyed in the caption
    # rather than by legends over the data. Polymer function is the class in
    # data/reference/excipient_descriptors.csv (Table S1).
    fig = plt.figure(figsize=(style.DOUBLE_COLUMN, style.MAX_HEIGHT))
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 0.025], hspace=0.42, wspace=0.04)
    pairs = pair_panel(fig.add_subplot(grid[0, 0]), df, classes=material_classes(),
                       letter="a", legend=False, cax=fig.add_subplot(grid[0, 1]))
    parallel_panel(fig.add_subplot(grid[1, 0]), df, letter="b", legend=False)
    parallel_panel(fig.add_subplot(grid[2, 0]), df, by_release=True, show_failed=False,
                   cax=fig.add_subplot(grid[2, 1]), letter="c", legend=False)
    fig.subplots_adjust(top=0.97, bottom=0.07, left=0.15, right=0.91)
    style.save(fig, "design_space")

    table = design_table(df)
    cfg.TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.TABLES / "design_space.csv", index=False)

    independent = table[table.type != "derived"]
    print(f"{len(df)} experiments, {int(df.feasible.sum())} formed pellets")
    print(f"independent variables: {len(independent)} "
          f"({int((independent.role == 'formulation').sum())} formulation, "
          f"{int((independent.role == 'process').sum())} process)")
    print(f"polymer combinations tried: {len(pairs)}, of which "
          f"{int((pairs['size'] == 1).sum())} once and "
          f"{int((pairs['first'] == pairs['second']).sum())} single-polymer")
    print(pairs.to_string(index=False))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
