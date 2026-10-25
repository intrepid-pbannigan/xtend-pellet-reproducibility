"""Manuscript Tables 1, 2 and S1-S3 as images, from the CSVs in outputs/tables/.

Each table is drawn in the three-rule style (top, below the header, bottom) at
the RSC double-column width and written to outputs/manuscript/tables/ as PDF
and 600 dpi PNG. Titles and captions are left to the manuscript.

Table 1 gives the permitted levels of the design space as specified for the
campaign; the continuous ranges are the observed ones from table1_design_space.csv.
The filler is all microcrystalline cellulose and is the balance of the
formulation, so there is no filler-ratio variable.

    python src/manuscript/render_tables.py
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg  # noqa: E402
from plotting import style  # noqa: E402

OUT = cfg.MANUSCRIPT_FIGURES / "tables"
FONT = 7.0
LINE = FONT * 1.45 / 72
PAD = 0.035
RULE_HEAVY, RULE_LIGHT = 0.9, 0.5
CHAR_WIDTH = FONT * 0.56 / 72


def wrap(text, width_in):
    chars = max(int(width_in / CHAR_WIDTH), 4)
    lines = []
    for part in str(text).split("\n"):
        lines += textwrap.wrap(part, chars, break_long_words=False) or [""]
    return lines


def draw(name, header, rows, widths, align=None, groups=None):
    """Draw a table; `widths` are column fractions, `groups` row indices that start a group."""
    width = style.DOUBLE_COLUMN
    columns = [w * width for w in widths]
    align = align or ["left"] * len(header)
    cells = [[wrap(c, w - 0.08) for c, w in zip(row, columns)] for row in [header] + rows]
    heights = [max(len(c) for c in row) * LINE + 2 * PAD for row in cells]
    height = sum(heights) + 0.04
    fig = plt.figure(figsize=(width, height))

    def y(inches):
        return 1 - inches / height

    def rule(at, weight):
        fig.add_artist(plt.Line2D([0, 1], [y(at), y(at)], lw=weight, color="k"))

    top = 0.02
    rule(top, RULE_HEAVY)
    at = top
    for i, (row, h) in enumerate(zip(cells, heights)):
        if groups and i - 1 in groups and i > 1:
            rule(at, 0.3)
        left = 0.0
        for lines, w, a in zip(row, columns, align):
            x = left + 0.04 if a == "left" else left + w / 2
            fig.text(x / width, y(at + PAD), "\n".join(lines), ha=a if a == "left" else "center",
                     va="top", fontsize=FONT, linespacing=1.45,
                     fontweight="bold" if i == 0 else "normal")
            left += w
        at += h
        if i == 0:
            rule(at, RULE_LIGHT)
    rule(at, RULE_HEAVY)

    OUT.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"savefig.bbox": "standard", "savefig.dpi": 600}):
        fig.savefig(OUT / f"{name}.pdf")
        fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  {name} -> {(OUT / name).relative_to(cfg.ROOT)}.pdf/.png")


def number(value, digits):
    return f"{float(value):,.{digits}f}"


def table1():
    t = pd.read_csv(cfg.TABLES / "table1_design_space.csv").set_index("variable")

    def span(var, digits, units):
        low, high = t.loc[var, "levels_or_range"].split(" to ")
        return f"Continuous: {number(low, digits)}–{number(high, digits)} {units}"

    formulation = [
        ("Drug w/w", "Discrete: 0.10–0.40 in steps of 0.05"),
        ("Total control-polymer w/w", "Discrete: 0.15–0.50 in steps of 0.05"),
        ("Control polymer 1", "Categorical: 10 materials"),
        ("Control polymer 2", "Categorical: 10 materials or none"),
        ("Polymer ratio (polymer 1 share)", "Discrete: 0–1.0 in steps of 0.1"),
        ("Mg stearate w/w", "Discrete: 0.005–0.015 in steps of 0.0025"),
        ("Filler (MCC) w/w", "Balance: 1 − drug − polymer − Mg stearate"),
    ]
    process = [
        ("Granulation time", span("gran_time", 1, "min")),
        ("Granulation speed", span("gran_rpm", 0, "rpm")),
        ("Spheronization time", span("spheronization_time", 1, "min")),
        ("Spheronization speed", span("spheronization_rpm", 0, "rpm")),
        ("Granulating fluid", "Categorical: water; water:ethanol 50:50; 5% PVP K30"),
        ("Fluid volume", span("fluid_vol", 1, "mL")),
    ]
    rows = [["Formulation", *r] for r in formulation] + [["Process", *r] for r in process]
    for i in range(1, len(formulation)):
        rows[i][0] = ""
    for i in range(len(formulation) + 1, len(rows)):
        rows[i][0] = ""
    draw("table1_design_space", ["", "Variable", "Type / permitted levels"], rows,
         [0.16, 0.34, 0.50], groups={len(formulation)})


def table2():
    t = pd.read_csv(cfg.TABLES / "table2_release_validation.csv")
    names = {"A": "Slot-specific one-hot encoding", "B": "Role-independent material w/w",
             "C": "B + four wet-mass descriptors"}
    rows = [[r.representation, names[r.representation], f"{r.leave_one_out_r2:.2f}",
             f"{r.forward_in_time_r2:.2f}"] for r in t.itertuples()]
    draw("table2_release_validation",
         ["Representation", "Polymer representation", "Leave-one-out R²", "Forward-in-time† R²"],
         rows, [0.18, 0.40, 0.21, 0.21], align=["left", "left", "center", "center"])


def tableS1():
    t = pd.read_csv(cfg.TABLES / "tableS1_excipient_classes.csv")
    classes = {"gel": "Gel-forming", "barrier": "Hydrophobic / insoluble matrix-forming",
               "soluble_wetting": "Soluble / wetting"}
    water = {"swelling_insoluble": "swelling, insoluble", "soluble_gel": "soluble gel",
             "insoluble_lipid": "insoluble lipid", "insoluble_matrix": "insoluble matrix",
             "soluble": "soluble"}
    products = {"hpmc": "HPMC (2,600–5,600 cP)", "cmc": "Sodium CMC", "pvp_30k": "PVP K30",
                "pvp_40k": "PVP K40"}
    rows = [[products.get(r.excipient, r["product"]),
             r.supplier.replace("Gattefosse", "Gattefossé"),
             r.chemistry if "CAS" in r.chemistry else re.sub(r"(\d)-(\d)", "\\1–\\2", r.chemistry),
             classes[r.functional_class], water[r.water_interaction], r.ionic_character,
             "–" if pd.isna(r.hlb) else f"{r.hlb:g}"] for _, r in t.iterrows()]
    draw("tableS1_excipient_classes",
         ["Material", "Supplier", "Chemistry", "Functional class", "Water interaction",
          "Ionic character", "HLB"],
         rows, [0.15, 0.12, 0.31, 0.16, 0.11, 0.10, 0.05],
         align=["left"] * 6 + ["center"])


def tableS2():
    t = pd.read_csv(cfg.TABLES / "tableS2_chimera_hierarchy.csv")
    names = {"release_area_diff": ("Release deviation, D", "fraction-hours"),
             "granule_aspect_ratio": ("Median aspect ratio", "–"),
             "drug_mass": ("Drug loading", "mg per 500 mg")}
    rows = []
    for r in t.itertuples():
        label, units = names[r.objective]
        sign = "<" if r.goal == "min" else ">"
        digits = 0 if r.objective == "drug_mass" else 2
        rows.append([f"{r.rank}", label, units, r.goal.capitalize(),
                     f"{sign} {r.threshold:.{digits}f}"])
    draw("tableS2_chimera_hierarchy",
         ["Priority", "Objective", "Units", "Goal", "Chimera tolerance"], rows,
         [0.12, 0.32, 0.22, 0.12, 0.22], align=["center", "left", "left", "center", "center"])


def tableS3():
    t = pd.read_csv(cfg.TABLES / "tableS3_segmentation_parameters.csv")
    units = {"um": "µm", "px": "px", "-": ""}
    rows = []
    for r in t.itertuples():
        value = f"{r.value:,g}"
        rows.append([r.description[0].upper() + r.description[1:],
                     f"{value} {units[r.units]}".strip()])
    rows.append(["Foreground threshold", "Otsu on HSV value, per image"])
    rows.append(["Aspect ratio", "Major / minor axis of the moment ellipse"])
    draw("tableS3_segmentation_parameters", ["Step", "Value"], rows, [0.62, 0.38])


def main():
    style.set_style()
    table1()
    table2()
    tableS1()
    tableS2()
    tableS3()


if __name__ == "__main__":
    main()
