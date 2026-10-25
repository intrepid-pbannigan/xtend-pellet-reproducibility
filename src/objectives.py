"""The retrospective Chimera hierarchy (manuscript Table S2).

Edit the numbers in HIERARCHY, then re-score:

    python src/analysis/plot_optimization_trajectories.py

Each row is one objective, in rank order. Chimera is a gate, not a weighted
sum: an experiment that clears the first bar is ranked on the second, and so
on. `absolute=True` means the bar is in the objective's own units.

    release_area_diff     min, 0.50   fraction-hours; below this is on-target enough
    granule_aspect_ratio  min, 1.50   1.0 is a sphere; below this is round enough
    drug_mass             max, 125    mg per 500 mg; keep recipes at or above this.
                                      Chimera's max gate is strict, so a bar of 150
                                      would drop every 150 mg experiment.

Set `absolute=False` on a row to switch that bar back to a fraction of the
observed range (0–1). Drug mass is a design variable, not a measurement; it
is last on purpose.
"""

from __future__ import annotations

# Edit this table. Order is the hierarchy.
HIERARCHY = (
    # name,                    goal,   tolerance, absolute
    ("release_area_diff",      "min",  0.50,      True),
    ("granule_aspect_ratio",   "min",  1.50,      True),
    ("drug_mass",              "max",  125.0,     True),
)

OBJECTIVES = tuple(row[0] for row in HIERARCHY)
GOALS = tuple(row[1] for row in HIERARCHY)
TOLERANCES = tuple(row[2] for row in HIERARCHY)
ABSOLUTES = tuple(row[3] for row in HIERARCHY)

UNITS = {
    "release_area_diff": "fraction-hours",
    "granule_aspect_ratio": "aspect ratio",
    "drug_mass": "mg per 500 mg",
}


def row_for(name):
    for row in HIERARCHY:
        if row[0] == name:
            return row
    raise KeyError(name)


def bar_text(name, goal, tolerance, absolute):
    """One human line for a caption or a log."""
    units = UNITS.get(name, "")
    if absolute:
        if goal == "min":
            return f"{name} < {tolerance:g} {units}".strip()
        return f"{name} > {tolerance:g} {units}".strip()
    if goal == "min":
        return f"{name} within {tolerance:.0%} of the observed range"
    return f"{name} within {tolerance:.0%} of the observed range (maximize)"


def caption():
    """One sentence the figures can print so the bars stay honest."""
    bits = [bar_text(*row) for row in HIERARCHY]
    kind = "absolute" if all(ABSOLUTES) else "mixed"
    return ("Chimera hierarchy, " + kind + " bars: " + "; ".join(bits)
            + ". Edit src/objectives.py and re-run this script.")


def figure_note():
    """Shorter than `caption`, for a figure footer."""
    bits = [bar_text(*row) for row in HIERARCHY]
    return "Chimera bars: " + "; ".join(bits) + "."


def describe():
    return " → ".join(bar_text(*row) for row in HIERARCHY)
