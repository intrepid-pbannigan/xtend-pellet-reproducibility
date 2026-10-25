"""Manuscript Figures 2-5, S1 and S2, plus the tables their scripts write.

Each figure script draws an annotated working figure into outputs/figures/
(titles that state the finding, annotation boxes, point labels). For the
submission artwork this script reruns the main-figure scripts with `style.save`
intercepted, strips the commentary from the finished figure, sets it to the RSC
double-column width, and writes it to outputs/manuscript/ under the manuscript
numbering. Everything removed is listed in outputs/manuscript/removed_text.txt,
so nothing said on the annotated version is lost before the caption says it.

What is removed: axes titles (replaced by the panel letter), every text inside
an axes, and every figure-level text. What stays: axes, ticks, axis and
colour-bar labels, legends, and text tagged `style.DATA_LABEL` (the
formed/tried counts in Figure 2a).

Figure S1 is a photograph montage whose panel titles are its labels, so it is
copied as drawn. Figures 6 and 7 are written directly by
`retrospective/ablation.py` and `retrospective/figure7.py`.

    python src/manuscript/build_figures.py
"""

from __future__ import annotations

import contextlib
import io
import re
import runpy
import shutil
import string
import sys
from pathlib import Path

import matplotlib
import matplotlib.collections
import matplotlib.lines
import matplotlib.text

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg  # noqa: E402
from plotting import style  # noqa: E402

# (manuscript name, figure stem the script saves, script, clean?)
FIGURES = [
    ("fig02_design_space", "design_space", "src/analysis/plot_design_space.py", True),
    ("fig03_campaign_and_feasibility", "campaign_and_feasibility",
     "src/analysis/plot_campaign_and_feasibility.py", True),
    ("fig04_release_exemplars", "release_exemplars",
     "src/analysis/plot_release_exemplars.py", True),
    ("fig05_multiobjective", "multiobjective", "src/analysis/plot_multiobjective.py", True),
    ("figS01_segmentation_pipeline", "segmentation_pipeline",
     "src/analysis/plot_segmentation_pipeline.py", False),
    ("figS02_particle_morphology_validation", "particle_morphology_validation",
     "src/analysis/analyze_particle_morphology.py", True),
]

OUT = cfg.MANUSCRIPT_FIGURES
LETTER = re.compile(r"^\s*\(([a-z])\)")
FONT_MIN, FONT_MAX, TICK_SIZE, LETTER_SIZE = 6.0, 7.5, 6.5, 9


def panel_axes(fig):
    """Axes that are panels: not colour bars, not twins of an earlier axes."""
    seen, panels = [], []
    for ax in fig.axes:
        colorbar = ax.get_label() == "<colorbar>" or getattr(ax, "_colorbar", None) is not None
        if colorbar or not ax.get_visible() or not ax.axison:
            continue
        bounds = tuple(np.round(ax.get_position().bounds, 4))
        if bounds in seen:
            continue
        seen.append(bounds)
        panels.append(ax)
    return panels


def publication(fig):
    """Resize to RSC double-column width and put all text on one scale.

    The analysis figures are drawn 10-13 in wide; shrinking them as they are
    would leave 4-5 pt text. Fonts are set to a fixed range instead, lines and
    markers thinned with the square root of the shrink, and the layout redone
    so the larger relative text still fits.
    """
    width, height = fig.get_size_inches()
    shrink = style.DOUBLE_COLUMN / width
    native = abs(shrink - 1) < 0.01
    fig.set_size_inches(style.DOUBLE_COLUMN, min(height * shrink, style.MAX_HEIGHT))
    thin = np.sqrt(min(shrink, 1.0))
    for text in fig.findobj(matplotlib.text.Text):
        size = text.get_fontsize()
        text.set_fontsize(float(np.clip(size * 0.85, FONT_MIN, FONT_MAX)))
    # Tick labels are rebuilt at draw time from the axis settings, not kept.
    for ax in fig.axes:
        ax.tick_params(labelsize=TICK_SIZE)
    for line in fig.findobj(matplotlib.lines.Line2D):
        line.set_linewidth(line.get_linewidth() * thin)
        line.set_markersize(line.get_markersize() * thin)
    for collection in fig.findobj(matplotlib.collections.Collection):
        sizes = collection.get_sizes() if hasattr(collection, "get_sizes") else ()
        if len(sizes):
            collection.set_sizes(sizes * thin ** 2)
        collection.set_linewidth(np.asarray(collection.get_linewidth()) * thin)
    return native


def relayout(fig, native):
    """Fit the panels to the new size, keeping figure legends clear."""
    if native:
        return
    bottom = 0.0
    fig.canvas.draw()
    for legend in fig.legends:
        box = legend.get_window_extent().transformed(fig.transFigure.inverted())
        bottom = max(bottom, box.y1 + 0.01)
    # Constrained layout handles the twin axes and mixed grids tight_layout
    # skips. Figures already drawn at page width keep their own layout.
    fig.set_layout_engine("constrained", rect=(0, bottom, 1, 1 - bottom),
                          w_pad=0.04, h_pad=0.04)


def strip(fig):
    """Remove commentary in place; return what was removed, for the caption."""
    removed = []
    panels = panel_axes(fig)
    letters = {}
    for ax in fig.axes:
        for loc in ("left", "center", "right"):
            title = ax.get_title(loc=loc)
            if title:
                removed.append(f"title: {title}")
                match = LETTER.match(title)
                if match:
                    letters[ax] = match.group(1)
                ax.set_title("", loc=loc)
        for text in list(ax.texts):
            if text.get_gid() == style.DATA_LABEL:
                continue
            if text.get_text().strip():
                removed.append(f"text: {text.get_text()}")
            text.remove()
    for text in list(fig.texts):
        if text.get_text().strip():
            removed.append(f"figure text: {text.get_text()}")
        text.remove()
    if len(panels) > 1:
        for index, ax in enumerate(panels):
            letter = letters.get(ax, string.ascii_lowercase[index])
            ax.set_title(f"({letter})", loc="left", fontsize=LETTER_SIZE, fontweight="bold")
    return removed


def run(script):
    with contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(str(cfg.ROOT / script), run_name="__main__")


def build_clean(name, stem, script):
    """Run a figure script; save the annotated figure and a cleaned copy."""
    captured = {}
    original = style.save

    def intercept(fig, figure_name, directory=None, extra=None):
        if figure_name != stem:
            original(fig, figure_name, directory=directory, extra=extra)
            return
        cfg.FIGURES.mkdir(parents=True, exist_ok=True)
        headings = style.drop_headings(fig, stem)
        for suffix in (".pdf", ".png"):
            fig.savefig(cfg.FIGURES / f"{stem}{suffix}")
        native = publication(fig)
        captured[name] = headings + strip(fig)
        relayout(fig, native)
        target = OUT / name
        # No tight bounding box: every figure keeps exactly the page width.
        with matplotlib.rc_context({"savefig.bbox": "standard"}):
            fig.savefig(target.with_suffix(".pdf"))
            fig.savefig(target.with_suffix(".png"), dpi=600)
        plt.close(fig)

    style.save = intercept
    try:
        run(script)
    finally:
        style.save = original
    if name not in captured:
        raise SystemExit(f"{script} never saved {stem}")
    return captured[name]


def build_as_drawn(name, stem, script):
    if not (cfg.ROOT / "data" / "raw").is_dir():
        print(f"{name}: skipped, needs the photographs in data/raw/ (see README)")
        return
    run(script)
    for suffix in (".pdf", ".png"):
        shutil.copy2(cfg.FIGURES / f"{stem}{suffix}", OUT / f"{name}{suffix}")
    print(f"{name}: copied as drawn")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg.TABLES.mkdir(parents=True, exist_ok=True)
    log = []
    for name, stem, script, clean in FIGURES:
        if clean:
            removed = build_clean(name, stem, script)
            log.append(f"== {name}\n" + "\n".join(f"  {r}" for r in removed))
            print(f"{name}: {len(removed)} text items moved to the caption")
        else:
            build_as_drawn(name, stem, script)
    (OUT / "removed_text.txt").write_text("\n\n".join(log) + "\n")
    print(f"manuscript figures -> {OUT.relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
