"""Shared figure style and save helper, so every figure in outputs/figures
looks the same and is produced the same way.
"""

import math
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg

# colourblind-safe, used consistently across all figures
FEASIBLE = "#0072B2"
FAILED = "#B0B0B0"
TARGET = "#D55E00"
ACCENT = "#009E73"

# The 2026 remakes and repeats. One colour and one marker per preparation
# everywhere they appear, and never a colour used for a campaign quality class,
# because they carry no visual score and belong to no batch: wherever they are
# drawn they sit outside the statistics of the panel they are drawn in.
REMAKE = "#B0006B"
REMAKE_MARKER = "D"
REPEAT = "#5D3A9B"
REPEAT_MARKER = "s"
BATCH12 = "#56B4E9"
BATCH12_MARKER = "^"
BATCH13 = "#CC6677"
BATCH13_MARKER = "v"
BATCH13_R2 = "#7B3294"
BATCH13_R2_MARKER = "P"

# Keyed by imaging series rather than by preparation number, because the series is
# what decides how a sample was measured: the camera, the background and whether
# there is a calibration at all. The preparation number varies independently --
# September holds the second preparation of formulations 43 and 44 and the third
# of 4 and 5 -- and is carried in the label instead.
SERIES_COLOURS = {cfg.CAMPAIGN_SERIES: FEASIBLE, cfg.REMAKE_SERIES: REMAKE,
                  cfg.REPEAT_SERIES: REPEAT, cfg.BATCH12_SERIES: BATCH12,
                  cfg.BATCH13_SERIES: BATCH13, cfg.BATCH13_R2_SERIES: BATCH13_R2}
SERIES_MARKERS = {cfg.CAMPAIGN_SERIES: "o", cfg.REMAKE_SERIES: REMAKE_MARKER,
                  cfg.REPEAT_SERIES: REPEAT_MARKER, cfg.BATCH12_SERIES: BATCH12_MARKER,
                  cfg.BATCH13_SERIES: BATCH13_MARKER,
                  cfg.BATCH13_R2_SERIES: BATCH13_R2_MARKER}
SERIES_LABELS = {cfg.CAMPAIGN_SERIES: "2024 campaign", cfg.REMAKE_SERIES: "August 2026",
                 cfg.REPEAT_SERIES: "September 2026", cfg.BATCH12_SERIES: "batch 12",
                 cfg.BATCH13_SERIES: "batch 13",
                 cfg.BATCH13_R2_SERIES: "batch 13 re-assay"}

QUALITY_COLOURS = {
    0: "#B0B0B0",  # failed to form
    1: "#CC79A7",  # unacceptable
    2: "#E69F00",  # acceptable
    3: "#009E73",  # excellent
}

QUALITY_LABELS = {
    0: "failed to form",
    1: "unacceptable",
    2: "acceptable",
    3: "excellent",
}


def quality_colour(quality):
    """Colour for a 0-3 pellet-quality score, or batch 12 when that score is absent.

    Kept for the analyses whose subject is the visual score itself. Figures
    that used the score as a stand-in for "how round is this pellet" use
    `aspect_colour` instead, which ranks the same property continuously.
    """
    if _missing_quality(quality):
        return BATCH12
    return QUALITY_COLOURS[int(quality)]


def quality_marker(quality):
    if _missing_quality(quality):
        return BATCH12_MARKER
    return "o"


def _missing_quality(quality):
    if quality is None:
        return True
    try:
        return math.isnan(float(quality))
    except (TypeError, ValueError):
        return True


# --- the continuous morphology ranking --------------------------------------
#
# Median granule aspect ratio, which replaced the four visual quality classes
# as the campaign's morphology objective. Those classes turned out to be
# non-overlapping intervals of exactly this quantity, with two experiments in
# each of the end classes, so a figure that colours by them is showing a
# coarsened version of what is plotted here.
#
# The ramp runs through the old class colours -- excellent green, acceptable
# amber, unacceptable pink -- so a point keeps roughly the colour it had in the
# four-class figures and the two generations stay comparable. That is
# continuity of appearance only; the stops carry no threshold.
#
# Limits are fixed rather than taken per figure, so the same colour means the
# same elongation everywhere. They bracket the campaign's measured 1.07-1.56.
ASPECT_LIMITS = (1.05, 1.60)
ASPECT_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "elongation", [QUALITY_COLOURS[3], QUALITY_COLOURS[2], QUALITY_COLOURS[1]]
)
# Feasible, but never photographed, so it has no elongation to rank. Distinct
# from FAILED, which has no pellets at all.
UNMEASURED = "#FFFFFF"


def aspect_norm():
    return mpl.colors.Normalize(vmin=ASPECT_LIMITS[0], vmax=ASPECT_LIMITS[1])


def aspect_colour(aspect_ratio, feasible=True):
    """Colour on the continuous elongation ramp.

    Failed runs stay grey and unmeasured ones stay hollow-white, because
    neither has an elongation: putting them on the ramp would invent a rank.
    """
    if not feasible:
        return FAILED
    if _missing_quality(aspect_ratio):
        return UNMEASURED
    return ASPECT_CMAP(aspect_norm()(float(aspect_ratio)))


def aspect_mappable():
    mappable = mpl.cm.ScalarMappable(norm=aspect_norm(), cmap=ASPECT_CMAP)
    mappable.set_array([])
    return mappable


def aspect_colourbar(fig, ax, label="median pellet aspect ratio\n(lower is rounder)",
                     **kwargs):
    bar = fig.colorbar(aspect_mappable(), ax=ax, **kwargs)
    bar.set_label(label, fontsize=7.5)
    bar.ax.tick_params(labelsize=7)
    return bar


def aspect_legend_handles(values=(1.10, 1.25, 1.40, 1.55), label="aspect ratio"):
    """The ramp as a few labelled swatches, for figures with no room for a bar.

    Several of these figures place their legend with hand-tuned geometry that a
    colourbar's stolen space would break. Sampling the same ramp keeps one
    visual language across the gallery.
    """
    return [plt.Line2D([], [], marker="o", ls="", markerfacecolor=aspect_colour(v),
                       markeredgecolor="k", markeredgewidth=0.4, markersize=7,
                       label=f"{label} {v:.2f}")
            for v in values]


def aspect_handles(include_unmeasured=True, include_failed=False):
    """Legend entries for the two states that are off the ramp."""
    handles = []
    if include_failed:
        handles.append(plt.Line2D([], [], marker="o", ls="", markerfacecolor=FAILED,
                                  markeredgecolor="k", markeredgewidth=0.4,
                                  markersize=7, label="failed to form"))
    if include_unmeasured:
        handles.append(plt.Line2D([], [], marker="o", ls="", markerfacecolor=UNMEASURED,
                                  markeredgecolor="k", markeredgewidth=0.4,
                                  markersize=7, label="never photographed"))
    return handles


# Text artists carrying data the reader needs, such as a count inside a marker.
# The clean manuscript build keeps these and strips every other in-axes text.
DATA_LABEL = "data-label"

# RSC artwork: 17.1 cm double column; a figure taller than the text block will
# not fit on the page with its caption.
DOUBLE_COLUMN = 17.1 / 2.54
MAX_HEIGHT = 9.0


def set_style():
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.4,
            "pdf.fonttype": 42,
        }
    )


# Photograph montages whose panel titles are the labels, not commentary.
KEEP_HEADINGS = {"segmentation_pipeline", "segmentation_examples"}
PANEL_LABEL = re.compile(r"^\s*\([a-z]\)(\s+experiment \d+)?")


def drop_headings(fig, name):
    """Remove figure and panel headings, keeping the panel letter.

    Returns the removed text, so a caller building captions can keep it.
    """
    if name in KEEP_HEADINGS:
        return []
    removed = []
    if fig._suptitle is not None and fig._suptitle.get_text().strip():
        removed.append(f"suptitle: {fig._suptitle.get_text()}")
        fig._suptitle.set_text("")
    for ax in fig.axes:
        for loc in ("left", "center", "right"):
            title = ax.get_title(loc=loc)
            if not title:
                continue
            label = PANEL_LABEL.match(title)
            kept = label.group(0).strip() if label else ""
            if kept != title.strip():
                removed.append(f"title: {title}")
            if label:
                ax.set_title(kept, loc="left", fontsize=9, fontweight="bold")
                if loc != "left":
                    ax.set_title("", loc=loc)
            else:
                ax.set_title("", loc=loc)
    return removed


def save(fig, name, directory=None, extra=None):
    """Save as vector PDF plus a 300 dpi PNG for drafts and slides.

    `directory` replaces the campaign gallery. `extra` writes the same files
    again, which is how a batch-owned figure also lands in batches/<nn>/figures.
    """
    drop_headings(fig, name)
    targets = [Path(directory) if directory is not None else cfg.FIGURES]
    if extra is not None:
        extras = extra if isinstance(extra, (list, tuple)) else [extra]
        targets.extend(Path(p) for p in extras)
    seen = set()
    for folder in targets:
        resolved = folder.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        folder.mkdir(parents=True, exist_ok=True)
        pdf = folder / f"{name}.pdf"
        png = folder / f"{name}.png"
        fig.savefig(pdf)
        fig.savefig(png)
        print(f"  figure -> {pdf.relative_to(cfg.ROOT)}, {png.relative_to(cfg.ROOT)}")
    plt.close(fig)


def batch_bands(ax, batches, alpha=0.06):
    """Faint alternating shading so batch boundaries are readable."""
    for b in sorted(set(batches)):
        if b % 2:
            continue
        lo = (b - 1) * cfg.BATCH_SIZE + 0.5
        ax.axvspan(lo, lo + cfg.BATCH_SIZE, color="k", alpha=alpha, lw=0)


# Where a point label is tried, in points, in order of preference: three rings
# outward from the marker, each covering right, left, above and below. Right and
# slightly below comes first, because that is where a label sits closest to its
# marker without covering it. The outer rings are far enough out to need a leader
# line, and exist because the middle of the size agreement panel holds five points
# within one marker width of each other, where the inner rings are all blocked.
# The list has to be long enough that it never runs out: a label placed at the
# last offset for want of anywhere else lands on top of the previous one.
LABEL_OFFSETS = [(7, -4), (7, 5), (-7, -4), (-7, 5), (0, 10), (0, -11),
                 (17, -12), (17, 10), (-17, -12), (-17, 10), (26, 0), (-26, 0),
                 (28, -20), (28, 18), (-28, -20), (-28, 18), (0, 23), (0, -24),
                 (40, -10), (40, 9), (-40, -10), (-40, 9), (0, 34), (0, -35)]


def place_labels(ax, items, avoid=(), fontsize=6.5, pad=1.5, radius=6.0):
    """Label scattered points, choosing an offset for each that stays clear.

    A crowded scatter cannot be labelled with one fixed offset. In the size
    agreement panel nineteen points share a narrow diagonal band, and three of the
    campaign's labels sat on top of each other there before the 2026 material was
    added. Each label is placed at the first offset in LABEL_OFFSETS whose box
    misses every marker and every label already placed, and gets a leader line
    where it ends up far enough out to be ambiguous without one.

    Works in display coordinates, so the packing reflects what the reader sees
    rather than the data range, and must therefore be called once the axis limits
    are final. Nothing here moves a point, so no placement can change a value.

    `items` is (x, y, text, colour); `avoid` is marker positions in data
    coordinates, which normally includes the labelled points themselves. Each is
    treated as a disc of `radius` points rather than as a position, because a label
    that merely grazes a marker it does not belong to is the worst outcome
    available: it reads as that marker's label.
    """
    renderer = ax.figure.canvas.get_renderer()
    markers = []
    for x, y in avoid:
        cx, cy = ax.transData.transform((x, y))
        markers.append((cx - radius, cy - radius, cx + radius, cy + radius))
    taken = []
    for x, y, text, colour in items:
        anchor = ax.transData.transform((x, y))
        width, height = _text_size(ax, text, fontsize, renderer)
        placed = None
        for dx, dy in LABEL_OFFSETS:
            left = anchor[0] + dx - (width if dx < 0 else 0)
            bottom = anchor[1] + dy - height / 2
            box = (left - pad, bottom - pad, left + width + pad, bottom + height + pad)
            if not any(_overlaps(box, m) for m in markers) \
                    and not any(_overlaps(box, t) for t in taken):
                placed = (dx, dy, box)
                break
        if placed is None:
            # Better to say so than to stack two labels and let a reader attribute
            # a number to the wrong point.
            raise SystemExit(f"no clear position for the label {text!r} among "
                             f"{len(LABEL_OFFSETS)} candidates; the panel needs more room or "
                             f"fewer labels")
        dx, dy, box = placed
        taken.append(box)
        far = abs(dx) > 8 or abs(dy) > 10
        ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=fontsize, color=colour, va="center",
                    ha="right" if dx < 0 else "left",
                    arrowprops=dict(arrowstyle="-", color="#CCCCCC", lw=0.5,
                                    shrinkA=0, shrinkB=1.5) if far else None)


def _text_size(ax, text, fontsize, renderer):
    probe = ax.text(0, 0, text, fontsize=fontsize, alpha=0)
    box = probe.get_window_extent(renderer=renderer)
    probe.remove()
    return box.width, box.height


def _overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
