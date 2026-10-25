"""Analysis F: the segmentation drawn over the photographs it was run on.

The best and worst photographs, each above the size distribution it produced.
Images are ranked by the per-image confidence score, which never sees the manual
tables, so the selection cannot be accused of picking the cases that happen to
agree with them.

The point of pairing each image with its distribution is that the two failure
modes look different. An image that segments badly shows it in the outlines; an
image that is simply unrepresentative of its sample shows it as a curve sitting
away from its siblings.

    python src/analysis/plot_segmentation_examples.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.morphology import binary_dilation, disk
from skimage.segmentation import find_boundaries

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg
from data_processing import segment_pellets as segment
from plotting import style

Image.MAX_IMAGE_PIXELS = None

ACCEPTED_COLOUR = (0.0, 0.85, 0.45)
FILTERED_COLOUR = (1.0, 0.25, 0.2)
BORDER_COLOUR = (1.0, 0.75, 0.0)
SIBLING_COLOUR = "#9E9E9E"


def outline(rgb, labels, records):
    """Draw the segmentation over the photograph.

    Green is a measured particle, red a shape the filters rejected, amber a
    region touching the frame edge. Boundaries come from the label image rather
    than the binary mask, so the line where two touching pellets were separated
    is visible and the separation rule can be judged directly.
    """
    accepted = np.isin(labels.cleared, np.array([r["label"] for r in records], dtype=int))
    overlay = rgb.copy()
    # Thick enough to survive the downsampling that fits the frame on a page.
    pen = disk(max(2, rgb.shape[1] // 600))
    for layer, colour in [
        (np.where((labels.separated > 0) & (labels.cleared == 0), labels.separated, 0), BORDER_COLOUR),
        (np.where((labels.cleared > 0) & ~accepted, labels.cleared, 0), FILTERED_COLOUR),
        (np.where(accepted, labels.cleared, 0), ACCEPTED_COLOUR),
    ]:
        if layer.max() == 0:
            continue
        overlay[binary_dilation(find_boundaries(layer, mode="thick"), pen)] = colour
    return overlay


def crop_to_particles(overlay, records, aspect=1.5):
    """Frame the pellet cluster rather than the empty card around it."""
    if not records:
        return overlay
    rows = np.array([r["centroid_row"] for r in records])
    cols = np.array([r["centroid_col"] for r in records])
    pad = 1.2 * np.median([r["axis_major_px"] for r in records])
    height, width = overlay.shape[:2]

    top, bottom = max(rows.min() - pad, 0), min(rows.max() + pad, height)
    left, right = max(cols.min() - pad, 0), min(cols.max() + pad, width)
    # Grow the short side so every panel has the same shape.
    if (right - left) / (bottom - top) < aspect:
        want = aspect * (bottom - top)
        centre = (left + right) / 2
        left, right = max(centre - want / 2, 0), min(centre + want / 2, width)
    else:
        want = (right - left) / aspect
        centre = (top + bottom) / 2
        top, bottom = max(centre - want / 2, 0), min(centre + want / 2, height)
    return overlay[int(top):int(bottom), int(left):int(right)]


def ecdf(values):
    values = np.sort(np.asarray(values, dtype=float))
    return values, np.arange(1, len(values) + 1) / len(values)


def distribution_panel(ax, is_leftmost, image, experiment, particles, manual):
    """One photograph's size distribution against its own siblings.

    The comparison that matters is not against the manual table, which covers a
    different subset of particles, but against the other photographs of the same
    sample. Those are independent draws from the same pellets, so a curve sitting
    away from its siblings is the visible form of the between-image scatter.
    """
    sample = particles[particles.experiment_num == experiment]
    for other, group in sample.groupby("source_file"):
        if other == image:
            continue
        ax.step(*ecdf(group.size_um_display), where="post", color=SIBLING_COLOUR, lw=1.0)

    hand = manual[manual.experiment_num == experiment]
    if len(hand):
        ax.step(*ecdf(hand.size_um), where="post", color="k", ls="--", lw=1.2)

    this = sample[sample.source_file == image]
    ax.step(*ecdf(this.size_um_display), where="post", color=style.FEASIBLE, lw=2.0)
    median = float(this.size_um_display.median())
    ax.axvline(median, color=style.FEASIBLE, lw=0.9, ls=":")

    siblings = sample.groupby("source_file").size_um_display.median()
    label = f"median {median:.0f} um"
    if len(siblings) > 1:
        offset = 100 * (median - siblings.median()) / siblings.median()
        label += f"  ({offset:+.0f}% vs siblings)"
    ax.set_title(label, fontsize=8)
    ax.set_xlim(600, 3700)
    ax.set_xticks([1000, 2000, 3000])
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("major axis (um)", fontsize=8)
    if is_leftmost:
        ax.set_ylabel("cumulative\nfraction", fontsize=8)
    else:
        ax.set_yticklabels([])


def main(n=4):
    style.set_style()
    # The campaign series only: this figure shows the range of photograph quality
    # the campaign's own conclusions rest on, and the heaped August 2026 remakes
    # would take both of the worst slots for a protocol nothing here analyses.
    particles = pd.read_csv(cfg.PARTICLES_CSV).query("series == @cfg.CAMPAIGN_SERIES")
    quality = pd.read_csv(cfg.IMAGE_QUALITY_CSV).query("series == @cfg.CAMPAIGN_SERIES")
    manifest = pd.read_csv(cfg.IMAGE_MANIFEST_CSV).set_index("file")

    segmented = particles[particles.source == "segmentation"]
    manual = particles[(particles.source == "manual_imagej") & ~particles.provisional]

    # One image per experiment: the worst four outright come from two samples and
    # would show the same failure twice over.
    ranked = quality.sort_values("confidence", ascending=False)
    best = ranked.drop_duplicates("experiment_num").head(n)
    worst = ranked.sort_values("confidence").drop_duplicates("experiment_num").head(n).iloc[::-1]

    fig = plt.figure(figsize=(4.0 * n, 10.4))
    grid = fig.add_gridspec(4, n, height_ratios=[2.5, 1.5, 2.5, 1.5], hspace=0.55, wspace=0.12)

    for block, (chosen, banner) in enumerate([(best, "highest confidence"),
                                              (worst, "lowest confidence")]):
        image_row, dist_row = 2 * block, 2 * block + 1
        dist_left = fig.add_subplot(grid[dist_row, 0])
        for column, row in enumerate(chosen.itertuples()):
            source = manifest.loc[row.source_file]
            with Image.open(cfg.ROOT / source.path) as img:
                rgb = np.asarray(img.convert("RGB"), dtype=float) / 255.0
            records, _, labels = segment.segment_image(rgb, source.um_per_px)

            ax = fig.add_subplot(grid[image_row, column])
            panel = crop_to_particles(outline(rgb, labels, records), records)
            # Downsample for the page; the pen width above survives it.
            step = max(int(panel.shape[1] / 1100), 1)
            ax.imshow(panel[::step, ::step])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            provisional = ("" if source.calibration_source != "operator_line_scale"
                           else "  ·  scale from the operator's own line")
            ax.set_title(
                f"exp {int(row.experiment_num)}  ·  {row.source_file.strip()}\n"
                f"confidence {row.confidence:.2f}  ·  {int(row.n_accepted)} particles{provisional}",
                fontsize=8.5,
            )
            if column == 0:
                ax.set_ylabel(banner, fontsize=10, labelpad=8)

            dist = dist_left if column == 0 else fig.add_subplot(grid[dist_row, column],
                                                                sharey=dist_left)
            distribution_panel(dist, column == 0, row.source_file, row.experiment_num,
                               segmented, manual)

    handles = [
        plt.Line2D([], [], color=ACCEPTED_COLOUR, lw=3, label="measured particle"),
        plt.Line2D([], [], color=FILTERED_COLOUR, lw=3, label="rejected: size or solidity"),
        plt.Line2D([], [], color=BORDER_COLOUR, lw=3, label="rejected: touches the frame edge"),
        plt.Line2D([], [], color=style.FEASIBLE, lw=2, label="this photograph"),
        plt.Line2D([], [], color=SIBLING_COLOUR, lw=1.2, label="other photographs of the same sample"),
        plt.Line2D([], [], color="k", ls="--", lw=1.2, label="manual ImageJ table"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(
        "Segmentation and the size distribution it produced, on the best and worst photographs\n"
        "ranked without reference to the manual tables",
        y=0.995, fontsize=10.5,
    )
    # Margins set directly: the image axes have a fixed aspect, which tight_layout
    # cannot reconcile with the shared distribution axes below them.
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.08)
    style.save(fig, "segmentation_examples")

    print(f"best:  {', '.join(f'exp {int(r.experiment_num)} ({r.confidence:.2f})' for r in best.itertuples())}")
    print(f"worst: {', '.join(f'exp {int(r.experiment_num)} ({r.confidence:.2f})' for r in worst.itertuples())}")


if __name__ == "__main__":
    main()
