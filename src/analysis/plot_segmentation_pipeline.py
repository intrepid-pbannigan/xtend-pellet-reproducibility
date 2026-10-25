"""Analysis F: the segmentation pipeline, step by step, on two campaign photographs.

The other segmentation figures show outcomes. This one shows the method, so a
reader can follow each operation in segment_pellets.py on real pixels:

    top row     a spherical sample where touching pellets have to be split
    bottom row  a dumbbell sample, under the fixed-distance watershed this
                replaced and under the concavity-depth rule kept, then the
                fitted ellipses the aspect ratio is read from

Every intermediate is computed by the production functions or with the same
config constants, so the figure cannot drift from what generated the data.
Both photographs are chosen by rule rather than by eye: the top one is the
trusted spherical-class photograph in which separation split the most touching
pellets, the bottom one the trusted dumbbell-class photograph on which the two
separation rules disagree most.

    python src/analysis/plot_segmentation_pipeline.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from matplotlib.patches import Ellipse
from PIL import Image
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import h_maxima
from skimage.segmentation import clear_border

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg
from analysis.plot_segmentation_examples import crop_to_particles, outline
from data_processing import segment_pellets as segment
from plotting import style

Image.MAX_IMAGE_PIXELS = None

ARTEFACT_COLOUR = (0.85, 0.2, 0.75)
MARKER_COLOUR = "#FF3B30"
MAX_PANEL_WIDTH_PX = 900


def read(path):
    with Image.open(cfg.ROOT / path) as img:
        return np.asarray(img.convert("RGB"), dtype=float) / 255.0


def candidates(quality, summary, scores):
    """Trusted campaign photographs of experiments carrying one of `scores`."""
    shape = summary[(summary.source == "segmentation") & (summary.series == cfg.CAMPAIGN_SERIES)]
    keep = set(shape[shape.particle_shape_score.isin(scores)].experiment_num)
    return quality[(quality.series == cfg.CAMPAIGN_SERIES) & quality.trusted
                   & quality.experiment_num.isin(keep)]


def pick_spherical(pool, manifest):
    """The photograph in which separation split the most touching pellets."""
    best, most = None, -1
    for row in pool.itertuples():
        source = manifest.loc[row.source_file]
        mask, _ = segment.build_mask(read(source.path), source.um_per_px)
        split = segment.separate(mask, source.um_per_px).max() - label(mask).max()
        if split > most:
            best, most = row.source_file, split
    return best


def pick_dumbbell(pool, manifest):
    """The photograph on which the two separation rules disagree most."""
    best, most = None, -1
    for row in pool.itertuples():
        source = manifest.loc[row.source_file]
        rgb = read(source.path)
        kept, _, _ = segment.segment_image(rgb, source.um_per_px)
        gap = len(segment.segment_image_baseline(rgb, source.um_per_px)) - len(kept)
        if gap > most:
            best, most = row.source_file, gap
    return best


def stages(rgb, um_per_px):
    """Every intermediate the top row draws, from the production constants."""
    with np.errstate(all="ignore"):
        hsv = rgb2hsv(rgb)
    value, saturation = hsv[..., 2], hsv[..., 1]
    threshold = threshold_otsu(value)
    bright = value > threshold
    artefact = bright & (saturation >= cfg.SEGMENT_SATURATION_MAX)
    mask, _ = segment.build_mask(rgb, um_per_px)
    distance = ndi.distance_transform_edt(mask) * um_per_px
    markers = label(h_maxima(distance / um_per_px, cfg.SEGMENT_H_DEPTH_UM / um_per_px))
    records, diagnostics, labels = segment.segment_image(rgb, um_per_px)
    return dict(value=value, threshold=threshold, bright=bright, artefact=artefact,
                mask=mask, distance=distance, markers=markers, records=records,
                diagnostics=diagnostics, labels=labels)


def baseline_labels(rgb, um_per_px):
    """The fixed-distance watershed, in the shape `outline` expects."""
    mask, _ = segment.build_mask(rgb, um_per_px)
    separated = segment.separate_fixed_distance(mask, um_per_px)
    cleared = clear_border(separated)
    records, _ = segment.measure_regions(cleared, um_per_px)
    return records, segment.Labels(separated, cleared)


def show(ax, image, records, title, **kwargs):
    panel = crop_to_particles(image, records)
    step = max(int(panel.shape[1] / MAX_PANEL_WIDTH_PX), 1)
    handle = ax.imshow(panel[::step, ::step], **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title.split(")")[0] + ")", fontsize=9, fontweight="bold", loc="left")
    return handle, step


def crop_offset(shape, records, aspect=1.5):
    """Top-left corner of the window `crop_to_particles` takes, for overlays."""
    probe = np.indices(shape[:2]).transpose(1, 2, 0)
    window = crop_to_particles(probe, records, aspect)
    return window[0, 0]


def spherical_row(axes, rgb, source, info):
    s = stages(rgb, source.um_per_px)
    records = s["records"]

    show(axes[0], rgb, records, f"(a) photograph\nexp {int(info.experiment_num)}, "
                                f"{source.um_per_px:.1f} µm px$^{{-1}}$")

    overlay = np.stack([s["bright"]] * 3, axis=-1).astype(float)
    overlay[s["artefact"]] = ARTEFACT_COLOUR
    show(axes[1], overlay, records,
         f"(b) Otsu on HSV value (t = {s['threshold']:.2f});\n"
         f"saturation ≥ {cfg.SEGMENT_SATURATION_MAX:.2f} rejected (magenta)")

    handle, step = show(axes[2], np.where(s["mask"], s["distance"], np.nan), records,
                        f"(c) distance transform; h-maxima\nmarkers at {cfg.SEGMENT_H_DEPTH_UM:.0f} µm depth",
                        cmap="viridis")
    axes[2].set_facecolor("black")
    top, left = crop_offset(s["mask"].shape, records)
    for region in regionprops(s["markers"]):
        row, col = region.centroid
        axes[2].plot((col - left) / step, (row - top) / step, "+", color=MARKER_COLOUR,
                     ms=5, mew=1.2)
    bar = plt.colorbar(handle, ax=axes[2], fraction=0.035, pad=0.02)
    # Above the bar: a side label falls under the next image.
    bar.ax.set_title("distance\n(µm)", fontsize=6.5, pad=4)
    bar.ax.tick_params(labelsize=6)

    rejected = sum(v for k, v in s["diagnostics"].items() if k.startswith("rejected_"))
    show(axes[3], outline(rgb, s["labels"], records), records,
         f"(d) watershed, border clearing, filters\n"
         f"{len(records)} measured, {rejected} rejected")


def dumbbell_row(axes, rgb, source, info):
    records, _, labels = segment.segment_image(rgb, source.um_per_px)
    old_records, old_labels = baseline_labels(rgb, source.um_per_px)

    show(axes[0], rgb, records, f"(e) photograph\nexp {int(info.experiment_num)}, "
                                f"dumbbell class")
    show(axes[1], outline(rgb, old_labels, records), records,
         f"(f) fixed-distance watershed\n({cfg.BASELINE_PEAK_DISTANCE_UM:.0f} µm): "
         f"{len(old_records)} particles")
    show(axes[2], outline(rgb, labels, records), records,
         f"(g) concavity-depth separation\n({cfg.SEGMENT_H_DEPTH_UM:.0f} µm): "
         f"{len(records)} particles")

    _, step = show(axes[3], rgb * 0.55, records,
                   "(h) moment-equivalent ellipses\nAR = major / minor axis")
    top, left = crop_offset(rgb.shape, records)
    kept = {r["label"] for r in records}
    ratios = []
    for region in regionprops(labels.cleared):
        if region.label not in kept:
            continue
        row, col = region.centroid
        ratio = region.axis_major_length / region.axis_minor_length
        ratios.append(ratio)
        colour = plt.cm.cool(np.clip((ratio - 1.0) / 1.0, 0, 1))
        axes[3].add_patch(Ellipse(
            ((col - left) / step, (row - top) / step),
            width=region.axis_major_length / step, height=region.axis_minor_length / step,
            angle=-np.degrees(region.orientation) + 90, fill=False, ec=colour, lw=1.1))
    scale = plt.cm.ScalarMappable(cmap="cool", norm=plt.Normalize(1.0, 2.0))
    bar = plt.colorbar(scale, ax=axes[3], fraction=0.035, pad=0.02)
    bar.set_label("aspect ratio", fontsize=6.5)
    bar.ax.tick_params(labelsize=6)


def main():
    style.set_style()
    manifest = pd.read_csv(cfg.IMAGE_MANIFEST_CSV).set_index("file")
    quality = pd.read_csv(cfg.IMAGE_QUALITY_CSV)
    summary = pd.read_csv(cfg.PARTICLE_SUMMARY_CSV)

    spherical = pick_spherical(candidates(quality, summary, cfg.SPHERICAL_SCORES), manifest)
    dumbbell = pick_dumbbell(candidates(quality, summary, cfg.DUMBBELL_SCORES), manifest)
    info = quality.set_index("source_file")

    fig, axes = plt.subplots(2, 4, figsize=(style.DOUBLE_COLUMN * 1.45, 4.3))
    spherical_row(axes[0], read(manifest.loc[spherical].path), manifest.loc[spherical],
                  info.loc[spherical])
    dumbbell_row(axes[1], read(manifest.loc[dumbbell].path), manifest.loc[dumbbell],
                 info.loc[dumbbell])

    handles = [
        plt.Line2D([], [], color=(0.0, 0.85, 0.45), lw=3, label="measured particle"),
        plt.Line2D([], [], color=(1.0, 0.25, 0.2), lw=3, label="rejected: size, sliver or solidity"),
        plt.Line2D([], [], color=(1.0, 0.75, 0.0), lw=3, label="rejected: touches frame edge"),
        plt.Line2D([], [], color=MARKER_COLOUR, marker="+", ls="", ms=7, label="watershed marker"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.01, right=0.97, top=0.91, bottom=0.08, wspace=0.12, hspace=0.38)
    style.save(fig, "segmentation_pipeline")
    print(f"spherical exemplar: {spherical}; dumbbell exemplar: {dumbbell}")


if __name__ == "__main__":
    main()
