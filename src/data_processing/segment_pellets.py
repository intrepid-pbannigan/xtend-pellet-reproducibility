"""Pellet segmentation: colour-based artefact rejection and concavity-depth separation.

Pure functions over arrays, no file access, so each step can be exercised on its
own and the fixed-distance watershed this replaces can be scored through exactly
the same measurement code.

Two departures from a plain Otsu-plus-watershed pipeline:

1. Artefacts are rejected by colour rather than by geometry. The pellets are
   neutral white; the pink paper at the top of the frame and the specular arcs on
   the black card are strongly coloured. A saturation ceiling removes them
   without any per-image region of interest, so one rule applies everywhere.
2. Markers come from the depth of a maximum rather than its distance from its
   neighbour, so deep necks between touching spheres split while shallow dumbbell
   waists stay intact. See the reasoning in config.SEGMENT_H_DEPTH_UM.
"""

import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import (
    binary_closing,
    disk,
    h_maxima,
    remove_small_holes,
    remove_small_objects,
)
from skimage.segmentation import clear_border, watershed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg

Labels = namedtuple("Labels", "separated cleared")


def build_mask(rgb, um_per_px):
    """Bright, colour-neutral objects, cleaned of speckle and interior holes.

    `rgb` is float in [0, 1]. Grayscale sources have zero saturation everywhere,
    so the colour rule passes silently and the shape filters carry the load.

    Also returns how much of the bright material the colour rule removed, which
    is the artefact load of the photograph and the quantity most likely to differ
    between imaging sessions.
    """
    # numpy 2.0 on the macOS Accelerate BLAS emits spurious matmul warnings
    with np.errstate(all="ignore"):
        hsv = rgb2hsv(rgb)
    value, saturation = hsv[..., 2], hsv[..., 1]

    bright = value > threshold_otsu(value)
    mask = bright & (saturation < cfg.SEGMENT_SATURATION_MAX)
    diagnostics = {
        "bright_frac": float(bright.mean()),
        "artefact_frac_of_bright": float(1 - mask.sum() / bright.sum()) if bright.any() else np.nan,
    }

    mask = binary_closing(mask, disk(cfg.SEGMENT_CLOSING_RADIUS_PX))
    mask = remove_small_holes(mask, area_threshold=_area_px(cfg.SEGMENT_FILL_HOLE_UM, um_per_px))
    mask = remove_small_objects(mask, min_size=_area_px(cfg.SEGMENT_MIN_DIAMETER_UM, um_per_px))
    return mask, diagnostics


def separate(mask, um_per_px, h_depth_um=None):
    """Split touching particles at deep necks, leaving shallow waists whole."""
    h_depth_um = cfg.SEGMENT_H_DEPTH_UM if h_depth_um is None else h_depth_um
    distance = ndi.distance_transform_edt(mask)
    if h_depth_um <= 0:
        return label(mask)
    markers = label(h_maxima(distance, h_depth_um / um_per_px))
    if markers.max() == 0:
        return label(mask)
    return watershed(-distance, markers, mask=mask)


def separate_fixed_distance(mask, um_per_px):
    """The fixed-distance watershed this replaces, kept for the contrast only."""
    from skimage.feature import peak_local_max

    distance = ndi.distance_transform_edt(mask)
    peaks = peak_local_max(
        distance,
        min_distance=max(int(cfg.BASELINE_PEAK_DISTANCE_UM / um_per_px), 1),
        labels=mask,
    )
    markers = np.zeros(distance.shape, dtype=int)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    if markers.max() == 0:
        return label(mask)
    return watershed(-distance, markers, mask=mask)


def measure_regions(labels, um_per_px):
    """One record per surviving particle, plus a tally of what was rejected.

    Size filters are stated in microns and converted here, so a single parameter
    set covers both resolution classes. Frame-touching regions are expected to
    have been removed already, since a partially imaged particle has no
    measurable extent.
    """
    rejected = {"undersize": 0, "oversize": 0, "sliver": 0, "low_solidity": 0, "degenerate": 0}
    records = []
    for region in regionprops(labels):
        if region.axis_minor_length <= 0 or region.perimeter <= 0:
            rejected["degenerate"] += 1
            continue
        major_um = region.axis_major_length * um_per_px
        if major_um < cfg.SEGMENT_MIN_DIAMETER_UM:
            rejected["undersize"] += 1
            continue
        if major_um > cfg.SEGMENT_MAX_DIAMETER_UM:
            rejected["oversize"] += 1
            continue
        # The floor applies to both axes. A scratch on the card or a line on the
        # boat rim is a few pixels wide but long enough to pass the major-axis
        # test, and being straight it is convex, so the solidity test passes it
        # too. Filtering on the minor axis is symmetric with the major one, so it
        # cannot bias elongation: it removes shapes too thin to be a pellet at
        # all, not shapes that are merely elongated.
        if region.axis_minor_length * um_per_px < cfg.SEGMENT_MIN_DIAMETER_UM:
            rejected["sliver"] += 1
            continue
        if region.solidity < cfg.SEGMENT_MIN_SOLIDITY:
            rejected["low_solidity"] += 1
            continue
        records.append(
            {
                "label": int(region.label),
                "centroid_row": float(region.centroid[0]),
                "centroid_col": float(region.centroid[1]),
                "area_px": float(region.area),
                "axis_major_px": float(region.axis_major_length),
                "axis_minor_px": float(region.axis_minor_length),
                "equivalent_diameter_px": float(region.equivalent_diameter_area),
                "aspect_ratio": float(region.axis_major_length / region.axis_minor_length),
                "circularity": float(min(4 * np.pi * region.area / region.perimeter**2, 1.0)),
                "solidity": float(region.solidity),
            }
        )
    diagnostics = {f"rejected_{reason}": count for reason, count in rejected.items()}
    diagnostics["n_accepted"] = len(records)
    return records, diagnostics


def segment_image(rgb, um_per_px):
    """Full pass over one photograph.

    Returns the particle records, per-image diagnostics, and both label images,
    the last so the outcome can be drawn over the photograph with regions the
    filters rejected distinguishable from regions touching the frame edge.
    """
    mask, diagnostics = build_mask(rgb, um_per_px)
    separated = separate(mask, um_per_px)
    # Particles running off the edge of the frame have no measurable extent, and
    # this also removes specular streaks that cross the whole image.
    cleared = clear_border(separated)
    diagnostics["rejected_border"] = int(separated.max() - len(np.unique(cleared)) + 1)

    records, measured = measure_regions(cleared, um_per_px)
    diagnostics.update(measured)

    mask_area = float((cleared > 0).sum())
    accepted_area = float(sum(r["area_px"] for r in records))
    # How much of the separated material became a measured particle. A low value
    # means the photograph produced many bright shapes the filters could not
    # accept, which is the clearest single warning that an image is difficult.
    diagnostics["area_accounted_frac"] = accepted_area / mask_area if mask_area else np.nan
    diagnostics["median_solidity"] = (
        float(np.median([r["solidity"] for r in records])) if records else np.nan
    )
    return records, diagnostics, Labels(separated, cleared)


def segment_image_baseline(rgb, um_per_px):
    """Same measurement code, fixed-distance watershed instead of concavity depth."""
    mask, _ = build_mask(rgb, um_per_px)
    labels = separate_fixed_distance(mask, um_per_px)
    records, _ = measure_regions(clear_border(labels), um_per_px)
    return records


def _area_px(diameter_um, um_per_px):
    return max(int(np.pi * (diameter_um / 2 / um_per_px) ** 2), 1)
