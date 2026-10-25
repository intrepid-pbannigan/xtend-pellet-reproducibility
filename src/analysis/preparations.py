"""Later preparations of campaign formulations, as measured.

Both 2026 series are the same kind of object: a formulation made again, long after
the campaign, and photographed. Neither carries a visual score or a release
profile, so neither can join a statistic that validates the campaign, but both can
be measured on the same axes and read beside it.

Both are also photographed the same two ways in one session. Some frames spread
the pellets in a single layer on a moulded tray, which the separation rule can
resolve; others show the bulk sample heaped in a weighing boat, which it cannot,
and which exists as a record of the sample rather than as a measurement of it.
The confidence score tells them apart with room to spare -- across the twenty 2026
photographs the trusted ones run 0.51 to 0.96 and the rest 0.06 to 0.30 -- so
every descriptor here is recomputed over the trusted frames alone. Recomputed
rather than read from the summary, so that a plotted point and the particles drawn
underneath it are always the same pellets.
"""

import sys
from collections import namedtuple
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg

Preparation = namedtuple("Preparation", "particles summary frames quality")


def load(series):
    """One 2026 series, restricted to the photographs the pipeline trusts.

    Both series now carry absolute size. The September photographs had none of
    their own, and for a time this returned ratios for them alone; the operator's
    later ImageJ session supplied a measured scale for twelve of the fourteen, and
    the two it missed are the two the confidence score had already rejected, so
    every frame that reaches this function has a scale measured on it.

    That coincidence is worth not relying on, so the size columns are still
    computed from `size_um`, which the build leaves empty wherever no scale was
    measured. A frame without one contributes shape and no size rather than a
    size derived from a nominal scale.
    """
    particles = pd.read_csv(cfg.PARTICLES_CSV).query(
        "series == @series and source == 'segmentation'")
    quality = pd.read_csv(cfg.IMAGE_QUALITY_CSV).query("series == @series")
    trusted = quality[quality.trusted]
    particles = particles[particles.source_file.isin(trusted.source_file)]
    # By stem, not by filename: an ImageJ table and the photograph it was measured
    # on are the same frame under two extensions.
    frames = {Path(f).stem for f in trusted.source_file}
    summary = pd.read_csv(cfg.PARTICLE_SUMMARY_CSV).query(
        "series == @series and source == 'segmentation'").set_index("sample")

    by_sample = particles.groupby("sample")
    deciles = by_sample.size_um.quantile([0.1, 0.5, 0.9]).unstack()
    summary = summary.assign(
        median_aspect_ratio=by_sample.aspect_ratio.median(),
        median_circularity=by_sample.circularity.median(),
        median_solidity=by_sample.solidity.median(),
        d10_um=deciles[0.1], d50_um=deciles[0.5], d90_um=deciles[0.9],
        span=(deciles[0.9] - deciles[0.1]) / deciles[0.5],
        # The same width measured inside each photograph and then pooled. It needs
        # no scale, so it was the only width the September series could report
        # before its tables arrived, and it is kept as the check on the pooled span
        # above: the two disagree exactly when the frames of one sample were shot
        # at different working distances, which in this session they were.
        span_in_image=trusted.groupby("sample").span_in_image.median(),
        n_particles=by_sample.size(),
        n_images=by_sample.source_file.nunique(),
    ).reset_index().sort_values("sample")
    return Preparation(particles=particles, summary=summary, frames=frames, quality=quality)


def frame_medians(particles, columns=("aspect_ratio", "circularity", "solidity")):
    """One row per photograph, which is the unit these samples actually replicate.

    A sample here is two or three photographs of a few dozen pellets each. Treating
    the particles as the replicates would put the count in the hundreds and imply a
    precision the design cannot support, since pellets within one photograph are
    from one spread of one sample. The scatter between photographs is the error bar
    that means something.
    """
    return particles.groupby(["sample", "source_file"])[list(columns)].median().reset_index()
