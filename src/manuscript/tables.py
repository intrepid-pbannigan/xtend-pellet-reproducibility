"""Manuscript Tables 1, S1, S2 and S3 as CSV.

Table 2 is written by `retrospective/ablation.py`, because it needs the
cross-validated fits.

    outputs/tables/table1_design_space.csv
    outputs/tables/tableS1_excipient_classes.csv
    outputs/tables/tableS2_chimera_hierarchy.csv
    outputs/tables/tableS3_segmentation_parameters.csv

    python src/manuscript/tables.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg  # noqa: E402
import objectives  # noqa: E402
from analysis.plot_design_space import design_table  # noqa: E402

EXCIPIENT_COLUMNS = ["excipient", "product", "supplier", "chemistry", "functional_class",
                     "water_interaction", "ionic_character", "hlb"]

SEGMENTATION = (
    ("SEGMENT_SATURATION_MAX", "HSV saturation ceiling for pellet pixels", "-"),
    ("SEGMENT_H_DEPTH_UM", "h-maximum depth separating touching granules", "um"),
    ("SEGMENT_CLOSING_RADIUS_PX", "morphological closing radius", "px"),
    ("SEGMENT_FILL_HOLE_UM", "largest internal hole filled", "um"),
    ("SEGMENT_MIN_DIAMETER_UM", "smallest equivalent diameter kept", "um"),
    ("SEGMENT_MAX_DIAMETER_UM", "largest equivalent diameter kept", "um"),
    ("SEGMENT_MIN_SOLIDITY", "smallest solidity kept", "-"),
    ("CONFIDENCE_FLOOR", "image-quality floor for a trusted photograph", "-"),
    ("BASELINE_PEAK_DISTANCE_UM", "fixed-distance watershed used only as the comparison", "um"),
)


def main():
    cfg.TABLES.mkdir(parents=True, exist_ok=True)
    design_table(pd.read_csv(cfg.CAMPAIGN_CSV)).to_csv(
        cfg.TABLES / "table1_design_space.csv", index=False)

    excipients = pd.read_csv(cfg.EXCIPIENT_DESCRIPTORS_CSV)
    excipients[EXCIPIENT_COLUMNS].to_csv(cfg.TABLES / "tableS1_excipient_classes.csv",
                                         index=False)

    pd.DataFrame([dict(rank=i + 1, objective=name, goal=goal, threshold=tol,
                       threshold_units=objectives.UNITS[name] if absolute else "fraction of range")
                  for i, (name, goal, tol, absolute) in enumerate(objectives.HIERARCHY)]
                 ).to_csv(cfg.TABLES / "tableS2_chimera_hierarchy.csv", index=False)

    pd.DataFrame([dict(parameter=name, description=text, value=getattr(cfg, name), units=units)
                  for name, text, units in SEGMENTATION]
                 ).to_csv(cfg.TABLES / "tableS3_segmentation_parameters.csv", index=False)

    for name in ("table1_design_space", "tableS1_excipient_classes",
                 "tableS2_chimera_hierarchy", "tableS3_segmentation_parameters"):
        print(f"{name}.csv")


if __name__ == "__main__":
    main()
