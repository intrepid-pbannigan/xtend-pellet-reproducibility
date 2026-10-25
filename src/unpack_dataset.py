"""Unpack the dataset workbook into the tables the analysis reads.

    python src/unpack_dataset.py
"""

from pathlib import Path

import pandas as pd

import config as cfg

WORKBOOK = cfg.ROOT / "data" / "ILX006_pellet_campaign_dataset.xlsx"

SHEETS = {
    "experiments": cfg.CAMPAIGN_CSV,
    "release_capsules": cfg.RELEASE_REPLICATES_CSV,
    "release_profiles": cfg.DISSOLUTION_CSV,
    "granules": cfg.PARTICLES_CSV,
    "morphology": cfg.PARTICLE_POOLED_CSV,
    "morphology_by_series": cfg.PARTICLE_SUMMARY_CSV,
    "photographs": cfg.IMAGE_MANIFEST_CSV,
    "image_quality": cfg.IMAGE_QUALITY_CSV,
    "excipients": cfg.EXCIPIENT_DESCRIPTORS_CSV,
    "fluids": cfg.FLUID_DESCRIPTORS_CSV,
}


def main():
    sheets = pd.read_excel(WORKBOOK, sheet_name=list(SHEETS))
    for name, path in SHEETS.items():
        frame = sheets[name]
        if name == "experiments":
            frame = frame.drop(columns=["cntrl_polymer_0_name", "cntrl_polymer_1_name"])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    print(f"  {len(SHEETS)} tables unpacked from {WORKBOOK.relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
