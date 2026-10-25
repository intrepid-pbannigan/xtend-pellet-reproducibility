"""Central configuration for the X-TEND pellet campaign analysis.

Every path, constant and variable role used by the analysis scripts lives here,
so the whole set of modelling decisions can be read in one file.
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

RAW_WORKBOOK = ROOT / "data" / "raw" / "rawdata_ilx006_capsules_manual_feb_2024.xlsx"
RAW_IMAGING = ROOT / "data" / "raw" / "imaging"
RAW_IMAGING_REMAKE = ROOT / "data" / "raw" / "New_images_August2026"
# Batches 12 and 13 (2026) have their own folders under data/raw/batches/<nn>/.
# Historical 2024 photographs stay in RAW_IMAGING; the 2026 remakes stay in
# RAW_IMAGING_REMAKE.
RAW_BATCHES = ROOT / "data" / "raw" / "batches"


def batch_raw(n):
    """Photographs and dissolution workbooks for one batch."""
    return RAW_BATCHES / f"{int(n):02d}"


RAW_IMAGING_BATCH12 = batch_raw(12) / "imaging"
RAW_IMAGING_BATCH13 = batch_raw(13) / "imaging"

# The dissolution working file behind the summary workbook, which is where the
# individual capsules live. RAW_WORKBOOK carries one profile per experiment, the
# mean over its replicates, and that is all the campaign analysis needed until
# the 2026 runs had to be pooled with it -- a mean cannot be pooled with two new
# capsules without knowing how many it already averages.
RAW_RELEASE_CAMPAIGN = ROOT / "data" / "raw" / "ILX006_release_capsules_2024.xlsx"

# The 2026 repeats: two capsules each of the four formulations that were made
# again, dissolved alongside the material the September photographs were taken
# of. Same preparation as those photographs, so a formulation's release and its
# morphology now come from one batch.
RAW_RELEASE_REPEAT = (ROOT / "data" / "raw" / "Repeated_release_experiments"
                      / "Dissolution_Summaries_F4-F44_with_Historical_Comparison.xlsx")

# Re-assay of experiment 27, 11 September 2026. The 2024 profile peaks at 1.06
# and drops 24 points between 180 and 240 minutes, which is not a physical
# cumulative release. These four HPLC capsules replace that curve in the
# official profile; the original two stay in release_replicates.csv as the
# campaign series so the two assays can still be compared.
RAW_RELEASE_F27 = (ROOT / "data" / "raw" / "Repeated_release_experiments"
                   / "20260911_F27_dissolution..xlsx")
RELEASE_SUPERSEDE_SERIES = {27: "campaign"}

# First dissolution of experiments 45-48. Same instrument-summary layout as the
# F4-F44 repeat workbook, two capsules each.
RAW_RELEASE_BATCH12 = (batch_raw(12) / "dissolution"
                       / "Dissolution_Summaries_F45-F48_with_Chart_15092026.xlsx")

# First dissolution of experiments 49-52. Same instrument-summary layout as
# the Batch 12 workbook, two capsules each. The right-hand column still
# carries the Batch 12 template labels F47/F48; those blocks are F51/F52.
# The chart sheet treats them as 51/52, which is what the block map uses.
RAW_RELEASE_BATCH13 = (batch_raw(13) / "dissolution"
                       / "Dissolution_Summaries_F49-F52_with_Chart_18092026.xlsx")
# Second dissolution of the same Batch 13 pellets, 23 September 2026. Same
# instrument-summary layout, plus two extra F50 capsules (n3, n4) that the
# Masses sheet lists and the chart mean omits. Official profiles pool both
# assays; neither supersedes the other.
RAW_RELEASE_BATCH13_R2 = (batch_raw(13) / "dissolution"
                          / "_Dissolution_Summaries_F49-F52_R2_with_Chart_23092026.xlsx")
PROCESSED = ROOT / "data" / "processed"
REFERENCE = ROOT / "data" / "reference"
OUTPUTS = ROOT / "outputs"
# Annotated working figures and every analysis table.
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
FIGURE_VARIANTS = FIGURES / "variants"
# Submission artwork, numbered as in the manuscript.
MANUSCRIPT_FIGURES = OUTPUTS / "manuscript"
EXCIPIENT_DESCRIPTORS_CSV = REFERENCE / "excipient_descriptors.csv"
FLUID_DESCRIPTORS_CSV = REFERENCE / "fluid_descriptors.csv"

CAMPAIGN_CSV = PROCESSED / "campaign.csv"
DISSOLUTION_CSV = PROCESSED / "dissolution_long.csv"
RELEASE_REPLICATES_CSV = PROCESSED / "release_replicates.csv"
DICTIONARY_CSV = PROCESSED / "data_dictionary.csv"
PARTICLES_CSV = PROCESSED / "particle_measurements.csv"
PARTICLE_SUMMARY_CSV = PROCESSED / "particle_summary.csv"
PARTICLE_POOLED_CSV = PROCESSED / "particle_pooled.csv"
CALIBRATION_CSV = PROCESSED / "particle_calibration.csv"
IMAGE_QUALITY_CSV = PROCESSED / "particle_image_quality.csv"
IMAGE_MANIFEST_CSV = PROCESSED / "particle_image_manifest.csv"
FAILURE_RECORD_CSV = PROCESSED / "repeated_failure_record.csv"
SCALE_SENSITIVITY_CSV = PROCESSED / "repeat_scale_sensitivity.csv"

# The campaign sheet carried forward to batches 12 and 13: the source of the
# formulation and process settings for experiments 45-52.
MASTER_EDIT_XLSX = ROOT / "data" / "raw" / "ILX006_campaign_sheet_batches_1_13.xlsx"

RANDOM_SEED = 20240201

# --- experimental design ---------------------------------------------------

# The 2024 campaign ran 44 experiments in 11 batches of four. Batches 12 and
# 13 added 45-52, so the live table has 52 numbers and 13 batches; 37 is still
# merged into 35, which is why the file holds 51 rows.
N_CAMPAIGN_EXPERIMENTS = 44
N_CAMPAIGN_BATCHES = 11
N_EXPERIMENTS = 52
N_BATCHES = 13
BATCH_SIZE = 4
TOTAL_BATCH_MASS_MG = 500.0

DISSOLUTION_TIMES_MIN = np.array([0, 15, 30, 45, 60, 120, 180, 240], dtype=float)

# The workbook records frac_release_0m as 0 for every experiment, but the SOP
# takes its first sample at 15 min. Time zero is an assumed structural zero.
ASSUMED_ZERO_TIME_MIN = 0.0

# Zero-order release completing at 240 min. The target_dissolution sheet gives
# 0/6/12.5/19/25/50/75/100 %, which is this profile rounded; the exact form is
# used so the objective is reproducible from a single stated rule.
TARGET_COMPLETION_MIN = 240.0
TARGET_PROFILE = DISSOLUTION_TIMES_MIN / TARGET_COMPLETION_MIN

# --- the duplicated design point --------------------------------------------
#
# Experiments 35 and 37 are the same experiment run twice. Their twelve
# formulation and process variables agree exactly, which no other pair in the
# campaign does, and the workbook gives them the same recorded size and the same
# shape score; only the dissolution differs, because it was measured twice.
#
# They are merged, and 37's identity disappears: its two capsules become
# replicates of 35, its five photographs become photographs of 35, and batch 10
# is left with three distinct formulations rather than four. Merging rather than
# dropping is what keeps the second measurement, and the second measurement is
# the campaign's only direct read on how much of the release spread is the assay
# rather than the formulation.
#
# Written as a mapping so that a later duplicate can be added without a new code
# path, and so that every script can ask the one question that matters -- what is
# this experiment's canonical number -- through `canonical_experiment`.
DUPLICATE_EXPERIMENTS = {37: 35}


def canonical_experiment(numbers):
    """Fold duplicated design points onto the number they are recorded under."""
    return numbers.replace(DUPLICATE_EXPERIMENTS)


# --- release normalisation ---------------------------------------------------
#
# Every profile is a capsule's released mass divided by the largest mass that
# capsule reached between 15 and 240 minutes, and an experiment's profile is the
# mean of its capsules. That is the campaign's own convention and what the target
# profile is written against: the target ends at 1.0 at 240 minutes, so a profile
# on any other denominator is being scored against the wrong curve.
#
# It was not applied consistently. Two of the 22 measured experiments divide by
# the nominal drug load of the design instead of by the capsule's own maximum --
# all four capsules of experiment 43, by 150 mg, and two of the four of
# experiment 44, by 100 mg -- and experiment 27 is on a third footing again,
# since its recorded profile peaks at 1.06 and a mean of max-normalised capsules
# cannot exceed 1. Those three are exactly where rebuilding from the individual
# capsules disagrees with the summary workbook; for the other 19 it reproduces it
# to the rounding in the summary or exactly.
#
# So the profiles are rebuilt from the capsules rather than read from the
# summary. This matters most for 43 and 44, which are two of the four
# formulations repeated in 2026: comparing a 2026 capsule against a nominal-load
# profile would have measured the change of denominator and called it a change in
# the material.
RELEASE_NORMALISATION = "per-capsule maximum over 15-240 min"

# In the working file's mass block, the 240 minute cell of 40 of the 48 capsules
# repeats the 30 minute cell exactly: a formula filled across one column too few.
# The concentration block beside it is unaffected, so the mass is recomputed from
# it, through the ratio the same row's other six timepoints agree on. That ratio
# is a dilution and a media volume and is constant within a row, but not between
# rows -- experiments 43 and 44 were run at a different dilution -- so it is read
# per row rather than assumed.
#
# The summary workbook does not carry the error, which is why campaign.csv has
# been right about 240 minutes all along and why this is a repair to the working
# file rather than a correction to any published number.
RELEASE_240_COPY_BUG = True

# Where the 2026 capsules sit in their workbook: the sheet is four pairs of
# blocks, two columns of them, each block a label row, a header row and seven
# timepoints. Declared because a block's position cannot be inferred from its
# label alone -- the sheet holds two blocks headed `F4` and two headed `F43`, and
# which is capsule 1 is a matter of which comes first down the column.
RELEASE_REPEAT_SHEET = "Dissolution "
RELEASE_REPEAT_BLOCKS = (
    # (label row, first column, formulation, capsule)
    (1, 0, 4, 1), (11, 0, 4, 2), (23, 0, 5, 1), (33, 0, 5, 2),
    (2, 10, 43, 1), (12, 10, 43, 2), (24, 10, 44, 1), (34, 10, 44, 2),
)

# Same sheet name and block geometry as the repeat workbook, F45/F46 in the
# left column where F4/F5 sat and F47/F48 on the right where F43/F44 sat.
RELEASE_BATCH12_SHEET = "Dissolution "
RELEASE_BATCH12_BLOCKS = (
    (1, 0, 45, 1), (11, 0, 45, 2), (23, 0, 46, 1), (33, 0, 46, 2),
    (2, 10, 47, 1), (12, 10, 47, 2), (24, 10, 48, 1), (34, 10, 48, 2),
)

# Same geometry. Optional fifth entry is the label written on the sheet
# when it is not F{experiment}: the right column was never renamed from
# the F45-F48 template.
RELEASE_BATCH13_SHEET = "Dissolution "
RELEASE_BATCH13_BLOCKS = (
    (1, 0, 49, 1), (11, 0, 49, 2), (23, 0, 50, 1), (33, 0, 50, 2),
    (2, 10, 51, 1, "F47"), (12, 10, 51, 2, "F47"),
    (24, 10, 52, 1, "F48"), (34, 10, 52, 2, "F48"),
)

# Same F47/F48 aliases on the right. F50 has four capsules: the first two sit
# in the usual left-column slots, n3/n4 continue below the F50 n2 block.
RELEASE_BATCH13_R2_SHEET = "Dissolution "
RELEASE_BATCH13_R2_BLOCKS = (
    (1, 0, 49, 1), (11, 0, 49, 2), (23, 0, 50, 1), (33, 0, 50, 2),
    (44, 0, 50, 3), (54, 0, 50, 4),
    (2, 10, 51, 1, "F47"), (12, 10, 51, 2, "F47"),
    (24, 10, 52, 1, "F48"), (34, 10, 52, 2, "F48"),
)

# The campaign working file's `Release_corrected` sheet. Header at row 24
# counting from zero, capsules below it, and the blocks of interest are the
# concentration and the released mass.
RELEASE_CAMPAIGN_SHEET = "Release_corrected"
RELEASE_CAMPAIGN_HEADER_ROW = 24
RELEASE_CAMPAIGN_COLS = {
    "experiment": 0, "replicate": 1, "drug_mass": 4,
    "concentration": range(13, 20), "mass": range(21, 28),
}

# Which preparation the 2026 dissolution capsules came from. The same batches the
# September photographs were taken of, so each of these four formulations now has
# release and morphology measured on one piece of material rather than on two
# that happen to share a recipe.
#
# The extraction sheet dates that session 2026-01-09. The September photographs
# of formulations 4 and 5 are timestamped 1 September 2026, and 01/09/2026 read
# day-first is that date, so the workbook date is almost certainly the same day
# entered in the other order. Nothing downstream uses it, and it is recorded here
# only so the reading is not silently assumed.
RELEASE_REPEAT_PREPARATION = {4: 3, 5: 3, 43: 2, 44: 2, 27: 1}

# --- the size unit correction ----------------------------------------------
#
# The recorded `_size` column is mean(ImageJ Length) * 1000, which assumes the
# ImageJ calibration unit was millimetres. 28 of the 44 TIFFs record unit=cm in
# their ImageJ header (roughly 590 px/cm on the 3028x2018 images, 320 px/cm on the
# 1440x960 ones); the remaining 16 record unit=unit, meaning the operator set a
# scale without naming it, and fall in the same numerical band. Independently, the
# CSV Area column equals Length times exactly one pixel width, which recovers the
# operator's own scale and agrees with the TIFF tags to 1-3%. Recorded values are
# therefore a factor of 10 low.
SIZE_UNIT_CORRECTION = 10.0
CM_TO_UM = 1e4

# --- particle segmentation -------------------------------------------------
#
# One parameter set for every photograph. The only branch in the pipeline is
# whether an image carries an ImageJ calibration, and that gates the micron
# columns only, never the shape descriptors.

RAW_IMAGING_JPEG = RAW_IMAGING / "Particle Size Analysis Pictures "  # trailing space is real

# Pellets are neutral white (HSV saturation 0.01-0.08). The pink paper at the top
# of the frame and the blue-purple specular arcs on the black card are strongly
# coloured, so a saturation ceiling removes them without any hand-drawn region of
# interest, and therefore without a per-image judgement call.
SEGMENT_SATURATION_MAX = 0.15

# Depth of the neck at which touching particles are separated, stated in microns
# so it transfers unchanged between the 14-17 and 30-33 um/px resolution classes.
# Two touching spheres meet at a saddle near zero, so the shallower maximum
# stands about a lobe radius above it and the pair splits; a dumbbell waist sits
# near 0.6 of the lobe radius, so its maximum is far shallower and the particle
# survives. A fixed-distance watershed cannot tell those cases apart and cuts
# both, which is what biased the earlier version against dumbbells.
SEGMENT_H_DEPTH_UM = 100.0
SEGMENT_CLOSING_RADIUS_PX = 3
SEGMENT_FILL_HOLE_UM = 200.0
SEGMENT_MIN_DIAMETER_UM = 400.0
SEGMENT_MAX_DIAMETER_UM = 4000.0
SEGMENT_MIN_SOLIDITY = 0.80

# The fixed-distance watershed this replaces, kept only so the validation can
# report the contrast that motivated the change.
BASELINE_PEAK_DISTANCE_UM = 500.0

# How many of an ImageJ calibration unit make a centimetre. Every scale is
# converted through this table rather than assumed to be in centimetres, because
# the two imaging series disagree: the 2024 files say cm, the 2026 remakes say mm,
# and reading one as the other is exactly the factor of ten that
# SIZE_UNIT_CORRECTION exists to undo. `unit` means the operator set a scale
# without naming it, and is read as centimetres only because those files' pixel
# scales sit inside the band of the files that do say centimetres.
IMAGEJ_UNITS_PER_CM = {"cm": 1.0, "mm": 10.0, "um": 1e4, "unit": 1.0}
PX_PER_CM_RANGE = (250.0, 800.0)

# The operator's own pixel scale, recovered from a table's Area/Length, is checked
# against the photograph's tag. Disagreement beyond this is not rounding: it means
# the table was measured before the scale was set, and its lengths are in a stale
# unit that has to be converted. Set above the 1-3% the two agree to when nothing
# is wrong, and far below the 11% seen when something is.
OPERATOR_SCALE_TOLERANCE_PCT = 5.0

# --- the August 2026 remakes ------------------------------------------------
#
# Formulations 4 and 5 were prepared again in August 2026 and photographed, which
# is why they are here at all: they are the only two feasible experiments the 2024
# imaging never covered. They are a different preparation of the same formulation,
# not a second look at the same pellets, so they carry the formulation's identity
# and none of its measured responses, and they are held out of every statistic
# that validates the 2024 measurements.
#
# The protocol also changed. The pellets are poured several deep instead of spread
# in one layer, which the separation rule cannot resolve, so three of the six
# photographs are unusable for automated sizing and say so through their
# confidence score. The rule itself is left alone: it was calibrated against the
# 2024 tables specifically so that it would not split dumbbells, and retuning it
# here would silently move every morphology number in the campaign analysis.
CAMPAIGN_SERIES = "campaign"
REMAKE_SERIES = "remake_2026"
REMAKE_SAMPLES = {"F4": 4, "F5": 5}
REMAKE_SESSION_DATE = "2026-08-24"  # from the capture timestamps in the ImageJ labels

# --- the September 2026 repeats ---------------------------------------------
#
# Batch 11 and formulations 4 and 5 prepared again, photographed on a phone
# instead of through ImageJ, together with eleven of the campaign's failures
# repeated for the record.
#
# Nothing in the photographs themselves fixes a pixel size. They are phone JPEGs
# with no ImageJ tag, no EXIF and no ruler, and the moulded tray they sit on is
# the only in-frame reference: its wells are a regular lattice, but they are dark
# on dark and span barely two periods per frame, and line detection, a 2D spectrum
# and a 2D autocorrelation all fail to measure their pitch. A scale could not be
# borrowed from the August session either, since that session's own two calibrated
# samples sit at 15.7 and 24.3 um/px.
#
# The scale arrived afterwards, in REPEAT_IMAGEJ below, and it turned out to vary
# from 21.5 to 27.8 um/px across twelve photographs, so borrowing one would have
# been wrong by up to 29% and no single nominal value could have been right.
REPEAT_SERIES = "repeat_2026"

# The operator's ImageJ session on the September material, delivered as one zip
# per formulation. Each holds three photographs saved as TIFF, one pooled table of
# line measurements, and the JPEGs the measurements were made on.
#
# The zips are read where they lie rather than unpacked. Their JPEGs are
# byte-identical to the photographs already in REPEAT_FOLDERS, so unpacking would
# duplicate twelve images and add 414 MB of TIFF whose only unique content is a
# calibration tag; and their TIFFs are named F4_n1.tif and the like, which would
# collide with the August remakes under the recursive glob that finds those. The
# build pairs a table to a photograph by hashing the zip's own JPEG and finding
# that hash in the tree, so the pairing is proved from content rather than assumed
# from a filename -- which matters here, because the operator renamed every frame
# and the two naming schemes have nothing in common.
#
# Declared rather than globbed, for the same reason REPEAT_FOLDERS is: a zip name
# cannot be trusted to say which formulation it holds.
REPEAT_IMAGEJ = {
    "F4_replicate 2.zip": 4,
    "F5_Replicate 2.zip": 5,
    "F43_04092026.zip": 43,
    "F44_04092026.zip": 44,
}

# One of those twelve TIFFs carries `unit=unit`, which is what ImageJ writes when
# a scale was set without naming its unit. It cannot be read through
# IMAGEJ_UNITS_PER_CM, which maps an unnamed unit to centimetres for the 2024
# files, because for this session that would be a factor of ten. Instead the unit
# is the one that puts the frame's field of view inside this range, which is the
# span of the eleven frames of the same session that do name their unit, widened a
# little. The unlabelled frame lands at 78 x 104 mm; read as centimetres it would
# be a photograph 7.8 metres across.
REPEAT_FOV_MM = (55.0, 95.0)

# A nominal scale was used for the filters before the tables arrived, since the
# size floor, the hole fill and the separation depth are stated in microns and had
# to land somewhere. It is kept only because REPEAT_SCALE_SENSITIVITY, the check
# that measured what assuming it cost, is reported as a limitation that the real
# scales have now retired.
NOMINAL_UM_PER_PX = 20.0
REPEAT_SCALE_SENSITIVITY = (12.0, 20.0, 28.0)

# Folder -> experiment, which preparation of that formulation it is, and whether
# it is measured. Declared rather than parsed: the folder names are consistent but
# the mapping from a name to a preparation is not something a filename can carry,
# and reading it wrong would attribute one preparation's pellets to another.
#
# Preparation 1 is always the 2024 campaign material. Formulations 4 and 5 reach 3
# because their August remake is preparation 2; the rest of these are the first
# time their formulation has been made since the campaign.
#
# The eleven marked False are repeats of experiments the campaign recorded as
# infeasible, and the operator reports that all eleven failed again. They are
# photographed for the record and are not measured, because what they contain is
# unspheronized extrudate, heaped and usually in a weighing boat, for which a
# pellet size distribution is not a meaningful thing to compute. The build does
# not re-adjudicate the failure, which is a visual call in 2026 exactly as it was
# in 2024; it measures how little of these photographs the pipeline can process,
# and reports that beside the four repeats that passed.
REPEAT_FOLDERS = {
    "F4_second_repeat": (4, 3, True),
    "F5_second_repeat": (5, 3, True),
    "F43_repeat": (43, 2, True),
    "F44_repeat": (44, 2, True),
    "F41_repeat": (41, 2, False),
    "F42_repeat": (42, 2, False),
    "F6_repeat": (6, 2, False),
    "F7_repeat": (7, 2, False),
    "F8_repeat": (8, 2, False),
    "F10_repeat": (10, 2, False),
    "F12_repeat": (12, 2, False),
    "F13_repeat": (13, 2, False),
    "F14_repeat": (14, 2, False),
    "F22_repeat": (22, 2, False),
    "F25_repeat": (25, 2, False),
}

# --- batch 12, 13 September 2026 --------------------------------------------
#
# Experiments 45-48, the first preparations of those designs. Photographed the
# day they were made (13 September) and again the next day at a closer working
# distance (the `_N2` folders, frames n4-n6). Each day-1 folder holds three
# ImageJ-calibrated TIFFs, the JPEGs they were saved from, and one operator
# table that covers all three frames. The N2 folders have the same TIFF/JPEG
# pair and no table: they are segmented from their own tags. The scales are in
# millimetres and differ between frames, so a table is split per photograph
# rather than pooled.
#
# Held out of every statistic that validates the 2024 measurements, for the same
# reason the remakes and repeats are: a 2026 session cannot score a pipeline
# that was calibrated on 2024 photographs.
BATCH12_SERIES = "batch12_2026"
BATCH12_SESSION_DATE = "2026-09-13"  # first session; N2 folders have their own date
BATCH12_FOLDERS = {
    "ILX006_F45_13092026": 45,
    "ILX006_F45_14092026_N2": 45,
    "ILX006_F46_13092026": 46,
    "ILX006_F46_14092026_N2": 46,
    "ILX006_F47_13092026": 47,
    "ILX006_F47_14092026_N2": 47,
    "ILX006_F48_13092026": 48,
    "ILX006_F48_14092026_N2": 48,
}
BATCH12_FOLDER_DATES = {
    "ILX006_F45_13092026": "2026-09-13",
    "ILX006_F45_14092026_N2": "2026-09-14",
    "ILX006_F46_13092026": "2026-09-13",
    "ILX006_F46_14092026_N2": "2026-09-14",
    "ILX006_F47_13092026": "2026-09-13",
    "ILX006_F47_14092026_N2": "2026-09-14",
    "ILX006_F48_13092026": "2026-09-13",
    "ILX006_F48_14092026_N2": "2026-09-14",
}
# Historical experiment each new design was built from. Used by the new-batch
# review figure so a row can be read against the pellets it was meant to improve.
BATCH12_EXPERIMENTS = (45, 46, 47, 48)
BATCH12_PARENTS = {45: 43, 46: 43, 47: 38, 48: 38}
BATCH12_NOTES = {
    45: "43 composition, 10 min gran",
    46: "43 polymers, drug 0.35, 6 min gran",
    47: "38 composition, 10 min gran",
    48: "38 polymers, drug 0.40, 6 min gran",
}
# Experiments 45-48 carry no visual score. They were never scored by eye, and
# the campaign's 1-5 ladder cannot supply one: its score-2 and score-4 classes
# hold two experiments each, and the three populated 0-3 quality levels are
# non-overlapping intervals of the aspect ratio these samples already have
# measured (1.072-1.262, 1.300-1.472, 1.523-1.558). Snapping a measurement onto
# that ladder would invent a label and lose resolution, so morphology for the
# new batches is carried by `granule_aspect_ratio` alone.
# No lactose was weighed in any experiment (confirmed October 2026). The planner
# proposed a cellulose:lactose split, but the operator weighed the whole filler
# as microcrystalline cellulose, so cellulose mass is the recorded cellulose plus
# lactose and the recorded split carries no information. Applied to every
# experiment in the dataset, including batch 12, whose sheet split was rebuilt
# from experiments 41-44.
FILLER_IS_CELLULOSE_ONLY = True

# Batch 13 photographs of experiments 49-52. Photographed the day they were
# made (17 September) and again the next day (18 September, frames n4-n6).
# Both days are ImageJ-calibrated TIFFs with no operator table: the same
# pattern as the Batch 12 N2 reshoot. Held out of the 2024 calibration set.
BATCH13_SERIES = "batch13_2026"
BATCH13_R2_SERIES = "batch13_r2_2026"
BATCH13_SESSION_DATE = "2026-09-17"  # first session; 18 September folders have their own date
BATCH13_FOLDERS = {
    "ILX006_F49_17092026": 49,
    "ILX006_F49_18092026": 49,
    "ILX006_F50_17092026": 50,
    "ILX006_F50_18092026": 50,
    "ILX006_F51_17092026": 51,
    "ILX006_F51_18092026": 51,
    "ILX006_F52_17092026": 52,
    "ILX006_F52_18092026": 52,
}
BATCH13_FOLDER_DATES = {
    "ILX006_F49_17092026": "2026-09-17",
    "ILX006_F49_18092026": "2026-09-18",
    "ILX006_F50_17092026": "2026-09-17",
    "ILX006_F50_18092026": "2026-09-18",
    "ILX006_F51_17092026": "2026-09-17",
    "ILX006_F51_18092026": "2026-09-18",
    "ILX006_F52_17092026": "2026-09-17",
    "ILX006_F52_18092026": "2026-09-18",
}
BATCH13_EXPERIMENTS = (49, 50, 51, 52)
BATCH13_PARENTS = {49: 39, 50: 39, 51: 39, 52: 43}
BATCH13_NOTES = {
    49: "39 chemistry, 10 min gran",
    50: "39 polymers, carbopol 0.15, 10 min gran",
    51: "39 polymers, carbopol 0.20, 14 min gran",
    52: "43 polymer load, carbopol 0.20, 8 min gran",
}
# Nominal 10 g weigh-up from the dissolution workbook's Masses used sheet,
# scaled to the campaign's 500 mg batch. Filler is cellulose only: that
# sheet lists no lactose. Process lock is experiment 39's spheronization
# and fluid, with granulation time as written in BATCH13_NOTES.
# F50's compritol was short on the actual weigh-up (3325 mg vs 3825 mg
# intended); the campaign row records the intended design.
BATCH13_PROCESS = {
    "gran_rpm": 146.052957,
    "spheronization_time": 36.025377,
    "spheronization_rpm": 3778.509325,
    "fluid_type": "water",
    "fluid_vol": 2.466378,
}
BATCH13_RECIPES = {
    49: {"drug_ww_frac": 0.30, "total_cntrl_polymer_ww_frac": 0.45,
         "cntrl_polymer_0_relative_ww_frac": 0.10, "mg_stearate_ww_frac": 0.0075,
         "cellulose_ww_frac": 0.2425, "lactose_ww_frac": 0.0, "gran_time": 10.0},
    50: {"drug_ww_frac": 0.30, "total_cntrl_polymer_ww_frac": 0.45,
         "cntrl_polymer_0_relative_ww_frac": 0.15, "mg_stearate_ww_frac": 0.0075,
         "cellulose_ww_frac": 0.2425, "lactose_ww_frac": 0.0, "gran_time": 10.0},
    51: {"drug_ww_frac": 0.30, "total_cntrl_polymer_ww_frac": 0.45,
         "cntrl_polymer_0_relative_ww_frac": 0.20, "mg_stearate_ww_frac": 0.0075,
         "cellulose_ww_frac": 0.2425, "lactose_ww_frac": 0.0, "gran_time": 14.0},
    52: {"drug_ww_frac": 0.30, "total_cntrl_polymer_ww_frac": 0.50,
         "cntrl_polymer_0_relative_ww_frac": 0.20, "mg_stearate_ww_frac": 0.0075,
         "cellulose_ww_frac": 0.1925, "lactose_ww_frac": 0.0, "gran_time": 8.0},
}
# Experiments 49-52 carry no visual score either, for the same reason as 45-48.
NEW_BATCH_EXPERIMENTS = BATCH12_EXPERIMENTS + BATCH13_EXPERIMENTS
NEW_BATCH_SERIES = (BATCH12_SERIES, BATCH13_SERIES)
NEW_BATCH_FOLDER_DATES = {**BATCH12_FOLDER_DATES, **BATCH13_FOLDER_DATES}
# Which campaign batch each post-2024 imaging series belongs to. These are new
# designs rather than remakes, so unlike the 2026 remakes they have a position
# on the campaign's batch axis and are drawn in campaign order.
SERIES_BATCH = {BATCH12_SERIES: 12, BATCH13_SERIES: 13,
                BATCH13_R2_SERIES: 13}
# Which preparation each imaging series photographed. The campaign photographed
# the original material; the August remakes were the second preparation of the two
# formulations they cover. Batches 12 and 13 are the first preparation of
# experiments 45-48 and 49-52.
SERIES_PREPARATION = {
    CAMPAIGN_SERIES: 1,
    REMAKE_SERIES: 2,
    BATCH12_SERIES: 1,
    BATCH13_SERIES: 1,
    BATCH13_R2_SERIES: 1,
}

# The August remakes are found by a recursive glob for `F4_*.tif` and `F5_*.tif`,
# and the September zips contain files of exactly those names. Nothing unpacks
# them today, so nothing collides, but the count is asserted at build time so that
# anyone who does unpack them gets an error rather than four September photographs
# quietly folded into the August series.
REMAKE_TIFFS_EXPECTED = 6

# Below this a photograph is reported but not trusted for an absolute size. It is
# a reporting threshold, not a filter: the pooled statistics still use every
# photograph, and each experiment carries a second median computed only from the
# photographs above it, so the two can be compared. Three of the 56 campaign
# photographs fall below (experiments 9, 11 and two of the three for 17) and two
# of the six remakes, the heaped ones at 0.19 and 0.29.
CONFIDENCE_FLOOR = 0.30

# Shot on 29 February, before the calibrated series begins, and their manual
# tables are the provisional "estimate" and "pilot" ones whose lengths disagree
# with the workbook. Their scale is recoverable from the Area/Length pixel width
# of those same tables, so a size is reported, but it rests on the operator's own
# calibration line and therefore cannot also be used to score the method.
PROVISIONAL_EXPERIMENTS = (9, 11)

# Its manual table holds 15 particles against three photographs, so it measures
# a fraction of one image and cannot score the method.
CALIBRATION_EXCLUDED = (16,)

# The late window is where the process was deliberately held constant, which is a
# fact about how the campaign was run rather than about how the pellets came out.
# Defining it on the outcome would make the contrast circular, so the split is
# also reported at every other batch, and it is the shape of that curve rather
# than the single chosen split that carries the claim.
LATE_BATCHES = (10, 11)

# Split of the raw 1-5 visual score for the class-bias test that decides whether
# the measurement is even-handed enough to carry a morphology response.
SPHERICAL_SCORES = (4, 5)
DUMBBELL_SCORES = (1, 2, 3)
N_PERMUTATIONS = 20_000

# Imaging session dates, pinned. Not one of the 57 JPEGs carries an EXIF
# timestamp, so the only record of when a photograph was taken is its file
# modification time, which copying or syncing the folder destroys. The pixel scale
# of an uncalibrated photograph is borrowed from another photograph of the same
# session, so that grouping must not depend on a filesystem attribute. These are
# the dates the files carried when the folder was first read; the build warns if
# the filesystem later disagrees.
IMAGING_SESSION_DATES = {
    9: "2024-02-29", 11: "2024-02-29", 15: "2024-03-04", 16: "2024-03-04",
    17: "2024-03-06", 21: "2024-03-06", 23: "2024-03-06", 24: "2024-03-07",
    26: "2024-03-12", 27: "2024-03-12", 30: "2024-03-15", 31: "2024-03-15",
    33: "2024-03-21", 35: "2024-03-21", 36: "2024-03-21", 37: "2024-03-27",
    38: "2024-03-27", 39: "2024-03-27", 43: "2024-04-11", 44: "2024-04-11",
}

# Smoothing for the density view of the size distributions. Fixed in microns, and
# about a fifth of a typical interquartile width, so it damps the noise of the
# roughly 25 particles in one photograph without flattening real differences.
DENSITY_BANDWIDTH_UM = 120.0

# --- response definitions --------------------------------------------------

# Ordered pellet quality, spanning all 44 experiments in a single response.
PELLET_QUALITY_LEVELS = {
    0: "failed to form",
    1: "unacceptable (shape score 1-2)",
    2: "acceptable (shape score 3)",
    3: "excellent (shape score 4-5)",
}


def shape_score_to_quality(shape_score, feasible):
    """Map the raw 1-5 visual shape score onto the acceptability scale.

    A feasible experiment with no visual score returns NaN rather than a
    level. The campaign batches were all scored by eye; the 2026 batches were
    not, and the ladder has too few experiments per class to invent one from a
    measurement. Without this branch an unscored batch silently reads as
    quality 1, "unacceptable".
    """
    if not feasible:
        return 0
    if shape_score is None or (isinstance(shape_score, float) and np.isnan(shape_score)):
        return np.nan
    if shape_score >= 4:
        return 3
    if shape_score == 3:
        return 2
    return 1


# Objectives the optimizer actually carried, with the direction of improvement.
# particle_shape_score was measured but NOT optimized, which is the subject of
# analysis E.
OBJECTIVES = {
    "release_area_diff": "minimize",
    "size_um": "unconfirmed",  # see README open questions
    "drug_mass_mg": "maximize",
}

# --- variable roles --------------------------------------------------------

FORMULATION_VARS = [
    "drug_ww_frac",
    "total_cntrl_polymer_ww_frac",
    "cntrl_polymer_0_type",
    "cntrl_polymer_0_relative_ww_frac",
    "cntrl_polymer_1_type",
    "mg_stearate_ww_frac",
]

PROCESS_VARS = [
    "gran_time",
    "gran_rpm",
    "spheronization_time",
    "spheronization_rpm",
    "fluid_type",
    "fluid_vol",
]

CATEGORICAL_VARS = ["cntrl_polymer_0_type", "cntrl_polymer_1_type", "fluid_type"]

# Recorded excipient grades that are the same material, folded onto one name
# before anything is encoded. Carbopol 71G NF is the granulated form of 971P NF:
# same polymer, crosslinking and viscosity specification. The two PVPs are
# Sigma 81420 and Sigma PVP40, both K-value 29-32 and Mw about 40,000; "40k" is
# the molecular weight, not K40. Supplier evidence is in
# data/reference/excipient_descriptors.csv. The recorded grade is kept in
# cntrl_polymer_{0,1}_grade, so 71G and 971P can still be told apart: they
# differ in physical form (granular 150-425 um against 2-7 um powder).
MATERIAL_ALIASES = {
    "carbopol_71g": "carbopol_71",
    "carbopol_971p": "carbopol_71",
    "pvp_40k": "pvp_30k",
}

# Seed for every Gaussian-process fit (torch). The bootstraps in
# retrospective/cv.py use numpy seed 0.
GP_SEED = 42

# The retrospective Chimera hierarchy (Table S2) lives in objectives.py.

CONTINUOUS_VARS = [v for v in FORMULATION_VARS + PROCESS_VARS if v not in CATEGORICAL_VARS]

# filler_ww_frac is exactly 1 - drug - polymer - stearate, so it is determined
# by the other composition variables and is never used as a model input.
DERIVED_VARS = ["filler_ww_frac"]
