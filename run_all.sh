#!/usr/bin/env bash
# Reproduce every figure, table and quoted number in the manuscript.
#
#   ./run_all.sh                about 1 h on a laptop CPU
#   ./run_all.sh --exploratory  also the 180-configuration search of Section 2.5
#                               (about 15 h more)
#
# Figure S1 also needs the photographs unzipped into data/raw/ (see README).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"

EXPLORATORY=0
for arg in "$@"; do
  case "$arg" in
    --exploratory) EXPLORATORY=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

(cd src && "$PY" unpack_dataset.py)

# Figures 2-5, S1, S2 and Tables 1, S1-S3
"$PY" src/manuscript/build_figures.py
"$PY" src/manuscript/tables.py

# Figures 6-7 and Table 2
"$PY" src/retrospective/fit_models.py
"$PY" src/retrospective/ablation.py
"$PY" src/retrospective/importance.py
"$PY" src/retrospective/figure7.py

if [[ $EXPLORATORY == 1 ]]; then
  (cd src/retrospective &&
    "$PY" explore.py --models linear &&
    "$PY" explore.py --models gp --sets E0 E1 E2 E3 E5 E6 --responses "release deviation" "aspect ratio" &&
    "$PY" explore.py --models gp --sets E0 E1 E2 E3 E5 E6 --responses feasibility &&
    "$PY" explore.py --variants --sets E0 E1 E5)
fi

"$PY" src/manuscript/render_tables.py
"$PY" src/manuscript/check_numbers.py
