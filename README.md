# Closed-loop formulation–process optimization under unknown feasibility for extended-release pellets

Code to reproduce every figure, table and quoted number in

> Bannigan P., Saputra N., Craig M., Allen C., Hickman R. J. *Closed-loop
> formulation–process optimization under unknown feasibility for extended-release
> pellets.*

Corresponding authors: Pauric Bannigan (pauric@intrepidlabs.ai) and Riley J. Hickman
(riley@intrepidlabs.ai), Intrepid Labs Inc., Toronto, Canada.

The ANDROMEDA experiment-planning software is not included and is not needed.

## Data

- `data/ILX006_pellet_campaign_dataset.xlsx`: all experimental data, one sheet per
  table, with a README sheet and data dictionary
  ([10.5281/zenodo.23651360](https://doi.org/10.5281/zenodo.23651360)).
- Pellet photographs, needed only for Figure S1
  ([10.5281/zenodo.23651221](https://doi.org/10.5281/zenodo.23651221)). Unzip
  the archive in the repository root.

The filler is microcrystalline cellulose only: `lactose_mass` is 0 throughout.
Use `release_deviation` as the release objective.

## Run

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh                # about 1 h on a laptop CPU
./run_all.sh --exploratory  # adds the 180-configuration search of Section 2.5 (about 15 h)
```

Figures are written to `outputs/manuscript/` and tables to `outputs/tables/`.
`outputs/tables/manuscript_numbers.csv` compares each number quoted in the text
with the recomputed value.

| Manuscript | Script |
|---|---|
| Figures 2–5, S1, S2 | `src/manuscript/build_figures.py` |
| Tables 1, S1–S3 | `src/manuscript/tables.py` |
| Figure 6, Table 2 | `src/retrospective/ablation.py` |
| Figure 7 | `src/retrospective/importance.py`, `figure7.py` |

Results were produced on macOS (Apple silicon, CPU) with the versions in
`requirements.txt`; Figure S1 needs scikit-image 0.24.

## Licence

Code: MIT. Data: CC BY 4.0. See `LICENSE`.
