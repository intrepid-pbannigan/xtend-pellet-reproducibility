"""Leave-one-input-out ablation for the Figure 7 surrogates (panels b, d, f).

For every feature -- or every one-hot block, which is one choice and is dropped
whole -- the surrogate is refitted under leave-one-out with that feature held
constant, and the importance is how much the leave-one-out score falls (AUC for
feasibility, R2 otherwise). Holding a column constant is removal for both model
families: it adds nothing to any GP distance and standardises to zero for the
linear models. It also keeps the mixed GP's categorical dimensions, which it
cannot fit without.

Out of sample on purpose. An in-sample importance (permuting inputs of a model
fitted to all rows, or reading ARD length-scales) measures what the fit leans
on, which on 29-51 runs is not the same as what predicts. A feature whose
removal leaves the cross-validated score unchanged is one the data cannot show
to matter, whatever the fitted model does with it. The interval is a paired
bootstrap over rows of the score difference.

Correlated features share credit: dropping one of two carriers of the same
information costs little, so a small drop is "not uniquely needed", not "has no
effect".

    python src/retrospective/importance.py --surrogate release_E1   # one surrogate
    python src/retrospective/importance.py                          # all three

Refits are cached in outputs/retrospective/cache/importance/. The combined table
is outputs/tables/input_ablation.csv.
"""

import argparse

import numpy as np
import pandas as pd

from common import CACHE, KIND, OUTPUTS, campaign, cfg, response
from cv import cross_predict, folds, paired_bootstrap, score, PRIMARY
from features import build

SURROGATES = {
    "release_E1": ("release deviation", "E1"),
    "aspect_E5": ("aspect ratio", "E5"),
    "feasibility_E3": ("feasibility", "E3"),
}


def groups(columns):
    out = {}
    for i, column in enumerate(columns):
        name = column.split("=")[0] if "=" in column and not column.startswith("ww=") else column
        out.setdefault(name, []).append(i)
    return out


def run(tag):
    resp, name = SURROGATES[tag]
    df = campaign()
    frame, y = response(df, resp)
    frame = frame.reset_index(drop=True)
    binary = KIND[resp] == "binary"
    X, columns, cat = build(frame, name)
    fl = folds(frame, y, "loo", binary)
    full_path = CACHE / f"{resp.replace(' ', '_')}__loo__gp__{name}__plain.npz"
    if not full_path.exists():
        raise FileNotFoundError(f"{full_path} -- run src/retrospective/fit_models.py first")
    full = np.load(full_path)["mean"]
    base = score(y, full, binary)[PRIMARY[binary]]
    rows = []
    for group, idx in groups(columns).items():
        path = CACHE / "importance" / f"{tag}__{group}.npy"
        if path.exists():
            mean = np.load(path)
        else:
            Xd = X.copy()
            Xd[:, idx] = 0.0
            _, mean, _, _ = cross_predict("gp", Xd, y, fl, binary, cat)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, mean)
        dropped = score(y, mean, binary)[PRIMARY[binary]]
        b = paired_bootstrap(y, full, mean, binary, n=2000)
        rows.append(dict(surrogate=tag, response=resp, set=name, feature=group,
                         n_columns=len(idx), score_full=base, score_dropped=dropped,
                         importance=b["diff"], low=b["low"], high=b["high"],
                         p_positive=b["p_better"]))
        print(f"  {tag} {group}: {base:.3f} -> {dropped:.3f}", flush=True)
    table = pd.DataFrame(rows).sort_values("importance", ascending=False)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUTS / f"importance_{tag}.csv", index=False)
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surrogate", nargs="*", default=list(SURROGATES))
    for tag in parser.parse_args().surrogate:
        run(tag)
    tables = [pd.read_csv(OUTPUTS / f"importance_{tag}.csv") for tag in SURROGATES
              if (OUTPUTS / f"importance_{tag}.csv").exists()]
    cfg.TABLES.mkdir(parents=True, exist_ok=True)
    pd.concat(tables).to_csv(cfg.TABLES / "input_ablation.csv", index=False)


if __name__ == "__main__":
    main()
