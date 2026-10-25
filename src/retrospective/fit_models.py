"""Fit every retrospective model the manuscript reports and cache its predictions.

    representation   GP and penalised linear, sets E0/E1/E5 (A/B/C), all three
                     responses, all three validation schemes  -> Figure 6, Table 2
    feasibility      GP on the physicochemical set E3, all schemes -> Figure 7e,
                     leave-one-polymer-pair-out AUC in Section 2.5

Held-out predictions are cached per fit in outputs/retrospective/cache/, so runs
can be split across processes and resumed; delete a file to refit it. Scoring
always reads the cache.

    python src/retrospective/fit_models.py                  # everything (~1 h)
    python src/retrospective/fit_models.py --models linear  # minutes
    python src/retrospective/fit_models.py --models gp --responses feasibility
"""

import argparse
import time

import numpy as np
import pandas as pd

from common import CACHE, KIND, OUTPUTS, RESPONSES, campaign, response
from cv import SCHEMES, cross_predict, folds, score
from features import build

REPRESENTATION_SETS = ("E0", "E1", "E5")
FEASIBILITY_SET = "E3"


def key(resp, scheme, model, name, variant="plain"):
    return f"{resp.replace(' ', '_')}__{scheme}__{model}__{name}__{variant}"


def cached(resp, scheme, model, name, compute):
    path = CACHE / f"{key(resp, scheme, model, name)}.npz"
    if path.exists():
        return dict(np.load(path))
    start = time.time()
    tested, mean, sigma, converged = compute()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, tested=tested, mean=mean, sigma=sigma, converged=converged)
    print(f"  {path.stem}: {time.time() - start:.0f} s", flush=True)
    return dict(tested=tested, mean=mean, sigma=sigma, converged=converged)


def plain(df, resp, scheme, model, name):
    frame, y = response(df, resp)
    frame = frame.reset_index(drop=True)
    binary = KIND[resp] == "binary"
    X, _, cat_dims = build(frame, name)
    fold_list = folds(frame, y, scheme, binary)
    return cached(resp, scheme, model, name,
                  lambda: cross_predict(model, X, y, fold_list, binary, cat_dims))


def plan(models, responses, schemes):
    for model in models:
        for resp in responses:
            for scheme in schemes:
                sets = list(REPRESENTATION_SETS)
                if model == "gp" and resp == "feasibility":
                    sets.append(FEASIBILITY_SET)
                for name in sets:
                    yield resp, scheme, model, name


def score_all():
    df = campaign()
    records = []
    for path in sorted(CACHE.glob("*.npz")):
        resp, scheme, model, name, variant = path.stem.split("__")
        resp = resp.replace("_", " ")
        _, y = response(df, resp)
        fit = np.load(path)
        tested = fit["tested"].astype(int)
        binary = KIND[resp] == "binary"
        records.append(dict(response=resp, scheme=scheme, model=model, set=name,
                            variant=variant, n=len(tested),
                            **score(y[tested], fit["mean"][tested], binary),
                            converged=float(fit["converged"])))
    table = pd.DataFrame(records)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUTS / "scores.csv", index=False)
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=["linear", "gp"],
                        choices=("linear", "gp"))
    parser.add_argument("--responses", nargs="*", default=list(RESPONSES))
    parser.add_argument("--schemes", nargs="*", default=list(SCHEMES))
    args = parser.parse_args()
    df = campaign()
    for resp, scheme, model, name in plan(args.models, args.responses, args.schemes):
        plain(df, resp, scheme, model, name)
    print(score_all().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
