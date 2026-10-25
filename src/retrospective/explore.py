"""The exploratory search behind Section 2.5.

Every model (GP, penalised linear) x feature set (E0-E6) x response x scheme,
plus the release-deviation variants tried before the A/B/C ablation was fixed:
log target, a batch-12/13 era indicator, 1/SEM^2 weights (linear only) and the
curve model (time as an input, deviation integrated from the predicted profile).
E4 is fitted with the linear model only: it describes the fluid by continuous
descriptors, so it has no categorical column for the mixed-kernel GP. In all,
180 combinations.

The reported fits (`fit_models.py`) are untouched: this has its own cache,
outputs/retrospective/exploratory_cache/, and writes exploratory_scores.csv.
Fits shared with fit_models.py are copied from its cache rather than refitted;
the GP is deterministic, so they are the same fits.

    python src/retrospective/explore.py --models linear
    python src/retrospective/explore.py --models gp --responses "release deviation"
    python src/retrospective/explore.py --variants --sets E0 E1 E5
    python src/retrospective/explore.py --score
"""

import argparse
import shutil
import time

import numpy as np
import pandas as pd

from common import CACHE as REPORTED, KIND, OUTPUTS, RESPONSES, campaign, cfg, release_sem, response
from cv import SCHEMES, cross_predict, folds, score
from features import SETS, build

CACHE = OUTPUTS / "exploratory_cache"
TIMES = [t for t in cfg.DISSOLUTION_TIMES_MIN if t > 0]


def key(resp, scheme, model, name, variant="plain"):
    return f"{resp.replace(' ', '_')}__{scheme}__{model}__{name}__{variant}"


def cached(resp, scheme, model, name, variant, compute):
    path = CACHE / f"{key(resp, scheme, model, name, variant)}.npz"
    reported = REPORTED / path.name
    CACHE.mkdir(parents=True, exist_ok=True)
    if not path.exists() and reported.exists():
        shutil.copy2(reported, path)
    if path.exists():
        return dict(np.load(path))
    start = time.time()
    tested, mean, sigma, converged = compute()
    np.savez(path, tested=tested, mean=mean, sigma=sigma, converged=converged)
    print(f"  {path.stem}: {time.time() - start:.0f} s", flush=True)
    return dict(tested=tested, mean=mean, sigma=sigma, converged=converged)


def plain(df, resp, scheme, model, name, extra=None, variant="plain", transform=None,
          weight=None):
    frame, y = response(df, resp)
    frame = frame.reset_index(drop=True)
    binary = KIND[resp] == "binary"
    X, _, cat_dims = build(frame, name, extra=None if extra is None else extra(frame))
    fold_list = folds(frame, y, scheme, binary)
    w = None if weight is None else weight(frame)
    return cached(resp, scheme, model, name, variant,
                  lambda: cross_predict(model, X, y, fold_list, binary, cat_dims,
                                        weight=w, transform=transform))


def era(frame):
    return {"era_2026": (frame.batch >= 12).astype(float).to_numpy()}


def sem_weight(frame):
    sem = release_sem(frame)["sem"].reindex(frame.experiment_num).to_numpy(float)
    floor = np.nanmedian(sem)
    sem = np.where(np.isfinite(sem) & (sem > 0), sem, floor)
    sem = np.maximum(sem, 0.25 * floor)
    weight = 1 / sem ** 2
    return weight / weight.mean()


def profile_deviation(profile):
    full = np.column_stack([np.zeros(len(profile)), profile])
    hours = np.asarray(cfg.DISSOLUTION_TIMES_MIN, float) / 60.0
    return np.trapezoid(np.abs(full - cfg.TARGET_PROFILE), hours, axis=1)


def curve(df, scheme, model, name):
    """Predict the release curve with time as an input, then integrate deviation.

    Linear model: every feature times 1, s and s^2 with s = t / 240 min. GP: the
    features plus t as one more continuous input. Folds are over experiments,
    so all seven points of a held-out run leave together.
    """
    resp = "release deviation"
    frame, y = response(df, resp)
    frame = frame.reset_index(drop=True)
    X, _, cat_dims = build(frame, name)
    s = np.asarray(TIMES, float) / max(TIMES)
    n, k = len(frame), len(TIMES)
    profile = frame[[f"frac_release_{int(t)}m" for t in TIMES]].to_numpy(float)
    assert np.allclose(profile_deviation(profile), y, atol=1e-6)
    rows = np.repeat(np.arange(n), k)
    st = np.tile(s, n)
    if model == "linear":
        base = X[rows]
        long_X = np.column_stack([base, base * st[:, None], base * st[:, None] ** 2, st, st ** 2])
        long_cat = []
    else:
        cont = [c for c in range(X.shape[1]) if c not in cat_dims]
        long_X = np.column_stack([X[rows][:, cont], st, X[rows][:, cat_dims]])
        long_cat = list(range(len(cont) + 1, len(cont) + 1 + len(cat_dims)))
    long_y = profile.ravel()
    row_folds = folds(frame, y, scheme, binary=False)

    def compute():
        long_folds = [(np.flatnonzero(np.isin(rows, tr)), np.flatnonzero(np.isin(rows, te)))
                      for tr, te in row_folds]
        tested, mean, _, converged = cross_predict(model, long_X, long_y, long_folds,
                                                   False, long_cat)
        deviation = profile_deviation(np.clip(mean.reshape(n, k), 0, 1.2))
        rows_tested = np.unique(rows[tested])
        out = np.full(n, np.nan)
        out[rows_tested] = deviation[rows_tested]
        return rows_tested, out, np.full(n, np.nan), converged

    return cached(resp, scheme, model, name, "curve", compute)


def score_all():
    df = campaign()
    records = []
    for path in sorted(CACHE.glob("*.npz")):
        resp, scheme, model, name, variant = path.stem.split("__")
        resp = resp.replace("_", " ")
        _, y = response(df, resp)
        fit = np.load(path)
        tested = fit["tested"].astype(int)
        records.append(dict(response=resp, scheme=scheme, model=model, set=name,
                            variant=variant, n=len(tested),
                            **score(y[tested], fit["mean"][tested], KIND[resp] == "binary"),
                            converged=float(fit["converged"])))
    table = pd.DataFrame(records)
    table.to_csv(OUTPUTS / "exploratory_scores.csv", index=False)
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", choices=("linear", "gp"))
    parser.add_argument("--responses", nargs="*", default=list(RESPONSES))
    parser.add_argument("--schemes", nargs="*", default=list(SCHEMES))
    parser.add_argument("--sets", nargs="*", default=list(SETS))
    parser.add_argument("--variants", action="store_true")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    df = campaign()

    if args.models:
        for resp in args.responses:
            for scheme in args.schemes:
                for name in args.sets:
                    plain(df, resp, scheme, args.models, name)
    if args.variants:
        resp = "release deviation"
        for name in args.sets:
            for scheme in args.schemes:
                for model in ("linear", "gp"):
                    plain(df, resp, scheme, model, name, variant="log",
                          transform=(np.log, np.exp))
                    plain(df, resp, scheme, model, name, variant="era", extra=era)
                    curve(df, scheme, model, name)
                plain(df, resp, scheme, "linear", name, variant="sem_weighted",
                      weight=sem_weight)
    table = score_all()
    if args.score:
        print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
