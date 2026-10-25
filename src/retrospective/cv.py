"""Folds, models, scores and bootstrap intervals for the retrospective analysis.

Three schemes, each answering a different question:

    loo       leave-one-out
    lopo      leave-one-polymer-pair-out: every run with the same set of polymers
              is held out together, so a model must predict a combination it has
              never seen.
    forward   train on batches before k, predict batch k: the sequential view.
              A batch is predicted only once training has at least MIN_TRAIN
              rows (and both classes, for feasibility), so every feature set and
              model is scored on the same rows.

Every fold refits from scratch, so any scaling a model does is learned from its
training rows only.
"""

import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import cfg
from features import material_fractions

import models

SCHEMES = ("loo", "lopo", "forward")
MIN_TRAIN = 5
ALPHAS = np.logspace(-3, 3, 13)
N_BOOT = 5000
EPS = 1e-6


def pair_groups(frame):
    fractions = material_fractions(frame)
    return np.array(["+".join(sorted(fractions.columns[row > 0])) or "no polymer"
                     for row in fractions.to_numpy()])


def folds(frame, y, scheme, binary):
    """List of (train, test) positional index arrays."""
    n = len(frame)
    if scheme == "loo":
        return [(np.delete(np.arange(n), i), np.array([i])) for i in range(n)]
    if scheme == "lopo":
        groups = pair_groups(frame)
        return [(np.flatnonzero(groups != g), np.flatnonzero(groups == g))
                for g in dict.fromkeys(groups)]
    if scheme == "forward":
        batch = frame.batch.to_numpy()
        out = []
        for k in sorted(set(batch)):
            train, test = np.flatnonzero(batch < k), np.flatnonzero(batch == k)
            if len(train) < MIN_TRAIN or (binary and len(set(y[train])) < 2):
                continue
            out.append((train, test))
        return out
    raise KeyError(scheme)


class Linear:
    """Standardised, penalised linear model with the penalty chosen inside the fold."""

    def __init__(self, binary):
        self.binary = binary

    def fit(self, X, y, weight=None):
        self.constant = None
        if self.binary:
            y = y.astype(int)
            minority = int(min(np.bincount(y, minlength=2)))
            if minority == 0:
                self.constant = float(y.mean())
                return self
            if minority >= 3:
                estimator = LogisticRegressionCV(Cs=1 / ALPHAS, cv=min(5, minority),
                                                 scoring="neg_log_loss", max_iter=5000)
            else:
                estimator = LogisticRegression(C=1.0, max_iter=5000)
        else:
            estimator = RidgeCV(alphas=ALPHAS)
        self.model = make_pipeline(StandardScaler(), estimator)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            kwargs = {} if weight is None else {f"{self.model.steps[-1][0]}__sample_weight": weight}
            self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        if self.constant is not None:
            return np.full(len(X), self.constant), np.full(len(X), np.nan)
        with np.errstate(all="ignore"):
            if self.binary:
                return self.model.predict_proba(X)[:, 1], np.full(len(X), np.nan)
            return self.model.predict(X), np.full(len(X), np.nan)


class GP:
    """Atlas-style classifier for feasibility, mixed GP otherwise."""

    def __init__(self, binary, cat_dims):
        if not cat_dims:
            raise ValueError("the mixed GP needs at least one categorical column")
        self.binary, self.cat_dims = binary, cat_dims

    def fit(self, X, y, weight=None):
        if weight is not None:
            raise ValueError("the GP takes no sample weights")
        self.model = models.surrogate_for("binary" if self.binary else "continuous",
                                          self.cat_dims, seed=cfg.GP_SEED)
        self.model.fit(X, y)
        self.converged = getattr(self.model, "converged", None)
        return self

    def predict(self, X):
        mean, sigma = self.model.predict(X)
        return np.asarray(mean, float), np.asarray(sigma, float)


def make(model, binary, cat_dims):
    return Linear(binary) if model == "linear" else GP(binary, cat_dims)


def cross_predict(model, X, y, fold_list, binary, cat_dims, weight=None,
                  transform=None):
    """Held-out predictions on the rows the folds test, in their original order.

    `transform` is a (forward, inverse) pair applied to the target in each fold.
    """
    tested = np.concatenate([test for _, test in fold_list])
    mean = np.full(len(y), np.nan)
    sigma = np.full(len(y), np.nan)
    converged = []
    for train, test in fold_list:
        target = y[train] if transform is None else transform[0](y[train])
        fitted = make(model, binary, cat_dims).fit(
            X[train], target, None if weight is None else weight[train])
        m, s = fitted.predict(X[test])
        mean[test] = m if transform is None else transform[1](m)
        sigma[test] = s
        converged.append(getattr(fitted, "converged", None))
    flags = [c for c in converged if c is not None]
    return np.sort(tested), mean, sigma, (float(np.mean(flags)) if flags else np.nan)


def score(y, pred, binary):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if binary:
        p = np.clip(pred, EPS, 1 - EPS)
        both = len(set(y)) > 1
        brier = np.mean((y - p) ** 2)
        return dict(auc=roc_auc_score(y, p) if both else np.nan,
                    log_loss=log_loss(y, p, labels=[0, 1]),
                    brier_skill=1 - brier / y.var() if y.var() > 0 else np.nan)
    rho = spearmanr(y, pred).statistic if len(set(pred)) > 1 else np.nan
    return dict(r2=1 - np.mean((y - pred) ** 2) / y.var(), rho=rho,
                rmse=float(np.sqrt(np.mean((y - pred) ** 2))))


PRIMARY = {True: "auc", False: "r2"}
N_CI = 1000


def metric_ci(y, pred, binary, seed=0):
    """95% percentile interval of the primary metric over resampled rows."""
    metric = PRIMARY[binary]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(N_CI):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        draws.append(score(y[idx], pred[idx], binary)[metric])
    return np.nanpercentile(draws, [2.5, 97.5])


def paired_bootstrap(y, pred, reference, binary, n=N_BOOT, seed=0):
    """Difference in the primary metric (pred minus reference) over resampled rows.

    Paired: both predictions are scored on the same resample. Returns the
    observed difference, a 95% interval and the share of resamples above zero.
    """
    metric = PRIMARY[binary]
    rng = np.random.default_rng(seed)
    observed = score(y, pred, binary)[metric] - score(y, reference, binary)[metric]
    draws = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if binary and len(set(y[idx])) < 2:
            continue
        if not binary and y[idx].var() == 0:
            continue
        draws.append(score(y[idx], pred[idx], binary)[metric]
                     - score(y[idx], reference[idx], binary)[metric])
    draws = np.asarray(draws)
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return dict(diff=observed, low=low, high=high, p_better=float(np.nanmean(draws > 0)))
