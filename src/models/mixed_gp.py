"""The retrospective mixed categorical/continuous Gaussian-process regressor.

This is the regression surrogate used for every retrospective analysis in the
manuscript (Section 2.5, Figures 6 and 7, Table 2). It is a standard BoTorch
`MixedSingleTaskGP` with the settings stated in the Supplementary Information.
It is not the campaign's experiment-planning model, which is not distributed.

An exact GP over the design space built by `design_space.py`, using BoTorch's
`MixedSingleTaskGP` rather than a single stationary kernel over the whole vector.
The distinction is the point of the model: excipient identity is a Hamming
comparison between one-hot blocks, w/w fractions and process settings are a
Matern 5/2 with a lengthscale per dimension, and the two are combined as
sum-plus-product so the model can express both "this polymer behaves like that
one" and "this polymer responds differently to granulation time".

Four settings here are load-bearing and none of them is the library default.

Matern, not RBF. BoTorch 0.12 changed the default continuous kernel inside
`MixedSingleTaskGP` to RBF, silently, and this repo runs 0.18.1. Verified against
the installed version: omitting `cont_kernel_factory` yields an `RBFKernel`. RBF
assumes the response surface is infinitely differentiable, which for a
granulation endpoint is an assumption nobody would make on purpose, so the
factory is passed explicitly and a test asserts the kernel that came back.

No outcome transform. BoTorch standardises the outcome by default. This class
z-scores `y` itself, inside `fit`, so leaving the default in place would
standardise twice and the returned sigma would no longer be in the target's
units -- which is exactly the failure that does not show up in a point-prediction
metric and does show up in coverage.

Jitter of 1e-1. Early batches contain near-duplicate compositions, and a kernel
matrix over near-duplicate rows is near-singular. This is a large jitter by the
usual standards and it is deliberate.

Observation noise on at predict time. The returned sigma describes the next
measurement, not the latent surface, because everything downstream that uses it
asks whether a measured outcome would fall inside an interval.

The categorical indices are computed by `design_space.build_matrix` while it lays
the columns out, never inferred afterwards, so they cannot drift from the column
layout when there is more than one categorical block.
"""

import os
import sys
import warnings
from pathlib import Path

# Before torch is imported, or the thread pool is already sized. These fits are
# a few dozen rows wide; extra threads contend rather than help, and a
# leave-one-out sweep runs many of them back to back.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import gpytorch  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from botorch.exceptions.warnings import OptimizationWarning  # noqa: E402
from botorch.fit import fit_gpytorch_mll  # noqa: E402
from botorch.models import MixedSingleTaskGP  # noqa: E402
from gpytorch.constraints import GreaterThan  # noqa: E402
from gpytorch.kernels import MaternKernel  # noqa: E402
from gpytorch.mlls import ExactMarginalLogLikelihood  # noqa: E402

torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.base import Surrogate  # noqa: E402

DTYPE = torch.double

# Matern 5/2, twice differentiable. The lengthscale floor keeps a dimension the
# campaign barely varied from collapsing to zero lengthscale and turning into a
# lookup table on that axis.
MATERN_NU = 2.5
LENGTHSCALE_FLOOR = 1e-04

# Large on purpose; see the module docstring.
CHOLESKY_JITTER = 1e-1

# A column the training fold never varied cannot be min-max scaled. Left alone
# rather than dropped, so the column count is a property of the design space and
# not of the fold.
CONSTANT_TOLERANCE = 1e-10


def matern_factory(batch_shape, ard_num_dims, active_dims):
    """The continuous kernel, passed explicitly so the default cannot apply."""
    return MaternKernel(
        nu=MATERN_NU,
        batch_shape=batch_shape,
        ard_num_dims=ard_num_dims,
        active_dims=active_dims,
        lengthscale_constraint=GreaterThan(LENGTHSCALE_FLOOR),
    )


class MixedGP(Surrogate):
    """Exact GP with a Hamming kernel on one-hot blocks and Matern on the rest.

    Preprocessing lives inside `fit` and uses only the rows `fit` was given,
    which is what makes a leave-one-out sweep over this class honest.
    """

    def __init__(self, cat_dims, seed=42):
        if not len(cat_dims):
            raise ValueError(
                "MixedSingleTaskGP requires at least one categorical dimension; "
                "cat_dims came back empty, which usually means the design space "
                "was built without its one-hot blocks")
        self.cat_dims = list(cat_dims)
        self.seed = seed
        self.model = None
        # Whether the marginal-likelihood optimiser reported success. Recorded
        # rather than warned about, because on the small folds it fails often and
        # a stream of warnings is easy to stop reading -- while a fit that did not
        # converge is a fit whose hyperparameters are wherever the optimiser gave
        # up, which is a different object from the one the method section claims.
        self.converged = None
        self._lo = self._hi = None
        self._mu = self._sigma = None

    def clone(self):
        return MixedGP(self.cat_dims, self.seed)

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()

        # Min-max from this fold's rows only. A constant column keeps its raw
        # value instead of being divided by a zero range: it contributes nothing
        # to any distance, which is the correct behaviour for a one-hot level
        # that no training row happens to use.
        self._lo = X.min(axis=0)
        self._hi = X.max(axis=0)
        # Z-score the target too. A fold with a constant target gets sigma 1, so
        # the model degrades to predicting the mean rather than producing NaNs.
        self._mu = float(y.mean())
        self._sigma = float(y.std(ddof=0)) or 1.0

        train_X = torch.as_tensor(self._scale(X), dtype=DTYPE)
        train_Y = torch.as_tensor(
            ((y - self._mu) / self._sigma)[:, None], dtype=DTYPE)

        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            self.model = MixedSingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                cat_dims=self.cat_dims,
                cont_kernel_factory=matern_factory,
                outcome_transform=None,
            )
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            with (gpytorch.settings.cholesky_jitter(CHOLESKY_JITTER),
                  warnings.catch_warnings(record=True) as caught):
                warnings.simplefilter("always", OptimizationWarning)
                fit_gpytorch_mll(mll)
            self.converged = not any(
                issubclass(entry.category, OptimizationWarning) for entry in caught)
        return self

    def _scale(self, X):
        span = self._hi - self._lo
        scaled = X.astype(float).copy()
        movable = span > CONSTANT_TOLERANCE
        scaled[:, movable] = (X[:, movable] - self._lo[movable]) / span[movable]
        return scaled

    def predict(self, X):
        """Posterior mean and standard deviation of a measured outcome."""
        self._require_fit()
        test_X = torch.as_tensor(self._scale(np.asarray(X, float)), dtype=DTYPE)
        self.model.eval()
        with (torch.no_grad(),
              gpytorch.settings.cholesky_jitter(CHOLESKY_JITTER),
              gpytorch.settings.fast_pred_var()):
            posterior = self.model.posterior(test_X, observation_noise=True)
            mean = posterior.mean.squeeze(-1).numpy()
            sigma = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1).numpy()
        return mean * self._sigma + self._mu, sigma * self._sigma

    def sample_joint(self, X, n_samples, seed=None):
        """Draws from the joint posterior over a set of candidates.

        Joint rather than marginal because the questions it answers are
        family-wise -- the probability that any candidate in a batch beats the
        incumbent depends on how their errors covary, and independent marginals
        would answer a different question. Cubic in the number of candidates, and
        without `fast_pred_var`, whose Lanczos marginals are not a covariance
        that can be factored.
        """
        self._require_fit()
        test_X = torch.as_tensor(self._scale(np.asarray(X, float)), dtype=DTYPE)
        self.model.eval()
        with (torch.no_grad(),
              gpytorch.settings.cholesky_jitter(CHOLESKY_JITTER),
              torch.random.fork_rng()):
            torch.manual_seed(self.seed if seed is None else seed)
            posterior = self.model.posterior(test_X, observation_noise=True)
            draws = posterior.rsample(torch.Size([n_samples])).squeeze(-1).numpy()
        return draws * self._sigma + self._mu

    def _require_fit(self):
        if self.model is None:
            raise RuntimeError("predict called before fit")

    def kernel_report(self):
        """What kernel actually got built, for the guard test and the record."""
        self._require_fit()
        names = sorted({type(module).__name__
                        for module in self.model.covar_module.modules()})
        return dict(
            kernels=names,
            has_matern="MaternKernel" in names,
            has_rbf="RBFKernel" in names,
            has_categorical="CategoricalKernel" in names,
            outcome_transform=getattr(self.model, "outcome_transform", None),
            n_categorical=len(self.cat_dims),
        )
