"""Atlas's variational GP classifier, for the unknown feasibility constraint.

The retrospective analysis has two surrogates, not one. The regression GP in
`mixed_gp.py` is for responses that exist.
Feasibility is a different object: it is observed on every experiment, it is
binary, and a failure means the dissolution objective is never measured. Atlas
and Anubis (Hickman, Aspuru-Guzik and co-workers) handle that with a separate
variational Bernoulli GP whose posterior is P(feasible | x), then fold that
probability into a feasibility-aware acquisition function.

This class is that classifier, transcribed from Atlas rather than imported, so
the analysis does not grow an Olympus/Atlas dependency. The variational scheme
is Atlas's `ClassificationGPMatern` plus `train_vgp`: training points as
inducing points, an unwhitened variational strategy, a Bernoulli likelihood,
and Adam on the variational ELBO. Two kernels are offered because Atlas ships
a single Matern over the whole vector, while this campaign's design space is
the mixed Hamming-plus-Matern kernel the regression surrogate already uses.

    kernel="matern"   ScaleKernel(Matern 5/2), no ARD. Atlas as shipped.
    kernel="mixed"    the MixedSingleTaskGP sum-plus-product kernel, so the
                      classifier sees excipient identity the same way the
                      regressor does.

Labels are this repository's convention, y = 1 if pellets form. Atlas internally
stores 1 for infeasible measurements and reports 1 - P(infeasible); the
probability that comes out of `predict` here is already P(pellets form), which
is what every other table in the project reports.

The inner three-fold epoch search in Atlas's `train_vgp` is omitted. That
search exists to pick a stopping time in an online loop; running it inside
leave-one-out would choose the iteration count from the same campaign being
scored. A fixed budget, the Atlas default learning rate, and the same Cholesky
jitter as the regression GP keep the comparison honest.

Transcribed from Atlas (MIT), https://github.com/aspuru-guzik-group/atlas:
    src/atlas/gps/gps.py     ClassificationGPMatern
    src/atlas/base/base.py   train_vgp
"""

import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import gpytorch  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from botorch.models.kernels.categorical import CategoricalKernel  # noqa: E402
from gpytorch.constraints import GreaterThan  # noqa: E402
from gpytorch.distributions import MultivariateNormal  # noqa: E402
from gpytorch.kernels import MaternKernel, ScaleKernel  # noqa: E402
from gpytorch.likelihoods import BernoulliLikelihood  # noqa: E402
from gpytorch.mlls import VariationalELBO  # noqa: E402
from gpytorch.models import ApproximateGP  # noqa: E402
from gpytorch.variational import (  # noqa: E402
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
)

torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.mixed_gp import (  # noqa: E402
    CHOLESKY_JITTER,
    CONSTANT_TOLERANCE,
    DTYPE,
    MATERN_NU,
    matern_factory,
)
from models.base import Surrogate  # noqa: E402

# Atlas GPPlanner defaults. The iteration count is the budget `train_vgp`
# starts from before early stopping; used here as the whole budget.
VGP_ITERS = 1000
VGP_LR = 0.1

KERNELS = ("matern", "mixed")


@contextmanager
def double_default():
    """Torch's global default dtype, set to double for one block and restored.

    The variational machinery builds parameters at the default dtype before
    anything gets a chance to cast them -- `CholeskyVariationalDistribution` is
    sized from the training set and allocated on the spot -- so the fit needs
    double to be the default and not merely requested.

    Set and restored here rather than once at import, which is what this module
    did first and which was wrong in a way worth recording. Importing it flipped
    the process-wide default, so the regression GP in `mixed_gp.py` fitted
    differently depending on whether a classifier had been imported first: on
    release deviation that moved leave-one-out R-squared from +0.432 to +0.440
    and individual predictions by up to 0.378, which is larger than the model's
    own RMSE. A number that depends on import order is not a result.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def mixed_covar_module(n_dims, cat_dims, batch_shape=torch.Size()):
    """The MixedSingleTaskGP kernel, copied so the two surrogates cannot drift.

    K = scale(Matern_cont + scale(Hamming_cat)) + scale(Matern_cont * Hamming_cat)
    """
    if not len(cat_dims):
        raise ValueError(
            "mixed kernel requires categorical dimensions; "
            "use kernel='matern' for a fully continuous vector")
    cat_dims = list(cat_dims)
    ord_dims = sorted(set(range(n_dims)) - set(cat_dims))
    if not ord_dims:
        return ScaleKernel(
            CategoricalKernel(
                batch_shape=batch_shape,
                ard_num_dims=len(cat_dims),
                lengthscale_constraint=GreaterThan(1e-06),
            )
        )
    sum_kernel = ScaleKernel(
        matern_factory(batch_shape, len(ord_dims), ord_dims)
        + ScaleKernel(
            CategoricalKernel(
                batch_shape=batch_shape,
                ard_num_dims=len(cat_dims),
                active_dims=cat_dims,
                lengthscale_constraint=GreaterThan(1e-06),
            )
        )
    )
    prod_kernel = ScaleKernel(
        matern_factory(batch_shape, len(ord_dims), ord_dims)
        * CategoricalKernel(
            batch_shape=batch_shape,
            ard_num_dims=len(cat_dims),
            active_dims=cat_dims,
            lengthscale_constraint=GreaterThan(1e-06),
        )
    )
    return sum_kernel + prod_kernel


def atlas_matern_module():
    """Atlas as shipped: one Matern 5/2, one lengthscale, no Hamming block."""
    return ScaleKernel(MaternKernel(nu=MATERN_NU))


class _VariationalClassifier(ApproximateGP):
    """Latent GP. The Bernoulli likelihood lives on AtlasClassifier, not here."""

    def __init__(self, train_x, covar_module):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = UnwhitenedVariationalStrategy(
            self,
            train_x,
            variational_distribution,
            learn_inducing_locations=False,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = covar_module

    def forward(self, x):
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class AtlasClassifier(Surrogate):
    """Variational Bernoulli GP, Atlas's unknown-constraint classifier.

    `predict` returns P(y = 1) and the Bernoulli observation sd, both in
    probability units, so the leave-one-out harness can score it next to the
    0/1 regression GP without changing the contract.
    """

    def __init__(self, cat_dims, kernel="mixed", seed=42,
                 n_iter=VGP_ITERS, lr=VGP_LR):
        if kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")
        if kernel == "mixed" and not len(cat_dims):
            raise ValueError(
                "mixed kernel requires at least one categorical dimension; "
                "cat_dims came back empty")
        self.cat_dims = list(cat_dims)
        self.kernel = kernel
        self.seed = seed
        self.n_iter = int(n_iter)
        self.lr = float(lr)
        self.model = None
        self.likelihood = None
        self.converged = None
        self._lo = self._hi = None
        self._constant_p = None

    def clone(self):
        return AtlasClassifier(
            self.cat_dims, self.kernel, self.seed, self.n_iter, self.lr)

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        if not set(np.unique(y)).issubset({0.0, 1.0}):
            raise ValueError("AtlasClassifier expects binary labels in {0, 1}")

        self._lo = X.min(axis=0)
        self._hi = X.max(axis=0)
        self.model = self.likelihood = None
        self._constant_p = None
        self.converged = True

        # A fold with one class has no boundary to learn. Predicting the
        # training rate is the same degradation the logistic baseline uses.
        if len(np.unique(y)) < 2:
            self._constant_p = float(y.mean())
            return self

        train_X = torch.as_tensor(self._scale(X), dtype=DTYPE)
        train_y = torch.as_tensor(y.copy(), dtype=DTYPE)
        covar = (mixed_covar_module(train_X.shape[-1], self.cat_dims)
                 if self.kernel == "mixed" else atlas_matern_module())

        with torch.random.fork_rng(), double_default():
            torch.manual_seed(self.seed)
            model = _VariationalClassifier(train_X, covar).to(dtype=DTYPE)
            likelihood = BernoulliLikelihood().to(dtype=DTYPE)
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            mll = VariationalELBO(likelihood, model, train_y.numel())
            loss = None
            with (gpytorch.settings.cholesky_jitter(CHOLESKY_JITTER),
                  warnings.catch_warnings()):
                warnings.filterwarnings("ignore", message=".*jitter.*")
                for _ in range(self.n_iter):
                    optimizer.zero_grad()
                    output = model(train_X)
                    loss = -mll(output, train_y)
                    if not torch.isfinite(loss):
                        self.converged = False
                        break
                    loss.backward()
                    optimizer.step()
            if loss is None or not torch.isfinite(loss):
                self.converged = False
                self._constant_p = float(y.mean())
                return self
            self.model = model
            self.likelihood = likelihood
        return self

    def _scale(self, X):
        span = self._hi - self._lo
        scaled = X.astype(float).copy()
        movable = span > CONSTANT_TOLERANCE
        scaled[:, movable] = (X[:, movable] - self._lo[movable]) / span[movable]
        return scaled

    def predict(self, X):
        """P(y = 1) and the sd of a Bernoulli observation at that probability."""
        X = np.asarray(X, float)
        if self._constant_p is not None:
            p = np.full(len(X), self._constant_p)
            return p, np.sqrt(np.clip(p * (1.0 - p), 0.0, None))
        self._require_fit()
        test_X = torch.as_tensor(self._scale(X), dtype=DTYPE)
        self.model.eval()
        self.likelihood.eval()
        with (torch.no_grad(), double_default(),
              gpytorch.settings.cholesky_jitter(CHOLESKY_JITTER)):
            posterior = self.likelihood(self.model(test_X))
            p = posterior.mean.clamp(1e-6, 1 - 1e-6).numpy()
        return p, np.sqrt(p * (1.0 - p))

    def predict_epistemic(self, X, n_samples=512):
        """P(y = 1) and how uncertain the model is about that probability.

        `predict` returns sqrt(p(1-p)), the sd of one Bernoulli draw, because
        that is the quantity the leave-one-out harness scores against a 0/1
        outcome. It is the wrong scale for asking whether a partial-dependence
        span is real: it is at its largest, 0.5, exactly when the model is most
        undecided, it never shrinks as evidence accumulates, and no partial
        dependence on a probability can exceed 1.0 -- so comparing spans
        against it would dismiss every driver of feasibility by construction.

        What that comparison needs is the model's uncertainty about the
        probability itself, which does shrink as the campaign pins the boundary
        down. It is obtained by sampling the latent GP and pushing the draws
        through the probit link the Bernoulli likelihood uses, rather than by a
        delta-method expansion, because the link saturates and the posterior is
        wide enough at the edges of this design space for a local linearisation
        to overstate the spread.
        """
        X = np.asarray(X, float)
        if self._constant_p is not None:
            # No boundary was learned, so there is nothing to be uncertain
            # about beyond the rate itself.
            return np.full(len(X), self._constant_p), np.zeros(len(X))
        self._require_fit()
        test_X = torch.as_tensor(self._scale(X), dtype=DTYPE)
        self.model.eval()
        with (torch.no_grad(), double_default(),
              gpytorch.settings.cholesky_jitter(CHOLESKY_JITTER),
              torch.random.fork_rng()):
            torch.manual_seed(self.seed)
            latent = self.model(test_X)
            draws = latent.rsample(torch.Size([int(n_samples)]))
            probs = torch.distributions.Normal(0.0, 1.0).cdf(draws)
            return probs.mean(0).numpy(), probs.std(0).numpy()

    def _require_fit(self):
        if self.model is None and self._constant_p is None:
            raise RuntimeError("predict called before fit")

    def kernel_report(self):
        """What kernel actually got built, for the guard test and the record."""
        self._require_fit()
        if self.model is None:
            return dict(kernels=[], has_matern=False, has_rbf=False,
                        has_categorical=False, likelihood="constant",
                        n_categorical=len(self.cat_dims), kernel=self.kernel)
        names = sorted({type(module).__name__
                        for module in self.model.covar_module.modules()})
        return dict(
            kernels=names,
            has_matern="MaternKernel" in names,
            has_rbf="RBFKernel" in names,
            has_categorical="CategoricalKernel" in names,
            likelihood=type(self.likelihood).__name__,
            n_categorical=len(self.cat_dims),
            kernel=self.kernel,
        )

    def fit_predict_loo(self, X, y):
        """Leave-one-out with fold progress, because a thousand Adam steps is slow."""
        X, y = np.asarray(X, float), np.asarray(y, float)
        mean = np.empty(len(y))
        sigma = np.empty(len(y))
        converged = []
        for i in range(len(y)):
            if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(y):
                print(f"    fold {i + 1}/{len(y)}", flush=True)
            keep = np.arange(len(y)) != i
            model = self.clone()
            model.fit(X[keep], y[keep])
            mean[i], sigma[i] = (value[0] for value in model.predict(X[i : i + 1]))
            converged.append(getattr(model, "converged", None))
        self.loo_converged = [flag for flag in converged if flag is not None]
        return mean, sigma
