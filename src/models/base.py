"""The contract every surrogate in this project satisfies.

Two methods, because that is all the cross-validation harness and the response
figures need, and keeping the surface this small is what lets the GP and the
penalised linear baseline be swapped for one another without the calling code
knowing which it has.

    fit(X, y)      -> self, having seen only these rows
    predict(X)     -> (mean, sigma), both in the target's original units

`sigma` is the standard deviation of the predictive distribution over a measured
outcome, not over the latent function. The distinction matters for anything that
checks calibration: a model can be well calibrated about where the response
surface lies and still be overconfident about what the next assay will read.

The important half of the contract is not the signature but where preprocessing
happens. Every scaling constant a model needs must be derived inside `fit`, from
the rows `fit` was given. Nothing may be fitted on the full dataset and reused
across folds. That is the only reason a leave-one-out number from this harness
can be read as out of sample.
"""

import numpy as np


class Surrogate:
    """Base class asserting the fit/predict contract, and nothing else."""

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_epistemic(self, X):
        """Mean, and the uncertainty about the mean rather than about a draw.

        These coincide for a Gaussian surrogate and are reported as such: its
        predictive sd is what a reader should compare a partial-dependence span
        against, since a span smaller than the model's own uncertainty at any
        single point has not been shown to be anything. A Bernoulli surrogate
        has to override this, because the sd it reports for scoring is the
        spread of the coin and not of the model's belief about it.
        """
        return self.predict(X)

    def fit_predict_loo(self, X, y):
        """Leave-one-out predictions, refitting from scratch for every held-out row.

        Refitting rather than using a closed-form leave-one-out identity, because
        the GP's preprocessing constants and its marginal-likelihood optimum both
        depend on which rows it saw, and a shortcut that reuses either of them
        across folds would quietly stop being out of sample.
        """
        X, y = np.asarray(X, float), np.asarray(y, float)
        mean = np.empty(len(y))
        sigma = np.empty(len(y))
        converged = []
        for i in range(len(y)):
            keep = np.arange(len(y)) != i
            model = self.clone()
            model.fit(X[keep], y[keep])
            mean[i], sigma[i] = (value[0] for value in model.predict(X[i : i + 1]))
            converged.append(getattr(model, "converged", None))
        # Carried on the harness rather than returned, so the contract stays two
        # arrays while the diagnostic is still available to whoever wants it.
        self.loo_converged = [flag for flag in converged if flag is not None]
        return mean, sigma

    @property
    def converged_fraction(self):
        """Share of leave-one-out fits whose optimiser reported success."""
        flags = getattr(self, "loo_converged", [])
        return float(np.mean(flags)) if flags else float("nan")

    def clone(self):
        raise NotImplementedError
