"""Surrogate models. See `base.py` for the contract they satisfy.

`mixed_gp.py` is the regression GP. `atlas_classifier.py` is the
variational Bernoulli GP Atlas uses for unknown feasibility constraints.

Which of the two owns which response is decided here, once, so the scores, the
representation ablation and the input-ablation analysis all describe the same
model.
"""

# The mixed Hamming-plus-Matern kernel rather than the single Matern Atlas
# ships with, so the classifier sees excipient identity the way the regressor does.
FEASIBILITY_KERNEL = "mixed"


def surrogate_for(kind, cat_dims, seed=42, kernel=None, **kwargs):
    """The model class a response's likelihood calls for.

    `kernel` overrides the classifier's kernel; "matern" is Atlas as shipped.

    Imports are inside the function to keep `import models` from pulling torch
    into processes that only want the constant above.
    """
    if kind == "binary":
        from models.atlas_classifier import AtlasClassifier

        return AtlasClassifier(cat_dims, kernel=kernel or FEASIBILITY_KERNEL,
                               seed=seed, **kwargs)
    from models.mixed_gp import MixedGP

    return MixedGP(cat_dims, seed=seed, **kwargs)
