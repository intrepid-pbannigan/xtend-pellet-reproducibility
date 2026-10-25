"""The X-TEND design space in the campaign's slot-specific encoding.

This is representation A of the manuscript's encoding ablation (Section 2.5,
Figure 6): one column block per formulation role, a one-hot saying which
excipient fills the role and a continuous w/w fraction saying how much of it
there is. The role-independent representations B and C are built in
`retrospective/features.py`.

Wet granulation and spheronization were varied continuously, so they are design
variables in exactly the sense composition is and enter as a second block.

What does not enter X: batch index, experiment number, anything measured after
the batch was made (size, aspect ratio, release), and molecular descriptors.
Excipient identity is a Hamming one-hot, so every polymer pair is equidistant.

Two things this layout has to get right that the campaign spreadsheet does not.

Absence is a category. Five experiments use a single control polymer, recorded as
a relative fraction of 0 or 1 while the type column still names a material -- so
experiment 17 reads `kolidon_sr` at zero w/w. Left alone the one-hot would assert
a polymer that is not in the pellet. The empty role is mapped to `"none"`, which
is a level of the vocabulary like any other.

Vocabularies are frozen from all retained experiments, not from the fold being
fitted, so the column count cannot change between leave-one-out folds. A fold
that happens to hold out the only experiment using CMC still has a CMC column;
it is simply all zeros in the training rows, which is why `MixedGP.fit`
tolerates constant columns instead of dividing by a zero range.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg

ABSENT = "none"


@dataclass(frozen=True)
class Role:
    """One formulation role: which material fills it, and how much.

    `type_column` is None for a role whose material never varies. The API is the
    only one here -- every experiment is ILX006 -- so it contributes a fraction
    and no one-hot.
    """

    name: str
    fraction: str
    type_column: str | None = None
    vocabulary: tuple[str, ...] = ()


@dataclass
class DesignSpace:
    """Column layout, categorical indices, and the frozen vocabularies."""

    columns: list[str] = field(default_factory=list)
    cat_dims: list[int] = field(default_factory=list)
    roles: list[Role] = field(default_factory=list)
    blocks: tuple[str, ...] = ()

    @property
    def n_continuous(self):
        return len(self.columns) - len(self.cat_dims)

    def describe(self):
        return (f"{len(self.columns)} columns, {len(self.cat_dims)} categorical, "
                f"{self.n_continuous} continuous, blocks {'+'.join(self.blocks)}")


# Composition. Absolute w/w fractions per polymer rather than the recorded
# total-and-relative pair: they carry the same information and the per-role
# "how much" is what the kernel is meant to see. The total is their sum and so
# is not a separate column, and filler is 1 minus everything else -- the balance
# term -- so it is determined and stays out.
COMPOSITION_FRACTIONS = ("drug_ww_frac", "cntrl_polymer_0_ww_frac",
                         "cntrl_polymer_1_ww_frac", "mg_stearate_ww_frac")

PROCESS_CONTINUOUS = ("gran_time", "gran_rpm", "spheronization_time",
                      "spheronization_rpm", "fluid_vol")

PROCESS_CATEGORICAL = ("fluid_type",)

ALL_BLOCKS = ("composition", "process")


def polymer_fractions(df):
    """Absolute w/w fraction of each control polymer, and the empty role's name.

    The spreadsheet stores a total and polymer 0's relative share of it. Splitting
    that into two absolute fractions is what lets an empty role be recognised: a
    zero fraction is the signal that the type column is naming a material the
    pellet does not contain.
    """
    total = df.total_cntrl_polymer_ww_frac.to_numpy(float)
    relative = df.cntrl_polymer_0_relative_ww_frac.to_numpy(float)
    return total * relative, total * (1.0 - relative)


def material(types, fractions):
    """Excipient names with empty roles replaced by an explicit absent level."""
    names = pd.Series(types).astype(str).to_numpy(dtype=object).copy()
    names[np.asarray(fractions, float) <= 0.0] = ABSENT
    return names


def prepare(df):
    """Campaign rows widened to the quantities the design space is built from."""
    frame = df.copy()
    first, second = polymer_fractions(frame)
    frame["cntrl_polymer_0_ww_frac"] = first
    frame["cntrl_polymer_1_ww_frac"] = second
    frame["cntrl_polymer_0_material"] = material(frame.cntrl_polymer_0_type, first)
    frame["cntrl_polymer_1_material"] = material(frame.cntrl_polymer_1_type, second)
    return frame


def roles_for(frame, blocks):
    """Roles with vocabularies frozen from every row of `frame`.

    Sorted so the column order is a function of the vocabulary and not of the
    order experiments happen to appear in, which keeps a rebuilt matrix
    comparable to a stored one.
    """
    roles = []
    if "composition" in blocks:
        roles.append(Role("api", "drug_ww_frac"))
        for index in (0, 1):
            column = f"cntrl_polymer_{index}_material"
            roles.append(Role(
                f"control_polymer_{index}",
                f"cntrl_polymer_{index}_ww_frac",
                column,
                tuple(sorted(set(frame[column]))),
            ))
        roles.append(Role("lubricant", "mg_stearate_ww_frac"))
    return roles


def build_matrix(df, blocks=ALL_BLOCKS, space=None):
    """The design matrix, its column names and its categorical indices.

    Categorical indices are counted while the columns are laid out, rather than
    inferred afterwards from the frame, because inferring them is where the
    upstream helper goes wrong: it does not advance past a categorical parameter
    and so duplicates or shifts indices once there is more than one. Here the
    index is whatever `len(columns)` was when the column was appended, which
    cannot drift.

    Passing a `space` reuses an existing layout, so a prediction matrix is
    guaranteed the same columns in the same order as the matrix it was fitted on.
    """
    frame = prepare(df)
    roles = space.roles if space is not None else roles_for(frame, blocks)
    blocks = space.blocks if space is not None else tuple(
        b for b in ALL_BLOCKS if b in blocks)

    columns, cat_dims, data = [], [], []

    def add(name, values, categorical=False):
        if categorical:
            cat_dims.append(len(columns))
        columns.append(name)
        data.append(np.asarray(values, float))

    for role in roles:
        if role.type_column is not None:
            observed = frame[role.type_column].to_numpy(dtype=object)
            for level in role.vocabulary:
                add(f"{role.name}={level}", observed == level, categorical=True)
        add(role.fraction, frame[role.fraction].to_numpy(float))

    if "process" in blocks:
        for column in PROCESS_CONTINUOUS:
            add(column, frame[column].to_numpy(float))
        for column in PROCESS_CATEGORICAL:
            vocabulary = (space_vocabulary(space, column) if space is not None
                          else tuple(sorted(set(frame[column].astype(str)))))
            observed = frame[column].astype(str).to_numpy(dtype=object)
            for level in vocabulary:
                add(f"{column}={level}", observed == level, categorical=True)

    X = np.column_stack(data) if data else np.empty((len(frame), 0))
    built = DesignSpace(columns=columns, cat_dims=cat_dims, roles=roles, blocks=blocks)
    return X, (space if space is not None else built)


def space_vocabulary(space, column):
    """Levels of a process categorical recovered from an existing layout."""
    prefix = f"{column}="
    return tuple(name[len(prefix):] for name in space.columns
                 if name.startswith(prefix))


def design_space(df, blocks=ALL_BLOCKS):
    """The layout alone, frozen from every row of `df`."""
    return build_matrix(df, blocks)[1]


def main():
    """Print the layout, which is the thing most worth checking by eye."""
    df = pd.read_csv(cfg.CAMPAIGN_CSV)
    for blocks in [("composition",), ALL_BLOCKS]:
        X, space = build_matrix(df, blocks)
        print(f"blocks {'+'.join(blocks)}: {space.describe()}")
        for index, name in enumerate(space.columns):
            kind = "categorical" if index in space.cat_dims else "continuous"
            values = X[:, index]
            print(f"  {index:>2}. {name:<40} {kind:<12} "
                  f"{values.min():.4g} to {values.max():.4g}")
        print()

    frame = prepare(df)
    empty = frame[(frame.cntrl_polymer_0_material == ABSENT)
                  | (frame.cntrl_polymer_1_material == ABSENT)]
    print(f"{len(empty)} experiments have an empty polymer role, mapped to "
          f"{ABSENT!r} rather than left naming a material at zero w/w:")
    for _, row in empty.iterrows():
        print(f"  experiment {int(row.experiment_num):>2}: "
              f"recorded {row.cntrl_polymer_0_type} / {row.cntrl_polymer_1_type}"
              f" -> {row.cntrl_polymer_0_material} / {row.cntrl_polymer_1_material}")


if __name__ == "__main__":
    main()
