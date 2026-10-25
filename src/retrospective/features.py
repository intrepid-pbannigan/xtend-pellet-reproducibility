"""Formulation representations for the retrospective surrogates.

    E0  representation A: the campaign's slot-specific one-hot (design_space.py)
    E1  representation B: role-independent w/w of each of the 10 materials
    E5  representation C: B plus four wet-mass descriptors
    E3  mass-weighted physicochemical descriptors (the feasibility model)

E2, E4 and E6 were part of the exploratory comparison (Section 2.5 reports
about 120 model x representation x validation combinations) and are kept so
that comparison can be rerun.

Every set except E0 is order-invariant: a material contributes the same columns
whichever role the campaign recorded it in, and an empty role contributes
nothing. E0 is the campaign's layout.

Every set carries the same non-polymer block -- drug and Mg stearate w/w and the
five continuous process settings -- so the sets differ only in how polymers (and,
for E4, the fluid) are described. The granulating fluid enters as a one-hot
everywhere except E4. With three fluids, any fluid descriptor is an injective
recoding of that one-hot, so it can change a model's inductive bias but cannot
add information; E4 uses the descriptors so that a model with no identity
columns at all is still well defined.

Fluid volume is mL per batch at a fixed batch size, and the masses are per 500 mg
formula, so the liquid ratios are proportional to the physical ones rather than
equal to them. Proportional is all a scaled model needs.

    python src/retrospective/features.py   # runs the checks
"""

import numpy as np
import pandas as pd

from common import FLUIDS_CSV, campaign, cfg

import design_space as ds

BASE = ["drug_ww_frac", "mg_stearate_ww_frac", "gran_time", "gran_rpm",
        "spheronization_time", "spheronization_rpm", "fluid_vol"]
SETS = ("E0", "E1", "E2", "E3", "E4", "E5", "E6")
WATER_CLASSES = ("swelling_insoluble", "soluble_gel", "insoluble_lipid",
                 "insoluble_matrix", "soluble")


def materials_table():
    """Descriptors per canonical material, from the per-grade reference table."""
    table = pd.read_csv(cfg.EXCIPIENT_DESCRIPTORS_CSV)
    table["material"] = table.excipient.replace(cfg.MATERIAL_ALIASES)

    def log_viscosity(text):
        if not isinstance(text, str):
            return np.nan
        low, high = (float(v) for v in text.split("-"))
        return np.log10(np.sqrt(low * high))

    def transition(text):
        try:
            low, *rest = str(text).split("-")
            return np.mean([float(low)] + [float(r) for r in rest])
        except ValueError:
            return np.nan

    table["log_viscosity"] = table.viscosity_cp.map(log_viscosity)
    table["transition_c"] = table.thermal_transition_c.map(transition)
    columns = ["functional_class", "water_interaction", "ionic_character", "hlb",
               "log_viscosity", "transition_c"]
    merged = table.groupby("material")[columns].first()
    for column in ("functional_class", "water_interaction", "ionic_character"):
        disagree = table.groupby("material")[column].nunique() > 1
        if disagree.any():
            raise ValueError(f"merged grades disagree on {column}: {list(disagree[disagree].index)}")
    return merged


def material_fractions(df):
    """w/w of each canonical material, summed over the two roles."""
    frame = ds.prepare(df)
    names = sorted((set(frame.cntrl_polymer_0_material) | set(frame.cntrl_polymer_1_material))
                   - {ds.ABSENT})
    out = pd.DataFrame(0.0, index=df.index, columns=names)
    for index in (0, 1):
        for name in names:
            hit = frame[f"cntrl_polymer_{index}_material"].to_numpy() == name
            out.loc[hit, name] += frame[f"cntrl_polymer_{index}_ww_frac"].to_numpy(float)[hit]
    return out


def granular_fraction(df):
    """w/w of polymer recorded as the granular 71G grade."""
    frame = ds.prepare(df)
    total = np.zeros(len(df))
    for index in (0, 1):
        grade = df[f"cntrl_polymer_{index}_grade"].to_numpy()
        total += np.where(grade == "carbopol_71g",
                          frame[f"cntrl_polymer_{index}_ww_frac"].to_numpy(float), 0.0)
    return total


def class_block(fractions, table):
    out = pd.DataFrame(index=fractions.index)
    for kind, label in (("gel", "gel_ww"), ("barrier", "barrier_ww"),
                        ("soluble_wetting", "soluble_ww")):
        members = [m for m in fractions if table.loc[m, "functional_class"] == kind]
        out[label] = fractions[members].sum(axis=1)
    out["complementarity"] = out.gel_ww * out.barrier_ww
    return out


def physchem_block(df, fractions, table):
    """Mass-weighted descriptors: each is sum over materials of w/w x property."""
    out = pd.DataFrame(index=fractions.index)
    for kind in WATER_CLASSES:
        members = [m for m in fractions if table.loc[m, "water_interaction"] == kind]
        out[f"{kind}_ww"] = fractions[members].sum(axis=1)

    def weighted(column, scale=1.0):
        values = table.loc[fractions.columns, column].fillna(0.0).to_numpy(float) / scale
        return fractions.to_numpy(float) @ values

    out["gel_log_viscosity_load"] = weighted("log_viscosity")
    out["hlb_load"] = weighted("hlb", scale=10.0)
    out["transition_load"] = weighted("transition_c", scale=100.0)
    for charge in ("anionic", "cationic"):
        members = [m for m in fractions if table.loc[m, "ionic_character"] == charge]
        out[f"{charge}_ww"] = fractions[members].sum(axis=1)
    out["granular_ww"] = granular_fraction(df)
    return out


def physics_block(df, fractions, table):
    """Wet-granulation quantities a formulator would compute for the wet mass.

    The filler is all microcrystalline cellulose (`config.FILLER_IS_CELLULOSE_ONLY`),
    so MCC is the whole filler and the lactose term of the soluble solids is zero.
    """
    out = pd.DataFrame(index=df.index)
    mcc = df.cellulose_mass / df.total_mass_mg
    lactose = df.lactose_mass / df.total_mass_mg
    soluble_polymer = [m for m in fractions if table.loc[m, "water_interaction"] == "soluble"]
    lipid = [m for m in fractions if table.loc[m, "water_interaction"] == "insoluble_lipid"]
    out["fluid_per_mcc"] = df.fluid_vol / mcc
    out["soluble_solids_frac"] = df.drug_ww_frac + lactose + fractions[soluble_polymer].sum(axis=1)
    out["hydrophobic_frac"] = fractions[lipid].sum(axis=1) + df.mg_stearate_ww_frac
    fluids = pd.read_csv(FLUIDS_CSV).set_index("fluid_type")
    binder = df.fluid_type.map(fluids.binder_w_w_pct).to_numpy(float) / 100.0
    out["binder_delivered"] = binder * df.fluid_vol
    return out, fluids


def fluid_onehot(df):
    levels = sorted(set(df.fluid_type))
    return pd.DataFrame({f"fluid={lvl}": (df.fluid_type == lvl).astype(float) for lvl in levels},
                        index=df.index)


def build(df, name, extra=None):
    """Return X, column names and categorical column indices for one set.

    `extra` is a dict of additional columns (e.g. an era indicator), appended as
    continuous inputs.
    """
    if name == "E0":
        X, space = ds.build_matrix(df, ds.ALL_BLOCKS)
        columns, cat_dims = list(space.columns), list(space.cat_dims)
        if extra:
            for key, values in extra.items():
                X = np.column_stack([X, np.asarray(values, float)])
                columns.append(key)
        return X, columns, cat_dims

    table = materials_table()
    fractions = material_fractions(df)
    parts = [df[BASE].astype(float)]
    if name in ("E1", "E5"):
        parts.append(fractions.add_prefix("ww="))
    if name == "E2":
        parts.append(class_block(fractions, table))
    if name in ("E3", "E6"):
        parts.append(physchem_block(df, fractions, table))
    if name in ("E4", "E5", "E6"):
        physics, fluids = physics_block(df, fractions, table)
        parts.append(physics)
    if name == "E4":
        for column in ("ethanol_frac", "surface_tension_mN_m", "viscosity_mPa_s"):
            parts.append(df.fluid_type.map(fluids[column]).astype(float).rename(f"fluid_{column}"))
    if extra:
        parts.append(pd.DataFrame(extra, index=df.index))
    frame = pd.concat(parts, axis=1)
    cat_dims = []
    if name != "E4":
        onehot = fluid_onehot(df)
        cat_dims = list(range(frame.shape[1], frame.shape[1] + onehot.shape[1]))
        frame = pd.concat([frame, onehot], axis=1)
    if frame.isna().any().any():
        raise ValueError(f"{name} has missing values in {list(frame.columns[frame.isna().any()])}")
    return frame.to_numpy(float), list(frame.columns), cat_dims


def check():
    df = campaign()
    fractions = material_fractions(df)
    total = df.total_cntrl_polymer_ww_frac.to_numpy(float)
    assert np.allclose(fractions.sum(axis=1), total), "material w/w must sum to total polymer"
    assert len(fractions.columns) == 10, fractions.columns

    swapped = df.copy()
    for a, b in (("cntrl_polymer_0_type", "cntrl_polymer_1_type"),
                 ("cntrl_polymer_0_grade", "cntrl_polymer_1_grade")):
        swapped[a], swapped[b] = df[b], df[a]
    swapped["cntrl_polymer_0_relative_ww_frac"] = 1 - df.cntrl_polymer_0_relative_ww_frac
    for name in SETS[1:]:
        X, columns, cat_dims = build(df, name)
        Xs, _, _ = build(swapped, name)
        assert np.allclose(X, Xs), f"{name} is not order-invariant"
        assert X.shape[0] == len(df)
        print(f"{name}: {X.shape[1]:>2} columns ({len(cat_dims)} categorical): "
              f"{', '.join(c for c in columns if c not in BASE)}")
    X0, columns0, cat0 = build(df, "E0")
    Xref, space = ds.build_matrix(df, ds.ALL_BLOCKS)
    assert np.array_equal(X0, Xref) and cat0 == list(space.cat_dims)
    print(f"E0: {X0.shape[1]} columns ({len(cat0)} categorical), identical to design_space.py")
    winner = df.experiment_num == 49
    phys = physchem_block(df, fractions, materials_table())[winner].round(4)
    print("experiment 49, physicochemical block:", phys.to_dict("records")[0])
    print("checks passed: material w/w sums, order invariance for E1-E6, E0 matches design_space.py")


if __name__ == "__main__":
    check()
