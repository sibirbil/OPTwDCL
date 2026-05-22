# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None


# =========================================================
# Paths
# =========================================================
syria_xlsx_path = Path("../../OptiCL/notebooks/WFP/processed-data/Syria_instance.xlsx")
wfp_csv_path = Path("../../OptiCL/notebooks/WFP/processed-data/WFP_dataset.csv")

concrete_xls_path = Path("Concrete_Data.xls")
concrete_xlsx_path = Path("Concrete_Data.xlsx")
concrete_csv_path = Path("Concrete_Data.csv")


# =========================================================
# Concrete constants
# =========================================================
CONCRETE_FEATURE_COLUMNS = [
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
]

CONCRETE_TARGET_COLUMN = "compressive_strength"

CONCRETE_COMPONENT_COLUMNS = [
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
]

CONCRETE_COST_VECTOR = np.array([
    0.050,  # cement
    0.040,  # blast furnace slag
    0.045,  # fly ash
    0.002,  # water
    1.800,  # superplasticizer
    0.020,  # coarse aggregate
    0.020,  # fine aggregate
    0.000,  # age
], dtype=float)

# Article Table 3 point-estimation RF reference.
# This is only for reporting/comparison, not a constraint.
CONCRETE_ARTICLE_POINT_REFERENCE = {
    "cement": 166.90,
    "blast_furnace_slag": 43.72,
    "fly_ash": 0.00,
    "water": 175.08,
    "superplasticizer": 0.00,
    "coarse_aggregate": 1149.93,
    "fine_aggregate": 694.37,
    "age": 78.00,
    "cost": 47.33,
}


def concrete_component_weight(x) -> float:
    """Total weight of the seven concrete material components, excluding age."""
    x = np.asarray(x, dtype=float).ravel()
    return float(np.sum(x[: len(CONCRETE_COMPONENT_COLUMNS)]))


def concrete_material_cost(x) -> float:
    """Material cost for one concrete solution vector."""
    x = np.asarray(x, dtype=float).ravel()
    return float(CONCRETE_COST_VECTOR @ x[: len(CONCRETE_COST_VECTOR)])


def concrete_solution_dict(x) -> dict:
    """Readable concrete decision dictionary."""
    x = np.asarray(x, dtype=float).ravel()
    return {
        name: float(x[i])
        for i, name in enumerate(CONCRETE_FEATURE_COLUMNS)
    }


# =========================================================
# WFP helpers
# =========================================================
@dataclass
class SyriaInstance:
    nutr_val: pd.DataFrame
    nutr_req: pd.DataFrame
    cost_p: pd.Series


def load_syria_instance(xlsx_path: str | Path) -> SyriaInstance:
    xlsx_path = Path(xlsx_path)

    nutr_val = pd.read_excel(
        xlsx_path,
        sheet_name="nutr_val",
        index_col="Food",
        engine="openpyxl",
    )

    nutr_req = pd.read_excel(
        xlsx_path,
        sheet_name="nutr_req",
        index_col="Type",
        engine="openpyxl",
    )

    cost_p = pd.read_excel(
        xlsx_path,
        sheet_name="FoodCost",
        index_col="Supplier",
        engine="openpyxl",
    ).iloc[0, :]

    return SyriaInstance(
        nutr_val=nutr_val,
        nutr_req=nutr_req,
        cost_p=cost_p,
    )


def load_wfp_dataset(
    csv_path: str | Path,
    *,
    shuffle_seed: int = 0,
) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)

    y = df["label"].astype(float)
    X = df.drop(columns=["label"])

    return X, y


def setup_problem_wfp(
    problem_model_type,
    *,
    shuffle_seed: int = 0,
    pal_req: float = 0.0,
):
    if gp is None or GRB is None:
        raise ImportError(
            "WFP requires gurobipy. Install Gurobi/gurobipy, "
            "or choose problem_name='rosenbrock'."
        )

    if problem_model_type != "gurobi":
        raise NotImplementedError(
            "WFP is currently implemented only for problem_model_type='gurobi'."
        )

    instance = load_syria_instance(syria_xlsx_path)
    X, y = load_wfp_dataset(wfp_csv_path, shuffle_seed=shuffle_seed)

    model = gp.Model()

    foods = list(instance.nutr_val.index)
    nutrients = list(instance.nutr_req.columns)

    A = instance.nutr_val.loc[foods, nutrients].to_numpy(dtype=float)
    req = instance.nutr_req.iloc[0][nutrients].to_numpy(dtype=float)
    cost = instance.cost_p.loc[foods].to_numpy(dtype=float)

    F = len(foods)

    input_vars = model.addMVar(
        shape=F,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="input_vars",
    )

    output_var = model.addMVar(
        shape=1,
        lb=-GRB.INFINITY,
        vtype=GRB.CONTINUOUS,
        name="output_vars",
    )

    z = input_vars

    model.addConstr(output_var[0] >= float(pal_req), name="pal_lb")
    model.addConstr(A.T @ z >= req, name="nutr_req")

    sugar_i = foods.index("Sugar")
    salt_i = foods.index("Salt")

    model.addConstr(z[sugar_i] == 0.2, name="fix_sugar")
    model.addConstr(z[salt_i] == 0.05, name="fix_salt")

    model.setObjective(cost @ z, GRB.MINIMIZE)
    model.update()

    return model, "gurobi", X, y


# =========================================================
# Concrete helpers
# =========================================================
def _find_existing_concrete_data_path(path: str | Path | None = None) -> Path:
    if path is not None:
        p = Path(path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Concrete data file not found: {p}")

    candidates = []

    try:
        here = Path(__file__).resolve().parent
        candidates.extend([
            here / "Concrete_Data.xls",
            here / "Concrete_Data.xlsx",
            here / "Concrete_Data.csv",
        ])
    except NameError:
        pass

    candidates.extend([
        Path.cwd() / "Concrete_Data.xls",
        Path.cwd() / "Concrete_Data.xlsx",
        Path.cwd() / "Concrete_Data.csv",
        concrete_xls_path,
        concrete_xlsx_path,
        concrete_csv_path,
    ])

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not find Concrete_Data.xls, Concrete_Data.xlsx, or Concrete_Data.csv. "
        "Put the data file in the same folder as dcl_problem.py or in the current working directory."
    )


def _standardize_concrete_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    compact_names = CONCRETE_FEATURE_COLUMNS + [CONCRETE_TARGET_COLUMN]

    if len(df.columns) >= 9:
        df = df.iloc[:, :9]
        df.columns = compact_names
        return df

    raise ValueError(
        "Concrete dataset must have at least 9 columns: "
        "8 feature columns and 1 compressive-strength target column."
    )


def load_concrete_dataset(
    path: str | Path | None = None,
    *,
    shuffle_seed: int = 0,
) -> Tuple[pd.DataFrame, pd.Series]:
    data_path = _find_existing_concrete_data_path(path)
    suffix = data_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(data_path)

    elif suffix == ".xlsx":
        df = pd.read_excel(data_path, engine="openpyxl")

    elif suffix == ".xls":
        try:
            df = pd.read_excel(data_path, engine="xlrd")
        except ImportError as exc:
            raise ImportError(
                "Reading Concrete_Data.xls requires xlrd. "
                "Install it with: conda install -c conda-forge xlrd "
                "or: pip install xlrd. "
                "Alternative: convert Concrete_Data.xls to Concrete_Data.xlsx or Concrete_Data.csv."
            ) from exc

    else:
        raise ValueError(
            f"Unsupported concrete data extension '{suffix}'. "
            "Use .xls, .xlsx, or .csv."
        )

    df = _standardize_concrete_columns(df)

    for col in CONCRETE_FEATURE_COLUMNS + [CONCRETE_TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=CONCRETE_FEATURE_COLUMNS + [CONCRETE_TARGET_COLUMN])
    df = df.reset_index(drop=True)
    df = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)

    X = df[CONCRETE_FEATURE_COLUMNS].astype(float)
    y = df[CONCRETE_TARGET_COLUMN].astype(float)

    return X, y


def setup_problem_concrete(
    problem_model_type,
    *,
    shuffle_seed: int = 0,
    strength_req: float = 45.0,
    weight_lb: float = 2230.0,
    weight_ub: float = 2450.0,
    data_path: str | Path | None = None,
):
    """
    Concrete point-estimation setup.

    This is deliberately WFP-style:
        learned_strength = ML_model(concrete variables)
        learned_strength >= strength_req

    No quantile, no superquantile, no chance-constraint reformulation.
    """
    if gp is None or GRB is None:
        raise ImportError(
            "Concrete requires gurobipy. Install Gurobi/gurobipy, "
            "or choose problem_name='rosenbrock'."
        )

    if problem_model_type != "gurobi":
        raise NotImplementedError(
            "Concrete is currently implemented only for problem_model_type='gurobi'."
        )

    X, y = load_concrete_dataset(
        data_path,
        shuffle_seed=shuffle_seed,
    )

    model = gp.Model()

    n = len(CONCRETE_FEATURE_COLUMNS)

    lb = X.min(axis=0).to_numpy(dtype=float)
    ub = X.max(axis=0).to_numpy(dtype=float)

    input_vars = model.addMVar(
        shape=n,
        lb=lb,
        ub=ub,
        vtype=GRB.CONTINUOUS,
        name="input_vars",
    )

    output_var = model.addMVar(
        shape=1,
        lb=-GRB.INFINITY,
        vtype=GRB.CONTINUOUS,
        name="output_vars",
    )

    # WFP-style learned-output constraint:
    # WFP:      predicted palatability >= pal_req
    # Concrete: predicted compressive strength >= 45 MPa
    model.addConstr(
        output_var[0] >= float(strength_req),
        name="strength_lb",
    )

    # Article component-weight constraint. Age is excluded.
    component_weight = gp.quicksum(input_vars[i] for i in range(7))

    model.addConstr(
        component_weight >= float(weight_lb),
        name="component_weight_lb",
    )

    model.addConstr(
        component_weight <= float(weight_ub),
        name="component_weight_ub",
    )

    cost_expr = gp.quicksum(
        float(CONCRETE_COST_VECTOR[i]) * input_vars[i]
        for i in range(n)
    )

    model.setObjective(cost_expr, GRB.MINIMIZE)
    model.update()

    return model, "gurobi", X, y


# =========================================================
# Rosenbrock helpers
# =========================================================
def rosenbrock_constraint_value(x):
    """
    g(x) = -x0 + 100*(x2 - x1^2)^2 + (1 - x1)^2
    Feasible region is g(x) <= 0.
    """
    x = np.asarray(x, dtype=float)

    return (
        -x[0]
        + 100.0 * (x[2] - x[1] ** 2) ** 2
        + (1.0 - x[1]) ** 2
    )


def generate_rosenbrock_dataset(seed=102, nsamples=100, noise_std=0.0):
    rng = np.random.RandomState(seed)

    X = 2.0 * rng.randn(nsamples, 3)

    for i in range(nsamples):
        x1 = X[i, 1]
        x2 = X[i, 2]
        valley = 100.0 * (x2 - x1 * x1) ** 2 + (1.0 - x1) ** 2
        #X[i, 0] = rng.randn(1)[0] + valley
        X[i, 0] = rng.randn(1)[0] * valley + rng.randn(1)[0]
    
    noise_std=1.0
    noise = noise_std * rng.randn(nsamples)
    
    y = np.array(
        [rosenbrock_constraint_value(X[i, :]) for i in range(nsamples)],
        dtype=float,
    ) + noise

    Xdf = pd.DataFrame(X, columns=["x0", "x1", "x2"])
    ysr = pd.Series(y, name="label", dtype=float)

    return Xdf, ysr


def setup_problem_rosenbrock(seed=102, nsamples=100, noise_std=0.0):
    X, y = generate_rosenbrock_dataset(
        seed=seed,
        nsamples=nsamples,
        noise_std=noise_std,
    )

    problem = {
        "name": "rosenbrock",
        "solver": "scipy",
        "objective": lambda x: float(x[0]),
        "bounds_lb": np.array([-np.inf, -5.0, -5.0], dtype=float),
        "bounds_ub": np.array([np.inf, 5.0, 5.0], dtype=float),
        "true_constraint": rosenbrock_constraint_value,
    }

    return problem, "scipy", X, y


# =========================================================
# Problem dispatcher
# =========================================================
def dcl_setup_problem(
    problem_name,
    learning_model_type,
    adjustable_problem_parameters,
    seed,
):
    if problem_name == "wfp":
        if learning_model_type not in ("cart", "nn"):
            raise NotImplementedError(
                "WFP currently supports learning_model_type='cart' or 'nn' only."
            )

        return setup_problem_wfp(
            "gurobi",
            shuffle_seed=seed,
            pal_req=adjustable_problem_parameters,
        )

    elif problem_name == "rosenbrock":
        return setup_problem_rosenbrock(
            seed=seed,
            nsamples=100,
            noise_std=0.0,
        )

    elif problem_name == "concrete":
        if learning_model_type not in ("rf", "nn"):
            raise NotImplementedError(
                "Concrete point-estimation setup currently supports "
                "learning_model_type='rf' or learning_model_type='nn'."
            )

        return setup_problem_concrete(
            "gurobi",
            shuffle_seed=seed,
            strength_req=float(adjustable_problem_parameters),
        )

    else:
        raise ValueError(
            f"Unknown problem_name='{problem_name}'. "
            f"Supported values are 'wfp', 'rosenbrock', and 'concrete'."
        )