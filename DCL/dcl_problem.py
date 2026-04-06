# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from pathlib import Path
from typing import Tuple
from dataclasses import dataclass

# This file contains procedures for setting up problems to be solved by dcl

syria_xlsx_path = Path("../../OptiCL/notebooks/WFP/processed-data/Syria_instance.xlsx")
wfp_csv_path = Path("../../OptiCL/notebooks/WFP/processed-data/WFP_dataset.csv")


# =========================================================
# WFP helpers
# =========================================================
@dataclass
class SyriaInstance:
    nutr_val: pd.DataFrame   # index=Food, columns=nutrients
    nutr_req: pd.DataFrame   # index=Type, columns=nutrients
    cost_p: pd.Series        # index=Food


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

    return SyriaInstance(nutr_val=nutr_val, nutr_req=nutr_req, cost_p=cost_p)


def load_wfp_dataset(csv_path: str | Path, *, shuffle_seed: int = 0) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
    y = df["label"].astype(float)
    X = df.drop(columns=["label"])
    return X, y


# In this routine we add regular constraints, bounds, etc. all components of the
# optimization problem except the learned constraint.
# We shuffle the data set here before training.
def setup_problem_wfp(
    problem_model_type,
    *,
    shuffle_seed: int = 0,
    pal_req: float = 0.0,
):
    instance = load_syria_instance(syria_xlsx_path)
    X, y = load_wfp_dataset(wfp_csv_path, shuffle_seed=shuffle_seed)

    if problem_model_type != "gurobi":
        raise NotImplementedError("WFP is currently implemented only for problem_model_type='gurobi'.")

    model = gp.Model()

    foods = list(instance.nutr_val.index)
    nutrients = list(instance.nutr_req.columns)

    A = instance.nutr_val.loc[foods, nutrients].to_numpy(dtype=float)
    req = instance.nutr_req.iloc[0][nutrients].to_numpy(dtype=float)
    cost = instance.cost_p.loc[foods].to_numpy(dtype=float)

    F = len(foods)

    # change notation to x = [z; output_var], output_var = g(z)
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

    # learned output must satisfy palatability lower bound
    model.addConstr(output_var[0] >= float(pal_req), name="pal_lb")

    # nutrition constraints
    model.addConstr(A.T @ z >= req, name="nutr_req")

    # fixed ingredients
    sugar_i = foods.index("Sugar")
    salt_i = foods.index("Salt")
    model.addConstr(z[sugar_i] == 0.2, name="fix_sugar")
    model.addConstr(z[salt_i] == 0.05, name="fix_salt")

    model.setObjective(cost @ z, GRB.MINIMIZE)
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
    return -x[0] + 100.0 * (x[2] - x[1] ** 2) ** 2 + (1.0 - x[1]) ** 2


def generate_rosenbrock_dataset(seed=102, nsamples=100, noise_std=0.0):
    rng = np.random.default_rng(seed)

    X = 2.0 * rng.standard_normal((nsamples, 3))

    for i in range(nsamples):
        x1 = X[i, 1]
        x2 = X[i, 2]
        valley = 100.0 * (x2 - x1 * x1) ** 2 + (1.0 - x1) ** 2
        X[i, 0] = rng.standard_normal() + valley

    noise = noise_std * rng.standard_normal(nsamples)
    y = np.array([rosenbrock_constraint_value(X[i, :]) for i in range(nsamples)], dtype=float) + noise

    Xdf = pd.DataFrame(X, columns=["x0", "x1", "x2"])
    ysr = pd.Series(y, name="label", dtype=float)
    return Xdf, ysr


def setup_problem_rosenbrock(seed=102, nsamples=100, noise_std=0.0):
    """
    Rosenbrock setup for SciPy local subproblems.

    Optimization model:
        min x0
        s.t. learned_g(x0, x1, x2) <= 0
             x1 in [-5, 5]
             x2 in [-5, 5]
    """
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
        "bounds_ub": np.array([ np.inf,  5.0,  5.0], dtype=float),
    }

    return problem, "scipy", X, y


# =========================================================
# setup
# =========================================================
def dcl_setup_problem(problem_name, learning_model_type, adjustable_problem_parameters, seed):
    # read problem data and put into input and output data structures
    # construct the objective / additional constraints of the problem

    if problem_name == "wfp":
        # Keep WFP as Gurobi-based in the current framework.
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

    else:
        raise ValueError(
            f"Unknown problem_name='{problem_name}'. "
            f"Supported values are 'wfp' and 'rosenbrock'."
        )