#nowdcl_core.py


import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from scipy.optimize import LinearConstraint, Bounds


@dataclass
class DietSpec:
    foods: list
    cost: np.ndarray  #(n,)
    t: float  #palatability threshold (0..1)

    #Optional constraints:
    A_ineq: Optional[np.ndarray]  #(m, n) for Ax >= b (nutritional minima)
    b_ineq: Optional[np.ndarray]  #(m,)
    A_eq: Optional[np.ndarray]    #(p, n) for Aeq x = beq (salt/sugar fixed)
    b_eq: Optional[np.ndarray]    #(p,)

    lb: np.ndarray  #(n,) lower bounds (>=0)
    ub: np.ndarray  #(n,) upper bounds (use +inf if none)

    degree: int = 2  #surrogate degree
    alpha: float = 1.0  #ridge


#f(x) = cost^T x
def f_maker(cost: np.ndarray):
    def f(x):
        xx = np.asarray(x, dtype=float)[: cost.shape[0]]
        return float(np.dot(cost, xx))
    return f


#g(x) target is y_g = t - label, so g(x) <= 0 means label >= t
def g_from_global(poly_global: PolynomialFeatures, global_model: Ridge, n: int):
    def g(x):
        xx = np.asarray(x, dtype=float)[:n].reshape(1, -1)
        return float(global_model.predict(poly_global.transform(xx))[0])
    return g


#load cost vector
COST_VECTOR = pd.read_excel(
    "/home/ezgi/İndirilenler/processed-data/Syria_instance.xlsx",
    sheet_name="FoodCost",
    index_col="Supplier",
    engine="openpyxl",
).iloc[0, :]

print("cost vector loaded (len={}):".format(len(COST_VECTOR)))


def prepare_from_csv(
    csv_path: str,
    t: float = 0.60,
    cost_vector: Optional[pd.Series] = COST_VECTOR,
    *,
    A_ineq: Optional[np.ndarray] = None,
    b_ineq: Optional[np.ndarray] = None,
    A_eq: Optional[np.ndarray] = None,
    b_eq: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    degree: int = 2,
    alpha: float = 1.0,
    auto_eq: bool = True,
    grams_per_unit: float = 100.0,      
    instance_xlsx: Optional[str] = None, 
    **_ignored,                          
) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    foods = [c for c in df.columns if c != "label"]
    n = len(foods)

    X = df[foods].to_numpy(dtype=float)
    #g-targets: g(x) = t - h(x). g(x) <= 0  <=>  h(x) >= t
    y_g = (t - df["label"].to_numpy(dtype=float))

    print("foods:", foods)

    #costs (need alignment?)
    if cost_vector is None:
        cost = np.ones(n, dtype=float)
    else:
        if not isinstance(cost_vector, pd.Series):
            cost = np.asarray(cost_vector, dtype=float)
        else:
            missing = [f for f in foods if f not in cost_vector.index]
            if missing:
                raise ValueError(f"Cost vector missing foods: {missing}")
            cost = cost_vector.loc[foods].to_numpy(dtype=float)
    if cost.shape != (n,):
        raise ValueError(f"cost must have shape {(n,)}, got {cost.shape}")

    #bounds
    if lb is None:
        lb = np.zeros(n, dtype=float)
    else:
        lb = np.asarray(lb, dtype=float)
    if lb.shape != (n,):
        raise ValueError(f"lb must have shape {(n,)}, got {lb.shape}")

    if ub is None:
        ub = np.full(n, np.inf, dtype=float)
    else:
        ub = np.asarray(ub, dtype=float)
    if ub.shape != (n,):
        raise ValueError(f"ub must have shape {(n,)}, got {ub.shape}")

    #equalities (in 100g-units, consistent?)
    #Salt = 0.05 (5g), Sugar = 0.2 (20g)
    if A_eq is None and b_eq is None:
        rows = []
        rhs = []
        if auto_eq:
            if "Salt" in foods:
                row = np.zeros(n, dtype=float)
                row[foods.index("Salt")] = 1.0
                rows.append(row)
                rhs.append(0.05)
            if "Sugar" in foods:
                row = np.zeros(n, dtype=float)
                row[foods.index("Sugar")] = 1.0
                rows.append(row)
                rhs.append(0.2)
        if rows:
            A_eq = np.vstack(rows)
            b_eq = np.asarray(rhs, dtype=float)
    elif (A_eq is None) != (b_eq is None):
        raise ValueError("Provide both A_eq and b_eq or neither.")

    if A_eq is not None:
        A_eq = np.asarray(A_eq, dtype=float)
        b_eq = np.asarray(b_eq, dtype=float)
        if A_eq.ndim != 2 or A_eq.shape[1] != n:
            raise ValueError(f"A_eq must have shape (p, {n}), got {A_eq.shape}")
        if b_eq.shape != (A_eq.shape[0],):
            raise ValueError(f"b_eq must have shape {(A_eq.shape[0],)}, got {b_eq.shape}")

    print("A_eq:", A_eq)
    print("b_eq:", b_eq)

    #nutritional inequality Ax >= b
    if A_ineq is None and b_ineq is None:
        nutr_val = pd.read_excel(
            "/home/ezgi/İndirilenler/processed-data/Syria_instance.xlsx",
            sheet_name="nutr_val",
            index_col="Food",
            engine="openpyxl",
        )
        missing = [f for f in foods if f not in nutr_val.index]
        if missing:
            raise ValueError(f"nutr_val is missing foods: {missing}")

        nutr_val = nutr_val.loc[foods]  #reorder to match foods
        A_ineq = nutr_val.to_numpy(dtype=float).T  #(m_nutrients, n_foods)

        nutr_req = pd.read_excel(
            "/home/ezgi/İndirilenler/processed-data/Syria_instance.xlsx",
            sheet_name="nutr_req",
            index_col="Type",
            engine="openpyxl",
        )
        num_req = nutr_req.select_dtypes(include=[np.number]).to_numpy(dtype=float)

        if num_req.ndim == 2:
            if num_req.shape[0] == 1 and num_req.shape[1] == A_ineq.shape[0]:
                b_ineq = num_req.reshape(-1)
            elif num_req.shape[0] == A_ineq.shape[0] and num_req.shape[1] == 1:
                b_ineq = num_req.reshape(-1)
            else:
                raise ValueError(
                    f"nutr_req numeric block shape {num_req.shape} "
                    f"does not match number of nutrients {A_ineq.shape[0]}"
                )
        else:
            b_ineq = np.asarray(num_req, dtype=float).reshape(-1)

    elif (A_ineq is None) != (b_ineq is None):
        raise ValueError("Provide both A_ineq and b_ineq or neither.")

    if A_ineq is not None:
        A_ineq = np.asarray(A_ineq, dtype=float)
        b_ineq = np.asarray(b_ineq, dtype=float)
        if A_ineq.ndim != 2 or A_ineq.shape[1] != n:
            raise ValueError(f"A_ineq must have shape (m, {n}), got {A_ineq.shape}")
        if b_ineq.shape != (A_ineq.shape[0],):
            raise ValueError(f"b_ineq must have shape {(A_ineq.shape[0],)}, got {b_ineq.shape}")

    print("A_ineq shape:", None if A_ineq is None else A_ineq.shape)
    print("b_ineq shape:", None if b_ineq is None else b_ineq.shape)

    #fit global surrogate g(x) ≈ t - h(x)
    poly_global = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly_global.fit_transform(X)
    global_model = Ridge(alpha=alpha, fit_intercept=False).fit(X_poly, y_g)

    #build constraints
    extra_constraints = []

    if A_ineq is not None:
        extra_constraints.append(
            LinearConstraint(
                A_ineq,
                lb=b_ineq,
                ub=np.full_like(b_ineq, np.inf, dtype=float),
            )
        )

    if A_eq is not None:
        extra_constraints.append(LinearConstraint(A_eq, lb=b_eq, ub=b_eq))

    bounds = Bounds(lb, ub)

    #initial point x0
    x0 = np.zeros(n, dtype=float)
    if A_eq is not None:
        for r, b in zip(A_eq, b_eq):
            idx = np.where(np.abs(r) > 0)[0]
            if len(idx) == 1:
                x0[idx[0]] = float(b)

    spec = DietSpec(
        foods=foods,
        cost=cost,
        t=float(t),
        A_ineq=A_ineq,
        b_ineq=b_ineq,
        A_eq=A_eq,
        b_eq=b_eq,
        lb=lb,
        ub=ub,
        degree=int(degree),
        alpha=float(alpha),
    )

    print("extra_constraints:", extra_constraints)

    return {
        "X": X,
        "y": y_g,
        "bounds": bounds,
        "x0": x0,
        "n": n,
        "foods": foods,
        "poly_global": poly_global,
        "global_model": global_model,
        "extra_constraints": extra_constraints,
        "f": f_maker(cost),
        "g": g_from_global(poly_global, global_model, n),
        "spec": spec,
    }
