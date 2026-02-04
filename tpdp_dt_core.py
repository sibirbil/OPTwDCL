#dt_core.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from scipy.optimize import linprog


_PYOMO_AVAILABLE = True
try:
    from pyomo.environ import (
        ConcreteModel, Var, Objective, Constraint,
        Binary, Reals, NonNegativeReals, minimize,
        SolverFactory, value,
    )
except Exception:
    _PYOMO_AVAILABLE = False


#data loading (Syria instance + WFP dataset)

@dataclass
class SyriaInstance:
    nutr_val: pd.DataFrame   #index=Food, columns=nutrients
    nutr_req: pd.DataFrame   #index=Type, columns=nutrients
    cost_p: pd.Series        #index=Food


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


#CART training

@dataclass
class TrainedTree:
    model: DecisionTreeRegressor
    features: List[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def train_cart_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.8,
    split_seed: int = 42,
    max_depth: int = 4,
    tree_seed: int = 0,
) -> TrainedTree:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=split_seed,
    )

    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=tree_seed)
    tree.fit(X_train, y_train)

    return TrainedTree(
        model=tree,
        features=list(X.columns),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


#tree -> constraint table extraction (leaves)

def _all_leaf_nodes(tree: DecisionTreeRegressor) -> np.ndarray:
    children_left = tree.tree_.children_left
    return np.where(children_left == -1)[0]


def _path_to_leaf(tree: DecisionTreeRegressor, leaf: int) -> List[Tuple[int, int]]:
    """
    Return a root-to-leaf path as list of (node, direction).
      direction=0 => took left branch at node
      direction=1 => took right branch at node
    """
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right

    parent: Dict[int, Tuple[int, int]] = {0: (-1, -1)}
    stack = [0]
    while stack:
        node = stack.pop()
        l, r = children_left[node], children_right[node]
        if l != -1:
            parent[l] = (node, 0)
            stack.append(l)
        if r != -1:
            parent[r] = (node, 1)
            stack.append(r)

    if leaf not in parent:
        raise ValueError(f"Leaf {leaf} not reachable from root (unexpected)")

    path: List[Tuple[int, int]] = []
    cur = leaf
    while cur != 0:
        p, direction = parent[cur]
        path.append((p, direction))
        cur = p
    path.reverse()
    return path


def extract_tree_table(
    tree: DecisionTreeRegressor,
    feature_names: List[str],
    *,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """
    Build a constraint table with columns:
      ID, <feature columns>, threshold, prediction

    Each row is one split inequality on the path to a leaf.
    For a split (feature f, threshold t):
      left branch:   x_f <= t
      right branch:  x_f >= t + eps   ->  -x_f <= -(t+eps)

    We store each inequality as:
      sum_i coeff_i * x_i <= threshold
    """
    feature_idx = tree.tree_.feature
    threshold = tree.tree_.threshold

    leaf_nodes = _all_leaf_nodes(tree)

    cols = ["ID"] + feature_names + ["threshold", "prediction"]
    rows: List[Dict[str, float]] = []

    for leaf_id, leaf in enumerate(leaf_nodes, start=1):
        path = _path_to_leaf(tree, leaf)
        pred = float(tree.tree_.value[leaf].reshape(-1)[0])

        for (node, direction) in path:
            f = int(feature_idx[node])
            thr = float(threshold[node])

            row = {c: 0.0 for c in cols}
            row["ID"] = float(leaf_id)
            row["prediction"] = pred

            if direction == 0:
                #left: x_f <= thr
                row[feature_names[f]] = 1.0
                row["threshold"] = thr
            else:
                #right: x_f >= thr + eps  => -x_f <= -(thr+eps)
                row[feature_names[f]] = -1.0
                row["threshold"] = -(thr + eps)

            rows.append(row)

    return pd.DataFrame(rows, columns=cols)


#bounds helper (optional?)

def compute_feature_bounds_from_data(X: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Bounds from data; lower bounded by 0 to match diet x >= 0."""
    bounds: Dict[str, Tuple[float, float]] = {}
    for c in X.columns:
        arr = X[c].to_numpy(dtype=float)
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        if hi < lo:
            lo, hi = hi, lo
        lo = max(0.0, lo)
        if hi <= lo:
            hi = lo + 1e-6
        bounds[c] = (lo, hi)
    return bounds


def _get_requirement_scalar(nutr_req: pd.DataFrame, nutrient: str) -> float:
    #nutr_req is 1-row DF? use first row.
    return float(nutr_req[nutrient].iloc[0])


#solver-free CART embedding: enumerate leaves + LP per leaf

def _features_from_tree_table(tree_table: pd.DataFrame) -> List[str]:
    return [c for c in tree_table.columns if c not in ("ID", "threshold", "prediction")]


def _format_solution_grams(features: List[str], x: np.ndarray, grams_per_unit: float = 100.0, tol: float = 1e-7) -> Dict[str, str]:
    sol: Dict[str, str] = {}
    for f, v in zip(features, x):
        g = float(v) * float(grams_per_unit)
        if g > tol:
            sol[f] = f"{np.round(g, 2)}g"
    return sol


def solve_tpdp_cart_via_leaf_enumeration(
    instance: SyriaInstance,
    tree_table: pd.DataFrame,
    *,
    pal_threshold: float = 0.5,
    feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    sugar_value: float = 0.2,   #20g in 100g-units
    salt_value: float = 0.05,   #5g in 100g-units
) -> Tuple[Any, Dict[str, str], float]:
    """
    Returns: (best_linprog_result, solution_dict_grams, leaf_prediction)
    """
    features = _features_from_tree_table(tree_table)

    #check foods exist in Syria instance
    missing = [f for f in features if f not in instance.nutr_val.index]
    if missing:
        raise ValueError(
            "These foods appear in the tree constraint table but not in Syria nutr_val:\n"
            f"{missing}"
        )

    #align instance matrices to features
    nutrients = list(instance.nutr_req.columns)
    nutr_val = instance.nutr_val.loc[features, nutrients]               #(n, m)
    cost_p = instance.cost_p.loc[features].astype(float).to_numpy()     #(n,)

    #nutrient constraints: sum_i x_i * nutr_val[i, req] >= req
    #linprog uses A_ub x <= b_ub, so: -nutr_val^T x <= -req
    A_nutr = nutr_val.to_numpy(dtype=float).T  #(m, n)
    b_nutr = np.array([_get_requirement_scalar(instance.nutr_req, r) for r in nutrients], dtype=float)

    #base LP pieces
    A_ub_base = -A_nutr
    b_ub_base = -b_nutr

    #variable bounds
    bounds_list: List[Tuple[float, Optional[float]]] = []
    for f in features:
        if feature_bounds is None:
            bounds_list.append((0.0, None))
        else:
            lo, hi = feature_bounds[f]
            lo = max(0.0, float(lo))
            hi = float(hi)
            bounds_list.append((lo, hi))

    #fix Sugar/Salt (best matches the notebook)
    if "Sugar" in features:
        i = features.index("Sugar")
        bounds_list[i] = (float(sugar_value), float(sugar_value))
    if "Salt" in features:
        i = features.index("Salt")
        bounds_list[i] = (float(salt_value), float(salt_value))

    #enumerate leaves
    best_res = None
    best_x = None
    best_pred = None
    best_cost = np.inf

    #robust ID handling
    ids = tree_table["ID"].to_numpy()
    #if some csvs store ID float, normalize
    leaf_ids = sorted({int(round(float(v))) for v in ids})

    for lid in leaf_ids:
        leaf_rows = tree_table.loc[tree_table["ID"].round().astype(int) == lid].copy()
        if leaf_rows.empty:
            continue

        pred = float(leaf_rows["prediction"].iloc[0])
        if pred < float(pal_threshold):
            continue

        A_leaf = leaf_rows[features].to_numpy(dtype=float)
        b_leaf = leaf_rows["threshold"].to_numpy(dtype=float)

        #drop completely-zero rows?
        keep = np.where(np.linalg.norm(A_leaf, ord=1, axis=1) > 0)[0]
        A_leaf = A_leaf[keep, :]
        b_leaf = b_leaf[keep]

        if A_leaf.size == 0:
            A_ub = A_ub_base
            b_ub = b_ub_base
        else:
            A_ub = np.vstack([A_leaf, A_ub_base])
            b_ub = np.concatenate([b_leaf, b_ub_base])

        res = linprog(
            c=cost_p,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds_list,
            method="highs",
        )

        if not res.success:
            continue

        if res.fun < best_cost:
            best_cost = float(res.fun)
            best_res = res
            best_x = res.x
            best_pred = pred

    if best_res is None or best_x is None or best_pred is None:
        raise RuntimeError(
            "No feasible leaf found for the given palatability threshold. "
            "Try lowering PAL_THRESHOLD or check constraints/bounds."
        )

    sol = _format_solution_grams(features, best_x, grams_per_unit=100.0)
    return best_res, sol, float(best_pred)


#Pyomo backend, optional

def _pyomo_solver_available(solver_name: str) -> bool:
    if not _PYOMO_AVAILABLE:
        return False
    try:
        opt = SolverFactory(solver_name)
        return bool(opt.available())
    except Exception:
        return False


#end-to-end runners

def run_cart_baseline_train_and_solve(
    *,
    wfp_csv_path: str,
    syria_xlsx_path: str,
    pal_threshold: float = 0.5,
    #training knobs
    test_size: float = 0.8,
    split_seed: int = 42,
    max_depth: int = 4,
    tree_seed: int = 0,
    #embedding knobs
    use_data_bounds: bool = True,
    eps: float = 1e-6,
    solver_name: str = "gurobi",   
    tee: bool = False,            
    backend: str = "auto",        
):
    instance = load_syria_instance(syria_xlsx_path)
    X, y = load_wfp_dataset(wfp_csv_path, shuffle_seed=0)

    trained = train_cart_regressor(
        X, y,
        test_size=test_size,
        split_seed=split_seed,
        max_depth=max_depth,
        tree_seed=tree_seed,
    )

    tree_table = extract_tree_table(trained.model, trained.features, eps=eps)

    bounds = compute_feature_bounds_from_data(trained.X_train) if use_data_bounds else None

    #default: safe backend
    if backend == "pyomo" and (_PYOMO_AVAILABLE and _pyomo_solver_available(solver_name)):
        pass

    res, sol, pred = solve_tpdp_cart_via_leaf_enumeration(
        instance,
        tree_table,
        pal_threshold=pal_threshold,
        feature_bounds=bounds,
        sugar_value=0.2,
        salt_value=0.05,
    )

    return res, sol, pred, tree_table


def run_cart_from_constraint_csv(
    *,
    constraints_csv_path: str,
    wfp_csv_path: str,
    syria_xlsx_path: str,
    pal_threshold: float = 0.5,
    use_data_bounds: bool = True,
    solver_name: str = "gurobi",   
    tee: bool = False,            
    backend: str = "auto",        
):
    """
    Best way to match the notebook's saved models:
    load the exported constraint table (e.g. results/cart/TPDP_v1_palatability_model.csv)
    and solve TPDP with it.
    """
    instance = load_syria_instance(syria_xlsx_path)
    tree_table = pd.read_csv(constraints_csv_path)

    bounds = None
    if use_data_bounds:
        X, _ = load_wfp_dataset(wfp_csv_path, shuffle_seed=0)
        bounds = compute_feature_bounds_from_data(X)

    if backend == "pyomo" and (_PYOMO_AVAILABLE and _pyomo_solver_available(solver_name)):
        pass

    res, sol, pred = solve_tpdp_cart_via_leaf_enumeration(
        instance,
        tree_table,
        pal_threshold=pal_threshold,
        feature_bounds=bounds,
        sugar_value=0.2,
        salt_value=0.05,
    )

    return res, sol, pred
