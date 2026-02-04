#main.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import random


#choose what to run
#"cart" | "cart_local" | "cart_local_globpoly" | "linear" | "nowdcl"
MODE = "cart_local_globpoly"

#paths
DATA_DIR = Path.home() / "İndirilenler" / "processed-data"
CSV_PATH = DATA_DIR / "WFP_dataset.csv"
XLSX_PATH = DATA_DIR / "Syria_instance.xlsx"


PAL_THRESHOLD = 0.50      #t: we learn g(x)=t-label, enforce g(x)<=0 <=> label>=t
GRAMS_PER_UNIT = 100.0    #decision variables are in 100g-units, printing converts to grams


#global CART settings
USE_EXPORTED_CART_CSV = False
CART_CONSTRAINTS_CSV = Path.home() / "İndirilenler" / "results" / "cart" / "TPDP_v1_palatability_model.csv"

TREE_MAX_DEPTH = 3
TREE_SEED = 0
SPLIT_SEED = 42
TEST_SIZE = 0.8

USE_DATA_BOUNDS = True    #for global CART only


#local CART settings
LC_ITERS = 6
LC_NEIGH_K = 250
LC_MAX_DEPTH = 3
LC_STANDARDIZE = True       #distance in z-score space
LC_WEIGHT_LAMBDA = 0.0      #0 disables distance weighting, >0 uses exp(-λ||x-xit||^2)
LC_USE_LOCAL_BOUNDS = True  #local bounds from neighborhood data (else global bounds)


#NOWDCL settings
SEED = 42
N_DATA_CAP = 120
N_ITERS = 3
MAX_SUBITER = 2000
TALK = True

NOWDCL_DEGREE = 2
NOWDCL_ALPHA = 1.0

LINEAR_DEGREE = 1
LINEAR_ALPHA = 0.0

#hybrid: Local CART + Global Polynomial guardrail
HY_ITERS = 6
HY_NEIGH_K = 400
HY_MAX_DEPTH = 3
HY_STANDARDIZE = True


#helpers
def _set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _cap_dataset(X: np.ndarray, y: np.ndarray, n_cap: int, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly subsample (without replacement) to at most n_cap rows."""
    if X.shape[0] <= n_cap:
        return X, y
    rng = random.Random(seed)
    idx = list(range(X.shape[0]))
    rng.shuffle(idx)
    keep = np.array(idx[:n_cap], dtype=int)
    return X[keep], y[keep]


def _format_solution_grams(features: List[str], x: np.ndarray, grams_per_unit: float = GRAMS_PER_UNIT, tol: float = 1e-7) -> Dict[str, str]:
    sol: Dict[str, str] = {}
    for f, v in zip(features, np.asarray(x, dtype=float).reshape(-1)):
        g = float(v) * float(grams_per_unit)
        if g > tol:
            sol[f] = f"{np.round(g, 2)}g"
    return sol


def _x0_with_fixed_salt_sugar(features: List[str], sugar_value: float = 0.2, salt_value: float = 0.05) -> np.ndarray:
    x0 = np.zeros(len(features), dtype=float)
    if "Salt" in features:
        x0[features.index("Salt")] = float(salt_value)
    if "Sugar" in features:
        x0[features.index("Sugar")] = float(sugar_value)
    return x0


def _require_files() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing dataset CSV: {CSV_PATH}")
    if not XLSX_PATH.exists():
        raise FileNotFoundError(f"Missing Syria instance XLSX: {XLSX_PATH}")


#polynomial prep
def _prepare_poly_from_csv(
    *,
    csv_path: str | Path,
    xlsx_path: str | Path,
    t: float,
    degree: int,
    alpha: float,
    auto_eq: bool = True,
    use_data_bounds: bool = False,
) -> Dict[str, Any]:
    """
    Produces the same objects main.py expects (X, y, bounds, x0, foods, poly_global,
    global_model, f, g, extra_constraints), but reads *all* XLSX data from xlsx_path.
    """
    import pandas as pd
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
    from scipy.optimize import LinearConstraint, Bounds
    import dt_core as DT

    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")
    foods = [c for c in df.columns if c != "label"]
    n = len(foods)

    X = df[foods].to_numpy(dtype=float)
    y_label = df["label"].to_numpy(dtype=float)
    y_g = (float(t) - y_label)  #g-target

    #load Syria instance
    instance = DT.load_syria_instance(xlsx_path)

    #cost vector aligned to foods
    missing_cost = [f for f in foods if f not in instance.cost_p.index]
    if missing_cost:
        raise ValueError(f"FoodCost sheet missing foods: {missing_cost}")
    cost = instance.cost_p.loc[foods].astype(float).to_numpy()

    #nutritional inequality A_ineq x >= b_ineq
    nutrients = list(instance.nutr_req.columns)
    missing_nutr = [f for f in foods if f not in instance.nutr_val.index]
    if missing_nutr:
        raise ValueError(f"nutr_val sheet missing foods: {missing_nutr}")
    nutr_val = instance.nutr_val.loc[foods, nutrients]
    A_ineq = nutr_val.to_numpy(dtype=float).T  #(m, n)
    b_ineq = np.array([float(instance.nutr_req[n].iloc[0]) for n in nutrients], dtype=float)

    #equalities (Salt/Sugar) in 100g-units
    A_eq = None
    b_eq = None
    if auto_eq:
        rows = []
        rhs = []
        if "Salt" in foods:
            r = np.zeros(n, dtype=float)
            r[foods.index("Salt")] = 1.0
            rows.append(r)
            rhs.append(0.05)
        if "Sugar" in foods:
            r = np.zeros(n, dtype=float)
            r[foods.index("Sugar")] = 1.0
            rows.append(r)
            rhs.append(0.2)
        if rows:
            A_eq = np.vstack(rows)
            b_eq = np.asarray(rhs, dtype=float)

    #bounds
    lb = np.zeros(n, dtype=float)
    if use_data_bounds:
        X_df = df[foods]
        bdict = DT.compute_feature_bounds_from_data(X_df)
        ub = np.array([bdict[f][1] for f in foods], dtype=float)
    else:
        ub = np.full(n, np.inf, dtype=float)
    bounds = Bounds(lb, ub)

    #SciPy LinearConstraint objects for NOWDCL subproblems
    extra_constraints: List[Any] = []
    extra_constraints.append(LinearConstraint(A_ineq, lb=b_ineq, ub=np.full_like(b_ineq, np.inf)))

    if A_eq is not None:
        extra_constraints.append(LinearConstraint(A_eq, lb=b_eq, ub=b_eq))

    #global polynomial surrogate for g(x)
    poly_global = PolynomialFeatures(degree=int(degree), include_bias=True)
    X_poly = poly_global.fit_transform(X)
    global_model = Ridge(alpha=float(alpha), fit_intercept=False).fit(X_poly, y_g)

    def f(x: np.ndarray) -> float:
        xx = np.asarray(x, dtype=float)[:n]
        return float(np.dot(cost, xx))

    def g(x: np.ndarray) -> float:
        xx = np.asarray(x, dtype=float)[:n].reshape(1, -1)
        return float(global_model.predict(poly_global.transform(xx))[0])

    x0 = _x0_with_fixed_salt_sugar(foods)

    return {
        "X": X,
        "y": y_g,
        "bounds": bounds,
        "x0": x0,
        "foods": foods,
        "poly_global": poly_global,
        "global_model": global_model,
        "extra_constraints": extra_constraints,
        "f": f,
        "g": g,
        "cost": cost,
        "A_ineq": A_ineq,
        "b_ineq": b_ineq,
        "A_eq": A_eq,
        "b_eq": b_eq,
        "instance": instance,
        "t": float(t),
    }


#linear baseline (LP): min cost^T x s.t. nutrition + salt/sugar + g_linear(x) <= 0
def run_linear():
    from scipy.optimize import linprog

    prep = _prepare_poly_from_csv(
        csv_path=CSV_PATH,
        xlsx_path=XLSX_PATH,
        t=PAL_THRESHOLD,
        degree=LINEAR_DEGREE,
        alpha=LINEAR_ALPHA,
        auto_eq=True,
        use_data_bounds=USE_DATA_BOUNDS,
    )

    foods = prep["foods"]
    n = len(foods)
    cost = prep["cost"]

    #linear g(x) from PolynomialFeatures(degree=1, include_bias=True)
    coef = np.ravel(prep["global_model"].coef_)
    if coef.shape[0] != n + 1:
        raise ValueError(f"Unexpected coef length for degree=1: got {coef.shape[0]}, expected {n+1}")
    b0 = float(coef[0])
    w = coef[1:].astype(float)

    #inequalities: A_ineq x >= b_ineq  ->  -A_ineq x <= -b_ineq
    A_ub = [-prep["A_ineq"]]
    b_ub = [-prep["b_ineq"]]
    #plus g(x) <= 0  -> w^T x <= -b0
    A_ub.append(w.reshape(1, -1))
    b_ub.append(np.array([-b0], dtype=float))

    A_ub = np.vstack(A_ub)
    b_ub = np.concatenate(b_ub)

    #equalities (Salt/Sugar)
    A_eq = prep["A_eq"]
    b_eq = prep["b_eq"]

    #linprog bounds
    lb = prep["bounds"].lb
    ub = prep["bounds"].ub
    bounds_list = [(float(lb[i]), None if np.isinf(ub[i]) else float(ub[i])) for i in range(n)]

    res = linprog(
        c=cost,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds_list,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"[linear] linprog failed: {res.message}")

    x_star = res.x
    sol = _format_solution_grams(foods, x_star)
    g_star = float(prep["g"](x_star))
    pal_hat = float(PAL_THRESHOLD - g_star)  #label_hat = t - g_hat

    print("The optimal solution is:\n", sol)
    print("g_hat(x*) =", g_star, "(feasible if ≤ 0)")
    print("Predicted palatability (label_hat) =", pal_hat)


#NOWDCL
def run_nowdcl():
    import diet_problem as algo

    prep = _prepare_poly_from_csv(
        csv_path=CSV_PATH,
        xlsx_path=XLSX_PATH,
        t=PAL_THRESHOLD,
        degree=NOWDCL_DEGREE,
        alpha=NOWDCL_ALPHA,
        auto_eq=True,
        use_data_bounds=USE_DATA_BOUNDS,
    )

    #global name "g"
    algo.g = prep["g"]

    Xs, ys = _cap_dataset(prep["X"], prep["y"], N_DATA_CAP, seed=SEED)

    exitcode, xis, local_models, poly_locals, subsets_used = algo.nowdcl(
        f=prep["f"],
        X=Xs,
        y=ys,
        bounds=prep["bounds"],
        x_0=prep["x0"],
        niters=N_ITERS,
        extra_constraints=prep["extra_constraints"],
        talk=TALK,
        max_subiter=MAX_SUBITER,
    )

    x_star = xis[-1] if xis else prep["x0"]
    g_star = float(prep["g"](x_star))
    pal_hat = float(PAL_THRESHOLD - g_star)

    print("\n=== NOWDCL Summary ===")
    print("Final x*:", x_star)
    print("g_hat(x*) =", g_star, "(feasible if ≤ 0)")
    print("Predicted palatability (label_hat) =", pal_hat)
    print("Exit code:", exitcode)


#CART (GLOBAL) — uses dt_core's end-to-end runner
def run_cart():
    import dt_core as DT
    print("dt_core imported from:", DT.__file__, flush=True)

    if USE_EXPORTED_CART_CSV:
        res, sol, pred = DT.run_cart_from_constraint_csv(
            constraints_csv_path=str(CART_CONSTRAINTS_CSV),
            wfp_csv_path=str(CSV_PATH),
            syria_xlsx_path=str(XLSX_PATH),
            pal_threshold=PAL_THRESHOLD,
            use_data_bounds=USE_DATA_BOUNDS,
            solver_name="ignored",
            tee=False,
        )
    else:
        res, sol, pred, _tree_table = DT.run_cart_baseline_train_and_solve(
            wfp_csv_path=str(CSV_PATH),
            syria_xlsx_path=str(XLSX_PATH),
            pal_threshold=PAL_THRESHOLD,
            test_size=TEST_SIZE,
            split_seed=SPLIT_SEED,
            max_depth=TREE_MAX_DEPTH,
            tree_seed=TREE_SEED,
            use_data_bounds=USE_DATA_BOUNDS,
            solver_name="ignored",
            tee=False,
        )

    print("The optimal solution is:\n", sol)
    print("The predicted palatability of the optimal solution is", pred)
    print("Solver status:", getattr(res, "message", getattr(res, "status", "n/a")))


#CART (LOCAL) — iterative kNN → local tree → leaf-enumeration LP → update x
def run_cart_local():
    import dt_core as DT
    from sklearn.tree import DecisionTreeRegressor

    X_df, y_ser = DT.load_wfp_dataset(str(CSV_PATH), shuffle_seed=SEED)
    foods = list(X_df.columns)
    X_all = X_df[foods].to_numpy(dtype=float)
    y_all = y_ser.to_numpy(dtype=float)

    xit = _x0_with_fixed_salt_sugar(foods)

    #optional z-score distance
    if LC_STANDARDIZE:
        mu = X_all.mean(axis=0)
        sd = X_all.std(axis=0, ddof=0)
        sd[sd == 0.0] = 1.0

        def _dist(x: np.ndarray) -> np.ndarray:
            return np.linalg.norm((X_all - x.reshape(1, -1)) / sd.reshape(1, -1), axis=1)
    else:
        def _dist(x: np.ndarray) -> np.ndarray:
            return np.linalg.norm(X_all - x.reshape(1, -1), axis=1)

    k = int(min(max(LC_NEIGH_K, 5 * len(foods)), len(X_all)))
    instance = DT.load_syria_instance(str(XLSX_PATH))

    #global bounds
    bounds_global = DT.compute_feature_bounds_from_data(X_df[foods]) if USE_DATA_BOUNDS else None

    best = {"cost": np.inf, "res": None, "sol": None, "pred": None}
    for it in range(max(1, LC_ITERS)):
        d = _dist(xit)
        idx = np.argsort(d)[:k]
        X_loc = X_all[idx, :]
        y_loc = y_all[idx]
        X_loc_df = X_df.iloc[idx][foods]

        sw = None
        if LC_WEIGHT_LAMBDA and LC_WEIGHT_LAMBDA > 0.0:
            sw = np.exp(-(LC_WEIGHT_LAMBDA) * (d[idx] ** 2))

        tree = DecisionTreeRegressor(max_depth=LC_MAX_DEPTH, random_state=SEED)
        tree.fit(X_loc, y_loc, sample_weight=sw)

        tree_table = DT.extract_tree_table(tree, foods)

        feature_bounds = DT.compute_feature_bounds_from_data(X_loc_df) if LC_USE_LOCAL_BOUNDS else bounds_global

        res, sol, pred = DT.solve_tpdp_cart_via_leaf_enumeration(
            instance,
            tree_table,
            pal_threshold=PAL_THRESHOLD,
            feature_bounds=feature_bounds,
            sugar_value=0.2,
            salt_value=0.05,
        )

        #track best by objective value
        if res.success and float(res.fun) < float(best["cost"]):
            best = {"cost": float(res.fun), "res": res, "sol": sol, "pred": float(pred)}

        x_new = np.asarray(res.x, dtype=float)
        step = float(np.linalg.norm(x_new - xit))
        print(f"[cart_local] iter {it+1}: step={step:.2e}, leaf_pred={pred:.3f}, cost={float(res.fun):.6g}")
        if step < 1e-6:
            break
        xit = x_new

    print("\n[cart_local] done.")
    print("The optimal solution is:\n", best["sol"])
    print("The predicted palatability of the optimal solution is", best["pred"])
    print("Solver status:", getattr(best["res"], "message", getattr(best["res"], "status", "n/a")))


#Hybrid helpers: global poly linearization + leaf-enumeration LP with extra rows
def _poly_grad_at(poly, coef_vec: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Analytic gradient for g(x) = sum_m coef[m] * prod_j x_j^{powers[m,j]}
    using poly.powers_ (PolynomialFeatures with include_bias=True).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    coef_vec = np.asarray(coef_vec, dtype=float).reshape(-1)

    powers = getattr(poly, "powers_", None)
    if powers is None:
        #very defensive fallback: central differences
        eps = 1e-6
        grad = np.zeros_like(x)
        base = float(coef_vec @ poly.transform(x.reshape(1, -1)).ravel())
        for j in range(x.size):
            xp = x.copy(); xm = x.copy()
            xp[j] += eps; xm[j] -= eps
            gp = float(coef_vec @ poly.transform(xp.reshape(1, -1)).ravel())
            gm = float(coef_vec @ poly.transform(xm.reshape(1, -1)).ravel())
            grad[j] = (gp - gm) / (2 * eps)
        return grad

    n = x.size
    grad = np.zeros(n, dtype=float)

    #term m value: coef[m] * Π_j x_j^{p[m,j]}
    for m in range(powers.shape[0]):
        cm = float(coef_vec[m])
        if cm == 0.0:
            continue
        pm = powers[m]  #length n
        #derivative for each coordinate
        for j in range(n):
            ej = int(pm[j])
            if ej == 0:
                continue
            #cm * ej * x_j^{ej-1} * Π_{k≠j} x_k^{pm[k]}
            prod = 1.0
            for k in range(n):
                pk = int(pm[k]) - (1 if k == j else 0)
                if pk > 0:
                    prod *= x[k] ** pk
            grad[j] += cm * ej * prod

    return grad


def _global_poly_linearization_row(poly_global, global_model, x0: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    One affine row a^T x <= b from first-order expansion of g at x0:
      g(x) ≈ g0 + ∇g(x0)^T (x - x0) <= 0  ⇒  (∇g)^T x <= -g0 + (∇g)^T x0
    """
    coef = np.ravel(global_model.coef_)
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    g0 = float(global_model.predict(poly_global.transform(x0.reshape(1, -1)))[0])
    grad = _poly_grad_at(poly_global, coef, x0)
    a = grad.astype(float)
    b = float(-g0 + float(grad @ x0))
    return a, b


def _solve_tpdp_cart_leaf_enum_with_extra_rows(
    instance,
    tree_table,
    *,
    pal_threshold: float,
    feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    extra_A: Optional[np.ndarray] = None,
    extra_b: Optional[np.ndarray] = None,
    sugar_value: float = 0.2,
    salt_value: float = 0.05,
):
    """
    Like dt_core.solve_tpdp_cart_via_leaf_enumeration, but appends extra_A x <= extra_b
    to each leaf LP.
    """
    from scipy.optimize import linprog

    #features from constraint table
    features = [c for c in tree_table.columns if c not in ("ID", "threshold", "prediction")]

    #align instance matrices to features
    nutrients = list(instance.nutr_req.columns)
    nutr_val = instance.nutr_val.loc[features, nutrients]               #(n, m)
    cost_p = instance.cost_p.loc[features].astype(float).to_numpy()     #(n,)

    #nutrition: -nutr_val^T x <= -req
    A_nutr = nutr_val.to_numpy(dtype=float).T
    b_nutr = np.array([float(instance.nutr_req[r].iloc[0]) for r in nutrients], dtype=float)
    A_ub_base = -A_nutr
    b_ub_base = -b_nutr

    #bounds
    bounds_list: List[Tuple[float, Optional[float]]] = []
    for f in features:
        if feature_bounds is None:
            bounds_list.append((0.0, None))
        else:
            lo, hi = feature_bounds[f]
            lo = max(0.0, float(lo))
            hi = float(hi)
            bounds_list.append((lo, hi))

    #fix Sugar/Salt
    if "Sugar" in features:
        i = features.index("Sugar")
        bounds_list[i] = (float(sugar_value), float(sugar_value))
    if "Salt" in features:
        i = features.index("Salt")
        bounds_list[i] = (float(salt_value), float(salt_value))

    #leaf ids
    ids = tree_table["ID"].to_numpy()
    leaf_ids = sorted({int(round(float(v))) for v in ids})

    best_res = None
    best_x = None
    best_pred = None
    best_cost = np.inf

    for lid in leaf_ids:
        leaf_rows = tree_table.loc[tree_table["ID"].round().astype(int) == lid].copy()
        if leaf_rows.empty:
            continue

        pred = float(leaf_rows["prediction"].iloc[0])
        if pred < float(pal_threshold):
            continue

        A_leaf = leaf_rows[features].to_numpy(dtype=float)
        b_leaf = leaf_rows["threshold"].to_numpy(dtype=float)

        keep = np.where(np.linalg.norm(A_leaf, ord=1, axis=1) > 0)[0]
        A_leaf = A_leaf[keep, :]
        b_leaf = b_leaf[keep]

        if A_leaf.size == 0:
            A_ub = A_ub_base
            b_ub = b_ub_base
        else:
            A_ub = np.vstack([A_leaf, A_ub_base])
            b_ub = np.concatenate([b_leaf, b_ub_base])

        if extra_A is not None and extra_b is not None:
            A_ub = np.vstack([A_ub, np.asarray(extra_A, dtype=float)])
            b_ub = np.concatenate([b_ub, np.asarray(extra_b, dtype=float).reshape(-1)])

        res = linprog(c=cost_p, A_ub=A_ub, b_ub=b_ub, bounds=bounds_list, method="highs")
        if not res.success:
            continue

        if float(res.fun) < float(best_cost):
            best_cost = float(res.fun)
            best_res = res
            best_x = res.x
            best_pred = pred

    if best_res is None:
        raise RuntimeError("No feasible leaf found (after adding extra rows). Consider relaxing PAL_THRESHOLD or bounds.")

    sol = _format_solution_grams(features, best_x, grams_per_unit=GRAMS_PER_UNIT)
    return best_res, sol, float(best_pred)


#Hybrid: Local CART + Global Polynomial (linearization) guardrail
def run_cart_local_globpoly():
    import dt_core as DT
    from sklearn.tree import DecisionTreeRegressor

    #global polynomial surrogate
    prep = _prepare_poly_from_csv(
        csv_path=CSV_PATH,
        xlsx_path=XLSX_PATH,
        t=PAL_THRESHOLD,
        degree=NOWDCL_DEGREE,
        alpha=NOWDCL_ALPHA,
        auto_eq=True,
        use_data_bounds=USE_DATA_BOUNDS,
    )
    foods = prep["foods"]
    poly_global = prep["poly_global"]
    global_model = prep["global_model"]
    xit = prep["x0"].copy()

    #full dataset for neighborhood selection
    X_df, y_ser = DT.load_wfp_dataset(str(CSV_PATH), shuffle_seed=SEED)
    X_all = X_df[foods].to_numpy(dtype=float)
    y_all = y_ser.to_numpy(dtype=float)

    instance = DT.load_syria_instance(str(XLSX_PATH))
    bounds_global = DT.compute_feature_bounds_from_data(X_df[foods]) if USE_DATA_BOUNDS else None

    if HY_STANDARDIZE:
        mu = X_all.mean(axis=0)
        sd = X_all.std(axis=0, ddof=0)
        sd[sd == 0.0] = 1.0

        def _dist(x: np.ndarray) -> np.ndarray:
            return np.linalg.norm((X_all - x.reshape(1, -1)) / sd.reshape(1, -1), axis=1)
    else:
        def _dist(x: np.ndarray) -> np.ndarray:
            return np.linalg.norm(X_all - x.reshape(1, -1), axis=1)

    k = int(min(max(HY_NEIGH_K, 5 * len(foods)), len(X_all)))

    best = {"cost": np.inf, "res": None, "sol": None, "pred": None, "g": None}
    for it in range(max(1, HY_ITERS)):
        d = _dist(xit)
        idx = np.argsort(d)[:k]
        X_loc = X_all[idx, :]
        y_loc = y_all[idx]
        X_loc_df = X_df.iloc[idx][foods]

        tree = DecisionTreeRegressor(max_depth=HY_MAX_DEPTH, random_state=SEED)
        tree.fit(X_loc, y_loc)
        tree_table = DT.extract_tree_table(tree, foods)

        bounds_local = DT.compute_feature_bounds_from_data(X_loc_df) if LC_USE_LOCAL_BOUNDS else bounds_global

        #global poly linearization at current xit: a^T x <= b
        a, b = _global_poly_linearization_row(poly_global, global_model, xit)
        extra_A = a.reshape(1, -1)
        extra_b = np.array([b], dtype=float)

        res, sol, pred = _solve_tpdp_cart_leaf_enum_with_extra_rows(
            instance,
            tree_table,
            pal_threshold=PAL_THRESHOLD,
            feature_bounds=bounds_local,
            extra_A=extra_A,
            extra_b=extra_b,
            sugar_value=0.2,
            salt_value=0.05,
        )

        x_new = np.asarray(res.x, dtype=float)
        step = float(np.linalg.norm(x_new - xit))
        g_new = float(prep["g"](x_new))

        if res.success and float(res.fun) < float(best["cost"]):
            best = {"cost": float(res.fun), "res": res, "sol": sol, "pred": float(pred), "g": g_new}

        print(f"[hybrid] iter {it+1}: step={step:.2e}, leaf_pred={pred:.3f}, cost={float(res.fun):.6g}, g_global(x_new)={g_new:.3g}")

        if step < 1e-6:
            break
        xit = x_new

    print("\n[hybrid] done.")
    print("The optimal solution is:\n", best["sol"])
    print("Leaf prediction at best leaf:", best["pred"])
    if best["g"] is not None:
        print("g_global(x*) =", best["g"], "(feasible if ≤ 0)")
        print("Predicted palatability (label_hat) =", float(PAL_THRESHOLD - best["g"]))
    print("Solver status:", getattr(best["res"], "message", getattr(best["res"], "status", "n/a")))


#entry
def main():
    _set_seeds(SEED)
    _require_files()

    if MODE == "cart":
        run_cart()
    elif MODE == "cart_local":
        run_cart_local()
    elif MODE == "cart_local_globpoly":
        run_cart_local_globpoly()
    elif MODE == "linear":
        run_linear()
    elif MODE == "nowdcl":
        run_nowdcl()
    else:
        raise ValueError("Unknown MODE. Use 'cart', 'cart_local', 'cart_local_globpoly', 'linear', or 'nowdcl'.")


if __name__ == "__main__":
    main()
