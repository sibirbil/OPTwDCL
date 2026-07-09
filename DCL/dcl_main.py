# -*- coding: utf-8 -*-

import random
import warnings
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize
from scipy.spatial import ConvexHull
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

try:
    import gurobipy as gp
    from gurobipy import GRB
    from gurobi_ml import add_predictor_constr
except ImportError:
    gp = None
    GRB = None
    add_predictor_constr = None

import dcl_core as dcl
import dcl_problem as dclp




SEED = 102
ACCEPT_FAILED_SCIPY_RESULTS = True

# Concrete controls. Used only when problem_name = "concrete".
CONCRETE_DATA_PATH = None          # Example: "Concrete_Data.xlsx"
CONCRETE_STRENGTH_REQ = 45.0
CONCRETE_WEIGHT_LB = 2230.0
CONCRETE_WEIGHT_UB = 2450.0
CONCRETE_DCL_ITERATIONS = 1
# Local concrete surrogate model.
# Use "cart" for one decision tree, "rf" for random forest, or "nn" for neural network.
CONCRETE_LEARNING_MODEL_TYPE = "cart" #or rf, cart
# The current global baseline function is RF-specific.
# Keep this False for clean CART local-DCL runs.
RUN_CONCRETE_GLOBAL_BASELINE = False
# Just define them as normal variables
USE_SOFT_CONSTRAINTS = True
SLACK_PENALTY_WEIGHT = 1000.0

# Rosenbrock global controls. Used only when problem_name = "rosenbrock".
RUN_ROSENBROCK_OLD_GLOBAL = True
RUN_ROSENBROCK_NEW_GLOBAL = True
ROSENBROCK_OLD_GLOBAL_DEGREE = 4


def _set_seeds(options, seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    options.seed = seed


def _as_numpy(X):
    return X.to_numpy(dtype=float) if hasattr(X, "to_numpy") else np.asarray(X, dtype=float)


def _as_vector(y):
    return y.to_numpy(dtype=float).ravel() if hasattr(y, "to_numpy") else np.asarray(y, dtype=float).ravel()


def _mse(y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_true = np.asarray(y_true, dtype=float).ravel()
    return float(np.mean((y_pred - y_true) ** 2))


def _rmse(y_pred, y_true):
    return float(np.sqrt(_mse(y_pred, y_true)))


def _relative_rmse(y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_true = np.asarray(y_true, dtype=float).ravel()
    denom = np.maximum(np.maximum(np.abs(y_pred), np.abs(y_true)), 1e-12)
    return float(np.sqrt(np.mean(((y_pred - y_true) / denom) ** 2)))


def _safe_float(v, default=np.nan):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _fmt(v, digits=6):
    v = _safe_float(v)
    return "nan" if not np.isfinite(v) else f"{v:.{digits}f}"


def _gurobi_status_name(status):
    if GRB is None:
        return str(status)

    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return status_map.get(status, str(status))


def _get_model_attr(model, attr, default=np.nan):
    try:
        return float(getattr(model, attr))
    except Exception:
        return default


# =========================================================
# Full-data convex hull helper
# =========================================================
def _full_data_hull_constraint(X):
    Xnp = _as_numpy(X)
    n = int(Xnp.shape[1])

    if Xnp.shape[0] == 0:
        raise ValueError("X is empty; cannot build a convex hull.")

    hull = ConvexHull(Xnp)
    hull_A = hull.equations[:, :n]
    hull_b = hull.equations[:, n]

    hull_constraint = LinearConstraint(
        hull_A,
        lb=-np.inf,
        ub=-hull_b,
    )

    def hull_violation(x):
        x = np.asarray(x, dtype=float)
        return float(np.max(hull_A @ x + hull_b))

    return hull_constraint, hull_violation, hull_A, hull_b


# =========================================================
# Rosenbrock true reference over full-data convex hull
# =========================================================
def solve_true_rosenbrock_over_convex_hull(problem, X, x_init=None):
    Xnp = _as_numpy(X)
    n = int(Xnp.shape[1])

    hull_constraint, hull_violation, _, _ = _full_data_hull_constraint(Xnp)
    bounds = Bounds(problem["bounds_lb"], problem["bounds_ub"])
    true_constraint = NonlinearConstraint(problem["true_constraint"], -np.inf, 0.0)

    lb_x = np.asarray(problem["bounds_lb"], dtype=float)
    ub_x = np.asarray(problem["bounds_ub"], dtype=float)

    def obj_fun(x):
        return float(problem["objective"](x))

    def true_g(x):
        return float(problem["true_constraint"](x))

    true_g_values = np.array([true_g(Xnp[i, :]) for i in range(Xnp.shape[0])], dtype=float)
    in_box = np.all((Xnp >= lb_x) & (Xnp <= ub_x), axis=1)
    feasible_idx = np.where(in_box & (true_g_values <= 0.0))[0]

    starts = []

    if feasible_idx.size > 0:
        ordered = feasible_idx[np.argsort(Xnp[feasible_idx, 0])]
        for i in ordered[:10]:
            starts.append(Xnp[i, :].copy())

    if x_init is not None:
        nearest = int(np.argmin(np.linalg.norm(Xnp - np.asarray(x_init, dtype=float), axis=1)))
        starts.append(Xnp[nearest, :].copy())

    starts.append(np.mean(Xnp, axis=0))

    best_feasible_res = None
    best_any_res = None

    for x0 in starts:
        res = minimize(
            obj_fun,
            x0,
            bounds=bounds,
            constraints=[true_constraint, hull_constraint],
            method="SLSQP",
            options={"disp": False, "maxiter": 1000, "ftol": 1e-10},
        )

        if not hasattr(res, "x") or res.x is None:
            continue

        x_res = np.asarray(res.x[:n], dtype=float)

        if not np.all(np.isfinite(x_res)):
            continue

        is_feasible = (
            true_g(x_res) <= 1e-6
            and hull_violation(x_res) <= 1e-6
            and np.all(x_res >= lb_x - 1e-6)
            and np.all(x_res <= ub_x + 1e-6)
        )

        if best_any_res is None or float(res.fun) < float(best_any_res.fun):
            best_any_res = res

        if is_feasible:
            if best_feasible_res is None or float(res.fun) < float(best_feasible_res.fun):
                best_feasible_res = res

    if best_feasible_res is not None:
        return best_feasible_res, hull_violation

    if best_any_res is not None:
        return best_any_res, hull_violation

    raise RuntimeError("True Rosenbrock full-hull reference solve failed from all starts.")


# =========================================================
# Rosenbrock old/new global baselines
# =========================================================
def _fit_unscaled_polynomial_model(X_train, y_train, degree):
    """Old x_new style: no scaling, include_bias=True, LinearRegression."""
    poly = PolynomialFeatures(degree=int(degree), include_bias=True)
    X_poly = poly.fit_transform(np.asarray(X_train, dtype=float))
    model = LinearRegression().fit(X_poly, np.asarray(y_train, dtype=float).ravel())
    return model, poly



def _predict_unscaled_polynomial(model, poly, X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return model.predict(poly.transform(X))

# =========================================================
# Print polynomial from the accepted 1-iteration DCL model
# =========================================================
def _sklearn_feature_to_python(term):
    """
    Convert sklearn PolynomialFeatures names to Python expressions.

    Examples:
        'x0'       -> 'x0'
        'x1^2'     -> '(x1**2)'
        'x0 x1'    -> 'x0*x1'
        'x0^2 x2'  -> '(x0**2)*x2'
    """
    if term == "1":
        return "1.0"

    pieces = []

    for part in term.split(" "):
        if "^" in part:
            var, power = part.split("^")
            pieces.append(f"({var}**{power})")
        else:
            pieces.append(part)

    return "*".join(pieces)


def print_one_iteration_dcl_polynomial(
    dcls,
    feature_names=("x0", "x1", "x2"),
    output_dir="outputs",
    tol=None,
):
    """
    Print the polynomial learned by the accepted DCL model.

    Use this after your 1-iteration/full-data DCL run.

    It prints and saves:
        1. a coefficient table
        2. a copy-paste Python function

    This does NOT refit anything.
    It uses exactly dcls.local_models[0].
    """
    if len(dcls.local_models) == 0:
        raise RuntimeError("No local model found in dcls.local_models.")

    accepted_model = dcls.local_models[0]

    if accepted_model.learned_model is None:
        raise RuntimeError("accepted_model.learned_model is None.")

    if accepted_model.poly_features is None:
        raise RuntimeError(
            "accepted_model.poly_features is None. "
            "This printer is for the Rosenbrock polynomial model."
        )

    model = accepted_model.learned_model
    poly = accepted_model.poly_features

    names = poly.get_feature_names_out(feature_names)
    coefs = np.asarray(model.coef_, dtype=float).ravel()
    intercept = float(np.asarray(model.intercept_, dtype=float).ravel()[0])

    # PolynomialFeatures(include_bias=True) creates a feature called '1'.
    # LinearRegression also has an intercept.
    # Therefore the actual constant is:
    #     intercept + coefficient_of_feature_1
    constant = intercept
    terms = []

    for name, coef in zip(names, coefs):
        coef = float(coef)

        if name == "1":
            constant += coef
        else:
            if tol is None or abs(coef) > float(tol):
                terms.append((name, coef, _sklearn_feature_to_python(name)))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coeff_csv_path = output_dir / "one_iter_full_data_dcl_polynomial_coefficients.csv"
    function_py_path = output_dir / "one_iter_full_data_dcl_polynomial_function.py"
    txt_path = output_dir / "one_iter_full_data_dcl_polynomial.txt"

    # Save coefficient table.
    with coeff_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "coefficient", "python_term"])
        writer.writerow(["1", f"{constant:.17e}", "1.0"])

        for name, coef, py_term in terms:
            writer.writerow([name, f"{coef:.17e}", py_term])

    # Build copy-paste Python function.
    lines = []
    lines.append("# Polynomial from accepted 1-iteration/full-data DCL model")
    lines.append("# Copy this function and evaluate it at any (x0, x1, x2).")
    lines.append("")
    lines.append("def alt_dcl_global_ghat(x0, x1, x2):")
    lines.append("    return (")
    lines.append(f"        {constant:+.17e}")

    for name, coef, py_term in terms:
        lines.append(f"        {coef:+.17e} * ({py_term})")

    lines.append("    )")
    lines.append("")

    function_text = "\n".join(lines)

    with function_py_path.open("w", encoding="utf-8") as f:
        f.write(function_text)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("ONE-ITERATION FULL-DATA DCL POLYNOMIAL\n")
        f.write(f"degree = {accepted_model.model_degree}\n")
        f.write(f"ndata_model = {accepted_model.ndata_model}\n")
        f.write(f"train_rmse = {accepted_model.train_rmse}\n")
        f.write(f"test_rmse = {accepted_model.test_rmse}\n")
        f.write(f"train_rel_rmse = {accepted_model.train_rel_rmse}\n")
        f.write(f"test_rel_rmse = {accepted_model.test_rel_rmse}\n")
        f.write("\n")
        f.write(function_text)

    print("\n" + "=" * 80)
    print("ONE-ITERATION FULL-DATA DCL POLYNOMIAL")
    print("=" * 80)
    print("degree =", accepted_model.model_degree)
    print("ndata_model =", accepted_model.ndata_model)
    print("train_rmse =", accepted_model.train_rmse)
    print("test_rmse =", accepted_model.test_rmse)
    print("train_rel_rmse =", accepted_model.train_rel_rmse)
    print("test_rel_rmse =", accepted_model.test_rel_rmse)

    if accepted_model.solution_x is not None:
        print("DCL solution_x =", np.asarray(accepted_model.solution_x, dtype=float))
        print("DCL local ghat(solution_x) =", accepted_model.solution_y)
        print("DCL slack =", accepted_model.solution_slack)

    print("\nCoefficient CSV saved to:")
    print(coeff_csv_path)

    print("\nCopy-paste Python function saved to:")
    print(function_py_path)

    print("\nPolynomial text saved to:")
    print(txt_path)

    print("\nCopy-paste function:")
    print(function_text)
    print("=" * 80)

    return function_text


def _rosenbrock_poly_metrics(model, poly, X_eval, y_eval):
    X_eval = _as_numpy(X_eval)
    y_eval = np.asarray(y_eval, dtype=float).ravel()
    pred = _predict_unscaled_polynomial(model, poly, X_eval)
    return {
        "mse": _mse(pred, y_eval),
        "rmse": _rmse(pred, y_eval),
        "rel_rmse": _relative_rmse(pred, y_eval),
    }


def _solve_rosenbrock_surrogate_over_convex_hull(problem, X, global_pred, x_init=None):
    Xnp = _as_numpy(X)
    n = int(Xnp.shape[1])

    hull_constraint, hull_violation, _, _ = _full_data_hull_constraint(Xnp)
    surrogate_constraint = NonlinearConstraint(global_pred, -np.inf, 0.0)
    bounds = Bounds(problem["bounds_lb"], problem["bounds_ub"])

    lb_x = np.asarray(problem["bounds_lb"], dtype=float)
    ub_x = np.asarray(problem["bounds_ub"], dtype=float)

    def obj_fun(x):
        return float(problem["objective"](x))

    pred_values = np.array([global_pred(Xnp[i, :]) for i in range(Xnp.shape[0])], dtype=float)
    in_box = np.all((Xnp >= lb_x) & (Xnp <= ub_x), axis=1)
    pred_feasible_idx = np.where(in_box & (pred_values <= 0.0))[0]

    starts = []

    if pred_feasible_idx.size > 0:
        ordered = pred_feasible_idx[np.argsort(Xnp[pred_feasible_idx, 0])]
        for i in ordered[:10]:
            starts.append(Xnp[i, :].copy())

    if x_init is not None:
        nearest = int(np.argmin(np.linalg.norm(Xnp - np.asarray(x_init, dtype=float), axis=1)))
        starts.append(Xnp[nearest, :].copy())

    starts.append(np.mean(Xnp, axis=0))

    best_feasible_res = None
    best_any_res = None

    for x0 in starts:
        res = minimize(
            obj_fun,
            x0,
            bounds=bounds,
            constraints=[surrogate_constraint, hull_constraint],
            method="SLSQP",
            options={"disp": False, "maxiter": 1000, "ftol": 1e-10},
        )

        if not hasattr(res, "x") or res.x is None:
            continue

        x_res = np.asarray(res.x[:n], dtype=float)

        if not np.all(np.isfinite(x_res)):
            continue

        pred_res = float(global_pred(x_res))
        hull_viol = hull_violation(x_res)

        is_feasible = (
            pred_res <= 1e-6
            and hull_viol <= 1e-6
            and np.all(x_res >= lb_x - 1e-6)
            and np.all(x_res <= ub_x + 1e-6)
        )

        if best_any_res is None or float(res.fun) < float(best_any_res.fun):
            best_any_res = res

        if is_feasible:
            if best_feasible_res is None or float(res.fun) < float(best_feasible_res.fun):
                best_feasible_res = res

    if best_feasible_res is not None:
        return best_feasible_res, hull_violation

    if best_any_res is not None:
        return best_any_res, hull_violation

    raise RuntimeError("Rosenbrock surrogate full-hull solve failed from all starts.")


def solve_old_global_rosenbrock(problem, X, y, degree=4, x_init=None):
    """
    Old global Rosenbrock baseline:
        degree fixed, default 4
        fit on all data
        no scaling
        PolynomialFeatures(include_bias=True)
        LinearRegression
    """
    Xnp = _as_numpy(X)
    ynp = _as_vector(y)

    model, poly = _fit_unscaled_polynomial_model(Xnp, ynp, degree)

    def global_pred(x):
        return float(_predict_unscaled_polynomial(model, poly, x)[0])

    train_metrics = _rosenbrock_poly_metrics(model, poly, Xnp, ynp)

    res, hull_violation = _solve_rosenbrock_surrogate_over_convex_hull(
        problem,
        Xnp,
        global_pred,
        x_init=x_init,
    )

    x_sol = np.asarray(res.x[: Xnp.shape[1]], dtype=float)
    pred_g = float(global_pred(x_sol))
    true_g = float(dclp.rosenbrock_constraint_value(x_sol))

    diagnostics = {
        "method": "old_global_fixed_degree_all_data",
        "degree": int(degree),
        "train_size": int(len(ynp)),
        "test_size": 0,
        "train_mse": train_metrics["mse"],
        "train_rmse": train_metrics["rmse"],
        "train_rel_rmse": train_metrics["rel_rmse"],
        "test_mse": np.nan,
        "test_rmse": np.nan,
        "test_rel_rmse": np.nan,
        "all_data_mse_after_refit": train_metrics["mse"],
        "all_data_rmse_after_refit": train_metrics["rmse"],
        "all_data_rel_rmse_after_refit": train_metrics["rel_rmse"],
        "solution_x": x_sol,
        "objective": float(problem["objective"](x_sol)),
        "predicted_g": pred_g,
        "true_g": true_g,
        "prediction_error": abs(pred_g - true_g),
        "hull_violation": float(hull_violation(x_sol)),
        "feasible_by_surrogate": bool(pred_g <= 1e-6),
        "feasible_by_true_constraint": bool(true_g <= 1e-6),
        "solver_success": bool(res.success),
        "solver_message": str(res.message),
    }

    return res, model, poly, global_pred, hull_violation, diagnostics


def solve_new_global_rosenbrock(
    problem,
    X,
    y,
    max_degree=4,
    train_data_ratio=0.70,
    seed=SEED,
    x_init=None,
):
    """
    New global Rosenbrock diagnostic baseline:
        no scaling
        PolynomialFeatures(include_bias=True)
        degrees 1,...,max_degree selected by STANDARD test RMSE
        selected degree is refit on all data before optimization
    """
    Xnp = _as_numpy(X)
    ynp = _as_vector(y)

    X_train, X_test, y_train, y_test = train_test_split(
        Xnp,
        ynp,
        train_size=float(train_data_ratio),
        random_state=int(seed),
        shuffle=True,
    )

    best = None
    degree_rows = []

    print("\n" + "=" * 70)
    print("NEW GLOBAL ROSENBROCK DEGREE SELECTION")
    print("unscaled PolynomialFeatures(include_bias=True) + LinearRegression")
    print("degree | train_mse | test_mse | train_rmse | test_rmse | train_rel_rmse | test_rel_rmse")
    print("-" * 70)

    for degree in range(1, int(max_degree) + 1):
        model, poly = _fit_unscaled_polynomial_model(X_train, y_train, degree)
        train_metrics = _rosenbrock_poly_metrics(model, poly, X_train, y_train)
        test_metrics = _rosenbrock_poly_metrics(model, poly, X_test, y_test)

        row = {
            "degree": int(degree),
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "train_mse": train_metrics["mse"],
            "test_mse": test_metrics["mse"],
            "train_rmse": train_metrics["rmse"],
            "test_rmse": test_metrics["rmse"],
            "train_rel_rmse": train_metrics["rel_rmse"],
            "test_rel_rmse": test_metrics["rel_rmse"],
        }
        degree_rows.append(row)

        print(
            f"{degree:6d} | "
            f"{row['train_mse']:.6e} | {row['test_mse']:.6e} | "
            f"{row['train_rmse']:.6e} | {row['test_rmse']:.6e} | "
            f"{row['train_rel_rmse']:.6e} | {row['test_rel_rmse']:.6e}"
        )

        if best is None or row["test_rel_rmse"] < best["test_rel_rmse"]:
            best = row

    model, poly = _fit_unscaled_polynomial_model(Xnp, ynp, best["degree"])

    def global_pred(x):
        return float(_predict_unscaled_polynomial(model, poly, x)[0])

    all_metrics = _rosenbrock_poly_metrics(model, poly, Xnp, ynp)

    res, hull_violation = _solve_rosenbrock_surrogate_over_convex_hull(
        problem,
        Xnp,
        global_pred,
        x_init=x_init,
    )

    x_sol = np.asarray(res.x[: Xnp.shape[1]], dtype=float)
    pred_g = float(global_pred(x_sol))
    true_g = float(dclp.rosenbrock_constraint_value(x_sol))

    diagnostics = {
        "method": "new_global_degree_selected_refit_all_data",
        "degree": int(best["degree"]),
        "degree_rows": degree_rows,
        "train_size": int(best["train_size"]),
        "test_size": int(best["test_size"]),
        "train_mse": float(best["train_mse"]),
        "test_mse": float(best["test_mse"]),
        "train_rmse": float(best["train_rmse"]),
        "test_rmse": float(best["test_rmse"]),
        "train_rel_rmse": float(best["train_rel_rmse"]),
        "test_rel_rmse": float(best["test_rel_rmse"]),
        "all_data_mse_after_refit": all_metrics["mse"],
        "all_data_rmse_after_refit": all_metrics["rmse"],
        "all_data_rel_rmse_after_refit": all_metrics["rel_rmse"],
        "solution_x": x_sol,
        "objective": float(problem["objective"](x_sol)),
        "predicted_g": pred_g,
        "true_g": true_g,
        "prediction_error": abs(pred_g - true_g),
        "hull_violation": float(hull_violation(x_sol)),
        "feasible_by_surrogate": bool(pred_g <= 1e-6),
        "feasible_by_true_constraint": bool(true_g <= 1e-6),
        "solver_success": bool(res.success),
        "solver_message": str(res.message),
    }

    return res, model, poly, global_pred, hull_violation, diagnostics


def print_global_rosenbrock_report(diagnostics):
    print("\n" + "=" * 70)
    print("ROSENBROCK GLOBAL FULL-HULL BASELINE")
    print("=" * 70)
    print("method =", diagnostics.get("method"))
    print("degree =", diagnostics.get("degree"))
    print("train_size =", diagnostics.get("train_size"), "test_size =", diagnostics.get("test_size"))

    for key in [
        "train_mse",
        "test_mse",
        "train_rmse",
        "test_rmse",
        "train_rel_rmse",
        "test_rel_rmse",
        "all_data_mse_after_refit",
        "all_data_rmse_after_refit",
        "all_data_rel_rmse_after_refit",
    ]:
        if key in diagnostics:
            print(key, "=", diagnostics[key])

    print("solution_x =", diagnostics.get("solution_x"))
    print("objective =", diagnostics.get("objective"))
    print("predicted g(x) =", diagnostics.get("predicted_g"))
    print("true g(x) =", diagnostics.get("true_g"))
    print("prediction error =", diagnostics.get("prediction_error"))
    print("hull violation =", diagnostics.get("hull_violation"))
    print("feasible by surrogate =", diagnostics.get("feasible_by_surrogate"))
    print("feasible by true constraint =", diagnostics.get("feasible_by_true_constraint"))
    print("solver success =", diagnostics.get("solver_success"))
    print("solver message =", diagnostics.get("solver_message"))
    print("=" * 70)


def _rosenbrock_local_row(row, true_ref_obj=np.nan, old_global_pred=None, new_global_pred=None, label="Local DCL"):
    x = np.asarray(row["solution_x"], dtype=float).ravel()
    objective = _safe_float(row.get("original_objective", row.get("objective")))
    true_g = float(dclp.rosenbrock_constraint_value(x))

    old_g = float(old_global_pred(x)) if old_global_pred is not None else np.nan
    new_g = float(new_global_pred(x)) if new_global_pred is not None else np.nan

    local_g = row.get("local_pred_solution")
    if local_g is None:
        local_g = row.get("solution_y")
    local_g = _safe_float(local_g)

    slack = _safe_float(row.get("relaxation_slack", 0.0), 0.0)
    relaxed_residual = _safe_float(row.get("relaxed_constraint_residual"))

    return {
        "method": label,
        "iter": row.get("iter"),
        "degree": row.get("degree"),
        "train_mse": _safe_float(row.get("train_mse")),
        "test_mse": _safe_float(row.get("test_mse")),
        "train_rmse": _safe_float(row.get("train_rmse")),
        "test_rmse": _safe_float(row.get("test_rmse")),
        "objective": objective,
        "gap": objective - _safe_float(true_ref_obj),
        "true_g": true_g,
        "old_global_g": old_g,
        "new_global_g": new_g,
        "local_g": local_g,
        "err_old_global": abs(old_g - true_g) if np.isfinite(old_g) else np.nan,
        "err_new_global": abs(new_g - true_g) if np.isfinite(new_g) else np.nan,
        "err_local": abs(local_g - true_g) if np.isfinite(local_g) else np.nan,
        "slack": slack,
        "relaxed_residual": relaxed_residual,
        "true_feasible": bool(true_g <= 1e-6),
        "model_feasible": bool(relaxed_residual <= 1e-6) if np.isfinite(relaxed_residual) else np.nan,
        "x": x,
    }


def print_rosenbrock_dcl_report_rows(dcls, old_global_pred=None, new_global_pred=None):
    print("\nROSENBROCK DCL ITERATION ROWS")
    print(
        "Iter & x0 & x1 & x2 & Obj & True g & Old Global ghat & New Global ghat & "
        "Local ghat & Err(Old Glob) & Err(New Glob) & Err(Local) & Slack & "
        "Relaxed Residual & Data Size & Degree & Train MSE & Test MSE & "
        "Train RMSE & Test RMSE & Train Rel RMSE & Test Rel RMSE \\\\"
    )

    for row in dcls.iteration_log:
        r = _rosenbrock_local_row(
            row,
            old_global_pred=old_global_pred,
            new_global_pred=new_global_pred,
        )
        x = r["x"]
        print(
            f"{row.get('iter')} & "
            f"{x[0]:.6f} & {x[1]:.6f} & {x[2]:.6f} & "
            f"{r['objective']:.6e} & {r['true_g']:.6e} & "
            f"{r['old_global_g']:.6e} & {r['new_global_g']:.6e} & {r['local_g']:.6e} & "
            f"{r['err_old_global']:.6e} & {r['err_new_global']:.6e} & {r['err_local']:.6e} & "
            f"{r['slack']:.6e} & {r['relaxed_residual']:.6e} & "
            f"{row.get('data_size')} & {row.get('degree')} & "
            f"{r['train_mse']:.6e} & {r['test_mse']:.6e} & "
            f"{r['train_rmse']:.6e} & {r['test_rmse']:.6e} & "
            f"{_safe_float(row.get('train_rel_rmse')):.6e} & "
            f"{_safe_float(row.get('test_rel_rmse')):.6e} \\\\"
        )


def print_rosenbrock_global_local_comparison(
    true_diag,
    old_global_diag,
    new_global_diag,
    dcls,
    old_global_pred=None,
    new_global_pred=None,
):
    rows = []
    true_obj = true_diag.get("objective", np.nan) if true_diag is not None else np.nan

    if true_diag is not None:
        rows.append({
            "method": "True full-hull reference",
            "iter": "-",
            "degree": "-",
            "train_rmse": np.nan,
            "test_rmse": np.nan,
            "objective": true_diag["objective"],
            "gap": 0.0,
            "true_g": true_diag["true_g"],
            "pred_g": np.nan,
            "slack": 0.0,
            "relaxed_residual": np.nan,
            "true_feasible": true_diag.get("true_feasible", True),
            "model_feasible": True,
        })

    for label, diag in [
        ("Old global fixed degree", old_global_diag),
        ("New global selected degree", new_global_diag),
    ]:
        if diag is not None:
            rows.append({
                "method": label,
                "iter": "-",
                "degree": diag.get("degree"),
                "train_rmse": diag.get("train_rmse"),
                "test_rmse": diag.get("test_rmse"),
                "objective": diag.get("objective"),
                "gap": _safe_float(diag.get("objective")) - _safe_float(true_obj),
                "true_g": diag.get("true_g"),
                "pred_g": diag.get("predicted_g"),
                "slack": 0.0,
                "relaxed_residual": diag.get("predicted_g"),
                "true_feasible": diag.get("feasible_by_true_constraint"),
                "model_feasible": diag.get("feasible_by_surrogate"),
            })

    if len(dcls.iteration_log) > 0:
        local_rows = [
            _rosenbrock_local_row(
                row,
                true_ref_obj=true_obj,
                old_global_pred=old_global_pred,
                new_global_pred=new_global_pred,
            )
            for row in dcls.iteration_log
        ]

        final = local_rows[-1].copy()
        final["method"] = "Local DCL final"
        rows.append(final)

        true_feasible = [r for r in local_rows if bool(r["true_feasible"])]
        if true_feasible:
            best_feasible = min(true_feasible, key=lambda r: r["objective"]).copy()
            best_feasible["method"] = "Local DCL best true-feasible"
            rows.append(best_feasible)

        best_test = min(local_rows, key=lambda r: r["test_rmse"]).copy()
        best_test["method"] = "Local DCL best test RMSE"
        rows.append(best_test)

        lowest_obj = min(local_rows, key=lambda r: r["objective"]).copy()
        lowest_obj["method"] = "Local DCL lowest objective"
        rows.append(lowest_obj)

    print("\n" + "=" * 132)
    print("ROSENBROCK GLOBAL vs LOCAL DIAGNOSTIC COMPARISON")
    print("=" * 132)
    print(
        "Method | Iter | Deg | Train RMSE | Test RMSE | Obj | Gap vs True | True g | "
        "Predicted/Local g | Slack | Relaxed Residual | True Feas | Model Feas"
    )
    print("-" * 132)

    for r in rows:
        print(
            f"{r['method']} | "
            f"{r['iter']} | "
            f"{r['degree']} | "
            f"{_fmt(r.get('train_rmse'))} | "
            f"{_fmt(r.get('test_rmse'))} | "
            f"{_fmt(r.get('objective'))} | "
            f"{_fmt(r.get('gap'))} | "
            f"{_fmt(r.get('true_g'))} | "
            f"{_fmt(r.get('pred_g', r.get('local_g')))} | "
            f"{_fmt(r.get('slack'))} | "
            f"{_fmt(r.get('relaxed_residual'))} | "
            f"{r.get('true_feasible')} | "
            f"{r.get('model_feasible')}"
        )

    print("=" * 132)
    print("Rosenbrock selection rule: first require true g(x) <= 0; among true-feasible rows, lower objective x0 is better.")
    print("Slack only relaxes local learned ghat(x), not the convex hull and not the true constraint.")


# =========================================================
# Concrete diagnostics and global full-hull baseline
# =========================================================
def _last_solved_local_model(dcls):
    for local_model in reversed(dcls.local_models):
        if local_model.solution_x is not None:
            return local_model
    return None


def print_concrete_dcl_report(dcls, strength_req=45.0):
    if dcls.xit is None:
        print("No concrete solution was produced.")
        return

    x = np.asarray(dcls.xit, dtype=float).ravel()
    solved_model = _last_solved_local_model(dcls)

    pred_strength = None
    if solved_model is not None:
        try:
            pred_strength = dcl.dcl_predict_local_model(solved_model, x)
        except Exception:
            pred_strength = solved_model.solution_y

    cost = dclp.concrete_material_cost(x)
    weight = dclp.concrete_component_weight(x)

    print("\n" + "=" * 70)
    print("CONCRETE DCL POINT-ESTIMATION REPORT")
    print("=" * 70)
    print("methodology = point estimation")
    print("model       =", dcls.options.learning_model_type)
    print("trust region=", dcls.options.trust_region)
    print("constraint  = predicted compressive strength >=", float(strength_req), "MPa")
    print("weight      =", CONCRETE_WEIGHT_LB, "<= sum(material components) <=", CONCRETE_WEIGHT_UB)
    print("material cost =", cost)
    print("component weight =", weight)

    if pred_strength is not None:
        print("predicted compressive strength =", float(pred_strength))
        print("strength margin =", float(pred_strength) - float(strength_req))

    print("\nmixture:")
    for name, value in dclp.concrete_solution_dict(x).items():
        unit = "days" if name == "age" else "kg"
        print(f"  {name}: {value:.6f} {unit}")

    if hasattr(dclp, "CONCRETE_ARTICLE_POINT_REFERENCE"):
        ref = dclp.CONCRETE_ARTICLE_POINT_REFERENCE
        print("\narticle point-estimation RF reference:")
        print("  article point cost =", float(ref["cost"]))
        print("  DCL cost gap       =", cost - float(ref["cost"]))

    print("=" * 70)


def solve_global_concrete_full_hull(problem, X, y, options, strength_req):
    if gp is None or GRB is None or add_predictor_constr is None:
        raise ImportError("Global concrete setup requires gurobipy and gurobi_ml.")

    Xnp = _as_numpy(X)
    ynp = _as_vector(y)

    Xtr, Xte, ytr, yte = train_test_split(
        Xnp,
        ynp,
        train_size=float(options.train_data_ratio),
        random_state=int(options.seed),
        shuffle=True,
    )

    best_rf = None
    best = None
    candidate_rows = []

    print("\n" + "=" * 70)
    print("GLOBAL CONCRETE RF DEGREE SELECTION  (degree = max_depth)")
    print("depth | train_mse | test_mse | train_rmse | test_rmse | train_rel_rmse | test_rel_rmse")
    print("-" * 70)

    for depth in range(1, int(options.max_model_degree) + 1):
        rf = RandomForestRegressor(
            n_estimators=int(getattr(options, "rf_n_estimators", 100)),
            max_depth=depth,
            min_samples_leaf=int(getattr(options, "rf_min_samples_leaf", 1)),
            random_state=int(options.seed),
            n_jobs=-1,
        ).fit(Xtr, ytr)

        pred_tr = rf.predict(Xtr)
        pred_te = rf.predict(Xte)

        stats = {
            "degree": int(depth),
            "train_size": int(len(ytr)),
            "test_size": int(len(yte)),
            "train_mse": _mse(pred_tr, ytr),
            "test_mse": _mse(pred_te, yte),
            "train_rmse": _rmse(pred_tr, ytr),
            "test_rmse": _rmse(pred_te, yte),
            "train_rel_rmse": _relative_rmse(pred_tr, ytr),
            "test_rel_rmse": _relative_rmse(pred_te, yte),
        }
        candidate_rows.append(stats)

        print(
            f"{depth:5d} | "
            f"{stats['train_mse']:.6e} | {stats['test_mse']:.6e} | "
            f"{stats['train_rmse']:.6e} | {stats['test_rmse']:.6e} | "
            f"{stats['train_rel_rmse']:.6e} | {stats['test_rel_rmse']:.6e}"
        )

        if best is None or stats["test_rel_rmse"] < best["test_rel_rmse"]:
            best = stats
            best_rf = rf

    ndata, n = Xnp.shape

    model = problem.copy()
    dcl._maybe_set_gurobi_tolerances(model, options)

    x_vars = [model.getVarByName(f"input_vars[{j}]") for j in range(n)]
    y_var = model.getVarByName("output_vars[0]")

    if any(v is None for v in x_vars) or y_var is None:
        raise RuntimeError("Could not find input_vars/output_vars in the Gurobi model.")

    pred_constr = dcl._add_predictor_constr_safely(
        model,
        best_rf,
        input_vars=np.array(x_vars, dtype=object).reshape(1, -1),
        output_vars=np.array([y_var], dtype=object).reshape(1, 1),
        options=options,
    )

    # Full-data convex hull:
    # x = sum_i lambda_i X_i, sum_i lambda_i = 1, lambda_i >= 0.
    lam = model.addMVar(
        ndata,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="lambda_global_hull",
    )

    model.addConstr(
        gp.quicksum(lam[i] for i in range(ndata)) == 1.0,
        name="global_hull_sum",
    )

    for j in range(n):
        model.addConstr(
            x_vars[j] == gp.quicksum(float(Xnp[i, j]) * lam[i] for i in range(ndata)),
            name=f"global_hull_x[{j}]",
        )

    model.update()

    diagnostics = {
        "method": "Global RF full hull",
        "candidate_rows": candidate_rows,
        "degree": best["degree"],
        "train_size": best["train_size"],
        "test_size": best["test_size"],
        "train_mse": best["train_mse"],
        "test_mse": best["test_mse"],
        "train_rmse": best["train_rmse"],
        "test_rmse": best["test_rmse"],
        "train_rel_rmse": best["train_rel_rmse"],
        "test_rel_rmse": best["test_rel_rmse"],
        "hull_points": int(ndata),
        "mip_cols_before_opt": int(model.NumVars),
        "mip_rows_before_opt": int(model.NumConstrs),
        "num_bin_before_opt": int(model.NumBinVars),
        "num_int_before_opt": int(model.NumIntVars),
        "num_genconstrs_before_opt": int(model.NumGenConstrs),
    }

    model.optimize()

    status = int(model.Status)
    sol_count = int(model.SolCount)

    diagnostics.update({
        "status": status,
        "status_name": _gurobi_status_name(status),
        "sol_count": sol_count,
        "success": bool(status == GRB.OPTIMAL),
        "has_solution": bool(sol_count > 0),
        "mip_cols": int(model.NumVars),
        "mip_rows": int(model.NumConstrs),
        "num_bin": int(model.NumBinVars),
        "num_int": int(model.NumIntVars),
        "num_genconstrs": int(model.NumGenConstrs),
        "runtime": float(model.Runtime),
        "nodes": float(model.NodeCount),
        "work": float(model.Work),
    })

    if sol_count == 0:
        print("\n" + "=" * 70)
        print("GLOBAL CONCRETE RF FULL-HULL BASELINE")
        print("=" * 70)
        print("chosen degree/max_depth =", best["degree"])
        print("Gurobi status =", diagnostics["status_name"])
        print("No incumbent solution was available, so cost/strength/weight cannot be reported.")
        print("=" * 70)
        return diagnostics

    x_sol = np.array([v.X for v in x_vars], dtype=float)
    lam_sol = np.asarray(lam.X, dtype=float).ravel()
    x_from_hull = Xnp.T @ lam_sol

    embedded_strength = float(y_var.X)
    sklearn_strength = float(best_rf.predict(x_sol.reshape(1, -1))[0])

    sklearn_output_diag, manual_predictor_error, gurobi_ml_predictor_error = dcl._compute_predictor_diagnostics(
        pred_constr,
        best_rf,
        x_sol,
        embedded_strength,
    )

    predictor_error = manual_predictor_error
    if np.isfinite(gurobi_ml_predictor_error):
        predictor_error = max(predictor_error, gurobi_ml_predictor_error)

    predictor_error_tol = float(getattr(options, "predictor_error_tol", 1e-4))
    embedding_valid = bool(predictor_error <= predictor_error_tol)

    cost = float(dclp.concrete_material_cost(x_sol))
    weight = float(dclp.concrete_component_weight(x_sol))

    feasible_by_embedded_strength = bool(embedded_strength >= float(strength_req) - 1e-6)
    feasible_by_sklearn_strength = bool(sklearn_strength >= float(strength_req) - predictor_error_tol)
    feasible_by_weight = bool(CONCRETE_WEIGHT_LB - 1e-6 <= weight <= CONCRETE_WEIGHT_UB + 1e-6)

    valid_point_estimation_solution = bool(
        embedding_valid
        and feasible_by_sklearn_strength
        and feasible_by_weight
    )

    diagnostics.update({
        "objective": _get_model_attr(model, "ObjVal"),
        "obj_bound": _get_model_attr(model, "ObjBound"),
        "mip_gap": _get_model_attr(model, "MIPGap"),
        "solution_x": x_sol,
        "material_cost": cost,
        "component_weight": weight,
        "embedded_strength": embedded_strength,
        "sklearn_strength": sklearn_strength,
        "sklearn_output_diag": sklearn_output_diag,
        "embedded_strength_margin": embedded_strength - float(strength_req),
        "sklearn_strength_margin": sklearn_strength - float(strength_req),
        "strength_mismatch_embedded_minus_sklearn": embedded_strength - sklearn_strength,
        "manual_predictor_error": manual_predictor_error,
        "gurobi_ml_predictor_error": gurobi_ml_predictor_error,
        "predictor_error": predictor_error,
        "predictor_error_tol": predictor_error_tol,
        "embedding_valid": embedding_valid,
        "valid_point_estimation_solution": valid_point_estimation_solution,
        "lambda_sum": float(np.sum(lam_sol)),
        "lambda_sum_violation": float(abs(np.sum(lam_sol) - 1.0)),
        "lambda_min": float(np.min(lam_sol)),
        "lambda_max": float(np.max(lam_sol)),
        "hull_eq_violation_inf": float(np.linalg.norm(x_sol - x_from_hull, ord=np.inf)),
        "feasible_by_embedded_strength": feasible_by_embedded_strength,
        "feasible_by_sklearn_strength": feasible_by_sklearn_strength,
        "feasible_by_weight": feasible_by_weight,
    })

    diagnostics["optimizer_success"] = diagnostics["success"]
    diagnostics["success"] = bool(diagnostics["success"] and valid_point_estimation_solution)

    print("\n" + "=" * 70)
    print("GLOBAL CONCRETE RF FULL-HULL BASELINE")
    print("=" * 70)
    print("chosen degree/max_depth =", diagnostics["degree"])
    print("train_size =", diagnostics["train_size"], "test_size =", diagnostics["test_size"])

    for key in [
        "train_mse",
        "test_mse",
        "train_rmse",
        "test_rmse",
        "train_rel_rmse",
        "test_rel_rmse",
    ]:
        print(key, "=", diagnostics[key])

    print("Gurobi status =", diagnostics["status_name"])
    print("solution count =", diagnostics["sol_count"])
    print("objective/model ObjVal =", diagnostics["objective"])
    print("material cost =", diagnostics["material_cost"])
    print("component weight =", diagnostics["component_weight"])
    print("embedded Gurobi strength =", diagnostics["embedded_strength"])
    print("sklearn RF strength =", diagnostics["sklearn_strength"])
    print("embedded strength margin =", diagnostics["embedded_strength_margin"])
    print("sklearn strength margin =", diagnostics["sklearn_strength_margin"])
    print("strength mismatch embedded - sklearn =", diagnostics["strength_mismatch_embedded_minus_sklearn"])
    print("manual predictor error =", diagnostics["manual_predictor_error"])
    print("gurobi_ml predictor error =", diagnostics["gurobi_ml_predictor_error"])
    print("predictor error used =", diagnostics["predictor_error"])
    print("predictor error tolerance =", diagnostics["predictor_error_tol"])
    print("embedding valid =", diagnostics["embedding_valid"])
    print("valid point-estimation solution =", diagnostics["valid_point_estimation_solution"])
    print("hull_eq_violation_inf =", diagnostics["hull_eq_violation_inf"])
    print("manual predictor error =", diagnostics["manual_predictor_error"])
    print("gurobi_ml predictor error =", diagnostics["gurobi_ml_predictor_error"])
    print("predictor error used =", diagnostics["predictor_error"])
    print("predictor error tolerance =", diagnostics["predictor_error_tol"])
    print("embedding valid =", diagnostics["embedding_valid"])
    print("valid point-estimation solution =", diagnostics["valid_point_estimation_solution"])
    print("lambda_sum_violation =", diagnostics["lambda_sum_violation"])
    print("mip_gap =", diagnostics["mip_gap"])
    print("obj_bound =", diagnostics["obj_bound"])
    print("runtime =", diagnostics["runtime"])
    print("nodes =", diagnostics["nodes"])
    print("mip_cols =", diagnostics["mip_cols"], "mip_rows =", diagnostics["mip_rows"])
    print("num_bin =", diagnostics["num_bin"], "num_int =", diagnostics["num_int"])
    print("num_genconstrs =", diagnostics["num_genconstrs"])

    print("\nmixture:")
    for name, value in dclp.concrete_solution_dict(x_sol).items():
        unit = "days" if name == "age" else "kg"
        print(f"  {name}: {value:.6f} {unit}")

    print("=" * 70)
    return diagnostics

def print_concrete_dcl_report_rows(dcls):
    print("\nCONCRETE DCL ITERATION ROWS")
    print(
        "Iter & Data Size & Degree/max_depth & Train rel. RMSE & Test rel. RMSE & "
        "Material Obj & Gurobi Obj incl. Slack & Slack & Slack Penalty & "
        "Sklearn Strength & Embedded Strength & Weight & MIP rows & MIP cols & "
        "Bin & Int & GenConstr & Nodes & MIP Runtime & MIP Work & Fit Time & Total Time \\\\"
    )

    for row in dcls.iteration_log:
        x = np.asarray(row["solution_x"], dtype=float)

        sklearn_strength = row.get("local_pred_solution", row.get("solution_y"))
        embedded_strength = row.get("solution_y")

        material_obj = _safe_float(row.get("original_objective", row.get("objective")))
        if not np.isfinite(material_obj):
            material_obj = float(dclp.concrete_material_cost(x))

        gurobi_obj = _safe_float(
            row.get("model_objective", row.get("relaxed_objective", material_obj))
        )

        slack = _safe_float(row.get("relaxation_slack", 0.0), 0.0)
        slack_cost = _safe_float(row.get("slack_penalty_cost", 0.0), 0.0)

        print(
            f"{row.get('iter')} & "
            f"{row.get('data_size')} & "
            f"{row.get('degree')} & "
            f"{_safe_float(row.get('train_rel_rmse')):.6f} & "
            f"{_safe_float(row.get('test_rel_rmse')):.6f} & "
            f"{material_obj:.6f} & "
            f"{gurobi_obj:.6f} & "
            f"{slack:.6f} & "
            f"{slack_cost:.6f} & "
            f"{float(sklearn_strength):.6f} & "
            f"{float(embedded_strength):.6f} & "
            f"{dclp.concrete_component_weight(x):.6f} & "
            f"{row.get('mip_rows')} & "
            f"{row.get('mip_cols')} & "
            f"{row.get('num_bin')} & "
            f"{row.get('num_int')} & "
            f"{row.get('num_genconstrs')} & "
            f"{_safe_float(row.get('mip_nodes')):.0f} & "
            f"{_safe_float(row.get('mip_runtime')):.6f} & "
            f"{_safe_float(row.get('mip_work')):.6f} & "
            f"{_safe_float(row.get('fit_time')):.6f} & "
            f"{_safe_float(row.get('total_iteration_time', row.get('elapsed_time'))):.6f} \\\\"
        )
        


def _concrete_local_row(row, label, strength_req=45.0):
    x = np.asarray(row["solution_x"], dtype=float).ravel()

    sklearn_pred = row.get("local_pred_solution")
    embedded_pred = row.get("solution_y")

    if sklearn_pred is None:
        sklearn_pred = embedded_pred

    sklearn_pred = _safe_float(sklearn_pred)
    embedded_pred = _safe_float(embedded_pred)

    cost = float(dclp.concrete_material_cost(x))
    weight = float(dclp.concrete_component_weight(x))

    slack = _safe_float(row.get("relaxation_slack", 0.0), 0.0)
    slack_penalty_cost = _safe_float(row.get("slack_penalty_cost", 0.0), 0.0)
    model_objective = _safe_float(
        row.get("model_objective", row.get("relaxed_objective", row.get("objective")))
    )

    predictor_error = _safe_float(row.get("predictor_error", 0.0), 0.0)
    predictor_error_tol = _safe_float(row.get("predictor_error_tol", 1e-4), 1e-4)
    #embedding_valid = bool(predictor_error <= predictor_error_tol)
    embedding_valid = bool(
        row.get("embedding_valid", predictor_error <= predictor_error_tol)
    )
    
    feasible_embedded_hard = bool(embedded_pred >= float(strength_req) - 1e-6)
    feasible_sklearn_hard = bool(sklearn_pred >= float(strength_req) - 1e-6)
    feasible_sklearn_soft = bool(sklearn_pred + slack >= float(strength_req) - 1e-6)
    feasible_weight = bool(CONCRETE_WEIGHT_LB - 1e-6 <= weight <= CONCRETE_WEIGHT_UB + 1e-6)

    return {
        "method": label,
        "iter": row.get("iter"),
        "degree": row.get("degree"),
        "train_rmse": _safe_float(row.get("train_rmse")),
        "test_rmse": _safe_float(row.get("test_rmse")),
        "train_rel_rmse": _safe_float(row.get("train_rel_rmse")),
        "test_rel_rmse": _safe_float(row.get("test_rel_rmse")),
        "cost": cost,
        "model_objective": model_objective,
        "slack": slack,
        "slack_penalty_cost": slack_penalty_cost,
        "embedded_strength": embedded_pred,
        "sklearn_strength": sklearn_pred,
        "embedded_margin": embedded_pred - float(strength_req),
        "sklearn_margin": sklearn_pred - float(strength_req),
        "weight": weight,
        "feasible_embedded_hard": feasible_embedded_hard,
        "feasible_sklearn_hard": feasible_sklearn_hard,
        "feasible_sklearn_soft": feasible_sklearn_soft,
        "feasible_weight": feasible_weight,
        "valid_hard_point_solution": bool(
            embedding_valid and feasible_sklearn_hard and feasible_weight
        ),
        "valid_soft_solution": bool(
            embedding_valid and feasible_sklearn_soft and feasible_weight
        ),
        "mip_cols": row.get("mip_cols"),
        "mip_rows": row.get("mip_rows"),
        "num_bin": row.get("num_bin"),
        "num_int": row.get("num_int"),
        "num_genconstrs": row.get("num_genconstrs"),
        "mip_nodes": _safe_float(row.get("mip_nodes")),
        "mip_runtime": _safe_float(row.get("mip_runtime")),
        "mip_work": _safe_float(row.get("mip_work")),
    }



def print_concrete_global_local_comparison(global_diag, dcls, strength_req=45.0):
    rows = []

    if global_diag is not None and global_diag.get("has_solution", False):
        rows.append({
            "method": global_diag.get("method", "Global model full hull"),
            "iter": "-",
            "degree": global_diag.get("degree"),
            "train_rmse": global_diag.get("train_rmse"),
            "test_rmse": global_diag.get("test_rmse"),
            "train_rel_rmse": global_diag.get("train_rel_rmse"),
            "test_rel_rmse": global_diag.get("test_rel_rmse"),
            "cost": global_diag.get("material_cost"),
            "model_objective": global_diag.get("objective"),
            "slack": global_diag.get("slack", 0.0),
            "slack_penalty_cost": global_diag.get("slack_penalty_cost", 0.0),
            "embedded_strength": global_diag.get("embedded_strength"),
            "sklearn_strength": global_diag.get("sklearn_strength"),
            "embedded_margin": global_diag.get("embedded_strength_margin"),
            "sklearn_margin": global_diag.get("sklearn_strength_margin"),
            "weight": global_diag.get("component_weight"),
            "feasible_sklearn_hard": global_diag.get("feasible_by_sklearn_strength"),
            "feasible_sklearn_soft": global_diag.get("feasible_by_sklearn_strength"),
            "feasible_weight": global_diag.get("feasible_by_weight"),
            "valid_hard_point_solution": global_diag.get("valid_point_estimation_solution"),
            "valid_soft_solution": global_diag.get("valid_point_estimation_solution"),
            "mip_cols": global_diag.get("mip_cols"),
            "mip_rows": global_diag.get("mip_rows"),
            "num_bin": global_diag.get("num_bin"),
            "num_int": global_diag.get("num_int"),
            "num_genconstrs": global_diag.get("num_genconstrs"),
            "mip_nodes": global_diag.get("nodes"),
            "mip_runtime": global_diag.get("runtime"),
            "mip_work": global_diag.get("work"),
        })

    if len(dcls.iteration_log) > 0:
        locals_ = [
            _concrete_local_row(row, "Local DCL", strength_req)
            for row in dcls.iteration_log
        ]

        final = locals_[-1].copy()
        final["method"] = "Local DCL final"
        rows.append(final)

        hard_feasible = [r for r in locals_ if r["valid_hard_point_solution"]]

        if hard_feasible:
            best_hard = min(hard_feasible, key=lambda r: r["cost"]).copy()
            best_hard["method"] = "Local DCL best hard-feasible"
            rows.append(best_hard)

        soft_feasible = [r for r in locals_ if r["valid_soft_solution"]]

        if soft_feasible:
            best_soft = min(soft_feasible, key=lambda r: r["model_objective"]).copy()
            best_soft["method"] = "Local DCL best soft-feasible"
            rows.append(best_soft)

        lowest_cost = min(locals_, key=lambda r: r["cost"]).copy()
        lowest_cost["method"] = "Local DCL lowest material cost"
        rows.append(lowest_cost)

        best_test = min(locals_, key=lambda r: r["test_rmse"]).copy()
        best_test["method"] = "Local DCL best test RMSE"
        rows.append(best_test)

    print("\n" + "=" * 150)
    print("CONCRETE GLOBAL vs LOCAL DIAGNOSTIC COMPARISON")
    print("=" * 150)
    print(
        "Method | Iter | Deg | Train rel RMSE | Test rel RMSE | "
        "Cost | Model Obj | Slack | Slack Cost | Sklearn Strength | Weight | "
        "Hard Sklearn Feas | Soft Sklearn Feas | Weight Feas | "
        "Valid Hard | Valid Soft | Rows | Cols | Bin | Int | GenConstr | Nodes | Runtime | Work"
    )
    print("-" * 150)

    for r in rows:
        print(
            f"{r['method']} | "
            f"{r['iter']} | "
            f"{r['degree']} | "
            f"{_fmt(r.get('train_rel_rmse'))} | "
            f"{_fmt(r.get('test_rel_rmse'))} | "
            f"{_fmt(r.get('cost'))} | "
            f"{_fmt(r.get('model_objective'))} | "
            f"{_fmt(r.get('slack'))} | "
            f"{_fmt(r.get('slack_penalty_cost'))} | "
            f"{_fmt(r.get('sklearn_strength'))} | "
            f"{_fmt(r.get('weight'))} | "
            f"{r.get('feasible_sklearn_hard')} | "
            f"{r.get('feasible_sklearn_soft')} | "
            f"{r.get('feasible_weight')} | "
            f"{r.get('valid_hard_point_solution')} | "
            f"{r.get('valid_soft_solution')} | "
            f"{r.get('mip_rows')} | "
            f"{r.get('mip_cols')} | "
            f"{r.get('num_bin')} | "
            f"{r.get('num_int')} | "
            f"{r.get('num_genconstrs')} | "
            f"{_fmt(r.get('mip_nodes'), digits=0)} | "
            f"{_fmt(r.get('mip_runtime'))} | "
            f"{_fmt(r.get('mip_work'))}"
        )

    print("=" * 150)
    print(
        "Concrete hard selection rule: sklearn strength >= 45 and weight in [2230, 2450]; "
        "among hard-feasible rows, lower material cost is better."
    )
    print(
        "Soft feasibility only means sklearn strength + slack >= 45; use it for debugging, "
        "not as a final point-estimation solution unless you explicitly allow relaxation."
    )
    
    
# =========================================================
# Main
# =========================================================
def main():
    warnings.filterwarnings(
        "ignore",
        message="delta_grad == 0.0. Check if the approximated function is linear. "
                "If the function is linear better results can be obtained by defining "
                "the Hessian as zero instead of using quasi-Newton approximations.",
    )
    warnings.filterwarnings(
        "ignore",
        message="Singular Jacobian matrix. Using SVD decomposition to perform the factorizations.",
    )

    # Change this one line to run another case.
    problem_name = "rosenbrock"   # "wfp", "rosenbrock", or "concrete"

    if problem_name == "wfp":
        adjustable_problem_parameters = 0.6
        learning_model_type = "cart" #or "nn"
        x_init = None

    elif problem_name == "rosenbrock":
        adjustable_problem_parameters = 0.0
        learning_model_type = "polynom_regression"
        x_init = np.array([-1.0, -2.0, 1.0], dtype=float)

    elif problem_name == "concrete":
        adjustable_problem_parameters = CONCRETE_STRENGTH_REQ
        learning_model_type = CONCRETE_LEARNING_MODEL_TYPE
        x_init = None
    
    else:
        raise ValueError(f"Unknown problem_name='{problem_name}'")
    
    options = dcl.dcl_options()

    if problem_name == "concrete":
        problem, problem_model_type, X, y = dclp.setup_problem_concrete(
            "gurobi",
            shuffle_seed=SEED,
            strength_req=CONCRETE_STRENGTH_REQ,
            weight_lb=CONCRETE_WEIGHT_LB,
            weight_ub=CONCRETE_WEIGHT_UB,
            data_path=CONCRETE_DATA_PATH,
            use_soft_constraints=USE_SOFT_CONSTRAINTS,   # Use the standalone variable
            slack_penalty_weight=SLACK_PENALTY_WEIGHT    # Use the standalone variable
        )
    else:
        problem, problem_model_type, X, y = dclp.dcl_setup_problem(
            problem_name,
            learning_model_type,
            adjustable_problem_parameters,
            SEED,
        )

    options = dcl.dcl_options()
    options.trust_region = "convex_hull"
    options.learning_model_type = learning_model_type

    if problem_name == "wfp":
        options.max_iterations = 10
        options.max_model_degree = 4
        options.train_data_ratio = 0.70
        options.max_ndata = 100000
        options.rmse_thresh = 0.15
        options.rmse_target = 0.08
        options.predictor_check_enabled = False
        options.tree_epsilon = None
        options.reject_predictor_mismatch = False

    elif problem_name == "rosenbrock":
        # Old local Rosenbrock options, with standard RMSE and slack retained.
        options.max_iterations = 1
        options.max_model_degree = 10
        # Old x_new local data schedule:
        # len(x_init)=3 -> learning_size=2*n=6 -> initial local size = 5*6 = 30.
        # dcl_core then tries 30, 42, 54, 66, ... because it adds 2*learning_size = 12.
        #options.init_ndata_model = 5 * (2 * len(x_init))
        options.init_ndata_model = 100
        options.max_ndata = 100000

        options.train_data_ratio = 0.70

        # Old values, now interpreted using STANDARD RMSE.
        options.rmse_thresh = 1.0
        options.rmse_target = 0.1
        options.delta_min = 1e-6
        options.scipy_accept_failed_result = ACCEPT_FAILED_SCIPY_RESULTS

        # Slack is kept:
        #     local ghat(x) - slack <= 0, slack >= 0
        # objective used by the subproblem:
        #     original objective + scipy_slack_penalty * slack
        options.scipy_relax_learned_constraint = True
        options.scipy_slack_penalty = 1e4
        options.scipy_slack_upper_bound = 1e4

        # Old-compatible geometry. Convex hull still enforces the trust region.
        options.scipy_use_local_coordinate_bounds = False
        options.scipy_maxiter = 20000
        options.scipy_gtol = 1e-7
        options.scipy_xtol = 1e-7
        options.scipy_barrier_tol = 1e-7
        options.scipy_initial_constr_penalty = 100.0

    elif problem_name == "concrete":
        options.max_iterations = CONCRETE_DCL_ITERATIONS
        options.delta_min = 0.0
        options.init_ndata_model = 1030
        options.train_data_ratio = 0.7
        options.rmse_thresh = np.inf
        options.rmse_target = 0.01 #rel for now
 #      options.max_ndata = len(y)

        options.predictor_check_enabled = True

        # Keep this False for local DCL so the loop can continue.
        # We will select the best sklearn-feasible row after the run.
        options.reject_predictor_mismatch = False

        options.predictor_error_tol = 1e-4
        options.gurobi_feasibility_tol = 1e-9
        options.gurobi_int_feas_tol = 1e-9
        
        # ========================================================
        # NEW: Manual Slack Variable Configuration
        # ========================================================
        #options.use_soft_constraints = True      # Toggle relaxation on/off
        #options.slack_penalty_weight = 1000.0    # 'Big M' penalty for the objective function
        options.use_soft_constraints = USE_SOFT_CONSTRAINTS
        options.slack_penalty_weight = SLACK_PENALTY_WEIGHT
        # ========================================================

        if learning_model_type == "cart":
            # CART = one DecisionTreeRegressor.
            # Here, "degree" means max_depth.
            options.max_model_degree = 20
            options.tree_epsilon = 1e-4
        
        elif learning_model_type == "rf":
            options.max_model_degree = 20
            options.rf_n_estimators = 1
            options.rf_min_samples_leaf = 1
            options.tree_epsilon = 1e-4

        elif learning_model_type == "nn":
            options.max_model_degree = 4
            options.tree_epsilon = None
        
        else:
            raise NotImplementedError(
                f"Unknown concrete learning_model_type='{learning_model_type}'. "
                "Use 'cart', 'rf', or 'nn'."
            )

    _set_seeds(options, SEED)

    true_rosenbrock_diag = None
    old_global_diag = None
    new_global_diag = None
    old_global_pred = None
    new_global_pred = None

    if problem_name == "rosenbrock":
        res_true, true_hull_violation = solve_true_rosenbrock_over_convex_hull(
            problem,
            X,
            x_init=x_init,
        )
        x_true = np.asarray(res_true.x[: X.shape[1]], dtype=float)
        true_rosenbrock_diag = {
            "solution_x": x_true,
            "objective": float(problem["objective"](x_true)),
            "true_g": float(dclp.rosenbrock_constraint_value(x_true)),
            "hull_violation": float(true_hull_violation(x_true)),
            "solver_success": bool(res_true.success),
            "solver_message": str(res_true.message),
            "true_feasible": bool(dclp.rosenbrock_constraint_value(x_true) <= 1e-6),
        }

        print("\n" + "=" * 70)
        print("ROSENBROCK TRUE FULL-HULL REFERENCE")
        print("=" * 70)
        print("solution_x =", true_rosenbrock_diag["solution_x"])
        print("objective =", true_rosenbrock_diag["objective"])
        print("true g(x) =", true_rosenbrock_diag["true_g"])
        print("hull violation =", true_rosenbrock_diag["hull_violation"])
        print("solver success =", true_rosenbrock_diag["solver_success"])
        print("solver message =", true_rosenbrock_diag["solver_message"])
        print("=" * 70)

        if RUN_ROSENBROCK_OLD_GLOBAL:
            _, _, _, old_global_pred, _, old_global_diag = solve_old_global_rosenbrock(
                problem,
                X,
                y,
                degree=ROSENBROCK_OLD_GLOBAL_DEGREE,
                x_init=x_init,
            )
            print_global_rosenbrock_report(old_global_diag)

        if RUN_ROSENBROCK_NEW_GLOBAL:
            _, _, _, new_global_pred, _, new_global_diag = solve_new_global_rosenbrock(
                problem,
                X,
                y,
                max_degree=options.max_model_degree,
                train_data_ratio=options.train_data_ratio,
                seed=SEED,
                x_init=x_init,
            )
            print_global_rosenbrock_report(new_global_diag)

        print("DCL initial point:", x_init)
        print("DCL initial objective:", problem["objective"](x_init))
        print("DCL initial true g(x):", dclp.rosenbrock_constraint_value(x_init))
        print("DCL local initial size:", options.init_ndata_model)
        print("DCL local max data:", options.max_ndata)
        print("DCL local data schedule: 30, 42, 54, 66, ... until accuracy stops improving")
        print("DCL local model: unscaled PolynomialFeatures(include_bias=True) + LinearRegression")
        print("DCL local error metric: standard RMSE")
        print("DCL slack active:", bool(options.scipy_relax_learned_constraint))
        print("Slack penalty:", options.scipy_slack_penalty)
        print("Slack upper bound:", options.scipy_slack_upper_bound)
        print("SciPy initial point style: x=xit, lambda=0")
        print("-" * 70)

    global_concrete_diag = None
    if problem_name == "concrete" and RUN_CONCRETE_GLOBAL_BASELINE:
        global_concrete_diag = solve_global_concrete_full_hull(
            problem,
            X,
            y,
            options,
            strength_req=CONCRETE_STRENGTH_REQ,
        )

    dcls = dcl.dcl_solver(
        problem,
        problem_model_type,
        options,
        X,
        y,
        x_init,
    )
    dcls.solve()

    print("final x =", dcls.xit)
    print("final objective =", dcls.final_obj)
    print("return code =", dcls.return_code)

    if problem_name == "rosenbrock" and dcls.xit is not None:
        x_final = np.asarray(dcls.xit, dtype=float)
        print("final true g(x) =", dclp.rosenbrock_constraint_value(x_final))
        if true_rosenbrock_diag is not None:
            print(
                "objective gap vs hull-restricted true solution =",
                problem["objective"](x_final) - true_rosenbrock_diag["objective"],
            )

    dcl.dcl_print_iteration_log(dcls)
    
    if problem_name == "rosenbrock":
        print_one_iteration_dcl_polynomial(dcls)
    
    if problem_name == "concrete":
        print_concrete_dcl_report(dcls, strength_req=adjustable_problem_parameters)
        print_concrete_dcl_report_rows(dcls)
        print_concrete_global_local_comparison(
            global_concrete_diag,
            dcls,
            strength_req=CONCRETE_STRENGTH_REQ,
        )

    if problem_name == "rosenbrock":
        print_rosenbrock_dcl_report_rows(
            dcls,
            old_global_pred=old_global_pred,
            new_global_pred=new_global_pred,
        )
        print_rosenbrock_global_local_comparison(
            true_rosenbrock_diag,
            old_global_diag,
            new_global_diag,
            dcls,
            old_global_pred=old_global_pred,
            new_global_pred=new_global_pred,
        )


#if __name__ == "__main__":
#    main()

if __name__ == "__main__":
    # Automatically save everything printed by this program to a timestamped file.
    # The output is also still shown in the terminal.
    from datetime import datetime
    from pathlib import Path
    import sys
    import traceback

    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, data):
            for file in self.files:
                file.write(data)
                file.flush()

        def flush(self):
            for file in self.files:
                file.flush()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"dcl_output_{datetime.now():%Y%m%d_%H%M%S}.txt"

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    failed = False

    with output_file.open("w", encoding="utf-8") as file:
        sys.stdout = Tee(original_stdout, file)
        sys.stderr = Tee(original_stderr, file)

        try:
            main()
        except Exception:
            traceback.print_exc()
            failed = True
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    print(f"Output saved to: {output_file.resolve()}")

    if failed:
        sys.exit(1)
    
    