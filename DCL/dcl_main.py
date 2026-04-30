
# -*- coding: utf-8 -*-

import random
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize
from scipy.spatial import ConvexHull
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import dcl_core as dcl
import dcl_problem as dclp


SEED = 102

#   False = do not accept SciPy subproblem results unless the solver reports success.
#   True  = reproduce x_new-style behavior, but failed trial points may be accepted.
ACCEPT_FAILED_SCIPY_RESULTS = True


def _set_seeds(options, seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    options.seed = seed


def _as_numpy(X):
    return X.to_numpy(dtype=float) if hasattr(X, "to_numpy") else np.asarray(X, dtype=float)


def _relative_rmse(y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_true = np.asarray(y_true, dtype=float).ravel()
    denom = np.maximum(np.maximum(np.abs(y_pred), np.abs(y_true)), 1e-12)
    rel_err = (y_pred - y_true) / denom
    return np.linalg.norm(rel_err) / max(len(y_true), 1)


def _full_data_hull_constraint(X):
    """
    Return a scipy LinearConstraint for x in conv(X), together with a hull-violation function.

    scipy.spatial.ConvexHull uses inequalities

        A x + b <= 0

    for points inside the hull.
    """
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


def solve_true_rosenbrock_over_convex_hull(problem, X, x_init=None):
    """
    Solve the true Rosenbrock reference problem restricted to conv(X):

        min  x0
        s.t. true_g(x) <= 0
             x in conv(X)
             problem bounds on x

    This is a reference/oracle solve. It does not use a learned surrogate.
    """
    Xnp = _as_numpy(X)
    n = int(Xnp.shape[1])

    if "true_constraint" not in problem:
        raise KeyError(
            "problem must contain problem['true_constraint'] for the true Rosenbrock reference solve."
        )

    hull_constraint, hull_violation, _, _ = _full_data_hull_constraint(Xnp)

    bounds = Bounds(problem["bounds_lb"], problem["bounds_ub"])

    true_constraint = NonlinearConstraint(
        problem["true_constraint"],
        -np.inf,
        0.0,
    )

    def obj_fun(x):
        return float(problem["objective"](x))

    def true_g_fun(x):
        return float(problem["true_constraint"](x))

    lb_x = np.asarray(problem["bounds_lb"], dtype=float)
    ub_x = np.asarray(problem["bounds_ub"], dtype=float)

    true_g_values = np.array([true_g_fun(Xnp[i, :]) for i in range(Xnp.shape[0])], dtype=float)
    in_box = np.all((Xnp >= lb_x) & (Xnp <= ub_x), axis=1)
    feasible_data_indices = np.where(in_box & (true_g_values <= 0.0))[0]

    starts = []

    if feasible_data_indices.size > 0:
        ordered = feasible_data_indices[np.argsort(Xnp[feasible_data_indices, 0])]
        for i in ordered[:10]:
            starts.append(Xnp[i, :].copy())

    if x_init is not None:
        nearest = int(
            np.argmin(
                np.linalg.norm(
                    Xnp - np.asarray(x_init, dtype=float),
                    axis=1,
                )
            )
        )
        starts.append(Xnp[nearest, :].copy())

    # Mean is guaranteed to be in conv(X).
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
            options={
                "disp": False,
                "maxiter": 1000,
                "ftol": 1e-10,
            },
        )

        if not hasattr(res, "x") or res.x is None:
            continue

        x_res = np.asarray(res.x[:n], dtype=float)

        if not np.all(np.isfinite(x_res)):
            continue

        true_g_res = true_g_fun(x_res)
        hull_viol = hull_violation(x_res)

        is_feasible = (
            true_g_res <= 1e-6
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

    raise RuntimeError("Hull-restricted true Rosenbrock reference solve failed from all starts.")


def solve_global_surrogate_rosenbrock_over_convex_hull(
    problem,
    X,
    y,
    degree=4,
    x_init=None,
):
    """
    Fit one global polynomial surrogate using all data and solve it over conv(X):

        min  x0
        s.t. global_surrogate_g(x) <= 0
             x in conv(X)
             problem bounds on x

    This is the missing global-baseline counterpart to DCL.
    """
    Xnp = _as_numpy(X)
    ynp = _as_numpy(y).ravel() if hasattr(y, "to_numpy") else np.asarray(y, dtype=float).ravel()

    n = int(Xnp.shape[1])

    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(Xnp)
    global_model = LinearRegression().fit(X_poly, ynp)

    def global_pred(x):
        x = np.asarray(x, dtype=float).reshape(1, -1)
        return float(global_model.predict(poly.transform(x))[0])

    train_pred = global_model.predict(poly.transform(Xnp))
    train_mse = float(np.mean((train_pred - ynp) ** 2))
    train_rel_rmse = float(_relative_rmse(train_pred, ynp))

    hull_constraint, hull_violation, _, _ = _full_data_hull_constraint(Xnp)

    surrogate_constraint = NonlinearConstraint(
        global_pred,
        -np.inf,
        0.0,
    )

    bounds = Bounds(problem["bounds_lb"], problem["bounds_ub"])

    def obj_fun(x):
        return float(problem["objective"](x))

    lb_x = np.asarray(problem["bounds_lb"], dtype=float)
    ub_x = np.asarray(problem["bounds_ub"], dtype=float)

    pred_values = np.array([global_pred(Xnp[i, :]) for i in range(Xnp.shape[0])], dtype=float)
    in_box = np.all((Xnp >= lb_x) & (Xnp <= ub_x), axis=1)
    pred_feasible_indices = np.where(in_box & (pred_values <= 0.0))[0]

    starts = []

    if pred_feasible_indices.size > 0:
        ordered = pred_feasible_indices[np.argsort(Xnp[pred_feasible_indices, 0])]
        for i in ordered[:10]:
            starts.append(Xnp[i, :].copy())

    if x_init is not None:
        nearest = int(
            np.argmin(
                np.linalg.norm(
                    Xnp - np.asarray(x_init, dtype=float),
                    axis=1,
                )
            )
        )
        starts.append(Xnp[nearest, :].copy())

    # Mean is guaranteed to be in conv(X).
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
            options={
                "disp": False,
                "maxiter": 1000,
                "ftol": 1e-10,
            },
        )

        if not hasattr(res, "x") or res.x is None:
            continue

        x_res = np.asarray(res.x[:n], dtype=float)

        if not np.all(np.isfinite(x_res)):
            continue

        pred_res = global_pred(x_res)
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
        chosen_res = best_feasible_res
    elif best_any_res is not None:
        chosen_res = best_any_res
    else:
        raise RuntimeError("Hull-restricted global surrogate solve failed from all starts.")

    diagnostics = {
        "train_mse": train_mse,
        "train_rel_rmse": train_rel_rmse,
    }

    return chosen_res, global_model, poly, global_pred, hull_violation, diagnostics


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

    # =========================
    # Choose problem
    # =========================
    problem_name = "rosenbrock"   # "wfp" or "rosenbrock"

    if problem_name == "wfp":
        adjustable_problem_parameters = 0.6   # palatability threshold
        learning_model_type = "cart"
        x_init = None

    elif problem_name == "rosenbrock":
        adjustable_problem_parameters = 0.0   # unused for Rosenbrock
        learning_model_type = "polynom_regression"
        x_init = np.array([-1.0, -2.0, 1.0], dtype=float)

    else:
        raise ValueError(f"Unknown problem_name='{problem_name}'")

    # =========================
    # Set problem
    # =========================
    problem, problem_model_type, X, y = dclp.dcl_setup_problem(
        problem_name,
        learning_model_type,
        adjustable_problem_parameters,
        SEED,
    )

    # =========================
    # Set options
    # =========================
    options = dcl.dcl_options()
    options.trust_region = "convex_hull"
    options.learning_model_type = learning_model_type

    if problem_name == "wfp":
        options.max_iterations = 10
        options.max_model_degree = 4

    elif problem_name == "rosenbrock":
        options.max_iterations = 50
        options.max_model_degree = 4

        # x_new equivalent:
        # learning_size = 2*n and ndata_model_0 = min(max_ndata, min(5*learning_size, ndata)).
        options.init_ndata_model = 5 * (2 * len(x_init))

        options.train_data_ratio = 0.70
        options.rmse_thresh = 0.9
        options.rmse_target = 0.1
        options.delta_min = 1e-6
        options.max_ndata = 100000
      
        options.scipy_accept_failed_result = ACCEPT_FAILED_SCIPY_RESULTS

    _set_seeds(options, SEED)

    # =========================
    # Rosenbrock reference and global surrogate baseline
    # =========================
    if problem_name == "rosenbrock":
        # 1) True/reference solve over full data convex hull.
        res_true, true_hull_violation = solve_true_rosenbrock_over_convex_hull(
            problem,
            X,
            x_init=x_init,
        )

        x_true = np.asarray(res_true.x[: X.shape[1]], dtype=float)

        print("Hull-restricted true solution:", x_true)
        print("Hull-restricted true objective:", problem["objective"](x_true))
        print("Hull-restricted true g(x):", dclp.rosenbrock_constraint_value(x_true))
        print("Hull-restricted true hull violation:", true_hull_violation(x_true))
        print("Hull-restricted true solver success:", bool(res_true.success))
        print("-" * 70)

        # 2) Global learned surrogate solve over full data convex hull.
        (
            res_global,
            global_model,
            global_poly,
            global_pred,
            global_hull_violation,
            global_diagnostics,
        ) = solve_global_surrogate_rosenbrock_over_convex_hull(
            problem,
            X,
            y,
            degree=4,
            x_init=x_init,
        )

        x_global = np.asarray(res_global.x[: X.shape[1]], dtype=float)

        print("Hull-restricted global surrogate solution:", x_global)
        print("Hull-restricted global surrogate objective:", problem["objective"](x_global))
        print("Hull-restricted global surrogate predicted g(x):", global_pred(x_global))
        print("Hull-restricted global surrogate true g(x):", dclp.rosenbrock_constraint_value(x_global))
        print("Hull-restricted global surrogate hull violation:", global_hull_violation(x_global))
        print("Hull-restricted global surrogate solver success:", bool(res_global.success))
        print("Global surrogate train MSE:", global_diagnostics["train_mse"])
        print("Global surrogate train relative RMSE:", global_diagnostics["train_rel_rmse"])

        if global_pred(x_global) > 1e-5 or global_hull_violation(x_global) > 1e-5:
            print(
                "WARNING: the returned global-surrogate point is not cleanly feasible "
                "for the surrogate convex-hull problem."
            )

        print("-" * 70)

    # =========================
    # Solve with DCL
    # =========================
    dcls = dcl.dcl_solver(problem, problem_model_type, options, X, y, x_init)
    dcls.solve()

    print("final x =", dcls.xit)
    print("final objective =", dcls.final_obj)
    print("return code =", dcls.return_code)

    if problem_name == "rosenbrock" and dcls.xit is not None:
        x_final = np.asarray(dcls.xit, dtype=float)
        print("final true g(x) =", dclp.rosenbrock_constraint_value(x_final))

        if "x_true" in locals():
            print("objective gap vs hull-restricted true solution =", problem["objective"](x_final) - problem["objective"](x_true))

    dcl.dcl_print_iteration_log(dcls)


if __name__ == "__main__":
    main()