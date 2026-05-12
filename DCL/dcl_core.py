# -*- coding: utf-8 -*-

import math
import random

import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from scipy.optimize import LinearConstraint, NonlinearConstraint, Bounds, minimize

try:
    import gurobipy as gp
    from gurobipy import GRB
    from gurobi_ml import add_predictor_constr
except ImportError:
    gp = None
    GRB = None
    add_predictor_constr = None


ENABLE_WEIGHTING = 0


# =========================================================
# Options
# =========================================================
class dcl_options:
    trust_region = "convex_hull"
    learning_model_type = "cart"
    train_data_ratio = 0.7
    init_data_ratio = 0.1
    init_ndata_model = None
    max_iterations = 1
    x_init = None
    delta_min = 1e-6
    rmse_thresh = 0.1
    rmse_target = 0.001
    max_model_degree = 4
    seed = 0
    max_ndata = 1000000
    scipy_accept_failed_result = False

    # SciPy/Rosenbrock relaxation and solver settings.
    scipy_relax_learned_constraint = True
    scipy_slack_penalty = 1e4
    scipy_slack_upper_bound = 1e4

    scipy_use_local_coordinate_bounds = True
    scipy_maxiter = 5000
    scipy_gtol = 1e-8
    scipy_xtol = 1e-8
    scipy_barrier_tol = 1e-8
    scipy_initial_constr_penalty = 100.0

    # Concrete RF options.
    rf_n_estimators = 100
    rf_min_samples_leaf = 1

    # Predictor embedding diagnostics.
    predictor_check_enabled = False
    tree_epsilon = None
    predictor_error_tol = 1e-4
    reject_predictor_mismatch = False
    gurobi_feasibility_tol = None
    gurobi_int_feas_tol = None


# =========================================================
# Local model container
# =========================================================
class dcl_local_model:
    def __init__(self):
        self.data_indices = None
        self.learned_model = None
        self.poly_features = None

        self.test_rmse = None
        self.train_rmse = None
        self.test_mse = None
        self.train_mse = None
        self.test_rel_rmse = None
        self.train_rel_rmse = None

        self.model_degree = 1
        self.ndata_model = None

        self.solution_x = None
        self.solution_y = None
        self.solution_slack = None

        self.mip_cols = None
        self.mip_rows = None
        self.mip_work = None
        self.mip_nodes = None
        self.mip_runtime = None
        self.num_bin = None
        self.num_int = None
        self.num_genconstrs = None

        self.solver_status = None
        self.solver_success = None
        self.solver_message = None
        self.solver_nit = None


# =========================================================
# Metrics
# =========================================================
def dcl_mse(y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_true = np.asarray(y_true, dtype=float).ravel()

    return float(np.mean((y_pred - y_true) ** 2))


def dcl_rmse(y_pred, y_true):
    return float(np.sqrt(dcl_mse(y_pred, y_true)))


def dcl_relative_rmse(y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_true = np.asarray(y_true, dtype=float).ravel()

    denom = np.maximum(np.maximum(np.abs(y_pred), np.abs(y_true)), 1.0)
    rel_err = (y_pred - y_true) / denom
    
    rel_mse = float(np.mean(rel_err ** 2))

    return float(np.sqrt(rel_mse))


# =========================================================
# Prediction helper
# =========================================================
def dcl_predict_local_model(accepted_model, x):
    x = np.asarray(x, dtype=float).reshape(1, -1)

    if accepted_model.poly_features is not None:
        x_feat = accepted_model.poly_features.transform(x)
        return float(accepted_model.learned_model.predict(x_feat)[0])

    return float(accepted_model.learned_model.predict(x)[0])


def _is_tree_or_rf_model(model):
    return isinstance(model, (DecisionTreeRegressor, RandomForestRegressor))


def _maybe_set_gurobi_tolerances(model, options):
    if options is None:
        return

    feasibility_tol = getattr(options, "gurobi_feasibility_tol", None)
    int_feas_tol = getattr(options, "gurobi_int_feas_tol", None)

    if feasibility_tol is not None:
        model.Params.FeasibilityTol = float(feasibility_tol)

    if int_feas_tol is not None:
        model.Params.IntFeasTol = float(int_feas_tol)


def _predictor_epsilon_kwargs(learned_model, options):
    predictor_kwargs = {}

    if options is None:
        return predictor_kwargs

    tree_epsilon = getattr(options, "tree_epsilon", None)

    if tree_epsilon is not None and _is_tree_or_rf_model(learned_model):
        predictor_kwargs["epsilon"] = float(tree_epsilon)

    return predictor_kwargs


def _add_predictor_constr_safely(model, learned_model, input_vars, output_vars, options):
    predictor_kwargs = _predictor_epsilon_kwargs(learned_model, options)

    try:
        return add_predictor_constr(
            model,
            learned_model,
            input_vars=input_vars,
            output_vars=output_vars,
            **predictor_kwargs,
        )
    except TypeError:
        return add_predictor_constr(
            model,
            learned_model,
            input_vars=input_vars,
            output_vars=output_vars,
        )


def _compute_predictor_diagnostics(pred_constr, learned_model, x_sol, gurobi_output):
    sklearn_output = float(
        np.asarray(
            learned_model.predict(np.asarray(x_sol, dtype=float).reshape(1, -1)),
            dtype=float,
        ).ravel()[0]
    )

    manual_error = float(abs(float(gurobi_output) - sklearn_output))

    try:
        gurobi_ml_error = float(
            np.max(
                np.asarray(pred_constr.get_error(), dtype=float)
            )
        )
    except Exception:
        gurobi_ml_error = np.nan

    return sklearn_output, manual_error, gurobi_ml_error


# =========================================================
# Solver
# =========================================================
class dcl_solver:
    def __init__(self, problem_, problem_model_type_, options_, X_, y_, x_init_):
        self.local_models = []
        self.problem_model = problem_
        self.problem_model_type = problem_model_type_
        self.options = options_
        self.X = X_
        self.y = y_
        self.x_init = x_init_

        self.return_code = 0
        self.xit = None
        self.final_obj = None
        self.iteration_log = []

    def solve(self):
        X = self.X
        y = self.y
        options = self.options

        Xnp = X.to_numpy(dtype=float) if hasattr(X, "to_numpy") else np.asarray(X, dtype=float)
        ynp = y.to_numpy(dtype=float) if hasattr(y, "to_numpy") else np.asarray(y, dtype=float)

        n = int(Xnp.shape[1])

        if self.x_init is None:
            self.x_init = dcl_init(Xnp)

        xit = np.asarray(self.x_init, dtype=float).copy()

        true_constraint = None
        if isinstance(self.problem_model, dict):
            true_constraint = self.problem_model.get("true_constraint", None)

        if callable(true_constraint):
            print(
                "iteration 0:",
                " trial_x: ", xit,
                "f(x): ", float(self.problem_model["objective"](xit)),
                "true_g(x): ", float(true_constraint(xit)),
            )

        dists = np.linalg.norm(Xnp - xit, axis=1)
        sorted_indices = np.argsort(dists).tolist()

        ndata = len(ynp)
        learning_size = 2 * n

        if options.init_ndata_model is not None:
            ndata_model_0 = int(options.init_ndata_model)
            ndata_model_0 = min(options.max_ndata, ndata_model_0, ndata)
        else:
            min_local_size = max(2 * n, 4)
            ndata_model_0 = max(min_local_size, math.ceil(options.init_data_ratio * ndata))
            ndata_model_0 = min(options.max_ndata, ndata_model_0, ndata)

        for i in range(options.max_iterations):
            x_start = xit.copy()
            ndata_model = ndata_model_0

            if ENABLE_WEIGHTING:
                weights = update_weights(xit, dists)
                ybar = weights * ynp
                Xbar = weights.reshape(-1, 1) * Xnp
            else:
                Xbar = X
                ybar = y

            best_test_rmse = np.inf
            best_train_rmse = None
            best_test_mse = None
            best_train_mse = None
            best_test_rel_rmse = np.inf
            best_train_rel_rmse = None
            best_ndata_model = 0
            local_model = None
            p_local = None
            model_degree = 1

            while ndata_model <= min(ndata, options.max_ndata):
                (
                    md_tmp,
                    test_rmse_tmp,
                    train_rmse_tmp,
                    lm_tmp,
                    p_tmp,
                    train_mse_tmp,
                    test_mse_tmp,
                    train_rel_rmse_tmp,
                    test_rel_rmse_tmp,
                ) = dcl_set_model_degree(
                    self.options.learning_model_type,
                    1,
                    sorted_indices,
                    ndata_model,
                    Xbar,
                    ybar,
                    self.options,
                )

                if test_rel_rmse_tmp < best_test_rel_rmse:
                    best_test_rmse = test_rmse_tmp
                    best_train_rmse = train_rmse_tmp
                    best_train_mse = train_mse_tmp
                    best_test_mse = test_mse_tmp
                    best_train_rel_rmse = train_rel_rmse_tmp
                    best_test_rel_rmse = test_rel_rmse_tmp
                    local_model = lm_tmp
                    p_local = p_tmp
                    model_degree = md_tmp
                    best_ndata_model = ndata_model
                else:
                    break

                if best_test_rel_rmse <= self.options.rmse_target:
                    break

                ndata_model += 2 * learning_size

            if local_model is None:
                print("Stopping: could not fit a local model.")
                self.return_code = -2
                self.xit = xit.copy()
                return

            idata_model = sorted_indices[:best_ndata_model]

            accepted_model = dcl_local_model()
            accepted_model.data_indices = idata_model
            accepted_model.learned_model = local_model
            accepted_model.poly_features = p_local
            accepted_model.test_rmse = best_test_rmse
            accepted_model.train_rmse = best_train_rmse
            accepted_model.test_mse = best_test_mse
            accepted_model.train_mse = best_train_mse
            accepted_model.test_rel_rmse = best_test_rel_rmse
            accepted_model.train_rel_rmse = best_train_rel_rmse
            accepted_model.model_degree = model_degree
            accepted_model.ndata_model = best_ndata_model

            self.local_models.append(accepted_model)
            print("Accepted model with degree ",model_degree," trained with ",best_ndata_model,"data points.")

            if best_test_rel_rmse > self.options.rmse_thresh:
                print("Stopping: local model not accurate enough (rmse =", best_test_rel_rmse, ")")
                self.return_code = -2
                self.xit = xit.copy()
                return

            if self.problem_model_type == "gurobi":
                x_trial, res_loc = dcl_solve_subproblem(
                    self.problem_model,
                    accepted_model,
                    self.options.trust_region,
                    self.X,
                    options=self.options,
                )

            elif self.problem_model_type == "scipy":
                x_trial, res_loc = dcl_solve_subproblem_scipy(
                    self.problem_model,
                    accepted_model,
                    self.options.trust_region,
                    self.X,
                    xit,
                    accept_failed_result=self.options.scipy_accept_failed_result,
                    options=self.options,
                )

            else:
                raise NotImplementedError(
                    f"Unknown problem_model_type='{self.problem_model_type}'."
                )

            if x_trial is None:
                print(f"Stopping: subproblem not solved (status={res_loc.get('status')})")
                if res_loc.get("message") is not None:
                    print("message:", res_loc.get("message"))
                self.return_code = -3
                self.xit = xit.copy()
                return

            accepted_model.solution_x = x_trial[:n].copy()
            accepted_model.solution_y = float(x_trial[n])
            accepted_model.solution_slack = float(res_loc.get("slack", 0.0))

            accepted_model.mip_cols = res_loc.get("mip_cols")
            accepted_model.mip_rows = res_loc.get("mip_rows")
            accepted_model.mip_work = res_loc.get("mip_work")
            accepted_model.mip_nodes = res_loc.get("mip_nodes")
            accepted_model.mip_runtime = res_loc.get("mip_runtime")
            accepted_model.num_bin = res_loc.get("num_bin")
            accepted_model.num_int = res_loc.get("num_int")
            accepted_model.num_genconstrs = res_loc.get("num_genconstrs")
            accepted_model.solver_status = res_loc.get("status")
            accepted_model.solver_success = res_loc.get("success")
            accepted_model.solver_message = res_loc.get("message")
            accepted_model.solver_nit = res_loc.get("nit")

            local_pred_start = None
            local_pred_solution = None
            true_g_start = None
            true_g_solution = None

            try:
                local_pred_start = dcl_predict_local_model(accepted_model, x_start)
                local_pred_solution = dcl_predict_local_model(
                    accepted_model,
                    accepted_model.solution_x,
                )
            except Exception:
                local_pred_solution = res_loc.get("output")

            if callable(true_constraint):
                true_g_start = float(true_constraint(x_start))
                true_g_solution = float(true_constraint(accepted_model.solution_x))

            self.iteration_log.append({
                "iter": i,
                "start_x": x_start.copy(),
                "objective": res_loc.get("obj"),
                "original_objective": res_loc.get("original_objective"),
                "relaxed_objective": res_loc.get("relaxed_objective"),
                "data_size": accepted_model.ndata_model,
                "degree": accepted_model.model_degree,
                "test_mse": accepted_model.test_mse,
                "train_mse": accepted_model.train_mse,
                "test_rmse": accepted_model.test_rmse,
                "train_rmse": accepted_model.train_rmse,
                "test_rel_rmse": accepted_model.test_rel_rmse,
                "train_rel_rmse": accepted_model.train_rel_rmse,
                "mip_cols": accepted_model.mip_cols,
                "mip_rows": accepted_model.mip_rows,
                "mip_work": accepted_model.mip_work,
                "mip_nodes": accepted_model.mip_nodes,
                "mip_runtime": accepted_model.mip_runtime,
                "num_bin": accepted_model.num_bin,
                "num_int": accepted_model.num_int,
                "num_genconstrs": accepted_model.num_genconstrs,
                "solver_status": accepted_model.solver_status,
                "solver_success": accepted_model.solver_success,
                "solver_message": accepted_model.solver_message,
                "solver_nit": accepted_model.solver_nit,
                "solution_x": accepted_model.solution_x.copy(),
                "solution_y": accepted_model.solution_y,
                "sklearn_output": res_loc.get("sklearn_output"),
                "manual_predictor_error": res_loc.get("manual_predictor_error"),
                "gurobi_ml_predictor_error": res_loc.get("gurobi_ml_predictor_error"),
                "predictor_error": res_loc.get("predictor_error"),
                "relaxation_slack": accepted_model.solution_slack,
                "relaxed_constraint_residual": res_loc.get("relaxed_constraint_residual"),
                "local_pred_start": local_pred_start,
                "local_pred_solution": local_pred_solution,
                "true_g_start": true_g_start,
                "true_g_solution": true_g_solution,
            })

            step = np.linalg.norm(x_trial[:n] - xit)

            if callable(true_constraint):
                print(
                    f"iter {i}: step = {step:.3e}, obj = {res_loc.get('obj')}, "
                    f"true_g = {true_g_solution}, local_g = {local_pred_solution}, "
                    f"slack = {accepted_model.solution_slack}"
                )
            else:
                print(
                    f"iter {i}: step = {step:.3e}, obj = {res_loc.get('obj')}, "
                    f"output = {accepted_model.solution_y}"
                )

            xit = x_trial[:n].copy()
            self.xit = xit.copy()
            self.final_obj = res_loc.get("obj")

            if step < options.delta_min:
                print("No significant progress; terminating.")
                self.return_code = 0
                return

            dists = np.linalg.norm(Xnp - xit, axis=1)
            sorted_indices = np.argsort(dists).tolist()

        self.xit = xit.copy()
        self.return_code = -1
        return


# =========================================================
# Initial point
# =========================================================
def dcl_init(X, size=None):
    if hasattr(X, "to_numpy"):
        Xnp = X.to_numpy(dtype=float)
    else:
        Xnp = np.asarray(X, dtype=float)

    if Xnp.size == 0:
        raise ValueError("X is empty, cannot compute arithmetic average.")

    if Xnp.ndim != 2:
        raise ValueError(f"X must be 2D. Got shape {Xnp.shape}")

    x0 = Xnp.mean(axis=0)

    if size is not None and len(x0) != size:
        print(f"Warning: dcl_init returned length {len(x0)} but size={size}")

    return x0



# =========================================================
# Gurobi subproblem
# =========================================================
def dcl_solve_subproblem(problem_model, accepted_model, trust_region, X, *, options=None):
    if gp is None or GRB is None or add_predictor_constr is None:
        raise ImportError(
            "The Gurobi path requires gurobipy and gurobi_ml. "
            "Install them or use a SciPy problem such as Rosenbrock."
        )

    m = problem_model.copy()

    # X -> numpy
    if hasattr(X, "to_numpy"):
        Xnp = X.to_numpy(dtype=float)
    else:
        Xnp = np.asarray(X, dtype=float)

    # number of input vars
    n = int(Xnp.shape[1])

    # get copied vars by names
    input_vars = [m.getVarByName(f"input_vars[{i}]") for i in range(n)]

    if any(v is None for v in input_vars):
        raise KeyError("Some 'input_vars[i]' were not found in copied model.")

    out0 = m.getVarByName("output_vars[0]")

    if out0 is None:
        raise KeyError("Could not find 'output_vars[0]' in copied model.")

    inp = np.array(input_vars, dtype=object).reshape(1, -1)
    out = np.array([out0], dtype=object).reshape(1, 1)

    # Old behavior: direct gurobi_ml embedding.
    add_predictor_constr(
        m,
        accepted_model.learned_model,
        input_vars=inp,
        output_vars=out,
    )

    # trust region
    if trust_region == "convex_hull":
        idx = list(accepted_model.data_indices) if accepted_model.data_indices is not None else []
        k = len(idx)

        if k == 0:
            raise ValueError(
                "accepted_model.data_indices is empty; cannot build convex hull trust region."
            )

        lam = m.addMVar(
            shape=k,
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name="lam_tr",
        )

        m.addConstr(
            gp.quicksum(lam[i] for i in range(k)) == 1.0,
            name="tr_sum",
        )

        for j in range(n):
            m.addConstr(
                input_vars[j]
                == gp.quicksum(lam[i] * float(Xnp[idx[i], j]) for i in range(k)),
                name=f"tr_x[{j}]",
            )

    elif trust_region in (None, "none"):
        pass

    else:
        raise NotImplementedError(f"Unknown trust_region='{trust_region}'.")

    # finalize model so size attributes are correct
    m.update()

    mip_cols = int(m.NumVars)
    mip_rows = int(m.NumConstrs)
    num_bin = int(m.NumBinVars)
    num_int = int(m.NumIntVars)
    num_genconstrs = int(m.NumGenConstrs)

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        return None, {
            "status": int(m.Status),
            "success": False,
            "message": f"Gurobi status {int(m.Status)}",
            "mip_cols": mip_cols,
            "mip_rows": mip_rows,
            "num_bin": num_bin,
            "num_int": num_int,
            "num_genconstrs": num_genconstrs,
            "slack": 0.0,
        }

    z_sol = np.array([v.X for v in input_vars], dtype=float)
    y_sol = float(out0.X)
    x_trial = np.concatenate([z_sol, np.array([y_sol])])

    return x_trial, {
        "status": int(m.Status),
        "success": True,
        "message": "OPTIMAL",
        "obj": float(m.ObjVal),
        "original_objective": float(m.ObjVal),
        "relaxed_objective": float(m.ObjVal),
        "output": y_sol,
        "sklearn_output": None,
        "manual_predictor_error": None,
        "gurobi_ml_predictor_error": None,
        "predictor_error": None,
        "slack": 0.0,
        "relaxed_constraint_residual": y_sol,
        "mip_cols": mip_cols,
        "mip_rows": mip_rows,
        "num_bin": num_bin,
        "num_int": num_int,
        "num_genconstrs": num_genconstrs,
        "mip_work": float(m.Work),
        "mip_nodes": float(m.NodeCount),
        "mip_runtime": float(m.Runtime),
    }


# =========================================================
# Gurobi subproblem
# =========================================================
def dcl_solve_subproblem_scipy(
    problem_model,
    accepted_model,
    trust_region,
    X,
    xit,
    *,
    accept_failed_result=False,
    options=None,
):
    if hasattr(X, "to_numpy"):
        Xnp = X.to_numpy(dtype=float)
    else:
        Xnp = np.asarray(X, dtype=float)

    n = int(Xnp.shape[1])

    idx = list(accepted_model.data_indices) if accepted_model.data_indices is not None else []
    k = len(idx)

    if k == 0:
        return None, {
            "status": -1,
            "success": False,
            "message": "empty local subset",
        }

    if trust_region != "convex_hull":
        raise NotImplementedError(
            "SciPy path currently supports only trust_region='convex_hull'."
        )

    relax_learned_constraint = bool(
        getattr(options, "scipy_relax_learned_constraint", True)
    )

    slack_penalty = float(
        getattr(options, "scipy_slack_penalty", 1e4)
    )

    slack_upper_bound = float(
        getattr(options, "scipy_slack_upper_bound", 1e4)
    )

    use_local_coordinate_bounds = bool(
        getattr(options, "scipy_use_local_coordinate_bounds", True)
    )

    scipy_maxiter = int(
        getattr(options, "scipy_maxiter", 5000)
    )

    scipy_gtol = float(
        getattr(options, "scipy_gtol", 1e-8)
    )

    scipy_xtol = float(
        getattr(options, "scipy_xtol", 1e-8)
    )

    scipy_barrier_tol = float(
        getattr(options, "scipy_barrier_tol", 1e-8)
    )

    scipy_initial_constr_penalty = float(
        getattr(options, "scipy_initial_constr_penalty", 100.0)
    )

    Xloc = Xnp[idx, :]

    slack_index = n + k if relax_learned_constraint else None
    pdim = n + k + (1 if relax_learned_constraint else 0)

    def original_ghat_fun(p):
        return float(dcl_predict_local_model(accepted_model, p[:n]))

    def relaxed_ghat_fun(p):
        value = original_ghat_fun(p)

        if relax_learned_constraint:
            value -= float(p[slack_index])

        return float(value)

    def obj_fun(p):
        value = float(problem_model["objective"](p[:n]))

        if relax_learned_constraint:
            value += slack_penalty * float(p[slack_index])

        return float(value)

    def obj_jac(p):
        grad = np.zeros(pdim, dtype=float)
        grad[0] = 1.0

        if relax_learned_constraint:
            grad[slack_index] = slack_penalty

        return grad

    A = np.zeros((n, pdim), dtype=float)
    A[:, :n] = -np.eye(n)
    A[:, n:n + k] = Xloc.T

    lc_x = LinearConstraint(
        A,
        lb=np.zeros(n),
        ub=np.zeros(n),
    )

    A_sum = np.zeros((1, pdim), dtype=float)
    A_sum[0, n:n + k] = 1.0

    lc_sum = LinearConstraint(
        A_sum,
        lb=np.array([1.0]),
        ub=np.array([1.0]),
    )

    jac_sparsity = np.zeros((1, pdim), dtype=bool)
    jac_sparsity[0, :n] = True

    if relax_learned_constraint:
        jac_sparsity[0, slack_index] = True

    nlc = NonlinearConstraint(
        relaxed_ghat_fun,
        -np.inf,
        0.0,
        finite_diff_jac_sparsity=jac_sparsity,
    )

    lb_x = np.asarray(problem_model["bounds_lb"], dtype=float)
    ub_x = np.asarray(problem_model["bounds_ub"], dtype=float)

    if use_local_coordinate_bounds:
        local_lb = np.min(Xloc, axis=0)
        local_ub = np.max(Xloc, axis=0)
        lb_x_eff = np.maximum(lb_x, local_lb)
        ub_x_eff = np.minimum(ub_x, local_ub)
    else:
        lb_x_eff = lb_x
        ub_x_eff = ub_x

    lb_vec_parts = [
        lb_x_eff,
        np.zeros(k),
    ]

    ub_vec_parts = [
        ub_x_eff,
        np.ones(k),
    ]

    if relax_learned_constraint:
        lb_vec_parts.append(np.array([0.0]))
        ub_vec_parts.append(np.array([slack_upper_bound]))

    lb_vec = np.concatenate(lb_vec_parts)
    ub_vec = np.concatenate(ub_vec_parts)

    # Old x_new-style SciPy initialization:
    #   x starts at the current iterate,
    #   all convex-hull lambda variables start at zero.
    # This is intentionally infeasible for sum(lambda)=1, but trust-constr
    # is allowed to repair feasibility during the solve.
    xit_arr = np.asarray(xit, dtype=float).copy()
    lam0 = np.zeros(k, dtype=float)

    p_init_parts = [
        xit_arr,
        lam0,
    ]

    if relax_learned_constraint:
        local_pred_init = float(dcl_predict_local_model(accepted_model, xit_arr))
        s0 = min(
            slack_upper_bound,
            max(0.0, local_pred_init) + 1e-8,
        )
        p_init_parts.append(np.array([s0], dtype=float))

    p_init = np.concatenate(p_init_parts)

    res = minimize(
        obj_fun,
        p_init,
        jac=obj_jac,
        bounds=Bounds(lb_vec, ub_vec),
        constraints=[nlc, lc_x, lc_sum],
        method="trust-constr",
        options={
            "disp": False,
            "maxiter": scipy_maxiter,
            "gtol": scipy_gtol,
            "xtol": scipy_xtol,
            "barrier_tol": scipy_barrier_tol,
            "initial_constr_penalty": scipy_initial_constr_penalty,
            "sparse_jacobian": True,
        },
    )

    raw_res_x = getattr(res, "x", None)

    has_usable_x = (
        raw_res_x is not None
        and np.asarray(raw_res_x, dtype=float).size >= pdim
        and np.all(np.isfinite(np.asarray(raw_res_x[:n], dtype=float)))
    )

    success = bool(getattr(res, "success", False))

    diagnostics = {
        "status": int(getattr(res, "status", -1)),
        "success": success,
        "message": str(getattr(res, "message", "SciPy subproblem failed")),
        "nit": int(getattr(res, "nit", -1)),
        "nfev": int(getattr(res, "nfev", -1)),
        "constr_violation": float(getattr(res, "constr_violation", np.nan)),
        "optimality": float(getattr(res, "optimality", np.nan)),
        "execution_time": float(getattr(res, "execution_time", np.nan)),
    }

    if (not has_usable_x) or ((not success) and (not accept_failed_result)):
        return None, diagnostics

    p_sol = np.asarray(raw_res_x, dtype=float)

    x_sol = p_sol[:n].copy()
    lam_sol = p_sol[n:n + k].copy()

    if relax_learned_constraint:
        slack_sol = float(p_sol[slack_index])
    else:
        slack_sol = 0.0

    original_local_pred = original_ghat_fun(p_sol)
    relaxed_constraint_residual = original_local_pred - slack_sol

    x_reconstructed = Xloc.T @ lam_sol

    original_objective = float(problem_model["objective"](x_sol))
    relaxed_objective = float(getattr(res, "fun", np.nan))

    diagnostics.update({
        "obj": original_objective,
        "original_objective": original_objective,
        "relaxed_objective": relaxed_objective,
        "output": original_local_pred,
        "slack": slack_sol,
        "relaxed_constraint_residual": float(relaxed_constraint_residual),
        "lambda_sum": float(np.sum(lam_sol)),
        "lambda_sum_violation": float(abs(np.sum(lam_sol) - 1.0)),
        "lambda_min": float(np.min(lam_sol)),
        "lambda_max": float(np.max(lam_sol)),
        "hull_eq_violation_inf": float(
            np.linalg.norm(x_sol - x_reconstructed, ord=np.inf)
        ),
        "x0": float(x_sol[0]),
        "min_local_data_x0": float(np.min(Xloc[:, 0])),
        "x0_below_min_local_data": float(
            max(0.0, np.min(Xloc[:, 0]) - x_sol[0])
        ),
    })

    x_trial = np.concatenate([
        x_sol,
        np.array([original_local_pred]),
    ])

    return x_trial, diagnostics




# =========================================================
# SciPy/Rosenbrock subproblem with relaxation
# =========================================================
def dcl_solve_subproblem_scipy(
    problem_model,
    accepted_model,
    trust_region,
    X,
    xit,
    *,
    accept_failed_result=False,
    options=None,
):
    if hasattr(X, "to_numpy"):
        Xnp = X.to_numpy(dtype=float)
    else:
        Xnp = np.asarray(X, dtype=float)

    n = int(Xnp.shape[1])

    idx = list(accepted_model.data_indices) if accepted_model.data_indices is not None else []
    k = len(idx)

    if k == 0:
        return None, {
            "status": -1,
            "success": False,
            "message": "empty local subset",
        }

    if trust_region != "convex_hull":
        raise NotImplementedError(
            "SciPy path currently supports only trust_region='convex_hull'."
        )

    relax_learned_constraint = bool(
        getattr(options, "scipy_relax_learned_constraint", True)
    )

    slack_penalty = float(
        getattr(options, "scipy_slack_penalty", 1e4)
    )

    slack_upper_bound = float(
        getattr(options, "scipy_slack_upper_bound", 1e4)
    )

    use_local_coordinate_bounds = bool(
        getattr(options, "scipy_use_local_coordinate_bounds", True)
    )

    scipy_maxiter = int(
        getattr(options, "scipy_maxiter", 5000)
    )

    scipy_gtol = float(
        getattr(options, "scipy_gtol", 1e-8)
    )

    scipy_xtol = float(
        getattr(options, "scipy_xtol", 1e-8)
    )

    scipy_barrier_tol = float(
        getattr(options, "scipy_barrier_tol", 1e-8)
    )

    scipy_initial_constr_penalty = float(
        getattr(options, "scipy_initial_constr_penalty", 100.0)
    )

    Xloc = Xnp[idx, :]

    slack_index = n + k if relax_learned_constraint else None
    pdim = n + k + (1 if relax_learned_constraint else 0)

    def original_ghat_fun(p):
        return float(dcl_predict_local_model(accepted_model, p[:n]))

    def relaxed_ghat_fun(p):
        value = original_ghat_fun(p)

        if relax_learned_constraint:
            value -= float(p[slack_index])

        return float(value)

    def obj_fun(p):
        value = float(problem_model["objective"](p[:n]))

        if relax_learned_constraint:
            value += slack_penalty * float(p[slack_index])

        return float(value)

    def obj_jac(p):
        grad = np.zeros(pdim, dtype=float)
        grad[0] = 1.0

        if relax_learned_constraint:
            grad[slack_index] = slack_penalty

        return grad

    A = np.zeros((n, pdim), dtype=float)
    A[:, :n] = -np.eye(n)
    A[:, n:n + k] = Xloc.T

    lc_x = LinearConstraint(
        A,
        lb=np.zeros(n),
        ub=np.zeros(n),
    )

    A_sum = np.zeros((1, pdim), dtype=float)
    A_sum[0, n:n + k] = 1.0

    lc_sum = LinearConstraint(
        A_sum,
        lb=np.array([1.0]),
        ub=np.array([1.0]),
    )

    jac_sparsity = np.zeros((1, pdim), dtype=bool)
    jac_sparsity[0, :n] = True

    if relax_learned_constraint:
        jac_sparsity[0, slack_index] = True

    nlc = NonlinearConstraint(
        relaxed_ghat_fun,
        -np.inf,
        0.0,
        finite_diff_jac_sparsity=jac_sparsity,
    )

    lb_x = np.asarray(problem_model["bounds_lb"], dtype=float)
    ub_x = np.asarray(problem_model["bounds_ub"], dtype=float)

    if use_local_coordinate_bounds:
        local_lb = np.min(Xloc, axis=0)
        local_ub = np.max(Xloc, axis=0)
        lb_x_eff = np.maximum(lb_x, local_lb)
        ub_x_eff = np.minimum(ub_x, local_ub)
    else:
        lb_x_eff = lb_x
        ub_x_eff = ub_x

    lb_vec_parts = [
        lb_x_eff,
        np.zeros(k),
    ]

    ub_vec_parts = [
        ub_x_eff,
        np.ones(k),
    ]

    if relax_learned_constraint:
        lb_vec_parts.append(np.array([0.0]))
        ub_vec_parts.append(np.array([slack_upper_bound]))

    lb_vec = np.concatenate(lb_vec_parts)
    ub_vec = np.concatenate(ub_vec_parts)

    xit_arr = np.asarray(xit, dtype=float)

    nearest_local = int(
        np.argmin(
            np.linalg.norm(Xloc - xit_arr, axis=1)
        )
    )

    lam0 = np.zeros(k, dtype=float)
    lam0[nearest_local] = 1.0

    x0_init = Xloc[nearest_local, :].copy()

    p_init_parts = [
        x0_init,
        lam0,
    ]

    if relax_learned_constraint:
        local_pred_init = float(dcl_predict_local_model(accepted_model, x0_init))
        s0 = min(
            slack_upper_bound,
            max(0.0, local_pred_init) + 1e-8,
        )
        p_init_parts.append(np.array([s0], dtype=float))

    p_init = np.concatenate(p_init_parts)

    res = minimize(
        obj_fun,
        p_init,
        jac=obj_jac,
        bounds=Bounds(lb_vec, ub_vec),
        constraints=[nlc, lc_x, lc_sum],
        method="trust-constr",
        options={
            "disp": False,
            "maxiter": scipy_maxiter,
            "gtol": scipy_gtol,
            "xtol": scipy_xtol,
            "barrier_tol": scipy_barrier_tol,
            "initial_constr_penalty": scipy_initial_constr_penalty,
            "sparse_jacobian": True,
        },
    )

    raw_res_x = getattr(res, "x", None)

    has_usable_x = (
        raw_res_x is not None
        and np.asarray(raw_res_x, dtype=float).size >= pdim
        and np.all(np.isfinite(np.asarray(raw_res_x[:n], dtype=float)))
    )

    success = bool(getattr(res, "success", False))

    diagnostics = {
        "status": int(getattr(res, "status", -1)),
        "success": success,
        "message": str(getattr(res, "message", "SciPy subproblem failed")),
        "nit": int(getattr(res, "nit", -1)),
        "nfev": int(getattr(res, "nfev", -1)),
        "constr_violation": float(getattr(res, "constr_violation", np.nan)),
        "optimality": float(getattr(res, "optimality", np.nan)),
        "execution_time": float(getattr(res, "execution_time", np.nan)),
    }

    if (not has_usable_x) or ((not success) and (not accept_failed_result)):
        return None, diagnostics

    p_sol = np.asarray(raw_res_x, dtype=float)

    x_sol = p_sol[:n].copy()
    lam_sol = p_sol[n:n + k].copy()

    if relax_learned_constraint:
        slack_sol = float(p_sol[slack_index])
    else:
        slack_sol = 0.0

    original_local_pred = original_ghat_fun(p_sol)
    relaxed_constraint_residual = original_local_pred - slack_sol

    x_reconstructed = Xloc.T @ lam_sol

    original_objective = float(problem_model["objective"](x_sol))
    relaxed_objective = float(getattr(res, "fun", np.nan))

    diagnostics.update({
        "obj": original_objective,
        "original_objective": original_objective,
        "relaxed_objective": relaxed_objective,
        "output": original_local_pred,
        "slack": slack_sol,
        "relaxed_constraint_residual": float(relaxed_constraint_residual),
        "lambda_sum": float(np.sum(lam_sol)),
        "lambda_sum_violation": float(abs(np.sum(lam_sol) - 1.0)),
        "lambda_min": float(np.min(lam_sol)),
        "lambda_max": float(np.max(lam_sol)),
        "hull_eq_violation_inf": float(
            np.linalg.norm(x_sol - x_reconstructed, ord=np.inf)
        ),
        "x0": float(x_sol[0]),
        "min_local_data_x0": float(np.min(Xloc[:, 0])),
        "x0_below_min_local_data": float(
            max(0.0, np.min(Xloc[:, 0]) - x_sol[0])
        ),
    })

    x_trial = np.concatenate([
        x_sol,
        np.array([original_local_pred]),
    ])

    return x_trial, diagnostics


# =========================================================
# Local model fitting and degree/model-complexity selection
# =========================================================
def dcl_set_model_degree(
    learning_model_type,
    degree_init,
    sorted_indices,
    ndata_model,
    Xbar,
    ybar,
    options,
):
    is_pandas_X = hasattr(Xbar, "iloc")
    is_pandas_y = hasattr(ybar, "iloc")

    ndata_train = math.floor(options.train_data_ratio * ndata_model)
    ndata_train = max(1, min(ndata_train, ndata_model - 1))
    ndata_test = ndata_model - ndata_train

    idata_model = sorted_indices[:ndata_model]

    idata_train = random.sample(idata_model, k=ndata_train)
    idata_test = np.setdiff1d(idata_model, idata_train).tolist()

    if is_pandas_X:
        Xbar_train = Xbar.iloc[idata_train, :]
        Xbar_test = Xbar.iloc[idata_test, :]
    else:
        Xbar_train = Xbar[idata_train, :]
        Xbar_test = Xbar[idata_test, :]

    if is_pandas_y:
        ybar_train = ybar.iloc[idata_train]
        ybar_test = ybar.iloc[idata_test]
    else:
        ybar_train = ybar[idata_train]
        ybar_test = ybar[idata_test]

    Xbar_train = np.asarray(Xbar_train, dtype=float)
    Xbar_test = np.asarray(Xbar_test, dtype=float)

    ybar_train = np.asarray(ybar_train, dtype=float).ravel()
    ybar_test = np.asarray(ybar_test, dtype=float).ravel()

    if learning_model_type == "nn":
        nn_architectures = [(10,), (20,), (50,), (100,)]
        nn_solver = "adam" if ndata_train >= 1000 else "lbfgs"
        nn_max_iter = 10000
        max_degree = min(options.max_model_degree, len(nn_architectures))
    else:
        max_degree = options.max_model_degree

    best = None

    for model_degree in range(1, max_degree + 1):
        p_local = None

        if learning_model_type == "cart":
            loc_model = DecisionTreeRegressor(
                max_depth=model_degree,
                random_state=options.seed,
            ).fit(Xbar_train, ybar_train)

            ypred_train = loc_model.predict(Xbar_train)
            ypred_test = loc_model.predict(Xbar_test)

        elif learning_model_type == "polynom_regression":
            # Rosenbrock polynomial regression in the old x_new style:
            # no scaling, PolynomialFeatures(include_bias=True), LinearRegression.
            p_local = PolynomialFeatures(
                degree=model_degree,
                include_bias=True,
            )

            X_poly_train = p_local.fit_transform(Xbar_train)
            X_poly_test = p_local.transform(Xbar_test)

            loc_model = LinearRegression().fit(X_poly_train, ybar_train)

            ypred_train = loc_model.predict(X_poly_train)
            ypred_test = loc_model.predict(X_poly_test)

        elif learning_model_type == "nn":
            arch_idx = min(model_degree - 1, len(nn_architectures) - 1)
            arch = nn_architectures[arch_idx]

            loc_model = Pipeline([
                ("scaler", StandardScaler()),
                ("nn", MLPRegressor(
                    hidden_layer_sizes=arch,
                    activation="relu",
                    solver=nn_solver,
                    max_iter=nn_max_iter,
                    random_state=options.seed,
                )),
            ])

            loc_model.fit(Xbar_train, ybar_train)

            ypred_train = loc_model.predict(Xbar_train)
            ypred_test = loc_model.predict(Xbar_test)

        elif learning_model_type == "rf":
            loc_model = RandomForestRegressor(
                n_estimators=int(getattr(options, "rf_n_estimators", 100)),
                max_depth=model_degree,
                min_samples_leaf=int(getattr(options, "rf_min_samples_leaf", 1)),
                random_state=options.seed,
                n_jobs=-1,
            ).fit(Xbar_train, ybar_train)

            ypred_train = loc_model.predict(Xbar_train)
            ypred_test = loc_model.predict(Xbar_test)

        else:
            raise NotImplementedError(
                f"Unknown learning_model_type='{learning_model_type}'."
            )

        train_mse = dcl_mse(ypred_train, ybar_train)
        test_mse = dcl_mse(ypred_test, ybar_test)

        train_rmse = float(np.sqrt(train_mse))
        test_rmse = float(np.sqrt(test_mse))

        train_rel_rmse = dcl_relative_rmse(ypred_train, ybar_train)
        test_rel_rmse = dcl_relative_rmse(ypred_test, ybar_test)

        print("train_mse :", train_mse, "test_mse :", test_mse)
        print("train_rmse :", train_rmse, "test_rmse :", test_rmse)
        print("train_rel_rmse :", train_rel_rmse, "test_rel_rmse :", test_rel_rmse)
        print(
            "ndata_model : ", ndata_model,
            "ndata_test : ", ndata_test,
            "ndata_train : ", ndata_train,
            "model_degree : ", model_degree,
        )

        candidate = (
            model_degree,
            test_rmse,
            train_rmse,
            loc_model,
            p_local,
            train_mse,
            test_mse,
            train_rel_rmse,
            test_rel_rmse,
        )

        if best is None:
            best = candidate

    return best


# =========================================================
# Weight update
# =========================================================
def update_weights(xit, dists):
    lam = 0.0
    return np.exp(-lam * dists)


# =========================================================
# Printing helpers
# =========================================================
def dcl_print_iteration_log(solver):
    print("\n===== DCL ITERATION LOG =====")

    for row in solver.iteration_log:
        print(
            f"iter={row['iter']}, "
            f"obj={row['objective']}, "
            f"relaxed_obj={row.get('relaxed_objective')}, "
            f"data_size={row['data_size']}, "
            f"degree={row['degree']}, "
            f"test_mse={row.get('test_mse')}, "
            f"test_rmse={row['test_rmse']}, "
            f"train_rmse={row['train_rmse']}, "
            f"status={row['solver_status']}, "
            f"success={row.get('solver_success')}"
        )

        print("start_x =", row.get("start_x"))
        print("solution_x =", row["solution_x"])
        print("solution_y/local_output =", row["solution_y"])

        if row.get("sklearn_output") is not None:
            print("sklearn_output =", row["sklearn_output"])

        if row.get("manual_predictor_error") is not None:
            print("manual_predictor_error =", row["manual_predictor_error"])

        if row.get("gurobi_ml_predictor_error") is not None:
            print("gurobi_ml_predictor_error =", row["gurobi_ml_predictor_error"])

        if row.get("predictor_error") is not None:
            print("predictor_error =", row["predictor_error"])

        if row.get("relaxation_slack") is not None:
            print("relaxation_slack =", row["relaxation_slack"])

        if row.get("relaxed_constraint_residual") is not None:
            print("relaxed_constraint_residual =", row["relaxed_constraint_residual"])

        if row.get("true_g_start") is not None:
            print("true_g_start =", row["true_g_start"])

        if row.get("local_pred_start") is not None:
            print("local_g_start/local_output_start =", row["local_pred_start"])

        if row.get("true_g_solution") is not None:
            print("true_g_solution =", row["true_g_solution"])

        if row.get("local_pred_solution") is not None:
            print("local_g_solution/local_output_solution =", row["local_pred_solution"])

        if row.get("solver_message") is not None:
            print("solver_message =", row["solver_message"])

        print("-" * 60)


def dcl_get_result_rows(solver):
    return solver.iteration_log