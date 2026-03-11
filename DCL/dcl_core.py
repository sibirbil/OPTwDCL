# -*- coding: utf-8 -*-
# This file contains the basic procedures of dynamic constraint learning

import numpy as np
import random
import math

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import gurobipy as gp
from gurobipy import GRB
from gurobi_ml import add_predictor_constr

ENABLE_WEIGHTING = 0


# =========================
# Options class
# =========================
class dcl_options:
    trust_region = "convex_hull"          # TR_MODE
    learning_model_type = "cart"   # "cart", "polynom_regression", "nn"
    train_data_ratio = 0.7
    init_data_ratio = 0.1
    max_iterations = 1
    x_init = None
    delta_min = 1e-6
    rmse_thresh = 0.1
    rmse_target = 0.001
    max_model_degree = 4
    seed = 0
    max_ndata = 1000000

    # def check_options(self):
    #     TODO


# =========================
# Learned model class
# =========================
class dcl_local_model:
    def __init__(self):
        self.data_indices = None
        self.learned_model = None
        self.test_rmse = None
        self.train_rmse = None
        self.model_degree = 1
        self.ndata_model = None

        # solution info
        self.solution_x = None   # input vector only
        self.solution_y = None   # output variable value

        # MIP statistics
        self.mip_cols = None
        self.mip_rows = None
        self.mip_work = None
        self.mip_nodes = None
        self.mip_runtime = None
        self.num_bin = None
        self.num_int = None
        self.num_genconstrs = None


# =========================
# Helper metric
# =========================
def dcl_relative_rmse(y_pred, y_true):
    #This is NOT the standard RMSE definition, will be corrected.
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_true = np.asarray(y_true, dtype=float).ravel()

    denom = np.maximum(np.maximum(np.abs(y_pred), np.abs(y_true)), 1e-12)
    rel_err = (y_pred - y_true) / denom
    return np.linalg.norm(rel_err) / max(len(y_true), 1)


# =========================
# Solver class
# =========================
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

        # one dict per outer iteration; ideal for result tables
        self.iteration_log = []

    def solve(self):
        X = self.X
        y = self.y
        options = self.options

        # work with numpy for distances
        Xnp = X.to_numpy(dtype=float) if hasattr(X, "to_numpy") else np.asarray(X, dtype=float)
        ynp = y.to_numpy(dtype=float) if hasattr(y, "to_numpy") else np.asarray(y, dtype=float)

        # problem size
        n = int(Xnp.shape[1])

        # initialize xit if needed
        if self.x_init is None:
            self.x_init = dcl_init(Xnp)

        xit = np.asarray(self.x_init, dtype=float).copy()

        # initial neighborhood
        dists = np.linalg.norm(Xnp - xit, axis=1)
        sorted_indices = np.argsort(dists).tolist()

        ndata = len(ynp)
        
        learning_size = 2 * n

        min_local_size = max(2 * n, 4)
        ndata_model_0 = max(min_local_size, math.ceil(options.init_data_ratio * ndata))
        ndata_model_0 = min(options.max_ndata, ndata_model_0, ndata)

        for i in range(options.max_iterations):
            model_degree = 1
            ndata_model = ndata_model_0

            # currently not using weighting
            if ENABLE_WEIGHTING:
                weights = update_weights(xit, dists)
                ybar = weights * y
                Xbar = weights.reshape(-1, 1) * X
            else:
                Xbar = X
                ybar = y

            best_test_rmse = np.inf
            best_train_rmse = None
            best_ndata_model = 0
            local_model = None
            p_local = None

            while ndata_model <= min(ndata, options.max_ndata):
                md_tmp, test_rmse_tmp, train_rmse_tmp, lm_tmp, p_tmp = dcl_set_model_degree(
                    self.options.learning_model_type,
                    model_degree,
                    sorted_indices,
                    ndata_model,
                    Xbar,
                    ybar,
                    self.options
                )

                if test_rmse_tmp < best_test_rmse:
                    best_test_rmse = test_rmse_tmp
                    best_train_rmse = train_rmse_tmp
                    local_model = lm_tmp
                    p_local = p_tmp
                    model_degree = md_tmp
                    best_ndata_model = ndata_model
                else:
                    break

                ndata_model += 2 * learning_size

            idata_model = sorted_indices[:best_ndata_model]

            accepted_model = dcl_local_model()
            accepted_model.data_indices = idata_model
            accepted_model.learned_model = local_model
            accepted_model.test_rmse = best_test_rmse
            accepted_model.train_rmse = best_train_rmse
            accepted_model.model_degree = model_degree
            accepted_model.ndata_model = best_ndata_model

            self.local_models.append(accepted_model)

            if best_test_rmse > self.options.rmse_thresh:
                print("Stopping: local model not accurate enough (rmse =", best_test_rmse, ")")
                self.return_code = -2
                self.xit = xit.copy()
                return

            x_trial, res_loc = dcl_solve_subproblem(
                self.problem_model,
                accepted_model,
                self.options.trust_region,
                self.X,
            )

            if x_trial is None:
                print(f"Stopping: subproblem not optimal (status={res_loc.get('status')})")
                self.return_code = -3
                self.xit = xit.copy()
                return

            # store solution and MIP stats
            accepted_model.solution_x = x_trial[:n].copy()
            accepted_model.solution_y = float(x_trial[n])

            accepted_model.mip_cols = res_loc.get("mip_cols")
            accepted_model.mip_rows = res_loc.get("mip_rows")
            accepted_model.mip_work = res_loc.get("mip_work")
            accepted_model.mip_nodes = res_loc.get("mip_nodes")
            accepted_model.mip_runtime = res_loc.get("mip_runtime")
            accepted_model.num_bin = res_loc.get("num_bin")
            accepted_model.num_int = res_loc.get("num_int")
            accepted_model.num_genconstrs = res_loc.get("num_genconstrs")

            self.iteration_log.append({
                "iter": i,
                "objective": res_loc.get("obj"),
                "data_size": accepted_model.ndata_model,
                "degree": accepted_model.model_degree,
                "test_rmse": accepted_model.test_rmse,
                "train_rmse": accepted_model.train_rmse,
                "mip_cols": accepted_model.mip_cols,
                "mip_rows": accepted_model.mip_rows,
                "mip_work": accepted_model.mip_work,
                "mip_nodes": accepted_model.mip_nodes,
                "mip_runtime": accepted_model.mip_runtime,
                "num_bin": accepted_model.num_bin,
                "num_int": accepted_model.num_int,
                "num_genconstrs": accepted_model.num_genconstrs,
                "solution_x": accepted_model.solution_x.copy(),
                "solution_y": accepted_model.solution_y,
            })

            # progress
            step = np.linalg.norm(x_trial[:n] - xit)
            print(f"iter {i}: step = {step:.3e}, obj = {res_loc.get('obj')}")

            # stop if no progress
            if step < options.delta_min:
                print("No significant progress; terminating.")
                self.return_code = -1
                self.xit = x_trial[:n].copy()
                self.final_obj = res_loc.get("obj")
                return

            # accept new iterate
            xit = x_trial[:n].copy()
            self.final_obj = res_loc.get("obj")

            # recompute neighborhood
            dists = np.linalg.norm(Xnp - xit, axis=1)
            sorted_indices = np.argsort(dists).tolist()

        # max_iterations reached
        self.xit = xit.copy()
        return


# =========================
# Initial point based on X
# =========================
def dcl_init(X, size=None):
    """
    Initial point = arithmetic average (centroid) of samples in X.
    Works with numpy arrays and pandas DataFrames.

    X: shape (n_samples, n_features)
    returns: shape (n_features,)
    """
    if hasattr(X, "to_numpy"):
        Xnp = X.to_numpy(dtype=float)
    else:
        Xnp = np.asarray(X, dtype=float)

    if Xnp.size == 0:
        raise ValueError("X is empty, cannot compute arithmetic average.")

    if Xnp.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features). Got shape {Xnp.shape}")

    x0 = Xnp.mean(axis=0)

    if size is not None and len(x0) != size:
        print(f"Warning: dcl_init returned length {len(x0)} but size={size}")

    return x0


# =========================
# Construct and solve subproblem
# =========================
def dcl_solve_subproblem(problem_model, accepted_model, trust_region, X):
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

    inp = np.array(input_vars, dtype=object).reshape(1, -1)   # (1, n)
    out = np.array([out0], dtype=object).reshape(1, 1)        # (1, 1)

    # learned constraint: output_vars = g(input_vars)
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
            raise ValueError("accepted_model.data_indices is empty; cannot build convex hull trust region.")

        lam = m.addMVar(shape=k, lb=0.0, vtype=GRB.CONTINUOUS, name="lam_tr")
        m.addConstr(gp.quicksum(lam[i] for i in range(k)) == 1.0, name="tr_sum")

        for j in range(n):
            m.addConstr(
                input_vars[j] == gp.quicksum(lam[i] * float(Xnp[idx[i], j]) for i in range(k)),
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
            "mip_cols": mip_cols,
            "mip_rows": mip_rows,
            "num_bin": num_bin,
            "num_int": num_int,
            "num_genconstrs": num_genconstrs,
        }

    z_sol = np.array([v.X for v in input_vars], dtype=float)
    y_sol = float(out0.X)
    x_trial = np.concatenate([z_sol, np.array([y_sol])])

    return x_trial, {
        "status": int(m.Status),
        "obj": float(m.ObjVal),
        "output": y_sol,
        "mip_cols": mip_cols,
        "mip_rows": mip_rows,
        "num_bin": num_bin,
        "num_int": num_int,
        "num_genconstrs": num_genconstrs,
        "mip_work": float(m.Work),
        "mip_nodes": float(m.NodeCount),
        "mip_runtime": float(m.Runtime),
    }


# =========================
# Fit local model and choose degree
# =========================
def dcl_set_model_degree(learning_model_type, degree_init, sorted_indices, ndata_model, Xbar, ybar, options):
    RMSE_THRESH = options.rmse_target

    is_pandas_X = hasattr(Xbar, "iloc")
    is_pandas_y = hasattr(ybar, "iloc")

    model_degree = degree_init

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

    test_rmse_prev = np.inf
    train_rmse_prev = None
    model_degree_prev = None
    loc_model_prev = None
    p_local_prev = None

    if learning_model_type == "nn":
        nn_architectures = [(10,), (20,), (50,), (100,)]
        nn_solver = "adam" if ndata_train >= 1000 else "lbfgs"
        nn_max_iter = 10000

    while True:
        p_local = None

        if learning_model_type == "cart":
            loc_model = DecisionTreeRegressor(
                max_depth=model_degree,
                random_state=options.seed
            ).fit(Xbar_train, ybar_train)

            ypred_train = loc_model.predict(Xbar_train)
            ypred_test = loc_model.predict(Xbar_test)

        elif learning_model_type == "polynom_regression":
            p_local = PolynomialFeatures(degree=model_degree, include_bias=True)
            X_poly_train = p_local.fit_transform(Xbar_train)
            loc_model = LinearRegression().fit(X_poly_train, ybar_train)

            ypred_train = loc_model.predict(X_poly_train)
            X_poly_test = p_local.transform(Xbar_test)
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
                    random_state=options.seed
                ))
            ])

            loc_model.fit(Xbar_train, ybar_train)
            ypred_train = loc_model.predict(Xbar_train)
            ypred_test = loc_model.predict(Xbar_test)

        else:
            raise NotImplementedError(
                f"Unknown learning_model_type='{learning_model_type}'."
            )

        train_rmse = dcl_relative_rmse(ypred_train, ybar_train)
        test_rmse = dcl_relative_rmse(ypred_test, ybar_test)

        train_mse = np.mean((ypred_train - ybar_train)**2)
        test_mse  = np.mean((ypred_test - ybar_test)**2)
        print("train_mse :", train_mse, "test_mse :", test_mse)

        print("ndata_model : ", ndata_model,
              "ndata_test : ", ndata_test,
              "ndata_train : ", ndata_train,
              "model_degree : ", model_degree,
              "test_rmse : ", test_rmse,
              "train_rmse : ", train_rmse)

        if (model_degree_prev is not None) and (test_rmse > test_rmse_prev):
            return model_degree_prev, test_rmse_prev, train_rmse_prev, loc_model_prev, p_local_prev

        if learning_model_type == "nn":
            reached_max = (model_degree >= min(options.max_model_degree, len(nn_architectures)))
        else:
            reached_max = (model_degree == options.max_model_degree)

        if test_rmse < RMSE_THRESH or reached_max:
            return model_degree, test_rmse, train_rmse, loc_model, p_local

        model_degree_prev = model_degree
        test_rmse_prev = test_rmse
        train_rmse_prev = train_rmse
        loc_model_prev = loc_model
        p_local_prev = p_local
        model_degree += 1


# =========================
# Weight update
# =========================
def update_weights(xit, dists):
    lam = 0.0
    return np.exp(-lam * dists)


# =========================
# print results
# =========================
def dcl_print_iteration_log(solver):
    print("\n===== DCL ITERATION LOG =====")
    for row in solver.iteration_log:
        print(f"iter={row['iter']}, "
              f"obj={row['objective']}, "
              f"data_size={row['data_size']}, "
              f"degree={row['degree']}, "
              f"test_rmse={row['test_rmse']}, "
              f"train_rmse={row['train_rmse']}, "
              f"mip_cols={row['mip_cols']}, "
              f"mip_rows={row['mip_rows']}, "
              f"mip_work={row['mip_work']}")
        print("solution_x =", row["solution_x"])
        print("solution_y =", row["solution_y"])
        print("-" * 60)


# =========================
# return rows for LaTeX table
# =========================
def dcl_get_result_rows(solver):
    return solver.iteration_log