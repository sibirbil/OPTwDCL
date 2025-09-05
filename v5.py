from scipy.optimize import (
    LinearConstraint,
    NonlinearConstraint,
    Bounds,
    minimize,
    linprog,
)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

import warnings
import math
import random

from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import pinv

global global_model
global local_model
global poly_global
global poly_local
global n

# Record subsets of data used each iteration
global subset_indices_list
subset_indices_list = []

SUBSET_COLOR      = "#FF7F0E"   # orange fill for "used"
ALLDATA_COLOR     = "lightgray" # fill for "not used"
EDGE_COLOR        = "black"
EDGE_LW_USED      = 0.6
EDGE_LW_NOT       = 0.6
POINT_SIZE_USED   = 60
POINT_SIZE_NOT    = 60

# Fixed axis ranges (x0 vertical on z-axis in this version)
X1_MIN, X1_MAX = -5.0, 5.0
X2_MIN, X2_MAX = -5.0, 5.0
X0_MIN, X0_MAX =  0.0, 1500.0

# ---- Trust-regions ----
TR_MODE = 1             # 1 = convex hull, 2 = Mahalanobis ellipsoid
ELLIPSOID_RADIUS = 1.0  # only used for TR_MODE=2 (γ); constraint is (x-xk)^T C^{-1} (x-xk) <= γ^2
EPS_COV = 0         # ridge

warnings.filterwarnings(
    "ignore",
    message="delta_grad == 0.0. Check if the approximated function is linear. If the function is linear better results can be obtained by defining the Hessian as zero instead of using quasi-Newton approximations.",
)
warnings.filterwarnings(
    "ignore",
    message="Singular Jacobian matrix. Using SVD decomposition to perform the factorizations.",
)

'''
We assume the following problem structure.
             min f(x)
 subject to   g(x)<=0
              c(x)<=0
We assume that the constraint function g(x) is to be learned from data.
'''

'''
Example : Rosenbrock
'''

def f(x):
    return x[0]

def g(x):
    return -x[0] + 100 * (x[2] - x[1] * x[1]) ** 2 + (1 - x[1]) ** 2


def g_globmodel(x):
    x = np.asarray(x).reshape(1, -1)
    x_poly = poly_global.fit_transform(x)
    return float(global_model.predict(x_poly)[0])

def g_locmodel(x):
    if (x.size > n):
        x = x[0:n]
    x = np.asarray(x).reshape(1, -1)
    x_poly_local = poly_local.fit_transform(x)
    return float(local_model.predict(x_poly_local)[0])

def g_locmodel_from(lm, pl, x):
    xx = np.asarray(x)
    if xx.size > n:
        xx = xx[:n]
    xx = xx.reshape(1, -1)
    Phi = pl.fit_transform(xx)  # keep v5/60 style
    return float(lm.predict(Phi)[0])

def update_weights(xit, dists):
    lam = 0.0
    return np.exp(-lam * dists)

def set_model_degree(degree_init, sorted_indices, ndata_model, Xbar, ybar):
    RMSE_THRESH = 1e-1
    MAX_MODEL_DEGREE = 4
    model_degree = degree_init

    ndata_train = math.floor(0.7 * ndata_model)
    ndata_test  = ndata_model - ndata_train

    idata_model = sorted_indices[0:ndata_model]
    idata_train = random.sample(idata_model, k=ndata_train)
    idata_test  = np.setdiff1d(idata_model, idata_train).tolist()

    Xbar_train = Xbar[idata_train, :]
    ybar_train = ybar[idata_train]
    Xbar_test  = Xbar[idata_test, :]
    ybar_test  = ybar[idata_test]

    isDone = False
    test_rmse_prev = np.inf

    while not isDone:
        p_local = PolynomialFeatures(degree=model_degree)
        X_poly_local = p_local.fit_transform(Xbar_train)
        loc_model = LinearRegression().fit(X_poly_local, ybar_train)

        X_poly_test = p_local.fit_transform(Xbar_test)
        ypred_test  = loc_model.predict(X_poly_test)

        test_rmse = np.linalg.norm(
            (ypred_test - ybar_test) / np.maximum(np.maximum(np.abs(ypred_test), np.abs(ybar_test)), 1e-12)
        ) / max(ndata_test, 1)

        print("ndata_model : ", ndata_model,
              "ndata_test : ",  ndata_test,
              "ndata_train : ", ndata_train,
              "model_degree : ", model_degree,
              "test_rmse : ",    test_rmse)

        if test_rmse > test_rmse_prev:
            return model_degree_prev, test_rmse_prev, loc_model_prev, p_local_prev

        if test_rmse < RMSE_THRESH or model_degree == MAX_MODEL_DEGREE:
            return model_degree, test_rmse, loc_model, p_local

        model_degree_prev = model_degree
        test_rmse_prev    = test_rmse
        loc_model_prev    = loc_model
        p_local_prev      = p_local
        model_degree     += 1


def subproblem_solver(xk, lm, pl, idata_model, X, bounds, tr_mode, ellipsoid_radius):
    """
    xk           : current iterate (length n)
    lm, pl       : local surrogate (predicts g via v60 style on full x)
    idata_model  : list/array of indices used for local model
    X            : all samples (for convex hull columns and covariance)
    bounds       : Bounds for x only (length n)
    tr_mode      : 1 = convex hull (as in 70), 2 = full Mahalanobis ellipsoid centered at iterate
    ellipsoid_radius : γ (radius) for tr_mode=2
    """
    nloc = len(idata_model)
    Xsub = X[idata_model, :]

    if tr_mode == 1:
        A = np.hstack((-np.eye(n), Xsub[:, :n].T))
        lc = LinearConstraint(A, lb=np.zeros(n), ub=np.zeros(n))

        def g_loc_on_p(p):
            return g_locmodel_from(lm, pl, p[:n])

        nlc = NonlinearConstraint(g_loc_on_p, -np.inf, 0.0)

        lb_vec = np.concatenate([bounds.lb, np.zeros(nloc)])
        ub_vec = np.concatenate([bounds.ub, np.ones(nloc)])
        p_init = np.concatenate([xk, np.zeros(nloc)])

        res = minimize(
            f, p_init,
            bounds=Bounds(lb_vec, ub_vec),
            constraints=[nlc, lc],
            method="trust-constr",
            options={'disp': False}
        )
        return res.x[:n], res

    elif tr_mode == 2:
        C_loc  = np.cov(Xsub, rowvar=False) + EPS_COV*np.eye(n)
        Cinv_l = pinv(C_loc)

        def md2_iter(x):
            d = x - xk
            return float(d @ Cinv_l @ d)

        gamma2 = float(ellipsoid_radius)**2

        # feasibility on the learned surrogate
        def g_loc_x(x):
            return g_locmodel_from(lm, pl, x)

        # TR constraint: md2_iter(x) <= gamma^2  <=>  gamma^2 - md2_iter(x) >= 0
        def cons_md(x):
            return gamma2 - md2_iter(x)

        nlc_md  = NonlinearConstraint(cons_md,  0.0,  np.inf)
        nlc_loc = NonlinearConstraint(g_loc_x, -np.inf, 0.0)

        res = minimize(
            f, xk,
            bounds=bounds,
            constraints=[nlc_loc, nlc_md],
            method="trust-constr",
            options={'disp': False}
        )
        return res.x, res

    else:
        raise ValueError("Unknown TR mode (use 1=cx-hull, 2=Mahalanobis)")

def nowdcl(f, X, y, bounds, x_0, niters):
    global local_model, poly_local, n, subset_indices_list

    MAX_NDATA   = 100000
    RMSE_THRESH = 9e-1
    DELTA_MIN   = 1e-6

    xit = np.array(x_0, dtype=float)
    xis = [np.array(x_0, copy=True)]

    local_models = []
    poly_locals  = []
    subsets_used = []

    dists = np.linalg.norm(xit - X, axis=1)
    sorted_indices = np.argsort(dists).tolist()

    ndata = len(y)
    learning_size  = 2 * len(x_0)
    ndata_model_0  = np.minimum(MAX_NDATA, np.minimum(5 * learning_size, ndata))

    print("iteration 0:", " trial_x: ", xit, "f(x): ", f(xit), "g(x): ", g(xit))

    for i in range(niters):
        model_degree = 1
        ndata_model  = ndata_model_0
        weights = update_weights(xit, dists)
        ybar = weights * y
        Xbar = weights.reshape(-1, 1) * X

        best_test_rmse   = np.inf
        best_ndata_model = 0
        local_model = None
        poly_local  = None

        while ndata_model <= ndata:
            md_tmp, rmse_tmp, lm_tmp, pl_tmp = set_model_degree(
                model_degree, sorted_indices, ndata_model, Xbar, ybar
            )
            if rmse_tmp < best_test_rmse:
                best_test_rmse   = rmse_tmp
                local_model      = lm_tmp
                model_degree     = md_tmp
                poly_local       = pl_tmp
                best_ndata_model = ndata_model
            else:
                break
            ndata_model += 2 * learning_size

        idata_model = sorted_indices[:best_ndata_model]
        subsets_used.append(idata_model.copy())
        subset_indices_list.append(idata_model.copy())  # keep your original global list

        local_models.append(local_model)
        poly_locals.append(poly_local)

        if best_test_rmse > RMSE_THRESH:
            print("Stopping: local model not accurate enough (rmse =", best_test_rmse, ")")
            return -2, xis, local_models, poly_locals, subsets_used

        x_trial, res_loc = subproblem_solver(
            xk=xit,
            lm=local_model,
            pl=poly_local,
            idata_model=idata_model,
            X=X,
            bounds=bounds,
            tr_mode=TR_MODE,
            ellipsoid_radius=ELLIPSOID_RADIUS
        )

        if np.linalg.norm(x_trial[:n] - xit) < DELTA_MIN:
            print("No significant progress; terminating.")
            return 0, xis, local_models, poly_locals, subsets_used

        xit = x_trial[:n]
        xis.append(xit.copy())
        print("iteration ", i+1, ", trial_x: ", xit, "f(x): ", f(xit), "g(x): ", g(xit), "g_locmodel(x) =",
              g_locmodel(np.array(xit)))

        dists = np.linalg.norm(xit - X, axis=1)
        sorted_indices = np.argsort(dists).tolist()

    return -1, xis, local_models, poly_locals, subsets_used

def _apply_fixed_axes(ax):
    """Apply the fixed ranges with x0 vertical (z-axis)."""
    ax.set_xlim3d(X2_MIN, X2_MAX)  # x-axis shows x2
    ax.set_ylim3d(X1_MIN, X1_MAX)  # y-axis shows x1
    ax.set_zlim3d(X0_MIN, X0_MAX)  # z-axis shows x0 (vertical)
    ax.set_xlabel('x₂'); ax.set_ylabel('x₁'); ax.set_zlabel('x₀')
    ax.view_init(elev=25, azim=45)

def animate_subsets_3d(X, xis, subsets, pause=1000, repeat=True):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1,1,1))
    try:
        ax.set_proj_type('ortho')
    except Exception:
        pass

    N = X.shape[0]

    def update(frame):
        ax.clear()
        idx = subsets[frame]
        xi  = xis[frame+1]  # skip initial point

        used = np.zeros(N, dtype=bool)
        used[idx] = True

        ax.scatter(X[~used,2], X[~used,1], X[~used,0],
                   c=ALLDATA_COLOR, s=POINT_SIZE_NOT, alpha=0.6,
                   edgecolors=EDGE_COLOR, linewidths=EDGE_LW_NOT, depthshade=False, label='not used')

        if len(idx):
            ax.scatter(X[used,2], X[used,1], X[used,0],
                       c=SUBSET_COLOR, s=POINT_SIZE_USED, marker='o',
                       edgecolors=EDGE_COLOR, linewidths=EDGE_LW_USED, depthshade=False,
                       label=f'local subset (size={len(idx)})')

        ax.scatter(xi[2], xi[1], xi[0],
                   c='C0', s=130, marker='*',
                   edgecolors='k', linewidths=0.6, depthshade=False, label='xit')

        _apply_fixed_axes(ax)
        ax.set_title(f"Iteration {frame+1}")
        ax.legend(loc='upper right')

    ani = animation.FuncAnimation(fig, update, frames=len(subsets), interval=pause, repeat=repeat)
    plt.show()

def plot_winner_iteration_3d(X, xis, subsets):
    subset_sizes = [len(s) for s in subsets]
    max_iter = subset_sizes.index(max(subset_sizes))      # 0‐based
    max_size = subset_sizes[max_iter]
    max_subset = subsets[max_iter]
    xi_max = xis[max_iter+1]

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1,1,1))
    try:
        ax.set_proj_type('ortho')
    except Exception:
        pass

    used = np.zeros(X.shape[0], dtype=bool); used[max_subset] = True

    ax.scatter(
        X[~used,2], X[~used,1], X[~used,0],
        c=ALLDATA_COLOR, s=POINT_SIZE_NOT, alpha=0.6, depthshade=False,
        edgecolors=EDGE_COLOR, linewidths=EDGE_LW_NOT, label='not used'
    )

    ax.scatter(
        X[used,2], X[used,1], X[used,0],
        c=SUBSET_COLOR, s=POINT_SIZE_USED, marker='o', depthshade=False,
        edgecolors=EDGE_COLOR, linewidths=EDGE_LW_USED,
        label=f'local subset (size={max_size})'
    )

    ax.scatter(
        xi_max[2], xi_max[1], xi_max[0],
        c='C0', s=130, marker='*',
        edgecolors='k', linewidths=0.6, depthshade=False, label='xit'
    )

    _apply_fixed_axes(ax)
    ax.set_title(f"Iteration {max_iter+1}: Largest Local Model")
    ax.legend(loc='upper right')
    plt.show()

def plot_union_used_points_3d(X, subsets):
    """Show ALL points ever used (union) vs not used — with x₀ vertical and fixed ranges."""
    used = sorted(set(j for idxs in subsets for j in idxs))
    mask_used = np.zeros(X.shape[0], dtype=bool)
    mask_used[used] = True

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1,1,1))
    try:
        ax.set_proj_type('ortho')
    except Exception:
        pass

    ax.scatter(
        X[~mask_used,2], X[~mask_used,1], X[~mask_used,0],
        c=ALLDATA_COLOR, s=POINT_SIZE_NOT, alpha=0.6, depthshade=False,
        edgecolors=EDGE_COLOR, linewidths=EDGE_LW_NOT, label='not used'
    )
    ax.scatter(
        X[mask_used,2], X[mask_used,1], X[mask_used,0],
        c=SUBSET_COLOR, s=POINT_SIZE_USED, marker='o', depthshade=False,
        edgecolors=EDGE_COLOR, linewidths=EDGE_LW_USED,
        label=f'used (union)  n={mask_used.sum()}'
    )

    _apply_fixed_axes(ax)
    ax.set_title("All points ever used vs not used")
    ax.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    x_0 = [-1.0, -2.0, 1.0]
    n   = 3

    np.random.seed(102)
    random.seed(102)
    print("Random seed fixed!")
    nsamples = 100
    X = 0.0 + 2 * np.random.randn(nsamples, 3)
    for i in range(nsamples):
        myrnd = np.random.randn(1)
        X[i,0]= myrnd[0] + (100 * (X[i,2] - X[i,1] * X[i,1]) ** 2 + (1 - X[i,1]) ** 2)

    noise = 0.0 * np.random.randn(nsamples)
    y = np.array([g(X[i, :]) for i in range(nsamples)]) + noise

    delta_plot = 0.25
    x1_plot = np.arange(-5.0, 5.0, delta_plot)
    x2_plot = np.arange(-7.0, 5.0, delta_plot)
    X1plot, X2plot = np.meshgrid(x1_plot, x2_plot)
    Z = 100 * (X2plot - X1plot ** 2) ** 2 + (1 - X1plot) ** 2
    fig2, ax2 = plt.subplots()
    cs = ax2.contourf(X1plot, X2plot, Z)
    plt.scatter(X[:, 1], X[:, 2])
    plt.scatter(x_0[1], x_0[2])
    plt.title("Constraint landscape & samples")
    plt.show()

    bounds = Bounds([-np.inf, -5, -5], [np.inf, 5, 5])
    nlc    = NonlinearConstraint(g, -np.inf, 0)
    res    = minimize(f, x_0, bounds=bounds, constraints=[nlc], method="trust-constr")
    print('True solution: ', res.x)
    print("f: ", f(res.x), "g: ", g(res.x))

    # Global 4th-degree polynomial model
    poly_global  = PolynomialFeatures(degree=4)
    X_poly       = poly_global.fit_transform(X)
    global_model = LinearRegression().fit(X_poly, y)
    nlc_gmodel   = NonlinearConstraint(g_globmodel, -np.inf, 0)
    res_g = minimize(f, x_0, bounds=bounds, constraints=[nlc_gmodel], method="trust-constr")
    print('Solution using the global model with ', nsamples, ' samples: ', res_g.x)
    print("true constraint value", g(res_g.x))

    niters = 30
    out = nowdcl(f, X, y, bounds, x_0, niters)

    if isinstance(out, tuple) and len(out) == 5:
        exitcode, xis, local_models, poly_locals, subsets_used = out
    else:
        exitcode = out
        xis = [np.array(x_0)]
        local_models, poly_locals, subsets_used = [], [], []

    # ===== Per-iteration diagnostics table =====
    records = []
    K = min(len(local_models), len(xis))
    for i in range(K):
        x_i = np.array(xis[i])
        true_val = g(x_i)
        glob_pred = g_globmodel(x_i)
        lm = local_models[i]
        pl = poly_locals[i]
        loc_pred = float(lm.predict(pl.fit_transform(x_i.reshape(1, -1)))[0])

        records.append({
            "Iter": i,
            "x0": x_i[0], "x1": x_i[1], "x2": x_i[2],
            "True g(x)":     true_val,
            "Global Pred.":  glob_pred,
            "Local Pred.":   loc_pred,
            "Err(Global)":   abs(glob_pred - true_val),
            "Err(Local)":    abs(loc_pred  - true_val)
        })

    if len(records):
        df = pd.DataFrame(records)
        print("\n=== Diagnostics (per-iteration, using each iter's own local model) ===")
        print(df.to_string(index=False))
        if len(df) > 20:
            print("\n=== Last 20 iterates ===")
            print(df.tail(20).to_string(index=False))
    else:
        print("\n(No per-iteration surrogates were recorded; algorithm terminated immediately.)")

    if len(subsets_used) and len(xis) > 1:
        plot_winner_iteration_3d(X, xis, subsets_used)
        plot_union_used_points_3d(X, subsets_used)
        animate_subsets_3d(X, xis, subsets_used, pause=1000)
