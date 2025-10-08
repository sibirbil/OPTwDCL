# === Quadratic + NOWDCL (Penalty Objective, Geometry-only Constraint) ===
# Objective:  minimize  x1 + x2 + 100*x0^2
# Constraint: q(x1, x2) <= 0   (no x0 in the constraint)
#
# Surrogates (global & local) are trained on Z = [x1, x2] only.
# Trust regions (convex hull or Mahalanobis) are enforced in (x1,x2) only.
# x0 remains unconstrained by TR (aside from variable bounds).

from scipy.optimize import (
    LinearConstraint,
    NonlinearConstraint,
    Bounds,
    minimize,
)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold, train_test_split   # CV + holdout for global degree
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import warnings, math, random
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import pinv, qr

# ---------------- Globals ----------------
global global_model
global local_model
global poly_global
global poly_local
global n

# Record subsets of data used each iteration (for your animations)
global subset_indices_list
subset_indices_list = []

SUBSET_COLOR      = "#FF7F0E"
ALLDATA_COLOR     = "lightgray"
EDGE_COLOR        = "black"
EDGE_LW_USED      = 0.6
EDGE_LW_NOT       = 0.6
POINT_SIZE_USED   = 60
POINT_SIZE_NOT    = 60

# Fixed axis ranges (x0 vertical on z-axis in this version)
X1_MIN, X1_MAX = -5.0, 5.0
X2_MIN, X2_MAX = -5.0, 5.0
X0_MIN, X0_MAX = -5.0, 5.0   # x0 is free; set a symmetric plotting range

# ---- Trust-regions for local/global solves ----
TR_MODE = 1             # 1 = convex hull in (x1,x2), 2 = Mahalanobis ellipsoid in (x1,x2)
ELLIPSOID_RADIUS = 1.0  # for TR_MODE=2: (z - zk)^T C^{-1} (z - zk) <= gamma^2  where z=(x1,x2)
EPS_COV = 0.0           # ridge for covariance (2x2)

warnings.filterwarnings(
    "ignore",
    message="delta_grad == 0.0. Check if the approximated function is linear. If the function is linear better results can be obtained by defining the Hessian as zero instead of using quasi-Newton approximations.",
)
warnings.filterwarnings(
    "ignore",
    message="Singular Jacobian matrix. Using SVD decomposition to perform the factorizations.",
)

# =======================  Quadratic setup q(x1,x2)  =======================
Q_H = None   # 2x2 SPD
Q_b = None   # length-2
Q_c = None   # scalar
Q_d = 0.0    # scalar
Q_mu = None  # center
Q_R  = None  # radius

def _make_spd_2x2(rng, lam_min=0.5, lam_max=3.0):
    A = rng.normal(size=(2,2))
    Qm,_ = qr(A)                       # random orthonormal
    l1,l2 = rng.uniform(lam_min, lam_max, size=2)
    H = Qm.T @ np.diag([l1,l2]) @ Qm
    return 0.5*(H+H.T)

def init_quadratic(seed=42):
    """Initialize random quadratic q(x1,x2) and store globals."""
    global Q_H, Q_b, Q_c, Q_d, Q_mu, Q_R
    rng  = np.random.default_rng(seed)
    Q_H  = _make_spd_2x2(rng)
    Q_mu = rng.normal(scale=2.0, size=2)
    Q_R  = float(rng.uniform(0.75, 2.5))
    Q_b  = -2.0 * (Q_H @ Q_mu)
    Q_c  = float(Q_mu.T @ Q_H @ Q_mu - Q_R**2)
    Q_d  = 0.0

def q_val(x1, x2):
    """q(x1,x2) = [x1 x2] H [x1;x2] + b^T[x1;x2] + c - d"""
    x2d = np.array([x1, x2], dtype=float)
    return float(x2d @ (Q_H @ x2d) + Q_b @ x2d + Q_c - Q_d)

# =======================  Objective/constraint (NEW)  =======================
def f(x):
    # minimize x1 + x2 + 100*x0^2
    return float(x[1] + x[2] + 100.0*(x[0]**2))

def g(x):
    # geometry-only inequality: q(x1,x2) <= 0
    return q_val(x[1], x[2])

# -------- Surrogate accessors (train/predict on z=[x1,x2]) --------
def _x_to_z(x):
    # map decision vector x=[x0,x1,x2] to z=[x1,x2] used by constraint model
    return np.asarray([x[1], x[2]], dtype=float).reshape(1, -1)

def g_globmodel(x):
    z = _x_to_z(x)
    z_phi = poly_global.fit_transform(z)
    return float(global_model.predict(z_phi)[0])

def g_locmodel(x):
    z = _x_to_z(x)
    z_phi = poly_local.fit_transform(z)
    return float(local_model.predict(z_phi)[0])

def g_locmodel_from(lm, pl, x):
    z = _x_to_z(x)
    z_phi = pl.fit_transform(z)
    return float(lm.predict(z_phi)[0])

# =======================  (NEW) Read-only verification helpers  =======================
def check_convex_hull_solution(p, Zsub, bounds, tol=1e-8):
    """
    p = [x0, x1, x2, λ_1..λ_m] from CH solve.
    Checks: x1 = Σλ z1, x2 = Σλ z2, Σλ=1, 0<=λ<=1, and variable bounds.
    """
    m = Zsub.shape[0]
    x0, x1, x2 = p[0], p[1], p[2]
    lam = p[3:]
    assert lam.size == m

    r1 = x1 - np.dot(lam, Zsub[:,0])
    r2 = x2 - np.dot(lam, Zsub[:,1])
    r3 = lam.sum() - 1.0
    in_box = (lam.min() >= -tol) and (lam.max() <= 1.0 + tol)
    in_bounds = (bounds.lb[0]-tol <= x0 <= bounds.ub[0]+tol and
                 bounds.lb[1]-tol <= x1 <= bounds.ub[1]+tol and
                 bounds.lb[2]-tol <= x2 <= bounds.ub[2]+tol)
    return {
        "ok": (abs(r1)<=tol and abs(r2)<=tol and abs(r3)<=tol and in_box and in_bounds),
        "r_eq_x1": float(r1), "r_eq_x2": float(r2), "r_eq_sumlam": float(r3),
        "lam_min": float(lam.min()), "lam_max": float(lam.max()),
        "in_bounds": bool(in_bounds)
    }

def check_model_constraints_surrogate_and_true(x_star, lm, pl, tol=1e-10):
    """
    Checks surrogate feasibility g_loc(x)<=0, global surrogate (if available), and true q(x1,x2)<=0.
    """
    g_loc = g_locmodel_from(lm, pl, x_star)
    out = {"ok_loc": bool(g_loc <= tol), "g_loc": float(g_loc)}
    try:
        g_glob = g_globmodel(x_star)
        out.update({"ok_glob": bool(g_glob <= tol), "g_glob": float(g_glob)})
    except Exception:
        pass
    g_true = g(x_star)  # = q(x1,x2)
    out.update({"ok_true": bool(g_true <= tol), "g_true": float(g_true)})
    out["ok_all"] = all(v for k,v in out.items() if k.startswith("ok_"))
    return out

# =======================  Local degree auto-selection (ENHANCED: trace)  =======================
def update_weights(xit, dists):
    lam = 0.0
    return np.exp(-lam * dists)

def set_model_degree(degree_init, sorted_indices, ndata_model, Zbar, ybar):
    """
    Returns:
      model_degree_opt, best_test_rmse, loc_model, poly_local, degree_trace
    where degree_trace = [(deg1, rmse1), (deg2, rmse2), ...] for this ndata_model.
    """
    RMSE_THRESH = 1e-1
    MAX_MODEL_DEGREE = 4
    model_degree = degree_init

    ndata_train = math.floor(0.7 * ndata_model)
    ndata_test  = ndata_model - ndata_train

    idata_model = sorted_indices[0:ndata_model]
    idata_train = random.sample(idata_model, k=ndata_train)
    idata_test  = np.setdiff1d(idata_model, idata_train).tolist()

    Zbar_train = Zbar[idata_train, :]
    ybar_train = ybar[idata_train]
    Zbar_test  = Zbar[idata_test, :]
    ybar_test  = ybar[idata_test]

    degree_trace = []
    isDone = False
    test_rmse_prev = np.inf

    while not isDone:
        p_local = PolynomialFeatures(degree=model_degree)
        Z_poly_local = p_local.fit_transform(Zbar_train)
        loc_model = LinearRegression().fit(Z_poly_local, ybar_train)

        Z_poly_test = p_local.fit_transform(Zbar_test)
        ypred_test  = loc_model.predict(Z_poly_test)

        # relative RMSE safeguard
        test_rmse = np.linalg.norm(
            (ypred_test - ybar_test) / np.maximum(np.maximum(np.abs(ypred_test), np.abs(ybar_test)), 1e-12)
        ) / max(ndata_test, 1)

        degree_trace.append((model_degree, float(test_rmse)))

        print("ndata_model : ", ndata_model,
              "ndata_test : ",  ndata_test,
              "ndata_train : ", ndata_train,
              "model_degree : ", model_degree,
              "test_rmse : ",    test_rmse)

        if test_rmse > test_rmse_prev:
            # go back to previous
            return model_degree_prev, float(test_rmse_prev), loc_model_prev, p_local_prev, degree_trace

        if test_rmse < RMSE_THRESH or model_degree == MAX_MODEL_DEGREE:
            return model_degree, float(test_rmse), loc_model, p_local, degree_trace

        model_degree_prev = model_degree
        test_rmse_prev    = test_rmse
        loc_model_prev    = loc_model
        p_local_prev      = p_local
        model_degree     += 1

# =======================  Local TR subproblem (TR in z=(x1,x2) only)  =======================
def subproblem_solver(xk, lm, pl, idata_model, X, Z, bounds, tr_mode, ellipsoid_radius):
    """
    xk: current iterate in R^3
    X:  all design points in R^3 (only used for animations & bounds)
    Z:  all geometry points in R^2 (x1,x2) used for TR construction
    idata_model: indices of local subset within Z
    """
    nloc = len(idata_model)
    Zsub = Z[idata_model, :]   # (m x 2), used by TR

    if tr_mode == 1:
        # Convex hull in z=(x1,x2): enforce
        #   x1 = sum_j λ_j * Zsub[j,0]
        #   x2 = sum_j λ_j * Zsub[j,1]
        #   sum_j λ_j = 1, 0<=λ<=1
        # x0 is free (aside from bounds).
        def f_on_p(p):
            # p = [x0, x1, x2, λ...]
            return f(p[:3])

        m = nloc
        # Build A for equalities in variables p=[x0,x1,x2, λ_1..λ_m]
        # Row1: x1 - Σ λ_j z1j = 0
        row1 = np.zeros(3 + m); row1[1] = 1.0; row1[3:] = -Zsub[:,0]
        # Row2: x2 - Σ λ_j z2j = 0
        row2 = np.zeros(3 + m); row2[2] = 1.0; row2[3:] = -Zsub[:,1]
        # Row3: Σ λ_j = 1
        row3 = np.zeros(3 + m); row3[3:] = 1.0
        Aeq = np.vstack([row1, row2, row3])
        beq = np.array([0.0, 0.0, 1.0])

        lc_eq = LinearConstraint(Aeq, lb=beq, ub=beq)

        # Nonlinear constraint via local model: g_loc(x) <= 0
        def g_loc_on_p(p):
            return g_locmodel_from(lm, pl, p[:3])

        nlc = NonlinearConstraint(g_loc_on_p, -np.inf, 0.0)

        lb_vec = np.concatenate([bounds.lb, np.zeros(m)])
        ub_vec = np.concatenate([bounds.ub, np.ones(m)])
        p_init = np.concatenate([xk, np.full(m, 1.0/max(m,1))])

        res = minimize(
            f_on_p, p_init,
            bounds=Bounds(lb_vec, ub_vec),
            constraints=[nlc, lc_eq],
            method="trust-constr",
            options={'disp': False}
        )

        # ---- READ-ONLY CHECKS (LOCAL CH) ----
        x_star = res.x[:3]
        diag_ch = check_convex_hull_solution(res.x, Zsub, bounds, tol=1e-8)
        diag_mod = check_model_constraints_surrogate_and_true(x_star, lm, pl, tol=1e-10)

        print("[LOCAL CH] CH_ok:", diag_ch["ok"],
              "eq_res:", (diag_ch["r_eq_x1"], diag_ch["r_eq_x2"], diag_ch["r_eq_sumlam"]),
              "lam_range:", (diag_ch["lam_min"], diag_ch["lam_max"]),
              "| g_loc:", diag_mod.get("g_loc"), "g_glob:", diag_mod.get("g_glob"), "g_true:", diag_mod["g_true"])

        return x_star, res

    elif tr_mode == 2:
        # Mahalanobis TR in z=(x1,x2)
        C_loc  = np.cov(Zsub, rowvar=False) + EPS_COV*np.eye(2)
        Cinv_l = pinv(C_loc)
        gamma2 = float(ellipsoid_radius)**2

        zk = np.asarray([xk[1], xk[2]], dtype=float)

        def md2_iter(x):
            z = np.array([x[1], x[2]])
            d = z - zk
            return float(d @ Cinv_l @ d)

        def g_loc_x(x):
            return g_locmodel_from(lm, pl, x)

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

# =======================  Global degree selection (old CV, kept)  =======================
def select_global_polynomial(Z, y, degrees=(1, 2, 3, 4), kfolds=5, random_state=102):
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=random_state)
    best = None
    report = []
    for d in degrees:
        rmses = []
        poly = PolynomialFeatures(degree=d)
        Z_phi = poly.fit_transform(Z)
        for train_idx, val_idx in kf.split(Z_phi):
            Ztr, Zval = Z_phi[train_idx], Z_phi[val_idx]
            ytr, yval = y[train_idx], y[val_idx]
            model = LinearRegression().fit(Ztr, ytr)
            yhat  = model.predict(Zval)
            rmse  = float(np.sqrt(np.mean((yhat - yval)**2)))
            rmses.append(rmse)
        mean_rmse = float(np.mean(rmses))
        std_rmse  = float(np.std(rmses))
        report.append((d, mean_rmse, std_rmse))
        if (best is None) or (mean_rmse < best[1]):
            best = (d, mean_rmse, std_rmse, poly, LinearRegression().fit(Z_phi, y))
    best_degree, best_mean_rmse, best_std_rmse, best_poly, best_model = best
    print("\n=== Global degree CV report (RMSE) ===")
    for d, m, s in report:
        print(f"degree={d}: mean={m:.6f}  std={s:.6f}")
    print(f"Selected global degree = {best_degree} (mean RMSE={best_mean_rmse:.6f})\n")
    return best_poly, best_model

# =======================  Global degree selection (NEW: monotone early-stop)  =======================
def select_global_polynomial_monotone(
    Z, y, degree_init=1, max_degree=6, test_size=0.3, random_state=102, tol=1e-12
):
    """
    Increase polynomial degree starting from degree_init. Split into train/test once.
    Fit on train, compute test RMSE. Stop when test RMSE increases (> prev + tol).
    Finally, refit on ALL data with the chosen degree and return (poly, model).
    """
    Ztr, Zte, ytr, yte = train_test_split(Z, y, test_size=test_size, random_state=random_state, shuffle=True)

    def rmse(yhat, ytrue):
        return float(np.sqrt(np.mean((yhat - ytrue)**2)))

    best_degree = None
    best_rmse   = np.inf
    best_poly   = None
    best_model  = None

    print("\n=== Global degree (holdout) monotone selection ===")
    prev_rmse = np.inf
    d = degree_init
    while d <= max_degree:
        poly = PolynomialFeatures(degree=d)
        Ztr_phi = poly.fit_transform(Ztr)
        model = LinearRegression().fit(Ztr_phi, ytr)
        yhat_te = model.predict(poly.transform(Zte))
        r = rmse(yhat_te, yte)
        print(f"degree={d}: test RMSE={r:.6f}")

        if r + tol > prev_rmse:  # increased (or not improved) -> stop at previous
            if best_degree is None:
                best_degree, best_rmse, best_poly, best_model = d, r, poly, model
            break

        best_degree, best_rmse, best_poly, best_model = d, r, poly, model
        prev_rmse = r
        d += 1

    print(f"Selected global degree = {best_degree} (test RMSE={best_rmse:.6f})")

    # Refit with chosen degree on ALL data
    final_poly = PolynomialFeatures(degree=best_degree)
    Z_all_phi  = final_poly.fit_transform(Z)
    final_model = LinearRegression().fit(Z_all_phi, y)
    return final_poly, final_model

# =======================  Global solve with optional TR (TR in z only)  =======================
def solve_global_with_tr(xk, X, Z, bounds, tr_mode=1, ellipsoid_radius=1.0, m_subset=None):
    """
    xk: current iterate in R^3
    X:  (N x 3) cloud (for plotting/animations)
    Z:  (N x 2) geometry points used for TR & nearest-neighbor subsets
    """
    n = len(xk)

    # distances only in z=(x1,x2)
    zk = np.asarray([xk[1], xk[2]])
    d  = np.linalg.norm(Z - zk, axis=1)
    order = np.argsort(d)
    if m_subset is None:
        m_subset = max(5*n, 2*n+4)
    m_subset = int(min(m_subset, len(order)))
    idata = order[:m_subset]
    Zsub  = Z[idata, :]

    if tr_mode == 1:
        # Convex hull in z; x0 is free.
        def f_on_p(p):
            return f(p[:3])

        m = m_subset
        row1 = np.zeros(3 + m); row1[1] = 1.0; row1[3:] = -Zsub[:,0]
        row2 = np.zeros(3 + m); row2[2] = 1.0; row2[3:] = -Zsub[:,1]
        row3 = np.zeros(3 + m); row3[3:] = 1.0
        Aeq = np.vstack([row1, row2, row3])
        beq = np.array([0.0, 0.0, 1.0])
        lc_eq = LinearConstraint(Aeq, lb=beq, ub=beq)

        def g_glob_on_p(p):
            return g_globmodel(p[:3])

        nlc_g = NonlinearConstraint(g_glob_on_p, -np.inf, 0.0)

        lb_vec = np.concatenate([bounds.lb, np.zeros(m)])
        ub_vec = np.concatenate([bounds.ub, np.ones(m)])
        p0 = np.concatenate([xk, np.full(m, 1.0/m)])

        res = minimize(f_on_p, p0,
                       bounds=Bounds(lb_vec, ub_vec),
                       constraints=[lc_eq, nlc_g],
                       method="trust-constr",
                       options={'disp': False})

        # ---- READ-ONLY CHECKS (GLOBAL CH) ----
        x_star = res.x[:3]
        diag_ch = check_convex_hull_solution(res.x, Zsub, bounds, tol=1e-8)
        g_glob_val = g_globmodel(x_star)
        g_true_val = g(x_star)
        ok_glob = bool(g_glob_val <= 1e-10)
        ok_true = bool(g_true_val <= 1e-10)

        print("[GLOBAL CH] CH_ok:", diag_ch["ok"],
              "eq_res:", (diag_ch["r_eq_x1"], diag_ch["r_eq_x2"], diag_ch["r_eq_sumlam"]),
              "lam_range:", (diag_ch["lam_min"], diag_ch["lam_max"]),
              "| g_glob:", float(g_glob_val), "g_true:", float(g_true_val),
              "ok_glob:", ok_glob, "ok_true:", ok_true)

        return x_star, res

    elif tr_mode == 2:
        C = np.cov(Zsub, rowvar=False) + EPS_COV*np.eye(2)
        Cinv = pinv(C)
        gamma2 = float(ellipsoid_radius)**2
        zk = zk.copy()

        def cons_md(x):
            z = np.asarray([x[1], x[2]])
            d = z - zk
            return gamma2 - float(d @ Cinv @ d)

        nlc_md = NonlinearConstraint(cons_md, 0.0, np.inf)
        nlc_g  = NonlinearConstraint(g_globmodel, -np.inf, 0.0)

        res = minimize(f, xk,
                       bounds=bounds,
                       constraints=[nlc_g, nlc_md],
                       method="trust-constr",
                       options={'disp': False})
        return res.x, res

    else:
        raise ValueError("tr_mode must be 1 or 2")

# =======================  Plot helpers (existing)  =======================
def _apply_fixed_axes(ax):
    ax.set_xlim3d(X2_MIN, X2_MAX)
    ax.set_ylim3d(X1_MIN, X1_MAX)
    ax.set_zlim3d(X0_MIN, X0_MAX)
    ax.set_xlabel('x₂'); ax.set_ylabel('x₁'); ax.set_zlabel('x₀')
    ax.view_init(elev=25, azim=45)

def animate_subsets_3d(X, xis, subsets, pause=1000, repeat=True):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1,1,1))
    try: ax.set_proj_type('ortho')
    except Exception: pass

    N = X.shape[0]

    def update(frame):
        ax.clear()
        idx = subsets[frame]
        xi  = xis[frame+1]

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
    max_iter = subset_sizes.index(max(subset_sizes))
    max_size = subset_sizes[max_iter]
    max_subset = subsets[max_iter]
    xi_max = xis[max_iter+1]

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1,1,1))
    try: ax.set_proj_type('ortho')
    except Exception: pass

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
    used = sorted(set(j for idxs in subsets for j in idxs))
    mask_used = np.zeros(X.shape[0], dtype=bool)
    mask_used[used] = True

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1,1,1))
    try: ax.set_proj_type('ortho')
    except Exception: pass

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

# ---------- 2D overview plot with key solutions ----------
def plot_data_and_solutions_2d(Z, x_start, x_true, x_global, x_local_last, title="Overview (x₁–x₂)"):
    # Contour of q(x1,x2)
    delta = 0.25
    x1 = np.arange(-5.0, 5.0, delta)
    x2 = np.arange(-5.0, 5.0, delta)
    X1, X2 = np.meshgrid(x1, x2)
    Zq = np.vectorize(lambda a,b: q_val(a,b))(X1, X2)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    cs = ax.contourf(X1, X2, Zq, levels=30)
    plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04, label="q(x₁,x₂)")

    # Data cloud
    ax.scatter(Z[:,0], Z[:,1], s=10, c='k', alpha=0.55, label='data (x₁,x₂)')

    # q=0 contour line for feasibility boundary
    ax.contour(X1, X2, Zq, levels=[0.0], colors='w', linewidths=1.8)

    # Markers
    ax.scatter(x_start[1],  x_start[2],  s=80, c='r', marker='x',  label='start x')
    ax.scatter(x_true[1],   x_true[2],   s=90, c='cyan', marker='*', edgecolors='k', linewidths=0.6, label='true optimum (x₁,x₂)')
    ax.scatter(x_global[1], x_global[2], s=75, c='magenta', marker='s', edgecolors='k', linewidths=0.5, label='global model sol.')
    ax.scatter(x_local_last[1], x_local_last[2], s=95, c='C0', marker='D', edgecolors='k', linewidths=0.6, label='last local sol.')

    ax.set_xlabel('x₁'); ax.set_ylabel('x₂')
    ax.set_xlim(X1_MIN, X1_MAX); ax.set_ylim(X2_MIN, X2_MAX)
    ax.set_title(title)
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.show()

# =======================  (NEW) Visuals focused on first K iterations  =======================
def plot_first_k_subsets_2d(Z, per_iter_info, K=3, title="İlk K iterasyonda seçilen alt-kümeler ve adımlar"):
    K = min(K, len(per_iter_info))
    if K == 0:
        print("(Görselleştirilecek iterasyon yok.)")
        return
    delta = 0.25
    x1 = np.arange(X1_MIN, X1_MAX+delta, delta)
    x2 = np.arange(X2_MIN, X2_MAX+delta, delta)
    X1m, X2m = np.meshgrid(x1, x2)
    Zq = np.vectorize(lambda a,b: q_val(a,b))(X1m, X2m)

    fig, axes = plt.subplots(1, K, figsize=(6*K, 5), squeeze=False)
    axes = axes[0]
    for j in range(K):
        info = per_iter_info[j]
        ax = axes[j]
        ax.contourf(X1m, X2m, Zq, levels=30, alpha=0.9)
        ax.contour(X1m, X2m, Zq, levels=[0.0], colors='w', linewidths=1.6)

        used_mask = np.zeros(Z.shape[0], dtype=bool)
        used_mask[info["subset_idx"]] = True
        ax.scatter(Z[~used_mask,0], Z[~used_mask,1], s=10, c='k', alpha=0.35, label='data (unused)')
        ax.scatter(Z[used_mask,0],  Z[used_mask,1],  s=28, c=SUBSET_COLOR, edgecolors='k', linewidths=0.3, label=f'local subset (n={info["subset_size"]})')

        # step from x_prev to x_new
        xprev = info["x_prev"]; xnew = info["x_new"]
        ax.plot([xprev[1], xnew[1]], [xprev[2], xnew[2]], '-o', lw=2.0, ms=5)
        ax.scatter(xprev[1], xprev[2], s=70, c='r', marker='x', label='x_prev' if j==0 else None)
        ax.scatter(xnew[1],  xnew[2],  s=80, c='C0', marker='D', edgecolors='k', linewidths=0.6, label='x_new' if j==0 else None)

        ax.set_xlim(X1_MIN, X1_MAX); ax.set_ylim(X2_MIN, X2_MAX)
        ax.set_xlabel('x₁'); ax.set_ylabel('x₂')
        ax.set_title(f"Iter {info['iter']}: deg={info['model_degree']}  g_true={info['g_true']:.3g}")
        if j==0:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_degree_rmse_traces(per_iter_info, K=3, title="İlk K iterasyon için derece–RMSE izleri"):
    K = min(K, len(per_iter_info))
    if K == 0:
        return
    fig, axes = plt.subplots(1, K, figsize=(5*K, 4), squeeze=False)
    axes = axes[0]
    for j in range(K):
        info = per_iter_info[j]
        degs  = [d for d,_ in info["degree_trace"]]
        rmses = [r for _,r in info["degree_trace"]]
        ax = axes[j]
        ax.plot(degs, rmses, marker='o', lw=2)
        ax.axvline(info["model_degree"], ls='--')
        ax.set_xlabel("Model derecesi")
        ax.set_ylabel("Test RMSE")
        ax.set_title(f"Iter {info['iter']}  (seçilen: {info['model_degree']})")
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_iterate_path_2d(xis, title="Iterasyon yolu (x₁–x₂)"):
    xs = np.array(xis)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    # feasibility boundary
    delta = 0.25
    x1 = np.arange(X1_MIN, X1_MAX+delta, delta)
    x2 = np.arange(X2_MIN, X2_MAX+delta, delta)
    X1m, X2m = np.meshgrid(x1, x2)
    Zq = np.vectorize(lambda a,b: q_val(a,b))(X1m, X2m)
    ax.contourf(X1m, X2m, Zq, levels=30, alpha=0.9)
    ax.contour(X1m, X2m, Zq, levels=[0.0], colors='w', linewidths=1.6)

    ax.plot(xs[:,1], xs[:,2], '-o', lw=2.0, ms=4)
    ax.scatter(xs[0,1], xs[0,2], s=80, c='r', marker='x', label='start')
    ax.scatter(xs[-1,1], xs[-1,2], s=95, c='C0', marker='D', edgecolors='k', linewidths=0.6, label='last')

    ax.set_xlim(X1_MIN, X1_MAX); ax.set_ylim(X2_MIN, X2_MAX)
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂')
    ax.set_title(title)
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.show()

def print_first_k_iteration_report(per_iter_info, K=3):
    K = min(K, len(per_iter_info))
    if K == 0:
        print("(Raporlanacak iterasyon yok.)")
        return
    rows = []
    for j in range(K):
        info = per_iter_info[j]
        rows.append({
            "Iter": info["iter"],
            "Subset n": info["subset_size"],
            "Degree": info["model_degree"],
            "Best RMSE": min(r for _,r in info["degree_trace"]),
            "f(x_new)": info["f_val"],
            "g_true(x_new)": info["g_true"],
            "g_loc(x_new)": info["g_loc"],
            "g_glob(x_new)": info["g_glob"]
        })
    df = pd.DataFrame(rows)
    print("\n=== Summary for the first K iterations ===")
    print(df.to_string(index=False))

# =======================  Data generation helpers  =======================
def sample_elliptical_gaussian(
    n, mu, var_long=3.0, var_short=0.03,
    orient="H_eig1", angle_rad=None,
    exclude_center=None, exclude_radius=0.0,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    # Build rotation
    if orient in ("H_eig1", "H_eig2"):
        w, V = np.linalg.eigh(Q_H)  # columns of V are eigenvectors
        if orient == "H_eig1":
            e = V[:, 0]  # eigenvector of smaller eigenvalue (flatter)
        else:
            e = V[:, 1]  # eigenvector of larger eigenvalue
        e = e / (np.linalg.norm(e) + 1e-15)
        e_perp = np.array([-e[1], e[0]])
        R = np.column_stack([e, e_perp])  # 2x2
    elif orient == "angle":
        if angle_rad is None:
            angle_rad = 0.0
        ca, sa = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[ca, -sa],
                      [sa,  ca]])
    else:
        raise ValueError("orient must be 'H_eig1', 'H_eig2', or 'angle'")

    S = np.diag([var_long, var_short])
    Sigma = R @ S @ R.T

    out = []
    mu = np.asarray(mu, dtype=float)
    while len(out) < n:
        batch = rng.multivariate_normal(mean=mu, cov=Sigma, size=(n - len(out)))
        if exclude_center is not None and exclude_radius > 0.0:
            keep = np.linalg.norm(batch - exclude_center, axis=1) >= exclude_radius
            batch = batch[keep]
        out.append(batch)
    Z = np.vstack(out)[:n]  # geometry points (x1,x2)
    return Z

# =======================  NOWDCL main loop (models on Z; TR in Z) =======================
def nowdcl(f, X, Z, y, bounds, x_0, niters):
    """
    X: (N x 3) cloud for animations (x0 column here is dummy; TR ignores it)
    Z: (N x 2) geometry points used by models/TR
    y: q(Z) values (labels for constraint surrogate)
    """
    global local_model, poly_local, n, subset_indices_list

    MAX_NDATA   = 100000
    RMSE_THRESH = 9e-1
    DELTA_MIN   = 1e-6

    xit = np.array(x_0, dtype=float)
    xis = [np.array(x_0, copy=True)]

    local_models = []
    poly_locals  = []
    subsets_used = []
    per_iter_info = []  # <-- NEW

    # distances in Z=(x1,x2)
    zk = np.asarray([xit[1], xit[2]])
    dists = np.linalg.norm(Z - zk, axis=1)
    sorted_indices = np.argsort(dists).tolist()

    ndata = len(y)
    learning_size  = 2 * len(x_0)
    ndata_model_0  = np.minimum(MAX_NDATA, np.minimum(5 * learning_size, ndata))

    print("iteration 0:", " trial_x: ", xit, "f(x): ", f(xit), "g(x)=q(x1,x2): ", g(xit))

    for i in range(niters):
        model_degree = 1
        ndata_model  = ndata_model_0
        weights = update_weights(xit, dists)
        ybar = weights * y
        Zbar = weights.reshape(-1, 1) * Z

        best_test_rmse   = np.inf
        best_ndata_model = 0
        local_model = None
        poly_local  = None
        iter_degree_trace = []  # <-- NEW: to store trace for the selected ndata_model

        while ndata_model <= ndata:
            md_tmp, rmse_tmp, lm_tmp, pl_tmp, degree_trace_tmp = set_model_degree(
                model_degree, sorted_indices, ndata_model, Zbar, ybar
            )
            if rmse_tmp < best_test_rmse:
                best_test_rmse   = rmse_tmp
                local_model      = lm_tmp
                model_degree     = md_tmp
                poly_local       = pl_tmp
                best_ndata_model = ndata_model
                iter_degree_trace = degree_trace_tmp[:]  # keep the trace for this setting
            else:
                break
            ndata_model += 2 * learning_size

        idata_model = sorted_indices[:best_ndata_model]

        local_models.append(local_model)
        poly_locals.append(poly_local)

        if best_test_rmse > RMSE_THRESH:
            print("Stopping: local model not accurate enough (rmse =", best_test_rmse, ")")
            return -2, xis, local_models, poly_locals, subsets_used, per_iter_info

        x_prev = xit.copy()
        x_trial, res_loc = subproblem_solver(
            xk=xit,
            lm=local_model,
            pl=poly_local,
            idata_model=idata_model,
            X=X,
            Z=Z,
            bounds=bounds,
            tr_mode=TR_MODE,
            ellipsoid_radius=ELLIPSOID_RADIUS
        )

        if np.linalg.norm(x_trial[:n] - xit) < DELTA_MIN:
            print("No significant progress; terminating.")
            # record the (non) step as well
            info = {
                "iter": i+1,
                "subset_idx": idata_model.copy(),
                "subset_size": len(idata_model),
                "model_degree": model_degree,
                "degree_trace": iter_degree_trace[:],
                "x_prev": x_prev,
                "x_new": xit.copy(),
                "f_val": f(xit),
                "g_true": g(xit),
                "g_loc": g_locmodel(np.array(xit)),
                "g_glob": g_globmodel(np.array(xit))
            }
            per_iter_info.append(info)
            return 0, xis, local_models, poly_locals, subsets_used, per_iter_info

        xit = x_trial[:n]
        xis.append(xit.copy())

        subsets_used.append(idata_model.copy())
        subset_indices_list.append(idata_model.copy())

        print("iteration ", i+1, ", trial_x: ", xit,
              "f(x): ", f(xit),
              "g_true(x)=q(x1,x2): ", g(xit),
              "g_locmodel(x) =", g_locmodel(np.array(xit)))

        # log iteration info
        info = {
            "iter": i+1,
            "subset_idx": idata_model.copy(),
            "subset_size": len(idata_model),
            "model_degree": model_degree,
            "degree_trace": iter_degree_trace[:],
            "x_prev": x_prev,
            "x_new": xit.copy(),
            "f_val": f(xit),
            "g_true": g(xit),
            "g_loc": g_locmodel(np.array(xit)),
            "g_glob": g_globmodel(np.array(xit))
        }
        per_iter_info.append(info)

        # update distances in Z
        zk = np.asarray([xit[1], xit[2]])
        dists = np.linalg.norm(Z - zk, axis=1)
        sorted_indices = np.argsort(dists).tolist()

    return -1, xis, local_models, poly_locals, subsets_used, per_iter_info

# =======================  Main  =======================
if __name__ == "__main__":
    # ---------- Initialize quadratic constraint ----------
    init_quadratic(seed=42)

    print("=== Quadratic constraint coefficients ===")
    print("H =\n", Q_H)
    print("b =", Q_b)
    print("c =", Q_c)
    print("d =", Q_d)
    print("Ellipsoid center mu =", Q_mu, "  radius R =", Q_R)
    print("Check q(mu) =", q_val(Q_mu[0], Q_mu[1]), "  (≈ -R^2)")

    # Decision variable dimension
    n   = 3
    x_0 = np.array([0.0, -1.0, -1.0])   # starting point (x0,x1,x2)

    np.random.seed(102)
    random.seed(102)
    rng = np.random.default_rng(102)
    print("Random seed fixed!")

    # --------- Generate long, skinny, rotated cloud away from the optimum ---------
    nsamples = 300
    w_eigs, V_eigs = np.linalg.eigh(Q_H)
    v_flat = V_eigs[:, 0]                  # flattest direction
    v_flat = v_flat / (np.linalg.norm(v_flat) + 1e-15)
    mu_shift = Q_mu + 3.0 * v_flat

    Z = sample_elliptical_gaussian(
        nsamples,
        mu=mu_shift,
        var_long=3.0,
        var_short=0.03,
        orient="H_eig1",
        exclude_center=Q_mu,
        exclude_radius=1.2,
        rng=rng
    )  # (N x 2) = (x1,x2)

    # Build labels for constraint surrogate: y = q(x1,x2)
    y = np.array([q_val(z[0], z[1]) for z in Z])

    # Build a 3D cloud X for plotting/animations: set x0=0 for each sample (TR ignores x0)
    X = np.c_[np.zeros(nsamples), Z]

    # ---------- Ground-truth nonlinear solve (with NEW objective/constraint) ----------
    bounds = Bounds([-np.inf, -5, -5], [np.inf, 5, 5])
    nlc    = NonlinearConstraint(g, -np.inf, 0.0)
    res    = minimize(f, x_0, bounds=bounds, constraints=[nlc], method="trust-constr")
    print('True solution: ', res.x)
    print("f: ", f(res.x), "g=q: ", g(res.x))

    # ---------- Global polynomial with monotone early-stop degree ----------
    poly_global, global_model = select_global_polynomial_monotone(
        Z, y, degree_init=1, max_degree=6, test_size=0.3, random_state=102, tol=1e-12
    )

    # Global + Convex-hull TR in (x1,x2)
    xg_ch, res_ch = solve_global_with_tr(
        xk=x_0, X=X, Z=Z, bounds=bounds,
        tr_mode=1,
        ellipsoid_radius=ELLIPSOID_RADIUS,
        m_subset=5*len(x_0)
    )
    print('Global model + CH TR : ', xg_ch, " true g=q:", g(xg_ch))

    # After printing xg_ch and xg_maha:
    x_global = xg_ch  # or xg_maha

    # Global + Mahalanobis TR in (x1,x2)
    xg_maha, res_maha = solve_global_with_tr(
        xk=x_0, X=X, Z=Z, bounds=bounds,
        tr_mode=2,
        ellipsoid_radius=ELLIPSOID_RADIUS,
        m_subset=5*len(x_0)
    )
    print('Global model + Maha  : ', xg_maha, " true g=q:", g(xg_maha))

    # ---------- Local refinement ----------
    niters = 5
    out = nowdcl(f, X, Z, y, bounds, x_0, niters)

    if isinstance(out, tuple) and len(out) >= 5:
        if len(out) == 6:
            exitcode, xis, local_models, poly_locals, subsets_used, per_iter_info = out
        else:
            exitcode, xis, local_models, poly_locals, subsets_used = out
            per_iter_info = []
    else:
        exitcode = out
        xis = [np.array(x_0)]
        local_models, poly_locals, subsets_used, per_iter_info = [], [], [], []

    # ===== Per-iteration diagnostics table =====
    records = []
    Ktab = min(len(local_models), len(xis))
    for i in range(Ktab):
        x_i = np.array(xis[i])
        true_val = g(x_i)             # q(x1,x2)
        glob_pred = g_globmodel(x_i)
        lm = local_models[i]
        pl = poly_locals[i]
        loc_pred = float(lm.predict(pl.fit_transform(_x_to_z(x_i)))[0])

        records.append({
            "Iter": i,
            "x0": x_i[0], "x1": x_i[1], "x2": x_i[2],
            "True q(x1,x2)": true_val,
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

    # Align for plotting: accepted steps = len(xis)-1
    if len(subsets_used) > len(xis) - 1:
        subsets_used = subsets_used[:len(xis) - 1]

    # ---------- 2D overview plot with all key points ----------
    x_local_last = np.array(xis[-1]) if len(xis) else np.array(x_0)
    plot_data_and_solutions_2d(
        Z,
        x_start=np.array(x_0),
        x_true=res.x,
        x_global=x_global,
        x_local_last=x_local_last,
        title="Data & Solutions in 2D (x₁–x₂)"
    )

    # ---------- NEW: First-3-iteration focused analysis ----------
    print_first_k_iteration_report(per_iter_info, K=3)
    plot_first_k_subsets_2d(Z, per_iter_info, K=3,
                            title="Local subsets in the first three iterations, q=0 and steps")
    plot_degree_rmse_traces(per_iter_info, K=3,
                            title="Degree for the first three iterations–RMSE trail")
    plot_iterate_path_2d(xis, title="Iteration (x₁–x₂)")

    # Optional extra 3D diagnostics/animations
    if len(subsets_used) and len(xis) > 1:
        plot_winner_iteration_3d(X, xis, subsets_used)
        plot_union_used_points_3d(X, subsets_used)
        animate_subsets_3d(X, xis, subsets_used, pause=1000)
