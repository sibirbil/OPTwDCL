#diet_problem.py

from scipy.optimize import (
    LinearConstraint, NonlinearConstraint, Bounds, minimize,
)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import linprog


def _jac_g_locmodel_p_factory(local_model, poly_local, n, local_n):
    """Return a Jacobian function for g_locmodel(p) with shape (n+local_n,).
    Assumes local_model is a sklearn LinearRegression-like model fitted on poly_local features of x (length n).
    z-variables do not enter g.
    """
    #approximate gradient from the polynomial basis directly if available. otherwise fall back to finite differences.
    try:
        powers = poly_local.powers_
        coef = getattr(local_model, "coef_", None)
        if coef is None:
            #Ridge/LinearRegression stores coef_ as (n_terms,) or (1, n_terms)
            coef = getattr(local_model, "coef", None)
            if coef is not None:
                coef = np.ravel(coef)

        def jac(p):
            x = np.asarray(p[:n], dtype=float)
            #compute gradient in x
            grad_x = np.zeros(n, dtype=float)
            if coef is None or powers is None:
                #fallback?
                eps = 1e-6
                base = float(local_model.predict(poly_local.transform(x.reshape(1, -1)))[0])
                for j in range(n):
                    xp = x.copy(); xm = x.copy()
                    xp[j] += eps; xm[j] -= eps
                    fp = float(local_model.predict(poly_local.transform(xp.reshape(1, -1)))[0])
                    fm = float(local_model.predict(poly_local.transform(xm.reshape(1, -1)))[0])
                    grad_x[j] = (fp - fm) / (2 * eps)
            else:
                #bias term doesn't contribute to gradient
                for j in range(n):
                    s = 0.0
                    for m in range(1, powers.shape[0]): 
                        e_j = powers[m, j]
                        if e_j == 0:
                            continue
                        term = 1.0
                        for k in range(n):
                            e = powers[m, k] - (1 if k == j else 0)
                            if e > 0:
                                term *= x[k] ** e
                        s += coef[m] * e_j * term
                    grad_x[j] = s
            return np.concatenate([grad_x, np.zeros(local_n, dtype=float)])

        return jac

    except Exception:
        #defensive?
        def jac(p):
            x = np.asarray(p[:n], dtype=float)
            eps = 1e-6
            grad_x = np.zeros(n, dtype=float)
            base = float(local_model.predict(poly_local.transform(x.reshape(1, -1)))[0])
            for j in range(n):
                xp = x.copy(); xm = x.copy()
                xp[j] += eps; xm[j] -= eps
                fp = float(local_model.predict(poly_local.transform(xp.reshape(1, -1)))[0])
                fm = float(local_model.predict(poly_local.transform(xm.reshape(1, -1)))[0])
                grad_x[j] = (fp - fm) / (2 * eps)
            return np.concatenate([grad_x, np.zeros(local_n, dtype=float)])

        return jac


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import warnings
import math
import random

#globals
global global_model
global local_model
global poly_global
global poly_local
global n

#record subsets of data used each iteration
global subset_indices_list
subset_indices_list = []

#3D plot styling
SUBSET_COLOR = "#FF7F0E"  #orange fill for "used"
ALLDATA_COLOR = "lightgray"  #fill for "not used"
EDGE_COLOR = "black"
EDGE_LW_USED = 0.6
EDGE_LW_NOT = 0.6
POINT_SIZE_USED = 60
POINT_SIZE_NOT = 60

#fixed axis ranges (for the 3D demo where x0 is vertical)
X1_MIN, X1_MAX = -5.0, 5.0
X2_MIN, X2_MAX = -5.0, 5.0
X0_MIN, X0_MAX = 0.0, 1500.0

warnings.filterwarnings(
    "ignore",
    message="delta_grad == 0.0. Check if the approximated function is linear. If the function is linear better results can be obtained by defining the Hessian as zero instead of using quasi-Newton approximations.",
)
warnings.filterwarnings(
    "ignore",
    message="Singular Jacobian matrix. Using SVD decomposition to perform the factorizations.",
)


#default f,g (Rosenbrock-style)
def f(x):
    return x[0]


def g(x):
    #classic Rosenbrock-shaped constraint
    return -x[0] + 100 * (x[2] - x[1] * x[1]) ** 2 + (1 - x[1]) ** 2


#surrogate wrappers
def g_globmodel(x):
    x = np.asarray(x).reshape(1, -1)
    x_poly = poly_global.transform(x)
    return float(global_model.predict(x_poly)[0])


def g_locmodel(x):
    if (np.size(x) > n):
        x = np.asarray(x)[:n]
    x = np.asarray(x).reshape(1, -1)
    x_poly_local = poly_local.transform(x)
    return float(local_model.predict(x_poly_local)[0])


#local model selection
def update_weights(xit, dists):
    lam = 0.0
    return np.exp(-lam * dists)


def set_model_degree(degree_init, sorted_indices, ndata_model, Xbar, ybar):
    RMSE_THRESH = 1e-1
    MAX_MODEL_DEGREE = 1
    model_degree = degree_init

    ndata_train = math.floor(0.7 * ndata_model)
    ndata_test = ndata_model - ndata_train
    idata_model = sorted_indices[0:ndata_model]
    idata_train = random.sample(idata_model, k=ndata_train)
    idata_test = np.setdiff1d(idata_model, idata_train).tolist()

    Xbar_train = Xbar[idata_train, :]
    ybar_train = ybar[idata_train]
    Xbar_test = Xbar[idata_test, :]
    ybar_test = ybar[idata_test]

    isDone = False
    test_rmse_prev = np.inf
    while not isDone:
        p_local = PolynomialFeatures(degree=model_degree)
        X_poly_local = p_local.fit_transform(Xbar_train)
        loc_model = LinearRegression().fit(X_poly_local, ybar_train)

        X_poly_test = p_local.fit_transform(Xbar_test)
        ypred_test = loc_model.predict(X_poly_test)
        test_rmse = np.linalg.norm(
            (ypred_test - ybar_test) / np.maximum(np.maximum(np.abs(ypred_test), np.abs(ybar_test)), 1e-12)
        ) / max(ndata_test, 1)

        print("ndata_model : ", ndata_model, "ndata_test : ", ndata_test, "ndata_train : ", ndata_train, "model_degree : ", model_degree, "test_rmse : ", test_rmse)

        if test_rmse > test_rmse_prev:
            return model_degree_prev, test_rmse_prev, loc_model_prev, p_local_prev

        if test_rmse < RMSE_THRESH or model_degree == MAX_MODEL_DEGREE:
            return model_degree, test_rmse, loc_model, p_local

        model_degree_prev = model_degree
        test_rmse_prev = test_rmse
        loc_model_prev = loc_model
        p_local_prev = p_local
        model_degree += 1


#LinearConstraints act on p = [x,z]
def _augment_linear_constraints_for_p(extra_constraints, n, local_n):
    """Make each LinearConstraint act on the augmented decision vector p=[x,z].
    If A has width n, pad with local_n zero columns to width n+local_n.
    If A already has width n+local_n, pass through.
    """
    augmented = []
    for c in (extra_constraints or []):
        if isinstance(c, LinearConstraint):
            A = np.asarray(c.A)
            if A.shape[1] == n:
                A_aug = np.hstack([A, np.zeros((A.shape[0], local_n))])
                augmented.append(LinearConstraint(A_aug, lb=c.lb, ub=c.ub))
            elif A.shape[1] == n + local_n:
                augmented.append(c)
            else:
                raise ValueError(
                    f"LinearConstraint width {A.shape[1]} incompatible with p-dim {n}+{local_n}."
                )
        else:
            augmented.append(c)
    return augmented


#bound checks, solver talk helpers
def _check_and_report_bounds(lb_vec, ub_vec, n, local_n, p_init=None, name="p", tol=1e-12):
    assert lb_vec.shape == ub_vec.shape, "lb_vec/ub_vec must have identical shapes"
    assert lb_vec.ndim == 1, "lb_vec/ub_vec must be 1-D"
    assert lb_vec.size == n + local_n, f"bounds length must be n+local_n = {n+local_n}"

    bad_nan = np.isnan(lb_vec) | np.isnan(ub_vec)
    bad_ord = lb_vec > ub_vec + tol

    #summary
    width = ub_vec - lb_vec
    fixed = np.isfinite(lb_vec) & np.isfinite(ub_vec) & (np.abs(width) <= tol)
    print(f"[bounds] Checking {name}: dim={n}+{local_n} (x+z)")
    print(f" NaNs: {bad_nan.sum()}, lb>ub: {bad_ord.sum()}, fixed vars: {fixed.sum()}")

    with np.errstate(invalid='ignore'):
        finite_width = width[np.isfinite(width)]
        if finite_width.size:
            print(f" width stats (finite only): min={finite_width.min():.3g}, "
                  f"median={np.median(finite_width):.3g}, max={finite_width.max():.3g}")

    #per block?
    print(f" x-block: [{lb_vec[:n].min():.3g}, {ub_vec[:n].max():.3g}] "
          f"| z-block: [{lb_vec[n:].min():.3g}, {ub_vec[n:].max():.3g}]")

    if bad_nan.any():
        raise ValueError("Found NaNs in bounds.")
    if bad_ord.any():
        idx = np.where(bad_ord)[0][:10]
        raise ValueError(f"Found lb>ub at indices (first 10): {idx}")

    if p_init is not None:
        below = p_init < lb_vec - 1e-9
        above = p_init > ub_vec + 1e-9
        if below.any() or above.any():
            idx = np.where(below | above)[0][:10]
            print(f"[bounds] WARNING: initial point outside bounds at indices (first 10): {idx}")
        else:
            print("[bounds] Initial point is within bounds.")


def _make_solver_callback(f, n):
    #trust-constr calls: callback(xk, state)
    def cb(xk, state):
        #defensive against SciPy version differences
        nit = getattr(state, "iteration", getattr(state, "nit", getattr(state, "niter", None)))
        cv = getattr(state, "constr_violation", getattr(state, "maxcv", None))
        opt = getattr(state, "optimality", None)
        tr = getattr(state, "tr_radius", getattr(state, "trust_radius", None))

        fx = None
        try:
            fx = float(f(xk))
        except Exception:
            pass

        msg = "[subsolve step]"
        if nit is not None:
            msg += f" nit={nit}"
        if fx is not None:
            msg += f" f={fx:.6g}"
        if cv is not None:
            msg += f" maxcv={cv:.3e}"
        if opt is not None:
            msg += f" opt={opt:.3e}"
        if tr is not None:
            msg += f" tr={tr:.3g}"
        print(msg)
        return False

    return cb


def _report_subproblem_result(res, n, local_n, f, g, g_locmodel, A_eq=None, b_eq=None, aug_lin=None, tol=1e-6):
    print("\n===== Subproblem Result =====")
    print(f"success={res.success} status={res.status} message={res.message}")
    print(f"nit={getattr(res, 'nit', None)} nfev={getattr(res, 'nfev', None)} "
          f"cg_niter={getattr(res, 'cg_niter', None)}")
    print(f"fun={getattr(res, 'fun', None)} constr_violation={getattr(res, 'constr_violation', None)}")

    p_star = res.x
    x_star = p_star[:n]
    z_star = p_star[n:]

    print(f"x* (first 10): {np.array2string(x_star[:10], precision=4, floatmode='maxprec_equal')}")
    print(f"z* (len={local_n}, first 10): {np.array2string(z_star[:10], precision=4, floatmode='maxprec_equal')}")

    #objective & constraints at x*
    try:
        print(f"f(x*)={float(f(p_star)):.6g}")
    except Exception:
        try:
            print(f"f(x*)={float(f(x_star)):.6g} (evaluated on x only)")
        except Exception:
            pass
    try:
        print(f"g_true(x*)={float(g(x_star)):.6g}")
    except Exception:
        pass
    try:
        print(f"g_local(x*)={float(g_locmodel(x_star)):.6g}")
    except Exception:
        pass

    #linear tie constraint residual: A p = 0
    if A_eq is not None:
        r = A_eq @ p_star
        print(f"||A_tie p||_inf = {np.linalg.norm(r, ord=np.inf):.3e}")

    #any extra linear constraints
    if aug_lin:
        worst = 0.0
        for k, c in enumerate(aug_lin):
            if not isinstance(c, LinearConstraint):
                continue
            Ax = c.A @ p_star
            lb = np.broadcast_to(c.lb, Ax.shape)
            ub = np.broadcast_to(c.ub, Ax.shape)
            viol = np.maximum(lb - Ax, 0) + np.maximum(Ax - ub, 0)
            v = float(np.max(viol))
            worst = max(worst, v)
            if v > tol:
                print(f"[extra lin #{k}] max violation={v:.3e}")
        print(f"max extra linear-constraint violation = {worst:.3e}")

    print("===== End Subproblem Result =====\n")


def _objective_vector_from_f(f, n, local_n):
    """Recover linear objective c for p=[x,z] from f. Assumes f is linear in x and ignores z."""
    c = np.zeros(n + local_n, dtype=float)
    #f(0) should be 0 for cost^T x ?
    p0 = np.zeros(n + local_n, dtype=float)
    f0 = float(f(p0))
    for j in range(n):
        pj = p0.copy()
        pj[j] = 1.0
        c[j] = float(f(pj)) - f0
    return c

def _affine_g_coeffs(local_model, poly_local, n):
    """
    Extract affine coefficients for g_loc(x) ~= a0 + a^T x for degree-1 PolynomialFeatures.
    Works even if the sklearn model keeps an intercept and the poly has a bias term.
    If shapes look odd, falls back to black-box probing.
    """
    #try structured extraction first
    try:
        degree = getattr(poly_local, "degree", None)
        include_bias = getattr(poly_local, "include_bias", True)
        if degree != 1:
            raise ValueError("Local surrogate degree is not 1; LP not applicable.")
        coef = np.ravel(getattr(local_model, "coef_", local_model.coef))
        intercept = float(getattr(local_model, "intercept_", 0.0))
        if include_bias:
            #features: [1, x1, x2, ..., xn]
            if coef.size < n + 1:
                raise ValueError("coef size mismatch")
            a0 = intercept + coef[0]
            a = coef[1:1+n].copy()
            if a.size != n:
                raise ValueError("a size mismatch")
            return a0, a
        else:
            #features: [x1, ..., xn], intercept separately
            if coef.size < n:
                raise ValueError("coef size mismatch")
            a0 = intercept
            a = coef[:n].copy()
            return a0, a
    except Exception:
        #robust black-box fallback: probe g at 0 and basis
        x0 = np.zeros(n, dtype=float)
        g0 = float(local_model.predict(poly_local.transform(x0.reshape(1, -1)))[0])
        a = np.zeros(n, dtype=float)
        for j in range(n):
            x = x0.copy(); x[j] = 1.0
            gij = float(local_model.predict(poly_local.transform(x.reshape(1, -1)))[0])
            a[j] = gij - g0
        a0 = g0
        return a0, a

def _convert_linear_constraints_to_lp(aug_extras, n, local_n):
    """
    Convert a list of LinearConstraint (already padded to [x,z]) into A_ub,b_ub,A_eq,b_eq for linprog.
    Handles ranges lb <= A p <= ub by splitting to <= and == rows where appropriate.
    """
    A_ub_rows, b_ub = [], []
    A_eq_rows, b_eq = [], []
    for c in aug_extras:
        if not isinstance(c, LinearConstraint):
            continue
        A = np.asarray(c.A, dtype=float)
        lb = np.asarray(c.lb, dtype=float).reshape(-1)
        ub = np.asarray(c.ub, dtype=float).reshape(-1)
        #broadcast if needed
        if lb.size == 1 and A.shape[0] > 1: lb = np.full(A.shape[0], lb.item())
        if ub.size == 1 and A.shape[0] > 1: ub = np.full(A.shape[0], ub.item())
        for i in range(A.shape[0]):
            Ai = A[i, :].reshape(1, -1)
            lbi = lb[i]
            ubi = ub[i]
            if np.isfinite(lbi) and np.isfinite(ubi) and np.isclose(lbi, ubi):
                A_eq_rows.append(Ai.ravel())
                b_eq.append(lbi)
            else:
                if np.isfinite(ubi):
                    A_ub_rows.append(Ai.ravel())
                    b_ub.append(ubi)
                if np.isfinite(lbi):
                    A_ub_rows.append((-Ai).ravel())
                    b_ub.append(-lbi)
    A_ub = np.vstack(A_ub_rows) if len(A_ub_rows) else None
    b_ub = np.asarray(b_ub, dtype=float) if len(b_ub) else None

    A_eq = np.vstack(A_eq_rows) if len(A_eq_rows) else None
    b_eq = np.asarray(b_eq, dtype=float) if len(b_eq) else None
    return A_ub, b_ub, A_eq, b_eq

#NOWDCL
def nowdcl(
    f, X, y, bounds, x_0, niters, extra_constraints=None,
    talk=True, check_bounds=True, tol=1e-9, max_subiter=500, max_local_n=120
):
    """
    f: objective callable (called on p=[x,z]; can ignore z if desired)
    X: (N, n) design points
    y: (N,) targets for the learned constraint g(x) (i.e., y = g_true(x_i))
    bounds: Bounds over the decision variables (length n)
    x_0: initial point (length n)
    niters: iteration cap
    extra_constraints: optional list of SciPy constraints (e.g., LinearConstraint) to include in the subproblem solve; can be None.

    Extra knobs:
    - talk: print solver progress + result report
    - check_bounds: validate & summarize bounds before solving
    - tol: numeric tolerance for tiny violations
    - max_subiter: max iterations for the trust-constr subsolver
    """

    USE_CONVEX_HULL = True  #True → add sum(z)=1; False → cone (no sum row)

    global local_model, poly_local, n, subset_indices_list

    if extra_constraints is None:
        extra_constraints = []

    #robustly define n from inputs
    n = int(np.asarray(x_0, dtype=float).size)

    MAX_NDATA = 100000
    RMSE_THRESH = 9e-1
    DELTA_MIN = 1e-6

    xit = np.array(x_0, dtype=float)
    xis = [np.array(x_0, copy=True)]

    #store per-iteration local surrogates + subset indices
    local_models = []
    poly_locals = []
    subsets_used = []

    dists = np.linalg.norm(xit - X, axis=1)
    sorted_indices = np.argsort(dists).tolist()
    ndata = len(y)
    learning_size = 2 * len(x_0)
    ndata_model_0 = np.minimum(MAX_NDATA, np.minimum(5 * learning_size, ndata))
    ndata_model_0 = int(min(ndata_model_0, max_local_n))

    print("iteration 0:", " trial_x: ", xit, "f(x): ", f(xit), "g(x): ", g(xit))

    for i in range(niters):
        model_degree = 1
        ndata_model = ndata_model_0
        weights = update_weights(xit, dists)
        ybar = weights * y
        Xbar = weights.reshape(-1, 1) * X

        best_test_rmse = np.inf
        best_ndata_model = 0
        local_model = None
        poly_local = None

        while ndata_model <= ndata and ndata_model <= max_local_n:
            md_tmp, rmse_tmp, lm_tmp, pl_tmp = set_model_degree(
                model_degree, sorted_indices, ndata_model, Xbar, ybar
            )
            if rmse_tmp < best_test_rmse:
                best_test_rmse = rmse_tmp
                local_model = lm_tmp
                model_degree = md_tmp
                poly_local = pl_tmp
                best_ndata_model = ndata_model
            else:
                break
            ndata_model += 2 * learning_size

        #record indices used this iteration
        idata_model = sorted_indices[:best_ndata_model]
        subsets_used.append(idata_model.copy())
        subset_indices_list.append(idata_model.copy())

        #record the surrogate for this iteration before stepping
        local_models.append(local_model)
        poly_locals.append(poly_local)

        if best_test_rmse > RMSE_THRESH:
            print("Stopping: local model not accurate enough (rmse =", best_test_rmse, ")")
            return -2, xis, local_models, poly_locals, subsets_used

        local_n = len(idata_model)

        #augment p = [x, z] with local slack variables
        A_tie = np.hstack((-np.eye(n), X[idata_model, :].T))  #shape (n, n+local_n)
        p_init = np.concatenate([xit, np.zeros(local_n)])

        lc_tie = LinearConstraint(A_tie, lb=np.zeros(n), ub=np.zeros(n))

        jac_fun = _jac_g_locmodel_p_factory(local_model, poly_local, n, local_n)
        nlc_locmodel = NonlinearConstraint(g_locmodel, -np.inf, 0.0, jac=jac_fun)

        #bounds for p = [x, z]
        lb_x, ub_x = bounds.lb, bounds.ub
        if lb_x.shape[0] != n or ub_x.shape[0] != n:
            raise ValueError("Bounds length must match n")
        lb_vec = np.concatenate([lb_x, np.zeros(local_n)])
        ub_vec = np.concatenate([ub_x, np.full(local_n, np.inf)])

        #known constraints to match p-dimension?
        aug_extras = _augment_linear_constraints_for_p(extra_constraints, n, local_n)

        #checks / summaries
        if check_bounds:
            _check_and_report_bounds(lb_vec, ub_vec, n, local_n, p_init=p_init, name="p")

        if talk:
            try:
                print(f"[subsolve: LP] starting; dimensions: x={n} z={local_n}")
                print(f"[subsolve: LP] initial f={float(f(p_init)):.6g} g_loc@x0={float(g_locmodel(p_init[:n])):.6g}")
            except Exception:
                print("[subsolve: LP] starting (initial evaluation failed to print)")

        #build tie equality: x - X_subset^T z = 0  ->  [-I | X^T] p = 0
        A_tie = np.hstack((-np.eye(n), X[idata_model, :].T))  #shape (n, n+local_n)
        b_tie = np.zeros(n)
        p_init = np.concatenate([xit, np.zeros(local_n)])  #for logging only


        #convert extra LinearConstraints (nutrition, equalities) to LP rows
        aug_extras = _augment_linear_constraints_for_p(extra_constraints, n, local_n)
        A_ub_e, b_ub_e, A_eq_e, b_eq_e = _convert_linear_constraints_to_lp(aug_extras, n, local_n)

        #local palatability: g_loc(x) <= 0  ->  (a^T x + a0) <= 0
        a0, a = _affine_g_coeffs(local_model, poly_local, n)
        A_g = np.zeros((1, n + local_n));
        A_g[0, :n] = a
        b_g = np.array([-a0], dtype=float)

        #stack all A_ub / A_eq
        A_ub_list = [A for A in [A_ub_e, A_g] if A is not None]
        b_ub_list = [b for b in [b_ub_e, b_g] if b is not None]
        A_eq_list = [A for A in [A_eq_e, A_tie] if A is not None]
        b_eq_list = [b for b in [b_eq_e, b_tie] if b is not None]

        A_ub = np.vstack(A_ub_list) if len(A_ub_list) else None
        b_ub = np.concatenate(b_ub_list) if len(b_ub_list) else None


        A_eq = np.vstack(A_eq_list) if len(A_eq_list) else None
        b_eq = np.concatenate(b_eq_list) if len(b_eq_list) else None

        #convex-hull row: sum(z) = 1
        if USE_CONVEX_HULL:
            p_dim = n + local_n  #(x, z) only in the current code
            sumz = np.zeros(p_dim, dtype=float)
            sumz[n:n + local_n] = 1.0  #only z's
            A_eq = sumz[None, :] if A_eq is None else np.vstack([A_eq, sumz])
            b_eq = np.array([1.0]) if b_eq is None else np.concatenate([b_eq, [1.0]])


        #per-variable bounds list
        bounds_list = [(float(lb_vec[i]), float(ub_vec[i]) if np.isfinite(ub_vec[i]) else None)
                       for i in range(n + local_n)]

        #linear objective c^T p (recover c from f)
        c = _objective_vector_from_f(f, n, local_n)

        if talk:
            print(f"[subsolve: LP] vars: x={n}, z={local_n}, "
                  f"A_ub rows={0 if A_ub is None else A_ub.shape[0]}, "
                  f"A_eq rows={0 if A_eq is None else A_eq.shape[0]}")
            try:
                print(f"[subsolve: LP] initial f={float(f(p_init)):.6g}, "
                      f"g_loc@x0={float(g_locmodel(p_init[:n])):.6g}")
            except Exception:
                pass
            print("extracons entering to the LP solver", aug_extras)

        res_loc = linprog(
            c=c,
            A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=bounds_list,
            method="highs",
        )

        #handle infeasible/failure early (res.x is None)
        if not res_loc.success or res_loc.x is None:
            print("Stopping: LP subproblem infeasible or failed. Message:", res_loc.message)
            return -2, xis, local_models, poly_locals, subsets_used

        #report if we have a solution vector
        if talk:
            _report_subproblem_result(
                res_loc, n, local_n, f=f, g=g, g_locmodel=g_locmodel,
                A_eq=A_tie, b_eq=np.zeros(n), aug_lin=aug_extras, tol=tol
            )

        x_trial = res_loc.x
        if np.linalg.norm(x_trial[:n] - xit) < 1e-6:
            print("No significant progress; terminating.")
            return 0, xis, local_models, poly_locals, subsets_used

        xit = x_trial[:n]
        xis.append(xit.copy())
        print("iteration ", i + 1, ", trial_x: ", xit, "f(x): ", f(xit),
              "g(x): ", g(xit), "g_locmodel(x) =", g_locmodel(np.array(xit)))

        #refresh neighborhood for next iter
        dists = np.linalg.norm(xit - X, axis=1)
        sorted_indices = np.argsort(dists).tolist()


    return -1, xis, local_models, poly_locals, subsets_used


#3D plots/animations
def _apply_fixed_axes(ax):
    ax.set_xlim3d(X2_MIN, X2_MAX)  #x-axis shows x2
    ax.set_ylim3d(X1_MIN, X1_MAX)  #y-axis shows x1
    ax.set_zlim3d(X0_MIN, X0_MAX)  #z-axis shows x0 (vertical)
    ax.set_xlabel('x₂'); ax.set_ylabel('x₁'); ax.set_zlabel('x₀')
    ax.view_init(elev=25, azim=45)


def plot_all_data_3d(X):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1,1,1))
    try:
        ax.set_proj_type('ortho')
    except Exception:
        pass

    ax.scatter(
        X[:,2], X[:,1], X[:,0],
        c=ALLDATA_COLOR, s=POINT_SIZE_NOT, alpha=0.8, depthshade=False,
        edgecolors=EDGE_COLOR, linewidths=EDGE_LW_NOT, label='data point'
    )
    _apply_fixed_axes(ax)
    ax.set_title("All Data")
    ax.legend(loc='upper right')
    plt.show()


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
        xi = xis[frame+1]

        used = np.zeros(N, dtype=bool)
        used[idx] = True

        ax.scatter(X[~used,2], X[~used,1], X[~used,0],
                   c=ALLDATA_COLOR, s=POINT_SIZE_NOT, alpha=0.6,
                   edgecolors=EDGE_COLOR, linewidths=EDGE_LW_NOT, depthshade=False,
                   label='not used')
        if len(idx):
            ax.scatter(X[used,2], X[used,1], X[used,0],
                       c=SUBSET_COLOR, s=POINT_SIZE_USED, marker='o',
                       edgecolors=EDGE_COLOR, linewidths=EDGE_LW_USED, depthshade=False,
                       label=f'local subset (size={len(idx)})')
        ax.scatter(xi[2], xi[1], xi[0],
                   c='C0', s=130, marker='*', edgecolors='k', linewidths=0.6,
                   depthshade=False, label='xit')

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
        c='C0', s=130, marker='*', edgecolors='k', linewidths=0.6, depthshade=False,
        label='xit'
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
        edgecolors=EDGE_COLOR, linewidths=EDGE_LW_USED, label=f'used (union) n={mask_used.sum()}'
    )
    _apply_fixed_axes(ax)
    ax.set_title("All points ever used vs not used")
    ax.legend(loc='upper right')
    plt.show()
