# -*- coding: utf-8 -*-

import warnings
import random
import numpy as np

from scipy.optimize import Bounds, NonlinearConstraint, minimize

import dcl_core as dcl
import dcl_problem as dclp


SEED = 102


def _set_seeds(options, seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    options.seed = seed


def main():
    warnings.filterwarnings(
        "ignore",
        message="delta_grad == 0.0. Check if the approximated function is linear. If the function is linear better results can be obtained by defining the Hessian as zero instead of using quasi-Newton approximations.",
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
        options.init_data_ratio = 0.30
        options.train_data_ratio = 0.70
        options.rmse_thresh = 0.9
        options.rmse_target = 0.1
        options.delta_min = 1e-6

    _set_seeds(options, SEED)

    # =========================
    # Reference solve for Rosenbrock
    # =========================
    if problem_name == "rosenbrock":
        bounds = Bounds(problem["bounds_lb"], problem["bounds_ub"])
        nlc = NonlinearConstraint(dclp.rosenbrock_constraint_value, -np.inf, 0.0)

        def f_true(x):
            return float(x[0])

        res_true = minimize(
            f_true,
            x_init,
            bounds=bounds,
            constraints=[nlc],
            method="trust-constr",
            options={"disp": False},
        )

        print("True solution:", res_true.x)
        print("True objective:", res_true.fun)
        print("True g(x):", dclp.rosenbrock_constraint_value(res_true.x))
        print("-" * 70)

    # =========================
    # Solve with DCL
    # =========================
    dcls = dcl.dcl_solver(problem, problem_model_type, options, X, y, x_init)
    dcls.solve()

    print("final x =", dcls.xit)
    print("final objective =", dcls.final_obj)
    print("return code =", dcls.return_code)
    dcl.dcl_print_iteration_log(dcls)


if __name__ == "__main__":
    main()