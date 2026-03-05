# -*- coding: utf-8 -*-
#This file contains the basic procedures of dynamic constraint learning
import numpy as np
import random
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
#from dataclasses import dataclass
import gurobipy as gp
from gurobipy import GRB

from gurobi_ml import add_predictor_constr

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ENABLE_WEIGHTING = 0

#options class
class dcl_options:
    trust_region = "convex_hull" #TR_MODE
    learning_model_type = "polynom_regression"   #or "cart" etc.
    train_data_ratio = 0.7
    init_data_ratio = 0.1 #BURADA KALDIK!!
    max_iterations = 100
    x_init = None
    delta_min = 1e-6
    rmse_thresh = 0.1
    max_model_degree = 4
    seed = 0
    max_ndata = 1000000
    
    #new
    # NN-specific
    nn_architectures = [
        (8,),
        (16,),
        (16, 16),
        (32, 16),
    ]
    nn_solver = "lbfgs"
    nn_alpha = 1e-4
    nn_max_iter = 3000
    nn_tol = 1e-6
    
    #subproblem_solver??

    #def check_options(self): 
        #TODO 

#learned model class
class dcl_local_model:
    def __init__(self):
        self.data_indices = None
        self.learned_model = None  # scikit learn object
        self.test_rmse = None
        self.model_degree = 1


#solver class
class dcl_solver:
    def __init__(self,problem_,problem_model_type_,options_,X_,y_,x_init_):
        self.local_models = [] #holds information on local models, new: initin icine tasindi
        self.problem_model = problem_
        self.problem_model_type = problem_model_type_
        self.options = options_
        self.X = X_
        self.y = y_
        self.x_init = x_init_
        self.return_code = 0
        self.xit = None
        
    def solve(self):
        #main loop
        X = self.X
        y = self.y
        options = self.options

        
        #new: need to work with numpy for distances
        Xnp = X.to_numpy(dtype=float) if hasattr(X, "to_numpy") else np.asarray(X, dtype=float)
      
        #get problem size
        n = int(Xnp.shape[1])    
        
        #new: initialize xit if needed
        if self.x_init is None:
            self.x_init = dcl_init(Xnp)   # returns length n
        xit = np.asarray(self.x_init, dtype=float)
        print("x_init",xit)

        #xit =self.x_init
        dists = np.linalg.norm(xit - X, axis=1)
        sorted_indices = np.argsort(dists).tolist()
        
        ndata = len(y)
        learning_size  = 2 * len(self.x_init)
        #FGN: TODO: Burada options.init_data_ratio kullanalım    
        #ndata_model_0  = np.minimum(options.max_ndata, np.minimum(5 * learning_size, ndata))
        #new
        ndata = len(y)
        min_local_size = max(2 * n, 4)
        ndata_model_0 = max(min_local_size, math.ceil(options.init_data_ratio * ndata))
        ndata_model_0 = min(options.max_ndata, ndata_model_0, ndata)

        for i in range(self.options.max_iterations):
            model_degree = 1
            ndata_model  = ndata_model_0
            #We are not currently using weighting
            if(ENABLE_WEIGHTING):
                weights = update_weights(xit, dists)
                ybar = weights * y
                Xbar = weights.reshape(-1, 1) * X
            else:
                Xbar = X
                ybar = y
    
            best_test_rmse   = np.inf
            best_ndata_model = 0
            local_model = None
            #poly_local  = None
    
            while ndata_model <= ndata:
                md_tmp, rmse_tmp, lm_tmp, p_tmp = dcl_set_model_degree(
                self.options.learning_model_type,  #new
                model_degree,
                sorted_indices,
                ndata_model,
                Xbar,
                ybar,
                self.options
                )
                if rmse_tmp < best_test_rmse:
                    best_test_rmse   = rmse_tmp
                    local_model      = lm_tmp
                    model_degree     = md_tmp
                    best_ndata_model = ndata_model
                else:
                    break
                ndata_model += 2 * learning_size
    
            idata_model = sorted_indices[:best_ndata_model]
            accepted_model = dcl_local_model()
            accepted_model.data_indices = idata_model
            accepted_model.learned_model = local_model
            accepted_model.test_rmse = best_test_rmse
            accepted_model.model_degree = model_degree
    
            self.local_models.append(accepted_model)
       
            if best_test_rmse > self.options.rmse_thresh:
                print("Stopping: local model not accurate enough (rmse =", best_test_rmse, ")")
                self.return_code = -2
                self.xit = xit
                return 
    
    #
            x_trial, res_loc = dcl_solve_subproblem(
                self.problem_model,
                accepted_model,
                self.options.trust_region,
                self.X,
                )

            if x_trial is None:
                print(f"Stopping: subproblem not optimal (status={res_loc.get('status')})")
                self.return_code = -3
                self.xit = xit
                return

            # ---- progress + update neighborhood ----
            step = np.linalg.norm(x_trial[:n] - xit)
            print(f"iter {i}: step = {step:.3e}, obj = {res_loc.get('obj')}")

            if step < options.delta_min:
                print("No significant progress; terminating.")
                self.return_code = -1
                self.xit = xit
                return

            # update iterate (inputs only)
            xit = x_trial[:n]

            # recompute neighborhood around new xit
            # (make sure Xnp exists; see note below)
            dists = np.linalg.norm(Xnp - xit, axis=1)
            sorted_indices = np.argsort(dists).tolist()
            

#Initialize
#def dcl_init(X, size):
    #choose initial_point based on X
    #TODO: Add a procedure for initial point computation

def dcl_init(X, size=None):
    """
    Initial point = arithmetic average (centroid) of samples in X.
    Works with numpy arrays and pandas DataFrames.

    X: shape (n_samples, n_features)
    returns: shape (n_features,)
    """
    # pandas DataFrame -> numpy
    if hasattr(X, "to_numpy"):
        Xnp = X.to_numpy(dtype=float)
    else:
        Xnp = np.asarray(X, dtype=float)

    if Xnp.size == 0:
        raise ValueError("X is empty, cannot compute arithmetic average.")

    if Xnp.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features). Got shape {Xnp.shape}")

    x0 = Xnp.mean(axis=0)  # arithmetic average of the set (column-wise mean)

    # Optional consistency check
    if size is not None and len(x0) != size:
        # only warn if size is inconsistent (often happens if size = total vars incl. output)
        print(f"Warning: dcl_init returned length {len(x0)} but size={size}")

    return x0

#Construct and solve subproblem
#def dcl_solve_subproblem(problem_model, accepted_model, trust_region):
    #TODO
    #add accepted_model.learned_model (scikit learn objesi) to problem_model as a constraint via gurobiml library, 
    
    #also add trust_region of desired type to problem_model (eğer convex hull ise önceki kullandığımız doğrusal kısıt)
    #learned_constraint y = g(z)
    #input_vars = x  #Mvar for z in the gurobbi model problem_model, getVarByName(name)
    #output_vars = y  #Mvar for y in the gurobbi model problem_model 
    #problem_model_local = problem_model
    #learned_constraint = add_predictor_constr( = problem_model, accepted_model.learned_model, input_vars, output_vars)
    #return [input_vars.X, output_vars.X]  x = [z;y]



def dcl_solve_subproblem(problem_model, accepted_model, trust_region, X):
    m = problem_model.copy()

    # X -> numpy
    if hasattr(X, "to_numpy"):
        Xnp = X.to_numpy(dtype=float)
    else:
        Xnp = np.asarray(X, dtype=float)

    # number of input vars (from original model's MVar)
    n = int(Xnp.shape[1])  

    # get copied vars by explicit names (these names come from addMVar(..., name="input_vars"))
    input_vars = [m.getVarByName(f"input_vars[{i}]") for i in range(n)]
    if any(v is None for v in input_vars):
        raise KeyError("Some 'input_vars[i]' were not found in copied model.")

    out0 = m.getVarByName("output_vars[0]")
    if out0 is None:
        raise KeyError("Could not find 'output_vars[0]' in copied model.")

    # new
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

    m.optimize()
    if m.Status != GRB.OPTIMAL:
        return None, {"status": int(m.Status)}

    z_sol = np.array([v.X for v in input_vars], dtype=float)
    y_sol = float(out0.X)
    x_trial = np.concatenate([z_sol, np.array([y_sol])])

    return x_trial, {"status": int(m.Status), "obj": float(m.ObjVal), "output": y_sol}


def dcl_set_model_degree(learning_model_type, degree_init, sorted_indices, ndata_model, Xbar, ybar, options):
    RMSE_THRESH = options.rmse_thresh

    is_pandas_X = hasattr(Xbar, "iloc")
    is_pandas_y = hasattr(ybar, "iloc")

    model_degree = degree_init

    ndata_train = math.floor(options.train_data_ratio * ndata_model)
    ndata_test  = ndata_model - ndata_train

    idata_model = sorted_indices[0:ndata_model]
    idata_train = random.sample(idata_model, k=ndata_train)
    idata_test  = np.setdiff1d(idata_model, idata_train).tolist()

    if is_pandas_X:
        Xbar_train = Xbar.iloc[idata_train, :]
        Xbar_test  = Xbar.iloc[idata_test, :]
    else:
        Xbar_train = Xbar[idata_train, :]
        Xbar_test  = Xbar[idata_test, :]

    if is_pandas_y:
        ybar_train = ybar.iloc[idata_train]
        ybar_test  = ybar.iloc[idata_test]
    else:
        ybar_train = ybar[idata_train]
        ybar_test  = ybar[idata_test]

    Xbar_train = np.asarray(Xbar_train, dtype=float)
    Xbar_test  = np.asarray(Xbar_test, dtype=float)
    ybar_train = np.asarray(ybar_train, dtype=float).ravel()
    ybar_test  = np.asarray(ybar_test, dtype=float).ravel()

    test_rmse_prev = np.inf
    model_degree_prev = None
    loc_model_prev = None
    p_local_prev = None

    while True:
        p_local = None

        if learning_model_type == "cart":
            loc_model = DecisionTreeRegressor(
                max_depth=model_degree,
                random_state=options.seed
            ).fit(Xbar_train, ybar_train)

            ypred_test = loc_model.predict(Xbar_test)

        elif learning_model_type == "polynom_regression":
            p_local = PolynomialFeatures(degree=model_degree, include_bias=True)
            X_poly_train = p_local.fit_transform(Xbar_train)
            loc_model = LinearRegression().fit(X_poly_train, ybar_train)

            X_poly_test = p_local.transform(Xbar_test)
            ypred_test  = loc_model.predict(X_poly_test)

        elif learning_model_type == "nn":
            arch_idx = min(model_degree - 1, len(options.nn_architectures) - 1)
            arch = options.nn_architectures[arch_idx]

            loc_model = Pipeline([
                ("scaler", StandardScaler()),
                ("nn", MLPRegressor(
                    hidden_layer_sizes=arch,
                    activation="relu",
                    solver=options.nn_solver,
                    alpha=options.nn_alpha,
                    max_iter=options.nn_max_iter,
                    tol=options.nn_tol,
                    random_state=options.seed
                ))
            ])

            loc_model.fit(Xbar_train, ybar_train)
            ypred_test = loc_model.predict(Xbar_test)

        else:
            raise NotImplementedError(
                f"Unknown learning_model_type='{learning_model_type}'."
            )

        denom = np.maximum(np.maximum(np.abs(ypred_test), np.abs(ybar_test)), 1e-12)
        test_rmse = np.linalg.norm((ypred_test - ybar_test) / denom) / max(len(ybar_test), 1)

        print("ndata_model : ", ndata_model,
              "ndata_test : ",  ndata_test,
              "ndata_train : ", ndata_train,
              "model_degree : ", model_degree,
              "test_rmse : ",    test_rmse)

        if test_rmse > test_rmse_prev:
            if model_degree_prev is None:
                return model_degree, test_rmse, loc_model, p_local
            return model_degree_prev, test_rmse_prev, loc_model_prev, p_local_prev

        if learning_model_type == "nn":
            reached_max = model_degree >= min(options.max_model_degree, len(options.nn_architectures))
        else:
            reached_max = model_degree == options.max_model_degree

        if test_rmse < RMSE_THRESH or reached_max:
            return model_degree, test_rmse, loc_model, p_local

        model_degree_prev = model_degree
        test_rmse_prev    = test_rmse
        loc_model_prev    = loc_model
        p_local_prev      = p_local
        model_degree     += 1

def update_weights(xit, dists):
    lam = 0.0
    return np.exp(-lam * dists)