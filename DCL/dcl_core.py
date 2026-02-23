# -*- coding: utf-8 -*-
#This file contains the basic procedures of dynamic constraint learning
import numpy as np

import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass
import gurobipy as gp
from gurobipy import GRB

ENABLE_WEIGHTING = 0

#options class
class dcl_options:
    trust_region = "convex_hull" #TR_MODE
    learning_model_type = "polynom_regression"   #or "cart" etc.
    test_data_ratio = 0.2
    max_iterations = 100
    x_init = None
    delta_min = 1e-6
    rmse_thresh = 0.1
    max_model_degree = 4
    tree_seed = 0 #?
    max_ndata = 2000
    
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
    local_models = [] #holds information on local models
    def __init__(self,problem_,problem_model_type_,options_,X_,y_,x_init_):
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
        problem_model = self.problem_model
        #get problem size
        if(self.problem_model_type=="gurobi"):
            n = problem_model.NumVars #CHECK
        #else...    
        if(self.x_init == None):
            self.x_init = dcl_init(self.X, n) #TODO
            
        xit =self.x_init
        dists = np.linalg.norm(xit - X, axis=1)
        sorted_indices = np.argsort(dists).tolist()
        
        ndata = len(y)
        learning_size  = 2 * len(self.x_init)
        ndata_model_0  = np.minimum(options.max_ndata, np.minimum(5 * learning_size, ndata))
    
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
                md_tmp, rmse_tmp, lm_tmp = dcl_set_model_degree(
                    model_degree, sorted_indices, ndata_model, Xbar, ybar, self.options
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
    
            x_trial, res_loc = dcl_solve_subproblem(self.problem_model, accepted_model, self.options.trust_region) #TODO
            
            if np.linalg.norm(x_trial[:n] - xit) < options.delta_min:
                print("No significant progress; terminating.")
                self.return_code = -1
                self.xit = xit
                return 
    
            #xit = x_trial[:n]


#Initialize
#def dcl_init(X, size):
    #choose initial_point based on X
    #TODO: Add a procedure for initial point computation


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

    from gurobi_ml import add_predictor_constr

    m = problem_model.copy()

    # ---- infer n from input_vars[i] names ----
    n = 0
    while True:
        v = m.getVarByName(f"input_vars[{n}]")
        if v is None:
            break
        n += 1
    if n == 0:
        raise KeyError(
            "No variables found named 'input_vars[0]'. "
            "Make sure your base model uses addMVar(..., name='input_vars')."
        )

    # ---- get input_vars and output_vars by name ----
    input_vars = [m.getVarByName(f"input_vars[{i}]") for i in range(n)]
    if any(v is None for v in input_vars):
        raise KeyError("Some 'input_vars[i]' were not found in the model copy.")

    out0 = m.getVarByName("output_vars[0]")
    if out0 is None:
        raise KeyError(
            "Could not find 'output_vars[0]'. "
            "Make sure your base model uses addMVar(shape=1, name='output_vars')."
        )

    # ---- X to numpy ----
    if hasattr(X, "to_numpy"):
        Xnp = X.to_numpy(dtype=float)
    else:
        Xnp = np.asarray(X, dtype=float)

    # ---- learned constraint: output_vars = g(input_vars) ----
    #m.Params.NonConvex = 2  # safe for polynomial/pipeline models
    add_predictor_constr(
        m,
        accepted_model.learned_model,
        input_vars=np.array(input_vars, dtype=object),
        output_vars=np.array([out0], dtype=object),
    )

    # ---- trust region ----
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
        raise NotImplementedError(f"Unknown trust_region='{trust_region}'. Use 'convex_hull' or 'none'.")

    # ---- solve ----
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        return None, {"status": int(m.Status)}

    # ---- return [input_vars; output_vars] ----
    z_sol = np.array([v.X for v in input_vars], dtype=float)
    y_sol = float(out0.X)
    x_trial = np.concatenate([z_sol, np.array([y_sol])])

    res = {"status": int(m.Status), "obj": float(m.ObjVal), "output": y_sol}
    return x_trial, res


def dcl_set_model_degree(learning_model_type, degree_init, sorted_indices, ndata_model, Xbar, ybar, options):
    RMSE_THRESH = options.rmse_thresh
    MAX_MODEL_DEGREE = options.max_model_degree
    
    model_degree = degree_init

    ndata_train = math.floor(0.7 * ndata_model)
    ndata_test  = ndata_model - ndata_train

    idata_model = sorted_indices[0:ndata_model]
    idata_train = np.random.sample(idata_model, k=ndata_train)
    idata_test  = np.setdiff1d(idata_model, idata_train).tolist()

    Xbar_train = Xbar[idata_train, :]
    ybar_train = ybar[idata_train]
    Xbar_test  = Xbar[idata_test, :]
    ybar_test  = ybar[idata_test]

    isDone = False
    test_rmse_prev = np.inf

    while not isDone:
        if(learning_model_type == "cart"):
            tree = DecisionTreeRegressor(max_depth=model_degree, random_state=tree_seed)
            loc_model = tree.fit(Xbar_train, ybar_train)
            #ypred_test = ...

        elif(learning_model_type == "polynom_regression"):
        	    p_local = PolynomialFeatures(degree=model_degree)
        	    X_poly_local = p_local.fit_transform(Xbar_train)
        	    loc_model = LinearRegression().fit(X_poly_local, ybar_train)

        X_poly_test = p_local.fit_transform(Xbar_test)
        ypred_test  = loc_model.predict(X_poly_test)
       
        #else:

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

        #model_degree_prev = model_degree
        test_rmse_prev    = test_rmse
        #loc_model_prev    = loc_model
        #p_local_prev      = p_local
        model_degree     += 1


def update_weights(xit, dists):
    lam = 0.0
    return np.exp(-lam * dists)