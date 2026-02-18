# -*- coding: utf-8 -*-
#This file contains the basic procedures of dynamic constraint learning
import numpy as np

import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass

#options class
class dcl_options:
    trust_region = "convex_hull"
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
    data_indices = None
    learned_model = None  #scikit learn object
    test_rmse = None
    model_degree = 1


#solver class
class dcl_solver:
    local_models = [] #holds information on local models
    def __init__(self,problem_,problem_model_type_,options_,X_,y_,x_init_):
        problem_model = problem_
        problem_model_type = problem_model_type_
        options = options_
        X = X_
        y = y_
        x_init = x_init_

    def solve(self):
        if(self.x_init == None):
            x_init = dcl_init(self.X)
        #main loop
        X = self.X
        y = self.y
        xit = x_init
        dists = np.linalg.norm(xit - X, axis=1)
        sorted_indices = np.argsort(dists).tolist()
    
        ndata = len(y)
        learning_size  = 2 * len(x_init)
        ndata_model_0  = np.minimum(max_ndata, np.minimum(5 * learning_size, ndata))
    
        for i in range(self.options.max_iterations):
            model_degree = 1
            ndata_model  = ndata_model_0
            weights = update_weights(xit, dists)
            ybar = weights * y
            Xbar = weights.reshape(-1, 1) * X
    
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
                return -2, xis, local_models, poly_locals, subsets_used
    
            #x_trial, res_loc = dcl_solve_subproblem(self.problem_model, learned_model, self.options.trust_region) #TODO
            
            x_trial = xit.copy()  # placeholder so code runs?
            
            if np.linalg.norm(x_trial[:n] - xit) < options.delta_min:
                print("No significant progress; terminating.")
                return 0, xis, local_models, poly_locals, subsets_used
    
            xit = x_trial[:n]


#Initialize
#def dcl_init(X):
    #choose initial_point based on X
    #TODO: Add a procedure for initial point computation


#Construct and solve subproblem
#def dcl_solve_subproblem():
    #TODO


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


