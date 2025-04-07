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
import warnings
import math
import random

global global_model
global local_model
global poly_global
global poly_local
global n

warnings.filterwarnings("ignore",message="delta_grad == 0.0. Check if the approximated function is linear. If the function is linear better results can be obtained by defining the Hessian as zero instead of using quasi-Newton approximations.")
warnings.filterwarnings("ignore",message="Singular Jacobian matrix. Using SVD decomposition to perform the factorizations.")


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
#Objective function definition for augmented Rosenbrock problem
def f(x):
    return x[0]
#Constraint function definition for controlled experiments
#  To generate constraint data for augmented Rosenbrock problem
def g(x): 
    return -x[0] + 100*(x[2] - x[1]*x[1])** 2 + (1-x[1]) ** 2 

'''
Model evaluation routines
'''
#Evaluate global model of g at a given point 
def g_globmodel(x):
    x = x.reshape(1,-1)
    x_poly = poly_global.fit_transform(x)
    return global_model.predict(x_poly)
#Evaluate local model of g at a given point
def g_locmodel(x):
    if(x.size>n):
        x = x[0:n]
    x = x.reshape(1,-1)
    x_poly_local = poly_local.fit_transform(x)
    return local_model.predict(x_poly_local)

def update_weights(xit, dists):
    lam = 1.0  # lambda
    weights = np.exp(-lam * dists)
    #for j in range(len(weights)):
    #    if(dists[j]<delta): 
    #        weights[j]=1
    #    else: 
    #        weights[j]=0
    #weights = weights / max(weights)    
    return weights

#Find the most suitable model degree given a local sample size
def set_model_degree(degree_init, sorted_indices, ndata_model, Xbar, ybar) :
    RMSE_THRESH = 1e-1
    MAX_MODEL_DEGREE = 4
    IMPRV_FACTOR = 0.2
    model_degree = degree_init
    ndata_train = math.floor(0.7*ndata_model)
    ndata_test = ndata_model-ndata_train
    idata_model = sorted_indices[0:ndata_model]
    #now, determine the train and test data
    idata_train = random.sample(range(ndata_model),k=ndata_train) #indices of training data
    idata_test = np.setdiff1d(idata_model,idata_train).tolist()
    Xbar_train = Xbar[idata_train, :]
    ybar_train = ybar[idata_train]
    Xbar_test = Xbar[idata_test, :]
    ybar_test = ybar[idata_test]
    isDone=False
    test_rmse_prev = np.inf
    while not isDone:
        #construct the model
        p_local = PolynomialFeatures(degree=model_degree)
        X_poly_local = p_local.fit_transform(Xbar_train)
        loc_model = LinearRegression().fit(X_poly_local, ybar_train)
        #compute the test error and check termination
        X_poly_test = p_local.fit_transform(Xbar_test)
        ypred_test = loc_model.predict(X_poly_test) 
        test_rmse = np.linalg.norm(np.divide((ypred_test-ybar_test), np.maximum(abs(ypred_test),abs(ybar_test))))
        test_rmse = (1.0/ndata_test)*test_rmse
        print("ndata_model : ", ndata_model, "ndata_test : ", ndata_test, "ndata_train : ", ndata_train, "model_degree : ", model_degree, "test_rmse : ", test_rmse)
        if test_rmse>test_rmse_prev:
            isDone=True
            return model_degree_prev, test_rmse_prev, loc_model_prev, p_local_prev
        if test_rmse<RMSE_THRESH or model_degree==MAX_MODEL_DEGREE: 
        #this criteria should rather be relative, the last one is a question
            isDone=True
            return model_degree, test_rmse, loc_model, p_local
        else :
            model_degree_prev = model_degree
            model_degree = model_degree+1
        test_rmse_prev = test_rmse
        loc_model_prev = loc_model
        p_local_prev = p_local


'''
Function nowdcl
Input:
f       objective function
(X, y)  data aavailable to learn g
c       additional constraints
bl, bu  bounds
x_0     initial point
'''
#TODO : Create a 'DCLproblem' class

def nowdcl(f, X, y, bounds, x_0, niters):
    global local_model
    global poly_local
    global n

    MAX_NDATA = 100000
    RMSE_THRESH = 9*1e-1
    DELTA_MIN = 1e-6

    #initialize
    delta = 1.0
    #penalty = 1000
    xit = np.array(x_0)
    dists = np.linalg.norm(xit - X, axis=1)
    sorted_indices = np.argsort(dists)
    sorted_indices = sorted_indices.tolist()
    '''
    closest_data = sorted_indices[0]
    closest_x = X[closest_data]
    closest_gx = y[closest_data]
    closest_fx = f(closest_x)            
    merit_xit = closest_fx + penalty*np.maximum(closest_gx,0)
    '''
    ndata = len(y) #total number of data points
    learning_size= len(x_0) #size of the (constraint) function to be learned locally
    ndata_model_0 = np.minimum(MAX_NDATA, np.minimum(3*learning_size, ndata)) #number of data points to be included in model construction
    for i in range(niters):
        #model construction -- initialize
        model_degree = 1
        ndata_model = ndata_model_0
        weights = update_weights(xit, dists)
        ybar = weights * y
        Xbar = weights.reshape(-1, 1) * X

        #choose the closest 'ndata_model' data points
        best_test_rmse = np.inf
        best_ndata_model = 0
        local_model = None
        poly_local = None
        while ndata_model <= ndata :
            model_degree_tmp, test_rmse_tmp, local_model_tmp, poly_local_tmp = set_model_degree(model_degree, sorted_indices, ndata_model, Xbar, ybar)
            if(test_rmse_tmp < best_test_rmse):
                best_test_rmse = test_rmse_tmp
                local_model = local_model_tmp
                model_degree = model_degree_tmp
                poly_local = poly_local_tmp
                best_ndata_model = ndata_model
                print("model_degree:",model_degree, "ndata_model:",ndata_model, "best_test_rmse:", best_test_rmse) 	
            else:
                break
            #try increasing ndata_model 
            ndata_model+=learning_size
        print("model_degree:",model_degree, "ndata_model:",best_ndata_model, "best_test_rmse:", best_test_rmse)
        ndata_model = best_ndata_model 	
        idata_model = sorted_indices[0:ndata_model]

        #stop if the best model that can be constructed at this point is not sufficiently good
        if best_test_rmse > RMSE_THRESH:
            return -2
         
        #set delta as the distance to the farthest point; to be replaced by a convex hull constraint
        #sorted_dists = np.sort(dists)
        #delta = np.maximum(sorted_dists[range(ndata_model)]) 

	#solve the local model
        '''
        next_iterate_found = False
        nlc_locmodel = NonlinearConstraint(g_locmodel, -np.inf, 0)
        while not next_iterate_found and delta>DELTA_MIN:
            local_bounds = Bounds([xit[0]-delta,max(xit[1]-delta,-5),max(xit[2]-delta,-5)], [xit[0]+delta, min(xit[2]+delta,5), min(xit[2]+delta,5)])
            res_loc = minimize(f, xit, bounds=local_bounds, constraints=[nlc_locmodel], method="trust-constr",options={'disp':False})
            x_trial = res_loc.x
            print("iteration ",i,", trial_x: ",xit, "f(x): ",f(xit), "g(x): ", g(xit), "g_locmodel(x) =", g_locmodel(np.array(xit)), "delta:", delta)
            #check acceptance / update delta
            dists_tmp = np.linalg.norm(x_trial - X, axis=1)
            sorted_indices_tmp = np.argsort(dists_tmp)
            closest_data = sorted_indices_tmp[0]
            closest_x = X[closest_data]
            closest_gx = y[closest_data]
            closest_fx = f(closest_x)            
            merit_trial = closest_fx + penalty*np.maximum(closest_gx,0)
            model_trial = closest_fx + penalty*np.maximum(g_locmodel(np.array(closest_x)),0)
            ratio = (merit_xit - merit_trial)/(merit_xit - model_trial)
            print("closest_x: ",closest_x, "f_closest: ",closest_fx, "g_closest: ", closest_gx, "g_locmodel(closest_x)", g_locmodel(np.array(closest_x)), "ratio :", ratio)
            if ratio>0.0 :
                next_iterate_found = True
                xit = x_trial
                dists = dists_tmp
                sorted_indices = sorted_indices_tmp
                merit_xit = merit_trial
            if ratio > 0.2 :
                delta = delta*2
            else if ratio <= 0.0 :
                delta = delta/2
                #lam = lam * 0.3
        '''
        #the solution should be in the convex hull of the data points used in model construction
        #let z be the weights of data points so that x = z^T*X[idata_model,:] => [-I X[idata_model,:]'][x;z] = 0 , 0<=z<=1.
        A = np.hstack((-1*np.eye(n),X[idata_model,:].T))
        p_init = np.vstack((xit.reshape([n,1]),np.zeros([ndata_model,1])))
        p_init = p_init.reshape(-1)
        print(A)
        lc = LinearConstraint(A,lb=[0]*(n) ,ub=[0]*(n))
        nlc_locmodel = NonlinearConstraint(g_locmodel, -np.inf, 0)
        lb = np.vstack((np.array([-np.inf,-5,-5]).reshape([3,1]),np.zeros([ndata_model,1])))
        ub = np.vstack((np.array([np.inf,5,5]).reshape([3,1]),np.ones([ndata_model,1])))
        lb = lb.reshape(-1)
        ub = ub.reshape(-1)
        res_loc = minimize(f, p_init, bounds=Bounds(lb, ub), constraints=[nlc_locmodel, lc], method="trust-constr",options={'disp':False})
        x_trial = res_loc.x
        print("iteration ",i,", trial_x: ",xit, "f(x): ",f(xit), "g(x): ", g(xit), "g_locmodel(x) =", g_locmodel(np.array(xit)))
        if(np.linalg.norm(x_trial-xit)<DELTA_MIN):
            return 0 
    #end while
    return -1


#MAIN PROGRAM

# Exact optimal solution should be [0, 1, 1]
x_0 = [-1.0, -2.0, 1.0]
n = 3

# Generate data from the nonlinear constraint
np.random.seed(102)
random.seed(102)
print("Random seed fixed!")
nsamples = 100  # number of samples
X = 0.0 + 3 * np.random.randn(nsamples, 3)
#noise = 0.0001 * np.random.randn(nsamples)
noise = 0.0 * np.random.randn(nsamples)
y = np.array([g(X[i, :]) for i in range(nsamples)]) + noise

#Plot contours
delta_plot=0.25
x1_plot = np.arange(-5.0,5.0,delta_plot) 
x2_plot = np.arange(-7.0,5.0,delta_plot)
X1plot,X2plot=np.meshgrid(x1_plot,x2_plot)
#100*(x[2] - x[1]*x[1])** 2 + (1-x[1]) ** 2
Z = 100*(X2plot-X1plot**2)**2+(1-X1plot)**2
fig, ax = plt.subplots()
cs = ax.contourf(X1plot,X2plot,Z)

#Plot data 
plt.scatter(X[:,1],X[:,2])
plt.scatter(x_0[1],x_0[2])
#plt.scatter(x_ast_model[1],x_ast_model[2])
#plt.scatter(x_ast[1],x_ast[2])
#plt.show()

#Solve with the true constraint function g(x)
bounds = Bounds([-np.inf,-5,-5], [np.inf, 5, 5])
nlc = NonlinearConstraint(g, -np.inf, 0) #g(x)<=0
res = minimize(f, x_0, bounds=bounds, constraints=[nlc], method="trust-constr")
#res = minimize(f, x_0, bounds=bounds, constraints=[nlc], method="slsqp")
print('True solution: ',res.x)
x_ast = res.x
print("f: ",f(x_ast), "g: ", g(x_ast))

#Fit a fourth order polynomial and solve globally
nlc_gmodel = NonlinearConstraint(g_globmodel, -np.inf, 0)
poly_global = PolynomialFeatures(degree=4)
X_poly = poly_global.fit_transform(X)
#poly.fit(X_poly, y)
global_model = LinearRegression().fit(X_poly, y)
res_g = minimize(f, x_0, bounds=bounds, constraints=[nlc_gmodel], method="trust-constr")
#res_g = minimize(f, x_0, bounds=bounds, constraints=[nlc_gmodel], method="slsqp")
print('Solution using the global model with ',nsamples, ' samples: ',res_g.x)
x_ast_model = res_g.x

#Solve via local modelling
niters = 15  # number of iterations
exitcode = nowdcl(f, X, y, bounds, x_0, niters)



