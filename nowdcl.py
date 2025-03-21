from scipy.optimize import (
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

global global_model
global local_model
global poly_global
global poly_local

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

def nowdcl(f, X, y, None, bounds, x_0, niters):
    MAX_NDATA = 100000
    RMSE_THRESH = 1e-2
    IMPRV_FACTOR = 0.2

    #initialize
    xit = np.array(x_0)
    ndata = len(y) #total number of data points
    learning_size= len(x_0) #size of the (constraint) function to be learned locally
    ndata_model_0 = np.minimum(MAX_NDATA, np.minimum(3*learning_size, ndata)) #number of data points to be included in model construction
    ndata_train_0 = math.floor(0.7*ndata_model)
    ndata_test_0 = ndata_model-ndata_train
    for i in range(niters):
        #model construction -- initialize
        model_degree = 2
        ndata_model = ndata_model_0
        ndata_train = ndata_train_0
        ndata_test = ndata_test_0
        weights = np.ones(ndata_model)
        dists = np.linalg.norm(xit - X, axis=1)
        update_weights(xit, dists)
        ybar = weights * y
        Xbar = weights.reshape(-1, 1) * X

        #choose the closest 'ndata_model' data points
        sorted_indices = np.argsort(dists)
        idata_model = sorted_indices[0:ndata_model]
        #now, determine the train and test data
        idata_train = random.sample(range(ndata_model),ndata_train) #indices of training data
        idata_test = [idata_model[i] for i notin idata_train]
        Xbar_train = Xbar[idata_train, :]
        ybar_train = ybar[idata_train]
        isDone=False
        while !isDone:
 	    #construct the model
            poly_local = PolynomialFeatures(degree=model_degree)
            X_poly_local = poly_local.fit_transform(Xbar_train)
            local_model = LinearRegression().fit(X_poly_local, ybar_train)
            #compute the test error and check termination
            #if test_rmse>imprv_factor*test_rmse_prev or test_rmse<rmse_thresh => stop
            #FGN : BURADA KALDIM
        
	#solve the local model
        nlc_locmodel = NonlinearConstraint(g_locmodel, -np.inf, 0)
        local_bounds = Bounds([xit[0]-delta,max(xit[1]-delta,-5),max(xit[2]-delta,-5)], [xit[0]+delta, min(xit[2]+delta,5), min(xit[2]+delta,5)])
        res_loc = minimize(f, xit, bounds=local_bounds, constraints=[nlc_locmodel], method="trust-constr",options={'disp':False})
        x_trial = res_loc.x
        print("iteration ",i,", iterate : ",xit, "f: ",f(xit), "g: ", g(xit), "delta:", delta)
        print("g(xit)= ", g(xit), "g_locmodel(xit) =", g_locmodel(np.array(xit)),"g_locmodel[0,1,1] =", g_locmodel(np.array([0.0,1.0,1.0])))
        xit = x_trial
        #lam = lam * 0.3
    #print("Solution: ", xit)



#MAIN PROGRAM

bounds = Bounds([-np.inf,-5,-5], [np.inf, 5, 5])
# Exact optimal solution should be [0, 1, 1]
x_0 = [-1.0, -2.0, 1.0]

# Generate data from the nonlinear constraint
np.random.seed(102)
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
plt.scatter(x_init[1],x_init[2])
plt.scatter(x_ast_model[1],x_ast_model[2])
plt.scatter(x_ast[1],x_ast[2])
#plt.show()

#Solve with the true constraint function g(x)
nlc = NonlinearConstraint(g, -np.inf, 0) #g(x)<=0
res = minimize(f, x_init, bounds=bounds, constraints=[nlc], method="trust-constr")
#res = minimize(f, x_init, bounds=bounds, constraints=[nlc], method="slsqp")
print('True solution: ',res.x)
x_ast = res.x
print("f: ",f(x_ast), "g: ", g(x_ast))

#Fit a fourth order polynomial and solve globally
nlc_gmodel = NonlinearConstraint(g_globmodel, -np.inf, 0)
poly_global = PolynomialFeatures(degree=4)
X_poly = poly_global.fit_transform(X)
#poly.fit(X_poly, y)
global_model = LinearRegression().fit(X_poly, y)
res_g = minimize(f, x_init, bounds=bounds, constraints=[nlc_gmodel], method="trust-constr")
#res_g = minimize(f, x_init, bounds=bounds, constraints=[nlc_gmodel], method="slsqp")
print('Solution using the global model with ',nsamples, ' samples: ',res_g.x)
x_ast_model = res_g.x

#Solve via local modelling
niters = 15  # number of iterations
nowdcl(f, X, y, None, bounds, x_0, niters)



