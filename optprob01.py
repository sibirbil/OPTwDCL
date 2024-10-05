from scipy.optimize import (
    NonlinearConstraint,
    Bounds,
    minimize,
    linprog,
)
from sklearn.linear_model import LinearRegression
from locreg import LocalRegression
import numpy as np


def f(x):
    return x[0] + x[1]


def g(x):
    return 2 - x[0] ** 2 - x[1] ** 2


bounds = Bounds([-10, 0], [10, 10])
nlc = NonlinearConstraint(g, 0, np.inf)


# Exact optimal solution is [-sqrt(2), 0]
res = minimize(f, [0, 0], bounds=bounds, constraints=[nlc], method="trust-constr")

print(res.x)

# Generate data from the nonlinear constraint
nsamples = 1000  # number of samples
X = -5.0 + 10 * np.random.rand(nsamples, 2)
noise = 0.1 * np.random.randn(nsamples)
y = np.array([g(X[i, :]) for i in range(nsamples)]) + noise

# Approach 1


# Approach 2
niters = 1000  # number of iterations
ssize = 5  # subset size

xit = np.array([0, 0])
for i in range(niters):
    dists = np.linalg.norm(xit - X, axis=1)
    indices = np.argsort(dists)[:ssize]
    Xsub, ysub = X[indices, :], y[indices]
    locmod = LinearRegression().fit(Xsub, ysub)
    res = linprog(
        np.array([1, 1]),
        A_ub=-locmod.coef_.reshape(1, -1),
        b_ub=locmod.intercept_,
        bounds=[(-10, 10), (0, 10)],
    )
    xit = res.x

print("Linear regression (subset size = 5): ", xit)

# Approach 3
niters = 100  # number of iterations
frac = 0.005  # fraction of samples
xit = np.array([0, 0])
for i in range(niters):

    locmod = LocalRegression(frac=frac).fit(X, y)
    locmod.predict(xit)
    res = linprog(
        np.array([1, 1]),
        A_ub=-locmod.coef_[0].reshape(1, -1),
        b_ub=locmod.intercept_[0],
        bounds=[(-10, 10), (0, 10)],
    )
    xit = res.x

print("Local regression (frac = 0.005): ", xit)
