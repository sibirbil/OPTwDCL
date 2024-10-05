#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sibirbil
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin, BaseEstimator


class SklearnEstimator:
    """
    This base class is dummy, just for guideline. Sklearn does not
    provide a base class that has both fit and predict
    """

    def fit(self, X: np.array, y: np.array):
        raise NotImplementedError("Needs to implement fit(X, y)")

    def predict(self, X0: np.array):
        raise NotImplementedError("Needs to implement predict(X, y)")


class LocalRegression(RegressorMixin, BaseEstimator, SklearnEstimator):
    """
    Parameters
    ----------
        frac : Fraction of total samples used for local fit

    Source
    ------
    Elements of Statistical Learning, T. Hastie, R. Tibshirani, J. Friedman,
    2nd Edition, Springer (Section 6.3)
    """

    def __init__(self, frac=0.2):

        self.X = None
        self.y = None
        self.frac = frac
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: np.array, y: np.array):

        self.X = X
        self.y = y

        return self

    def predict(self, X0: np.array):

        if self.X is None:
            raise ValueError("You need to fit the LocalRegression model first")

        N = self.X.shape[0]
        n_neighbors = int(np.ceil(self.frac * N))

        X0 = X0.reshape(1, -1)

        N0 = X0.shape[0]
        y0 = np.zeros(N0)

        self.coef_ = np.zeros(X0.shape)
        self.intercept_ = np.zeros(N0)

        h = [
            np.sort(np.linalg.norm(self.X - X0[i], axis=1))[n_neighbors]
            for i in range(N0)
        ]

        for i in range(N0):
            if h[i] < 1.0e-6:
                # Just to avoid division by zero
                t = np.linalg.norm(self.X - X0[i], axis=1) / (h[i] + 1.0e-6)
            else:
                t = np.linalg.norm(self.X - X0[i], axis=1) / h[i]
            mask = t <= 1.0
            Kn = np.zeros(N)
            Kn[mask] = (1.0 - t[mask] ** 3.0) ** 3.0
            Wy = np.sqrt(Kn) * self.y
            WX = np.sqrt(Kn[:, None]) * self.X
            locmod = LinearRegression().fit(WX, Wy)
            y0[i] = locmod.predict(X0[i].reshape(1, -1))
            self.coef_[i] = locmod.coef_
            self.intercept_[i] = locmod.intercept_

        return y0
