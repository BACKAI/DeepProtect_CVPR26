import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from scipy import linalg

class OALDA(LinearDiscriminantAnalysis):
    def __init__(self, order=None, selected_indices=None, lambda_reg=1.0, delta=0.1, 
                 n_components=1, solver='lsqr', shrinkage='auto', priors=None, 
                 tol=1e-4, max_iter=1000, lr=0.01):
        super().__init__(n_components=n_components, solver=solver, shrinkage=shrinkage, 
                        priors=priors, tol=tol)
        self.order = order
        self.selected_indices = selected_indices
        self.lambda_reg = lambda_reg
        self.delta = delta
        self.max_iter = max_iter
        self.lr = lr

    def _solve_lstsq(self, X, y, shrinkage, covariance_estimator):
        from sklearn.discriminant_analysis import _class_means, _class_cov
        self.means_ = _class_means(X, y)
        self.covariance_ = _class_cov(
            X, y, self.priors_, shrinkage, covariance_estimator
        )

        Sw = self.covariance_

        R = np.zeros_like(Sw)
        for i, j in zip(self.order[:-1], self.order[1:]):
            diff = X[i] - X[j]
            R += diff

        lambda_rank = 1
        Sw_mod = Sw + lambda_rank * R

        self.coef_ = linalg.lstsq(Sw_mod, self.means_.T)[0].T
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )
