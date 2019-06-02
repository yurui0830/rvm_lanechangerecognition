import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from random import randint

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y

"""
kernels documentation: 
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html
pairwise_kernels(X, Y=None, metric=’linear’, filter_params=False, n_jobs=None, **kwds)
matric = ‘rbf’, ‘sigmoid’, ‘polynomial’, ‘poly’, ‘linear’, ‘cosine’
"""


class Base_RVM(BaseEstimator, RegressorMixin):

    """
    1 The Relevance Vector Machine
      M. E. Tipping

    2 Fast Marginal Likelihood Maximisation for Sparsity Bayesian Models
      M. E. Tipping and A. C. Faul, 2003

    :param:


    """

    def __init__(
        self,
        kernel='linear',  # kernel type
        max_iter=3000,  # maximum iteration times
        tol=1e-3,  # convergence criterion
        alpha=1e-6,  # initial alpha
        threshold_alpha=1e5,  # alpha will be kept if below this threshold
        beta=1e-6,  # initial beta
        verbose=False
    ):
        """Copy params to object properties, no validation."""
        self.kernel = kernel
        self.n_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.threshold_alpha = threshold_alpha
        self.beta = beta
        self.verbose = verbose

    def get_params(self, deep=True):
        """Return parameters as a dictionary."""
        params = {
            'kernel': self.kernel,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'alpha': self.alpha,
            'threshold_alpha': self.threshold_alpha,
            'beta': self.beta,
            'verbose': self.verbose
        }
        return params

    def set_params(self, **parameters):
        """Set parameters using kwargs."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _apply_kernel(self, x1, x2):
        """Apply the selected kernel function to the data."""
        phi = pairwise_kernels(X=x1,Y=x2, metric=self.kernel)
        return phi

    def _posterior(self):
        """Compute the posterior distribution (Sigma and weight) over weights."""
        # P3 Equation(6)
        quantity = np.diag(self.alpha_) + self.beta_ * np.dot(self.phi, self.phi.T)  # phi: ndarray m*n
        # sigma_: ndarray m*m; mean_: ndarray m*1
        self.sigma_ = np.linalg.inv(quantity)
        self.mean_ = self.beta_ * np.dot(self.sigma_, np.dot(self.phi, self.y))

    def _prune(self):
        """Remove basis functions based on alpha values."""
        # if alpha is smaller than the threshold, keep the alpha (the basic vector)
        keep_alpha = self.alpha_ < self.threshold_alpha
        # ensure at least one basic vector is reserved
        if not np.any(keep_alpha):
            keep_alpha[0] = True
        # update relevance vectors/new alpha/current alpha for next loop/gamma/phi/sigma/mean according to keep_alpha
        self.relevance_ = self.relevance_[keep_alpha]
        self.alpha_ = self.alpha_[keep_alpha]
        self.alpha_old = self.alpha_old[keep_alpha]
        self.gamma = self.gamma[keep_alpha]
        self.phi = self.phi[keep_alpha, :]
        self.sigma_ = self.sigma_[np.ix_(keep_alpha, keep_alpha)]
        self.mean_ = self.mean_[keep_alpha]

    def fit(self, X, y):
        """Fit the RVR to the training data."""

        X, y = check_X_y(X, y)
        # obtain n,d from training set
        n_samples, n_features = X.shape
        # original Phi: ndarray n*n
        self.phi = self._apply_kernel(X, X)
        # number of basic functions, equal to the number of the rows (m) in Phi
        n_basis_functions = self.phi.shape[0]

        self.relevance_ = X
        self.y = y
        # initialize alpha, beta, means and record current alpha
        self.alpha_ = self.alpha * np.ones(n_basis_functions,)
        self.beta_ = self.beta
        self.mean_ = np.zeros(n_basis_functions,)
        self.alpha_old = self.alpha_

        # loop
        for i in range(self.n_iter):
            # update Sigma and mean
            self._posterior()

        # algorithm I: The Relevance Vector Machine, M.Tipping
            # P654 Equation(9)
            self.gamma = 1 - self.alpha_*np.diag(self.sigma_)
            self.alpha_ = np.divide(self.gamma, self.mean_ ** 2)
            # P654 Equation(6)(10)
            self.beta_ = (n_samples - np.sum(self.gamma))/((y - np.dot(self.phi.T, self.mean_) ** 2).sum())

            # prune basic vectors
            self._prune()
            if self.verbose:
                print("Iteration: {}".format(i))
                print("Alpha: {}".format(self.alpha_))
                print("Beta: {}".format(self.beta_))
                print("Gamma: {}".format(self.gamma))
                print("m: {}".format(self.mean_))
                print("Relevance Vectors: {}".format(self.relevance_.shape[0]))
            # convergence criterion
            delta = np.amax(np.absolute(self.alpha_ - self.alpha_old))
            if delta < self.tol and i > 1:
                break
            self.alpha_old = self.alpha_

        print(self.alpha_, self.beta_)
        # return value
        return self

    def predict(self, test_X):
        """Evaluate the RVR model at x."""
        # test_X: ndarray n_test*d; self.relevance_: ndarray m*d
        # test_phi: ndarray n_test*m
        test_phi = self._apply_kernel(test_X, self.relevance_)
        y = np.dot(test_phi, self.mean_)

        # return value
        return y
