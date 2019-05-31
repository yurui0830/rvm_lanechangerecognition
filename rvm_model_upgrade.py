import numpy as np
from numpy import linalg as la
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


class OneClass_RVM(BaseEstimator, RegressorMixin):

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
        alpha=1e9,  # initial alpha
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

        x, t = check_X_y(X, y)
        # obtain n,d from training set
        n_samples, n_features = X.shape
        # original Phi: ndarray n*n
        self.phi = self._apply_kernel(x, x)
        # number of basic functions, equal to the number of the rows (m) in Phi
        n_basis_functions = self.phi.shape[0]

        self.relevance_ = x
        self.y = t
        # initialize alpha, beta, means and record current alpha
        self.alpha_ = self.alpha * np.ones(n_basis_functions,)
        self.beta_ = self.beta
        self.mean_ = np.zeros(n_basis_functions,)
        self.alpha_old = self.alpha_

    # Algorithm II: Fast Marginal Likelihood Maximisation for Sparse Bayesian Models, M.Tipping
        # pick a sample, initialize hyper-parameters
        # P7 Equation(26): initialize model with a single sample and set a[i]
        # randomly find a single sample i
        i = randint(self.n_sample)
        self.alpha_[i] = la.norm(self.phi[i])**2/((la.norm(self.phi[i] * self.y)/la.norm(self.phi[i]))**2-1/self.beta_)

        # initialize a quantity C parameter
        quantity_c = np.ones((self.n_sample, self.n_sample))
        # initialize a zero array for theta
        theta = np.zeros((self.n_sample,))
        # P4 Equation 16: initialize theta
        theta[i] = np.linalg.norm(self.kernel[i] * self.y.T) / (self.n_class * np.square(self.kernel[i]).sum())

        # record active samples
        active_sample = [i]

        for i in range(self.n_iter):
            quantity_c = np.diag(1/self.beta_) + self.phi[active_sample].T*self.



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



class mRVM(Base_RVM):
    """

    :param
        kernel
        a(array: n_sample * n_class): scales matrix
        w(array: n_sample * n_class): weight/regressor, most elements are zeros
        y(array: n_class * n_sample): given a sample n, we assign it to the class c with the highest y_cn


    The training phase follows the consecutive updates of A, W and Y

    """
    def __init__(self, n_class: int=3):
        self.n_class = n_class
        self.a = np.full((self.n_sample, n_class), np.inf)
        self.w = np.zeros((self.n_sample, n_class))
        self.y = np.zeros((n_class, self.n_sample))

    def __update_a(self, i: int, s, q):
        """
        P3 Equation 11,12

        :param
            i(int): denote the i-th sample
            s(float): sparsity factor (a scale)
            q(array: n_class,): quality factor (n_class scales)
        :return:
        """
        # P3 Equation 11,12: update A
        if np.square(q).sum() > self.n_class * s:
            self.a[i] = self.n_class*np.square(s) / (np.square(q).sum() - self.n_class*s)
        else:
            self.a[i] = np.inf
        # return model itself
        return  self

    def __update_w(self, active_sample, quantity_kka):
        # P3 Equation 14: update W
        self.w[active_sample] = quantity_kka*self.kernel[active_sample]*self.y.T
        # return model itself
        return self

    def __update_y(self):
        # P2 Equation 3,4: update Y
        return self

    def train(self, training_set, t):
        """
        train this model by labels

        :param
            training_set(array: n_class * n_feature)
            t(array: n_class * n_sample): labels
        :return:

        """

    # pick a sample, initialize hyper-parameters
        # initialize Y to follow target labels t
        self.y = t
        # P3 Equation 15: initialize model with a single sample and set a[i]
        # randomly find a single sample i
        i = randint(self.n_sample)
        temp = 0
        for cls in range(self.n_class):
            temp = temp + np.dot(self.kernel[i], self.y[cls,:]) ** 2
        self.a[i] = self.n_class*np.square(self.kernel[i]).sum() / (temp/np.square(self.kernel[i]).sum() - self.n_class)
        # initialize a quantity C parameter
        quantity_c = np.ones((self.n_sample, self.n_sample))
        # initialize a zero array for theta
        theta = np.zeros((self.n_sample,))
        # P4 Equation 16: initialize theta
        theta[i] = np.linalg.norm(self.kernel[i] * self.y.T) / (self.n_class * np.square(self.kernel[i]).sum())
        # record active samples
        active_sample = [i]

    # update hyper-parameters
        # Pe Equation 17
        # use a quantity to replace inv(K_star*K_star_transpose+A_star), save computational cost
        quantity_kka = np.linalg.inv(self.kernel[active_sample]*self.kernel[active_sample].T + self.a[active_sample])
        # P4 Equation 18/19: calculate S and Q
        # S(array: n_sample * n_sample): sparsity factor (diagonal matrix)
        # Q(array: n_sample * n_class): quality factor
        s = self.kernel*self.kernel.T - self.kernel*self.kernel[active_sample].T*quantity_kka*self.kernel[active_sample]*self.kernel
        q = self.kernel*self.y.T - self.kernel*self.kernel[active_sample].T*quantity_kka*self.kernel[active_sample]*self.y.T
        if theta[i] > 0 and self.a[i] < np.inf: # update A[i] (sample i is already in the model)
            # P4 Equation 20,21: tune Q and S in order not to include the existing sample i
            s[i,i] = self.a[i] * s[i,i] / (self.a[i] - s[i,i])
            q[i] = self.a[i] * q[i] / (self.a[i] - s[i,i])
            # update A
            self.__update_a(i, s[i,i], q[i])
        elif theta[i] > 0 and self.a[i] < np.inf: # add sample i into the model
            # update A
            self.__update_a(i, s[i,i], q[i])
            # add sample i
            #active_sample = np.sort(np.insert(active_sample, i), axis=None)
            active_sample.insert(i).sort()
        elif theta[i] <= 0 and self.a[i] < np.inf: # delete sample i from the model
            # delete A[i]
            self.a[i] = np.inf
            # delete sample i
            active_sample.remove(i)
        # update W
        self.__update_w(active_sample, quantity_kka)
        # update Y
        self.__update_y()

    # recalculate theta for all samples
        # P3 Equation 13: update theta
        for i in range(self.n_sample):
            theta[i] = np.square(q[i]).sum() - self.n_class * s[i,i]

        return self

    def predict(self, test_set, n_quapoint: int=20):
        """

        :param
            test_set():
        :return:

        """

        return self
