import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import pairwise_kernels
from numpy.random import randint
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y
import warnings
warnings.filterwarnings("error")

"""
kernels documentation: 
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html
pairwise_kernels(X, Y=None, metric=’linear’, filter_params=False, n_jobs=None, **kwds)
matric = ‘rbf’, ‘sigmoid’, ‘polynomial’, ‘poly’, ‘linear’, ‘cosine’
"""


class MultiClass_RVM(BaseEstimator, RegressorMixin):

    """
    Multiclass Relevance Vector Machines: Sparsity and Accuracy
    Ioannis Psorakis, Theodoros Damoulasm, Mark A. Girolami

    Fast Marginal Likelihood Maximisation for Sparsity Bayesian Models
    M. E. Tipping and A. C. Faul, 2003

    :param:


    """

    def __init__(
        self,
        kernel='linear',  # kernel type
        max_iter=100,  # maximum iteration times
        conv=1e-3,  # convergence criterion
        beta=1e-6,  # initial beta
    ):
        """Copy params to object properties, no validation."""
        self.kernel = kernel
        self.n_iter = max_iter
        self.conv = conv
        self.beta = beta
        self.alpha, self.alpha_, self.alpha_old, self.beta_, self.relevance, self.phi, self.phi_,\
            self.y, self.mean_, self.sigma_ = [None]*10

    def get_params(self, deep=True):
        """Return parameters as a dictionary."""
        params = {
            'kernel': self.kernel,
            'max_iter': self.max_iter,
            'conv': self.conv,
            'beta': self.beta,
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
        quantity = np.diag(self.alpha) + self.beta_ * (self.phi @ self.phi.T)  # phi: ndarray m*n
        # sigma_: ndarray m*m; mean_: ndarray m*c
        self.sigma_ = np.linalg.inv(quantity)
        self.mean_ = self.beta_ * (self.sigma_ @ self.phi @ self.y.T)

    def fit(self, X, Y):
        """Fit the RVR to the training data."""

        # obtain n,d from training set
        n_class, n_samples = Y.shape
        # phi_: original Phi, ndarray n_samples*n_samples
        self.phi_ = self._apply_kernel(X, X)
        self.y = Y
        # initialize alpha, beta for the first iteration
        self.alpha_ = np.full((n_samples,), 1e9)
        self.beta_ = self.beta

    # Algorithm: Fast Marginal Likelihood Maximisation for Sparse Bayesian Models, M.Tipping
        # find a single sample i
        i = np.argmax(np.sum((self.phi_ @ self.y.T)**2, 1)/(n_class * np.sum(self.phi_ ** 2, 1)))
        # i = randint(n_samples)
        # P3 Equation(15) + Fast P7 Equation(26): initialize alpha for the first sample
        self.alpha_[i] = n_class * (self.phi_[i]**2).sum()/\
                         (((self.phi_[i]*self.y)**2).sum()/(self.phi_[i]**2).sum() - n_class/self.beta_)
        # record active samples and the number of active samples
        active_sample = np.zeros((n_samples,), dtype=bool)
        active_sample[i] = True
        self.alpha = self.alpha_[active_sample]
        self.alpha_old = self.alpha_
        self.phi = self.phi_[active_sample]
        # Fast P3 Equation(8): update B (noise) and C: array n_samples*n_samples
        B = np.zeros((n_samples, n_samples))
        np.fill_diagonal(B, 1 / self.beta_)
        A = np.diag(self.alpha)
        quantity_c_inv = np.linalg.inv(B + self.phi.T @ A @ self.phi)
        # calculate sparsity, quality quantities and contribution (s, q and theta)
        s = np.zeros((n_samples,))
        q = np.zeros((n_samples,))
        theta = np.zeros((n_samples,))
        for i in range(n_samples):
            # Fast P5 Equation(19) + P2 Equation(11,12): compute s_i and q_i
            s[i] = self.phi_[i].T @ quantity_c_inv @ self.phi_[i]
            temp_q = self.phi_[i].T @ quantity_c_inv @ self.y.T
            # P7 Step(5): compute theta_i
            if active_sample[i]:
                # P4 Equation (20,21): compute s_i and q_i
                s[i] = self.alpha_[i] * s[i] / (self.alpha_[i] - s[i])
                q[i] = ((self.alpha_[i] * temp_q / (self.alpha_[i] - s[i]))**2).sum()
            else:
                q[i] = (temp_q ** 2).sum()
            theta[i] = q[i] - n_class*s[i]
        i = randint(n_samples)
        if theta[i] > 0:
            add_sample = i
            delete_sample = None
        else:
            add_sample = None
            delete_sample = i
        update = 0
        # loop
        for loop in range(self.n_iter):

            # update alpha
            # add non-active but contributed samples
            if add_sample:
                add_sample = i
                # update active samples / hyper-parameters
                active_sample[add_sample] = True
                n_basis_functions = np.sum(active_sample)
                print('add samples', add_sample, 'samples:', n_basis_functions)
                self.phi = self.phi_[active_sample]
                self.relevance = X[active_sample]
                # update alpha and beta
                # P3 Equation(11,12): update alpha
                try:
                    self.alpha_[add_sample] = n_class * s[add_sample]**2 / theta[add_sample]
                except RuntimeWarning:
                    print('warning')
                self.alpha = self.alpha_[active_sample]
                # update Sigma and mean
                self._posterior()
                # P7 Step(9): update beta
                self.beta_ = n_class * (n_samples - n_basis_functions + np.dot(self.alpha, np.diag(self.sigma_))) / \
                             ((self.y.T - self.phi.T @ self.mean_)**2).sum()
                # quantize the update during this iteration
                update = abs(self.alpha_[add_sample])
            # delete active but non-contributed samples
            elif delete_sample:
                # update active samples / hyper-parameters
                active_sample[delete_sample] = False
                n_basis_functions = np.sum(active_sample)
                print('delete samples', delete_sample, 'samples:', n_basis_functions)
                self.phi = self.phi_[active_sample]
                self.relevance = X[active_sample]
                self.alpha_[delete_sample] = 1e9
                self.alpha = self.alpha_[active_sample]
                self._posterior()
                self.beta_ = n_class * (n_samples - n_basis_functions + np.dot(self.alpha, np.diag(self.sigma_))) / \
                             ((self.y.T - self.phi.T @ self.mean_)**2).sum()
                # quantize the update during this iteration
                update = abs(self.alpha_old[delete_sample])
            # check convergence criterion
            if update < 1e-5:
                print('model converged')
                print(self.alpha)
                print(self.beta_)
                break
            else:
                self.alpha_old = self.alpha_
                # update A, B and c
                np.fill_diagonal(B, 1 / self.beta_)
                A = np.diag(self.alpha)
                quantity_c_inv = np.linalg.inv(B + self.phi.T @ A @ self.phi)
                # calculate sparsity, quality quantities and contribution (s, q and theta)
                for i in range(n_samples):
                    # Fast P5 Equation(19) + P2 Equation(11,12): compute s_i and q_i
                    s[i] = self.phi_[i].T @ quantity_c_inv @ self.phi_[i]
                    temp_q = self.phi_[i].T @ quantity_c_inv @ self.y.T
                    # P7 Step(5): compute theta_i
                    if active_sample[i]:
                        # P4 Equation (20,21): compute s_i and q_i
                        s[i] = self.alpha_[i] * s[i] / (self.alpha_[i] - s[i])
                        q[i] = ((self.alpha_[i] * temp_q / (self.alpha_[i] - s[i])) ** 2).sum()
                    else:
                        q[i] = (temp_q ** 2).sum()
                    theta[i] = q[i] - n_class*s[i]
                try:
                    add = np.max(theta[~active_sample])
                    delete = np.min(theta[active_sample])
                except ValueError:
                    i = randint(n_samples)
                    if theta[i] > 0:
                        add_sample = i
                        delete_sample = None
                    else:
                        add_sample = None
                        delete_sample = i
                else:
                    if add > abs(delete):
                        add_sample = np.argwhere(theta == add)[0][0]
                        # prevent infinite loop
                        if add_sample == delete_sample:
                            i = randint(n_samples)
                            if theta[i] > 0:
                                add_sample = i
                                delete_sample = None
                            else:
                                add_sample = None
                                delete_sample = i
                        # prevent overwrite active sample
                        if active_sample[add_sample]:
                            add_sample = np.argwhere(theta == add)[1][0]
                            delete_sample = None
                    else:
                        delete_sample = np.argwhere(theta == delete)[0][0]
                        # prevent infinite loop
                        if delete_sample == add_sample:
                            i = randint(n_samples)
                            if theta[i] > 0:
                                add_sample = i
                                delete_sample = None
                            else:
                                add_sample = None
                                delete_sample = i

        # return value
        return self

    def predict(self, test_X):
        """Evaluate the RVR model at x."""
        # test_X: ndarray n_test*d; self.relevance_: ndarray m*d
        # test_phi: ndarray n_test*m
        test_phi = self._apply_kernel(test_X, self.relevance)
        y = np.dot(test_phi, self.mean_)
        t = np.argmax(y, axis=1)
        # return value
        return t
