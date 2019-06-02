import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import pairwise_kernels
from numpy.random import randint

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
    Fast Marginal Likelihood Maximisation for Sparsity Bayesian Models
    M. E. Tipping and A. C. Faul, 2003

    :param:


    """

    def __init__(
        self,
        kernel='linear',  # kernel type
        max_iter=1000,  # maximum iteration times
        tol=1e-3,  # convergence criterion
        beta=1e-6,  # initial beta
    ):
        """Copy params to object properties, no validation."""
        self.kernel = kernel
        self.n_iter = max_iter
        self.tol = tol
        self.beta = beta
        self.alpha, self.alpha_, self.beta_, self.relevance, self.phi, self.phi_, self.mean_, self.sigma_ = [None]*8

    def get_params(self, deep=True):
        """Return parameters as a dictionary."""
        params = {
            'kernel': self.kernel,
            'max_iter': self.max_iter,
            'tol': self.tol,
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
        # sigma_: ndarray m*m; mean_: ndarray m*1
        self.sigma_ = np.linalg.inv(quantity)
        self.mean_ = self.beta_ * np.dot(self.sigma_, np.dot(self.phi, self.y))

    def fit(self, X, y):
        """Fit the RVR to the training data."""

        x, t = check_X_y(X, y)
        # obtain n,d from training set
        n_samples, n_features = X.shape
        # phi_: original Phi, ndarray n_samples*n_samples
        self.phi_ = self._apply_kernel(x, x)
        self.y = t
        # initialize alpha, beta for the first iteration
        self.alpha_ = np.full((n_samples,), 1e9)
        self.beta_ = self.beta

    # Algorithm II: Fast Marginal Likelihood Maximisation for Sparse Bayesian Models, M.Tipping
        # P7 Equation(26): initialize model with a single sample and set a[i]
        # randomly find a single sample i
        i = np.argmax(norm(np.dot(self.phi_, self.y))/norm(np.sum(self.phi_, 1)))
        # i = randint(n_samples)
        self.alpha_[i] = norm(self.phi_[i])**2/((norm(self.phi_[i]*self.y)/norm(self.phi_[i]))**2-1/self.beta_)
        # record active samples and the number of active samples
        active_sample = np.zeros((n_samples,), dtype=bool)
        active_sample[i] = True
        self.alpha = self.alpha_[active_sample]
        self.phi = self.phi_[active_sample]

        # loop
        for loop in range(self.n_iter):
            # P3 Equation(8): update B (noise) and C: array n_samples*n_samples
            B = np.zeros((n_samples, n_samples))
            np.fill_diagonal(B, 1/self.beta_)
            A = np.diag(self.alpha)
            quantity_c = B + self.phi.T @ A @ self.phi

            # find a sample among non-active samples (selected sample is not included in C)
            s = np.zeros((n_samples, 1))
            q = np.zeros((n_samples, 1))
            theta = np.zeros((n_samples, 1))
            for i in range(n_samples):
                if ~active_sample[i]:
                    # P5 Equation(19): compute s_i and q_i
                    # P7 Step(5): compute theta_i
                    s[i] = self.phi_[i].T @ np.linalg.inv(quantity_c) @ self.phi_[i]
                    q[i] = self.phi_[i].T @ np.linalg.inv(quantity_c) @ self.y
                    theta[i] = q[i]**2 - s[i]
                else:
                    continue
            # select the one which has the highest theta (contribution)
            if np.max(theta) > 0:
                # update active samples
                next_sample = np.argmax(theta)
                active_sample[next_sample] = True
                n_basis_functions = np.sum(active_sample)
                self.phi = self.phi_[active_sample]
                self.relevance = x[active_sample]
                # update alpha and beta
                # P5 Equation(20): update alpha
                self.alpha_[next_sample] = s[next_sample]**2 / (q[next_sample]**2 - s[next_sample])
                self.alpha = self.alpha_[active_sample]
                # update Sigma and mean
                self._posterior()
                # P7 Step(9): update beta
                self.beta_ = (n_samples - n_basis_functions + np.dot(self.alpha, np.diag(self.sigma_))) / \
                             norm(self.y - self.phi.T @ self.mean_)**2
            else:
                # if there is no useful samples, throw an exception and quit this training
                if loop == 0:
                    print('unsolved problem')
                    exit()
                else:
                    B = np.zeros((n_samples, n_samples))
                    np.fill_diagonal(B, 1 / self.beta_)
                    A = np.diag(self.alpha)
                    quantity_c = B + self.phi.T @ A @ self.phi
                    s = np.zeros((n_samples, 1))
                    q = np.zeros((n_samples, 1))
                    theta = np.zeros((n_samples, 1))
                    for i in range(n_samples):
                        if active_sample[i]:
                            # P5 Equation(19): compute s_i and q_i
                            # P7 Step(5): compute theta_i
                            s[i] = self.phi_[i].T @ np.linalg.inv(quantity_c) @ self.phi_[i]
                            q[i] = self.phi_[i].T @ np.linalg.inv(quantity_c) @ self.y
                            theta[i] = q[i] ** 2 - s[i]
                        else:
                            continue
                    delete_sample = np.argwhere(theta < 0)
                    active_sample[delete_sample] = False
                    print('sample deleted:', delete_sample)
                    self.alpha = self.alpha_[active_sample]
                    self.phi = self.phi_[active_sample]
                    self.relevance = x[active_sample]
                    self._posterior()
                    print(self.alpha)
                    break

        # return value
        return self

    def predict(self, test_X):
        """Evaluate the RVR model at x."""
        # test_X: ndarray n_test*d; self.relevance_: ndarray m*d
        # test_phi: ndarray n_test*m
        test_phi = self._apply_kernel(test_X, self.relevance)
        y = np.dot(test_phi, self.mean_)

        # return value
        return y


class mRVM():


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

