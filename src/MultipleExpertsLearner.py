import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import LogisticRegression


class MultipleExpertLearner:
    def __init__(self, params={}):
        self.phi = params.get('phi', 1)
        self.psi = params.get('psi', 1)
        self.theta = params.get('theta', 1)
        self.gamma = params.get('gamma', 0.5)
        self.eps = params.get('eps', 1.0e-3)
        self.tol_params = params.get('TolParams', 1.0e-12)
        self.tol_ll = params.get('TolLikelihood', 1.0e-4)
        self.iters = params.get('MaxIterations', 500)
        self.regulariser = params.get('Regulariser', None)
        self.verbose = params.get('Verbose', False)

    def fit(self, X, Z, feats=None):
        """
        Function to perform learning of ground truth labels from a set of noisy expert labels and an optional set of
        features for jointly learning a supervised classifier.

        :param X: NxM numpy array of M expert votes for N instances
        :param Z: Initial guess for the true ground truth labels. Suggest using the ground_truth_heuristic function in
        tools.
        :param feats: Optional NxK (sparse) numpy array of K features for the N instances enabling training of a binary
        classifier jointly with learning the ground truth labels.
        """
        iters = self.iters
        gamma = self.gamma
        verbose = self.verbose

        shp = X.shape
        N = shp[0]
        M = shp[1]
        if feats:
            K = feats.shape[1]
            w_old = np.zeros([K])
        else:
            w_old = None

        alpha_old = np.zeros(M)
        beta_old = np.zeros(M)
        pi_old = 0
        ll_old = 0
        Z_old = Z

        if verbose: print("Beginning EM algorithm...")
        convergence = False
        iteration = 0
        while not convergence and iteration < iters:
            if verbose: print("Iteration %d of %d" % (iteration+1, iters))

            pi, alpha, beta, clf = self._m_step(X, Z, feats, N)
            Z = self._e_step(X, pi, alpha, beta)

            ll = self._get_likelihood(X, alpha, beta, pi)
            if verbose: print("    Log-likelihood: ", ll)

            Z_shift = sum(np.array([Z >= gamma], dtype=int) != Z_old)

            if verbose:
                if feats:
                    conv2 = norm(w - w_old, ord=1)
                    print("    Movement in logistic regression parameters: ", conv2)

                conv3 = pi - pi_old
                print("    Movement in ground truth probability: ", conv3)
                print("    Movement in ground truth labels: ", Z_shift)

            if feats:
                convergence = self._convergence_test(alpha, alpha_old, beta, beta_old, 0, 0, ll, ll_old, Z_shift)
                w = clf.coef_

            else:
                convergence = self._convergence_test(alpha, alpha_old, beta, beta_old, pi, pi_old, ll, ll_old, Z_shift)

            alpha_old = alpha
            beta_old = beta
            pi_old = pi
            Z_old = np.array([Z >= gamma], dtype=int)
            ll_old = ll
            iteration += 1

        self.Z_ = Z
        self.alpha_ = alpha
        self.beta_ = beta
        self.pi_ = pi
        if feats:
            self.clf_ = clf

        if convergence:
            print("EM algorithm terminated; convergence achieved.")
        else:
            print("EM algorithm terminated; hit iteration cap. Consider reducing convergence tolerances or increasing "
                  "the number of iterations.")

        return

    def evaluate(self, X, feats=None):
        """
        Once trained, this function will evaluate soft class probabilities for a new set of data points.

        :param X: NxM numpy array of M expert votes for N instances
        :param feats: Optional NxK (sparse) numpy array of K features for the N instances enabling training of a binary
        classifier jointly with learning the ground truth labels. Requires model to have been trained with a
        corresponding feature set.
        :return: Returns an Nx1 array of soft probability predictions for the ground truth labels
        """
        if feats and self.clf_:
            pi = self.clf_.predict_proba(feats)[:,1]
        else:
            pi = self.pi

        alpha = self.alpha
        beta = self.beta

        return self._e_step(X, pi, alpha, beta)

    def _m_step(self, X, Z, feats, N):

        phi = self.phi
        psi = self.psi
        theta = self.theta
        eps = self.eps

        sum_Z = Z.sum()
        sum_ZX = np.dot(Z, X)

        if feats:
            clf = self._performRegression(Z, feats)
            pi = clf.predict_proba(feats)[:,1]
        else:
            clf = None
            pi = (theta + sum_Z)/(2 * theta + N)
        alpha = sum_ZX / (phi + sum_Z)
        beta = (X.sum(axis=0) - sum_ZX) / (psi + N - sum_Z)

        pi = np.clip(pi, eps, 1-eps)
        np.clip(alpha, eps, 1-eps, out=alpha)
        np.clip(beta, eps, 1-eps, out=beta)

        return pi, alpha, beta, clf

    @staticmethod
    def _e_step(X, pi, alpha, beta):

        a = beta / alpha
        b = (1 - beta) / (1 - alpha)

        arr = (X * np.log(a) + (1-X) * np.log(b)).sum(axis=1)
        arr += np.log(1 - pi) - np.log(pi)
        arr = np.exp(arr)
        arr += 1
        Z = 1 / arr

        np.clip(Z, 0, 1, out=Z)

        return Z

    def _get_likelihood(self, X, alpha, beta, pi):

        phi = self.phi
        psi = self.psi

        a = np.prod(alpha**X * (1 - alpha)**(1-X), axis=1)
        b = np.prod(beta**X * (1-beta)**(1-X), axis=1)

        ll = np.log(a*pi + b*(1-pi)).sum()

        ll += (phi * np.log(1-alpha)).sum()
        ll += (psi * np.log(1-beta)).sum()

        return ll

    def _convergence_test(self, a1, a0, b1, b0, pi1, pi0, ll, ll_old, Z_shift):

        tol_params = self.tol_params
        tol_ll = self.tol_ll

        if Z_shift > 10: return False

        val = norm(a1 - a0, ord=1) + norm(b1 - b0, ord=1) + abs(pi1 - pi0)

        if val <= tol_params:
            return True

        if ll - ll_old < tol_ll and ll_old != 0:
            return True

        return False

    def _perform_regression(self, Z, feats):

        gamma = self.gamma
        verbose = self.verbose
        if self.regulariser:
            regulariser = self.regulariser

        if verbose: print("    Performing logistic regression on probabilistic labels...")
        clf = LogisticRegression(solver='sag')

        clf.fit(feats, np.array(Z >= gamma, dtype=int))

        return clf

    @property
    def Z(self):
        return self.Z_

    @property
    def alpha(self):
        return self.alpha_

    @property
    def beta(self):
        return self.beta_

    @property
    def pi(self):
        return self.pi_

    @property
    def clf(self):
        return self.clf_