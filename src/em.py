import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import LogisticRegression


def EMAlgorithm(X, Z, phi=1, psi=1, theta=0, gamma=0.5, delta=1e-12, eps=1e-3):
    """
    Function to perform learning of ground truth labels from
    """
    shp = X.shape
    N = shp[0]
    M = shp[1]

    alpha_old = np.zeros(M)
    beta_old = np.zeros(M)
    pi_old = 0
    ll_old = 0
    Z_old = Z

    def MStep(X, Z, phi, psi, theta, eps):

        sum_Z = Z.sum()
        sum_ZX = np.dot(Z, X)
        #print()
        #print(sum_ZX)
        #print()

        pi = (theta + sum_Z)/(2 * theta + N)
        alpha = sum_ZX / (phi + sum_Z)
        beta = (X.sum(axis=0) - sum_ZX) / (psi + N - sum_Z)

        pi = np.clip(pi, eps, 1-eps)
        np.clip(alpha, eps, 1-eps, out=alpha)
        np.clip(beta, eps, 1-eps, out=beta)

        return pi, alpha, beta


    def EStep(X, pi, alpha, beta, eps):

        a = beta / alpha
        b = (1 - beta) / (1 - alpha)

        arr = (X * np.log(a) + (1-X) * np.log(b)).sum(axis=1)
        arr += np.log(1 - pi) - np.log(pi)
        arr = np.exp(arr)
        arr += 1
        Z = 1 / arr

        np.clip(Z, 0, 1, out=Z)

        return Z

    def getLikelihood(X, alpha, beta, pi, phi, psi):
        a = np.prod(alpha**X * (1 - alpha)**(1-X), axis=1)
        b = np.prod(beta**X * (1-beta)**(1-X), axis=1)

        ll = np.log(a*pi + b*(1-pi)).sum()

        ll += (phi * np.log(1-alpha)).sum()
        ll += (psi * np.log(1-beta)).sum()

        return ll


    def convergenceTest(a1, a0, b1, b0, pi1, pi0, ll, ll_old, delta, features=False):

        val = norm(a1 - a0, ord=1) + norm(b1 - b0, ord=1) + abs(pi1 - pi0)

        if val <= delta:
            return True

        if ll - ll_old < 1e-4 and ll_old != 0:
            return True

        return False

    print("Beginning EM algorithm...")
    convergence = False
    iteration = 0
    while not(convergence) and iteration < 500:
        print("Iteration %d of 500" % iteration)
        pi, alpha, beta = MStep(X, Z, phi, psi, theta, eps)
        #print(pi)
        #print(alpha)
        #print(beta)
        Z = EStep(X, pi, alpha, beta, eps)
        #print(sum(Z))


        ll = getLikelihood(X, alpha, beta, pi, phi, psi)
        print("    Log-likelihood: ", ll)

        convergence = convergenceTest(alpha, alpha_old, beta, beta_old, pi, pi_old, ll, ll_old, delta)

        conv3 = pi - pi_old
        Z_shift = sum(np.array(Z >= gamma, dtype=int) != Z_old)

        print("    Movement in ground truth probability: ", conv3)
        print("    Movement in ground truth labels: ", Z_shift)


        alpha_old = alpha
        beta_old = beta
        pi_old = pi
        Z_old = np.array(Z >= gamma, dtype=int)
        ll_old = ll
        iteration += 1

    if convergence:
        print("EM algorithm terminated; convergence achieved.")

    return Z, alpha, beta, pi


def EMAlgorithm_LogisticRegression(X, Z, feats, phi, psi, iters=50, gamma=0.5, delta=1e-12, eps=1e-3):

    shp = X.shape
    N = shp[0]
    M = shp[1]
    K = feats.shape[1]

    alpha_old = np.zeros(M)
    beta_old = np.zeros(M)
    pi_old = np.zeros(N)
    params_old = np.zeros(K)
    Z_old = Z
    ll_old = 0

    def MStep(X, Z, feats, phi, psi, gamma, eps):

        sum_Z = Z.sum()
        sum_ZX = np.dot(Z, X)
        #print()
        #print(sum_ZX)
        #print()

        clf = PerformRegression(Z, feats, gamma)
        #np.clip(pi, eps, 1-eps, out=pi)

        alpha = sum_ZX / (phi + sum_Z)
        beta = (X.sum(axis=0) - sum_ZX) / (psi + N - sum_Z)

        np.clip(alpha, eps, 1-eps, out=alpha)
        np.clip(beta, eps, 1-eps, out=beta)

        return clf, alpha, beta


    def EStep(X, pi, alpha, beta, eps):

        a = beta / alpha
        b = (1 - beta) / (1 - alpha)

        arr = (X * np.log(a) + (1-X) * np.log(b)).sum(axis=1)
        arr += np.log(1 - pi) - np.log(pi)
        arr = np.exp(arr)
        arr += 1
        Z = 1 / arr

        np.clip(Z, 0, 1, out=Z)

        return Z

    def PerformRegression(Z, feats, gamma):
        print("    Performing logistic regression on probabilistic labels...")
        clf = LogisticRegression(solver='sag')

        clf.fit(feats, np.array(Z >= gamma, dtype=int))

        #probs = clf.predict_proba(feats)

        return clf


    def convergenceTest(a1, a0, b1, b0, ll, ll_old, delta):

        val = norm(a1 - a0, ord=1) + norm(b1 - b0, ord=1)# + norm(pi1 - pi0, ord=1)
        print("    Total movement in alpha + beta: ", val)
        if val <= delta:
            return True

        if ll - ll_old < 1e-4 and ll_old != 0:
            return True

        return False

    def getLikelihood(X, alpha, beta, pi, phi, psi):
        a = np.prod(alpha**X * (1 - alpha)**(1-X), axis=1)
        b = np.prod((1-beta)**X * beta**(1-X), axis=1)

        ll = np.log(a*pi + b*(1-pi)).sum()

        ll += (phi * np.log(1-alpha)).sum()
        ll += (psi * np.log(1-beta)).sum()

        return ll


    print("Beginning EM algorithm with logistic regression...")
    convergence = False
    iteration = 0
    while (not convergence) and iteration < iters:
        print("Iteration %d of %d" % (iteration+1, iters))
        clf, alpha, beta = MStep(X, Z, feats, phi, psi, gamma, eps)
        #print(pi)
        #print(alpha)
        #print(beta)
        pi = clf.predict_proba(feats)[:,1]
        np.clip(pi, eps, 1-eps, out=pi)

        Z = EStep(X, pi, alpha, beta, eps)
        #print(sum(Z))
        params = clf.coef_
        #print(params.shape)
        #return params

        ll = getLikelihood(X, alpha, beta, pi, phi, psi)
        convergence = convergenceTest(alpha, alpha_old, beta, beta_old, ll, ll_old, delta)
        conv2 = norm(params - params_old, ord=1)
        conv3 = norm(pi - pi_old, ord=1)
        Z_shift = sum(np.array(Z >= gamma, dtype=int) != Z_old)
        if Z_shift > 10: convergence = False

        print("    Movement in logistic regression parameters: ", conv2)
        print("    Movement in ground truth probabilities: ", conv3)
        print("    Movement in ground truth labels: ", Z_shift)


        print("    Log-likelihood: ", ll)

        alpha_old = alpha
        beta_old = beta
        pi_old = pi
        params_old = params
        Z_old = np.array(Z >= gamma, dtype=int)
        ll_old = ll
        iteration += 1

    if convergence:
        print("EM algorithm terminated; convergence achieved.")

    return Z, alpha, beta, clf