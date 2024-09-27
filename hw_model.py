import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multinomial, norm
from sklearn.cluster import KMeans as km
from sklearn.exceptions import ConvergenceWarning


def fit_hw_model(X, c: float = 1e-4, max_iter: int = 10_000):
    warnings.filterwarnings("error")

    n = np.shape(X)[0]  # length of data set
    assert n > 0
    pi = np.ones(3) / 3
    r = np.zeros([3, n])  # initial responsibility matrix
    sigma = np.ones(2) * np.std(X)
    f = np.ravel(X).astype(float).reshape(-1, 1)
    kmeans = km(n_init="auto", n_clusters=3)  # k-means for initial means

    try:
        kmeans.fit(f)
        centers = np.sort(np.ravel(kmeans.cluster_centers_))

    # if there arent 3 centers than we already know that the variant is not in HW equilibrium
    # or at least doesn't match our expectations
    # if len(centers) < 3:
    except ConvergenceWarning:
        # NOTE: this might not be true
        pi = np.array([1, 0, 0])
        mu = np.array([0, 0])
        sigma = np.array([0, 0])
        log_likelihood = 0
        return {"pi": pi, "mu": mu, "sigma": sigma, "log-likelihood": log_likelihood}

    mu = centers[1:]

    log_likelihoods = []
    distances = []  # distances between log-likelihoods
    iteration = 0
    distance = 1

    while (distance > c) and (iteration < max_iter):
        ## expectation
        r[0][X == 0] = pi[0]  # responsibility towards first component is just the zeros
        r[0][X != 0] = 0

        # responsibility of each value to second and third components
        r[1] = pi[1] * norm.pdf(X, mu[0], sigma[0])
        r[2] = pi[2] * norm.pdf(X, mu[1], sigma[1])
        r = r / np.sum(r, axis=0)

        ## maximization
        # total responsibility of a component divided by total # of observations
        # pi = np.clip(np.sum(r, axis=1), a_min=1, a_max=None) / (2 * n)
        pi = np.sum(r, axis=1) / n

        # MLE for mean in normal distribution
        mu[0] = np.average(X, weights=r[1])
        mu[1] = np.average(X, weights=r[2])

        mu[0] = (mu[0] + 0.5 * mu[1]) / 2  # fixing 2mu_2 = mu_3
        mu[1] = 2 * mu[0]

        # MLE for standard deviation in normal distribution
        sigma[0] = np.average((X - mu[0]) ** 2, weights=r[1]) ** 0.5
        sigma[1] = np.average((X - mu[1]) ** 2, weights=r[2]) ** 0.5
        sigma = np.clip(sigma, a_min=1e-4, a_max=None)

        ## score
        #  likelihood of each observation in hurdle model
        hurdle = np.where(X == 0, pi[0], 0)

        # likelihood of each observation in normal distribution
        gmm = pi[1] * norm.pdf(X, mu[0], sigma[0]) + pi[2] * norm.pdf(
            X, mu[1], sigma[1]
        )
        log_likelihood = np.sum(np.log(hurdle + gmm))
        log_likelihoods.append(log_likelihood)

        iteration += 1
        if iteration > 1:
            distance = np.abs(log_likelihoods[-2] - log_likelihoods[-1])
            distances.append(distance)

    return {"pi": pi, "mu": mu, "sigma": sigma, "log-likelihood": log_likelihood}


def plot_hw_model(
    X: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    p: float | str,  # p-value of KS test
    output: str,
):
    """
    X: input counts
    pi, mu, sigma: model parameters
    """
    n = X.shape[0]
    M = np.max(X)
    ax = plt.subplots(figsize=(10, 6))[1]
    plt.hist(
        X, alpha=0.20, color="grey", edgecolor="white", linewidth=3, bins=range(M + 2)
    )

    plt.scatter(0, n * pi[0], color="green")
    curve2 = np.linspace(mu[0] - 10 * sigma[0], mu[0] + 10 * sigma[0], 1000)
    curve3 = np.linspace(mu[1] - 10 * sigma[1], mu[1] + 10 * sigma[1], 1000)
    plt.fill_between(
        curve2,
        n * norm.pdf(curve2, mu[0], sigma[0]) * pi[1],
        color="orange",
        alpha=0.5,
    )
    plt.fill_between(
        curve3,
        n * norm.pdf(curve3, mu[1], sigma[1]) * pi[2],
        color="blue",
        alpha=0.5,
    )

    q = np.sqrt(pi[2])

    ax.set_yscale("log")
    ax.set_ylim(bottom=1)
    ax.set_xlim(left=0, right=M + 1)
    # ax.set_xticks(range(0, M + 1, 5))
    plt.xlabel("Evidence Depth")
    plt.ylabel("Number of Samples")
    plt.title(f"q={q}; null model p-value={p}")
    plt.savefig(output)
    plt.close()


def theta_to_data(N, pi, mu_2, sigma):
    draws = multinomial.rvs(n=N, p=pi)

    X = [0 for _ in range(draws[0])]
    X_g = norm.rvs(mu_2, sigma[0], size=draws[1])
    X_g2 = norm.rvs(2 * mu_2, sigma[1], size=draws[2])

    while sum(X_g < -0.5) > 0:
        X_g_new = norm.rvs(mu_2, sigma[0], size=sum(X_g < -0.5))
        X_g = [x for x in X_g if x >= -0.5]
        X_g = np.concatenate((X_g, X_g_new))

    while sum(X_g2 < -0.5) > 0:
        X_g2_new = norm.rvs(2 * mu_2, sigma[1], size=sum(X_g2 < -0.5))
        X_g2 = [x for x in X_g2 if x >= -0.5]
        X_g2 = np.concatenate((X_g2, X_g2_new))

    X = np.concatenate((X, X_g))
    X = np.concatenate((X, X_g2))
    X = np.round(X).astype(int)

    return X
