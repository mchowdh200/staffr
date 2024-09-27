import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon, geom, multinomial
from sklearn.cluster import KMeans as km


def nonhw_model(Y, c=1e-4):
    assert Y.any(), f"input all zero -- nonhw_model"
    # we add a "fake sample" to avoid the case when there are no zero values
    X = np.zeros(len(Y) + 1)
    X[1:] = Y
    n = X.shape[0]  # length of data set

    # responsibility matrix
    r = np.zeros(n)

    # probability of an observation belonging to zero component
    alpha = np.sum(X == 0) / n

    # expected value of the geometric distribution
    lambda_ = n / np.sum(X != 0)

    log_likelihoods = []
    iteration = 0
    distance = 1

    while distance > c:
        ## expectation
        # responsibility of zero values to first mode
        r[X == 0] = alpha

        # responsibility of nonzero values to first component
        r[X != 0] = (1 - alpha) * geom.pmf(X[X != 0], 1 / lambda_)
        r[r != 0] /= r[r != 0]  # normalization for nonzero values

        ## maximization
        alpha = np.sum(r[X == 0]) / np.sum(r)  # MLE for
        # reciprocal of MLE for p in geometric distribution
        lambda_ = np.dot(X[X != 0], r[X != 0]) / ((1 - alpha) * np.sum(r))
        lambda_ = 6 if lambda_ > 6 else lambda_

        ## score
        #  likelihood of each observation in hurdle model
        hurdle = np.clip(
            np.where(X == 0, alpha, (1 - alpha) * geom.pmf(X, 1 / lambda_)),
            a_min=1e-8,
            a_max=None,
        )
        log_likelihood = np.sum(np.log(hurdle))
        log_likelihoods.append(log_likelihood)

        iteration += 1
        if iteration > 1:
            distance = np.abs(log_likelihoods[-2] - log_likelihoods[-1])

    return alpha, lambda_, log_likelihoods[-1]


def plot_nonhw_model(X, n, alpha, lambda_):
    ax = plt.subplots(figsize=(6, 6))[1]
    plt.hist(X, alpha=0.20, bins=max(X), color="grey", edgecolor="white", linewidth=3)

    curve1 = np.linspace(1, 10 * lambda_, 1000)
    plt.fill_between(
        curve1,
        n * expon.pdf(curve1, loc=1, scale=lambda_) * (1 - alpha),
        color="red",
        alpha=0.5,
    )
    plt.scatter(0, n * alpha, color="green")

    ax.set_yscale("log")
    ax.set_ylim(bottom=1)
    ax.set_xlim(left=0, right=max(X) + 1)
    plt.xlabel("Evidence Depth")
    plt.ylabel("Number of Samples")
    plt.show()


def psi_to_data(N, alpha, lambda_):
    if alpha == 1.0:
        raise ValueError(f"alpha == 1: {alpha=}, {lambda_=}")
    elif np.isnan(alpha):
        raise ValueError(f"alpha is NaN: {alpha}")
    # draw N samples from a multinomial distribution
    draws = multinomial.rvs(n=N, p=[alpha, 1 - alpha])

    X = [0 for _ in range(draws[0])]
    X_l = geom.rvs(1 / lambda_, size=draws[1])

    X = np.concatenate((X, X_l))
    X = np.round(X).astype(int)

    # ensure at least one non-zero value to avoid division by zero
    if not X.any():
        X[-1] = 1

    return X


def ks_calc(X, show_plot, runs=1000):
    X = np.sort(X)
    X_unique = np.unique(X)
    N = len(X)

    alpha_hat_x, lambda_hat_x, _ = nonhw_model(X)
    if np.isnan(alpha_hat_x):
        return "NA"

    EDF_x = [np.sum(X <= x) / len(X) for x in X_unique]
    CDF_x = [
        alpha_hat_x
        if x == 0
        else alpha_hat_x + (1 - alpha_hat_x) * geom.cdf(x, 1 / lambda_hat_x)
        for x in X_unique
    ]
    ks_x = max([abs(cdf_val - edf_val) for cdf_val, edf_val in zip(CDF_x, EDF_x)])

    null_dist = []

    warnings.resetwarnings()
    for _ in range(runs):
        Y_i = psi_to_data(N, alpha_hat_x, lambda_hat_x)
        Y_i = np.sort(Y_i)
        assert Y_i.ravel().astype(int).any(), f"Y_i all zero: {Y_i}"
        Y_i_unique = np.unique(Y_i)

        alpha_hat_y, lambda_hat_y, _ = nonhw_model(Y_i)
        EDF_y = [np.sum(Y_i <= y) / len(Y_i) for y in Y_i_unique]
        CDF_y = [
            alpha_hat_y
            if y == 0
            else alpha_hat_y + (1 - alpha_hat_y) * geom.cdf(y, 1 / lambda_hat_y)
            for y in Y_i_unique
        ]
        ks_y = max([abs(cdf_val - edf_val) for cdf_val, edf_val in zip(CDF_y, EDF_y)])
        null_dist.append(ks_y)

    if show_plot:
        plt.hist(
            null_dist,
            weights=np.zeros_like(null_dist) + 1.0 / len(null_dist),
            bins=30,
            color="grey",
            alpha=0.5,
        )
        plt.axvline(ks_x, color="red", label="D_x")
        plt.xlabel("D (KS Distance)")
        plt.ylabel("Pr(D)")
        plt.legend()
        plt.show()

    p_val = np.sum([x for x in null_dist if x > ks_x]) / np.sum(null_dist)
    return p_val
