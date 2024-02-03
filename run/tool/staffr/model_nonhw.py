import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, geom, expon, multinomial
from sklearn.cluster import KMeans as km
import csv

def plot_model_nonhw(X, n, alpha, lambda_):
    ax = plt.subplots(figsize = (6,6))[1]
    plt.hist(X, alpha = 0.20, bins = max(X), color = 'grey', edgecolor = 'white', linewidth = 3) # plot histogram of input data set

    curve1 = np.linspace(1, 10 * lambda_, 1000)
    plt.plot(curve1, n * expon.pdf(curve1, loc = 0, scale = lambda_) * (1-alpha), linewidth = 3, color = 'red')
    plt.scatter(0, n * alpha, color = 'red')

    ax.set_yscale("log") 
    ax.set_ylim(bottom=1)
    ax.set_xlim(left=0, right = max(X)+1)
    plt.xlabel('Evidence Depth')
    plt.ylabel('Number of Samples')
    plt.show()

def model_nonhw(X, show_plot, show_distance):

    # TODO: fine-tune initial paramters, put in separate function
    # initial values for parameters
    n = np.shape(X)[0] # length of data set
    r = np.zeros([3,n]) # responsibility matrix
    alpha = np.sum(X == 0) / n # probability of an observation belonging to zero component
    lambda_ = n / np.sum(X != 0) # expected value of the geometric distribution
    log_likelihoods = []
    iteration = 0
    distance = 1

    while distance > (1/(n*10)): # convergence criterion

        # expectation
        r[0][X==0] = alpha # responsibility of zero values to first mode
        r[0][X!=0] = (1-alpha) * geom.pmf(X[X!=0], 1/lambda_) # responsibility of nonzero values to first component
        r = r / np.sum(r, axis = 0) # normalization

        # maximization
        alpha = np.sum(r[0][X==0]) / np.sum(r[0]) # MLE for 
        lambda_ = np.dot(X[X != 0], r[0][X != 0]) / ((1-alpha)*np.sum(r[0])) # reciprocal of MLE for p in geometric distribution
        lambda_ = 6 if lambda_ > 6 else lambda_

        # score
        hurdle = np.where(X == 0, alpha, (1-alpha) * geom.pmf(X,1/lambda_)) #  likelihood of each observation in hurdle model
        log_likelihood = np.sum(np.log(hurdle)) # sum of log of likelihood of each observation
        log_likelihoods.append(log_likelihood) 

        iteration += 1
        if iteration > 1:
            distance = np.abs(log_likelihoods[-2]-log_likelihoods[-1]) # magnitude of difference between each 
            if(show_distance):
                print(distance)

    if(show_plot):
        plot_model_nonhw(X, n, alpha, lambda_)

    return alpha, lambda_,log_likelihoods[-1]

def psi_to_data(N, alpha, lambda_):

    draws = multinomial.rvs(n = N, p = [alpha, 1 - alpha])

    X = [0 for _ in range(draws[0])]
    X_l = geom.rvs(1/lambda_, size = draws[1])

    X = np.concatenate((X,X_l))
    X = np.round(X).astype(int)

    return X

def ks_calc(X, show_plot):
    X = np.sort(X)
    X_unique = np.unique(X)
    N = len(X)

    alpha_hat_x, lambda_hat_x, _ = model_nonhw(X, False, False)
    EDF_x = [np.sum(X <= x) / len(X) for x in X_unique]
    CDF_x = [alpha_hat_x if x == 0 else alpha_hat_x + (1 - alpha_hat_x) * geom.cdf(x, 1/lambda_hat_x) for x in X_unique]
    ks_x = max([abs(cdf_val - edf_val) for cdf_val, edf_val in zip(CDF_x, EDF_x)])

    null_dist = []

    for _ in range(1000):
        Y_i = psi_to_data(N, alpha_hat_x, lambda_hat_x)
        Y_i = np.sort(Y_i)
        Y_i_unique = np.unique(Y_i)

        alpha_hat_y, lambda_hat_y, _ = model_nonhw(Y_i, False, False)
        EDF_y = [np.sum(Y_i <= y) / len(Y_i) for y in Y_i_unique]
        CDF_y = [alpha_hat_y if y == 0 else alpha_hat_y + (1 - alpha_hat_y) * geom.cdf(y, 1/lambda_hat_y) for y in Y_i_unique]
        ks_y = max([abs(cdf_val - edf_val) for cdf_val, edf_val in zip(CDF_y, EDF_y)])
        null_dist.append(ks_y)

    if(show_plot):
        plt.hist(null_dist, weights=np.zeros_like(null_dist) + 1. / len(null_dist), bins = 30, color = 'grey', alpha = 0.5)
        plt.axvline(ks_x, color='red', label = r'D$_x$')
        plt.xlabel('D (KS Distance)')
        plt.ylabel('Pr(D)')
        plt.legend()
        plt.show()

    p_val = np.sum([x for x in null_dist if x > ks_x]) / np.sum(null_dist)
    return p_val

def psi_to_data(N, alpha, lambda_):

    draws = multinomial.rvs(n = N, p = [alpha, 1 - alpha])

    X = [0 for _ in range(draws[0])]
    X_l = geom.rvs(1/lambda_, size = draws[1])

    X = np.concatenate((X,X_l))
    X = np.round(X).astype(int)

    return X

X = psi_to_data (1000, 0.5, 5)

# def theta_to_data(N, pi, alpha, lambda_, mu_2, sigma):

#     draws = multinomial.rvs(n = N, p = pi)
#     draws_alpha = multinomial.rvs(n = draws[0], p = [alpha, 1 - alpha])

#     X = [0 for _ in range(draws_alpha[0])]
#     X_l = geom.rvs(1/lambda_, size = draws_alpha[1])
#     X_g = norm.rvs(mu_2, sigma[0], size = draws[1])
#     X_g2 = norm.rvs(2 * mu_2, sigma[1], size = draws[2])

#     while sum(X_g < -0.5) > 0:
#         X_g_new = norm.rvs(mu_2, sigma[0], size = sum(X_g < -0.5))
#         X_g = [x for x in X_g if x >= -0.5]
#         X_g = np.concatenate((X_g, X_g_new))

#     while sum(X_g2 < -0.5) > 0:
#         X_g2_new = norm.rvs(2 * mu_2, sigma[1], size = sum(X_g2 < -0.5))
#         X_g2 = [x for x in X_g2 if x >= -0.5]
#         X_g2 = np.concatenate((X_g2, X_g2_new))

#     X = np.concatenate((X,X_l))
#     X = np.concatenate((X,X_g))
#     X = np.concatenate((X,X_g2))
#     X = np.round(X).astype(int)

#     return X

# N = 1000
# pi = [1/3, 1/3, 1/3]
# alpha = 0.5
# lambda_ = 2
# mu_2 = 20
# sigma = [3, 3]

# X_gen = theta_to_data(N, pi, alpha, lambda_, mu_2, sigma)
# print(SVMM(X_gen, True, False))

# test values 
# N_gen = 3000
# alpha_gen = 0.5
# lambda_gen = 3
# mu_gen = 30
# sigma_gen = 3
# pi_gen = [1/3, 1/3, 1/3]

# X = [0 for _ in range(int(pi_gen[0] * N_gen * alpha_gen))] # generating zeros
# X_l = geom.rvs(1/lambda_gen, size=(int(pi_gen[0] * N_gen * (1-alpha_gen)))) # generating observations following geometric 
# X.extend(X_l)

# X_g = norm.rvs(mu_gen, sigma_gen, size =(int(pi_gen[1] * N_gen))) # generating observations following normal
# X_g = [round(g) for g in X_g] # converting to integers
# X.extend(X_g)

# X_g2 = norm.rvs(2*mu_gen, sigma_gen, size =(int(pi_gen[2] * N_gen))) # generating observations following normal
# X_g2 = [round(g) for g in X_g2] # converting to integers
# X.extend(X_g2)

# X = np.array(X)

# with open('test_3mode.csv', 'w') as f:
#     writethis = csv.writer(f, delimiter =',')
#     writethis.writerows([X])

# SVMM(X, show_plot = True, show_distance = True)