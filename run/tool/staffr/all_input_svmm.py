import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, geom, expon, multinomial
from sklearn.cluster import KMeans as km
import csv
from pandas import read_csv

def plot_SVMM(X, n, pi, alpha, lambda_, mu, sigma):
    ax = plt.subplots(figsize = (6,6))[1]
    plt.hist(X, alpha = 0.20, bins = max(X), color = 'grey', edgecolor = 'white', linewidth = 3) # plot histogram of input data set

    curve = np.linspace(mu - 10 * sigma[0], mu + 10 * sigma[0], 1000)
    curve2 = np.linspace(1, 10 * lambda_, 1000)
    plt.plot(curve, n * norm.pdf(curve, mu, sigma[0]) * pi[1], linewidth = 3, color='red')
    plt.plot(curve2, n * expon.pdf(curve2, loc = 0, scale = lambda_) * pi[0], linewidth = 3, color = 'red')
    plt.scatter(0, n * alpha * pi[0], color = 'red')

    ax.set_yscale("log") 
    ax.set_ylim(bottom=1)
    ax.set_xlim(left=0, right = max(X)+1)
    plt.xlabel('Evidence Depth')
    plt.ylabel('Number of Samples')
    plt.show()

def SVMM(X, show_plot, show_distance):

    # TODO: fine-tune initial paramters, put in separate function
    # initial values for parameters
    n = np.shape(X)[0] # length of data set
    pi = [1.0/3 for _ in range(3)] # mixing coefficients
    r = np.zeros([3,n]) # responsibility matrix
    alpha = np.sum(float(input("Enter true alpha, or leave blank: ") or (np.sum(X == 0) / n))) # probability of an observation belonging to zero component
    lambda_ = np.sum(int(input("Enter true lambda, or leave blank: ") or (n / np.sum(X != 0)))) # expected value of the geometric distribution
    sigma_1 = np.sum(float(input("Enter true sigma_1, or leave blank: ") or (np.std(X)))) # standard deviation for normal distribution
    sigma_2 = np.sum(float(input("Enter true sigma_2, or leave blank: ") or (np.std(X))))
    sigma = [sigma_1, sigma_2]
    f = np.ravel(X).astype(float)
    f=f.reshape(-1,1)
    kmeans = km(n_clusters=3)
    kmeans.fit(f)
    centers = np.sort(np.ravel(kmeans.cluster_centers_))
    centers = np.delete(centers, 0)
    mu_2 = np.sum(float(input("Enter true mu_2, or leave blank: ") or (centers[0])))
    mu = [mu_2, 2*mu_2]
    log_likelihoods = [] 
    iteration = 0
    distance = 1
    print(alpha, lambda_, sigma, mu)
    while distance > (1/(n*10)): # convergence criterion

        # expectation
        r[0][X==0] = pi[0] * alpha # responsibility of zero values to first mode
        r[0][X!=0] = pi[0] * (1-alpha) * geom.pmf(X[X!=0], 1/lambda_) # responsibility of nonzero values to first component
        r[1] = pi[1] * norm.pdf(X, mu[0], sigma[0]) # responsibility of each value to second component
        r[2] = pi[2] * norm.pdf(X, mu[1], sigma[1]) # responsibility of each value to second component
        r = r / np.sum(r, axis = 0) # normalization

        # maximization
        pi = np.sum(r, axis = 1) / n # total responsibility of a component divided by total # of observations
        alpha = np.sum(r[0][X==0]) / np.sum(r[0]) # MLE for 
        lambda_ = np.dot(X[X != 0], r[0][X != 0]) / ((1-alpha)*np.sum(r[0])) # reciprocal of MLE for p in geometric distribution
        mu[0] = np.average(X, weights = r[1]) # MLE for mean in normal distribution
        mu[1] = np.average(X, weights = r[2]) # MLE for mean in normal distribution
        sigma[0] = np.average((X-mu[0])**2, weights=r[1])**.5 # MLE for standard deviation in normal distribution
        sigma[1] = np.average((X-mu[1])**2, weights=r[2])**.5 # MLE for standard deviation in normal distribution

        # score
        hurdle = np.where(X == 0, pi[0] * alpha, pi[0] * (1-alpha) * geom.pmf(X,1/lambda_)) #  likelihood of each observation in hurdle model
        gmm = pi[1] * norm.pdf(X, mu[0], sigma[0]) +  pi[2] * norm.pdf(X, mu[1], sigma[1])# likelihood of each observation in normal distribution
        log_likelihood = np.sum(np.log(hurdle + gmm)) # sum of log of likelihood of each observation
        log_likelihoods.append(log_likelihood) 

        iteration += 1
        if iteration > 1:
            distance = np.abs(log_likelihoods[-2]-log_likelihoods[-1]) # magnitude of difference between each 
            if(show_distance):
                print(distance)

    if(show_plot):
        plot_SVMM(X, n, pi, alpha, lambda_, mu, sigma)

    return pi, alpha, lambda_, mu, sigma

def theta_to_data(N, pi, alpha, lambda_, mu_2, sigma):

    draws = multinomial.rvs(n = N, p = pi)
    draws_alpha = multinomial.rvs(n = draws[0], p = [alpha, 1 - alpha])

    X = [0 for _ in range(draws_alpha[0])]
    X_l = geom.rvs(1/lambda_, size = draws_alpha[1])
    X_g = norm.rvs(mu_2, sigma[0], size = draws[1])
    X_g2 = norm.rvs(2 * mu_2, sigma[1], size = draws[2])

    while sum(X_g < -0.5) > 0:
        X_g_new = norm.rvs(mu_2, sigma[0], size = sum(X_g < -0.5))
        X_g = [x for x in X_g if x >= -0.5]
        X_g = np.concatenate((X_g, X_g_new))

    while sum(X_g2 < -0.5) > 0:
        X_g2_new = norm.rvs(2 * mu_2, sigma[1], size = sum(X_g2 < -0.5))
        X_g2 = [x for x in X_g2 if x >= -0.5]
        X_g2 = np.concatenate((X_g2, X_g2_new))

    X = np.concatenate((X,X_l))
    X = np.concatenate((X,X_g))
    X = np.concatenate((X,X_g2))
    X = np.round(X).astype(int)

    return X

N = 1000
pi = [1/3, 1/3, 1/3]
alpha = 0.5
lambda_ = 2
mu_2 = 5
sigma = [5, 5]

X_gen = theta_to_data(N, pi, alpha, lambda_, mu_2, sigma)
print(SVMM(X_gen, True, False))

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