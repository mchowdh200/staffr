import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, geom, expon
from sklearn.cluster import KMeans as km
import csv

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

def SVMM(X, true_lambda, show_plot, show_distance):

    # TODO: fine-tune initial paramters, put in separate function
    # initial values for parameters
    n = np.shape(X)[0] # length of data set
    pi = [1.0/3 for _ in range(3)] # mixing coefficients
    r = np.zeros([3,n]) # responsibility matrix
    alpha = np.sum(X == 0) / n # probability of an observation belonging to zero component
    lambda_ = true_lambda # expected value of the geometric distribution
    sigma = [np.std(X), np.std(X)] # standard deviation for normal distribution
    f = np.ravel(X).astype(float)
    f=f.reshape(-1,1)
    kmeans = km(n_clusters=3)
    kmeans.fit(f)
    centers = np.sort(np.ravel(kmeans.cluster_centers_))
    centers = np.delete(centers, 0)
    mu = centers
    log_likelihoods = [] 
    iteration = 0
    distance = 1

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