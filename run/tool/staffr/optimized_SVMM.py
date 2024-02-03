import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, geom

def plot_SVMM(X, n, pi, alpha, lambda_, mu, sigma):
    ax = plt.subplots(figsize = (6,6))[1]
    plt.hist(X, alpha = 0.5, bins = 20) # plot histogram of input data set

    X_hurdle = [0 for _ in range(n)]
    X_hurdle = [0 for _ in range(int(pi[0] * n * alpha))] # generate zeros based on estimated parameters
    X_hurdle.extend(geom.rvs(1/lambda_, size = int(n * (1-alpha) * pi[0]))) # generate geometric based on estimated parameters
    hist_hurdle, bin_hurdle = np.histogram(X_hurdle, bins = 30) # derive points and bin edges in histogram form
    plt.scatter(0.5 * (bin_hurdle[1:] + bin_hurdle[:-1]), hist_hurdle, color = 'orange') # plot points against bin centers

    X_normal1 = norm.rvs(mu, sigma, size = int(pi[1] * n)) # generate normal based on estimated parameters
    hist_normal1, bin_normal1 = np.histogram(X_normal1, 30) # derive points and bin edges in histogram form
    plt.scatter(0.5 * (bin_normal1[1:] + bin_normal1[:-1]), hist_normal1, color = 'green') # plot points against bin centers

    ax.set_yscale("log") 
    ax.set_ylim(bottom=1)
    ax.set_xlim(left=0)
    plt.xlabel('Evidence Depth')
    plt.ylabel('Number of Samples')
    plt.show()

def SVMM(X, show_plot, show_distance):

    # TODO: fine-tune initial paramters, put in separate function
    # initial values for parameters
    n = np.shape(X)[0] # length of data set
    pi = [1.0/2 for _ in range(2)] # mixing coefficients
    r = np.zeros([2,n]) # responsibility matrix
    alpha = np.sum(X == 0) / n # probability of an observation belonging to zero component
    lambda_ = n / np.sum(X != 0) # expected value of the geometric distribution
    sigma = np.std(X) # standard deviation for normal distribution
    mu = np.mean(X) # mean for normal distribution 
    log_likelihoods = [] 
    iteration = 0
    distance = 1

    while distance > (1/(n*10)): # convergence criterion

        # expectation
        r[0][X==0] = pi[0] * alpha # responsibility of zero values to first mode
        r[0][X!=0] = pi[0] * (1-alpha) * geom.pmf(X[X!=0], 1/lambda_) # responsibility of nonzero values to first component
        r[1] = pi[1] * norm.pdf(X, mu, sigma) # responsibility of each value to second component
        r = r / np.sum(r, axis = 0) # normalization

        # maximization
        pi = np.sum(r, axis = 1) / n # total responsibility of a component divided by total # of observations
        alpha = np.sum(r[0][X==0]) / np.sum(r[0]) # MLE for 
        lambda_ = np.dot(X[X != 0], r[0][X != 0]) / ((1-alpha)*np.sum(r[0])) # reciprocal of MLE for p in geometric distribution
        mu = np.average(X, weights = r[1]) # MLE for mean in normal distribution
        sigma = np.average((X-mu)**2, weights=r[1])**.5 # MLE for standard deviation in normal distribution

        # score
        hurdle = np.where(X == 0, pi[0] * alpha, pi[0] * (1-alpha) * geom.pmf(X,1/lambda_)) #  likelihood of each observation in hurdle model
        gmm = pi[1] * norm.pdf(X, mu, sigma) # likelihood of each observation in normal distribution
        log_likelihood = np.sum(np.log(hurdle + gmm)) # sum of log of likelihood of each observation
        log_likelihoods.append(log_likelihood) 

        iteration += 1
        if iteration > 1:
            distance = np.abs(log_likelihoods[-2]-log_likelihoods[-1]) # magnitude of difference between each 
            if(show_distance):
                print(distance)

    if(show_plot):
        plot_SVMM(X, n, pi, alpha, lambda_, mu, sigma)

    return pi[0], pi[1], alpha, lambda_, mu, sigma

# # test values 
# N_gen = 3000
# alpha_gen = 0.5
# lambda_gen = 3
# mu_gen = 30
# sigma_gen = 3
# pi_gen = [1/2, 1/2]

# X = [0 for _ in range(int(pi_gen[0] * N_gen * alpha_gen))] # generating zeros
# X_l = geom.rvs(1/lambda_gen, size=(int(pi_gen[0] * N_gen * (1-alpha_gen)))) # generating observations following geometric 
# X.extend(X_l)

# X_g = norm.rvs(mu_gen, sigma_gen, size =(int(pi_gen[1] * N_gen))) # generating observations following normal
# X_g = [round(g) for g in X_g] # converting to integers
# X.extend(X_g)

# X = np.array(X)
# # with open('test.csv', 'w') as f:
# #     writethis = csv.writer(f, delimiter =',')
# #     writethis.writerows([X])
# SVMM(X, show_plot = True, show_distance = True)