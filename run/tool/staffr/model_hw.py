import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, geom, expon, multinomial
from sklearn.cluster import KMeans as km
import csv

def plot_model_hw(X, n, pi, mu, sigma):
    ax = plt.subplots(figsize = (6,6))[1]
    plt.hist(X, alpha = 0.20, bins = max(X), color = 'grey', edgecolor = 'white', linewidth = 3) # plot histogram of input data set

    curve2 = np.linspace(mu[0] - 10 * sigma[0], mu[0] + 10 * sigma[0], 1000)
    curve3 = np.linspace(mu[1] - 10 * sigma[1], mu[1] + 10 * sigma[1], 1000)
    plt.plot(curve2, n * norm.pdf(curve2, mu[0], sigma[0]) * pi[1], linewidth = 3, color='red')
    plt.plot(curve3, n * norm.pdf(curve3, mu[1], sigma[1]) * pi[2], linewidth = 3, color='red')
    plt.scatter(0, n * pi[0], color = 'red')

    ax.set_yscale("log") 
    ax.set_ylim(bottom=1)
    ax.set_xlim(left=0, right = max(X)+1)
    plt.xlabel('Evidence Depth')
    plt.ylabel('Number of Samples')
    plt.show()

def model_hw(X, show_plot, show_distance):

    # TODO: fine-tune initial paramters, put in separate function
    # initial values for parameters
    n = np.shape(X)[0] # length of data set
    pi = [1.0/3 for _ in range(3)] # mixing coefficients
    r = np.zeros([3,n]) # responsibility matrix
    sigma = [np.std(X), np.std(X)] # standard deviation for normal distribution
    f = np.ravel(X).astype(float)
    f=f.reshape(-1,1)
    kmeans = km(n_init='auto',n_clusters=3)
    kmeans.fit(f)
    centers = np.sort(np.ravel(kmeans.cluster_centers_))
    centers = np.delete(centers, 0)
    mu = centers
    log_likelihoods = [] 
    distances = []
    iteration = 0
    distance = 1

    while distance > (1/(n*10)): # convergence criterion

        # expectation
        r[0][X==0] = pi[0] 
        r[0][X!=0] = 0
        r[1] = pi[1] * norm.pdf(X, mu[0], sigma[0]) # responsibility of each value to second component
        r[2] = pi[2] * norm.pdf(X, mu[1], sigma[1]) # responsibility of each value to second component
        r = r / np.sum(r, axis = 0) # normalization

        # maximization
        pi = np.sum(r, axis = 1) / n # total responsibility of a component divided by total # of observations
        mu[0] = np.average(X, weights = r[1]) # MLE for mean in normal distribution
        mu[1] = np.average(X, weights = r[2]) # MLE for mean in normal distribution
        mu[0] = (mu[0] + 0.5*mu[1])/2
        mu[1] = 2*mu[0]
        sigma[0] = np.average((X-mu[0])**2, weights=r[1])**.5 # MLE for standard deviation in normal distribution
        sigma[1] = np.average((X-mu[1])**2, weights=r[2])**.5 # MLE for standard deviation in normal distribution

        # score
        hurdle = np.where(X == 0, pi[0], 0) #  likelihood of each observation in hurdle model
        gmm = pi[1] * norm.pdf(X, mu[0], sigma[0]) +  pi[2] * norm.pdf(X, mu[1], sigma[1])# likelihood of each observation in normal distribution
        log_likelihood = np.sum(np.log(hurdle + gmm)) # sum of log of likelihood of each observation
        log_likelihoods.append(log_likelihood) 

        iteration += 1
        if iteration > 1:
            distance = np.abs(log_likelihoods[-2]-log_likelihoods[-1]) # magnitude of difference between each 
            distances.append(distance)
            # if iteration > 2:
            #     if np.abs(distances[-2]-distances[-1]) < 1e-5:
            #         return pi, mu, sigma, log_likelihoods[-1]
            if(show_distance):
                print(distance)

    if(show_plot):
        plot_model_hw(X, n, pi, mu, sigma)

    return pi, mu, sigma