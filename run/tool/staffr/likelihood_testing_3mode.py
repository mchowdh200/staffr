import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.cluster import KMeans as km
from sklearn.mixture import GaussianMixture as GMM
import sklearn.metrics as metrics
import random
import math

def plot_SVEM_3mode(alpha,pi,l,mu,sigma,r,N, show_affinity_plot):
    if math.isnan(pi[0]):
        X = [0 for _ in range(N)]
    else:
        X = [0 for _ in range(int(pi[0]*N*alpha))]
        if alpha != 1:
            X_l = stats.geom.rvs(1/l, size=(int(N*(1-alpha)*pi[0])))
            X.extend(X_l)
            X_g = np.random.normal(mu[0], sigma[0], int(pi[1]*N))
            X_g = [int(g) for g in X_g]
            X.extend(X_g)
            X_a = np.random.normal(2*mu[0], sigma[0], int(pi[2]*N))
            X_a = [int(a) for a in X_a]
            X.extend(X_a)

    hist_1, bin_edges_1 = np.histogram(X, bins=30, density=False)
    bin_center_1 = (bin_edges_1[:-1] + bin_edges_1[1:])/2

    plt.scatter(bin_center_1, hist_1, color = 'orange')

    if(show_affinity_plot):
        fig_1, ax_1 = plt.subplots(figsize = (6,6))
        ax_1.set_yscale("log")
        plt.xlabel('Affinity')
        plt.ylabel('Number of Samples')
        # TODO: move the plots side by side
        plt.hist(r[0], alpha = 0.5, label = 'first mode') 
        plt.hist(r[1], alpha = 0.5, label = 'second mode')
        plt.hist(r[2], alpha = 0.5, label = 'third mode')
        plt.legend(loc="upper left")
        plt.show()

def SVEM_3mode(data, plot_it, show_affinity_plot, ax): 
    def dist(a,b):
        return np.abs(b - a)
    N = len(data)
    f = np.ravel(data).astype(float) 
    f=f.reshape(-1,1) # reshaping data so built-in kmeans can process data
    kmeans = km(n_clusters=3)
    kmeans.fit(f)
    centers = np.sort(np.ravel(kmeans.cluster_centers_))
    mu = centers # initial guess for mu 
    mu = np.delete(mu, 0) # not needed for first mode
    sigma = [np.std(X)] # initial guess for standard deviations 
    pi = np.ones(3) * (1.0/3) # initial weight across 3 modes
    r = np.zeros([3,N]) # responsibility matrix for 3 modes
    ll_list = list() # log-likelihood list
    iteration = 0
    distance = 1 # distance between log-likelihoods
    alpha = np.count_nonzero(data == 0) / len(data) # intial guess for proportion of zeros that comprise data
    l = 2 # expected value for first mode hurdle model geometric

    while distance > (1 / (N*10)): # tolerance
        r_zero = [] # affinity for first mode 
        for i in range(len(data)): # for each data point
            if data[i]==0: 
                r_zero.append(alpha*pi[0]) # if the data point is a 0, set affinity to proportion of zeros times first mode weight 
            else: 
                geo = stats.geom.pmf(data[i],1/l) # otherwise, the data point is noise and its affinity is associated with
                                                  # its geometric pmf and expected value
                r_zero.append(geo*(1-alpha)*pi[0]) # the noise affinity is geometric pmf (w/ inverse expected value)
                                                   # * proportion of nonzero * first mode weight
                
        r[0] = r_zero # set responsibility matrix row associated with first mode to found affinities
        r[1] = pi[1] * stats.norm.pdf(x=data, loc=mu[0], scale=sigma[0]) # second mode responsibility is equal to
                                                                         # normal pdf (w/ mu_2, sigma_3)
        r[2] = pi[2] * stats.norm.pdf(x=data, loc=2*mu[0], scale=sigma[0]) # third mode responsibility is equal to 
                                                                         # normal pdf (w/ mu_3, sigma_3)

        # TODO: condense this to one for-loop
        # background affinity to prevent numerical errors
        for idx, r_i in enumerate(r[0]):
            if r_i== 0 or math.isnan(r_i):
                r_i[0][idx] = 1e-10
        for idx, r_i in enumerate(r[1]):
            if r_i == 0 or math.isnan(r_i):
                r[1][idx] = 1e-10
        for idx, r_i in enumerate(r[2]):
            if r_i == 0 or math.isnan(r_i):
                r[2][idx] = 1e-10
        r = r / np.sum(r, axis=0) # normalize responsibility matrix

        N_k = np.sum(r, axis=1) 
        r_sum = 0 
        for i in range(N): 
            if data[i] == 0: # I(x_i = 0) -> {1 if x_i = 0, 0 otherwise}; given that the indicator variable returns 1
                r_sum += r[0][i] # take the sum of the responsibilities for each point in the first mode
        alpha = r_sum / sum(r[0])   # take the proportion of the sum of responsibilites which have an associated x_i = 0 
                                    # over the sum of all first mode responsibilites
        l_sum = 0
        for i in range(N): 
            if data[i] != 0: # I(x_i != 0) -> {1 if x_i != 0, 0 otherwise}; given that the indicator variable returns 1
                l_sum += data[i] * r[0][i] # take the sum of responsibilities for each point in the first mode * each data point
        if alpha != 1: # if the hurdle model doesn't only have zero-inflation; if not all data points are zero
            l = l_sum / ((1-alpha)*sum(r[0])) # take the proportion of the sum of responsibilites for each point in first mode * each data point
                                              # which have an associated x_i = 0
                                              # over the proportion of non-zero elements * sum of all first mode responsibilites
                                              # as this equation is based on the MLE for p in Geo(p), and l = 1/p
        else:
            l = 1e-3 # lower-bound for expected value associated with geometric distribution

        # TODO: condense into one for-loop
        # MLEs for mean and standard deviations associated with normal distributions
        mu[0] = (np.sum(r[1] * data)) / N_k[1]  
        numerator = r[1] * (data - mu[0])**2
        sigma[0] = np.sqrt(np.sum(numerator) / N_k[1]) 
        if sigma[0] < 0.01: # lower-bound for standard deviation
            sigma[0] = 0.01
        mu[1] = (np.sum(r[2] * data)) / N_k[2]  
        numerator = r[2] * (data - mu[1])**2
        pi = N_k/N # update mixing coefficients / weights for each mode

        # TODO: what if there is no noise, only zero-inflation and normals? 

        hurdle = np.where(X == 0, pi[0] * alpha, pi[0] * (1-alpha) * stats.geom.pmf(X,1/l))
        gmm = pi[1] * stats.norm.pdf(X, mu[0], sigma[0]) + pi[2] * stats.norm.pdf(X, 2*mu[0], sigma[0])
        log_likelihood = np.sum(np.log(hurdle + gmm))
        ll_list.append(log_likelihood)

        iteration += 1
        if iteration > 1:
            distance = dist(ll_list[-1],ll_list[-2])
            print(distance)
    if(plot_it):
        plot_SVEM_3mode(alpha,pi,l,mu,sigma,r,N, show_affinity_plot)
    return alpha, l, ll_list

N = 1000
a = 0.7
l = 2

X = [0 for _ in range(int(N*a))] # zero-inflation
X_l = stats.geom.rvs(1/l, size=(int(N*(1-a)))) # geometric distribution (p = 1/l, size)
X.extend(X_l)

X_g = np.random.normal(30, 3, 1000) # normal distribution (mu, std, size)
X_g = [round(g) for g in X_g] # convert to integers, in line with structural variant data
X.extend(X_g)

X_g2 = np.random.normal(60, 3, 1000) # normal distribution (mu, std, size)
X_g2 = [round(g) for g in X_g2] # convert to integers, in line with structural variant data
X.extend(X_g2)

fig, ax = plt.subplots(figsize = (6,6))

ax.hist(X, bins = 25, alpha = 0.25)
ax.set_yscale("log")
ax.set_ylim(bottom=1)
ax.set_xlim(left=0)
plt.xlabel('Evidence Depth')
plt.ylabel('Number of Samples')

alpha, l, ll_list = SVEM_3mode(np.array(X), True, True, ax)