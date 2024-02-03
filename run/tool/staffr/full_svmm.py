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

def plot_SVEM(rhat,pi,l,mu,sigma,r,N):
    if math.isnan(pi[0]):
        X = [0 for _ in range(N)]
    else:
        X = [0 for _ in range(int(pi[0]*N*rhat))]
        if rhat != 1:
            X_l = stats.geom.rvs(1/l, size=(int(N*(1-rhat)*pi[0])))
            X.extend(X_l)
            X_g = np.random.normal(mu[0], sigma[0], int(pi[1]*N))
            X_g = [int(g) for g in X_g]
            X.extend(X_g)
            X_a = np.random.normal(mu[1], sigma[1], int(pi[2]*N))
            X_a = [int(a) for a in X_a]
            X.extend(X_a)

    hist_1, bin_edges_1 = np.histogram(X, bins=30, density=False)
    bin_center_1 = (bin_edges_1[:-1] + bin_edges_1[1:])/2
    fig, ax = plt.subplots(figsize = (6,6))

    plt.hist(X, bins = 20, alpha = 0.25, rwidth = 1)   
    plt.scatter(bin_center_1, hist_1, color = 'orange')
    ax.set_yscale("log")
    ax.set_ylim(bottom=1)
    ax.set_xlim(left=0)
    plt.xlabel('Evidence Depth')
    plt.ylabel('Number of Samples')
    plt.show()
    fig_1, ax_1 = plt.subplots(figsize = (6,6))
    ax_1.set_yscale("log")
    plt.xlabel('Affinity')
    plt.ylabel('Number of Samples')
    # TODO: move the plots side by side
    plt.hist(r[0], alpha = 0.5) 
    plt.hist(r[1], alpha = 0.5)
    plt.hist(r[2], alpha = 0.5)
    plt.show()

def SVEM(data): 
    def dist(a,b):
        return np.abs(b - a)
    N = len(data)
    f = np.ravel(data).astype(float)
    f=f.reshape(-1,1)
    kmeans = km(n_clusters=3)
    kmeans.fit(f)
    centers = np.sort(np.ravel(kmeans.cluster_centers_))
    mu = centers
    mu = np.delete(mu, 0)
    sigma = [np.std(data), np.std(data)]
    pi = np.ones(3) * (1.0/3) 
    r = np.zeros([3,N]) 
    ll_list = list()
    iteration = 0
    distance = 1
    rhat = np.count_nonzero(data == 0) / len(data) 
    l = 2

    while distance > (1 / (N*10)): 
        r[0][X==0] = pi[0] * rhat # responsibility of zero values to first mode
        r[0][X!=0] = pi[0] * (1-rhat) * stats.geom.pmf(X[X!=0], 1/l) # responsibility of nonzero values to first component
        r[1] = pi[1] * stats.norm.pdf(data, mu[0], sigma[0]) # responsibility of each value to second component
        r[2] = pi[2] * stats.norm.pdf(data, mu[1], sigma[1])
        # TODO: condense this to one for-loop
        for idx, r_i in enumerate(r[0]):
            if r_i == 0 or math.isnan(r_i):
                r[0][idx] = 1e-10
        for idx, r_i in enumerate(r[1]):
            if r_i == 0 or math.isnan(r_i):
                r[1][idx] = 1e-10
        for idx, r_i in enumerate(r[2]):
            if r_i == 0 or math.isnan(r_i):
                r[2][idx] = 1e-10
        r = r / np.sum(r, axis=0) 

        N_k = np.sum(r, axis=1)
        r_sum = 0
        for i in range(N):
            if data[i] == 0:
                r_sum += r[0][i]
        rhat = r_sum / sum(r[0])
        l_sum = 0
        for i in range(N):
            if data[i] != 0:
                l_sum += data[i] * r[0][i]
        if rhat != 1:
            l = l_sum / ((1-rhat)*sum(r[0]))
        else:
            l = 1e-3
        # TODO: condense into one for-loop
        mu[0] = (np.sum(r[1] * data)) / N_k[1]  
        numerator = r[1] * (data - mu[0])**2
        sigma[0] = np.sqrt(np.sum(numerator) / N_k[1]) 
        if sigma[0] < 0.01:
            sigma[0] = 0.01
        mu[1] = (np.sum(r[2] * data)) / N_k[2]  
        numerator = r[2] * (data - mu[1])**2
        sigma[1] = np.sqrt(np.sum(numerator) / N_k[2]) 
        if sigma[1] < 0.01:
            sigma[1] = 0.01
        pi = N_k/N

        hurdle = np.where(X == 0, pi[0] * rhat, pi[0] * (1-rhat) * stats.geom.pmf(data,1/l)) #  likelihood of each observation in hurdle model
        gmm = pi[1] * stats.norm.pdf(X, mu[0], sigma[0]) + pi[2] * stats.norm.pdf(X, mu[1], sigma[1]) # likelihood of each observation in normal distribution
        log_likelihood = np.sum(np.log(hurdle + gmm)) # sum of log of likelihood of each observation
        ll_list.append(log_likelihood)

        iteration += 1
        if iteration > 1:
            distance = dist(ll_list[-1],ll_list[-2])
    
    plot_SVEM(rhat,pi,l,mu,sigma,r,N)
    return rhat, l, ll_list

N = 1000
a = 0.7
l = 2

X = [0 for _ in range(int(N*a))] # zero-inflation
# X_l = stats.geom.rvs(1/l, size=(int(N*(1-a)))) # geometric distribution (p = 1/l, size)
# X.extend(X_l)

# X_g = np.random.normal(30, 3, 1000) # normal distribution (mu, std, size)
# X_g = [round(g) for g in X_g] # convert to integers, in line with structural variant data
# X.extend(X_g)

# X_g2 = np.random.normal(60, 3, 1000) # normal distribution (mu, std, size)
# X_g2 = [round(g) for g in X_g2] # convert to integers, in line with structural variant data
# X.extend(X_g2)

rhat, l, ll_list = SVEM(np.array(X))

# 1. make a function that estimates 3-mode model, coupled modes 2 and 3, independent variance
# 2. redo the p-value algo but with new model
# - rerun old p-value analysis and check that it's similar
# 3. apply to Ryan's data
# a) pick some data (x10) (to show Ryan)
# - fit model
# - make figure showing data and fit
# - annotate figure with estimated params and p-value
# b) run code on all data
# - make plot of q vs mu_2 scatter plot 