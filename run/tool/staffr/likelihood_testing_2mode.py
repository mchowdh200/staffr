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

def plot_SVEM_2mode(alpha,pi,l,mu,sigma,r,N, show_affinity_plot, ax):
    if math.isnan(pi[0]):
        X = [0 for _ in range(N)]
    else:
        X = [0 for _ in range(int(pi[0]*N*alpha))]
        if alpha != 1:
            X_l = stats.geom.rvs(1/l, size=(int(N*(1-alpha)*pi[0])))
            X.extend(X_l)
            X_g = np.random.normal(mu[0], sigma[0], int(pi[1]*N))
            X_g = [round(g) for g in X_g]
            X.extend(X_g)

    hist_1, bin_edges_1 = np.histogram(X, bins=20, density=False)
    bin_center_1 = (bin_edges_1[:-1] + bin_edges_1[1:])/2
    plt.scatter(bin_center_1, hist_1, color = 'orange')
    
    if(show_affinity_plot):
        fig_1, ax_1 = plt.subplots(figsize = (6,6))
        ax_1.set_yscale("log")
        plt.xlabel('Affinity')
        plt.ylabel('Number of Samples')
        plt.hist(r[0], alpha = 0.5)
        plt.hist(r[1], alpha = 0.5)
        plt.show()

def SVEM_2mode(data, plot_it, show_affinity_plot, ax): 
    def dist(a,b):
        return np.abs(b - a)
    N = len(data)
    f = np.ravel(data).astype(float)
    f=f.reshape(-1,1)
    kmeans = km(n_clusters=2)
    kmeans.fit(f)
    centers = np.sort(np.ravel(kmeans.cluster_centers_))
    mu = centers
    mu = np.delete(mu, 0)
    sigma = [np.std(data)]
    pi = np.ones(2) * (1.0/2) 
    r = np.zeros([2,N]) 
    ll_list = list()
    iteration = 0
    distance = 1
    num_zeros = np.count_nonzero(data == 0)
    alpha = num_zeros / len(data) 
    l = 2

    while distance > (1 / (N*10)): 
        r_zero = []
        for i in range(len(data)):
            if data[i]==0:
                r_zero.append(alpha*pi[0])
            else:
                geo = stats.geom.pmf(data[i],1/l)
                r_zero.append(geo*(1-alpha)*pi[0])
                
        r[0] = r_zero
        r[1] = pi[1] * stats.norm.pdf(x=data, loc=mu[0], scale=sigma[0]) 
        for idx, r_i in enumerate(r[0]):
            if r_i == 0 or math.isnan(r_i):
                r[0][idx] = 1e-10
        for idx, r_i in enumerate(r[1]):
            if r_i == 0 or math.isnan(r_i):
                r[1][idx] = 1e-10
        r = r / np.sum(r, axis=0) 

        N_k = np.sum(r, axis=1)
        r_sum = 0
        for i in range(N):
            if data[i] == 0:
                r_sum += r[0][i]
        alpha = r_sum / sum(r[0])
        l_sum = 0
        for i in range(N):
            if data[i] != 0:
                l_sum += data[i] * r[0][i]
        if alpha != 1:
            l = l_sum / ((1-alpha)*sum(r[0]))
        else:
            l = 1e-3
        mu[0] = (np.sum(r[1] * data)) / N_k[1]  
        numerator = r[1] * (data - mu[0])**2
        sigma[0] = np.sqrt(np.sum(numerator) / N_k[1]) 
        if sigma[0] < 0.01:
            sigma[0] = 0.01
        pi = N_k/N

        likelihood_zero = (pi[0] * alpha) + (pi[1] * stats.norm.pdf(x = 0, loc = mu[0], scale = sigma[0]))
        likelihood = [likelihood_zero for _ in range(num_zeros)]
        for i in np.nonzero(data)[0]:
            nonzero_hurdle_likelihood = pi[0] * (1-alpha) * stats.geom.pmf(data[i], 1/l)
            likelihood.append(nonzero_hurdle_likelihood + (pi[1] * stats.norm.pdf(x = data[i], loc = mu[0], scale = sigma[0])))
        log_likelihood = np.sum(np.log(likelihood))
        ll_list.append(log_likelihood)

        # hurdle = np.where(data == 0, pi[0] * alpha, pi[0] * (1-alpha) * stats.geom.pmf(data, 1/l))
        # gmm = pi[1] * stats.norm.pdf(data, mu[0], sigma[0])
        # log_likelihood = np.log(hurdle + gmm)
        # log_likelihood = np.sum(log_likelihood)
        # ll_list.append(log_likelihood)

        iteration += 1
        print(iteration)
        if iteration > 1:
            distance = dist(ll_list[-1],ll_list[-2])
        print(distance)
    if(plot_it):
        plot_SVEM_2mode(alpha,pi,l,mu,sigma,r,N, show_affinity_plot, ax)
    return alpha, l, ll_list

N = 1000
a = 0.5
l = 2

X = [0 for _ in range(int(N*a))] # zero-inflation
X_l = stats.geom.rvs(1/l, size=(int(N*(1-a)))) # geometric distribution (p = 1/l, size)
X.extend(X_l)

X_g = np.random.normal(30, 3, 1000) # normal distribution (mu, std, size)
X_g = [round(g) for g in X_g] # convert to integers, in line with structural variant data
X.extend(X_g)

fig, ax = plt.subplots(figsize = (6,6))

ax.hist(X, bins = 25, alpha = 0.25)
ax.set_yscale("log")
ax.set_ylim(bottom=1)
ax.set_xlim(left=0)
plt.xlabel('Evidence Depth')
plt.ylabel('Number of Samples')

alpha, l, ll_list = SVEM_2mode(np.array(X), True, True, ax)

# ps = []
# for i in range(5000):
#     gaussian_numbers = np.random.normal(0, 1, size=1000)
#     gaussian_numbers2 = np.random.normal(0, 1, size=1000)
#     t, p = stats.ttest_ind(gaussian_numbers, gaussian_numbers2, equal_var=True)
#     ps.append(p)