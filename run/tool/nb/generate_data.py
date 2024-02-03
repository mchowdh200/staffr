import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.cluster import KMeans as km
from sklearn.mixture import GaussianMixture as GMM
import sklearn.metrics as metrics

# Plotting the data according to the mean, standard deviation, weight, and if the histogram should be plotted
# - the function checks a boolean to see if the data should be plotted on a histogram for comparison purposes
# - the output plot will be a lineplot, with the first mode being blue (the lowest mean being zero)
# - the third mode will be green, have the highest mean as the mean should be non-negative and twice the mean of the second mode
# - the second mode will be orange, and its mean will be between the other two
# - the x-values, being the curve, will be an evenly-spaced interval of 100 numbers
# - the lower-bound of the x-value interval will be 3 standard deviations below the mean
# - the upper-bound of the x-value interval will be 3 standard deviations above the mean
# - from the curve, we get our resulting lineplot by inputting the curve values into the PDF of the Normal Distribution
# - these values from the Normal PDF are then normalized according to the weight input from the GMM 

def plot_distributions(data, mu, sigma, pi, hist):
    if hist == True: # histogram underlay
        plt.hist(data, bins=30, density = True, alpha = 0.5)
    color = ''
    for k in range(3):
        if mu[k] == min(mu): # first mode
            color = 'blue'
        elif mu[k] == max(mu): # third mode
            color = 'green'
        else: # second mode 
            color = 'orange'
        curve = np.linspace(mu[k] - 3*sigma[k], mu[k] + 3*sigma[k], 100) # 100 numbers surrounding mean, bounded by 3 * std
        plt.plot(curve, pi[k]*stats.norm.pdf(curve, mu[k], sigma[k]), color) # Normal PDF values scaled by weight

def dist(a,b):
    return np.abs(b - a)

data = []
bins = 30
N = 1000
N_0 = int(N/6)
N_1 = int(N/6)
N_2 = N - (N_0+N_1)

mu = 20
sigma = 5
mode_1 = np.random.normal(0, sigma, N_0)
mode_2 = np.random.normal(mu, sigma, N_1)
mode_3 = np.random.normal(2*mu, sigma, N_2)

data.extend(mode_1)
data.extend(mode_2)
data.extend(mode_3)

plt.hist(mode_1, bins = bins, color = 'blue')
plt.hist(mode_2, bins = bins, color = 'orange')
plt.hist(mode_3, bins = bins, color = 'green')
plt.show()

plt.hist(data, bins = bins, density = True)
plt.show()

# TODO: potentially find better inital guess method?
# First Step: k-means clustering

k = 3 # number of modes
f = np.ravel(data).astype(float) # prepping data for built-in k-means
f = f.reshape(-1,1) 
kmeans = km(n_init = 3, n_clusters=3)
kmeans.fit(f) 
centers = np.sort(np.ravel(kmeans.cluster_centers_)) # storing centroids
mu = centers # initial mu values
sigma = [np.std(data), np.std(data), np.std(data)] # initial std values
pi = np.ones(3) * (1.0/3) # mixing coefficients
r = np.zeros([3,N]) # responsibilities
ll_list = [] # log-likelihood list
iteration = 0 
distance = 1

while distance > 1e-6: # convergence criterion

    # Expectation Step
    for k in range(3):
        # set responsibility of each point to PDF using estimated parameters, scaled by the mixing coefficient
        r[k,:] = pi[k] * stats.norm.pdf(x=data, loc=mu[k], scale=sigma[k])
    r = r / np.sum(r, axis=0) # normalize the data 
        

    # Maximization Step
    N_k = np.sum(r, axis=1) # total responsibility for each mode
    for k in range(3):
        mu[k] = np.sum(r[k,:] * data) / N_k[k] # update mean
        numerator = r[k] * (data - mu[k])**2
        sigma[k] = np.sqrt(np.sum(numerator) / N_k[k]) # update std
    pi = N_k/N # update mixing coefficient
        
    # Likelihood Calculation
    likelihood = 0.0
    for k in range(3):
        likelihood += pi[k] * stats.norm.pdf(x=data, loc=mu[k], scale=sigma[k])
    ll_list.append(np.sum(np.log(likelihood)))
    
    iteration += 1
    if iteration > 1: # as soon as there are 2 points to compare
        distance = dist(ll_list[-1],ll_list[-2]) # take distance

print('weights', np.sort(np.ravel(pi)))
print('means', np.sort(np.ravel(mu)))
print('stds', np.sort(np.ravel(sigma)))
plot_distributions(data, mu, sigma, pi, True)