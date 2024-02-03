from scipy.stats import norm, multinomial
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.cluster import KMeans as km

def plot_SVMM_gmm(X, n, pi, mu, sigma):
    ax = plt.subplots(figsize = (6,6))[1]
    plt.hist(X, alpha = 0.20, bins = max(X), color = 'grey', edgecolor = 'white', linewidth = 3) # plot histogram of input data set

    curve2 = np.linspace(mu[0] - 10 * sigma[0], mu[0] + 10 * sigma[0], 1000)
    curve3 = np.linspace(mu[1] - 10 * sigma[1], mu[1] + 10 * sigma[1], 1000)
    plt.plot(curve2, n * norm.pdf(curve2, mu[0], sigma[0]) * pi[0], linewidth = 3, color='red')
    plt.plot(curve3, n * norm.pdf(curve3, mu[1], sigma[1]) * pi[1], linewidth = 3, color='red')

    ax.set_yscale("log") 
    ax.set_ylim(bottom=1)
    ax.set_xlim(left=0, right = max(X)+1)
    plt.xlabel('Evidence Depth')
    plt.ylabel('Number of Samples')
    plt.show()

def SVMM_gmm(X, show_plot, show_distance):

    # TODO: fine-tune initial paramters, put in separate function
    # initial values for parameters
    n = np.shape(X)[0] # length of data set
    pi = [1.0/2 for _ in range(2)] # mixing coefficients
    r = np.zeros([2,n]) # responsibility matrix
    sigma = [np.std(X), np.std(X)] # standard deviation for normal distribution
    f = np.ravel(X).astype(float)
    f=f.reshape(-1,1)
    kmeans = km(n_clusters=2)
    kmeans.fit(f)
    centers = np.sort(np.ravel(kmeans.cluster_centers_))
    # mu = [max(X)/3.0, 2.0*max(X)/3.0]
    mu = centers
    log_likelihoods = [] 
    iteration = 0
    distance = 1

    while distance > (1/(n*10)): # convergence criterion

        # expectation
        r[0] = pi[0] * norm.pdf(X, mu[0], sigma[0]) # responsibility of each value to second component
        r[1] = pi[1] * norm.pdf(X, mu[1], sigma[1]) # responsibility of each value to second component
        r = r / np.sum(r, axis = 0) # normalization
        
        # maximization
        pi = np.sum(r, axis = 1) / n # total responsibility of a component divided by total # of observations

        # mu_sum = 0
        # for i in range(n):
        #     mu_sum += (X[i] * r[0][i]) + ((X[i]- mu[0]) * r[1][i])

        # mu[0] = (1/((n))) * mu_sum
        mu[0] = (mu[0] + 0.5*mu[1])/2
        mu[1] = 2 * mu[0] # MLE for mean in normal distribution

        # sigma_sum = 0
        # for i in range(n):
        #     sigma_sum += ((X[i] - mu[0])**2) * r[0][i]

        # sigma[0] = (1/n) * sigma_sum

        # sigma_sum = 0
        # for i in range(n):
        #     sigma_sum += ((X[i] - mu[1])**2) * r[1][i]
        # sigma[0] = np.average((X-mu[0])**2, weights=r[1])**.5
        # sigma[1] = (1/n) * sigma_sum # MLE for standard deviation in normal distribution
        sigma[0] = np.average((X-mu[0])**2, weights=r[0])**.5 # MLE for standard deviation in normal distribution
        sigma[1] = np.average((X-mu[1])**2, weights=r[1])**.5
        sigma[0] = (sigma[0] + sigma[1]) / 2.0
        sigma[1] = sigma[0]
        # score
        gmm = pi[0] * norm.pdf(X, mu[0], sigma[0]) +  pi[1] * norm.pdf(X, mu[1], sigma[1])# likelihood of each observation in normal distribution
        log_likelihood = np.sum(np.log(gmm)) # sum of log of likelihood of each observation
        log_likelihoods.append(log_likelihood) 

        iteration += 1
        if iteration > 1:
            distance = np.abs(log_likelihoods[-2]-log_likelihoods[-1]) # magnitude of difference between each 
            if(show_distance):
                print(distance)

    if(show_plot):
        plot_SVMM_gmm(X, n, pi, mu, sigma)

    return pi, mu, sigma

def theta_to_data_nm1(N, pi, mu_2, sigma):
# without first mode
    draws = multinomial.rvs(n = N, p = pi)
    
    X = []
    X_g = norm.rvs(mu_2, sigma[0], size = draws[0])
    X_g2 = norm.rvs(2 * mu_2, sigma[1], size = draws[1])

    while sum(X_g < -0.5) > 0:
        X_g_new = norm.rvs(mu_2, sigma[0], size = sum(X_g < -0.5))
        X_g = [x for x in X_g if x >= -0.5]
        X_g = np.concatenate((X_g, X_g_new))

    while sum(X_g2 < -0.5) > 0:
        X_g2_new = norm.rvs(1.1 * mu_2, sigma[1], size = sum(X_g2 < -0.5))
        X_g2 = [x for x in X_g2 if x >= -0.5]
        X_g2 = np.concatenate((X_g2, X_g2_new))

    X = np.concatenate((X,X_g))
    X = np.concatenate((X,X_g2))
    X = np.round(X).astype(int)
    return X

N = 1000
pi = [1/2,1/2]
sigma = [5, 5]

for mu_2 in range(5,10):
    print('mu2:',mu_2)
    X_gen = theta_to_data_nm1(N, pi, mu_2, sigma)
    plt.hist(X_gen, alpha = 0.20, bins = max(X_gen), color = 'grey', edgecolor = 'white', linewidth = 3) # plot histogram of input data set
    print(SVMM_gmm(X_gen, False, False))
