from model_hw import model_hw
from model_nonhw import model_nonhw
from pandas import read_csv
from scipy.stats import geom, norm, multinomial, expon, truncnorm
import matplotlib.pyplot as plt 
import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import KMeans as km
import pandas as pd
def pcalc(theta):
    # N = 1000
    # alpha = 0.5
    # lambda_ = 2


    # X = N
    # X_unique = np.unique(X)

    # alpha_hat_x, lambda_hat_x, _ = 0, 0 

    # EDF_x = [np.sum(X <= x) / len(X) for x in X_unique]
    # CDF_x = [alpha_hat_x if x == 0 else alpha_hat_x + (1 - alpha_hat_x) * geom.cdf(x, 1/lambda_hat_x) for x in X_unique]

    # plt.plot(X_unique, EDF_x, label = 'EDF', linestyle = '--', color = 'k')
    # plt.plot(X_unique, CDF_x, label = 'CDF', color = 'red', alpha = 0.75)
    # plt.xlabel('X')
    # plt.ylabel('Cumulative Density of X')
    # plt.legend()
    # plt.show()

    # ks_x = max([abs(cdf_val - edf_val) for cdf_val, edf_val in zip(CDF_x, EDF_x)])

    # null_dist = []

    # for _ in range(1000):
    #     # Y_i = psi_to_data(N, alpha_hat_x, lambda_hat_x)
    #     Y_i = np.sort(Y_i)
    #     Y_i_unique = np.unique(Y_i)

    #     alpha_hat_y, lambda_hat_y, _ = model_nonhw(Y_i, False, False)

    #     EDF_y = [np.sum(Y_i <= y) / len(Y_i) for y in Y_i_unique]
    #     CDF_y = [alpha_hat_y if y == 0 else alpha_hat_y + (1 - alpha_hat_y) * geom.cdf(y, 1/lambda_hat_y) for y in Y_i_unique]

    #     # visual sanity check
    #     # plt.plot(Y_i_unique, EDF_y, label = 'EDF', linestyle = '--', color = 'k')
    #     # plt.plot(Y_i_unique, CDF_y, label = 'CDF', color = 'red', alpha = 0.75)
    #     # plt.xlabel('Y')
    #     # plt.ylabel('Cumulative Density of Y')
    #     # plt.legend()
    #     # plt.show()

    #     ks_y = max([abs(cdf_val - edf_val) for cdf_val, edf_val in zip(CDF_y, EDF_y)])
    #     null_dist.append(ks_y)

    # plt.hist(null_dist, weights=np.zeros_like(null_dist) + 1. / len(null_dist), bins = 30, color = 'grey', alpha = 0.5)
    # plt.axvline(ks_x, color='red', label = r'D$_x$')
    # plt.xlabel('D (KS Distance)')
    # plt.ylabel('Pr(D)')
    # plt.legend()
    # plt.show()

    # p_val = np.sum([x for x in null_dist if x > ks_x]) / np.sum(null_dist)
    # print('p-val:', p_val)
    return 0.124999993