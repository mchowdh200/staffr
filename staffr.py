from argparse import ArgumentParser
from hw_model import hw_model_iter, hw_model_converge, plot_hw_model
from nonhw_model import nonhw_model, plot_nonhw_model, ks_calc
from pandas import read_csv
import numpy as np
import json
import os

if __name__ == '__main__':
    parser = ArgumentParser(description='retrieve probability of recessive allele and associated p-value from structural variant')
    parser.add_argument('input', type=str, nargs='?', help='data input', default=None)
    parser.add_argument('-c', type=float, help='convergence tolerance for EM, default = 1e-4')
    parser.add_argument('-i', type=float, help='iteration limit for EM, default = 10')
    parser.add_argument('-p', action='store_true', help='p-value calculation, requires theta, p-value')
    parser.add_argument('-n', action='store_true', help='plot null-model')
    parser.add_argument('-o', type=float, help='')
    parser.add_argument('-v', type=str, nargs='?', const=True, default=False, help='visualize data, requires theta')
    args = parser.parse_args()

    input_file = args.input
    output_file = os.path.splitext(input_file)[0] + '_output.json'

    file = read_csv(input_file, header=None)
    X = file.values[0]

    if args.c is not None:
        c = args.c
        theta_hw = list(hw_model_converge(X, c))
    elif args.i is not None:
        iter = args.i
        theta_hw = list(hw_model_iter(X, iter))
    else: 
        theta_hw = list(hw_model_iter(X))

    q = np.sqrt(theta_hw[0][2])
    theta_hw.append(q)

    param_names = {
        'pi': 0,
        'mu': 1,
        'sigma': 2,
        'log-likelihood': 3,
        'q': 4
    }
    output_data = {}
    for param_name, index in param_names.items():
        if isinstance(theta_hw[index], np.ndarray):
            output_data[param_name] = theta_hw[index].tolist()
        else:
            output_data[param_name] = theta_hw[index]

    with open(output_file, 'w') as f:
        json.dump(output_data, f)

    if args.v:
        with open(output_file, "r") as f:
            theta_data = json.load(f)
        pi = theta_data["pi"]
        mu = theta_data["mu"]
        sigma = theta_data["sigma"]

        plot_hw_model(X, len(X), pi, mu, sigma)

    if args.p:
        p_val = ks_calc(X, False)
        with open(output_file, "r") as f:
            theta_data = json.load(f)
        theta_data['p-value'] = p_val
        with open(output_file, "w") as f:
            json.dump(theta_data, f)
    
    if args.n:
        psi_nonhw = list(nonhw_model(X))
        plot_nonhw_model(X, len(X), psi_nonhw[0], psi_nonhw[1])