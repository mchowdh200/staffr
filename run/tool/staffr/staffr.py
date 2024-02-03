from argparse import ArgumentParser
from model_hw import model_hw, plot_model_hw
from model_nonhw import model_nonhw, ks_calc
from pandas import read_csv
import numpy as np
import json
import os

if __name__ == '__main__':
    parser = ArgumentParser(description='retrieve probability of recessive allele and associated p-value from structural variant')
    parser.add_argument('inputData', type=str, help='data input')
    parser.add_argument('-viz', action='store_true', help='')
    parser.add_argument('-p', action='store_true', help='')
    parser.add_argument('-c', type=float, help='convergence tolerance for EM')
    parser.add_argument('-v', action='store_true', help='calculate p-val')
    parser.add_argument('-m', type=str, help='theta file')
    parser.add_argument('-o', action='store_true', help='outlier detection')
    args = parser.parse_args()
    
    inputFile = args.inputData
    show_plot = args.viz
    outputFile = os.path.join(os.path.dirname(inputFile), 'output', os.path.basename(inputFile).split('.')[0] + '_output.json')

    file = read_csv(inputFile, header=None)
    X = file.values[0]

    if not args.m:
        theta_hw = list(model_hw(X, show_plot, False))
        theta_hw.append(np.sqrt(theta_hw[0][2]))

        param_names = {
            'pi': 0,
            'mu': 1,
            'sigma': 2,
            'q': 3
        }

        outputData = {}
        for param_name, index in param_names.items():
            if isinstance(theta_hw[index], np.ndarray):
                outputData[param_name] = theta_hw[index].tolist()
            else:
                outputData[param_name] = theta_hw[index]

        with open(outputFile, 'w') as f:
            json.dump(outputData, f)

    if args.v:
        p_val = ks_calc(X, True)
        with open(outputFile, 'a') as f:
            f.write(', ' + str(p_val))

    if args.viz and args.m:
        theta_file = args.m
        with open(theta_file, 'r') as f:
            data = f.read().split(',')
            theta_values = []
            for item in data:
                item = item.strip() 
                if item.startswith('[') and item.endswith(']'):
                    item = item.strip('[]').split()
                    theta_values.append([float(i) for i in item])
                else:
                    theta_values.append(float(item))
        plot_model_hw(X, len(X), theta_values[0], theta_values[1], theta_values[2])

# if args.p == '' or args.p is None:
#     file = read_csv('../data/'+args.file_name, header = None)
#     X = file.values[0]
#     N = len(X)
#     pi1, pi2, alpha, lambda_, mu, sigma = SVMM(X, show_display, False)
#     q = 1 - sqrt(pi1)
#     with open('params.txt','w') as f: #append '-params' to input file name, stick necessary flags into output file name
#         f.write('{},{},{},{},{},{},{}\n'.format(N, pi1, pi2, alpha, lambda_, mu, sigma, q))
# else:
#     with open(args.p, 'r') as f:
#         params = f.readline().split(',')
#     params = [float(param) for param in params]
#     params[0] = int(params[0])
#     file = read_csv('../data/'+args.file_name, header = None)
#     X = file.values[0]
#     plot_SVMM(X=X,
#               n=params[0],
#               pi=[params[1],params[2]],
#               alpha=params[3],
#               lambda_=params[4],    
#               mu=params[5],
#               sigma=params[6]
#     )

    