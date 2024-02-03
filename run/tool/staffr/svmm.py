#!/usr/bin/env python3
from argparse import ArgumentParser
from pandas import read_csv
from numpy import sqrt
from model_hw import model_hw

parser = ArgumentParser(
    description='retrieve probability of recessive allele and associated p-value from structural variant'
)
parser.add_argument(
    'file_name',
    type=str,
    help='genotype count data set'
)
parser.add_argument(
    '-p',
    nargs='?',
    const='',
    default=None,
    help='optional figure display'
)
parser.add_argument(
    '-c',
    type=float,
    help='convergence tolerance for EM'
)
parser.add_argument(
    '-nop',
    action='store_false'
)

args = parser.parse_args()

show_display = True if args.p is not None else False

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
