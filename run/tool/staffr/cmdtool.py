from argparse import ArgumentParser
from hw import model_hw
from model_nonhw import model_nonhw
from pandas import read_csv
import numpy as np
from pcalc import pcalc

if __name__ == '__main__':
    parser = ArgumentParser(
        description='retrieve probability of recessive allele and associated p-value from structural variant'
    )
    parser.add_argument('inputData', type=str, help='data input')
    parser.add_argument('-viz', nargs='?', const='', default=None, help='')
    parser.add_argument('-p', nargs='?', const='', default=None, help='')
    parser.add_argument('-c', type=float, help='convergence tolerance for EM')
    parser.add_argument('-v', nargs='?', const='', default=None, help='outlier detection')
    parser.add_argument('outputData', type=str, help='data output')
    args = parser.parse_args()
    
    file = read_csv(args.inputData, header = None)
    X = file.values[0]
    theta_hw = model_hw(X, False, False)

    with open(args.outputData, 'w') as f:
        f.write(', '.join(str(param) for param in theta_hw)
)
    if (args.v =='' or args.v == None):
        with open(args.outputData, 'a') as f:
            f.write(', '+str(pcalc(theta_hw))
    )
    
    