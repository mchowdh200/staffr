import argparse
import os
import sys
from functools import partial
from multiprocessing import Pool
from typing import Any

import numpy as np
import pandas as pd

from hw_model import hw_model_converge, hw_model_iter
from nonhw_model import ks_calc, nonhw_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="retrieve probability of alternate allele and associated p-value from structural variant"
    )
    parser.add_argument("input", type=str, nargs="?", help="data input", default=None)
    parser.add_argument(
        "-c",
        "--convergence",
        type=float,
        default=1e-4,
        help="convergence tolerance for EM",
    )
    parser.add_argument(
        "-p",
        "--p-value",
        action="store_true",
        help="p-value calculation, requires theta, p-value",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="output file",
        required=True,
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="number of threads to use",
        default=1,
    )
    parser.add_argument(
        "--data-column-start",
        type=int,
        required=True,
        help="data column start (0-based)",
    )
    parser.add_argument(
        "--data-column-format",
        type=str,
        default="sample:count",
        help="""Data column format. Expected to be a string with colon separated fields.
        This program will expect one of the fields to be "count" and will use that field as the data.
        """,
    )
    return parser.parse_args()


def parse_data_item(x: str, data_index: int) -> int:
    return int(x.split(":")[data_index])
    # count_index = fields.index("count")


def fit_model(X: np.ndarray[Any, Any], c: float, p: bool) -> dict[str, Any]:
    theta_hw = list(hw_model_converge(X, c))

    q = np.sqrt(theta_hw[0][2])
    theta_hw.append(q)

    output_data = {
        "pi": theta_hw[0],
        "mu": theta_hw[1],
        "sigma": np.array(theta_hw[2]),  # just to get it to print better
        "log-likelihood": theta_hw[3],
        "q": theta_hw[4],
    }

    if p:
        p_value = ks_calc(X, False)
        output_data["p-value"] = p_value

    return output_data


def main():
    np.set_printoptions(threshold=sys.maxsize)
    args = parse_args()
    pool = Pool(args.threads)

    data = pd.read_csv(args.input, sep="\t", header=None)
    data_field_index = args.data_column_format.split(":").index("count")
    print(data.head())
    X = (
        data.iloc[:, args.data_column_start :]
        .applymap(partial(parse_data_item, data_index=data_field_index))
        .values
    )
    print(X.shape)
    exit()

    O = pool.map(partial(fit_model, c=args.convergence, p=args.p_value), X)

    output_data = pd.DataFrame(O)

    # concatenate with original data minus the data columns
    output_data = pd.concat(
        [data.iloc[:, : args.data_column_start], output_data], axis=1
    )
    output_data.to_csv(args.output, sep="\t", index=False, header=True)


if __name__ == "__main__":
    main()
