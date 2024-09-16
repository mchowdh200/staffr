import warnings

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import argparse
import os
import sys
from functools import partial
from multiprocessing import Pool, set_start_method
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from hw_model import hw_model_converge, hw_model_iter
from nonhw_model import ks_calc, nonhw_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="retrieve probability of alternate allele and associated p-value from structural variant"
    )
    parser.add_argument("input", type=str, nargs="?", help="data input", default=None)
    parser.add_argument(
        "--iteration-limit",
        default=1000,
        help="iteration limit for EM",
    )
    parser.add_argument(
        "-c",
        "--convergence",
        default=None,
        help="convergence tolerance for EM",
    )
    parser.add_argument(
        "-p",
        "--p-value",
        action="store_true",
        help="p-value calculation, requires theta, p-value",
    )
    # parser.add_argument(
    #     "--output",
    #     type=str,
    #     help="output file",
    #     required=True,
    # )
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


def fit_model(
    X: np.ndarray[Any, Any],
    iteration_limit: int | None = None,
    c: float | None = None,
    p: bool = False,
) -> dict[str, Any]:
    if iteration_limit is not None:
        theta_hw = list(hw_model_iter(X, iteration_limit))
    elif c is not None:
        theta_hw = list(hw_model_converge(X, c))
    else:
        theta_hw = list(hw_model_converge(X, 1e-4))

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
    # set_start_method("spawn")
    np.set_printoptions(threshold=sys.maxsize)
    args = parse_args()
    pool = Pool(args.threads)

    data = pd.read_csv(args.input, sep="\t", header=None)
    data_field_index = args.data_column_format.split(":").index("count")
    X = (
        data.iloc[:, args.data_column_start :]
        .applymap(partial(parse_data_item, data_index=data_field_index))
        .values
    )

    # O = pool.map(
    #     partial(fit_model, c=args.convergence, p=args.p_value),
    #     X,
    # )
    result_keys = ["pi", "mu", "sigma", "log-likelihood", "q"]
    if args.p_value:
        result_keys.append("p-value")

    header = "\t".join(map(str, data.columns[: args.data_column_start])) + "\t".join(
        result_keys
    )
    print(header)

    for (_, row), result in tqdm(
        zip(
            data.iterrows(),
            pool.imap(partial(fit_model, iteration_limit=args.iteration_limit, p=args.p_value), X),
        ),
        total=len(X),
    ):
    # for (_, row), x in tqdm(zip(data.iterrows(), X), total=len(X)):
        # result = fit_model(x, args.iteration_limit, args.p_value)
        print("\t".join(map(str, row.iloc[: args.data_column_start])), end="\t")
        print("\t".join(str(result[k]) for k in result_keys))


if __name__ == "__main__":
    main()
