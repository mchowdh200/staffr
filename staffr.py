import argparse
import os
import sys
from functools import partial
from multiprocessing import Pool
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from hw_model import fit_hw_model, plot_hw_model
from nonhw_model import ks_calc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrieve probability of alternate allele and associated p-value from structural variant. "
        "Input/output data is tab-separated. See args for more details on input format. Output is printed to stdout "
        "and follows the format of the input data."
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
        type=float,
        default=1e-4,
        help="convergence tolerance for EM",
    )
    parser.add_argument(
        "-p",
        "--p-value",
        action="store_true",
        help="p-value calculation",
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
    parser.add_argument(
        "--acceptance-threshold",
        type=int,
        default=5,
        help="minimum depth of stix evidence to consider as non-zero",
    )
    parser.add_argument(
        "--plot-distributions",
        type=str,
        default=None,
        help="plots distributions in path specified as argument",
    )
    return parser.parse_args()


def parse_data_item(x: str, data_index: int) -> int:
    return int(x.split(":")[data_index])


def fit_model(
    data: pd.Series,
    iteration_limit: int,
    c: float,
    p: bool,
    data_field_index: int,
    data_column_start: int,
    acceptance_threshold: int,
    plot_dir: str,
) -> dict[str, Any]:
    X = (
        data[data_column_start:]
        .apply(partial(parse_data_item, data_index=data_field_index))
        .values
    )
    X[X < acceptance_threshold] = 0
    if not X.any():
        return {
            "pi": np.array([1, 0, 0]),
            "mu": np.array([0, 0]),
            "sigma": np.array([0, 0]),
            "log-likelihood": 0,
            "q": 0,
            "p-value": "NA",
        }
    output_data = fit_hw_model(X, c=c, max_iter=iteration_limit)
    output_data["q"] = np.sqrt(output_data["pi"][2])

    if p:
        if output_data["q"] > 0.0:
            p_value = ks_calc(X, False)
        else:
            p_value = "NA"
        output_data["p-value"] = p_value

    if plot_dir and output_data["pi"][2] > 0.0:
        plot_hw_model(
            X=X,
            pi=output_data["pi"],
            mu=output_data["mu"],
            sigma=output_data["sigma"],
            output=f"{plot_dir}/{'_'.join(map(str, data[: data_column_start]))}.png",
            p=output_data["p-value"],
        )
    return output_data


def main():
    np.set_printoptions(threshold=sys.maxsize)
    args = parse_args()
    if args.plot_distributions:
        os.makedirs(args.plot_distributions, exist_ok=True)

    data = pd.read_csv(args.input, sep="\t", header=None)
    data_field_index = args.data_column_format.split(":").index("count")

    result_keys = ["pi", "mu", "sigma", "log-likelihood", "q"]
    if args.p_value:
        result_keys.append("p-value")

    header = (
        "\t".join(map(str, data.columns[: args.data_column_start]))
        + "\t"
        + "\t".join(result_keys)
    )
    print(header)

    with Pool(args.threads) as pool:
        for (_, row), result in tqdm(
            zip(
                data.iterrows(),
                pool.imap(
                    partial(
                        fit_model,
                        c=args.convergence,
                        iteration_limit=args.iteration_limit,
                        p=args.p_value,
                        plot_dir=args.plot_distributions,
                        data_column_start=args.data_column_start,
                        data_field_index=data_field_index,
                        acceptance_threshold=args.acceptance_threshold,
                    ),
                    [d[1] for d in data.iterrows()],
                ),
            ),
            total=len(data),
        ):
            print("\t".join(map(str, row.iloc[: args.data_column_start])), end="\t")
            print("\t".join(str(result[k]) for k in result_keys))


if __name__ == "__main__":
    main()
