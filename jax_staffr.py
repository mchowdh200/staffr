import argparse
import os
import sys
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import pandas as pd
from tqdm import tqdm

from jax_model_hw import fit_hw_model


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
        "--batch-size",
        type=int,
        help="number of lines to process at a time",
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
    # parser.add_argument(
    #     "--acceptance-threshold",
    #     type=int,
    #     default=5,
    #     help="minimum depth of stix evidence to consider as non-zero",
    # )
    # parser.add_argument(
    #     "--plot-distributions",
    #     type=str,
    #     default=None,
    #     help="plots distributions in path specified as argument",
    # )
    return parser.parse_args()


def parse_data_item(x: str, data_index: int) -> int:
    return int(x.split(":")[data_index])


def main(
    input: str,
    iteration_limit: int,
    convergence: float,
    p_value: bool,
    batch_size: int,
    data_column_start: int,
    data_column_format: str,
):
    jnp.set_printoptions(threshold=sys.maxsize)

    data = pd.read_csv(input, sep="\t", header=None)
    data_field_index = data_column_format.split(":").index("count")

    result_keys = ["pi", "mu", "sigma", "log-likelihood", "q"]
    # if p_value:
    #     result_keys.append("p-value")

    header = (
        "\t".join(map(str, data.columns[:data_column_start]))
        + "\t"
        + "\t".join(result_keys)
    )
    print(header)

    X = (
        data.iloc[:, data_column_start:]
        .map(partial(parse_data_item, data_index=data_field_index))
        .values
    )

    output = jax.vmap(fit_hw_model, in_axes=(0, None, None, None))(
        X, X.shape[1], convergence, iteration_limit
    )

    for (_, row), out in zip(data.iterrows(), output):
        print("\t".join(map(str, row.iloc[:data_column_start])), end="\t")
        print("\t".join(map(str, [out.pi, out.mu, out.sigma, out.log_likelihood, out.q])))


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
