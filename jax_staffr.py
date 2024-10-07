import argparse
import os

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"
import sys
from functools import partial

import jax
import jax.numpy as jnp
import pandas as pd

from jax_model_hw import fit_hw_model


def unshard(xs):
    return jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), xs)


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
        default=32,
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
    # TODO reinstate plotting functionality
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
    # TODO reimplement null model in jax
    # if p_value:
    #     result_keys.append("p-value")

    header = (
        "\t".join(map(str, data.columns[:data_column_start]))
        + "\t"
        + "\t".join(result_keys)
    )
    print(header)

    X = jnp.array(
        data.iloc[:, data_column_start:]
        .map(partial(parse_data_item, data_index=data_field_index))
        .values,
        dtype=jnp.float32,
    )
    print(X.shape)

    # set number of devices (cpu cores) based on batch size
    fit_func = jax.pmap(
        fit_hw_model,
        in_axes=0,
        # in_axes=(0, None, None, None),
        static_broadcasted_argnums=(1, 2, 3),
        devices=jax.devices("cpu")[:batch_size],
    )

    for batch in range(0, X.shape[0], batch_size):
        # pad the array to ensure equal batch sizes.  The zip later will skip outputs corresponding to padding.
        if batch + batch_size > X.shape[0]:
            x = jnp.pad(X[batch:], ((0, batch + batch_size - X.shape[0]), (0, 0)))
        else:
            x = X[batch : batch + batch_size]

        output = fit_func(x, X.shape[1], convergence, iteration_limit)
        pi, mu, sigma, log_likelihood, Q = (
            output.pi,
            output.mu,
            output.sigma,
            output.log_likelihood,
            jnp.sqrt(output.pi[:, 3]),
        )
        print(pi.shape, mu.shape, sigma.shape, log_likelihood.shape)
        for (_, row), p, m, s, l, q in zip(
            data.iloc[batch : batch + batch_size].iterrows(),
            pi,
            mu,
            sigma,
            log_likelihood,
            Q,
        ):
            print("\t".join(map(str, row.iloc[:data_column_start])), end="\t")
            print("\t".join(map(str, [p, m, s, l, q])))


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
