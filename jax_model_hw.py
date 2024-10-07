from dataclasses import dataclass, fields
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct
from jax import lax, tree_util
from jax.scipy.stats import norm
from jax.typing import ArrayLike
from ott.tools.k_means import k_means

# jax.config.update("jax_debug_nans", True)

@struct.dataclass
class State:
    pi: jax.Array
    mu: jax.Array
    sigma: jax.Array
    r: jax.Array
    iterations: int
    log_likelihood: float
    prev_log_likelihood: float


@partial(jax.jit, static_argnums=(1, 2, 3))
def fit_hw_model(X: jax.Array, n: int, c: float = 1e-5, max_iter: int = 10_000):
    km_out = k_means(X.reshape((-1, 1)), 3)
    centers = jnp.sort(km_out.centroids.ravel().sort()[1:])

    initial_state = State(
        pi=jnp.ones(3, dtype=jnp.float32) / 3.,
        mu=centers,
        sigma=jnp.ones(2) * jnp.std(X),
        r=jnp.zeros((3, n)),
        iterations=0,
        log_likelihood=0.0,
        prev_log_likelihood=1.0,
    )

    ## ------------------------------------------------------------------------
    ## EM algorithm loop
    ## ------------------------------------------------------------------------
    def EM_main_loop(initial_state: State):
        def loop_condition(state: State):
            return (state.iterations < max_iter) & (
                jnp.abs(state.log_likelihood - state.prev_log_likelihood) > c
            )

        def loop_body(state: State):
            ## Unpack state
            pi, mu, sigma, r, iterations, log_likelihood, prev_log_likelihood = (
                state.pi,
                state.mu,
                state.sigma,
                state.r,
                state.iterations,
                state.log_likelihood,
                state.prev_log_likelihood,
            )


            ## E-step -------------------------------------------------------------
            # zero component
            r = r.at[0].set(jnp.where(X == 0, pi[0], 0.0))

            # gaussian components
            r = r.at[1].set(pi[1] * norm.pdf(X, mu[0], sigma[0]))
            r = r.at[2].set(pi[2] * norm.pdf(X, mu[1], sigma[1]))
            r = r / jnp.maximum(jnp.sum(r, axis=0), 1e-6)

            ## M-step -------------------------------------------------------------
            # total responsibility of a component divided by total # of observations
            pi = jnp.sum(r, axis=1) / jnp.float32(n)

            # MLE for gaussian means
            mu = mu.at[0].set(jnp.average(X, weights=r[1]))
            mu = mu.at[1].set(jnp.average(X, weights=r[2]))

            # constrain mu[1] = 2mu[0]
            mu = mu.at[0].set((mu[0] + 0.5 * mu[1]) / 2)
            mu = mu.at[1].set(2 * mu[0])

            # MLE for gaussian variances
            sigma = sigma.at[0].set(jnp.sqrt(jnp.average((X - mu[0]) ** 2, weights=r[1])))
            sigma = sigma.at[1].set(jnp.sqrt(jnp.average((X - mu[1]) ** 2, weights=r[2])))
            sigma = jnp.maximum(sigma, 1e-6)

            ## Compute log likelihood ---------------------------------------------
            hurdle = jnp.where(X == 0, pi[0], 0)
            gmm = pi[1] * norm.pdf(X, mu[0], sigma[0]) + pi[2] * norm.pdf(
                X, mu[1], sigma[1]
            ) + 1e-6

            return State(
                pi=pi,
                mu=mu,
                sigma=sigma,
                r=r,
                iterations=iterations + 1,
                log_likelihood=jnp.sum(jnp.log(hurdle + gmm)),
                prev_log_likelihood=log_likelihood,
            )

        return lax.while_loop(loop_condition, loop_body, initial_state)

    ## Does our data have 3 distinct clusters? --------------------------------
    output = lax.cond(
        jnp.any(X != 0), # & km_out.converged & (centers[0] != centers[1]),
        ## TRUE
        lambda: EM_main_loop(initial_state),
        ## FALSE
        lambda: State(
            pi=jnp.array([1., 0., 0.]),
            mu=jnp.array([jnp.nan, jnp.nan]),
            sigma=jnp.array([jnp.nan, jnp.nan]),
            r=jnp.zeros((3, n)),
            iterations=0,
            log_likelihood=jnp.nan,
            prev_log_likelihood=jnp.nan,
        ),
    )
    # output = EM_main_loop(initial_state)

    return output


if __name__ == "__main__":
    x = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=jnp.float32)

    out = k_means(x.ravel().reshape(-1, 1), 2)
    print(jnp.sort(out.centroids.ravel(), axis=0))
