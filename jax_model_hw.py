from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.stats import norm
from jax.typing import ArrayLike
from ott.tools.k_means import k_means


@dataclass(kw_only=True)
class State:
    pi: jax.Array
    mu: jax.Array
    sigma: jax.Array
    r: jax.Array
    iterations: int
    log_likelihood: float
    prev_log_likelihood: float


@partial(jax.jit, static_argnums=(1, 2, 3))
def fit_hw_model(X: jax.Array, n: int, c: float = 1e-4, max_iter: int = 10_000):
    f = X.reshape(-1, 1)
    km_out = k_means(f, 3)
    centers = jnp.sort(km_out.centroids.ravel())
    initial_state = State(
        pi=jnp.ones(3) / 3,
        mu=centers[1:],
        sigma=jnp.ones(2) * jnp.std(X),
        r=jnp.zeros((3, n)),
        iterations=0,
        log_likelihood=0.0,
        prev_log_likelihood=jnp.inf,
    )

    ## ------------------------------------------------------------------------
    ## EM algorithm loop
    ## ------------------------------------------------------------------------
    def while_loop(initial_state: State):
        def loop_condition(state: State):
            return (state.iterations < max_iter) or (
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
            r.at[0, X == 0].set(pi[0])
            r.at[0, X != 0].set(0.0)

            # gaussian components
            r.at[1, :].set(pi[1] * norm.pdf(X, mu[0], sigma[0]))
            r.at[2, :].set(pi[2] * norm.pdf(X, mu[1], sigma[1]))
            r = r / jnp.sum(r, axis=0)

            ## M-step -------------------------------------------------------------
            # total responsibility of a component divided by total # of observations
            pi = jnp.sum(r, axis=1) / n

            # MLE for gaussian means
            mu.at[0].set(jnp.average(X, weights=r[1]))
            mu.at[1].set(jnp.average(X, weights=r[2]))

            # constrain mu[1] = 2mu[0]
            mu.at[0].set((mu[0] + 0.5 * mu[1]) / 2)
            mu.at[1].set(2 * mu[0])

            # MLE for gaussian variances
            sigma.at[0].set(jnp.sqrt(jnp.average((X - mu[0]) ** 2, weights=r[1])))
            sigma.at[1].set(jnp.sqrt(jnp.average((X - mu[1]) ** 2, weights=r[2])))
            sigma = jnp.maximum(sigma, 1e-6)

            ## Compute log likelihood ---------------------------------------------
            hurdle = jnp.where(X == 0, pi[0], 0)
            gmm = pi[1] * norm.pdf(X, mu[0], sigma[0]) + pi[2] * norm.pdf(
                X, mu[1], sigma[1]
            )
            prev_log_likelihood = log_likelihood
            log_likelihood = jnp.sum(jnp.log(hurdle + gmm))
            iterations = iterations + 1

            return State(
                pi=pi,
                mu=mu,
                sigma=sigma,
                r=r,
                iterations=iterations,
                log_likelihood=log_likelihood,
                prev_log_likelihood=prev_log_likelihood,
            )

        return lax.while_loop(loop_condition, loop_body, initial_state)

    ## Does our data have 3 distinct clusters? --------------------------------
    output = lax.cond(
        km_out.centroids.shape[0] != 3,
        ## TRUE
        lambda: while_loop(initial_state),
        ## FALSE
        lambda: State(
            pi=jnp.array([1, 0, 0]),
            mu=jnp.array([0, 0, 0]),
            sigma=jnp.array([0, 0]),
            r=jnp.zeros((3, n)),
            iterations=0,
            log_likelihood=0.0,
            prev_log_likelihood=jnp.inf,
        )
    )

    return output


if __name__ == "__main__":
    x = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=jnp.float32)

    out = k_means(x.ravel().reshape(-1, 1), 2)
    print(jnp.sort(out.centroids.ravel(), axis=0))
