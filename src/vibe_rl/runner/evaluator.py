"""JIT-compiled evaluation with vmap-parallel episodes.

Rolls out a greedy policy across multiple episodes simultaneously using
``jax.vmap`` and ``jax.lax.while_loop``, avoiding Python-level loops.

Usage::

    eval_metrics = evaluate(
        act_fn, agent_state,
        env, env_params,
        n_episodes=10, max_steps=500,
        rng=jax.random.PRNGKey(99),
    )
    # eval_metrics.mean_return, eval_metrics.std_return, eval_metrics.mean_length
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp

from vibe_rl.env.base import EnvParams, EnvState, Environment


class EvalMetrics(NamedTuple):
    """Aggregated evaluation results."""

    mean_return: chex.Array
    std_return: chex.Array
    mean_length: chex.Array


class _EpisodeCarry(NamedTuple):
    """Per-episode loop state."""

    obs: chex.Array
    env_state: EnvState
    rng: chex.PRNGKey
    total_reward: chex.Array
    length: chex.Array
    done: chex.Array


def evaluate(
    act_fn: callable,
    agent_state: chex.ArrayTree,
    env: Environment,
    env_params: EnvParams,
    *,
    n_episodes: int,
    max_steps: int,
    rng: chex.PRNGKey,
) -> EvalMetrics:
    """Evaluate a greedy policy over multiple episodes in parallel.

    The evaluation is fully JIT-compiled. Each episode runs inside a
    ``lax.while_loop`` (bounded by *max_steps*), and the *n_episodes*
    episodes are vmapped in parallel.

    Args:
        act_fn: ``(agent_state, obs) -> action``  — greedy action selector.
            Must be JIT-compatible and **not** update agent state (pure
            inference). This function should already handle ``explore=False``.
        agent_state: Agent state pytree (broadcast across episodes).
        env: Pure-JAX environment.
        env_params: Environment parameters.
        n_episodes: Number of parallel evaluation episodes.
        max_steps: Maximum steps per episode (must be static for XLA).
        rng: PRNG key; split across episodes.

    Returns:
        ``EvalMetrics`` with mean/std return and mean episode length.
    """

    def _run_episode(key: chex.PRNGKey) -> tuple[chex.Array, chex.Array]:
        """Run one episode, returning (total_reward, length)."""
        key_reset, key_steps = jax.random.split(key)
        obs, env_state = env.reset(key_reset, env_params)

        carry = _EpisodeCarry(
            obs=obs,
            env_state=env_state,
            rng=key_steps,
            total_reward=jnp.float32(0.0),
            length=jnp.int32(0),
            done=jnp.bool_(False),
        )

        def _cond(carry: _EpisodeCarry) -> chex.Array:
            return ~carry.done & (carry.length < max_steps)

        def _body(carry: _EpisodeCarry) -> _EpisodeCarry:
            action = act_fn(agent_state, carry.obs)
            rng, step_key = jax.random.split(carry.rng)
            obs, env_state, reward, done, _info = env.step(
                step_key, carry.env_state, action, env_params,
            )
            return _EpisodeCarry(
                obs=obs,
                env_state=env_state,
                rng=rng,
                total_reward=carry.total_reward + reward,
                length=carry.length + 1,
                done=done,
            )

        final = jax.lax.while_loop(_cond, _body, carry)
        return final.total_reward, final.length

    keys = jax.random.split(rng, n_episodes)
    returns, lengths = jax.vmap(_run_episode)(keys)

    return EvalMetrics(
        mean_return=jnp.mean(returns),
        std_return=jnp.std(returns),
        mean_length=jnp.mean(lengths.astype(jnp.float32)),
    )


# Pre-jitted version for convenience (caller can also jit externally).
jit_evaluate = jax.jit(evaluate, static_argnames=("act_fn", "env", "n_episodes", "max_steps"))
