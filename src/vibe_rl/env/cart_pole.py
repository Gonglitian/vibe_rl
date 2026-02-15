"""Pure-JAX CartPole environment.

Physics match the classic Barto, Sutton & Anderson (1983) formulation
and Gymnasium's CartPole-v1 defaults.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from vibe_rl.env.base import Environment, EnvParams, EnvState
from vibe_rl.env.spaces import Box, Discrete


class CartPoleState(EnvState):
    x: jax.Array
    x_dot: jax.Array
    theta: jax.Array
    theta_dot: jax.Array


class CartPoleParams(EnvParams):
    gravity: float = eqx.field(static=True, default=9.8)
    masscart: float = eqx.field(static=True, default=1.0)
    masspole: float = eqx.field(static=True, default=0.1)
    length: float = eqx.field(static=True, default=0.5)
    force_mag: float = eqx.field(static=True, default=10.0)
    tau: float = eqx.field(static=True, default=0.02)
    theta_threshold: float = eqx.field(static=True, default=0.2094395)  # 12 degrees
    x_threshold: float = eqx.field(static=True, default=2.4)
    max_steps: int = eqx.field(static=True, default=500)


class CartPole(Environment):
    """Classic CartPole balancing task in pure JAX.

    Observation: ``[x, x_dot, theta, theta_dot]``
    Actions: ``0`` (push left) or ``1`` (push right)
    Reward: ``+1`` per timestep the pole stays upright.

    Episode terminates when the pole angle exceeds ±12° or the cart
    leaves ±2.4 from center, or after ``max_steps``.
    """

    def default_params(self) -> CartPoleParams:
        return CartPoleParams()

    def reset(
        self,
        key: jax.Array,
        params: CartPoleParams,
    ) -> tuple[jax.Array, CartPoleState]:
        init = jax.random.uniform(key, shape=(4,), minval=-0.05, maxval=0.05)
        state = CartPoleState(
            x=init[0],
            x_dot=init[1],
            theta=init[2],
            theta_dot=init[3],
            time=jnp.int32(0),
        )
        return self._get_obs(state), state

    def step(
        self,
        key: jax.Array,
        state: CartPoleState,
        action: jax.Array,
        params: CartPoleParams,
    ) -> tuple[jax.Array, CartPoleState, jax.Array, jax.Array, dict[str, Any]]:
        force = jnp.where(action == 1, params.force_mag, -params.force_mag)

        total_mass = params.masscart + params.masspole
        polemass_length = params.masspole * params.length

        cos_th = jnp.cos(state.theta)
        sin_th = jnp.sin(state.theta)

        temp = (force + polemass_length * state.theta_dot ** 2 * sin_th) / total_mass
        theta_acc = (params.gravity * sin_th - cos_th * temp) / (
            params.length * (4.0 / 3.0 - params.masspole * cos_th ** 2 / total_mass)
        )
        x_acc = temp - polemass_length * theta_acc * cos_th / total_mass

        # Euler integration
        x = state.x + params.tau * state.x_dot
        x_dot = state.x_dot + params.tau * x_acc
        theta = state.theta + params.tau * state.theta_dot
        theta_dot = state.theta_dot + params.tau * theta_acc

        time = state.time + 1
        new_state = CartPoleState(
            x=x, x_dot=x_dot, theta=theta, theta_dot=theta_dot, time=time,
        )

        terminated = (
            (x < -params.x_threshold)
            | (x > params.x_threshold)
            | (theta < -params.theta_threshold)
            | (theta > params.theta_threshold)
        )
        truncated = time >= params.max_steps
        done = terminated | truncated
        reward = jnp.float32(1.0)

        return self._get_obs(new_state), new_state, reward, done, {"terminated": terminated, "truncated": truncated}

    def observation_space(self, params: CartPoleParams) -> Box:
        high = jnp.array(
            [params.x_threshold * 2, jnp.finfo(jnp.float32).max,
             params.theta_threshold * 2, jnp.finfo(jnp.float32).max],
            dtype=jnp.float32,
        )
        return Box(low=-high, high=high)

    def action_space(self, params: CartPoleParams) -> Discrete:
        return Discrete(n=2)

    @staticmethod
    def _get_obs(state: CartPoleState) -> jax.Array:
        return jnp.array([state.x, state.x_dot, state.theta, state.theta_dot], dtype=jnp.float32)
