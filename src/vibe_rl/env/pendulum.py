"""Pure-JAX Pendulum environment.

Classic inverted pendulum swingup task with continuous action space.
Matches the Gymnasium Pendulum-v1 dynamics and reward function.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from vibe_rl.env.base import Environment, EnvParams, EnvState
from vibe_rl.env.spaces import Box


class PendulumState(EnvState):
    theta: jax.Array  # angle (radians)
    theta_dot: jax.Array  # angular velocity


class PendulumParams(EnvParams):
    max_speed: float = eqx.field(static=True, default=8.0)
    max_torque: float = eqx.field(static=True, default=2.0)
    dt: float = eqx.field(static=True, default=0.05)
    g: float = eqx.field(static=True, default=10.0)
    m: float = eqx.field(static=True, default=1.0)
    l: float = eqx.field(static=True, default=1.0)
    max_steps: int = eqx.field(static=True, default=200)


class Pendulum(Environment):
    """Classic pendulum swingup task in pure JAX.

    Observation: ``[cos(theta), sin(theta), theta_dot]``
    Actions: continuous torque in ``[-max_torque, max_torque]``
    Reward: ``-(theta^2 + 0.1 * theta_dot^2 + 0.001 * torque^2)``

    The goal is to swing the pendulum upright (theta=0) and keep it there.
    """

    def default_params(self) -> PendulumParams:
        return PendulumParams()

    def reset(
        self,
        key: jax.Array,
        params: PendulumParams,
    ) -> tuple[jax.Array, PendulumState]:
        k1, k2 = jax.random.split(key)
        theta = jax.random.uniform(k1, shape=(), minval=-jnp.pi, maxval=jnp.pi)
        theta_dot = jax.random.uniform(k2, shape=(), minval=-1.0, maxval=1.0)
        state = PendulumState(
            theta=theta,
            theta_dot=theta_dot,
            time=jnp.int32(0),
        )
        return self._get_obs(state), state

    def step(
        self,
        key: jax.Array,
        state: PendulumState,
        action: jax.Array,
        params: PendulumParams,
    ) -> tuple[jax.Array, PendulumState, jax.Array, jax.Array, dict[str, Any]]:
        # Clip action to torque bounds
        u = jnp.clip(action.reshape(()), -params.max_torque, params.max_torque)

        theta = state.theta
        theta_dot = state.theta_dot

        # Reward: penalize angle, velocity, and torque
        # Normalize angle to [-pi, pi]
        norm_theta = _angle_normalize(theta)
        reward = -(norm_theta ** 2 + 0.1 * theta_dot ** 2 + 0.001 * u ** 2)

        # Physics update (Euler integration)
        new_theta_dot = (
            theta_dot
            + (3.0 * params.g / (2.0 * params.l) * jnp.sin(theta)
               + 3.0 / (params.m * params.l ** 2) * u)
            * params.dt
        )
        new_theta_dot = jnp.clip(new_theta_dot, -params.max_speed, params.max_speed)
        new_theta = theta + new_theta_dot * params.dt

        time = state.time + 1
        new_state = PendulumState(
            theta=new_theta,
            theta_dot=new_theta_dot,
            time=time,
        )

        done = time >= params.max_steps
        return (
            self._get_obs(new_state),
            new_state,
            jnp.float32(reward),
            done,
            {"truncated": done},
        )

    def observation_space(self, params: PendulumParams) -> Box:
        high = jnp.array([1.0, 1.0, params.max_speed], dtype=jnp.float32)
        return Box(low=-high, high=high)

    def action_space(self, params: PendulumParams) -> Box:
        return Box(
            low=-params.max_torque,
            high=params.max_torque,
            shape=(1,),
        )

    @staticmethod
    def _get_obs(state: PendulumState) -> jax.Array:
        return jnp.array(
            [jnp.cos(state.theta), jnp.sin(state.theta), state.theta_dot],
            dtype=jnp.float32,
        )


def _angle_normalize(x: jax.Array) -> jax.Array:
    """Normalize angle to [-pi, pi]."""
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi
