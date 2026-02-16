"""Train SAC on Pendulum (pure-JAX, continuous control).

Demonstrates the off-policy training loop for SAC with a continuous
action space. Uses the built-in Pendulum-v1 environment.

Usage::

    python examples/train_sac_pendulum.py
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from vibe_rl.algorithms.sac import SAC, SACConfig
from vibe_rl.dataprotocol.replay_buffer import ReplayBuffer
from vibe_rl.env import make
from vibe_rl.env.wrappers import AutoResetWrapper


def main() -> None:
    seed = 42
    total_steps = 50_000
    eval_every = 5_000
    min_buffer_size = 1_000

    config = SACConfig(
        hidden_sizes=(256, 256),
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        gamma=0.99,
        batch_size=256,
        tau=0.005,
        init_alpha=1.0,
        autotune_alpha=True,
        action_low=-2.0,
        action_high=2.0,
    )

    env, env_params = make("Pendulum-v1")
    env = AutoResetWrapper(env)
    env_params = env.default_params()

    rng = jax.random.PRNGKey(seed)
    rng, env_key, agent_key = jax.random.split(rng, 3)

    obs, env_state = env.reset(env_key, env_params)
    state = SAC.init(agent_key, obs_shape=(3,), action_dim=1, config=config)

    buffer = ReplayBuffer(
        capacity=50_000, obs_shape=(3,), action_shape=(1,), action_dtype=np.float32,
    )

    for step in range(total_steps):
        action, state = SAC.act(state, obs, config=config, explore=True)

        rng, step_key = jax.random.split(rng)
        next_obs, env_state, reward, done, info = env.step(
            step_key, env_state, action, env_params,
        )

        buffer.push(
            obs=np.asarray(obs),
            action=np.asarray(action),
            reward=float(reward),
            next_obs=np.asarray(next_obs),
            done=float(done),
        )

        obs = next_obs

        if len(buffer) >= min_buffer_size:
            batch = buffer.sample(config.batch_size)
            state, metrics = SAC.update(state, batch, config=config)

        if (step + 1) % eval_every == 0:
            eval_returns = _evaluate(state, env, env_params, config, rng)
            mean_return = float(jnp.mean(eval_returns))
            alpha = float(jnp.exp(state.log_alpha))
            print(
                f"Step {step + 1:6d} | "
                f"alpha={alpha:.3f} | "
                f"eval_return={mean_return:.1f}"
            )

    print("Training complete.")


def _evaluate(
    state,
    env,
    env_params,
    config: SACConfig,
    rng,
    n_episodes: int = 5,
) -> jax.Array:
    """Run deterministic evaluation episodes and return total returns."""
    returns = []
    for i in range(n_episodes):
        rng, eval_key = jax.random.split(rng)
        obs, env_state = env.reset(eval_key, env_params)
        episode_return = 0.0
        for _ in range(200):
            action, _ = SAC.act(state, obs, config=config, explore=False)
            rng, step_key = jax.random.split(rng)
            obs, env_state, reward, done, _ = env.step(
                step_key, env_state, action, env_params,
            )
            episode_return += float(reward)
            if float(done):
                break
        returns.append(episode_return)
    return jnp.array(returns)


if __name__ == "__main__":
    main()
