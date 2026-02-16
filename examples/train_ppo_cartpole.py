"""Train PPO on CartPole (pure-JAX, single environment).

Demonstrates the full collect-update loop using ``PPO.collect_rollout``
and ``PPO.update``, both driven by ``jax.lax.scan``.

Usage::

    python examples/train_ppo_cartpole.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from vibe_rl.algorithms.ppo import PPO, PPOConfig
from vibe_rl.env import make
from vibe_rl.env.wrappers import AutoResetWrapper


def main() -> None:
    seed = 42
    total_updates = 200

    config = PPOConfig(
        hidden_sizes=(64, 64),
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        n_steps=128,
        n_minibatches=4,
        n_epochs=4,
    )

    env, env_params = make("CartPole-v1")
    env = AutoResetWrapper(env)
    env_params = env.default_params()

    rng = jax.random.PRNGKey(seed)
    rng, env_key, agent_key = jax.random.split(rng, 3)

    obs, env_state = env.reset(env_key, env_params)
    state = PPO.init(agent_key, obs_shape=(4,), n_actions=2, config=config)

    for update in range(total_updates):
        # Collect rollout (jax.lax.scan internally)
        state, trajectories, obs, env_state, last_value = PPO.collect_rollout(
            state, obs, env_state, env.step, env_params, config=config,
        )

        # PPO update (multiple epochs of mini-batch SGD)
        state, metrics = PPO.update(
            state, trajectories, last_value, config=config,
        )

        if (update + 1) % 10 == 0:
            eval_returns = _evaluate(state, env, env_params, config, rng, n_episodes=5)
            mean_return = jnp.mean(eval_returns)
            print(
                f"Update {update + 1:4d} | "
                f"loss={float(metrics.total_loss):.4f} | "
                f"entropy={float(metrics.entropy):.4f} | "
                f"eval_return={float(mean_return):.1f}"
            )

    print("Training complete.")


def _evaluate(
    state,
    env,
    env_params,
    config: PPOConfig,
    rng,
    n_episodes: int = 5,
) -> jax.Array:
    """Run greedy evaluation episodes and return total returns."""
    returns = []
    for i in range(n_episodes):
        rng, eval_key = jax.random.split(rng)
        obs, env_state = env.reset(eval_key, env_params)
        episode_return = 0.0
        for _ in range(500):
            action, _, _, _ = PPO.act(state, obs, config=config)
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
