"""Benchmark: vmap parallel environments vs single environment throughput.

Compares environment step throughput when running N parallel environments
via ``jax.vmap`` versus a sequential single-environment loop.

Also benchmarks the PPO collect_rollout_batch (vmapped) vs collect_rollout
(single env) for end-to-end comparison.

Usage::

    python benchmarks/bench_vmap.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from vibe_rl.algorithms.ppo import PPO, PPOConfig
from vibe_rl.env import make
from vibe_rl.env.wrappers import AutoResetWrapper


def _time_fn(fn, *args, warmup: int = 3, repeats: int = 20, **kwargs) -> float:
    """Time a function, returning seconds per call (excluding warmup)."""
    for _ in range(warmup):
        result = fn(*args, **kwargs)
        jax.tree.map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            result,
        )

    start = time.perf_counter()
    for _ in range(repeats):
        result = fn(*args, **kwargs)
        jax.tree.map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            result,
        )
    elapsed = time.perf_counter() - start
    return elapsed / repeats


# ── Raw Environment Step Benchmark ───────────────────────────────────


def bench_env_step():
    """Benchmark raw env.step throughput: vmap vs sequential."""
    print("=" * 60)
    print("Raw Environment Step Throughput")
    print("=" * 60)

    n_steps = 256

    for n_envs in [1, 4, 16, 64]:
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        rng = jax.random.PRNGKey(0)

        if n_envs == 1:
            # Single env: step n_steps times sequentially via lax.scan
            env_key = jax.random.PRNGKey(1)
            obs, env_state = env.reset(env_key, env_params)

            @jax.jit
            def _single_scan(obs, env_state, rng):
                def _step(carry, _):
                    obs, state, rng = carry
                    rng, step_key, act_key = jax.random.split(rng, 3)
                    action = jax.random.randint(act_key, (), 0, 2)
                    obs, state, reward, done, info = env.step(
                        step_key, state, action, env_params,
                    )
                    return (obs, state, rng), reward

                (obs, state, rng), rewards = jax.lax.scan(
                    _step, (obs, env_state, rng), None, n_steps,
                )
                return obs, state, rewards

            t = _time_fn(_single_scan, obs, env_state, rng)
            steps_per_sec = n_steps / t
            print(f"  {n_envs:3d} env(s), {n_steps} steps: "
                  f"{t * 1000:8.2f} ms | {steps_per_sec:,.0f} steps/sec")
        else:
            # Vectorized: vmap over n_envs, step n_steps times
            batch_reset = jax.vmap(env.reset, in_axes=(0, None))
            batch_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

            env_keys = jax.random.split(jax.random.PRNGKey(1), n_envs)
            obs_batch, env_states = batch_reset(env_keys, env_params)

            @jax.jit
            def _vmap_scan(obs_batch, env_states, rng):
                def _step(carry, _):
                    obs, states, rng = carry
                    rng, step_key, act_key = jax.random.split(rng, 3)
                    step_keys = jax.random.split(step_key, n_envs)
                    actions = jax.random.randint(act_key, (n_envs,), 0, 2)
                    obs, states, rewards, dones, infos = batch_step(
                        step_keys, states, actions, env_params,
                    )
                    return (obs, states, rng), rewards

                (obs, states, rng), rewards = jax.lax.scan(
                    _step, (obs_batch, env_states, rng), None, n_steps,
                )
                return obs, states, rewards

            t = _time_fn(_vmap_scan, obs_batch, env_states, rng)
            total_steps = n_envs * n_steps
            steps_per_sec = total_steps / t
            print(f"  {n_envs:3d} env(s), {n_steps} steps: "
                  f"{t * 1000:8.2f} ms | {steps_per_sec:,.0f} steps/sec")

    print()


# ── PPO Collect Benchmark ────────────────────────────────────────────


def bench_ppo_collect():
    """Benchmark PPO collect_rollout: single env vs vmap batch."""
    print("=" * 60)
    print("PPO Collect Rollout: Single vs Vectorized")
    print("=" * 60)

    config = PPOConfig(
        hidden_sizes=(64, 64),
        n_steps=128,
        n_minibatches=4,
        n_epochs=4,
    )

    env, env_params = make("CartPole-v1")
    env = AutoResetWrapper(env)
    env_params = env.default_params()

    rng = jax.random.PRNGKey(0)

    # Single environment
    rng, env_key, agent_key = jax.random.split(rng, 3)
    obs, env_state = env.reset(env_key, env_params)
    state = PPO.init(agent_key, obs_shape=(4,), n_actions=2, config=config)

    def _single_collect(state, obs, env_state):
        return PPO.collect_rollout(
            state, obs, env_state, env.step, env_params, config=config,
        )

    t_single = _time_fn(_single_collect, state, obs, env_state)
    print(f"  Single env (128 steps): {t_single * 1000:8.2f} ms | "
          f"{128 / t_single:,.0f} steps/sec")

    # Vectorized environments
    for n_envs in [4, 16, 64]:
        rng, *env_keys = jax.random.split(rng, n_envs + 1)
        env_keys = jnp.stack(env_keys)
        batch_reset = jax.vmap(env.reset, in_axes=(0, None))
        obs_batch, env_states = batch_reset(env_keys, env_params)

        rng, agent_key = jax.random.split(rng)
        state_batch = PPO.init(agent_key, obs_shape=(4,), n_actions=2, config=config)

        def _batch_collect(state, obs_batch, env_states, *, _n=n_envs):
            return PPO.collect_rollout_batch(
                state, obs_batch, env_states, env.step, env_params, config=config,
            )

        t_batch = _time_fn(_batch_collect, state_batch, obs_batch, env_states)
        total_steps = n_envs * 128
        print(f"  {n_envs:3d} envs (128 steps each): {t_batch * 1000:8.2f} ms | "
              f"{total_steps / t_batch:,.0f} steps/sec")

    print()


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    backend = jax.default_backend()
    devices = jax.devices()
    print(f"JAX backend: {backend}")
    print(f"Devices: {devices}")
    print()

    bench_env_step()
    bench_ppo_collect()

    print("Benchmark complete.")


if __name__ == "__main__":
    main()
