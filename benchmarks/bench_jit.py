"""Benchmark: JIT vs no-JIT performance comparison.

Compares training step throughput with and without ``jax.jit`` for
each algorithm (DQN, PPO, SAC).

Usage::

    python benchmarks/bench_jit.py
"""

from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp

from vibe_rl.algorithms.dqn import DQN, DQNConfig
from vibe_rl.algorithms.ppo import PPO, PPOConfig
from vibe_rl.algorithms.sac import SAC, SACConfig
from vibe_rl.dataprotocol.transition import PPOTransition, Transition
from vibe_rl.env import make
from vibe_rl.env.wrappers import AutoResetWrapper


def _time_fn(fn, *args, warmup: int = 3, repeats: int = 50, **kwargs) -> float:
    """Time a function, returning seconds per call (excluding warmup)."""
    for _ in range(warmup):
        result = fn(*args, **kwargs)
        # Force completion for JAX async dispatch
        jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, result)

    start = time.perf_counter()
    for _ in range(repeats):
        result = fn(*args, **kwargs)
        jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, result)
    elapsed = time.perf_counter() - start
    return elapsed / repeats


# ── DQN Benchmark ────────────────────────────────────────────────────


def bench_dqn_update():
    """Benchmark DQN update step: JIT vs no-JIT."""
    print("=" * 60)
    print("DQN Update Step")
    print("=" * 60)

    config = DQNConfig(hidden_sizes=(128, 128), lr=1e-3, batch_size=64)
    rng = jax.random.PRNGKey(0)
    state = DQN.init(rng, obs_shape=(4,), n_actions=2, config=config)

    k1, k2, k3 = jax.random.split(rng, 3)
    batch = Transition(
        obs=jax.random.normal(k1, (64, 4)),
        action=jax.random.randint(k2, (64,), 0, 2),
        reward=jax.random.normal(k3, (64,)),
        next_obs=jax.random.normal(k1, (64, 4)),
        done=jnp.zeros(64),
    )

    # JIT version (uses @jax.jit decorator on DQN.update)
    t_jit = _time_fn(DQN.update, state, batch, config=config)

    # No-JIT version
    def _update_no_jit(state, batch, *, config):
        with jax.disable_jit():
            return DQN.update.__wrapped__(state, batch, config=config)

    t_nojit = _time_fn(_update_no_jit, state, batch, config=config, warmup=1, repeats=5)

    speedup = t_nojit / t_jit if t_jit > 0 else float("inf")
    print(f"  JIT:    {t_jit * 1000:8.2f} ms/step")
    print(f"  No-JIT: {t_nojit * 1000:8.2f} ms/step")
    print(f"  Speedup: {speedup:.1f}x")
    print()


# ── PPO Benchmark ────────────────────────────────────────────────────


def bench_ppo_update():
    """Benchmark PPO collect + update: JIT vs no-JIT."""
    print("=" * 60)
    print("PPO Collect + Update")
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
    rng, env_key, agent_key = jax.random.split(rng, 3)
    obs, env_state = env.reset(env_key, env_params)
    state = PPO.init(agent_key, obs_shape=(4,), n_actions=2, config=config)

    def _jit_step(state, obs, env_state):
        state, traj, obs, env_state, lv = PPO.collect_rollout(
            state, obs, env_state, env.step, env_params, config=config,
        )
        state, metrics = PPO.update(state, traj, lv, config=config)
        return state, obs, env_state, metrics

    t_jit = _time_fn(_jit_step, state, obs, env_state)

    def _nojit_step(state, obs, env_state):
        with jax.disable_jit():
            state, traj, obs, env_state, lv = PPO.collect_rollout.__wrapped__(
                state, obs, env_state, env.step, env_params, config=config,
            )
            state, metrics = PPO.update.__wrapped__(state, traj, lv, config=config)
        return state, obs, env_state, metrics

    t_nojit = _time_fn(_nojit_step, state, obs, env_state, warmup=1, repeats=3)

    speedup = t_nojit / t_jit if t_jit > 0 else float("inf")
    print(f"  JIT:    {t_jit * 1000:8.2f} ms/update (128 env steps + SGD)")
    print(f"  No-JIT: {t_nojit * 1000:8.2f} ms/update")
    print(f"  Speedup: {speedup:.1f}x")
    print()


# ── SAC Benchmark ────────────────────────────────────────────────────


def bench_sac_update():
    """Benchmark SAC update step: JIT vs no-JIT."""
    print("=" * 60)
    print("SAC Update Step")
    print("=" * 60)

    config = SACConfig(hidden_sizes=(256, 256), batch_size=256)
    rng = jax.random.PRNGKey(0)
    state = SAC.init(rng, obs_shape=(3,), action_dim=1, config=config)

    k1, k2, k3 = jax.random.split(rng, 3)
    batch = Transition(
        obs=jax.random.normal(k1, (256, 3)),
        action=jax.random.normal(k2, (256, 1)),
        reward=jax.random.normal(k3, (256,)),
        next_obs=jax.random.normal(k1, (256, 3)),
        done=jnp.zeros(256),
    )

    t_jit = _time_fn(SAC.update, state, batch, config=config)

    def _update_no_jit(state, batch, *, config):
        with jax.disable_jit():
            return SAC.update.__wrapped__(state, batch, config=config)

    t_nojit = _time_fn(_update_no_jit, state, batch, config=config, warmup=1, repeats=3)

    speedup = t_nojit / t_jit if t_jit > 0 else float("inf")
    print(f"  JIT:    {t_jit * 1000:8.2f} ms/step")
    print(f"  No-JIT: {t_nojit * 1000:8.2f} ms/step")
    print(f"  Speedup: {speedup:.1f}x")
    print()


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    backend = jax.default_backend()
    devices = jax.devices()
    print(f"JAX backend: {backend}")
    print(f"Devices: {devices}")
    print()

    bench_dqn_update()
    bench_ppo_update()
    bench_sac_update()

    print("Benchmark complete.")


if __name__ == "__main__":
    main()
