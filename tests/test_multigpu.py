"""Tests for multi-GPU training via jit + NamedSharding (shard_map).

Uses XLA_FLAGS to simulate multiple devices on a single CPU, so tests
can run anywhere without real GPUs.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import pytest

# Force 4 fake CPU devices for multi-GPU tests.
# Must be set before any JAX operations.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

from vibe_rl.algorithms.ppo import PPO, PPOConfig
from vibe_rl.checkpoint import load_checkpoint, save_checkpoint
from vibe_rl.env import make
from vibe_rl.env.wrappers import AutoResetWrapper
from vibe_rl.runner import (
    PPOMetricsHistory,
    PPOTrainState,
    RunnerConfig,
    get_num_devices,
    replicate,
    train_ppo_multigpu,
    unreplicate,
)


# ---------------------------------------------------------------------------
# Device utility tests
# ---------------------------------------------------------------------------


class TestDeviceUtils:
    def test_get_num_devices_auto(self):
        n = get_num_devices(None)
        assert n >= 1

    def test_get_num_devices_explicit(self):
        n = get_num_devices(2)
        assert n == 2

    def test_get_num_devices_too_many(self):
        with pytest.raises(ValueError, match="Requested .* devices"):
            get_num_devices(9999)

    def test_replicate_shape(self):
        x = jnp.ones((3, 4))
        replicated = replicate(x, 2)
        assert replicated.shape == (2, 3, 4)

    def test_replicate_unreplicate_roundtrip(self):
        x = jnp.array([1.0, 2.0, 3.0])
        r = replicate(x, 4)
        u = unreplicate(r)
        assert jnp.array_equal(u, x)

    def test_replicate_pytree(self):
        """Replicate works on NamedTuples and nested pytrees."""
        state = {"w": jnp.ones((2, 3)), "b": jnp.zeros(3)}
        r = replicate(state, 2)
        assert r["w"].shape == (2, 2, 3)
        assert r["b"].shape == (2, 3)

    def test_unreplicate_pytree(self):
        state = {"w": jnp.ones((4, 2, 3)), "b": jnp.zeros((4, 3))}
        u = unreplicate(state)
        assert u["w"].shape == (2, 3)
        assert u["b"].shape == (3,)


# ---------------------------------------------------------------------------
# Multi-GPU PPO training tests
# ---------------------------------------------------------------------------


class TestTrainPPOMultiGPU:
    def test_basic_multigpu_training(self):
        """Run multi-GPU PPO training with 2 fake devices."""
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        ppo_config = PPOConfig(
            n_steps=8,
            n_minibatches=2,
            n_epochs=2,
            hidden_sizes=(8, 8),
        )
        runner_config = RunnerConfig(
            total_timesteps=128,
            seed=0,
            num_devices=2,
            num_envs=2,
        )

        train_state, history = train_ppo_multigpu(
            env, env_params,
            ppo_config=ppo_config,
            runner_config=runner_config,
        )

        # n_updates = 128 // (2 devices * 2 envs * 8 steps) = 4
        n_updates = 128 // (2 * 2 * 8)
        assert n_updates == 4
        assert isinstance(train_state, PPOTrainState)
        assert isinstance(history, PPOMetricsHistory)

        # shard_map outer axis is devices, scan inner axis is updates
        # -> shape is (n_devices, n_updates)
        assert history.total_loss.shape == (2, n_updates)
        assert history.actor_loss.shape == (2, n_updates)

        # Agent state has leading device dim
        assert train_state.agent_state.step.shape == (2,)
        assert int(train_state.agent_state.step[0]) == n_updates

    def test_single_device_multigpu_runner(self):
        """Multi-GPU runner works with num_devices=1 (single GPU fallback)."""
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        ppo_config = PPOConfig(
            n_steps=16,
            n_minibatches=2,
            n_epochs=1,
            hidden_sizes=(8, 8),
        )
        runner_config = RunnerConfig(
            total_timesteps=64,
            seed=42,
            num_devices=1,
            num_envs=2,
        )

        train_state, history = train_ppo_multigpu(
            env, env_params,
            ppo_config=ppo_config,
            runner_config=runner_config,
        )

        # n_updates = 64 // (1 * 2 * 16) = 2
        n_updates = 64 // (1 * 2 * 16)
        assert n_updates == 2
        assert history.total_loss.shape == (1, n_updates)
        assert int(train_state.agent_state.step[0]) == n_updates

    def test_multigpu_metrics_finite(self):
        """All metrics should be finite (no NaN/inf)."""
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        ppo_config = PPOConfig(
            n_steps=8,
            n_minibatches=2,
            n_epochs=2,
            hidden_sizes=(8, 8),
        )
        runner_config = RunnerConfig(
            total_timesteps=64,
            seed=0,
            num_devices=2,
            num_envs=2,
        )

        _, history = train_ppo_multigpu(
            env, env_params,
            ppo_config=ppo_config,
            runner_config=runner_config,
        )

        assert jnp.all(jnp.isfinite(history.total_loss))
        assert jnp.all(jnp.isfinite(history.actor_loss))
        assert jnp.all(jnp.isfinite(history.critic_loss))
        assert jnp.all(jnp.isfinite(history.entropy))
        assert jnp.all(jnp.isfinite(history.approx_kl))

    def test_multigpu_deterministic(self):
        """Same seed produces same results."""
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        ppo_config = PPOConfig(
            n_steps=8,
            n_minibatches=2,
            n_epochs=1,
            hidden_sizes=(8, 8),
        )
        runner_config = RunnerConfig(
            total_timesteps=64,
            seed=7,
            num_devices=2,
            num_envs=2,
        )

        _, h1 = train_ppo_multigpu(
            env, env_params,
            ppo_config=ppo_config,
            runner_config=runner_config,
        )
        _, h2 = train_ppo_multigpu(
            env, env_params,
            ppo_config=ppo_config,
            runner_config=runner_config,
        )

        assert jnp.allclose(h1.total_loss, h2.total_loss)

    def test_multigpu_gradient_sync(self):
        """Verify that params are synchronised across devices (pmean effect)."""
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        ppo_config = PPOConfig(
            n_steps=8,
            n_minibatches=2,
            n_epochs=1,
            hidden_sizes=(8, 8),
        )
        runner_config = RunnerConfig(
            total_timesteps=64,
            seed=0,
            num_devices=2,
            num_envs=2,
        )

        train_state, _ = train_ppo_multigpu(
            env, env_params,
            ppo_config=ppo_config,
            runner_config=runner_config,
        )

        # After training with pmean, params on device 0 and device 1
        # should be identical (they started identical and gradients
        # were averaged across devices at every step).
        params_leaves = jax.tree.leaves(train_state.agent_state.params)
        for leaf in params_leaves:
            if hasattr(leaf, 'shape') and len(leaf.shape) > 0:
                assert jnp.allclose(leaf[0], leaf[1]), (
                    f"Params diverged across devices! shape={leaf.shape}"
                )

    def test_too_few_timesteps_raises(self):
        """Should raise ValueError when total_timesteps < one update."""
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        ppo_config = PPOConfig(n_steps=128, hidden_sizes=(8, 8))
        runner_config = RunnerConfig(
            total_timesteps=10,  # way too few
            num_devices=2,
            num_envs=4,
        )

        with pytest.raises(ValueError, match="total_timesteps"):
            train_ppo_multigpu(
                env, env_params,
                ppo_config=ppo_config,
                runner_config=runner_config,
            )

    def test_explicit_shapes(self):
        """Can pass obs_shape and n_actions explicitly."""
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        ppo_config = PPOConfig(
            n_steps=8,
            n_minibatches=2,
            n_epochs=1,
            hidden_sizes=(8, 8),
        )
        runner_config = RunnerConfig(
            total_timesteps=64,
            seed=0,
            num_devices=2,
            num_envs=2,
        )

        train_state, history = train_ppo_multigpu(
            env, env_params,
            ppo_config=ppo_config,
            runner_config=runner_config,
            obs_shape=(4,),
            n_actions=2,
        )

        n_updates = 64 // (2 * 2 * 8)
        assert int(train_state.agent_state.step[0]) == n_updates

    def test_four_devices(self):
        """Verify training with 4 fake devices."""
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        ppo_config = PPOConfig(
            n_steps=4,
            n_minibatches=2,
            n_epochs=1,
            hidden_sizes=(8, 8),
        )
        runner_config = RunnerConfig(
            total_timesteps=64,
            seed=0,
            num_devices=4,
            num_envs=1,
        )

        train_state, history = train_ppo_multigpu(
            env, env_params,
            ppo_config=ppo_config,
            runner_config=runner_config,
        )

        # n_updates = 64 // (4 * 1 * 4) = 4
        n_updates = 64 // (4 * 1 * 4)
        assert history.total_loss.shape == (4, n_updates)
        assert train_state.agent_state.step.shape == (4,)


# ---------------------------------------------------------------------------
# Sharding module tests
# ---------------------------------------------------------------------------


class TestShardingModule:
    def test_make_mesh_default(self):
        """Default mesh with fsdp_devices=1 creates a (n_devices, 1) mesh."""
        from vibe_rl.sharding import make_mesh
        mesh = make_mesh(num_fsdp_devices=1)
        assert mesh.axis_names == ("batch", "fsdp")
        n_devices = len(jax.devices())
        assert mesh.shape == {"batch": n_devices, "fsdp": 1}

    def test_make_mesh_fsdp(self):
        """Mesh with fsdp_devices=2 splits devices across both axes."""
        from vibe_rl.sharding import make_mesh
        mesh = make_mesh(num_fsdp_devices=2)
        assert mesh.axis_names == ("batch", "fsdp")
        n_devices = len(jax.devices())
        assert mesh.shape == {"batch": n_devices // 2, "fsdp": 2}

    def test_make_mesh_invalid_fsdp(self):
        """Should raise when fsdp_devices doesn't divide device count."""
        from vibe_rl.sharding import make_mesh
        with pytest.raises(ValueError, match="divisible"):
            make_mesh(num_fsdp_devices=3)  # 4 devices not divisible by 3

    def test_data_sharding(self):
        """data_sharding produces a PartitionSpec with DATA_AXIS."""
        from vibe_rl.sharding import data_sharding, make_mesh
        mesh = make_mesh()
        spec = data_sharding(mesh)
        assert spec.spec == (("batch", "fsdp"),)

    def test_replicate_sharding(self):
        """replicate_sharding produces an empty PartitionSpec."""
        from vibe_rl.sharding import make_mesh, replicate_sharding
        mesh = make_mesh()
        spec = replicate_sharding(mesh)
        assert spec.spec == ()


# ---------------------------------------------------------------------------
# Checkpoint with unreplicate/replicate tests
# ---------------------------------------------------------------------------


class TestCheckpointMultiDevice:
    def test_save_unreplicate_load(self, tmp_path: Path):
        """Save with unreplicate, load into single-device template."""
        state = {
            "w": jnp.ones((2, 3, 4)),  # (n_devices=2, 3, 4)
            "b": jnp.zeros((2, 4)),     # (n_devices=2, 4)
        }

        save_checkpoint(tmp_path / "ckpt", state, unreplicate=True)

        template = {"w": jnp.zeros((3, 4)), "b": jnp.zeros(4)}
        loaded = load_checkpoint(tmp_path / "ckpt", template)

        assert loaded["w"].shape == (3, 4)
        assert loaded["b"].shape == (4,)
        assert jnp.array_equal(loaded["w"], jnp.ones((3, 4)))

    def test_load_with_replicate(self, tmp_path: Path):
        """Load and replicate to n_devices."""
        state = {"w": jnp.ones((3, 4)), "b": jnp.zeros(4)}
        save_checkpoint(tmp_path / "ckpt", state)

        template = {"w": jnp.zeros((3, 4)), "b": jnp.zeros(4)}
        loaded = load_checkpoint(
            tmp_path / "ckpt", template, replicate_to=3,
        )

        assert loaded["w"].shape == (3, 3, 4)
        assert loaded["b"].shape == (3, 4)

    def test_unreplicate_save_replicate_load_roundtrip(self, tmp_path: Path):
        """Full roundtrip: replicated -> save(unreplicate) -> load(replicate)."""
        original = {"w": jnp.array([1.0, 2.0, 3.0])}
        replicated = replicate(original, 4)
        assert replicated["w"].shape == (4, 3)

        save_checkpoint(tmp_path / "ckpt", replicated, unreplicate=True)

        template = {"w": jnp.zeros(3)}
        restored = load_checkpoint(
            tmp_path / "ckpt", template, replicate_to=2,
        )
        assert restored["w"].shape == (2, 3)
        assert jnp.array_equal(restored["w"][0], original["w"])
        assert jnp.array_equal(restored["w"][1], original["w"])

    def test_multigpu_train_checkpoint_roundtrip(self, tmp_path: Path):
        """Train with multi-GPU, save unreplicated, verify loadable."""
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        ppo_config = PPOConfig(
            n_steps=8,
            n_minibatches=2,
            n_epochs=1,
            hidden_sizes=(8, 8),
        )
        runner_config = RunnerConfig(
            total_timesteps=64,
            seed=0,
            num_devices=2,
            num_envs=2,
        )

        train_state, _ = train_ppo_multigpu(
            env, env_params,
            ppo_config=ppo_config,
            runner_config=runner_config,
        )

        # Save with unreplicate
        save_checkpoint(
            tmp_path / "ckpt",
            train_state.agent_state,
            unreplicate=True,
        )

        # Create a single-device template
        template = PPO.init(
            jax.random.PRNGKey(99),
            obs_shape=(4,),
            n_actions=2,
            config=ppo_config,
        )

        loaded = load_checkpoint(tmp_path / "ckpt", template)

        # Should have single-device shapes
        assert loaded.step.shape == ()


# ---------------------------------------------------------------------------
# RunnerConfig multi-GPU fields
# ---------------------------------------------------------------------------


class TestRunnerConfigMultiGPU:
    def test_default_num_devices_none(self):
        cfg = RunnerConfig()
        assert cfg.num_devices is None

    def test_default_num_envs(self):
        cfg = RunnerConfig()
        assert cfg.num_envs == 1

    def test_default_fsdp_devices(self):
        cfg = RunnerConfig()
        assert cfg.fsdp_devices == 1

    def test_custom_values(self):
        cfg = RunnerConfig(num_devices=4, num_envs=8, fsdp_devices=2)
        assert cfg.num_devices == 4
        assert cfg.num_envs == 8
        assert cfg.fsdp_devices == 2

    def test_frozen(self):
        cfg = RunnerConfig(num_devices=2)
        with pytest.raises(AttributeError):
            cfg.num_devices = 4  # type: ignore
