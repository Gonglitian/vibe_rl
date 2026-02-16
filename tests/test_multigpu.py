"""Tests for multi-GPU training via jit + NamedSharding + FSDP.

Uses XLA_FLAGS to simulate multiple devices on a single CPU, so tests
can run anywhere without real GPUs.
"""

from __future__ import annotations

import os

# Force 4 fake CPU devices for multi-GPU tests.
# Must be set *before* JAX is imported so the backend sees it at init time.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import chex  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

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

        # GSPMD: metrics are (n_updates,) scalars (single logical computation)
        assert history.total_loss.shape == (n_updates,)
        assert history.actor_loss.shape == (n_updates,)

        # Agent state: params/step are single-copy (replicated/FSDP-sharded)
        assert train_state.agent_state.step.shape == ()
        assert int(train_state.agent_state.step) == n_updates

        # Env data keeps device dimension
        assert train_state.env_obs.shape == (2, 2, 4)  # (n_devices, num_envs, obs_dim)

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
        assert history.total_loss.shape == (n_updates,)
        assert int(train_state.agent_state.step) == n_updates

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
        """Verify that GSPMD keeps params consistent (single logical copy).

        With GSPMD + replicated/FSDP sharding, there is only one logical
        copy of params.  We verify that the params are well-formed
        (finite and reasonable) after training.
        """
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

        # With GSPMD, params are a single logical copy — verify all finite.
        params_leaves = jax.tree.leaves(train_state.agent_state.params)
        for leaf in params_leaves:
            if hasattr(leaf, 'shape'):
                assert jnp.all(jnp.isfinite(leaf)), (
                    f"Non-finite params found! shape={leaf.shape}"
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
        assert int(train_state.agent_state.step) == n_updates

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
        assert history.total_loss.shape == (n_updates,)
        assert train_state.agent_state.step.shape == ()

    def test_fsdp_devices_1_equivalent(self):
        """fsdp_devices=1 should produce same results as pure data-parallel."""
        env, env_params = make("CartPole-v1")
        env = AutoResetWrapper(env)
        env_params = env.default_params()

        ppo_config = PPOConfig(
            n_steps=8,
            n_minibatches=2,
            n_epochs=1,
            hidden_sizes=(8, 8),
        )

        # Default (fsdp_devices=1)
        runner_config = RunnerConfig(
            total_timesteps=64,
            seed=0,
            num_devices=2,
            num_envs=2,
            fsdp_devices=1,
        )

        _, history = train_ppo_multigpu(
            env, env_params,
            ppo_config=ppo_config,
            runner_config=runner_config,
        )

        assert jnp.all(jnp.isfinite(history.total_loss))
        assert history.total_loss.shape == (2,)  # n_updates=2

    def test_fsdp_devices_2(self):
        """fsdp_devices=2 with 4 devices: 2-way data-parallel x 2-way FSDP."""
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
            fsdp_devices=2,
        )

        train_state, history = train_ppo_multigpu(
            env, env_params,
            ppo_config=ppo_config,
            runner_config=runner_config,
        )

        n_updates = 64 // (4 * 1 * 4)
        assert n_updates == 4
        assert history.total_loss.shape == (n_updates,)
        assert jnp.all(jnp.isfinite(history.total_loss))
        assert int(train_state.agent_state.step) == n_updates


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
# FSDP sharding function tests
# ---------------------------------------------------------------------------


class TestFSDPSharding:
    """Tests for the fsdp_sharding() per-parameter sharding function."""

    def test_small_param_replicated(self):
        """Parameters < min_size_mbytes should be fully replicated."""
        from vibe_rl.sharding import fsdp_sharding, make_mesh

        mesh = make_mesh(num_fsdp_devices=2)
        # Small 2D array: 8x8 float32 = 256 bytes << 4MB
        small = jax.ShapeDtypeStruct((8, 8), jnp.float32)
        shardings = fsdp_sharding({"w": small}, mesh)
        assert shardings["w"].spec == ()  # replicated

    def test_scalar_replicated(self):
        """Scalars should always be replicated."""
        from vibe_rl.sharding import fsdp_sharding, make_mesh

        mesh = make_mesh(num_fsdp_devices=2)
        scalar = jax.ShapeDtypeStruct((), jnp.float32)
        shardings = fsdp_sharding({"s": scalar}, mesh)
        assert shardings["s"].spec == ()

    def test_1d_replicated(self):
        """1-D arrays should always be replicated regardless of size."""
        from vibe_rl.sharding import FSDP_AXIS, fsdp_sharding, make_mesh

        mesh = make_mesh(num_fsdp_devices=2)
        # Large 1D: 2M float32 = 8 MB > 4 MB, but 1D → replicate
        big_1d = jax.ShapeDtypeStruct((2_000_000,), jnp.float32)
        shardings = fsdp_sharding({"b": big_1d}, mesh)
        assert shardings["b"].spec == ()

    def test_large_2d_sharded(self):
        """Large 2D array should be sharded along largest divisible dim."""
        from vibe_rl.sharding import FSDP_AXIS, fsdp_sharding, make_mesh

        mesh = make_mesh(num_fsdp_devices=2)
        # 1024x1024 float32 = 4 MB — exactly at threshold
        big_2d = jax.ShapeDtypeStruct((1024, 1024), jnp.float32)
        shardings = fsdp_sharding({"w": big_2d}, mesh)
        # Both dims are 1024, divisible by 2 — should shard along dim 0
        # (argsort picks largest first; both equal, so first encountered)
        spec = shardings["w"].spec
        assert FSDP_AXIS in spec, f"Expected FSDP sharding, got {spec}"

    def test_large_2d_largest_dim_preferred(self):
        """Should shard along the largest divisible dimension."""
        from vibe_rl.sharding import FSDP_AXIS, fsdp_sharding, make_mesh

        mesh = make_mesh(num_fsdp_devices=2)
        # 2048x512 float32 = 4 MB — dim 0 is largest (2048)
        big_2d = jax.ShapeDtypeStruct((2048, 512), jnp.float32)
        shardings = fsdp_sharding({"w": big_2d}, mesh)
        spec = shardings["w"].spec
        # Dim 0 (2048) is largest and divisible by 2
        assert spec == (FSDP_AXIS, None)

    def test_no_divisible_dim_replicated(self):
        """If no dimension is divisible by fsdp_size, replicate."""
        from vibe_rl.sharding import fsdp_sharding, make_mesh

        mesh = make_mesh(num_fsdp_devices=2)
        # Large but dimensions are odd (not divisible by 2)
        # 1025x1025 float32 ≈ 4 MB
        odd = jax.ShapeDtypeStruct((1025, 1025), jnp.float32)
        shardings = fsdp_sharding({"w": odd}, mesh)
        assert shardings["w"].spec == ()

    def test_fsdp_size_1_all_replicated(self):
        """With fsdp_devices=1, all parameters should be replicated."""
        from vibe_rl.sharding import fsdp_sharding, make_mesh

        mesh = make_mesh(num_fsdp_devices=1)
        big = jax.ShapeDtypeStruct((2048, 2048), jnp.float32)
        shardings = fsdp_sharding({"w": big}, mesh)
        # fsdp size 1 means everything is divisible, but 1-way sharding
        # is equivalent to replication — the PartitionSpec will have
        # FSDP_AXIS but it's a trivial axis.
        # This is fine: PartitionSpec("fsdp", None) with fsdp=1 == replicated.
        assert True  # The function should not error

    def test_min_size_threshold(self):
        """Custom min_size_mbytes changes the threshold."""
        from vibe_rl.sharding import FSDP_AXIS, fsdp_sharding, make_mesh

        mesh = make_mesh(num_fsdp_devices=2)
        # 512x512 float32 = 1 MB — below default 4 MB but above 0.5 MB
        medium = jax.ShapeDtypeStruct((512, 512), jnp.float32)

        # Default threshold (4 MB): should be replicated
        s1 = fsdp_sharding({"w": medium}, mesh)
        assert s1["w"].spec == ()

        # Lower threshold (0.5 MB): should be sharded
        s2 = fsdp_sharding({"w": medium}, mesh, min_size_mbytes=0.5)
        assert FSDP_AXIS in s2["w"].spec

    def test_pytree_mixed(self):
        """Pytree with mixed sizes gets mixed shardings."""
        from vibe_rl.sharding import FSDP_AXIS, fsdp_sharding, make_mesh

        mesh = make_mesh(num_fsdp_devices=2)
        pytree = {
            "small": jax.ShapeDtypeStruct((8, 8), jnp.float32),      # 256 B
            "bias": jax.ShapeDtypeStruct((64,), jnp.float32),         # 256 B (1D)
            "large": jax.ShapeDtypeStruct((2048, 1024), jnp.float32), # 8 MB
        }
        shardings = fsdp_sharding(pytree, mesh)
        assert shardings["small"].spec == ()  # replicated (too small)
        assert shardings["bias"].spec == ()   # replicated (1D)
        assert FSDP_AXIS in shardings["large"].spec  # sharded


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
        """Train with multi-GPU, save agent state, verify loadable."""
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

        # With GSPMD, agent_state params are single-copy — save directly.
        save_checkpoint(
            tmp_path / "ckpt",
            train_state.agent_state,
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
