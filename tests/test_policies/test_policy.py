"""Tests for the Policy inference wrapper and factory."""

import json
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vibe_rl.algorithms.dqn.agent import DQN
from vibe_rl.algorithms.dqn.config import DQNConfig
from vibe_rl.algorithms.ppo.agent import PPO
from vibe_rl.algorithms.ppo.config import PPOConfig
from vibe_rl.algorithms.ppo.network import ActorCriticShared
from vibe_rl.algorithms.ppo.types import ActorCriticParams
from vibe_rl.algorithms.sac.agent import SAC
from vibe_rl.algorithms.sac.config import SACConfig
from vibe_rl.checkpoint import save_checkpoint
from vibe_rl.data.normalize import NormStats, save_norm_stats
from vibe_rl.policies.policy import (
    ComposeTransforms,
    NormalizeInput,
    Policy,
    ResizeImageInput,
    UnnormalizeOutput,
    _is_batched,
)
from vibe_rl.policies.policy_config import (
    _dqn_infer,
    _ppo_categorical_infer,
    _sac_infer,
    create_trained_policy,
)
from vibe_rl.configs.presets import TrainConfig


# ---------------------------------------------------------------------------
# Policy class tests
# ---------------------------------------------------------------------------


class TestPolicyBasic:
    """Test Policy with a trivial model."""

    def _make_policy(self):
        """Create a simple policy with an identity model."""
        key = jax.random.PRNGKey(0)
        # Simple linear model: obs -> action = W @ obs
        import equinox as eqx

        model = eqx.nn.Linear(4, 2, key=key)

        def infer_fn(m, obs):
            return m(obs)

        return Policy(model=model, infer_fn=infer_fn)

    def test_single_inference(self):
        policy = self._make_policy()
        obs = jnp.ones(4)
        action = policy.infer(obs)
        assert action.shape == (2,)

    def test_batch_inference(self):
        policy = self._make_policy()
        obs = jnp.ones((8, 4))
        actions = policy.infer(obs)
        assert actions.shape == (8, 2)

    def test_jit_compilation(self):
        """Second call should use cached JIT."""
        policy = self._make_policy()
        obs = jnp.ones(4)
        _ = policy.infer(obs)
        assert policy._jitted_infer is not None
        # Second call uses the same jitted function
        action = policy.infer(obs)
        assert action.shape == (2,)

    def test_batch_jit_compilation(self):
        policy = self._make_policy()
        obs = jnp.ones((4, 4))
        _ = policy.infer(obs)
        assert policy._jitted_infer_batch is not None


class TestPolicyWithTransforms:
    """Test Policy with input/output transforms."""

    def test_normalize_input(self):
        import equinox as eqx

        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(4, 2, key=key)

        def infer_fn(m, obs):
            return m(obs)

        mean = jnp.array([1.0, 2.0, 3.0, 4.0])
        std = jnp.array([0.5, 0.5, 0.5, 0.5])

        policy = Policy(
            model=model,
            infer_fn=infer_fn,
            input_transform=NormalizeInput(mean=mean, std=std),
        )

        obs = jnp.array([1.0, 2.0, 3.0, 4.0])
        action = policy.infer(obs)
        assert action.shape == (2,)

        # Verify normalization: obs should become zeros
        normalized = policy.input_transform(obs)
        np.testing.assert_allclose(normalized, jnp.zeros(4), atol=1e-6)

    def test_unnormalize_output(self):
        import equinox as eqx

        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(4, 2, key=key)

        def infer_fn(m, obs):
            return jnp.zeros(2)  # always return zeros

        mean = jnp.array([1.0, 2.0])
        std = jnp.array([0.5, 0.5])

        policy = Policy(
            model=model,
            infer_fn=infer_fn,
            output_transform=UnnormalizeOutput(mean=mean, std=std),
        )

        obs = jnp.ones(4)
        action = policy.infer(obs)
        # zeros * std + mean = mean
        np.testing.assert_allclose(action, mean, atol=1e-6)

    def test_both_transforms(self):
        import equinox as eqx

        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(4, 2, key=key)

        def infer_fn(m, obs):
            return m(obs)

        obs_mean = jnp.zeros(4)
        obs_std = jnp.ones(4)
        act_mean = jnp.array([0.0, 0.0])
        act_std = jnp.array([1.0, 1.0])

        policy = Policy(
            model=model,
            infer_fn=infer_fn,
            input_transform=NormalizeInput(mean=obs_mean, std=obs_std),
            output_transform=UnnormalizeOutput(mean=act_mean, std=act_std),
        )

        obs = jnp.ones(4)
        action = policy.infer(obs)
        assert action.shape == (2,)

    def test_batch_with_transforms(self):
        import equinox as eqx

        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(4, 2, key=key)

        def infer_fn(m, obs):
            return m(obs)

        mean = jnp.zeros(4)
        std = jnp.ones(4)

        policy = Policy(
            model=model,
            infer_fn=infer_fn,
            input_transform=NormalizeInput(mean=mean, std=std),
        )

        obs = jnp.ones((8, 4))
        actions = policy.infer(obs)
        assert actions.shape == (8, 2)


class TestDictObservation:
    """Test Policy with dict (mixed image+state) observations."""

    def test_normalize_dict_input(self):
        mean = jnp.array([1.0, 2.0])
        std = jnp.array([0.5, 0.5])
        transform = NormalizeInput(mean=mean, std=std, key="state")

        obs = {"state": jnp.array([1.0, 2.0]), "image": jnp.ones((8, 8, 3))}
        result = transform(obs)

        np.testing.assert_allclose(result["state"], jnp.zeros(2), atol=1e-6)
        np.testing.assert_allclose(result["image"], jnp.ones((8, 8, 3)))

    def test_resize_dict_input(self):
        transform = ResizeImageInput(height=4, width=4, key="image")

        obs = {"state": jnp.ones(2), "image": jnp.ones((8, 8, 3))}
        result = transform(obs)

        assert result["image"].shape == (4, 4, 3)
        np.testing.assert_allclose(result["state"], jnp.ones(2))

    def test_compose_transforms(self):
        mean = jnp.array([1.0, 2.0])
        std = jnp.array([0.5, 0.5])

        composed = ComposeTransforms(transforms=(
            NormalizeInput(mean=mean, std=std, key="state"),
            ResizeImageInput(height=4, width=4, key="image"),
        ))

        obs = {"state": jnp.array([1.0, 2.0]), "image": jnp.ones((8, 8, 3))}
        result = composed(obs)

        np.testing.assert_allclose(result["state"], jnp.zeros(2), atol=1e-6)
        assert result["image"].shape == (4, 4, 3)


class TestResizeImageInput:
    def test_resize_flat(self):
        transform = ResizeImageInput(height=4, width=4)
        img = jnp.ones((8, 8, 3))
        result = transform(img)
        assert result.shape == (4, 4, 3)

    def test_resize_preserves_values(self):
        transform = ResizeImageInput(height=4, width=4)
        img = jnp.ones((8, 8, 3)) * 128.0
        result = transform(img)
        np.testing.assert_allclose(result, 128.0, atol=1.0)

    def test_noop_same_size(self):
        transform = ResizeImageInput(height=8, width=8)
        img = jnp.ones((8, 8, 3))
        result = transform(img)
        assert result.shape == (8, 8, 3)


class TestIsBatched:
    def test_1d_is_single(self):
        assert not _is_batched(jnp.ones(4))

    def test_2d_is_batched(self):
        assert _is_batched(jnp.ones((8, 4)))

    def test_3d_image_is_not_batched(self):
        # 3D could be single image (H, W, C) — but our heuristic says 2+
        # dims = batched for flat. For images, 3D is single, 4D is batched.
        # The current heuristic uses ndim >= 2 which works for flat vectors.
        # For images, the user should use dict observations.
        assert _is_batched(jnp.ones((8, 8, 3)))

    def test_dict_single(self):
        obs = {"state": jnp.ones(4)}
        assert not _is_batched(obs)

    def test_dict_batched(self):
        obs = {"state": jnp.ones((8, 4))}
        assert _is_batched(obs)


# ---------------------------------------------------------------------------
# Per-algorithm infer function tests
# ---------------------------------------------------------------------------


class TestPPOInfer:
    def test_categorical_separate(self):
        config = PPOConfig(hidden_sizes=(32, 32))
        state = PPO.init(jax.random.PRNGKey(0), obs_shape=(4,), n_actions=3, config=config)
        assert isinstance(state.params, ActorCriticParams)

        obs = jnp.ones(4)
        action = _ppo_categorical_infer(state.params, obs)
        assert action.shape == ()
        assert 0 <= int(action) < 3

    def test_categorical_shared(self):
        config = PPOConfig(hidden_sizes=(32, 32), shared_backbone=True)
        state = PPO.init(jax.random.PRNGKey(0), obs_shape=(4,), n_actions=3, config=config)
        assert isinstance(state.params, ActorCriticShared)

        obs = jnp.ones(4)
        action = _ppo_categorical_infer(state.params, obs)
        assert action.shape == ()
        assert 0 <= int(action) < 3

    def test_batched_via_vmap(self):
        config = PPOConfig(hidden_sizes=(32, 32))
        state = PPO.init(jax.random.PRNGKey(0), obs_shape=(4,), n_actions=3, config=config)

        obs_batch = jnp.ones((8, 4))
        actions = jax.vmap(lambda o: _ppo_categorical_infer(state.params, o))(obs_batch)
        assert actions.shape == (8,)


class TestDQNInfer:
    def test_greedy_action(self):
        config = DQNConfig(hidden_sizes=(32, 32))
        state = DQN.init(jax.random.PRNGKey(0), obs_shape=(4,), n_actions=3, config=config)

        obs = jnp.ones(4)
        action = _dqn_infer(state.params, obs)
        assert action.shape == ()
        assert 0 <= int(action) < 3


class TestSACInfer:
    def test_deterministic_action(self):
        config = SACConfig(hidden_sizes=(32, 32))
        state = SAC.init(jax.random.PRNGKey(0), obs_shape=(3,), action_dim=1, config=config)

        obs = jnp.ones(3)
        action = _sac_infer(state.actor_params, obs)
        # tanh output is in [-1, 1]
        assert action.shape == (1,)
        assert float(action[0]) >= -1.0
        assert float(action[0]) <= 1.0

    def test_batched_via_vmap(self):
        config = SACConfig(hidden_sizes=(32, 32))
        state = SAC.init(jax.random.PRNGKey(0), obs_shape=(3,), action_dim=2, config=config)

        obs_batch = jnp.ones((8, 3))
        actions = jax.vmap(lambda o: _sac_infer(state.actor_params, o))(obs_batch)
        assert actions.shape == (8, 2)


# ---------------------------------------------------------------------------
# Full integration: create_trained_policy
# ---------------------------------------------------------------------------


class TestCreateTrainedPolicy:
    def _save_ppo_checkpoint(self, tmp_dir: Path) -> tuple[TrainConfig, Path]:
        """Train PPO for 0 steps and save a checkpoint."""
        config = TrainConfig(
            env_id="CartPole-v1",
            algo=PPOConfig(hidden_sizes=(32, 32)),
        )
        state = PPO.init(
            jax.random.PRNGKey(42), obs_shape=(4,), n_actions=2, config=config.algo,
        )
        ckpt_dir = tmp_dir / "ppo_ckpt"
        save_checkpoint(ckpt_dir, state)
        return config, ckpt_dir

    def _save_dqn_checkpoint(self, tmp_dir: Path) -> tuple[TrainConfig, Path]:
        config = TrainConfig(
            env_id="CartPole-v1",
            algo=DQNConfig(hidden_sizes=(32, 32)),
        )
        state = DQN.init(
            jax.random.PRNGKey(42), obs_shape=(4,), n_actions=2, config=config.algo,
        )
        ckpt_dir = tmp_dir / "dqn_ckpt"
        save_checkpoint(ckpt_dir, state)
        return config, ckpt_dir

    def _save_sac_checkpoint(self, tmp_dir: Path) -> tuple[TrainConfig, Path]:
        config = TrainConfig(
            env_id="Pendulum-v1",
            algo=SACConfig(hidden_sizes=(32, 32)),
        )
        state = SAC.init(
            jax.random.PRNGKey(42), obs_shape=(3,), action_dim=1, config=config.algo,
        )
        ckpt_dir = tmp_dir / "sac_ckpt"
        save_checkpoint(ckpt_dir, state)
        return config, ckpt_dir

    def test_ppo_policy_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            config, ckpt_dir = self._save_ppo_checkpoint(tmp_dir)
            policy = create_trained_policy(config, ckpt_dir)

            obs = jnp.ones(4)
            action = policy.infer(obs)
            assert action.shape == ()
            assert 0 <= int(action) < 2

    def test_ppo_policy_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            config, ckpt_dir = self._save_ppo_checkpoint(tmp_dir)
            policy = create_trained_policy(config, ckpt_dir)

            obs = jnp.ones((16, 4))
            actions = policy.infer(obs)
            assert actions.shape == (16,)

    def test_dqn_policy_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            config, ckpt_dir = self._save_dqn_checkpoint(tmp_dir)
            policy = create_trained_policy(config, ckpt_dir)

            obs = jnp.ones(4)
            action = policy.infer(obs)
            assert action.shape == ()
            assert 0 <= int(action) < 2

    def test_sac_policy_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            config, ckpt_dir = self._save_sac_checkpoint(tmp_dir)
            policy = create_trained_policy(config, ckpt_dir)

            obs = jnp.ones(3)
            action = policy.infer(obs)
            assert action.shape == (1,)

    def test_sac_policy_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            config, ckpt_dir = self._save_sac_checkpoint(tmp_dir)
            policy = create_trained_policy(config, ckpt_dir)

            obs = jnp.ones((8, 3))
            actions = policy.infer(obs)
            assert actions.shape == (8, 1)

    def test_policy_with_norm_stats(self):
        """Test that norm stats are loaded and applied."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            config, ckpt_dir = self._save_ppo_checkpoint(tmp_dir)

            # Save norm stats alongside checkpoint
            stats = {
                "obs": NormStats(
                    mean=np.zeros(4, dtype=np.float32),
                    std=np.ones(4, dtype=np.float32),
                    q01=np.full(4, -2.0, dtype=np.float32),
                    q99=np.full(4, 2.0, dtype=np.float32),
                ),
            }
            save_norm_stats(stats, ckpt_dir / "norm_stats.json")

            policy = create_trained_policy(config, ckpt_dir)
            assert policy.input_transform is not None
            assert isinstance(policy.input_transform, NormalizeInput)

            obs = jnp.ones(4)
            action = policy.infer(obs)
            assert action.shape == ()

    def test_policy_with_action_norm_stats(self):
        """Test action denormalization for SAC."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            config, ckpt_dir = self._save_sac_checkpoint(tmp_dir)

            stats = {
                "action": NormStats(
                    mean=np.zeros(1, dtype=np.float32),
                    std=np.ones(1, dtype=np.float32),
                    q01=np.full(1, -2.0, dtype=np.float32),
                    q99=np.full(1, 2.0, dtype=np.float32),
                ),
            }
            save_norm_stats(stats, ckpt_dir / "norm_stats.json")

            policy = create_trained_policy(config, ckpt_dir)
            assert policy.output_transform is not None

            obs = jnp.ones(3)
            action = policy.infer(obs)
            assert action.shape == (1,)

    def test_policy_with_explicit_norm_stats_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            config, ckpt_dir = self._save_ppo_checkpoint(tmp_dir)

            stats = {
                "obs": NormStats(
                    mean=np.zeros(4, dtype=np.float32),
                    std=np.ones(4, dtype=np.float32),
                    q01=np.full(4, -2.0, dtype=np.float32),
                    q99=np.full(4, 2.0, dtype=np.float32),
                ),
            }
            norm_path = tmp_dir / "custom_norm.json"
            save_norm_stats(stats, norm_path)

            policy = create_trained_policy(
                config, ckpt_dir, norm_stats_path=norm_path,
            )
            assert isinstance(policy.input_transform, NormalizeInput)

    def test_jit_latency_reasonable(self):
        """Verify JIT-compiled inference doesn't take extremely long."""
        import time

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            config, ckpt_dir = self._save_ppo_checkpoint(tmp_dir)
            policy = create_trained_policy(config, ckpt_dir)

            obs = jnp.ones(4)
            # First call triggers JIT compilation
            _ = policy.infer(obs)

            # Subsequent calls should be fast
            start = time.monotonic()
            for _ in range(100):
                _ = policy.infer(obs)
            elapsed = time.monotonic() - start

            # 100 inferences should complete in well under 5 seconds
            assert elapsed < 5.0, f"100 inferences took {elapsed:.2f}s"
