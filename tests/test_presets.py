"""Tests for preset configuration registry and CLI."""

from __future__ import annotations

import pytest

from vibe_rl.algorithms.dqn.config import DQNConfig
from vibe_rl.algorithms.ppo.config import PPOConfig
from vibe_rl.algorithms.sac.config import SACConfig
from vibe_rl.configs.presets import PRESETS, TrainConfig, cli


class TestPresetRegistry:
    """Verify that all presets are well-formed."""

    def test_presets_non_empty(self) -> None:
        assert len(PRESETS) > 0

    @pytest.mark.parametrize("name", list(PRESETS.keys()))
    def test_preset_structure(self, name: str) -> None:
        desc, config = PRESETS[name]
        assert isinstance(desc, str) and len(desc) > 0
        assert isinstance(config, TrainConfig)
        assert isinstance(config.env_id, str) and len(config.env_id) > 0

    @pytest.mark.parametrize("name", list(PRESETS.keys()))
    def test_preset_algo_type(self, name: str) -> None:
        _, config = PRESETS[name]
        assert isinstance(config.algo, (PPOConfig, DQNConfig, SACConfig))

    def test_expected_presets_exist(self) -> None:
        expected = {"cartpole_ppo", "cartpole_dqn", "pendulum_sac", "gridworld_dqn"}
        assert expected.issubset(set(PRESETS.keys()))


class TestCLI:
    """Verify overridable_config_cli integration."""

    def test_select_preset(self) -> None:
        config = cli(["cartpole_ppo"])
        assert config.env_id == "CartPole-v1"
        assert isinstance(config.algo, PPOConfig)

    def test_override_algo_field(self) -> None:
        config = cli(["cartpole_ppo", "--algo.lr", "1e-3"])
        assert isinstance(config.algo, PPOConfig)
        assert config.algo.lr == pytest.approx(1e-3)

    def test_override_runner_field(self) -> None:
        config = cli(["cartpole_ppo", "--runner.total_timesteps", "500000"])
        assert config.runner.total_timesteps == 500_000

    def test_override_env_id(self) -> None:
        config = cli(["cartpole_ppo", "--env_id", "GridWorld-v0"])
        assert config.env_id == "GridWorld-v0"

    def test_sac_preset(self) -> None:
        config = cli(["pendulum_sac"])
        assert config.env_id == "Pendulum-v1"
        assert isinstance(config.algo, SACConfig)

    def test_sac_override_batch_size(self) -> None:
        config = cli(["pendulum_sac", "--algo.batch_size", "512"])
        assert isinstance(config.algo, SACConfig)
        assert config.algo.batch_size == 512

    def test_dqn_preset(self) -> None:
        config = cli(["cartpole_dqn"])
        assert config.env_id == "CartPole-v1"
        assert isinstance(config.algo, DQNConfig)

    def test_dqn_override_batch_size(self) -> None:
        config = cli(["cartpole_dqn", "--algo.batch_size", "128"])
        assert isinstance(config.algo, DQNConfig)
        assert config.algo.batch_size == 128

    def test_multiple_overrides(self) -> None:
        config = cli([
            "cartpole_ppo",
            "--algo.lr", "1e-3",
            "--algo.n_steps", "256",
            "--runner.seed", "42",
        ])
        assert config.algo.lr == pytest.approx(1e-3)
        assert config.algo.n_steps == 256
        assert config.runner.seed == 42

    def test_preset_defaults_preserved_without_override(self) -> None:
        config = cli(["cartpole_ppo"])
        _, preset = PRESETS["cartpole_ppo"]
        assert config.algo.lr == preset.algo.lr
        assert config.algo.n_steps == preset.algo.n_steps
        assert config.runner.total_timesteps == preset.runner.total_timesteps

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(SystemExit):
            cli(["nonexistent_preset"])
