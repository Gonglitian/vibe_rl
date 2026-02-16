"""Preset experiment configurations.

Each preset bundles an environment, algorithm config, and runner settings
tuned for a specific experiment.  Use :func:`cli` in a training script to
get a :class:`TrainConfig` with ``overridable_config_cli`` — the user picks
a preset and optionally overrides individual fields::

    python scripts/train.py cartpole_ppo --ppo.lr 1e-3
    python scripts/train.py pendulum_sac --runner.total-timesteps 200000
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Union

import tyro

from vibe_rl.algorithms.dqn.config import DQNConfig
from vibe_rl.algorithms.ppo.config import PPOConfig
from vibe_rl.algorithms.sac.config import SACConfig
from vibe_rl.runner.config import RunnerConfig

# ---------------------------------------------------------------------------
# Unified training config
# ---------------------------------------------------------------------------

AlgoConfig = Annotated[
    Union[
        Annotated[PPOConfig, tyro.conf.subcommand("ppo", prefix_name=False)],
        Annotated[DQNConfig, tyro.conf.subcommand("dqn", prefix_name=False)],
        Annotated[SACConfig, tyro.conf.subcommand("sac", prefix_name=False)],
    ],
    tyro.conf.AvoidSubcommands,
]


@dataclass(frozen=True)
class TrainConfig:
    """Full training configuration — algorithm + environment + runner."""

    # Environment
    env_id: str = "CartPole-v1"

    # Algorithm (one of PPOConfig / DQNConfig / SACConfig)
    algo: AlgoConfig = field(default_factory=PPOConfig)

    # Runner / outer-loop settings
    runner: RunnerConfig = field(default_factory=RunnerConfig)


# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------

PRESETS: dict[str, tuple[str, TrainConfig]] = {
    "cartpole_ppo": (
        "PPO on CartPole-v1 (fast sanity-check)",
        TrainConfig(
            env_id="CartPole-v1",
            algo=PPOConfig(
                hidden_sizes=(64, 64),
                lr=2.5e-4,
                n_steps=128,
                n_minibatches=4,
                n_epochs=4,
                num_envs=4,
            ),
            runner=RunnerConfig(
                total_timesteps=100_000,
                eval_every=5_000,
            ),
        ),
    ),
    "cartpole_dqn": (
        "DQN on CartPole-v1",
        TrainConfig(
            env_id="CartPole-v1",
            algo=DQNConfig(
                hidden_sizes=(128, 128),
                lr=1e-3,
                batch_size=64,
                target_update_freq=1_000,
                epsilon_decay_steps=50_000,
            ),
            runner=RunnerConfig(
                total_timesteps=100_000,
                eval_every=5_000,
                buffer_size=100_000,
                warmup_steps=1_000,
            ),
        ),
    ),
    "pendulum_ppo": (
        "PPO on Pendulum-v1 (continuous control)",
        TrainConfig(
            env_id="Pendulum-v1",
            algo=PPOConfig(
                hidden_sizes=(64, 64),
                lr=3e-4,
                n_steps=2048,
                n_minibatches=32,
                n_epochs=10,
                num_envs=1,
                ent_coef=0.0,
            ),
            runner=RunnerConfig(
                total_timesteps=200_000,
                eval_every=10_000,
            ),
        ),
    ),
    "pendulum_sac": (
        "SAC on Pendulum-v1 (continuous control)",
        TrainConfig(
            env_id="Pendulum-v1",
            algo=SACConfig(
                hidden_sizes=(256, 256),
                actor_lr=3e-4,
                critic_lr=3e-4,
                alpha_lr=3e-4,
                batch_size=256,
                tau=0.005,
            ),
            runner=RunnerConfig(
                total_timesteps=200_000,
                eval_every=5_000,
                buffer_size=100_000,
                warmup_steps=1_000,
            ),
        ),
    ),
    "gridworld_dqn": (
        "DQN on GridWorld-v0 (tabular-scale discrete)",
        TrainConfig(
            env_id="GridWorld-v0",
            algo=DQNConfig(
                hidden_sizes=(64, 64),
                lr=5e-4,
                batch_size=32,
                target_update_freq=500,
                epsilon_decay_steps=20_000,
            ),
            runner=RunnerConfig(
                total_timesteps=50_000,
                eval_every=5_000,
                buffer_size=50_000,
                warmup_steps=500,
            ),
        ),
    ),
}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def cli(
    args: list[str] | None = None,
    **kwargs: object,
) -> TrainConfig:
    """Parse a preset + overrides from the command line.

    Usage::

        config = cli()                          # parse sys.argv
        config = cli(["cartpole_ppo", "--ppo.lr", "1e-3"])  # explicit args
    """
    return tyro.extras.overridable_config_cli(
        PRESETS,
        args=args,
        use_underscores=True,
        **kwargs,  # type: ignore[arg-type]
    )
