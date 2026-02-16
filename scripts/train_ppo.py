#!/usr/bin/env python3
"""Train PPO via CLI.

Usage::

    python scripts/train_ppo.py --help
    python scripts/train_ppo.py --env-id CartPole-v1
    python scripts/train_ppo.py --env-id CartPole-v1 --ppo.lr 1e-3 --ppo.n-steps 256
    python scripts/train_ppo.py --runner.total-timesteps 500000 --runner.seed 123
"""

from __future__ import annotations

from dataclasses import dataclass

import tyro

from vibe_rl.algorithms.ppo.config import PPOConfig
from vibe_rl.env import make
from vibe_rl.env.wrappers import AutoResetWrapper
from vibe_rl.runner import RunnerConfig, train_ppo


@dataclass(frozen=True)
class TrainPPOArgs:
    """PPO training configuration."""

    # Environment
    env_id: str = "CartPole-v1"

    # Algorithm hyperparameters
    ppo: PPOConfig = PPOConfig()

    # Runner / outer-loop settings
    runner: RunnerConfig = RunnerConfig()


def main(args: TrainPPOArgs) -> None:
    env, env_params = make(args.env_id)
    env = AutoResetWrapper(env)
    env_params = env.default_params()

    train_state, metrics = train_ppo(
        env,
        env_params,
        ppo_config=args.ppo,
        runner_config=args.runner,
    )

    final_loss = float(metrics.total_loss[-1])
    final_entropy = float(metrics.entropy[-1])
    print(
        f"Training complete | "
        f"final_loss={final_loss:.4f} | "
        f"final_entropy={final_entropy:.4f}"
    )


if __name__ == "__main__":
    main(tyro.cli(TrainPPOArgs))
