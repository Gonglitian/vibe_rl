#!/usr/bin/env python3
"""Serve a trained policy over WebSocket.

Start a WebSocket inference server that accepts observations and returns
actions.  Uses MessagePack binary serialization for efficient transport
of image observations.

Usage::

    python scripts/serve_policy.py cartpole_ppo \\
        --checkpoint_dir runs/cartpole/checkpoints
    python scripts/serve_policy.py pendulum_sac \\
        --checkpoint_dir runs/pendulum/checkpoints --port 9000
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import tyro

from vibe_rl.configs.presets import PRESETS, TrainConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServeConfig:
    """Configuration for the policy serving script."""

    # Model config — select a preset to define architecture / algorithm
    config: TrainConfig = field(default_factory=TrainConfig)

    # Checkpoint path (required)
    checkpoint_dir: str = ""

    # Optional checkpoint step (latest if omitted)
    step: int | None = None

    # Optional path to normalization statistics
    norm_stats_path: str | None = None

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    )

    serve_config = tyro.extras.overridable_config_cli(
        {
            name: (desc, ServeConfig(config=cfg))
            for name, (desc, cfg) in PRESETS.items()
        },
        use_underscores=True,
    )

    if not serve_config.checkpoint_dir:
        raise SystemExit("--checkpoint_dir is required")

    checkpoint_dir = Path(serve_config.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise SystemExit(f"Checkpoint directory not found: {checkpoint_dir}")

    # Build policy from checkpoint
    from vibe_rl.policies import create_trained_policy

    norm_path = serve_config.norm_stats_path
    policy = create_trained_policy(
        serve_config.config,
        checkpoint_dir,
        step=serve_config.step,
        norm_stats_path=norm_path,
    )
    logger.info("Policy loaded from %s", checkpoint_dir)

    # Warm up JIT
    import jax.numpy as jnp

    from vibe_rl.env import make

    env, env_params = make(serve_config.config.env_id)
    obs_shape = env.observation_space(env_params).shape
    dummy_obs = jnp.zeros(obs_shape)
    _ = policy.infer(dummy_obs)
    logger.info("JIT warmup complete")

    # Start server
    from vibe_rl.serving import PolicyServer

    server = PolicyServer(
        policy=policy,
        host=serve_config.host,
        port=serve_config.port,
    )
    logger.info(
        "Starting WebSocket server on ws://%s:%d",
        serve_config.host,
        serve_config.port,
    )
    server.run_blocking()


if __name__ == "__main__":
    main()
