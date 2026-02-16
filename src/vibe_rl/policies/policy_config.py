"""Factory for creating trained policies from checkpoints.

Loads model parameters and normalization statistics from a checkpoint
directory and assembles a complete inference pipeline wrapped in a
:class:`~vibe_rl.policies.policy.Policy`.

Usage::

    from vibe_rl.policies.policy_config import create_trained_policy
    from vibe_rl.configs.presets import TrainConfig

    config = TrainConfig(env_id="Pendulum-v1", algo=SACConfig())
    policy = create_trained_policy(config, "/path/to/checkpoint")
    action = policy.infer(obs)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from vibe_rl.algorithms.dqn.config import DQNConfig
from vibe_rl.algorithms.ppo.config import PPOConfig
from vibe_rl.algorithms.sac.config import SACConfig
from vibe_rl.checkpoint import load_checkpoint, load_eqx, load_metadata
from vibe_rl.configs.presets import TrainConfig
from vibe_rl.data.normalize import NormStats, load_norm_stats
from vibe_rl.policies.policy import (
    ComposeTransforms,
    NormalizeInput,
    Policy,
    UnnormalizeOutput,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-algorithm inference functions
# ---------------------------------------------------------------------------


def _ppo_categorical_infer(model: Any, obs: jax.Array) -> jax.Array:
    """PPO categorical: take the greedy (argmax) action."""
    from vibe_rl.algorithms.ppo.types import ActorCriticParams

    if isinstance(model, ActorCriticParams):
        logits = model.actor(obs)
    else:
        # Shared backbone: returns (logits, value)
        logits, _ = model(obs)
    return jnp.argmax(logits, axis=-1)


def _dqn_infer(model: Any, obs: jax.Array) -> jax.Array:
    """DQN: greedy action from Q-values."""
    q_values = model(obs)
    return jnp.argmax(q_values, axis=-1)


def _sac_infer(model: Any, obs: jax.Array) -> jax.Array:
    """SAC: deterministic action via tanh(mean).

    The model here is the ``GaussianActor``.  We take the mean action
    (no sampling) and apply tanh squashing.  Rescaling to action bounds
    is handled by the output transform if needed.
    """
    mean, _log_std = model(obs)
    return jnp.tanh(mean)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_infer_fn(
    config: TrainConfig,
) -> tuple[Any, Any]:
    """Return ``(infer_fn, model_extractor)`` for the given config.

    ``model_extractor`` is a callable that, given the full checkpoint
    pytree, extracts just the model parameters needed for inference.
    """
    algo = config.algo

    if isinstance(algo, PPOConfig):
        return _ppo_categorical_infer, _extract_ppo_model
    elif isinstance(algo, DQNConfig):
        return _dqn_infer, _extract_dqn_model
    elif isinstance(algo, SACConfig):
        return _sac_infer, _extract_sac_model
    else:
        raise ValueError(f"Unsupported algorithm config: {type(algo).__name__}")


def _extract_ppo_model(state: Any) -> Any:
    """Extract actor (or shared model) from PPOState."""
    return state.params


def _extract_dqn_model(state: Any) -> Any:
    """Extract Q-network from DQNState."""
    return state.params


def _extract_sac_model(state: Any) -> Any:
    """Extract actor from SACState."""
    return state.actor_params


def _build_template_state(config: TrainConfig, key: jax.Array) -> Any:
    """Build a template (skeleton) state for checkpoint deserialization.

    Creates a fresh state with random parameters that has the same
    pytree structure as the checkpoint, so ``load_checkpoint`` can
    restore into it.
    """
    from vibe_rl.algorithms.dqn.agent import DQN
    from vibe_rl.algorithms.ppo.agent import PPO
    from vibe_rl.algorithms.sac.agent import SAC
    from vibe_rl.env import make

    env, env_params = make(config.env_id)
    obs_space = env.observation_space(env_params)
    act_space = env.action_space(env_params)

    algo = config.algo

    if isinstance(algo, PPOConfig):
        from vibe_rl.env.spaces import Discrete

        assert isinstance(act_space, Discrete)
        return PPO.init(key, obs_shape=obs_space.shape, n_actions=act_space.n, config=algo)
    elif isinstance(algo, DQNConfig):
        from vibe_rl.env.spaces import Discrete

        assert isinstance(act_space, Discrete)
        return DQN.init(key, obs_shape=obs_space.shape, n_actions=act_space.n, config=algo)
    elif isinstance(algo, SACConfig):
        from vibe_rl.env.spaces import Box

        assert isinstance(act_space, Box)
        action_dim = act_space.shape[0] if act_space.shape else 1
        return SAC.init(key, obs_shape=obs_space.shape, action_dim=action_dim, config=algo)
    else:
        raise ValueError(f"Unsupported algorithm config: {type(algo).__name__}")


def _build_input_transform(
    norm_stats: dict[str, NormStats] | None,
    obs_key: str = "obs",
) -> Any:
    """Build input transform from normalization statistics."""
    if norm_stats is None or obs_key not in norm_stats:
        return None

    stats = norm_stats[obs_key]
    return NormalizeInput(
        mean=jnp.asarray(stats.mean),
        std=jnp.asarray(stats.std),
    )


def _build_output_transform(
    norm_stats: dict[str, NormStats] | None,
    action_key: str = "action",
) -> Any:
    """Build output transform (action denormalization)."""
    if norm_stats is None or action_key not in norm_stats:
        return None

    stats = norm_stats[action_key]
    return UnnormalizeOutput(
        mean=jnp.asarray(stats.mean),
        std=jnp.asarray(stats.std),
    )


def create_trained_policy(
    config: TrainConfig,
    checkpoint_dir: str | Path,
    *,
    step: int | None = None,
    norm_stats_path: str | Path | None = None,
    obs_norm_key: str = "obs",
    action_norm_key: str = "action",
) -> Policy:
    """Load a trained model from a checkpoint and wrap it as a Policy.

    Parameters
    ----------
    config:
        The training configuration used to produce the checkpoint.
        Needed to reconstruct the model architecture for deserialization.
    checkpoint_dir:
        Path to the checkpoint directory.  Supports both single-step
        checkpoints (``save_checkpoint``) and managed checkpoints
        (``CheckpointManager``).
    step:
        Specific checkpoint step to load.  If ``None``, loads the latest.
    norm_stats_path:
        Path to a ``norm_stats.json`` file with normalization statistics.
        If ``None``, looks for ``norm_stats.json`` inside ``checkpoint_dir``.
        If not found, no normalization is applied.
    obs_norm_key:
        Key in the norm stats dict for observation normalization.
    action_norm_key:
        Key in the norm stats dict for action denormalization.

    Returns
    -------
    A :class:`Policy` ready for inference.
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Build template state for deserialization
    key = jax.random.PRNGKey(0)
    template_state = _build_template_state(config, key)

    # Load checkpoint
    restored_state = load_checkpoint(checkpoint_dir, template_state, step=step)
    logger.info("Loaded checkpoint from %s", checkpoint_dir)

    # Extract just the model needed for inference
    infer_fn, model_extractor = _build_infer_fn(config)
    model = model_extractor(restored_state)

    # Load normalization statistics
    norm_stats = _load_norm_stats_auto(checkpoint_dir, norm_stats_path)

    # Build transforms
    input_transform = _build_input_transform(norm_stats, obs_norm_key)
    output_transform = _build_output_transform(norm_stats, action_norm_key)

    # For SAC, compose the tanh output with action denorm if needed
    if isinstance(config.algo, SACConfig) and output_transform is not None:
        # SAC infer_fn outputs tanh(mean) in [-1, 1].
        # We need to rescale to action bounds, then denormalize.
        algo = config.algo
        rescale = _SACRescale(
            action_low=jnp.float32(algo.action_low),
            action_high=jnp.float32(algo.action_high),
        )
        output_transform = ComposeTransforms(
            transforms=(rescale, output_transform),
        )

    kwargs: dict[str, Any] = {"model": model, "infer_fn": infer_fn}
    if input_transform is not None:
        kwargs["input_transform"] = input_transform
    if output_transform is not None:
        kwargs["output_transform"] = output_transform

    return Policy(**kwargs)


def _load_norm_stats_auto(
    checkpoint_dir: Path,
    explicit_path: str | Path | None,
) -> dict[str, NormStats] | None:
    """Try to load norm stats from explicit path, then checkpoint dir."""
    if explicit_path is not None:
        p = Path(explicit_path)
        if p.exists():
            logger.info("Loading normalization stats from %s", p)
            return load_norm_stats(p)
        logger.warning("Norm stats path %s not found, skipping", p)
        return None

    # Look inside checkpoint directory
    for candidate in [
        checkpoint_dir / "norm_stats.json",
        checkpoint_dir.parent / "norm_stats.json",
    ]:
        if candidate.exists():
            logger.info("Found normalization stats at %s", candidate)
            return load_norm_stats(candidate)

    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass(frozen=True)
class _SACRescale:
    """Rescale tanh output [-1, 1] to [action_low, action_high]."""

    action_low: jax.Array
    action_high: jax.Array

    def __call__(self, action: jax.Array) -> jax.Array:
        return self.action_low + 0.5 * (action + 1.0) * (
            self.action_high - self.action_low
        )
