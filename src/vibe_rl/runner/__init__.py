"""Training runners for vibe_rl algorithms.

Two styles:

- **PureJaxRL** (PPO): The entire training loop lives inside
  ``jax.lax.scan`` — one JIT compilation, then zero Python overhead.
  Ideal for on-policy algorithms with no mutable replay buffer.

- **Hybrid** (DQN, SAC): Python outer loop for replay-buffer management
  and logging, with ``jax.jit``-compiled inner steps for env interaction
  and gradient updates.  Flexible and easy to extend.

Evaluation is shared: ``evaluate`` / ``jit_evaluate`` run a greedy
policy across vmapped parallel episodes inside a single JIT call.
"""

from vibe_rl.runner.config import RunnerConfig
from vibe_rl.runner.device_utils import (
    get_num_devices,
    replicate,
    replicate_on_mesh,
    shard_pytree,
    split_key_across_devices,
    unreplicate,
)
from vibe_rl.runner.evaluator import EvalMetrics, evaluate, jit_evaluate
from vibe_rl.runner.train_dqn import DQNTrainResult, train_dqn
from vibe_rl.runner.train_ppo import PPOMetricsHistory, PPOTrainState, train_ppo
from vibe_rl.runner.train_ppo_multigpu import train_ppo_multigpu
from vibe_rl.runner.train_sac import SACTrainResult, train_sac

__all__ = [
    # Config
    "RunnerConfig",
    # Device utilities
    "get_num_devices",
    "replicate",
    "replicate_on_mesh",
    "shard_pytree",
    "unreplicate",
    "split_key_across_devices",
    # Evaluator
    "EvalMetrics",
    "evaluate",
    "jit_evaluate",
    # PPO (PureJaxRL)
    "PPOTrainState",
    "PPOMetricsHistory",
    "train_ppo",
    # PPO multi-GPU (jit + NamedSharding)
    "train_ppo_multigpu",
    # DQN (Hybrid)
    "DQNTrainResult",
    "train_dqn",
    # SAC (Hybrid)
    "SACTrainResult",
    "train_sac",
]
