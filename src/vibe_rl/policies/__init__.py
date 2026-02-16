"""Policy wrappers for trained model inference.

Provides a unified ``Policy`` interface that wraps trained model parameters
with input/output transforms (normalization, resize) and JIT-compiled
inference.

Usage::

    from vibe_rl.policies.policy_config import create_trained_policy

    policy = create_trained_policy(config, checkpoint_dir)
    action = policy.infer(observation)
    actions = policy.infer(batched_observations)
"""

from vibe_rl.policies.policy import Policy
from vibe_rl.policies.policy_config import create_trained_policy

__all__ = ["Policy", "create_trained_policy"]
