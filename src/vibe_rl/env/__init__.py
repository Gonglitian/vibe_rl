"""Pure-JAX environment module.

Quick start::

    import jax
    from vibe_rl.env import make

    env, params = make("CartPole-v1")
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, params)
    obs, state, reward, done, info = env.step(key, state, jnp.int32(1), params)
"""

from vibe_rl.env.base import Environment, EnvParams, EnvState
from vibe_rl.env.cart_pole import CartPole, CartPoleParams, CartPoleState
from vibe_rl.env.grid_world import GridWorld, GridWorldParams, GridWorldState
from vibe_rl.env.pendulum import Pendulum, PendulumParams, PendulumState
from vibe_rl.env.pixel_grid_world import PixelGridWorld, PixelGridWorldParams
from vibe_rl.env.spaces import Box, Discrete, Image, MultiBinary
from vibe_rl.env.wrappers import (
    AutoResetWrapper,
    GymnasiumWrapper,
    ObsNormWrapper,
    RewardScaleWrapper,
)

# ---- Registry ----

_REGISTRY: dict[str, type[Environment]] = {
    "CartPole-v1": CartPole,
    "GridWorld-v0": GridWorld,
    "Pendulum-v1": Pendulum,
    "PixelGridWorld-v0": PixelGridWorld,
}


def register(name: str, cls: type[Environment]) -> None:
    """Register a custom environment class under *name*."""
    _REGISTRY[name] = cls


def make(name: str, **kwargs: object) -> tuple[Environment, EnvParams]:
    """Create an environment and its default params by name.

    Built-in names: ``"CartPole-v1"``, ``"GridWorld-v0"``.

    Returns:
        ``(env, params)`` tuple ready for ``env.reset(key, params)``.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown environment {name!r}. Available: {available}")
    env = _REGISTRY[name](**kwargs)
    return env, env.default_params()


__all__ = [
    # Base
    "Environment",
    "EnvState",
    "EnvParams",
    # Spaces
    "Box",
    "Discrete",
    "Image",
    "MultiBinary",
    # Environments
    "CartPole",
    "CartPoleParams",
    "CartPoleState",
    "GridWorld",
    "GridWorldParams",
    "GridWorldState",
    "Pendulum",
    "PendulumParams",
    "PendulumState",
    "PixelGridWorld",
    "PixelGridWorldParams",
    # Wrappers
    "AutoResetWrapper",
    "RewardScaleWrapper",
    "ObsNormWrapper",
    "GymnasiumWrapper",
    # Registry
    "make",
    "register",
]
