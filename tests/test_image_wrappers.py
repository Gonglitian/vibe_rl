"""Tests for image processing wrappers."""

import jax
import jax.numpy as jnp
import pytest

from vibe_rl.env import (
    FrameStackWrapper,
    GrayscaleWrapper,
    ImageNormWrapper,
    ImageResizeWrapper,
    PixelGridWorld,
    PixelGridWorldParams,
)


# -- Helpers -----------------------------------------------------------------

def _make_env_and_params(**kwargs):
    """Create a PixelGridWorld with given params."""
    env = PixelGridWorld()
    params = PixelGridWorldParams(**kwargs)
    return env, params


def _reset(env, params, seed=0):
    key = jax.random.PRNGKey(seed)
    return env.reset(key, params), key


# -- ImageResizeWrapper ------------------------------------------------------

class TestImageResizeWrapper:
    def test_output_shape(self):
        env, params = _make_env_and_params(size=5, cell_px=8)
        # Original: (40, 40, 3). Resize to (64, 64).
        wrapped = ImageResizeWrapper(env, height=64, width=64)
        (obs, state), _ = _reset(wrapped, params)
        assert obs.shape == (64, 64, 3)

    def test_preserves_dtype(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = ImageResizeWrapper(env, height=32, width=32)
        (obs, _), _ = _reset(wrapped, params)
        assert obs.dtype == jnp.uint8

    def test_step_shape(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = ImageResizeWrapper(env, height=24, width=24)
        (obs, state), key = _reset(wrapped, params)
        obs, state, reward, done, info = wrapped.step(key, state, jnp.int32(1), params)
        assert obs.shape == (24, 24, 3)

    def test_observation_space(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = ImageResizeWrapper(env, height=32, width=48)
        space = wrapped.observation_space(params)
        assert space.shape == (32, 48, 3)

    def test_non_square_resize(self):
        env, params = _make_env_and_params(size=5, cell_px=8)
        wrapped = ImageResizeWrapper(env, height=32, width=64)
        (obs, _), _ = _reset(wrapped, params)
        assert obs.shape == (32, 64, 3)

    def test_jit(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = ImageResizeWrapper(env, height=24, width=24)
        key = jax.random.PRNGKey(0)
        obs, state = jax.jit(wrapped.reset)(key, params)
        assert obs.shape == (24, 24, 3)
        obs, state, r, d, info = jax.jit(wrapped.step)(key, state, jnp.int32(0), params)
        assert obs.shape == (24, 24, 3)

    def test_vmap(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = ImageResizeWrapper(env, height=24, width=24)
        n = 4
        keys = jax.random.split(jax.random.PRNGKey(0), n)
        batch_reset = jax.vmap(wrapped.reset, in_axes=(0, None))
        obs, states = batch_reset(keys, params)
        assert obs.shape == (n, 24, 24, 3)


# -- FrameStackWrapper -------------------------------------------------------

class TestFrameStackWrapper:
    def test_reset_shape(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = FrameStackWrapper(env, n_frames=4)
        (obs, state), _ = _reset(wrapped, params)
        assert obs.shape == (4, 12, 12, 3)

    def test_reset_all_frames_identical(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = FrameStackWrapper(env, n_frames=3)
        (obs, _), _ = _reset(wrapped, params)
        # All frames should be identical on reset
        for i in range(1, 3):
            assert jnp.array_equal(obs[0], obs[i])

    def test_step_shifts_frames(self):
        env, params = _make_env_and_params(size=5, cell_px=4)
        wrapped = FrameStackWrapper(env, n_frames=3)
        (obs, state), key = _reset(wrapped, params)
        initial_frame = obs[0]

        obs2, state, _, _, _ = wrapped.step(key, state, jnp.int32(1), params)
        # After one step: frames[0] and frames[1] should be the old initial frame
        # frames[2] should be the new observation
        assert jnp.array_equal(obs2[0], initial_frame)
        assert jnp.array_equal(obs2[1], initial_frame)
        assert obs2.shape == (3, 20, 20, 3)

    def test_observation_space(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = FrameStackWrapper(env, n_frames=4)
        space = wrapped.observation_space(params)
        assert space.shape == (4, 12, 12, 3)

    def test_jit(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = FrameStackWrapper(env, n_frames=2)
        key = jax.random.PRNGKey(0)
        obs, state = jax.jit(wrapped.reset)(key, params)
        assert obs.shape == (2, 12, 12, 3)
        obs, state, r, d, info = jax.jit(wrapped.step)(key, state, jnp.int32(0), params)
        assert obs.shape == (2, 12, 12, 3)

    def test_vmap(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = FrameStackWrapper(env, n_frames=3)
        n = 4
        keys = jax.random.split(jax.random.PRNGKey(0), n)
        batch_reset = jax.vmap(wrapped.reset, in_axes=(0, None))
        obs, states = batch_reset(keys, params)
        assert obs.shape == (n, 3, 12, 12, 3)

    def test_lax_scan(self):
        env, params = _make_env_and_params(size=3, cell_px=4, max_steps=20)
        from vibe_rl.env import AutoResetWrapper
        wrapped = FrameStackWrapper(AutoResetWrapper(env), n_frames=2)
        key = jax.random.PRNGKey(0)
        obs, state = wrapped.reset(key, params)

        def scan_step(carry, _):
            state, key = carry
            key, subkey = jax.random.split(key)
            obs, state, reward, done, info = wrapped.step(subkey, state, jnp.int32(1), params)
            return (state, key), obs

        (final_state, _), all_obs = jax.lax.scan(scan_step, (state, key), None, length=10)
        assert all_obs.shape == (10, 2, 12, 12, 3)


# -- GrayscaleWrapper --------------------------------------------------------

class TestGrayscaleWrapper:
    def test_output_shape(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = GrayscaleWrapper(env)
        (obs, _), _ = _reset(wrapped, params)
        assert obs.shape == (12, 12, 1)

    def test_preserves_dtype(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = GrayscaleWrapper(env)
        (obs, _), _ = _reset(wrapped, params)
        assert obs.dtype == jnp.uint8

    def test_luminance_weights(self):
        # White (255,255,255) should map to ~255 grayscale
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = GrayscaleWrapper(env)
        (obs, _), _ = _reset(wrapped, params)
        # Agent is at (0,0), rendered as white. Check those pixels.
        # cell_px=4, so agent occupies pixels [0:4, 0:4]
        agent_gray = obs[0, 0, 0]
        # 0.2989*255 + 0.5870*255 + 0.1140*255 ≈ 255 (uint8 round)
        assert agent_gray >= 250  # approximately 255

    def test_step_shape(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = GrayscaleWrapper(env)
        (obs, state), key = _reset(wrapped, params)
        obs, state, r, d, info = wrapped.step(key, state, jnp.int32(1), params)
        assert obs.shape == (12, 12, 1)

    def test_observation_space(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = GrayscaleWrapper(env)
        space = wrapped.observation_space(params)
        assert space.shape == (12, 12, 1)

    def test_jit(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = GrayscaleWrapper(env)
        key = jax.random.PRNGKey(0)
        obs, state = jax.jit(wrapped.reset)(key, params)
        assert obs.shape == (12, 12, 1)
        obs, state, r, d, info = jax.jit(wrapped.step)(key, state, jnp.int32(0), params)
        assert obs.shape == (12, 12, 1)

    def test_vmap(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = GrayscaleWrapper(env)
        n = 4
        keys = jax.random.split(jax.random.PRNGKey(0), n)
        batch_reset = jax.vmap(wrapped.reset, in_axes=(0, None))
        obs, _ = batch_reset(keys, params)
        assert obs.shape == (n, 12, 12, 1)


# -- ImageNormWrapper --------------------------------------------------------

class TestImageNormWrapper:
    def test_output_range(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = ImageNormWrapper(env)
        (obs, _), _ = _reset(wrapped, params)
        assert obs.dtype == jnp.float32
        assert obs.min() >= -1.0
        assert obs.max() <= 1.0

    def test_zero_maps_to_minus_one(self):
        # A pixel value of 0 should map to -1.0
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = ImageNormWrapper(env)
        (obs, _), _ = _reset(wrapped, params)
        # Empty cells are black (0,0,0) → should be -1.0
        # Find a pixel that's in an empty cell (e.g. far corner if not goal)
        # Goal is at (2,2), agent at (0,0). Cell (1,0) is empty.
        empty_pixel = obs[4, 0, 0]  # row 4 = cell row 1, col 0 = cell col 0
        assert jnp.isclose(empty_pixel, -1.0)

    def test_255_maps_to_one(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = ImageNormWrapper(env)
        (obs, _), _ = _reset(wrapped, params)
        # Agent at (0,0) is white (255,255,255) → should be 1.0
        agent_pixel = obs[0, 0, 0]
        assert jnp.isclose(agent_pixel, 1.0)

    def test_output_shape_unchanged(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = ImageNormWrapper(env)
        (obs, _), _ = _reset(wrapped, params)
        assert obs.shape == (12, 12, 3)

    def test_step(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = ImageNormWrapper(env)
        (obs, state), key = _reset(wrapped, params)
        obs, state, r, d, info = wrapped.step(key, state, jnp.int32(1), params)
        assert obs.dtype == jnp.float32
        assert obs.shape == (12, 12, 3)

    def test_observation_space(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = ImageNormWrapper(env)
        space = wrapped.observation_space(params)
        assert space.shape == (12, 12, 3)
        assert jnp.isclose(space.low.min(), -1.0)
        assert jnp.isclose(space.high.max(), 1.0)

    def test_jit(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = ImageNormWrapper(env)
        key = jax.random.PRNGKey(0)
        obs, state = jax.jit(wrapped.reset)(key, params)
        assert obs.dtype == jnp.float32
        obs, state, r, d, info = jax.jit(wrapped.step)(key, state, jnp.int32(0), params)
        assert obs.dtype == jnp.float32

    def test_vmap(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = ImageNormWrapper(env)
        n = 4
        keys = jax.random.split(jax.random.PRNGKey(0), n)
        batch_reset = jax.vmap(wrapped.reset, in_axes=(0, None))
        obs, _ = batch_reset(keys, params)
        assert obs.shape == (n, 12, 12, 3)
        assert obs.dtype == jnp.float32


# -- Composition tests -------------------------------------------------------

class TestWrapperComposition:
    def test_grayscale_then_normalize(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = ImageNormWrapper(GrayscaleWrapper(env))
        (obs, _), _ = _reset(wrapped, params)
        assert obs.shape == (12, 12, 1)
        assert obs.dtype == jnp.float32
        assert obs.min() >= -1.0
        assert obs.max() <= 1.0

    def test_resize_then_grayscale(self):
        env, params = _make_env_and_params(size=5, cell_px=8)
        wrapped = GrayscaleWrapper(ImageResizeWrapper(env, height=32, width=32))
        (obs, _), _ = _reset(wrapped, params)
        assert obs.shape == (32, 32, 1)

    def test_resize_grayscale_normalize(self):
        env, params = _make_env_and_params(size=5, cell_px=8)
        wrapped = ImageNormWrapper(GrayscaleWrapper(ImageResizeWrapper(env, height=32, width=32)))
        (obs, _), _ = _reset(wrapped, params)
        assert obs.shape == (32, 32, 1)
        assert obs.dtype == jnp.float32

    def test_full_pipeline_jit(self):
        env, params = _make_env_and_params(size=3, cell_px=4)
        wrapped = FrameStackWrapper(ImageNormWrapper(GrayscaleWrapper(env)), n_frames=4)
        key = jax.random.PRNGKey(0)
        obs, state = jax.jit(wrapped.reset)(key, params)
        assert obs.shape == (4, 12, 12, 1)
        assert obs.dtype == jnp.float32
        obs, state, r, d, info = jax.jit(wrapped.step)(key, state, jnp.int32(0), params)
        assert obs.shape == (4, 12, 12, 1)

    def test_resize_then_framestack(self):
        env, params = _make_env_and_params(size=5, cell_px=8)
        wrapped = FrameStackWrapper(ImageResizeWrapper(env, height=32, width=32), n_frames=3)
        key = jax.random.PRNGKey(0)
        obs, state = wrapped.reset(key, params)
        assert obs.shape == (3, 32, 32, 3)
