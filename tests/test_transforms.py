"""Tests for vibe_rl.data.transforms module."""

from __future__ import annotations

import numpy as np

from vibe_rl.data.transforms import (
    LambdaTransform,
    Normalize,
    Pad,
    Resize,
    Tokenize,
    Transform,
    TransformGroup,
)

# ── Protocol conformance ─────────────────────────────────────────────


class TestTransformProtocol:
    def test_builtin_transforms_conform(self):
        transforms = [
            Resize(keys=["img"], height=32, width=32),
            Normalize(keys=["obs"], loc=0.0, scale=1.0),
            Tokenize(keys=["action"]),
            Pad(keys=["seq"], max_len=10),
            LambdaTransform(fn=lambda s: s),
            TransformGroup(transforms=[]),
        ]
        for t in transforms:
            assert isinstance(t, Transform)

    def test_callable_is_not_transform(self):
        # A bare function does NOT satisfy the protocol (no __call__ method sig)
        # but a callable class does
        class NotTransform:
            pass

        assert not isinstance(NotTransform(), Transform)

    def test_custom_callable_conforms(self):
        class Custom:
            def __call__(self, sample):
                return sample

        assert isinstance(Custom(), Transform)


# ── TransformGroup ────────────────────────────────────────────────────


class TestTransformGroup:
    def test_empty_group(self):
        group = TransformGroup(transforms=[])
        sample = {"obs": np.array([1.0, 2.0])}
        result = group(sample)
        np.testing.assert_array_equal(result["obs"], sample["obs"])

    def test_composition_order(self):
        """Transforms are applied left-to-right."""
        log: list[str] = []

        def make_fn(name):
            def fn(sample):
                log.append(name)
                return sample

            return LambdaTransform(fn=fn)

        group = TransformGroup(transforms=[make_fn("a"), make_fn("b"), make_fn("c")])
        group({"x": 1})
        assert log == ["a", "b", "c"]

    def test_transforms_can_add_keys(self):
        def add_key(sample):
            sample = dict(sample)
            sample["new_key"] = np.array(42)
            return sample

        group = TransformGroup(transforms=[LambdaTransform(fn=add_key)])
        result = group({"old_key": 1})
        assert "new_key" in result
        assert result["new_key"] == 42

    def test_nested_groups(self):
        inner = TransformGroup(
            transforms=[LambdaTransform(fn=lambda s: {**s, "a": 1})]
        )
        outer = TransformGroup(
            transforms=[inner, LambdaTransform(fn=lambda s: {**s, "b": 2})]
        )
        result = outer({})
        assert result == {"a": 1, "b": 2}


# ── Resize ────────────────────────────────────────────────────────────


class TestResize:
    def test_basic_resize(self):
        img = np.random.rand(100, 80, 3).astype(np.float32)
        t = Resize(keys=["img"], height=32, width=64)
        result = t({"img": img})
        assert result["img"].shape == (32, 64, 3)

    def test_noop_when_same_size(self):
        img = np.random.rand(32, 32, 3).astype(np.float32)
        t = Resize(keys=["img"], height=32, width=32)
        result = t({"img": img})
        np.testing.assert_array_equal(result["img"], img)

    def test_channels_first(self):
        img = np.random.rand(3, 100, 80).astype(np.float32)
        t = Resize(keys=["img"], height=32, width=64, channels_first=True)
        result = t({"img": img})
        assert result["img"].shape == (3, 32, 64)

    def test_missing_key_skipped(self):
        t = Resize(keys=["img"], height=32, width=32)
        sample = {"obs": np.array([1.0])}
        result = t(sample)
        assert "obs" in result
        assert "img" not in result

    def test_does_not_mutate_input(self):
        img = np.random.rand(10, 10, 3).astype(np.float32)
        original = img.copy()
        sample = {"img": img}
        t = Resize(keys=["img"], height=5, width=5)
        t(sample)
        np.testing.assert_array_equal(sample["img"], original)

    def test_batched_image(self):
        imgs = np.random.rand(4, 100, 80, 3).astype(np.float32)
        t = Resize(keys=["img"], height=32, width=64)
        result = t({"img": imgs})
        assert result["img"].shape == (4, 32, 64, 3)


# ── Normalize ─────────────────────────────────────────────────────────


class TestNormalize:
    def test_basic_normalize(self):
        t = Normalize(keys=["obs"], loc=2.0, scale=4.0)
        sample = {"obs": np.array([6.0, 10.0])}
        result = t(sample)
        np.testing.assert_allclose(result["obs"], [1.0, 2.0])

    def test_array_loc_scale(self):
        loc = np.array([1.0, 2.0])
        scale = np.array([2.0, 4.0])
        t = Normalize(keys=["obs"], loc=loc, scale=scale)
        result = t({"obs": np.array([3.0, 6.0])})
        np.testing.assert_allclose(result["obs"], [1.0, 1.0])

    def test_zero_scale_clamped(self):
        t = Normalize(keys=["obs"], loc=0.0, scale=0.0, eps=1.0)
        result = t({"obs": np.array([5.0])})
        np.testing.assert_allclose(result["obs"], [5.0])

    def test_missing_key_skipped(self):
        t = Normalize(keys=["missing"], loc=0.0, scale=1.0)
        sample = {"obs": np.array([1.0])}
        result = t(sample)
        np.testing.assert_array_equal(result["obs"], [1.0])


# ── Tokenize ──────────────────────────────────────────────────────────


class TestTokenize:
    def test_basic_tokenize(self):
        t = Tokenize(keys=["action"], num_tokens=256, vmin=-1.0, vmax=1.0)
        result = t({"action": np.array([0.0])})
        # 0.0 maps to midpoint: 0.5 * 255 = 127.5 → 128
        assert result["action"][0] == 128

    def test_boundaries(self):
        t = Tokenize(keys=["x"], num_tokens=256, vmin=0.0, vmax=1.0)
        result = t({"x": np.array([0.0, 1.0])})
        assert result["x"][0] == 0
        assert result["x"][1] == 255

    def test_clipping(self):
        t = Tokenize(keys=["x"], num_tokens=10, vmin=0.0, vmax=1.0)
        result = t({"x": np.array([-0.5, 1.5])})
        assert result["x"][0] == 0
        assert result["x"][1] == 9

    def test_output_dtype(self):
        t = Tokenize(keys=["x"], num_tokens=256)
        result = t({"x": np.array([0.0])})
        assert result["x"].dtype == np.int32


# ── Pad ───────────────────────────────────────────────────────────────


class TestPad:
    def test_pad_shorter_sequence(self):
        t = Pad(keys=["seq"], max_len=5, pad_value=0.0)
        result = t({"seq": np.array([1.0, 2.0, 3.0])})
        expected = np.array([1.0, 2.0, 3.0, 0.0, 0.0])
        np.testing.assert_array_equal(result["seq"], expected)

    def test_truncate_longer_sequence(self):
        t = Pad(keys=["seq"], max_len=3)
        result = t({"seq": np.array([1.0, 2.0, 3.0, 4.0, 5.0])})
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result["seq"], expected)

    def test_exact_length_noop(self):
        arr = np.array([1.0, 2.0, 3.0])
        t = Pad(keys=["seq"], max_len=3)
        result = t({"seq": arr})
        np.testing.assert_array_equal(result["seq"], arr)

    def test_2d_padding(self):
        t = Pad(keys=["seq"], max_len=4, pad_value=-1.0)
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = t({"seq": arr})
        assert result["seq"].shape == (4, 2)
        np.testing.assert_array_equal(result["seq"][2], [-1.0, -1.0])
        np.testing.assert_array_equal(result["seq"][3], [-1.0, -1.0])

    def test_custom_pad_value(self):
        t = Pad(keys=["seq"], max_len=4, pad_value=99.0)
        result = t({"seq": np.array([1.0])})
        np.testing.assert_array_equal(result["seq"], [1.0, 99.0, 99.0, 99.0])


# ── LambdaTransform ──────────────────────────────────────────────────


class TestLambdaTransform:
    def test_basic(self):
        t = LambdaTransform(fn=lambda s: {**s, "added": True})
        result = t({"obs": 1})
        assert result["added"] is True
        assert result["obs"] == 1


# ── Integration: full pipeline ────────────────────────────────────────


class TestPipelineIntegration:
    def test_resize_then_normalize(self):
        pipeline = TransformGroup(
            transforms=[
                Resize(keys=["img"], height=4, width=4),
                Normalize(keys=["img"], loc=0.5, scale=0.5),
            ]
        )
        sample = {"img": np.ones((8, 8, 3), dtype=np.float32) * 0.5}
        result = pipeline(sample)
        assert result["img"].shape == (4, 4, 3)
        np.testing.assert_allclose(result["img"], 0.0, atol=1e-6)

    def test_multi_key_pipeline(self):
        pipeline = TransformGroup(
            transforms=[
                Normalize(keys=["obs"], loc=0.0, scale=1.0),
                Tokenize(keys=["action"], num_tokens=256, vmin=-1.0, vmax=1.0),
                Pad(keys=["action"], max_len=5),
            ]
        )
        sample = {
            "obs": np.array([1.0, 2.0, 3.0]),
            "action": np.array([0.0, 0.5]),
        }
        result = pipeline(sample)
        assert result["obs"].shape == (3,)
        assert result["action"].shape == (5,)
        assert result["action"].dtype == np.int32
