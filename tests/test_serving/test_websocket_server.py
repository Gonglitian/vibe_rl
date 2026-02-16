"""Tests for the WebSocket policy server."""

import asyncio
import json

import jax
import jax.numpy as jnp
import msgpack
import numpy as np
import websockets

from vibe_rl.policies.policy import Policy
from vibe_rl.serving.websocket_server import (
    PolicyServer,
    _decode_observation,
    _encode_action,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_echo_policy():
    """Create a trivial policy that returns a 2-d action from 4-d obs."""
    import equinox as eqx

    key = jax.random.PRNGKey(0)
    model = eqx.nn.Linear(4, 2, key=key)

    def infer_fn(m, obs):
        return m(obs)

    policy = Policy(model=model, infer_fn=infer_fn)
    # Warm up JIT
    _ = policy.infer(jnp.zeros(4))
    return policy


def _pack_obs(obs_dict: dict[str, np.ndarray]) -> bytes:
    """Encode an observation dict to MessagePack bytes."""
    meta = {}
    payload: dict[bytes, bytes | dict] = {}
    for key, arr in obs_dict.items():
        arr = np.ascontiguousarray(arr)
        bkey = key.encode()
        payload[bkey] = arr.tobytes()
        meta[bkey] = {b"dtype": arr.dtype.str, b"shape": list(arr.shape)}
    payload[b"__meta__"] = meta
    return msgpack.packb(payload, use_bin_type=True)


def _unpack_action(data: bytes) -> tuple[np.ndarray, dict]:
    """Decode a MessagePack response."""
    resp = msgpack.unpackb(data, raw=True)
    meta = resp[b"__meta__"]
    act_meta = meta[b"action"]
    dtype = act_meta[b"dtype"].decode()
    shape = tuple(act_meta[b"shape"])
    action = np.frombuffer(resp[b"action"], dtype=dtype).reshape(shape)
    latency_ms = meta.get(b"latency_ms", 0.0)
    return action, {"latency_ms": latency_ms}


async def _run_with_server(coro_fn):
    """Start a server, run ``coro_fn(url, srv)``, then tear down."""
    policy = _make_echo_policy()
    srv = PolicyServer(policy=policy, host="127.0.0.1", port=0)

    ws_server = await websockets.serve(
        srv._handle_ws,
        "127.0.0.1",
        0,
        process_request=srv._handle_http,
    )
    port = ws_server.sockets[0].getsockname()[1]
    url = f"ws://127.0.0.1:{port}"

    try:
        await coro_fn(url, srv, port)
    finally:
        ws_server.close()
        await ws_server.wait_closed()


# ---------------------------------------------------------------------------
# Serialization tests (no server needed)
# ---------------------------------------------------------------------------


class TestSerialization:
    """Test encode/decode helpers without running a server."""

    def test_encode_action_shape(self):
        action = jnp.array([1.0, 2.0, 3.0])
        encoded = _encode_action(action)
        assert b"action" in encoded
        assert b"__meta__" in encoded
        meta = encoded[b"__meta__"]
        assert meta[b"action"][b"shape"] == [3]

    def test_encode_action_roundtrip(self):
        action = jnp.array([0.5, -0.3], dtype=jnp.float32)
        encoded = _encode_action(action)
        arr = np.frombuffer(encoded[b"action"], dtype="float32")
        np.testing.assert_allclose(arr, [0.5, -0.3], atol=1e-6)

    def test_decode_flat_observation(self):
        obs_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        payload = {
            b"__meta__": {
                b"obs": {b"dtype": "<f4", b"shape": [4]},
            },
            b"obs": obs_np.tobytes(),
        }
        obs = _decode_observation(payload)
        assert isinstance(obs, jax.Array)
        np.testing.assert_allclose(obs, obs_np, atol=1e-6)

    def test_decode_dict_observation(self):
        state = np.array([1.0, 2.0], dtype=np.float32)
        image = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        payload = {
            b"__meta__": {
                b"state": {b"dtype": "<f4", b"shape": [2]},
                b"image": {b"dtype": "|u1", b"shape": [8, 8, 3]},
            },
            b"state": state.tobytes(),
            b"image": image.tobytes(),
        }
        obs = _decode_observation(payload)
        assert isinstance(obs, dict)
        assert "state" in obs
        assert "image" in obs
        np.testing.assert_allclose(obs["state"], state, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(obs["image"]), image)

    def test_decode_image_observation_preserves_shape(self):
        """Ensure high-dimensional image obs is correctly reshaped."""
        image = np.random.rand(64, 64, 3).astype(np.float32)
        payload = {
            b"__meta__": {
                b"obs": {b"dtype": "<f4", b"shape": [64, 64, 3]},
            },
            b"obs": image.tobytes(),
        }
        obs = _decode_observation(payload)
        assert obs.shape == (64, 64, 3)
        np.testing.assert_allclose(obs, image, atol=1e-6)

    def test_full_msgpack_roundtrip(self):
        """Test full pack → send → unpack roundtrip."""
        obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        packed = _pack_obs({"obs": obs})

        # Simulate server-side decode
        payload = msgpack.unpackb(packed, raw=True)
        decoded = _decode_observation(payload)
        np.testing.assert_allclose(decoded, obs, atol=1e-6)

        # Simulate server-side encode
        action = jnp.array([0.1, 0.2])
        encoded = _encode_action(action)
        packed_resp = msgpack.packb(encoded, use_bin_type=True)

        # Simulate client-side decode
        action_out, _ = _unpack_action(packed_resp)
        np.testing.assert_allclose(action_out, [0.1, 0.2], atol=1e-6)


# ---------------------------------------------------------------------------
# Server integration tests (use asyncio.run directly)
# ---------------------------------------------------------------------------


class TestPolicyServer:
    """Integration tests that start a real server."""

    def test_single_inference(self):
        async def _test(url, srv, port):
            obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            msg = _pack_obs({"obs": obs})

            async with websockets.connect(url) as ws:
                await ws.send(msg)
                resp = await ws.recv()

            action, meta = _unpack_action(resp)
            assert action.shape == (2,)
            assert meta["latency_ms"] > 0

        asyncio.run(_run_with_server(_test))

    def test_multiple_inferences(self):
        async def _test(url, srv, port):
            async with websockets.connect(url) as ws:
                for _ in range(5):
                    obs = np.random.randn(4).astype(np.float32)
                    await ws.send(_pack_obs({"obs": obs}))
                    resp = await ws.recv()
                    action, _ = _unpack_action(resp)
                    assert action.shape == (2,)

            assert srv._request_count == 5

        asyncio.run(_run_with_server(_test))

    def test_healthz(self):
        async def _test(url, srv, port):
            # Use raw socket — http.client hangs with websockets' HTTP server
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            data = await asyncio.wait_for(reader.read(4096), timeout=5)
            writer.close()

            # Parse the HTTP response body (after blank line)
            text = data.decode()
            body_str = text.split("\r\n\r\n", 1)[1]
            body = json.loads(body_str)
            assert body["status"] == "ok"
            assert "requests" in body
            assert "avg_latency_ms" in body

        asyncio.run(_run_with_server(_test))

    def test_latency_tracking(self):
        async def _test(url, srv, port):
            assert srv._request_count == 0

            async with websockets.connect(url) as ws:
                obs = np.zeros(4, dtype=np.float32)
                await ws.send(_pack_obs({"obs": obs}))
                await ws.recv()

            assert srv._request_count == 1
            assert srv._total_latency_ms > 0

        asyncio.run(_run_with_server(_test))

    def test_action_values_deterministic(self):
        """Same obs should produce the same action."""

        async def _test(url, srv, port):
            obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            msg = _pack_obs({"obs": obs})

            actions = []
            async with websockets.connect(url) as ws:
                for _ in range(3):
                    await ws.send(msg)
                    resp = await ws.recv()
                    action, _ = _unpack_action(resp)
                    actions.append(action)

            np.testing.assert_array_equal(actions[0], actions[1])
            np.testing.assert_array_equal(actions[1], actions[2])

        asyncio.run(_run_with_server(_test))
