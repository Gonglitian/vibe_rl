"""Async WebSocket inference server.

Accepts observations via WebSocket (MessagePack-encoded), runs policy
inference, and returns actions.  Also exposes an HTTP ``/healthz`` endpoint.

Protocol
--------
Client sends a MessagePack-encoded dict with string keys.  Values are
raw bytes that will be deserialized to NumPy arrays via ``np.frombuffer``.
Each message must include a ``__meta__`` key with serialization metadata::

    {
        "__meta__": {
            "<key>": {"dtype": "float32", "shape": [H, W, C]},
            ...
        },
        "<key>": <raw bytes>,
        ...
    }

The server responds with a MessagePack-encoded dict::

    {
        "action": <raw bytes>,
        "__meta__": {
            "action": {"dtype": "float32", "shape": [action_dim]},
            "latency_ms": 1.23,
        },
    }

Usage::

    server = PolicyServer(policy, host="0.0.0.0", port=8000)
    await server.run()              # or server.run_blocking()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


def _require(module_name: str):
    """Import an optional dependency and raise a clear error if missing."""
    try:
        return __import__(module_name)
    except ImportError:
        raise ImportError(
            f"{module_name} is required for the serving module. "
            f"Install it with: pip install 'vibe-rl[serving]'"
        ) from None


@dataclass
class PolicyServer:
    """Async WebSocket server that wraps a :class:`~vibe_rl.policies.policy.Policy`.

    Parameters
    ----------
    policy:
        A :class:`Policy` instance ready for inference.
    host:
        Bind address.
    port:
        Bind port.
    """

    policy: object  # Policy — kept generic to avoid import at module level
    host: str = "0.0.0.0"
    port: int = 8000

    # Runtime stats (read-only from outside)
    _request_count: int = field(default=0, init=False, repr=False)
    _total_latency_ms: float = field(default=0.0, init=False, repr=False)

    # --- public API --------------------------------------------------------

    def run_blocking(self) -> None:
        """Start the server (blocking).  Use from ``if __name__``."""
        asyncio.run(self.run())

    async def run(self) -> None:
        """Start the server (async)."""
        websockets = _require("websockets")

        server = await websockets.serve(
            self._handle_ws,
            self.host,
            self.port,
            process_request=self._handle_http,
        )
        logger.info("PolicyServer listening on ws://%s:%d", self.host, self.port)
        await server.wait_closed()

    # --- WebSocket handler -------------------------------------------------

    async def _handle_ws(self, websocket) -> None:
        """Handle a single WebSocket connection."""
        msgpack = _require("msgpack")

        logger.info("Client connected: %s", websocket.remote_address)
        try:
            async for raw in websocket:
                t0 = time.monotonic()

                # Decode observation
                payload = msgpack.unpackb(raw, raw=True)
                obs = _decode_observation(payload)

                # Run inference
                action = self.policy.infer(obs)

                # Encode response
                response = _encode_action(action)

                latency_ms = (time.monotonic() - t0) * 1000
                response[b"__meta__"][b"latency_ms"] = latency_ms

                self._request_count += 1
                self._total_latency_ms += latency_ms

                await websocket.send(msgpack.packb(response, use_bin_type=True))
        except Exception:
            logger.exception("Error handling client %s", websocket.remote_address)
        finally:
            logger.info("Client disconnected: %s", websocket.remote_address)

    # --- HTTP health check -------------------------------------------------

    def _handle_http(self, connection, request):
        """Respond to ``/healthz`` with a simple 200 OK.

        Uses the websockets v14+ ``process_request`` signature:
        ``(connection, request) -> Response | None``.
        """
        from websockets.datastructures import Headers
        from websockets.http11 import Response

        if request.path == "/healthz":
            avg = (
                self._total_latency_ms / self._request_count
                if self._request_count
                else 0.0
            )
            body = (
                f'{{"status":"ok",'
                f'"requests":{self._request_count},'
                f'"avg_latency_ms":{avg:.2f}}}\n'
            ).encode()
            return Response(
                200,
                "OK",
                Headers([("Content-Type", "application/json")]),
                body,
            )
        # Return None to proceed with WebSocket handshake
        return None


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _decode_observation(payload: dict) -> jnp.ndarray | dict[str, jnp.ndarray]:
    """Decode a MessagePack observation dict into JAX arrays.

    The ``__meta__`` key contains dtype/shape metadata for each observation
    key.  All other keys are raw byte buffers.
    """
    meta = payload[b"__meta__"]
    if isinstance(meta, bytes):
        import msgpack

        meta = msgpack.unpackb(meta, raw=True)

    obs_keys = [k for k in payload if k != b"__meta__"]

    if len(obs_keys) == 1 and obs_keys[0] == b"obs":
        key = obs_keys[0]
        arr = _unpack_array(meta[key], payload[key])
        return jnp.asarray(arr)

    # Dict observation
    obs: dict[str, jnp.ndarray] = {}
    for key in obs_keys:
        arr = _unpack_array(meta[key], payload[key])
        name = key.decode() if isinstance(key, bytes) else key
        obs[name] = jnp.asarray(arr)
    return obs


def _unpack_array(info: dict, raw: bytes) -> np.ndarray:
    """Unpack a single array from metadata + raw bytes."""
    dtype_val = info[b"dtype"]
    dtype = dtype_val.decode() if isinstance(dtype_val, bytes) else dtype_val
    shape_val = info[b"shape"]
    shape = tuple(shape_val) if isinstance(shape_val, list) else shape_val
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def _encode_action(action: jnp.ndarray) -> dict:
    """Encode a JAX action array into a MessagePack-friendly dict."""
    arr = np.asarray(action)
    return {
        b"action": arr.tobytes(),
        b"__meta__": {
            b"action": {
                b"dtype": arr.dtype.str,
                b"shape": list(arr.shape),
            },
        },
    }
