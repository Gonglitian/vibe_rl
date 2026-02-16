#!/usr/bin/env python3
"""Example WebSocket client for the PolicyServer.

Connects to a running PolicyServer, sends random observations, and
prints the returned actions.

Requirements::

    pip install websockets msgpack numpy

Usage::

    # Terminal 1: start the server
    python scripts/serve_policy.py cartpole_ppo \\
        --checkpoint_dir runs/cartpole/checkpoints

    # Terminal 2: run this client
    python examples/simple_client.py                    # flat obs (default)
    python examples/simple_client.py --obs_shape 3      # Pendulum
    python examples/simple_client.py --image             # dict obs with image
    python examples/simple_client.py --url ws://remote:8000
"""

from __future__ import annotations

import argparse
import asyncio
import time

import msgpack
import numpy as np
import websockets


def encode_observation(obs: dict[str, np.ndarray]) -> bytes:
    """Encode an observation dict to MessagePack bytes.

    Parameters
    ----------
    obs:
        Dict mapping string keys to NumPy arrays.  A special ``__meta__``
        entry is added automatically with dtype/shape info for each key.
    """
    meta = {}
    payload: dict[bytes, bytes | dict] = {}

    for key, arr in obs.items():
        arr = np.ascontiguousarray(arr)
        bkey = key.encode()
        payload[bkey] = arr.tobytes()
        meta[bkey] = {b"dtype": arr.dtype.str, b"shape": list(arr.shape)}

    payload[b"__meta__"] = meta
    return msgpack.packb(payload, use_bin_type=True)


def decode_action(data: bytes) -> tuple[np.ndarray, dict]:
    """Decode a MessagePack response into (action_array, meta_dict)."""
    resp = msgpack.unpackb(data, raw=True)
    meta = resp[b"__meta__"]
    action_meta = meta[b"action"]
    dtype = action_meta[b"dtype"].decode()
    shape = tuple(action_meta[b"shape"])
    action = np.frombuffer(resp[b"action"], dtype=dtype).reshape(shape)
    latency_ms = meta.get(b"latency_ms", 0.0)
    return action, {"latency_ms": latency_ms}


async def run_client(
    url: str,
    obs_shape: tuple[int, ...],
    *,
    image: bool = False,
    n_steps: int = 10,
) -> None:
    """Connect and send ``n_steps`` random observations."""
    async with websockets.connect(url) as ws:
        print(f"Connected to {url}")

        for i in range(n_steps):
            # Build observation
            if image:
                obs = {
                    "state": np.random.randn(obs_shape[0]).astype(np.float32),
                    "image": np.random.randint(
                        0, 255, (64, 64, 3), dtype=np.uint8
                    ),
                }
            else:
                obs = {"obs": np.random.randn(*obs_shape).astype(np.float32)}

            t0 = time.monotonic()
            await ws.send(encode_observation(obs))
            resp = await ws.recv()
            rtt_ms = (time.monotonic() - t0) * 1000

            action, meta = decode_action(resp)
            print(
                f"step {i:3d} | action={action} | "
                f"server={meta['latency_ms']:.2f}ms | rtt={rtt_ms:.2f}ms"
            )

        print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="PolicyServer test client")
    parser.add_argument("--url", default="ws://localhost:8000", help="Server URL")
    parser.add_argument(
        "--obs_shape",
        type=int,
        nargs="+",
        default=[4],
        help="Observation shape (e.g. 4 for CartPole, 3 for Pendulum)",
    )
    parser.add_argument(
        "--image",
        action="store_true",
        help="Send dict observations with image + state",
    )
    parser.add_argument(
        "--n_steps", type=int, default=10, help="Number of steps to send"
    )
    args = parser.parse_args()
    asyncio.run(
        run_client(
            args.url,
            tuple(args.obs_shape),
            image=args.image,
            n_steps=args.n_steps,
        )
    )


if __name__ == "__main__":
    main()
