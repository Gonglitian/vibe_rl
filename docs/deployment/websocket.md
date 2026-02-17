# WebSocket 推理服务

`PolicyServer` 是一个异步 WebSocket 服务，将训练好的 Policy 暴露为网络推理端点。使用 MessagePack 二进制序列化实现高效传输。

> **依赖安装**：`pip install 'vibe-rl[serving]'`（需要 `websockets>=14.0` + `msgpack>=1.0`）

---

## 架构

```
┌─────────────┐         WebSocket (msgpack)         ┌──────────────┐
│   Client    │  ──── observation ──────────────▶   │              │
│  (Python /  │                                     │ PolicyServer │
│   Robot /   │  ◀──── action ──────────────────   │              │
│   Unity)    │                                     │   ┌────────┐ │
└─────────────┘         HTTP /healthz               │   │ Policy │ │
                   ◀──── {"status":"ok"} ─────────  │   └────────┘ │
                                                    └──────────────┘
```

---

## PolicyServer

```python
from vibe_rl.serving import PolicyServer

server = PolicyServer(
    policy=policy,          # Policy 实例
    host="0.0.0.0",         # 绑定地址
    port=8000,              # 绑定端口
)

# 阻塞启动
server.run_blocking()

# 或 async 启动
await server.run()
```

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `policy` | `Policy` | 已准备好的推理策略 | 必填 |
| `host` | `str` | 绑定地址 | `"0.0.0.0"` |
| `port` | `int` | 绑定端口 | `8000` |

---

## 通信协议

### 请求格式（Client → Server）

MessagePack 编码的字典，包含观测数据和元信息：

```python
{
    b"__meta__": {
        b"obs": {b"dtype": "float32", b"shape": [4]},
    },
    b"obs": <raw bytes>,     # np.ndarray.tobytes()
}
```

**Dict 观测**（图像 + 状态）：

```python
{
    b"__meta__": {
        b"state": {b"dtype": "float32", b"shape": [14]},
        b"image": {b"dtype": "uint8",   b"shape": [64, 64, 3]},
    },
    b"state": <raw bytes>,
    b"image": <raw bytes>,
}
```

### 响应格式（Server → Client）

```python
{
    b"action": <raw bytes>,     # action 数组的 tobytes()
    b"__meta__": {
        b"action": {b"dtype": "float32", b"shape": [14]},
        b"latency_ms": 1.23,   # 服务端推理延迟 (ms)
    },
}
```

### 序列化流程

1. Client：`np.ndarray` → `tobytes()` → MessagePack → WebSocket
2. Server：MessagePack → `np.frombuffer()` → `jnp.asarray()` → Policy.infer()
3. Server：`jax.Array` → `np.asarray()` → `tobytes()` → MessagePack → WebSocket

---

## 健康检查

PolicyServer 同时处理 HTTP 请求，提供 `/healthz` 端点：

```bash
curl http://localhost:8000/healthz
```

```json
{
  "status": "ok",
  "requests": 1042,
  "avg_latency_ms": 1.85
}
```

| 字段 | 说明 |
|------|------|
| `status` | 始终为 `"ok"` |
| `requests` | 累计请求数 |
| `avg_latency_ms` | 平均推理延迟（毫秒） |

---

## CLI: serve_policy.py

使用预设配置快速启动服务：

```bash
# 启动 CartPole PPO 推理服务
python scripts/serve_policy.py cartpole_ppo \
    --checkpoint_dir runs/cartpole/checkpoints

# 启动 Pendulum SAC 推理服务，指定端口
python scripts/serve_policy.py pendulum_sac \
    --checkpoint_dir runs/pendulum/checkpoints \
    --port 9000

# 指定 checkpoint step 和归一化统计
python scripts/serve_policy.py pendulum_sac \
    --checkpoint_dir runs/pendulum/checkpoints \
    --step 50000 \
    --norm_stats_path runs/pendulum/norm_stats.json
```

### CLI 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| 位置参数 | 预设名（如 `cartpole_ppo`、`pendulum_sac`） | 必填 |
| `--checkpoint_dir` | checkpoint 目录路径 | 必填 |
| `--step` | 加载指定 step | `None`（最新） |
| `--norm_stats_path` | 归一化统计文件路径 | `None`（自动查找） |
| `--host` | 绑定地址 | `"0.0.0.0"` |
| `--port` | 绑定端口 | `8000` |

脚本启动流程：
1. 根据预设创建 `TrainConfig`
2. 调用 `create_trained_policy()` 加载模型
3. JIT warmup（用零值 dummy obs 预热）
4. 创建 `PolicyServer` 并阻塞运行

---

## Client 示例

`examples/simple_client.py` 提供了一个完整的测试客户端：

```bash
# Flat 观测 (CartPole, obs_shape=4)
python examples/simple_client.py

# Pendulum (obs_shape=3)
python examples/simple_client.py --obs_shape 3

# Dict 观测 (image + state)
python examples/simple_client.py --image

# 连接远程服务
python examples/simple_client.py --url ws://remote-host:8000

# 发送更多 step
python examples/simple_client.py --n_steps 100
```

### Client 代码要点

**编码观测**：

```python
import msgpack
import numpy as np

def encode_observation(obs: dict[str, np.ndarray]) -> bytes:
    meta = {}
    payload = {}
    for key, arr in obs.items():
        arr = np.ascontiguousarray(arr)
        bkey = key.encode()
        payload[bkey] = arr.tobytes()
        meta[bkey] = {b"dtype": arr.dtype.str, b"shape": list(arr.shape)}
    payload[b"__meta__"] = meta
    return msgpack.packb(payload, use_bin_type=True)
```

**解码动作**：

```python
def decode_action(data: bytes) -> tuple[np.ndarray, dict]:
    resp = msgpack.unpackb(data, raw=True)
    meta = resp[b"__meta__"]
    action_meta = meta[b"action"]
    dtype = action_meta[b"dtype"].decode()
    shape = tuple(action_meta[b"shape"])
    action = np.frombuffer(resp[b"action"], dtype=dtype).reshape(shape)
    latency_ms = meta.get(b"latency_ms", 0.0)
    return action, {"latency_ms": latency_ms}
```

**完整推理循环**：

```python
import asyncio
import websockets

async def infer_loop(url: str, obs: dict[str, np.ndarray]):
    async with websockets.connect(url) as ws:
        await ws.send(encode_observation(obs))
        resp = await ws.recv()
        action, meta = decode_action(resp)
        print(f"action={action}, latency={meta['latency_ms']:.2f}ms")

asyncio.run(infer_loop("ws://localhost:8000", {"obs": np.zeros(4, dtype=np.float32)}))
```

---

## 性能指标

| 指标 | 说明 |
|------|------|
| `_request_count` | 服务端累计处理请求数 |
| `_total_latency_ms` | 累计推理延迟 |
| `/healthz` avg_latency | 平均单次推理延迟 |

延迟测量范围：从收到 WebSocket 消息到发送响应前，包含反序列化 + 推理 + 序列化。

首次请求会触发 JIT 编译，延迟显著更高。`serve_policy.py` 脚本在启动时自动执行 warmup 以避免这个问题。

---

## API 速查

| 类 / 函数 | 所在模块 | 说明 |
|-----------|---------|------|
| `PolicyServer` | `vibe_rl.serving.websocket_server` | WebSocket 推理服务 |
| `PolicyServer.run()` | — | async 启动 |
| `PolicyServer.run_blocking()` | — | 阻塞启动 |
| `/healthz` | HTTP endpoint | 健康检查 + 性能指标 |
