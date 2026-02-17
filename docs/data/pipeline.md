# Dataset & DataLoader

vibe-rl 的离线数据管线负责将外部数据集（如 LeRobot 机器人演示数据）加载为 JAX `Transition` 批次，供训练循环使用。

> **依赖安装**：`pip install 'vibe-rl[data]'`（需要 PyTorch + LeRobot）

---

## 架构总览

```
HuggingFace Hub
    │
    ▼
LeRobotDataset (torch)        ← lerobot 库原生格式
    │
    ▼
LeRobotDatasetAdapter         ← 适配为 vibe-rl Transition
    │
    ▼
JaxDataLoader                  ← torch DataLoader → JAX arrays
    │
    ▼
Transition(obs, action, reward, next_obs, done)   ← 训练循环消费
```

---

## Dataset 协议

`Dataset` 是一个 `@runtime_checkable` 协议，定义了离线数据集的最小接口：

```python
from vibe_rl.data import Dataset

class Dataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Transition: ...
```

任何实现了 `__len__` + `__getitem__` 并返回 `Transition` 的对象都可以与 `JaxDataLoader` 配合使用。

---

## LeRobotDatasetAdapter

将 LeRobot 格式的 HuggingFace 数据集适配为 vibe-rl `Transition`。

### LeRobotDatasetConfig

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class LeRobotDatasetConfig:
    repo_id: str                                    # HuggingFace repo id
    root: str | None = None                         # 本地缓存目录
    episodes: list[int] | None = None               # 过滤特定 episode
    delta_timestamps: dict[str, list[float]] | None = None  # action horizon
    obs_keys: list[str] = field(
        default_factory=lambda: ["observation.state"]
    )                                               # 要包含的观测键
    action_key: str = "action"                      # action 特征键
    reward_key: str | None = "next.reward"          # reward 键 (None=无reward)
    image_transforms: Any = None                    # torchvision 变换
    revision: str | None = None                     # HF dataset 版本
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `repo_id` | HuggingFace 仓库 ID，如 `"lerobot/aloha_sim_insertion_human"` | 必填 |
| `root` | 本地缓存路径，`None` 使用 HF 默认缓存 | `None` |
| `episodes` | 只加载指定 episode 索引列表 | `None`（全部） |
| `delta_timestamps` | 时间偏移字典，用于 action chunk 堆叠 | `None` |
| `obs_keys` | 要包含在 `obs` 中的 LeRobot 特征键 | `["observation.state"]` |
| `action_key` | action 对应的 LeRobot 特征键 | `"action"` |
| `reward_key` | reward 对应的键，`None` 表示数据集无 reward | `"next.reward"` |

### 观测模式

| 模式 | obs_keys 示例 | obs 类型 |
|------|---------------|----------|
| **State-only** | `["observation.state"]` | 扁平向量 `(D,)` |
| **Image-only** | `["observation.images.top"]` | 图像数组 `(H, W, C)` |
| **Multi-key** | `["observation.state", "observation.state2"]` | 拼接向量 `(D1+D2,)` |

当只有一个 `obs_key` 时返回原始数组；多个键时会自动拼接所有 1-D 向量。

### Delta Timestamps（Action Horizons）

```python
# 以 10 fps 数据集为例，获取 4 步 action chunk
config = LeRobotDatasetConfig(
    repo_id="lerobot/aloha_sim_insertion_human",
    delta_timestamps={"action": [0/10, 1/10, 2/10, 3/10]},
)
# action 形状变为 (4, action_dim)
```

### 基础用法

```python
from vibe_rl.data import LeRobotDatasetAdapter

# 快捷方式：直接传 repo_id 字符串
ds = LeRobotDatasetAdapter("lerobot/aloha_sim_insertion_human")

# 等价于
from vibe_rl.data import LeRobotDatasetConfig
config = LeRobotDatasetConfig(repo_id="lerobot/aloha_sim_insertion_human")
ds = LeRobotDatasetAdapter(config)

# 访问单条数据
t = ds[0]  # Transition(obs, action, reward, next_obs, done)
print(t.obs.shape)     # (14,)  — state vector
print(t.action.shape)  # (14,)  — action vector

# 数据集元信息
print(ds.metadata)
# {'repo_id': '...', 'fps': 50, 'total_episodes': 50, ...}
```

### Episode 边界处理

适配器自动处理 episode 边界：
- 非末帧：`next_obs` 取同 episode 的下一帧
- 末帧：`next_obs = obs`，`done = True`

---

## JaxDataLoader

基于 torch `DataLoader` 的多进程数据加载器，自动将 batch 转为 JAX 数组。

### 参数

```python
from vibe_rl.data import JaxDataLoader

loader = JaxDataLoader(
    dataset,                    # 实现 Dataset 协议的对象
    batch_size=256,             # 每 batch 样本数
    num_workers=0,              # 并行 worker 数 (0=主进程加载)
    shuffle=True,               # 每 epoch 是否打乱
    drop_last=True,             # 丢弃不完整的最后一个 batch
    devices=None,               # JAX 设备列表 (多卡分片用)
    prefetch_factor=2,          # 每 worker 预取 batch 数
)
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `batch_size` | 批量大小 | `256` |
| `num_workers` | 数据加载并行 worker 数 | `0` |
| `shuffle` | 打乱数据 | `True` |
| `drop_last` | 丢弃不完整 batch | `True` |
| `devices` | 多设备分片，传入 `jax.devices()` | `None` |
| `prefetch_factor` | 预取因子，`num_workers > 0` 时生效 | `2` |

### 单卡训练

```python
from vibe_rl.data import LeRobotDatasetAdapter, JaxDataLoader

ds = LeRobotDatasetAdapter("lerobot/aloha_sim_insertion_human")
loader = JaxDataLoader(ds, batch_size=64, num_workers=4)

for batch in loader:
    # batch: Transition，各字段形状 (64, ...)
    state = train_step(state, batch)
```

### 多卡分片

传入 `devices` 参数后，`JaxDataLoader` 自动将 batch 按设备数切分并放置到对应设备：

```python
import jax

devices = jax.devices()  # e.g. 4 GPUs
loader = JaxDataLoader(
    ds,
    batch_size=64,
    num_workers=4,
    devices=devices,
)

for batch in loader:
    # batch.obs 形状: (4, 16, ...) — 4 设备 × 16 样本/设备
    state = pmap_train_step(state, batch)
```

分片实现使用 `jax.sharding.NamedSharding` + `jax.device_put`，要求 `batch_size` 能被设备数整除。

### 内部流程

1. **torch DataLoader**：多进程 `__getitem__` 读取 + collate 为 numpy `Transition`
2. **numpy → JAX**：`jnp.asarray()` 零拷贝转换到 JAX 设备
3. **分片**（可选）：`_shard_batch()` reshape 为 `(num_devices, B // num_devices, ...)` 并放置

### pack_obs / shard_batch

这两个是 DataLoader 的内部辅助函数：

- `_pack_obs(item, obs_keys)`：从 LeRobot 样本字典中提取并打包观测
- `_shard_batch(batch, devices)`：将 batch 按设备数重塑并分片

---

## 完整示例：加载 LeRobot 数据集

```python
from vibe_rl.data import (
    JaxDataLoader,
    LeRobotDatasetAdapter,
    LeRobotDatasetConfig,
)

# 1. 配置数据集
config = LeRobotDatasetConfig(
    repo_id="lerobot/aloha_sim_insertion_human",
    episodes=[0, 1, 2],           # 只加载前 3 个 episode
    obs_keys=["observation.state"],
)

# 2. 创建适配器
dataset = LeRobotDatasetAdapter(config)
print(f"Loaded {len(dataset)} transitions")

# 3. 创建 DataLoader
loader = JaxDataLoader(
    dataset,
    batch_size=128,
    num_workers=2,
    shuffle=True,
)

# 4. 迭代训练
for epoch in range(10):
    for batch in loader:
        # batch.obs:      (128, 14)
        # batch.action:   (128, 14)
        # batch.reward:   (128,)
        # batch.done:     (128,)
        loss = train_step(state, batch)
```

---

## API 速查

| 类 / 函数 | 所在模块 | 说明 |
|-----------|---------|------|
| `Dataset` | `vibe_rl.data.dataset` | 数据集协议（Protocol） |
| `LeRobotDatasetConfig` | `vibe_rl.data.lerobot_dataset` | LeRobot 数据集配置 |
| `LeRobotDatasetAdapter` | `vibe_rl.data.lerobot_dataset` | LeRobot → Transition 适配器 |
| `JaxDataLoader` | `vibe_rl.data.data_loader` | 多进程 JAX DataLoader |
| `Transition` | `vibe_rl.dataprotocol.transition` | `(obs, action, reward, next_obs, done)` 元组 |
