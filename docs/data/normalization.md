# Normalization

vibe-rl 的归一化模块负责计算、保存和应用数据集的逐特征统计量。支持 Z-score 和 quantile 两种归一化方式。

---

## NormStats

`NormStats` 是一个 frozen dataclass，存储单个特征键的统计量：

```python
from vibe_rl.data.normalize import NormStats

@dataclass(frozen=True)
class NormStats:
    mean: np.ndarray    # 均值，形状 (D,) 或标量
    std: np.ndarray     # 标准差
    q01: np.ndarray     # 第 1 百分位（quantile 归一化下界）
    q99: np.ndarray     # 第 99 百分位（quantile 归一化上界）
```

### 序列化

```python
# NormStats → dict (JSON 兼容)
d = stats.to_dict()
# {"mean": [...], "std": [...], "q01": [...], "q99": [...]}

# dict → NormStats
stats = NormStats.from_dict(d)
```

---

## 计算统计量

### compute_norm_stats()

遍历数据集，计算指定键的 mean / std / q01 / q99：

```python
from vibe_rl.data.normalize import compute_norm_stats

stats = compute_norm_stats(
    dataset,                # 实现 __len__ + __getitem__ 的数据集
    keys=["obs", "action"], # 要计算的键
    max_samples=None,       # 最大采样数 (None=全部)
)

# stats["obs"]    → NormStats
# stats["action"] → NormStats
```

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `dataset` | `Dataset` | 支持 `__len__` + `__getitem__` 的数据集 | 必填 |
| `keys` | `list[str]` | 要统计的 Transition 字段名 | 必填 |
| `max_samples` | `int \| None` | 最大采样数，大数据集可加速 | `None` |

**支持的数据集类型**：
- dict-like 对象（`sample["obs"]`）
- NamedTuple（如 `Transition`，自动调用 `_asdict()`）

### 使用示例

```python
from vibe_rl.data import LeRobotDatasetAdapter
from vibe_rl.data.normalize import compute_norm_stats

dataset = LeRobotDatasetAdapter("lerobot/aloha_sim_insertion_human")

# 只用前 1000 条数据快速计算
stats = compute_norm_stats(dataset, keys=["obs", "action"], max_samples=1000)

print(f"obs mean: {stats['obs'].mean[:5]}")
print(f"obs std:  {stats['obs'].std[:5]}")
print(f"action q01: {stats['action'].q01[:5]}")
print(f"action q99: {stats['action'].q99[:5]}")
```

---

## 归一化函数

### Z-score 归一化

将数据标准化为零均值、单位方差：

```python
from vibe_rl.data.normalize import z_score_normalize, z_score_unnormalize

# 归一化: (x - mean) / max(std, eps)
normalized = z_score_normalize(raw_obs, stats["obs"])

# 反归一化: x * max(std, eps) + mean
original = z_score_unnormalize(normalized, stats["obs"])
```

### Quantile 归一化

将 `[q01, q99]` 区间映射到 `[-1, 1]`：

```python
from vibe_rl.data.normalize import quantile_normalize, quantile_unnormalize

# 归一化: 2 * (x - q01) / max(q99 - q01, eps) - 1
normalized = quantile_normalize(raw_action, stats["action"])

# 反归一化: (x + 1) / 2 * max(q99 - q01, eps) + q01
original = quantile_unnormalize(normalized, stats["action"])
```

**特点**：
- 异常值**不会**被 clip，只是会超出 `[-1, 1]` 范围
- 适合 action 数据，因为 quantile 归一化对离群值更鲁棒

### 对比

| 方法 | 公式 | 适用场景 |
|------|------|---------|
| Z-score | `(x - mean) / std` | 近似正态分布的 state |
| Quantile | `2(x - q01) / (q99 - q01) - 1` | 有离群值的 action |

---

## 保存 / 加载

统计量以 JSON 格式持久化，方便版本控制和跨运行复用：

```python
from vibe_rl.data.normalize import save_norm_stats, load_norm_stats

# 保存
save_norm_stats(stats, "norm_stats.json")

# 加载
stats = load_norm_stats("norm_stats.json")
# stats["obs"]    → NormStats
# stats["action"] → NormStats
```

JSON 文件格式：

```json
{
  "obs": {
    "mean": [0.12, -0.03, ...],
    "std": [1.05, 0.98, ...],
    "q01": [-2.3, -1.8, ...],
    "q99": [2.5, 2.1, ...]
  },
  "action": {
    "mean": [0.0, 0.0, ...],
    "std": [0.5, 0.3, ...],
    "q01": [-1.0, -0.8, ...],
    "q99": [1.0, 0.9, ...]
  }
}
```

- `save_norm_stats` 会自动创建父目录
- 所有数组以 Python list 形式序列化，加载时恢复为 `float32` numpy 数组

---

## CLI: compute_norm_stats.py

提供命令行脚本快速计算统计量：

```bash
# 基本用法
python scripts/compute_norm_stats.py --repo-id lerobot/aloha_sim_insertion_human

# 指定键和输出路径
python scripts/compute_norm_stats.py \
    --repo-id lerobot/aloha_sim_insertion_human \
    --keys obs action \
    --output norm_stats.json

# 限制采样数（加速）
python scripts/compute_norm_stats.py \
    --repo-id lerobot/aloha_sim_insertion_human \
    --max-samples 1000

# 只统计特定 episode
python scripts/compute_norm_stats.py \
    --repo-id lerobot/aloha_sim_insertion_human \
    --episodes 0 1 2 3
```

### CLI 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--repo-id` | HuggingFace 数据集 ID | 必填 |
| `--root` | 本地缓存目录 | `None` |
| `--episodes` | 筛选 episode 列表 | `None`（全部） |
| `--keys` | 要计算的 Transition 键 | `["obs", "action"]` |
| `--max-samples` | 最大采样数 | `None`（全部） |
| `--output` | 输出 JSON 路径 | `"norm_stats.json"` |

---

## API 速查

| 类 / 函数 | 所在模块 | 说明 |
|-----------|---------|------|
| `NormStats` | `vibe_rl.data.normalize` | 归一化统计量 dataclass |
| `compute_norm_stats` | `vibe_rl.data.normalize` | 从数据集计算统计量 |
| `z_score_normalize` | `vibe_rl.data.normalize` | Z-score 归一化 |
| `z_score_unnormalize` | `vibe_rl.data.normalize` | Z-score 反归一化 |
| `quantile_normalize` | `vibe_rl.data.normalize` | Quantile 归一化到 [-1, 1] |
| `quantile_unnormalize` | `vibe_rl.data.normalize` | Quantile 反归一化 |
| `save_norm_stats` | `vibe_rl.data.normalize` | 保存到 JSON |
| `load_norm_stats` | `vibe_rl.data.normalize` | 从 JSON 加载 |
