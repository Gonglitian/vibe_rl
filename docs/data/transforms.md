# Transforms

vibe-rl 提供一套可组合的数据变换管线，用于在样本进入训练循环之前进行预处理（缩放、归一化、填充、离散化等）。

---

## Transform 协议

```python
from vibe_rl.data.transforms import Transform

class Transform(Protocol):
    def __call__(self, sample: Sample) -> Sample: ...
```

- `Sample = dict[str, Any]` — 键值对样本字典
- Transform 是纯函数：所有配置在构造时捕获，`__call__` 不修改内部状态
- 任何满足 `(Sample) -> Sample` 签名的 callable 都可以作为 Transform 使用

---

## TransformGroup 组合

`TransformGroup` 将多个 Transform 按顺序串联：

```python
from vibe_rl.data.transforms import TransformGroup, Resize, Normalize, Pad

pipeline = TransformGroup([
    Resize(keys=["obs", "next_obs"], height=64, width=64),
    Normalize(keys=["obs", "next_obs"], loc=mean, scale=std),
    Pad(keys=["action"], max_len=10, pad_value=0.0),
])

processed_sample = pipeline(raw_sample)
```

变换按列表顺序依次执行，每个 Transform 的输出作为下一个的输入。

---

## 内置 Transforms

### Resize

将图像数组缩放到指定分辨率。使用纯 numpy 最近邻插值，无需 PIL/cv2 依赖。

```python
from vibe_rl.data.transforms import Resize

resize = Resize(
    keys=["obs", "next_obs"],   # 要处理的样本键
    height=64,                  # 目标高度
    width=64,                   # 目标宽度
    channels_first=False,       # True = (..., C, H, W), False = (..., H, W, C)
)
```

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `keys` | `Sequence[str]` | 要缩放的样本键 | 必填 |
| `height` | `int` | 目标高度 | 必填 |
| `width` | `int` | 目标宽度 | 必填 |
| `channels_first` | `bool` | 通道维在前 | `False` |

- 缺失的键会被静默跳过
- 支持任意前导维度（batch、时间步等）

### Normalize

逐元素仿射归一化：`(x - loc) / scale`。

```python
from vibe_rl.data.transforms import Normalize
import numpy as np

normalize = Normalize(
    keys=["obs", "next_obs"],
    loc=np.array([0.5, 0.3, ...]),   # 偏移量（如均值或 q01）
    scale=np.array([1.2, 0.8, ...]), # 缩放量（如标准差或 q99-q01）
    eps=1e-8,                        # 防止除零
)
```

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `keys` | `Sequence[str]` | 要归一化的键 | 必填 |
| `loc` | `ndarray \| float` | 偏移量，可广播 | 必填 |
| `scale` | `ndarray \| float` | 缩放量，可广播 | 必填 |
| `eps` | `float` | 防止除零常数 | `1e-8` |

### Tokenize

将连续值离散化为整数 token，映射 `[vmin, vmax]` → `[0, num_tokens - 1]`。

```python
from vibe_rl.data.transforms import Tokenize

tokenize = Tokenize(
    keys=["action"],
    num_tokens=256,   # 离散 bin 数
    vmin=-1.0,        # 连续值下界
    vmax=1.0,         # 连续值上界
)
```

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `keys` | `Sequence[str]` | 要离散化的键 | 必填 |
| `num_tokens` | `int` | bin 数量 | `256` |
| `vmin` | `float` | 连续范围下界 | `-1.0` |
| `vmax` | `float` | 连续范围上界 | `1.0` |

输出数组 dtype 为 `int32`，值被 clip 到 `[0, num_tokens - 1]`。

### Pad

沿第 0 轴填充或截断到 `max_len`。

```python
from vibe_rl.data.transforms import Pad

pad = Pad(
    keys=["action"],
    max_len=10,        # 目标长度
    pad_value=0.0,     # 填充值
)
```

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `keys` | `Sequence[str]` | 要填充的键 | 必填 |
| `max_len` | `int` | 目标第 0 轴长度 | 必填 |
| `pad_value` | `float` | 填充值 | `0.0` |

- 长度 ≥ `max_len`：截断到 `max_len`
- 长度 < `max_len`：末尾填充

### LambdaTransform

将任意 callable 包装为 Transform：

```python
from vibe_rl.data.transforms import LambdaTransform

# 自定义变换：将 obs 裁剪到 [-5, 5]
clip_transform = LambdaTransform(
    fn=lambda s: {**s, "obs": np.clip(s["obs"], -5.0, 5.0)}
)
```

---

## 管线示例：图像数据预处理

```python
import numpy as np
from vibe_rl.data.transforms import (
    TransformGroup,
    Resize,
    Normalize,
    Tokenize,
)

# 预计算的统计量
image_mean = np.array([0.485, 0.456, 0.406])
image_std = np.array([0.229, 0.224, 0.225])

pipeline = TransformGroup([
    # 1. 将图像缩放到 64×64
    Resize(keys=["obs", "next_obs"], height=64, width=64),

    # 2. ImageNet 风格归一化
    Normalize(keys=["obs", "next_obs"], loc=image_mean, scale=image_std),

    # 3. 将 action 离散化为 256 个 token
    Tokenize(keys=["action"], num_tokens=256, vmin=-1.0, vmax=1.0),
])

# 应用到单个样本
sample = {"obs": raw_image, "next_obs": raw_next_image, "action": raw_action}
processed = pipeline(sample)
```

## 管线示例：机器人状态数据

```python
from vibe_rl.data.transforms import TransformGroup, Normalize, Pad
from vibe_rl.data.normalize import compute_norm_stats

# 从数据集计算统计量
stats = compute_norm_stats(dataset, keys=["obs", "action"])

pipeline = TransformGroup([
    # 1. Z-score 归一化状态和动作
    Normalize(
        keys=["obs", "next_obs"],
        loc=stats["obs"].mean,
        scale=stats["obs"].std,
    ),
    Normalize(
        keys=["action"],
        loc=stats["action"].mean,
        scale=stats["action"].std,
    ),

    # 2. 将 action chunk 填充到固定长度
    Pad(keys=["action"], max_len=16, pad_value=0.0),
])
```

---

## API 速查

| 类 | 所在模块 | 说明 |
|---|---------|------|
| `Transform` | `vibe_rl.data.transforms` | 变换协议（Protocol） |
| `TransformGroup` | `vibe_rl.data.transforms` | 变换组合器 |
| `Resize` | `vibe_rl.data.transforms` | 图像缩放 |
| `Normalize` | `vibe_rl.data.transforms` | 仿射归一化 |
| `Tokenize` | `vibe_rl.data.transforms` | 连续值离散化 |
| `Pad` | `vibe_rl.data.transforms` | 定长填充/截断 |
| `LambdaTransform` | `vibe_rl.data.transforms` | 自定义函数包装 |
