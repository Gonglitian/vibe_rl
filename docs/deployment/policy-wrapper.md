# Policy Wrapper

`Policy` 类为训练好的 RL 模型提供统一的推理接口，封装了输入归一化、前向推理、输出反归一化和 JIT 编译。

---

## 架构

```
Observation
    │
    ▼
InputTransform          ← 归一化 / 图像缩放
    │
    ▼
InferFn(model, obs)     ← JIT 编译的前向推理
    │
    ▼
OutputTransform         ← action 反归一化 / 缩放
    │
    ▼
Action
```

---

## Policy 类

```python
from vibe_rl.policies import Policy

policy = Policy(
    model=trained_model,              # Equinox 模型或 JAX pytree
    infer_fn=my_infer_fn,             # (model, obs) -> action 纯函数
    input_transform=my_input_xform,   # obs 预处理（可选）
    output_transform=my_output_xform, # action 后处理（可选）
)

# 单条推理
action = policy.infer(obs)            # obs: (*obs_shape,)

# 批量推理
actions = policy.infer(obs_batch)     # obs_batch: (B, *obs_shape)
```

### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | `Any` | 训练好的模型参数 |
| `infer_fn` | `InferFn` | 纯函数 `(model, obs) -> action`，须 JIT 兼容 |
| `input_transform` | `InputTransform` | 观测预处理变换，默认恒等变换 |
| `output_transform` | `OutputTransform` | 动作后处理变换，默认恒等变换 |

### 自动 batch 检测

`policy.infer()` 会根据数组维度自动区分单条和批量推理：

| 观测类型 | 单条 | 批量 |
|---------|------|------|
| Flat state | `(D,)` | `(B, D)` |
| Image | `(H, W, C)` | `(B, H, W, C)` |
| Dict | 各值 `(*shape)` | 各值 `(B, *shape)` |

- 单条推理：调用 JIT 编译的 `_raw_infer_single`
- 批量推理：调用 `jax.vmap` + JIT 编译的 `_raw_infer_batch`

### JIT 编译

推理管线在**首次调用时**自动 JIT 编译并缓存。后续调用直接使用编译后的版本，无额外编译开销。

---

## 协议定义

### InferFn

```python
class InferFn(Protocol):
    def __call__(self, model: Any, obs: Observation) -> jax.Array: ...
```

### InputTransform

```python
class InputTransform(Protocol):
    def __call__(self, obs: Observation) -> Observation: ...
```

### OutputTransform

```python
class OutputTransform(Protocol):
    def __call__(self, action: jax.Array) -> jax.Array: ...
```

---

## 内置 Transforms

### NormalizeInput

Z-score 归一化观测值：

```python
from vibe_rl.policies.policy import NormalizeInput

normalize = NormalizeInput(
    mean=jnp.array([...]),   # 逐特征均值 (D,)
    std=jnp.array([...]),    # 逐特征标准差 (D,)
    key=None,                # dict obs 时指定键名，None=flat array
    eps=1e-8,
)
```

对 dict 观测：只归一化 `key` 指定的字段，其余透传。

### UnnormalizeOutput

Z-score 反归一化动作：

```python
from vibe_rl.policies.policy import UnnormalizeOutput

unnorm = UnnormalizeOutput(
    mean=jnp.array([...]),   # action 均值
    std=jnp.array([...]),    # action 标准差
    eps=1e-8,
)
```

### ResizeImageInput

使用 `jax.image.resize` 缩放图像观测（JIT 兼容）：

```python
from vibe_rl.policies.policy import ResizeImageInput

resize = ResizeImageInput(
    height=64,
    width=64,
    key=None,   # dict obs 时指定键名
)
```

### ComposeTransforms

将多个 transform 串联：

```python
from vibe_rl.policies.policy import ComposeTransforms

composed = ComposeTransforms(transforms=(
    NormalizeInput(mean=obs_mean, std=obs_std),
    ResizeImageInput(height=64, width=64, key="image"),
))
```

---

## create_trained_policy()

从 checkpoint 一键创建 Policy，自动处理模型加载和变换配置：

```python
from vibe_rl.policies import create_trained_policy
from vibe_rl.configs.presets import TrainConfig
from vibe_rl.algorithms.sac.config import SACConfig

config = TrainConfig(env_id="Pendulum-v1", algo=SACConfig())
policy = create_trained_policy(
    config,
    checkpoint_dir="/path/to/checkpoint",
    step=None,                           # None = 最新 step
    norm_stats_path=None,                # 自动查找 checkpoint 目录下的 norm_stats.json
    obs_norm_key="obs",
    action_norm_key="action",
)

action = policy.infer(obs)
```

### 参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `config` | `TrainConfig` | 训练时的配置（用于重建模型架构） | 必填 |
| `checkpoint_dir` | `str \| Path` | checkpoint 目录路径 | 必填 |
| `step` | `int \| None` | 指定 step，`None` 加载最新 | `None` |
| `norm_stats_path` | `str \| Path \| None` | norm_stats.json 路径 | `None`（自动查找） |
| `obs_norm_key` | `str` | 观测归一化键名 | `"obs"` |
| `action_norm_key` | `str` | 动作归一化键名 | `"action"` |

### 自动 norm_stats 查找

当 `norm_stats_path=None` 时，按以下顺序查找：
1. `{checkpoint_dir}/norm_stats.json`
2. `{checkpoint_dir}/../norm_stats.json`
3. 找不到则跳过归一化

### 各算法推理行为

| 算法 | InferFn | 说明 |
|------|---------|------|
| **PPO** | `argmax(logits)` | Categorical actor 贪心选择 |
| **DQN** | `argmax(Q)` | Q 值最大的动作 |
| **SAC** | `tanh(mean)` | 确定性策略（不采样），→ 通过 OutputTransform 缩放到 action bounds |

### SAC 特殊处理

SAC 的 `infer_fn` 输出 `tanh(mean)` ∈ `[-1, 1]`。`create_trained_policy` 会自动串联：
1. `_SACRescale`：`[-1, 1]` → `[action_low, action_high]`
2. `UnnormalizeOutput`：反归一化到原始尺度

---

## 完整示例

```python
import jax.numpy as jnp
from vibe_rl.policies import create_trained_policy
from vibe_rl.configs.presets import TrainConfig
from vibe_rl.algorithms.ppo.config import PPOConfig

# 1. 加载训练好的 PPO 策略
config = TrainConfig(env_id="CartPole-v1", algo=PPOConfig())
policy = create_trained_policy(config, "runs/cartpole/checkpoints")

# 2. 单条推理
obs = jnp.zeros((4,))  # CartPole 观测
action = policy.infer(obs)
print(f"Action: {action}")  # 0 或 1

# 3. 批量推理
obs_batch = jnp.zeros((32, 4))
actions = policy.infer(obs_batch)
print(f"Actions shape: {actions.shape}")  # (32,)
```

---

## API 速查

| 类 / 函数 | 所在模块 | 说明 |
|-----------|---------|------|
| `Policy` | `vibe_rl.policies.policy` | 推理封装类 |
| `create_trained_policy` | `vibe_rl.policies.policy_config` | 从 checkpoint 创建 Policy |
| `InferFn` | `vibe_rl.policies.policy` | 推理函数协议 |
| `InputTransform` | `vibe_rl.policies.policy` | 输入变换协议 |
| `OutputTransform` | `vibe_rl.policies.policy` | 输出变换协议 |
| `NormalizeInput` | `vibe_rl.policies.policy` | 观测 Z-score 归一化 |
| `UnnormalizeOutput` | `vibe_rl.policies.policy` | 动作反归一化 |
| `ResizeImageInput` | `vibe_rl.policies.policy` | 图像缩放（JIT 兼容） |
| `ComposeTransforms` | `vibe_rl.policies.policy` | 变换串联组合 |
