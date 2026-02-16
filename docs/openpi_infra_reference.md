# OpenPI Infrastructure Reference

> Physical-Intelligence/openpi — JAX/Flax 机器人 VLA 训练框架
> 本地参考代码: `refs/openpi/`
> 本文档提炼 openpi 中可复用的基础设施模式，供 vibe-rl 工程师参考。

---

## 目录

1. [项目总览](#1-项目总览)
2. [Tyro 配置系统](#2-tyro-配置系统)
3. [分布式训练 (FSDP + Data Parallel)](#3-分布式训练)
4. [Checkpoint 与断点续训](#4-checkpoint-与断点续训)
5. [优化器与学习率调度](#5-优化器与学习率调度)
6. [数据加载管线](#6-数据加载管线)
7. [日志与实验追踪](#7-日志与实验追踪)
8. [对 vibe-rl 的适用性分析](#8-对-vibe-rl-的适用性分析)

---

## 1. 项目总览

### 源码结构

```
openpi/
├── scripts/
│   ├── train.py                  # JAX 训练入口 (主)
│   ├── train_pytorch.py          # PyTorch DDP 训练入口
│   ├── serve_policy.py           # WebSocket 推理服务
│   └── compute_norm_stats.py     # 计算归一化统计
├── src/openpi/
│   ├── models/                   # 模型定义 (Pi-0, Pi-0-FAST, Pi-0.5)
│   ├── training/
│   │   ├── config.py             # TrainConfig + DataConfig (tyro)
│   │   ├── sharding.py           # FSDP 分片逻辑
│   │   ├── checkpoints.py        # Orbax checkpoint 管理
│   │   ├── optimizer.py          # LR schedule + AdamW
│   │   ├── data_loader.py        # 数据集抽象
│   │   ├── utils.py              # TrainState dataclass
│   │   └── weight_loaders.py     # 权重加载策略
│   ├── policies/                 # 推理策略封装
│   ├── shared/                   # 通用工具
│   └── transforms.py             # 数据变换管线
├── examples/                     # 各机器人平台示例
└── pyproject.toml                # uv 管理, 依赖声明
```

### 核心依赖

| 依赖 | 用途 |
|------|------|
| `jax[cuda12]` | GPU 加速计算 |
| `flax` (NNX API) | 神经网络, 有状态模块 |
| `optax` | 优化器、LR schedule |
| `orbax-checkpoint` | checkpoint 持久化 |
| `tyro` | CLI 配置解析 |
| `wandb` | 实验追踪 |

---

## 2. Tyro 配置系统

> 关键文件: `src/openpi/training/config.py`

### 2.1 核心模式: frozen dataclass + tyro.cli

OpenPI 用 **frozen dataclass** 定义所有配置，通过 tyro 自动生成 CLI:

```python
import dataclasses
import tyro

@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # 必填字段 — tyro.MISSING 强制用户提供
    exp_name: str = tyro.MISSING

    # 带默认值的普通字段
    seed: int = 42
    batch_size: int = 32
    num_train_steps: int = 30_000
    log_interval: int = 100
    save_interval: int = 1000

    # 嵌套配置 — 支持 --model.xxx 点分覆盖
    model: BaseModelConfig = dataclasses.field(default_factory=Pi0Config)
    lr_schedule: LRScheduleConfig = dataclasses.field(default_factory=CosineDecaySchedule)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=AdamW)
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # 隐藏字段 — 不暴露到 CLI
    name: tyro.conf.Suppress[str] = ""
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # 分布式
    fsdp_devices: int = 1

    # 续训
    resume: bool = False
    overwrite: bool = False

    def __post_init__(self):
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()
```

### 2.2 预设配置注册表 + overridable_config_cli

OpenPI 预定义了大量配置组合，通过 `tyro.extras.overridable_config_cli` 实现"选基线 + 覆盖字段":

```python
# 预定义配置注册
_CONFIGS = [
    TrainConfig(
        name="pi0_libero",
        model=Pi0Config(),
        data=LeRobotLiberoDataConfig(repo_id="physical-intelligence/libero"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_droid",
        model=Pi0Config(),
        data=RLDSDroidDataConfig(repo_id="droid_100"),
        num_train_steps=100_000,
    ),
    # ...
]

_CONFIGS_DICT = {c.name: c for c in _CONFIGS}

def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli(
        {k: (k, v) for k, v in _CONFIGS_DICT.items()}
    )
```

使用方式:

```bash
# 选择预设 + 覆盖
python scripts/train.py pi0_libero --exp-name=run1 --batch_size=64
python scripts/train.py pi0_libero --lr_schedule.peak_lr=1e-4
```

### 2.3 关键 tyro 注解

| 注解 | 用途 | 示例 |
|------|------|------|
| `tyro.MISSING` | 标记必填字段 | `exp_name: str = tyro.MISSING` |
| `tyro.conf.Suppress[T]` | 从 CLI 隐藏 | `name: tyro.conf.Suppress[str]` |
| `tyro.extras.overridable_config_cli()` | 预设选择 + 覆盖 | 见上方 |

### 2.4 设计要点

- **所有 dataclass 用 `frozen=True`** — 配置不可变，避免运行时篡改
- **嵌套 dataclass** — 逻辑分组 (model/optimizer/data)，CLI 自动生成 `--model.xxx`
- **`tyro.conf.Suppress`** — 隐藏复杂 JAX 类型 (如 `nnx.filterlib.Filter`)，避免 CLI 解析失败
- **`__post_init__` 校验** — 互斥参数检查
- **computed property** — `checkpoint_dir`, `trainable_filter` 等

---

## 3. 分布式训练

> 关键文件: `src/openpi/training/sharding.py`, `scripts/train.py`

### 3.1 策略: FSDP + Data Parallelism (非 pmap)

OpenPI **不使用 `jax.pmap`**，而是使用现代的 `jax.jit` + Named Sharding:

```
设备拓扑: 2D Mesh = (batch_axis, fsdp_axis)

例: 8 GPU, fsdp_devices=2
   mesh_shape = (4, 2)
   → 4 路数据并行 x 2 路 FSDP 分片
```

### 3.2 Mesh 创建

```python
BATCH_AXIS = "batch"
FSDP_AXIS = "fsdp"
DATA_AXIS = (BATCH_AXIS, FSDP_AXIS)  # 数据沿两个轴分片

def make_mesh(num_fsdp_devices: int) -> jax.sharding.Mesh:
    if jax.device_count() % num_fsdp_devices != 0:
        raise ValueError(...)
    mesh_shape = (jax.device_count() // num_fsdp_devices, num_fsdp_devices)
    return jax.make_mesh(mesh_shape, (BATCH_AXIS, FSDP_AXIS))
```

### 3.3 智能 FSDP 分片

参数按大小和形状自动选择分片策略:

```python
def fsdp_sharding(pytree, mesh, *, min_size_mbytes=4, log=False):
    """对 pytree 中每个数组决定分片策略"""
    def _get_sharding(kp, array):
        size_mb = array.size * array.dtype.itemsize / (1024 * 1024)

        # 规则 1: < 4MB 的小张量 → 全复制
        if size_mb < min_size_mbytes:
            return NamedSharding(mesh, PartitionSpec())

        # 规则 2: 标量/1D → 全复制
        if len(array.shape) < 2:
            return NamedSharding(mesh, PartitionSpec())

        # 规则 3: 沿最大且可整除的维度分片
        axes = np.argsort(array.shape)[::-1]  # 从大到小
        spec = [None] * len(axes)
        for i in axes:
            if array.shape[i] % mesh.shape[FSDP_AXIS] == 0:
                spec[i] = FSDP_AXIS
                return NamedSharding(mesh, PartitionSpec(*spec))

        # 兜底: 全复制
        return NamedSharding(mesh, PartitionSpec())
```

### 3.4 JIT 训练步 (显式声明 sharding)

```python
ptrain_step = jax.jit(
    functools.partial(train_step, config),
    in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
    out_shardings=(train_state_sharding, replicated_sharding),
    donate_argnums=(1,),  # 捐赠旧 train_state 内存
)
```

### 3.5 全局 Mesh 上下文管理

```python
@contextlib.contextmanager
def set_mesh(mesh: jax.sharding.Mesh):
    _MeshState.active_mesh = mesh
    try:
        yield
    finally:
        _MeshState.active_mesh = None

# 在模型前向中使用
def activation_sharding_constraint(pytree):
    if _MeshState.active_mesh is None:
        return pytree
    return jax.lax.with_sharding_constraint(
        pytree,
        NamedSharding(_MeshState.active_mesh, PartitionSpec(DATA_AXIS))
    )
```

### 3.6 关键设计决策

| 决策 | 说明 |
|------|------|
| `jax.jit` 而非 `pmap` | 更灵活，支持混合并行策略 |
| 隐式梯度同步 | 无需手动 `pmean` — JAX sharding 自动处理 |
| 大小阈值 4MB | 小参数复制开销 < 通信开销 |
| `donate_argnums` | 旧状态原地更新，减少内存分配 |

---

## 4. Checkpoint 与断点续训

> 关键文件: `src/openpi/training/checkpoints.py`

### 4.1 基于 Orbax 的 Checkpoint 管理

```python
def initialize_checkpoint_dir(
    checkpoint_dir, *, keep_period, overwrite, resume
) -> tuple[ocp.CheckpointManager, bool]:

    options = ocp.CheckpointManagerOptions(
        max_to_keep=1,               # 仅保留最新
        keep_period=keep_period,     # 每 N 步永久保留 (如 5000)
        async_options=ocp.AsyncOptions(timeout_secs=7200),  # 异步写入
    )

    item_handlers = {
        "params": ocp.PyTreeCheckpointHandler(),
        "train_state": ocp.PyTreeCheckpointHandler(),
        "assets": CallbackHandler(save_fn),  # 自定义异步处理
    }

    return ocp.CheckpointManager(checkpoint_dir, options=options, item_handlers=item_handlers)
```

### 4.2 保存内容

每个 checkpoint 包含三部分:

| 组件 | 内容 | 用途 |
|------|------|------|
| `params` | 模型参数 (优先 EMA 参数) | 推理 |
| `train_state` | optimizer state + step + graphdef | 续训 |
| `assets` | norm_stats.json | 数据归一化 |

### 4.3 保存策略

```python
save_interval: int = 1000      # 每 1000 步保存
keep_period: int | None = 5000 # 每 5000 步永久保留

# 示例: steps 1000, 2000, 3000, 4000 被轮替删除
#        steps 5000, 10000, 15000 永久保留
```

### 4.4 续训流程

```python
# 1. 初始化 — 检测是否续训
checkpoint_manager, resuming = initialize_checkpoint_dir(
    config.checkpoint_dir,
    keep_period=config.keep_period,
    overwrite=config.overwrite,
    resume=config.resume,
)

# 2. 恢复状态
if resuming:
    train_state = restore_state(checkpoint_manager, train_state, data_loader)

# 3. 从恢复的 step 继续
start_step = int(train_state.step)
for step in range(start_step, config.num_train_steps):
    # ... 训练 ...
    if step % config.save_interval == 0 and step > start_step:
        save_state(checkpoint_manager, train_state, data_loader, step)

# 4. 等待异步写入完成
checkpoint_manager.wait_until_finished()
```

### 4.5 EMA 参数分离保存

```python
def _split_params(state: TrainState):
    if state.ema_params is not None:
        params = state.ema_params        # 推理用 EMA
        train_state = replace(state, ema_params=None)
    else:
        params = state.params
        train_state = replace(state, params={})
    return train_state, params
```

### 4.6 WandB 运行续接

```python
if resuming:
    run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
    wandb.init(id=run_id, resume="must")
else:
    wandb.init(name=config.exp_name, ...)
    (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)
```

### 4.7 Checkpoint 目录结构

```
checkpoints/<config_name>/<exp_name>/
├── <step>/
│   ├── params/            # 模型参数 (array shards)
│   ├── train_state/       # 优化器状态
│   └── assets/
│       └── <asset_id>/
│           └── norm_stats.json
├── wandb_id.txt           # WandB run ID
└── checkpoint_state.json  # Orbax 元数据
```

---

## 5. 优化器与学习率调度

> 关键文件: `src/openpi/training/optimizer.py`

### 5.1 LR Schedule

两种内建调度:

```python
@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Warmup + Cosine 退火 (fine-tuning 常用)"""
    peak_lr: float = 2.5e-5
    warmup_steps: int = 1000
    end_lr: float = 0.0

@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule(LRScheduleConfig):
    """Warmup + 1/sqrt(t) 衰减 (pre-training 常用)"""
    peak_lr: float = 1e-4
    warmup_steps: int = 1000
    timescale: int = 10_000
```

### 5.2 优化器构建

```python
@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    weight_decay: float = 1e-10
    max_grad_norm: float = 10.0
    b1: float = 0.9
    b2: float = 0.95

    def create(self, lr_schedule) -> optax.GradientTransformation:
        return optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adamw(lr_schedule, b1=self.b1, b2=self.b2,
                        weight_decay=self.weight_decay),
        )
```

---

## 6. 数据加载管线

> 关键文件: `src/openpi/training/data_loader.py`

### 6.1 管线流程

```
Dataset (LeRobot/RLDS/Fake)
  → repack_transforms     # 统一数据格式
  → data_transforms        # 任务/机器人特定变换
  → normalize              # z-score 或 quantile 归一化
  → model_transforms       # tokenization, padding, resize
  → TorchDataLoader        # 多进程 batch
  → JAX arrays             # 送入 jit 训练步
```

### 6.2 分布式数据分配

```python
# 全局 batch_size 按设备数切分
local_batch_size = batch_size // jax.process_count()

# 数据沿 DATA_AXIS=("batch", "fsdp") 分片
data_sharding = NamedSharding(mesh, PartitionSpec(DATA_AXIS))
```

### 6.3 DataConfig 工厂模式

```python
@dataclasses.dataclass(frozen=True)
class DataConfigFactory:
    repo_id: str = tyro.MISSING
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    def create(self, assets_dir, model_config) -> DataConfig:
        """延迟创建 — 训练时才实例化"""
        ...
```

---

## 7. 日志与实验追踪

### 7.1 WandB 集成

```python
def init_wandb(config, resuming):
    if resuming:
        run_id = read_wandb_id(config.checkpoint_dir)
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(name=config.exp_name, project=config.project_name)
        save_wandb_id(config.checkpoint_dir, wandb.run.id)

# 训练循环中
wandb.log({"loss": loss, "lr": lr, "step": step}, step=step)
```

### 7.2 控制台日志

自定义 formatter，缩写级别名:

```python
class AbbreviatedFormatter(logging.Formatter):
    LEVEL_MAP = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}
    # 输出: I 2024-01-01 12:00:00.000 train.py:100 Step 1000 loss=0.123
```

---

## 8. 对 vibe-rl 的适用性分析

### 8.1 直接可借鉴

| OpenPI 模式 | vibe-rl 适用场景 | 优先级 |
|---|---|---|
| **tyro 配置系统** | 替代当前硬编码的 `PPOConfig`/`RunnerConfig`，支持 CLI 覆盖 | 高 |
| **Orbax checkpoint + 续训** | 长时间 RL 训练的断点续训 | 高 |
| **WandB 续训 ID** | `wandb_id.txt` 写入 checkpoint 目录 | 高 |
| **`jax.jit` + Named Sharding** | 替代当前 `pmap` 多卡方案，更灵活 | 中 |
| **Cosine/Rsqrt LR schedule** | optax schedule 可直接复用 | 低 (已有) |

### 8.2 迁移建议

#### A. Tyro 配置 (推荐优先)

当前 vibe-rl 的 `PPOConfig`, `RunnerConfig` 等是纯 dataclass。迁移到 tyro 只需:

```python
# 现在
config = PPOConfig(n_steps=128, hidden_sizes=(64, 64))

# 迁移后 — 同时支持代码和 CLI
if __name__ == "__main__":
    config = tyro.cli(PPOConfig)
# python train.py --n_steps=256 --hidden_sizes 128 128
```

- 添加 `tyro.MISSING` 给必填字段
- 对复杂 JAX 类型用 `tyro.conf.Suppress`
- 考虑引入 `overridable_config_cli` 做预设实验配置

#### B. Checkpoint 续训

当前 vibe-rl 有 `orbax-checkpoint` 依赖但可能未充分利用。建议:

1. 保存完整 TrainState (params + opt_state + step)
2. 添加 `--resume` / `--overwrite` CLI 参数
3. 用 `keep_period` 策略管理磁盘空间
4. `wandb_id.txt` 写入 checkpoint 目录以支持续训

#### C. 分布式训练

当前 vibe-rl 使用 `jax.pmap` + `pmean`。如需更灵活的并行策略 (如 FSDP):

1. 迁移到 `jax.jit` + `NamedSharding` (JAX 官方推荐的新范式)
2. 用 2D mesh `(batch, fsdp)` 替代 1D pmap
3. 但对于纯 RL 小模型，`pmap` 已足够 — **仅在模型变大时迁移**

### 8.3 不适用的部分

| OpenPI 模式 | 原因 |
|---|---|
| 数据管线 (LeRobot/RLDS) | vibe-rl 用 gymnax 环境，非离线数据集 |
| 推理服务 (WebSocket) | RL 不需要在线推理服务 |
| Weight loaders | vibe-rl 不需要加载预训练 VLA 权重 |
| EMA 参数 | RL 通常不需要 EMA (DQN target net 除外) |
| 归一化统计 | RL 环境提供 obs/action space |
