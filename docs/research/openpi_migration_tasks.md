# OpenPI Feature Migration — Task Breakdown

> 共 7 个工作流、14 个 issue，最大并行度 5

---

## 依赖关系图

```
并行层 0 (立即开始，互不依赖):
  [A1] Tyro 配置迁移
  [B1] Checkpoint 增强 + 断点续训
  [D1] Vision Env 基础 + CNN/ViT 网络
  [E1] LeRobot 数据管线集成
  [G1] WandB 续训 + 日志增强

并行层 1 (依赖层 0):
  [A2] 预设配置注册表        ← 依赖 A1
  [C1] jit+NamedSharding 迁移 ← 依赖 A1
  [D2] Image Env Wrappers     ← 依赖 D1
  [E2] 数据变换管线 + 归一化  ← 依赖 E1

并行层 2 (依赖层 1):
  [C2] FSDP 支持              ← 依赖 C1
  [F1] 推理策略封装           ← 依赖 D1, E2

并行层 3:
  [F2] WebSocket 推理服务     ← 依赖 F1
  [C3] 修复多卡测试           ← 依赖 C1 或 C2
  [H1] 端到端集成测试         ← 依赖 all
```

---

## Stream A: Tyro 配置系统

### [A1] 迁移现有配置到 tyro CLI (层 0 — 可立即开始)

**优先级**: P0
**估计改动**: `pyproject.toml` + 所有 `config.py` + 训练入口脚本

**描述**:
将现有 frozen dataclass 配置迁移到 tyro，使训练脚本支持 CLI 参数覆盖。

**具体工作**:
1. `pyproject.toml` 添加 `tyro>=0.9` 依赖
2. 迁移配置类:
   - `PPOConfig` → 添加 `tyro.MISSING` 给必填字段
   - `SACConfig` / `DQNConfig` 同理
   - `RunnerConfig` → 添加 `fsdp_devices: int = 1`, `resume: bool`, `overwrite: bool`
3. 对复杂 JAX 类型使用 `tyro.conf.Suppress`
4. 为每个训练 runner 添加 `if __name__ == "__main__": tyro.cli(main)` 入口
5. 添加 `scripts/train_ppo.py`, `scripts/train_sac.py`, `scripts/train_dqn.py` 作为 CLI 入口

**验收标准**:
- `python scripts/train_ppo.py --help` 正常显示所有参数
- `python scripts/train_ppo.py --lr=1e-3 --n_steps=256` 可覆盖默认值
- 所有现有测试通过

**参考**: `refs/openpi/src/openpi/training/config.py`

---

### [A2] 预设配置注册表 (层 1 — 依赖 A1)

**优先级**: P1

**描述**:
实现 `tyro.extras.overridable_config_cli` 预设注册，支持 `python train.py cartpole_ppo --lr=1e-3` 选择预设并覆盖。

**具体工作**:
1. 创建 `src/vibe_rl/configs/presets.py`
2. 预定义典型实验配置 (cartpole_ppo, pendulum_sac, 等)
3. 实现 `cli() -> TrainConfig` 注册逻辑
4. 更新训练脚本入口使用 `overridable_config_cli`

**参考**: `refs/openpi/src/openpi/training/config.py` L560-979

---

## Stream B: Checkpoint 与断点续训

### [B1] 增强 Checkpoint 系统 + 断点续训 (层 0 — 可立即开始)

**优先级**: P0
**估计改动**: `checkpoint.py` + 训练 runner

**描述**:
增强现有 orbax checkpoint 系统，对齐 openpi 的续训模式。

**具体工作**:
1. 增强 `CheckpointManager`:
   - 添加 `keep_period` 参数 (每 N 步永久保留)
   - 启用异步保存 `AsyncOptions(timeout_secs=7200)`
   - 保存内容分三项: `params`, `train_state` (opt_state + step), `assets` (元数据)
2. 实现 `initialize_checkpoint_dir(dir, *, keep_period, overwrite, resume)`:
   - 已有 checkpoint + `resume=True` → 恢复
   - 已有 checkpoint + 无 flag → 报错提示
   - `overwrite=True` → 清除旧 checkpoint
3. 在训练 runner 中集成:
   - 恢复 step 计数，从中断处继续
   - 保存/恢复 optimizer state
4. WandB 续训:
   - `wandb_id.txt` 写入 checkpoint 目录
   - 续训时读取 run ID，`wandb.init(id=..., resume="must")`
5. 训练 runner 添加 `--resume` / `--overwrite` 参数

**验收标准**:
- 训练 1000 步 → 中断 → `--resume` → 从 1000 步继续
- WandB 曲线无断裂
- `keep_period=500` 时磁盘上保留正确的 checkpoint 子集

**参考**: `refs/openpi/src/openpi/training/checkpoints.py`

---

## Stream C: 分布式训练

### [C1] 从 pmap 迁移到 jit + NamedSharding (层 1 — 依赖 A1)

**优先级**: P1
**估计改动**: `runner/device_utils.py`, `runner/train_ppo_multigpu.py`

**描述**:
将当前 pmap 多卡方案迁移到 JAX 推荐的 jit + NamedSharding 范式。

**具体工作**:
1. 新建 `src/vibe_rl/sharding.py`:
   - `make_mesh(num_fsdp_devices)` → 2D mesh `(batch, fsdp)`
   - `set_mesh()` 上下文管理器
   - `activation_sharding_constraint()`
2. 重构 `train_ppo_multigpu.py`:
   - 移除 `jax.pmap` + `pmean` 模式
   - 改用 `jax.jit(fn, in_shardings=..., out_shardings=...)`
   - 数据沿 `DATA_AXIS = ("batch", "fsdp")` 分片
3. 更新 `device_utils.py`:
   - `replicate()` / `unreplicate()` 改为 sharding-aware 版本
4. RunnerConfig 添加 `fsdp_devices: int = 1` (纯数据并行时 = 1)
5. 保持 `fsdp_devices=1` 时行为与当前 pmap 一致

**验收标准**:
- 单卡: 与 pmap 版本训练结果一致
- 多卡 (XLA_FLAGS 模拟): 梯度同步正确
- 不引入性能退化

**参考**: `refs/openpi/src/openpi/training/sharding.py`

---

### [C2] 添加 FSDP 支持 (层 2 — 依赖 C1)

**优先级**: P2

**描述**:
在 C1 基础上添加 FSDP 参数分片，支持大模型训练。

**具体工作**:
1. 实现 `fsdp_sharding(pytree, mesh, *, min_size_mbytes=4)`:
   - < 4MB 参数全复制
   - >= 4MB 参数沿最大可整除维度分片
2. 训练初始化时用 `jax.eval_shape` + `fsdp_sharding` 计算分片策略
3. `jax.jit` 显式声明 `in_shardings` / `out_shardings`

**参考**: `refs/openpi/src/openpi/training/sharding.py` L48-102

---

### [C3] 修复多卡测试 (层 3 — 依赖 C1)

**优先级**: P0

**描述**:
修复 `tests/test_multigpu.py` 中 9 个失败测试 (已有 kanban issue)。

**具体工作**:
1. 在 `test_multigpu.py` 顶部设置 `XLA_FLAGS=--xla_force_host_platform_device_count=4`
2. 确保在 JAX 导入前设置环境变量
3. 适配测试到新 sharding 方案 (如果 C1 已完成)

---

## Stream D: Vision 环境 + 网络

### [D1] Vision 环境基础 + CNN/ViT 网络架构 (层 0 — 可立即开始)

**优先级**: P0
**估计改动**: 新建 `env/vision_*`, `algorithms/networks/`

**描述**:
添加 image 观测环境支持和对应的视觉网络架构。

**具体工作**:
1. 添加视觉网络模块 `src/vibe_rl/networks/`:
   - `cnn.py`: 简单 CNN encoder (Nature DQN style + 可配置)
   - `vit.py`: 轻量 ViT encoder (参考 openpi SigLIP)
   - `encoder.py`: 统一 encoder 接口 (支持 MLP/CNN/ViT)
2. 扩展 observation space:
   - `spaces.py` 添加 `Image(height, width, channels)` space
   - `Box` space 支持高维 image shape
3. 创建示例 vision env:
   - 包装 gymnax 或 brax 环境添加 pixel rendering
   - 或实现简单的 grid-world pixel 版本
4. 扩展 `Transition` 支持 image observations (高维 obs)

**验收标准**:
- CNN encoder 可 `jax.jit` 编译
- Image space 正确声明 shape/dtype
- 至少一个 vision env 可运行

**参考**: `refs/openpi/src/openpi/models/vit.py`, `refs/openpi/src/openpi/models/siglip.py`

---

### [D2] Image Environment Wrappers (层 1 — 依赖 D1)

**优先级**: P1

**描述**:
添加 image 处理 wrappers (resize, normalize, frame stack 等)。

**具体工作**:
1. `wrappers.py` 添加:
   - `ImageResizeWrapper(height, width)`: JAX image resize with padding
   - `FrameStackWrapper(n_frames)`: 堆叠连续帧
   - `GrayscaleWrapper`: RGB → grayscale
   - `ImageNormWrapper`: [0,255] → [-1,1] 归一化
2. 参考 openpi 的 `image_tools.resize_with_pad()` (JAX 实现，JIT 友好)

**参考**: `refs/openpi/src/openpi/shared/image_tools.py`

---

## Stream E: 数据管线 (LeRobot)

### [E1] LeRobot 数据集集成 (层 0 — 可立即开始)

**优先级**: P0
**估计改动**: 新建 `src/vibe_rl/data/`

**描述**:
集成 LeRobot 数据集格式，支持从离线数据集加载 (obs, action, reward) 用于离线 RL 或 imitation learning。

**具体工作**:
1. `pyproject.toml` 添加可选依赖 `[data]`: `lerobot`, `datasets`
2. 新建 `src/vibe_rl/data/`:
   - `dataset.py`: Dataset 协议 (random access + iterable)
   - `lerobot_dataset.py`: LeRobot 数据集加载器
     - 支持 HuggingFace `repo_id` 加载
     - 支持 delta timestamps (action horizon)
     - 返回 `Transition` 格式
   - `data_loader.py`: JAX-compatible DataLoader
     - 基于 torch DataLoader 的多进程加载
     - 自动转换为 JAX arrays
     - 支持 sharding (分布式)
3. 数据格式适配: LeRobot dict → vibe-rl Transition

**验收标准**:
- 可加载 LeRobot 格式数据集
- DataLoader 返回正确 shape 的 JAX batch
- 支持 image + state 混合观测

**参考**: `refs/openpi/src/openpi/training/data_loader.py`

---

### [E2] 数据变换管线 + 归一化统计 (层 1 — 依赖 E1)

**优先级**: P1

**描述**:
实现可组合的数据变换管线和归一化统计计算。

**具体工作**:
1. 新建 `src/vibe_rl/data/transforms.py`:
   - `Transform` 协议: `__call__(sample) -> sample`
   - `TransformGroup`: 组合多个 transform
   - 内建 transforms: resize, normalize, tokenize, pad
2. 新建 `src/vibe_rl/data/normalize.py`:
   - `NormStats` dataclass: mean, std, q01, q99
   - `compute_norm_stats(dataset)`: 遍历数据集计算统计
   - `z_score_normalize` / `quantile_normalize`
   - 统计保存/加载 (JSON)
3. 添加 `scripts/compute_norm_stats.py` 入口脚本

**参考**: `refs/openpi/src/openpi/shared/normalize.py`, `refs/openpi/src/openpi/transforms.py`

---

## Stream F: 推理策略与服务

### [F1] 推理策略封装 (层 2 — 依赖 D1, E2)

**优先级**: P1
**估计改动**: 新建 `src/vibe_rl/policies/`

**描述**:
实现 Policy 封装层，将训练好的模型包装为可部署的推理接口。

**具体工作**:
1. 新建 `src/vibe_rl/policies/`:
   - `policy.py`: Policy 基类
     - `infer(observation) -> action`: 统一推理接口
     - 自动应用 input transforms (normalize, resize)
     - 自动应用 output transforms (unnormalize)
   - `policy_config.py`: `create_trained_policy(config, checkpoint_dir)` 工厂
     - 从 checkpoint 加载模型参数
     - 加载归一化统计
     - 组装完整推理管线
2. 支持 JAX jit 编译推理

**参考**: `refs/openpi/src/openpi/policies/policy.py`, `refs/openpi/src/openpi/policies/policy_config.py`

---

### [F2] WebSocket 推理服务 (层 3 — 依赖 F1)

**优先级**: P2

**描述**:
实现 WebSocket 推理服务器，支持远程策略部署 (机器人控制场景)。

**具体工作**:
1. 新建 `src/vibe_rl/serving/`:
   - `websocket_server.py`: 异步 WebSocket 服务
     - MessagePack 二进制序列化
     - 接收 observation dict → 返回 action
     - 服务端计时指标
     - `/healthz` 健康检查
2. 添加 `scripts/serve_policy.py` 入口
3. 添加 client 示例代码

**参考**: `refs/openpi/src/openpi/serving/websocket_policy_server.py`

---

## Stream G: 日志增强

### [G1] WandB 续训 + 日志增强 (层 0 — 可立即开始)

**优先级**: P0
**估计改动**: `metrics.py`, 训练 runner

**描述**:
增强日志系统，支持 WandB 续训和结构化日志。

**具体工作**:
1. `MetricsLogger` 增加 WandB 续训支持:
   - `wandb_id.txt` 持久化到 run_dir
   - `resume_wandb(run_dir)` → 读取 ID 续训
2. 增强 console logging:
   - 缩写级别名 (D/I/W/E/C) + 毫秒时间戳
   - 训练步进度百分比
3. 训练 runner 统一 logging 初始化

**参考**: `refs/openpi/scripts/train.py` (init_logging, init_wandb)

---

## 并行执行计划

```
Week 1 — 层 0 (5 tasks 并行):
  工程师 1: [A1] Tyro 配置迁移
  工程师 2: [B1] Checkpoint 增强
  工程师 3: [D1] Vision Env + 网络
  工程师 4: [E1] LeRobot 数据集
  工程师 5: [G1] 日志增强

Week 2 — 层 1 (4 tasks 并行):
  工程师 1: [A2] 预设注册表
  工程师 2: [C1] jit + NamedSharding
  工程师 3: [D2] Image Wrappers
  工程师 4: [E2] 变换管线 + 归一化

Week 3 — 层 2+3:
  工程师 1: [C2] FSDP 支持
  工程师 2: [F1] 推理策略封装
  工程师 3: [C3] 修复多卡测试
  工程师 4: [F2] WebSocket 服务
  全员: [H1] 端到端集成测试
```
