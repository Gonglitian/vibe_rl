# Plotting

vibe-rl includes a plotting module for generating publication-quality reward curves from JSONL metrics files. It supports multi-seed aggregation, algorithm comparison, configurable smoothing, and a colorblind-friendly DeepMind palette.

## Installation

Plotting requires the `plotting` extra (installs matplotlib + seaborn):

```bash
uv sync --extra plotting
# or
pip install 'vibe-rl[plotting]'
```

## Quick start

```python
from vibe_rl.plotting import plot_reward_curve

# Single run
fig = plot_reward_curve("runs/ppo_seed0")

# Multi-seed aggregation (mean +/- std shading)
fig = plot_reward_curve(
    ["runs/ppo_seed0", "runs/ppo_seed1", "runs/ppo_seed2"],
    smooth=10,
)

# Compare algorithms
fig = plot_reward_curve(
    ["runs/ppo_run", "runs/dqn_run", "runs/sac_run"],
    group_by="auto",
)
fig.savefig("comparison.png")
```

## `plot_reward_curve()`

The main function reads JSONL metrics, applies smoothing, and renders the reward curve.

```python
from vibe_rl.plotting import plot_reward_curve, PlotConfig

fig = plot_reward_curve(
    run_dirs,
    x_key="step",
    y_key="episode_return",
    smooth=10,
    style="seaborn-v0_8-darkgrid",
    group_by="auto",
    config=PlotConfig(),
    save_path="reward.png",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `run_dirs` | path or list | required | Run directories or direct JSONL paths |
| `x_key` | `str` | `"step"` | JSON key for the x-axis |
| `y_key` | `str` | `"episode_return"` | JSON key for the y-axis |
| `smooth` | `int \| None` | `None` | Override `PlotConfig.smooth_radius` |
| `style` | `str \| None` | `None` | Override matplotlib style |
| `group_by` | `str` | `"auto"` | `"auto"` groups by algorithm name, `"none"` plots each run separately |
| `config` | `PlotConfig \| None` | `None` | Full config; keyword overrides are applied on top |
| `save_path` | path \| None | `None` | Save figure to this path |

**Returns:** `matplotlib.figure.Figure`

### Algorithm detection

When `group_by="auto"`, the function reads `config.json` inside each run directory (checking keys `algorithm`, `algo`, `algo_name`). If not found, it falls back to the directory name with timestamp suffixes stripped (e.g. `ppo_cartpole_20260215_143022` becomes `ppo_cartpole`).

Runs with the same detected name are treated as different seeds and aggregated.

## Single run

Pass a single run directory to plot one reward curve:

```python
fig = plot_reward_curve("runs/CartPole-v1_ppo_20260216_143022")
```

The function looks for `logs/metrics.jsonl` inside the run directory. If you pass a direct file path instead, it uses that file.

## Multi-seed aggregation

Pass multiple directories with the same algorithm name to aggregate across seeds:

```python
fig = plot_reward_curve([
    "runs/ppo_seed0",
    "runs/ppo_seed1",
    "runs/ppo_seed2",
])
```

The aggregation pipeline:

1. Each seed's `(x, y)` series is loaded and smoothed independently.
2. All series are interpolated onto a common x-grid (using `np.interp`).
3. The mean is plotted with a shaded region controlled by `PlotConfig.shaded`:
   - `"std"` (default) — mean +/- 1 standard deviation
   - `"stderr"` — mean +/- standard error
   - `"none"` — no shading

## Multi-algorithm comparison

To compare algorithms, pass runs from different algorithms and use `group_by="auto"`:

```python
fig = plot_reward_curve(
    ["runs/ppo_run", "runs/dqn_run", "runs/sac_run"],
    group_by="auto",
)
```

Each algorithm gets a distinct color from the palette and appears in the legend. If you have multiple seeds per algorithm, they are aggregated within each group.

## Smoothing

Two smoothing modes are available, selected via `PlotConfig.smooth_mode`:

### Window averaging (default)

Symmetric moving average, SpinningUp-style. Each output sample is the mean of a window of size `2 * radius + 1`:

```python
from vibe_rl.plotting import smooth_window

smoothed = smooth_window(y, radius=10)
# y'[i] = mean(y[max(0, i-10) : i+11])
```

Edges are padded to avoid boundary artifacts.

### Exponential moving average

EMA smoothing with `alpha = 2 / (span + 1)`, matching pandas EWM semantics:

```python
from vibe_rl.plotting import smooth_ema

smoothed = smooth_ema(y, span=10)
```

Select the mode in config:

```python
from vibe_rl.plotting import PlotConfig

# Window averaging (default)
cfg = PlotConfig(smooth_radius=10, smooth_mode="window")

# EMA
cfg = PlotConfig(smooth_radius=20, smooth_mode="ema")
```

## Color palette

The default palette is adapted from DeepMind publications — 10 distinct, colorblind-friendly colors:

```python
from vibe_rl.plotting import DEEPMIND, get_colors, set_palette, reset_palette, color_for

# View current palette
print(get_colors())
# ['#0077BB', '#EE7733', '#009988', '#CC3311', '#33BBEE',
#  '#EE3377', '#BBBBBB', '#AA3377', '#DDCC77', '#44BB99']

# Get color by index (cycles if more groups than colors)
color = color_for(0)  # '#0077BB'

# Use a custom palette
set_palette(["#e41a1c", "#377eb8", "#4daf4a"])

# Restore default
reset_palette()
```

Colors are assigned automatically by `plot_reward_curve()` — one per algorithm group. If there are more groups than palette entries, colors cycle.

## `PlotConfig` reference

All plot settings are controlled by the frozen `PlotConfig` dataclass:

```python
from vibe_rl.plotting import PlotConfig

cfg = PlotConfig(
    smooth_radius=10,
    smooth_mode="window",
    shaded="std",
    figsize=(8, 6),
    dpi=150,
    style="seaborn-v0_8-darkgrid",
    save_format="png",
    title=None,
    xlabel="Step",
    ylabel="Episode Return",
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `smooth_radius` | `int` | `10` | Half-window size for window averaging, or span for EMA |
| `smooth_mode` | `str` | `"window"` | `"window"` or `"ema"` |
| `shaded` | `str` | `"std"` | `"std"`, `"stderr"`, or `"none"` |
| `figsize` | `tuple` | `(8, 6)` | Figure size `(width, height)` in inches |
| `dpi` | `int` | `150` | Resolution for raster output |
| `style` | `str` | `"seaborn-v0_8-darkgrid"` | Matplotlib style name |
| `save_format` | `str` | `"png"` | Default file extension (`"png"`, `"pdf"`, `"svg"`) |
| `title` | `str \| None` | `None` | Figure title |
| `xlabel` | `str` | `"Step"` | X-axis label |
| `ylabel` | `str` | `"Episode Return"` | Y-axis label |

Override individual fields while keeping defaults for the rest:

```python
import dataclasses

cfg = PlotConfig(smooth_radius=20, dpi=300)
# or override from an existing config
cfg2 = dataclasses.replace(cfg, shaded="stderr")
```

## CLI script

`scripts/plot_rewards.py` provides a command-line interface for quick plotting:

```bash
# Plot by experiment name (auto-discovers matching runs)
python scripts/plot_rewards.py --experiment-name cartpole_ppo

# Plot specific run directories
python scripts/plot_rewards.py --run-dirs runs/ppo_seed0 runs/ppo_seed1

# Compare algorithms
python scripts/plot_rewards.py --compare \
    --run-dirs runs/ppo_run runs/dqn_run runs/sac_run

# Custom y-axis and smoothing
python scripts/plot_rewards.py --experiment-name cartpole_ppo \
    --y-key total_loss --smooth-radius 20 --output loss_curve.png
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--run-dirs` | — | Explicit run directories to plot |
| `--experiment-name` | — | Auto-discover runs matching this name |
| `--base-dir` | `runs` | Base directory for run discovery |
| `--x-key` | `step` | JSONL key for x-axis |
| `--y-key` | `episode_return` | JSONL key for y-axis |
| `--output` | auto | Output path (defaults to `<run>/artifacts/reward_curve.png`) |
| `--compare` | `false` | Enable multi-algorithm comparison mode |
| `--smooth-radius` | `10` | Smoothing radius |
| `--smooth-mode` | `window` | `window` or `ema` |
| `--shaded` | `std` | `std`, `stderr`, or `none` |
| `--figsize` | `(8, 6)` | Figure size in inches |
| `--dpi` | `150` | Output DPI |
| `--style` | `seaborn-v0_8-darkgrid` | Matplotlib style |
| `--title` | — | Figure title |

At least one of `--run-dirs` or `--experiment-name` is required.

## Auto-plotting after training

When `runner.plot=True` (the default) and matplotlib is installed, the runner automatically saves a reward curve to `artifacts/reward_curve.png` after training completes:

```bash
# Auto-plot is on by default
python scripts/train.py cartpole_ppo
# → runs/CartPole-v1_ppo_*/artifacts/reward_curve.png

# Disable auto-plotting
python scripts/train.py cartpole_ppo --runner.plot false
```

The auto-generated plot uses default `PlotConfig` settings. For custom styling, use the CLI script or Python API after training.

## Full example

```python
import dataclasses
from vibe_rl.plotting import PlotConfig, plot_reward_curve, set_palette

# Custom palette
set_palette(["#e41a1c", "#377eb8", "#4daf4a"])

# Custom config
cfg = PlotConfig(
    smooth_radius=20,
    smooth_mode="ema",
    shaded="stderr",
    figsize=(10, 6),
    dpi=300,
    title="CartPole: PPO vs DQN vs SAC",
    ylabel="Episode Return",
    save_format="pdf",
)

# Plot comparison
fig = plot_reward_curve(
    [
        "runs/ppo_seed0", "runs/ppo_seed1", "runs/ppo_seed2",
        "runs/dqn_seed0", "runs/dqn_seed1", "runs/dqn_seed2",
        "runs/sac_seed0", "runs/sac_seed1", "runs/sac_seed2",
    ],
    group_by="auto",
    config=cfg,
    save_path="cartpole_comparison.pdf",
)
```

This produces a 3-algorithm comparison with EMA smoothing, standard-error shading, and PDF output — ready for a paper.
