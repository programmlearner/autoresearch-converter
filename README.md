# autoresearch-converter

AI agent autonomous research framework for power converter electromagnetic
transient (EMT) surrogate models.

受 [karpathy/autoresearch](https://github.com/karpathy/autoresearch) 启发，为电力电子领域搭建的 AI agent 自主实验框架。让 AI agent 自动迭代训练变流器电磁暂态代理模型，自主探索模型架构、超参数和训练策略。

---

## Overview / 项目概述

This project provides a framework where an AI coding agent (e.g., Claude Code)
autonomously experiments with neural network architectures to build accurate
surrogate models for power electronic converter EMT simulation.

本项目提供一个框架，让 AI 编程 agent（如 Claude Code）自主实验神经网络架构，
构建电力电子变流器电磁暂态仿真的精确代理模型。

**Key idea**: The agent reads `program.md` for instructions, modifies `train.py`
to try different model architectures and training strategies, and iterates to
minimize validation NRMSE — all autonomously.

**核心思想**：Agent 读取 `program.md` 中的指令，修改 `train.py` 尝试不同的模型架构
和训练策略，通过迭代最小化验证集 NRMSE — 全程自主完成。

## Quick Start / 快速开始

### Prerequisites / 前置条件

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Setup / 安装

```bash
git clone https://github.com/programmlearner/autoresearch-converter.git
cd autoresearch-converter

# Install dependencies
uv sync

# Generate synthetic training data
uv run generate_sample_data.py

# Run baseline training (takes ~5 minutes)
uv run train.py
```

### Run with AI Agent / 使用 AI Agent 运行

Point your AI coding agent to `program.md` and let it iterate:

将你的 AI 编程 agent 指向 `program.md`，让它自主迭代：

```bash
# With Claude Code
claude "Read program.md and follow the instructions to run experiments"
```

## Project Structure / 项目结构

| File | Description | 描述 |
|------|-------------|------|
| `program.md` | Agent instructions (human-written, agent reads) | Agent 指令文件 |
| `prepare.py` | Data loading, preprocessing, evaluation (**do not modify**) | 数据基础设施（不可修改） |
| `train.py` | Model definition + training loop (**agent modifies this**) | 模型与训练（agent 修改此文件） |
| `generate_sample_data.py` | Synthetic VSC data generator | 合成数据生成器 |

## Data / 数据

### Synthetic Data / 合成数据

The built-in generator simulates a simplified two-level three-phase voltage
source converter (VSC) with LC filter. It produces 200 scenarios covering:

内置生成器模拟简化的两电平三相电压源变流器（VSC）+ LC 滤波器，生成 200 个场景：

- Steady-state at various power levels / 不同功率水平的稳态
- Load steps / 负荷阶跃
- Voltage sags (single-phase and three-phase) / 电压跌落（单相/三相）
- Frequency deviations / 频率偏差
- Harmonic injection / 谐波注入

**Format**: CSV with columns `time, va, vb, vc, ia, ib, ic`

**Split**: Scenarios 0–159 for training, 160–199 for validation.

### Using Your Own Data / 使用自己的数据

You can replace the synthetic data with real simulation data from PSCAD,
Simulink, or other EMT tools:

你可以用 PSCAD、Simulink 等 EMT 工具的仿真数据替换合成数据：

1. Export time-domain waveforms as CSV files
2. Each CSV must have columns: `time, va, vb, vc, ia, ib, ic`
3. Sampling rate should be 50 kHz (or resample to match)
4. Place files as `scenario_XXXX.csv` in `~/.cache/autoresearch-converter/data/`
5. Use scenarios 0000–0159 for training, 0160–0199 for validation

## Evaluation / 评估方法

The primary metric is **NRMSE** (Normalized Root Mean Square Error):

主要指标是 **NRMSE**（归一化均方根误差）：

```
NRMSE = RMSE / (max(i) - min(i))
```

Averaged across all three phases and all validation scenarios.
在所有三相和所有验证场景上取平均。

Additional metrics / 辅助指标:
- **Peak error**: Relative error in peak current / 峰值电流相对误差
- **THD error**: Absolute error in total harmonic distortion / 总谐波畸变绝对误差
- **Power balance**: Relative error in average power / 平均功率相对误差

## License

MIT
