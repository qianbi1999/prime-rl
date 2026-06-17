# Ascend NPU RL Training Guide

在 Ascend NPU (910B/C) 上跑通 prime-rl Wordle RL 训练的完整指南。

## 基础环境

| 项目 | 版本/路径 |
|------|----------|
| 基础镜像 | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` |
| CANN | 9.0.0 |
| torch | 2.10.0 + torch_npu 2.10.0 |
| vllm | 0.20.2 + vllm_ascend 0.20.2rc1 |
| transformers | 5.5.3+ |
| Python | 3.11+ |

## 快速开始

### 1. 应用 NPU 适配 patch

```bash
cd prime-rl
bash patches/apply.sh
```

三层 patch：
- `0001-device-abstraction.patch` — 设备抽象层 (`_device.py`)，自动检测 NPU/CUDA
- `0002-source-adaptation.patch` — 11 个源文件的 CUDA→NPU 替换
- `0003-npu-config.patch` — Wordle RL NPU 训练配置 (`rl_npu.toml`)

### 2. 安装依赖

```bash
# 基础依赖
pip install -r requirements-npu.txt

# prime-rl 本地包
pip install -e packages/prime-rl-configs
pip install -e . --no-deps

# torchtitan
pip install git+https://github.com/pytorch/torchtitan@a1fdd7e

# wordle 环境（从 git submodule）
git config submodule.verifiers.url https://github.com/PrimeIntellect-ai/verifiers.git
git submodule update --init deps/verifiers
pip install -e deps/verifiers --no-deps
pip install -e deps/verifiers/environments/wordle
```

### 3. NLTK 数据

```bash
python3 -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
```

### 4. 准备模型权重

将 HuggingFace 格式的 Qwen3-1.7B 权重（SFT 后）放到本地路径，例如：
`/data/model/checkpoint_wordle_sft/step-20/`

### 5. 配置

编辑 `examples/wordle/rl_npu.toml`：
```toml
[model]
name = "/data/model/checkpoint_wordle_sft/step-20"  # 你的模型路径
```

### 6. 启动训练

```bash
# 先查卡
npu-smi info

# 两张空闲卡，例如 phy-id 4,5
ASCEND_RT_VISIBLE_DEVICES=4,5 WANDB_MODE=offline \
  python3 -m prime_rl.entrypoints.rl @ examples/wordle/rl_npu.toml
```

### 7. 查看日志

```bash
tail -f outputs/logs/orchestrator.log   # reward, turns, error 率
tail -f outputs/logs/trainer.log        # loss, entropy, grad norm, 吞吐
tail -f outputs/logs/inference.log      # vLLM 引擎状态
```

## 配置说明

| 参数 | 值 | 为什么 |
|------|-----|--------|
| `seq_len = 1024` | Wordle 6 轮 × 150 token ≈ 900，1024 刚好 | 防 OOM |
| `optimization_dtype = float32` | NPU bf16 下 softmax 溢出 NaN | 精度换稳定性 |
| `attn = eager` | flash-attn 在 NPU 上不可用 | 小模型影响不大 |
| `matmul_precision = highest` | NPU/ROCm 需要完整 FP32 matmul | 防 softmax 精度崩 |
| `gpu_memory_utilization = 0.80` | 默认 0.90 太激进 | 留空间给碎片 |
| `weight_broadcast.type = filesystem` | HCCL 不完全兼容 | 走文件最稳 |
| `batch_size = 8, group_size = 4` | float32 下内存受限 | 小 batch 噪声大 |
| `lr = 1e-7` | 1e-6 导致权重爆炸 | RL 对 lr 比 SFT 敏感 |

## 常见问题

### NaN logprob / "Out of range float values"

vLLM 返回 400，orchestrator 日志 Error 率飙升。

**根因**：bf16 下 log_softmax 溢出（词表 152064 太大），logprob=NaN，JSON 不可序列化。

**修复**：切 `optimization_dtype = float32`。

### Rollout inflight 挂起

16 inflight 始终不 buffered，推理请求不完成。

**排查**：
```bash
curl -s http://localhost:8000/v1/models  # 如果挂起 → vLLM 死锁
grep "200 OK" outputs/logs/inference.log  # 检查是否有成功请求
```

**修复**：换一张 NPU 卡，或重启训练。

### Grad Norm 爆炸

Grad Norm 从 1 飙到 4 亿。

**根因**：小 batch 下一条异常 rollout 导致梯度尖峰。

**修复**：加大 `batch_size` 或降低 `lr`。

### acl 硬件异常

`aclnnNonzeroV2 (507015)` / `vector core exception (507035)`。

**507015**：NPU boolean mask 索引 bug，patch 已含 workaround（先移到 CPU 再做 mask）。

**507035**：硬件层异常，换 NPU 卡或重启。

### 僵尸进程清理

```bash
# 查看
ps aux | grep "PRIME-RL\|VLLM::EngineCore"

# 清理
kill -9 $(ps aux | grep -E "PRIME-RL|VLLM::EngineCore" | grep -v grep | awk '{print $2}')
```

### NPU 卡被占

```bash
npu-smi info | grep -A40 "Process"
```

如果目标卡有别人的进程，换空闲卡：
```bash
ASCEND_RT_VISIBLE_DEVICES=<free-ids> ...
```

## 已知限制

- **flash-attn / ring-flash-attn / liger-kernel**：NPU 不支持，attn 必须 eager
- **NCCL weight broadcast**：走 filesystem，不支持 NCCL
- **Context Parallel (CP)**：不可用（依赖 ring-flash-attn）
- **FP8 训练**：不支持（需要 Hopper SM90+）
- **bf16 训练**：大词表模型不稳定，建议 float32

## 性能参考

Qwen3-1.7B Wordle RL, 2×Ascend 910C, float32, seq_len=1024, batch=8:

| 指标 | 值 |
|------|-----|
| 训练吞吐 | ~2000 tokens/s |
| MFU | ~7% |
| 峰值内存 | ~37 GiB |
| 每步耗时 | ~2s（编译后） |
| Reward | 0.3-0.8（29步，SFT 初始化） |
