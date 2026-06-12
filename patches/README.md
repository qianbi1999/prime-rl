# NPU (Ascend) Migration Patches

Patches to run prime-rl RL training on Ascend NPU (910B/C).

## Environment

| Component | Version / Path |
|-----------|---------------|
| NPU | Ascend 910C × 16 |
| CANN | 9.0.0 |
| torch | 2.10.0 + torch_npu 2.10.0 |
| vllm | 0.20.2 + vllm_ascend 0.20.2rc1 |
| transformers | 5.5.3 |
| Model | `/data/z00949579/rl_course/model/Qwen3-1.7B` |

## Patch Files

| Patch | Scope | Description |
|-------|-------|-------------|
| `0001-npu-migration.patch` | All files (combined) | Full migration diff — apply this to reproduce all changes |
| `0002-device-abstraction.patch` | `src/prime_rl/_device.py` (new) | Device abstraction layer — auto-detects NPU/CUDA |
| `0003-npu-wordle-config.patch` | `examples/wordle/rl_npu.toml` (new) | NPU-specific Wordle RL config |

## What Was Changed

### 1. New: `src/prime_rl/_device.py` — Device Abstraction Layer

Central module that auto-detects the accelerator type and provides device-agnostic helpers:

- `get_device_string()` → `"npu"` / `"cuda"`
- `get_device(local_rank)` → `torch.device(...)`
- `get_dist_backend(enable_gloo)` → `"hccl"` / `"nccl"`
- Memory APIs: `reset_peak_memory_stats()`, `max_memory_reserved()`, `mem_get_info()`, etc.
- `get_visible_devices_env()` → `ASCEND_RT_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`
- `get_alloc_conf_env()` → `PYTORCH_NPU_ALLOC_CONF` / `PYTORCH_CUDA_ALLOC_CONF`

### 2. Launcher (`entrypoints/rl.py`)

- `pynvml` → removed; uses `torch.npu.device_count()` instead
- `CUDA_VISIBLE_DEVICES` → `ASCEND_RT_VISIBLE_DEVICES` (dynamic)
- `PYTORCH_CUDA_ALLOC_CONF` → `PYTORCH_NPU_ALLOC_CONF` (dynamic)

### 3. Distributed / Device Setup

| File | Change |
|------|--------|
| `parallel_dims.py` | `device_type` from `_device.py` instead of `_get_available_device_type()` |
| `utils.py` | `setup_torch_distributed` uses `hccl` backend; all `torch.cuda.*` → `_device` helpers |
| `utils/utils.py` | `get_cuda_visible_devices` → reads dynamic env var |
| `utils/nccl.py` | `pynvml` import guarded with `try/except ImportError` |

### 4. Trainer Core

| File | Change |
|------|--------|
| `trainer/rl/train.py` | All `.to("cuda")` → `.to(dt)`; `ProfilerActivity.CUDA` → dynamic; `torch.cuda.*` → `_device` helpers |
| `trainer/model.py` | Device strings → `get_device_string()`; `torch.cuda.get_device_capability()` guarded; `_move_buffers_to_cuda` uses dynamic device |
| `trainer/perf.py` | Ascend 910 FLOPS (~320 TFLOPS BF16) added to peak FLOPS table |
| `trainer/ckpt.py` | `opt._move_states("cuda")` → `opt._move_states(get_device_string())` |
| `trainer/optim.py` | Same `_move_states` fix |
| `trainer/rl/broadcast/__init__.py` | `torch.cuda.current_device()` → `current_device()` from `_device` |

### 5. New Config: `examples/wordle/rl_npu.toml`

Minimal 2-card NPU config with safe defaults:

- `matmul_precision = "highest"` — required for non-CUDA devices (like ROCm)
- `attn = "eager"` — flash-attention not available on NPU
- `optimization_dtype / reduce_dtype = "float32"` — safe starting point
- `impl = "hf"` — HuggingFace model path (avoids custom PrimeRL kernels)
- `weight_broadcast.type = "filesystem"` — avoids NCCL entirely
- 1 trainer NPU + 1 inference NPU

## Usage

```bash
# 1. Apply the combined patch (if starting from clean repo)
git apply patches/0001-npu-migration.patch

# 2. Start vLLM inference server on NPU 15
ASCEND_RT_VISIBLE_DEVICES=15 uv run inference \
  --model.name /data/z00949579/rl_course/model/Qwen3-1.7B \
  --model.impl hf \
  --dtype float32

# 3. Start RL training on NPU 14
ASCEND_RT_VISIBLE_DEVICES=14 uv run rl @ examples/wordle/rl_npu.toml
```

## Known Limitations

- **flash-attn / ring-flash-attn**: Not available → `attn = "eager"` only
- **liger-kernel**: Not available → skipped (Qwen3-1.7B doesn't depend on it)
- **NCCL weight broadcast**: Not supported → `filesystem` mode used instead
- **Context Parallel (CP)**: Not supported (requires ring-flash-attn)
- **FP8 training**: Not supported (requires Hopper SM90+)
- **Memory profiler snapshots**: May not be compatible with PyTorch memory_viz
