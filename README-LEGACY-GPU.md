# llama.cpp-legacy-gpu

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) with native C++ patches for
**heterogeneous multi-GPU inference** on pools combining legacy Tesla M10/M60 GPUs with
modern RTX cards.

> This is a **focused fork** — only `common/fit.cpp` is modified. All other upstream
> llama.cpp functionality is preserved and regularly rebased from the main repository.

---

## What is changed and why

### `common/fit.cpp` — GPU Layer Fitting Algorithm

The function `common_params_fit_impl()` determines how model layers are distributed
across GPU devices. The standard algorithm uses a "false position" iterative method
that converges to an approximately equal distribution weighted by free VRAM.

**Problem with heterogeneous pools:** On N04-RTX (see hardware spec below), the
standard algorithm distributes `qwen3.6:35b` (22 GiB, 42 layers) across all 12 GPUs —
including Tesla M10 (83 GB/s). Each decode token then traverses a pipeline of up to
12 GPU hops over PCIe. The Tesla M10 stages add ~5-10ms of PCIe latency per token,
reducing throughput from ~18 tok/s to ~5 tok/s.

#### Patch 1: Greedy Fill (`OLLAMA_FORCE_GPU_LAYERS`)

When `OLLAMA_FORCE_GPU_LAYERS` is set to a positive integer, the standard fitting
algorithm is bypassed and replaced with a greedy sequential fill:

```cpp
// Fills GPU with highest CUDA index (best bandwidth) first
for (int id = nd - 1; id >= 0 && layers_left > 0; id--) {
    effective_budget = (free_vram[id] - margins[id]) / bandwidth_factor[id]
    n_layers = effective_budget / bytes_per_layer
    assign n_layers to GPU id
}
```

The greedy fill uses an **adaptive overhead scale** that accounts for the ratio of
fixed compute overhead (KV cache + CUDA context) to model weight size. This scale is
also tunable via `OLLAMA_LAYER_OVERHEAD_SCALE`.

**Result:** `qwen3.6:35b` on N04-RTX uses 3–4 RTX GPUs instead of 12, reducing
pipeline hops and improving throughput from ~5 to ~24 tok/s.

#### Patch 2: Bandwidth-Weighted Budget Scaling (`OLLAMA_GPU_BANDWIDTHS`)

When `OLLAMA_GPU_BANDWIDTHS=83,83,83,83,160,160,192,336,336,336,360,360` is set
(comma-separated bandwidth in GB/s for each GPU in CUDA order), the raw VRAM budget
for each GPU is divided by its bandwidth factor:

```cpp
bandwidth_factor = max_bw / this_gpu_bw
effective_budget = raw_budget / bandwidth_factor
```

For Tesla M10 (83 GB/s) with max_bw = 360 GB/s: `factor = 4.3×`  
→ M10 gets 4.3× fewer layers than its VRAM would theoretically allow.

**Why this prevents pipeline bottlenecks:** Even if M10 has free VRAM, assigning many
layers to it forces every decode token to spend 4.3× longer on M10 compared to RTX.
Reducing M10's layer count proportionally keeps the pipeline balanced by throughput,
not just VRAM.

#### Patch 3: VRAM-Weighted Fallback (replaces equal distribution)

When greedy fill cannot place all layers (scale too tight, model too large for best
GPUs alone), the fallback is VRAM-weighted distribution:

```cpp
tensor_split[i] = (free_vram[i] - margins[i]) / sum(free_vram - margins)
```

The original equal distribution (`tensor_split = [1, 1, 1, ...]`) caused OOM on
Tesla M10 for large models because the primary CUDA device (CUDA 0 = M10 in standard
worst→best ordering) needs to allocate the full gallocr compute buffer, which scales
with `n_ctx × n_heads × 4 bytes` regardless of layer count.

VRAM-weighted distribution gives M10 a proportionally smaller share of model layers,
reducing its memory pressure while still allowing it to contribute.

#### Patch 4: Tier-Threshold Bypass

When `OLLAMA_GPU_TIER_THRESHOLD > 0`, the "no changes needed" early return is skipped.
This allows tier-aware layer assignment (used by Ollama's higher-level patches) to run
even when the standard fitting algorithm would already be satisfied.

---

## Reference Hardware: N04-RTX

| Component | Spec |
|-----------|------|
| CPU | AMD EPYC 3151 4-Core |
| RAM | 128 GiB DDR4 ECC |
| OS | Ubuntu 22.04.5 LTS |
| CUDA driver | 12.0.1 |
| GPU interconnect | PCIe (no NVLink) |

**GPUs (CUDA order, worst → best bandwidth):**

| CUDA | Device | Arch | CC | VRAM | Bandwidth |
|------|--------|------|----|------|-----------|
| 0–3 | Tesla M10 (4 GPU dies) | Maxwell | 5.0 | 8 GiB | 83 GB/s |
| 4–5 | Tesla M60 (2 GPU dies) | Maxwell | 5.2 | ~7.7 GiB | 160 GB/s |
| 6 | GTX 1060 6GB | Pascal | 6.1 | 6 GiB | 192 GB/s |
| 7–9 | RTX 2060 12GB (×3) | Turing | 7.5 | 12 GiB | 336 GB/s |
| 10–11 | RTX 3060 (×2) | Ampere | 8.6 | 12 GiB | 360 GB/s |
| **Total** | **12 endpoints** | | | **~114 GiB** | |

The CUDA ordering (worst → best) is intentional: the greedy fill starts at CUDA 11
(RTX 3060, 360 GB/s) and extends to CUDA 0 (Tesla M10, 83 GB/s) only if needed.

---

## Usage

This fork is designed to be used as the `LLAMA_CPP_FORK` in the
[ollama-legacy-gpu](https://github.com/h3rb3rn/ollama-legacy-gpu) build:

```dockerfile
ARG LLAMA_CPP_FORK=https://github.com/h3rb3rn/llama.cpp-legacy-gpu.git
```

The fork patches activate only when the relevant `OLLAMA_*` environment variables are
set by Ollama's `selectGPUPool()` function. Without these env vars, the standard
llama.cpp fitting algorithm runs unmodified.

### Environment variables

| Variable | Type | Effect |
|----------|------|--------|
| `OLLAMA_FORCE_GPU_LAYERS` | integer > 0 | Activate greedy fill (bypass standard fitting) |
| `OLLAMA_LAYER_OVERHEAD_SCALE` | float (1.0–5.0) | Override adaptive scale (set by auto-optimizer) |
| `OLLAMA_OVERHEAD_PER_GPU_MB` | integer | Fixed compute overhead per GPU in MB (default: 1536) |
| `OLLAMA_GPU_BANDWIDTHS` | CSV of GB/s | Per-GPU bandwidth in CUDA order for budget scaling |
| `OLLAMA_GPU_MAX_BANDWIDTH` | integer | Maximum bandwidth (denominator for scaling factors) |
| `OLLAMA_GPU_TIER_THRESHOLD` | integer | CUDA index below which GPUs are "legacy" tier |

---

## Compatibility

- **CUDA 12.0.1+** required for CC 5.0 support
- Builds with Maxwell (CC 5.0/5.2), Pascal (CC 6.x), Turing (CC 7.5), Ampere (CC 8.x)
- Upstream llama.cpp changes are regularly rebased onto the `legacy-gpu-support` branch

---

## Related

- **Upstream llama.cpp**: https://github.com/ggml-org/llama.cpp (MIT)
- **ollama-legacy-gpu** (Ollama fork, uses this fork): https://github.com/h3rb3rn/ollama-legacy-gpu

---

## License

MIT License — same as upstream llama.cpp.

Modifications in this repository are Copyright (c) 2025-2026 Philipp Horn.  
Original llama.cpp code is Copyright (c) 2023-2026 The ggml authors.  
See [LICENSE](LICENSE) for the full MIT license text.
