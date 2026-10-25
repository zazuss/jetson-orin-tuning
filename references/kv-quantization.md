# KV cache quantization — why it decides your context ceiling

> **中文速览** —— 这份文档讲 KV 量化为什么决定你的上下文天花板：哪些推理引擎支持、llama.cpp 具体怎么传参（`-ctk q4_0 -ctv q4_0 -fa on -np 1`，其中 `-fa on` 是必须的、`-np 1` 比 KV 量化更关键）、q4_0 在 256K 下的实测代价（只剩 39 MB 余量）、MLC-LLM 没有这个能力的三条证据、改不动引擎时用 sliding_window_size 兜底，以及 NvMap/CMA 那个经典误诊。


Worth knowing before you pick an engine: **KV-cache quantization is an architectural property of the runtime, not an optional flag you can assume exists.** Two engines on the same board, same model, same weights can differ by 8x in feasible context purely because one of them can store K/V in 8 bits and the other can't.

## The arithmetic that actually matters

Per-token KV cost:

```
bytes/token = 2 (K and V) × n_layers_caching × n_kv_heads × head_dim × bytes_per_element
```

On hybrid-attention models (Qwen3-Next, many new MoE designs), only a fraction of layers keep a full KV cache — divide by `full_attention_interval`. Getting this wrong overestimates by 4x and makes you conclude 64K is impossible when it isn't.

Measured on this board (Qwen3-1.7B, 28 layers, 8 KV heads, head_dim 128, dense attention, fp16):

```
2 × 28 × 8 × 128 × 2 B = 114 688 B/token = 112 KB/token
```

| Context | KV cache alone |
|---|---|
| 4 096 | 459 MB |
| 8 192 | 918 MB |
| 32 768 | 3 672 MB |
| 65 536 | 7 344 MB |

On a board with **7 485 MB** usable, fp16 KV puts 64K out of reach by construction — 7.3 GB of KV plus 0.9 GB of weights exceeds managed memory before you account for the working buffers.

## The same model, two engines, two ceilings

| Engine | Model | KV bytes/token | KV quant available | Largest context actually run | Speed there |
|---|---|---|---|---|---|
| llama.cpp | Ternary-Bonsai-2 9B (hybrid, `full_attention_interval=4`) | **32 KB** | ✅ `-ctk q8_0` / `q4_0` | **262 144** (256K) | 36.08 t/s decode, `q4_0` KV, single slot |
| llama.cpp | Ternary-Bonsai-2 9B | 32 KB | ✅ `-ctk q8_0` | **131 072** (128K) | 36.67 t/s decode, 1.4 GB headroom left |
| llama.cpp | Ternary-Bonsai-2 9B | 32 KB | ✅ `-ctk q8_0` | **65 536** (64K) | 36.10 t/s decode, 300.56 t/s prefill |
| llama.cpp | Qwen3-1.7B Q4_0 | 112 KB | ✅ `-ctk q8_0` | not pushed to the limit | 40.8 t/s at short ctx |
| MLC-LLM | Qwen3-1.7B q4f16_1 | 112 KB | ❌ **none** | **8 192** | 14.96 t/s decode, 129 t/s prefill |
| MLC-LLM | Qwen3-1.7B q4f16_1 | 112 KB | ❌ | 32 768 | **OOM** |

**Context length is nearly free in decode.** Measured on this board, 64K → 128K → 256K all hold 36 t/s. The cost of a longer context is paid entirely in memory, never in generation speed — because the 24 of 32 layers that use linear/recurrent attention keep a state whose size does not grow, so the per-token weight stream that bounds decode is unchanged. What *does* grow is the prefill buffer, and that is what OOMs first.

The 9B model reaches 64K while the 1.7B model cannot reach 32K. That is not a model-size paradox — it's two independent multipliers:

1. **Architecture**: hybrid attention makes the 9B store ~38 KB/token (fp16, measured) instead of 128 KB — a 3.4x saving from the layer count alone.
2. **Runtime**: llama.cpp can quantize K/V, MLC cannot.

Note that **weight quantization does not shrink KV**. Qwen3-1.7B's weights go from ~3.4 GB fp16 to 961 MB with `q4f16_1` — an active, working 3.5x saving on weights — while KV stays at a fixed 112 KB/token. People conflate the two constantly, then wonder why their 4-bit model still can't hold long context.

## llama.cpp: how to turn it on

```bash
llama-cli -m model.gguf -ctk q8_0 -ctv q8_0 -fa on -ngl 99
```

- `-ctk` / `-ctv` quantize the K and V caches independently. `q8_0` is the safe default: ~2x smaller, quality loss is negligible at 8 bits.
- **`-fa on` (flash attention) is required** for quantized KV to be efficient — the quantized path is implemented in the flash-attention kernels. Without it you get correctness with poor speed, or a fallback.
- `q4_0` for KV is what makes 256K possible on an 8 GB board, and on this model it costs nothing measurable in decode speed (36.08 t/s at 256K vs 36.10 t/s at 64K with fp16). It costs memory headroom instead: `q4_0` at 128K leaves 1.4 GB free, `q8_0` at 128K leaves 360 MB, `q4_0` at 256K leaves 39 MB. Prefer `q8_0` below 128K; reach for `q4_0` only when the context demand forces it.
- Check `llama-bench -p <ctx> -n 64 -fa on -ctk q8_0` rather than trusting the flag took effect. Note `llama-bench` rejects `-fa 1` — it takes `on|off|auto` — while `-c` is not a `llama-bench` flag at all; context there comes from `-p` and `-n`.

Verify the saving landed by watching the reported KV cache size at load, or by comparing process RSS at a large `-p`.

## MLC-LLM: confirmed absent (three independent checks)

As of mlc-llm commit `8978ea9` (2026-10-06):

| Check | Result |
|---|---|
| GitHub issue search `repo:mlc-ai/mlc-llm kv cache quantiz` | **0 results** |
| `EngineConfig` (24 fields) | Only `kv_state_kind`, `Literal["kv_cache","rnn_state"]` — selects KV vs RNN state for hybrid models, **nothing to do with precision** |
| Grep `kv_cache_dtype` across `.py/.cc/.h/.cu` (excluding 3rdparty) | **3 hits**, all in `model_preset.py` → `transformers.js_config`, hardcoded `"float16"` (a browser-runtime compat field) |

There is no supported path. If you want it, it's a source change:

| Layer | File | What has to change |
|---|---|---|
| Python API | `python/mlc_llm/nn/kv_cache.py` | `dtype: str` parameter must flow through to `mlc.create_paged_kv_cache_generic` |
| Dispatch | `python/mlc_llm/op/attention.py` | `and q.dtype == "float16"` — this single condition pins the flashinfer path to fp16 and is the hardest blocker |
| Storage | `3rdparty/tvm/src/runtime/vm/paged_kv_cache.cc` | Storage layout, page packing, read/write paths |

The real cost isn't those three edits — it's that the **attention kernel has to understand low-precision KV**. You either insert a dequant step (loses the bandwidth win you were chasing) or write a fused dequant+attention CUDA kernel. llama.cpp's `fattn-vec` took roughly two years to mature. Budget 2–3 weeks for something worse than the existing alternative.

## What to do instead, in order

1. **Compute bytes/token first** (with `full_attention_interval` applied). If `context × bytes/token` exceeds ~60% of usable memory, you have a structural problem, not a tuning problem.
2. **Prefer an engine that supports KV quantization** if you need long context. On this board that means llama.cpp-family, where `-ctk q8_0 -ctv q8_0 -fa on` is one flag away.
3. **If stuck on an engine without KV quant**, cap the context instead of letting it OOM:
   - Compile-time sliding window: `mlc_llm compile . --overrides "sliding_window_size=8192"` → KV fixed at ~900 MB, long conversations stop growing, at the cost of forgetting everything past 8K tokens.
   - Or pick a hybrid-attention model (Qwen3-Next style) where only a quarter of layers cache anything.
4. **Do not** reach for swapping weight quantization levels hoping it helps context. It doesn't move KV at all.

## Diagnostic trap: `NvMap ... error 12`

When MLC runs out of memory during KV cache allocation, the log fills with:

```
NvMapMemAllocInternalTagged failed: error 12
NvMapMemHandleAlloc failed: error 12
```

These are **collateral noise, not the cause**. They lead people to blame CMA — the 256 MB `CmaTotal` on Jetson that shows up in every forum thread. Measured directly: `cudaMallocManaged` allocates **4 GB** successfully and `cudaHostAlloc` allocates 1 GB; CMA and the CUDA allocation path are unrelated. The real error is always the last line of the trace:

```
CachedPagedKVCacheAuxDataManager::CachedPagedKVCacheAuxDataManager
  → Tensor::Empty → cuda_device_api.cc:173 CUDADeviceAPI::AllocDataSpace
  → cudaErrorMemoryAllocation (2): out of memory
```

Also: "3801 MB free" reported by the system says nothing about **single contiguous allocation**, which caps at about **4 576 MB** on this board (5 376 MB cumulative across blocks).

## Platform

Jetson Orin Nano 8GB (P3767), L4T R39.2.1, CUDA 13.2, sm_87, kernel 6.8.12-tegra, managed memory 7.31 GiB. All numbers above measured on this board; re-measure before quoting on NX/AGX.
