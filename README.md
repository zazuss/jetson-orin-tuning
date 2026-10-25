# Jetson Orin tuning notes

Measured, reproducible performance notes for LLM inference on Jetson Orin (Nano 8GB, Super, NX, AGX). Every number here came from a real measurement on a 7.3 GiB Orin Nano; nothing is copied from a datasheet or estimated from a formula that wasn't then tested.

The point of this repo is to stop you from tuning the wrong knob. On an 8 GB Orin, most of what people adjust — thread counts, batch size, GPU clocks, speculative-decoding depth — is **noise**, and the things that actually move the needle are structural: quantization layout, vocabulary size, memory bandwidth, whether the workload is compute- or bandwidth-bound at all, and whether the runtime can quantize its KV cache.

## Seven conclusions, each with evidence

### 1. Find out which bottleneck you have before touching anything

Sample `tegrastats` **while generation is running**, not at idle:

| GPU utilization | EMC utilization | Diagnosis | What actually helps |
|---|---|---|---|
| ≥90% | <60% | **Compute-bound** (dequant/ALU) | Nothing tunable. Smaller model or cheaper quant format. |
| <70% | ≥85% | **Bandwidth-bound** | Only reducing weight bytes helps |
| Both low | Both low | Serialization / kernel launch overhead | Batching concurrent requests, not single-stream tuning |

Measured on a 2B Q4 model: **GR3D median 94%, EMC median 43%, CPU 5%, board 6.9 W** → compute-bound. In that state, no flag matters. High GPU utilization is *not* good news by itself — GPU busy plus bandwidth idle means the ALU is burning cycles on work that produces no tokens.

→ [`references/throughput-bottleneck.md`](references/throughput-bottleneck.md)

### 2. Quantization *format* matters more than size — and "smaller is faster" is false

Same model, same prompt, `-ngl 99 -fa 1 --temp 0 -n 150`:

| Model / quant | Size | Single-stream tok/s |
|---|---|---|
| Qwen3-0.6B Q4_K_M | 378 MB | **64.4** |
| Qwen3-1.7B Q3_K_S | 827 MB | **22.4** ← smaller, twice as slow |
| Qwen3-1.7B Q4_0 | 1008 MB | 40.8 |
| Qwen3.5-2B Q4_0 | 1159 MB | 34.7 |
| Qwen3.5-2B Q4_K_M | 1222 MB | 32.1 |

**On a weak GPU, prefer simple block quantization (Q4_0/Q5_0) and avoid K-quants.** Q4_0 packs 32 weights per block with one fp16 scale — dequant is a single multiply-add that folds into the trailing integer accumulation. K-quant uses 256-weight super-blocks with a 6-bit scale *and* 6-bit min, re-split into 8 sub-blocks, each requiring its own min+scale decode. The instruction count per weight multiplies. That is why Q3_K_S is 45% slower than a *larger* Q4_0 file.

### 3. A large vocabulary is a hidden per-token tax

`lm_head params = n_vocab × n_embd`, and every single generated token streams that entire block to produce logits:

| Model | n_vocab × n_embd | Share of params |
|---|---|---|
| Qwen3.5-2B | 248320 × 2048 = 508 M | **27%** |
| Llama-3.2-3B | 128256 × 3072 ≈ 392 M | 12% |
| SmolLM2-1.7B | 49152 × 2048 ≈ 101 M | 6% |

This is why a 1.19 GB Qwen3.5-2B (34.7 tok/s) is slower than a 1.9 GB Llama-3.2-3B (43.07 tok/s official). If vocab share exceeds ~20%, expect speed that looks wrong for the file size.

### 4. Long context is nearly free on hybrid-attention models

Ternary-Bonsai-2 9B (`PTQ1_0`), 7.3 GiB Orin, `llama-bench`:

| Context | Prefill t/s | Generation t/s |
|---|---|---|
| 128 | 360.75 | 36.57 |
| 2 048 | 376.12 | 36.46 |
| 8 192 | 367.23 | 36.59 |
| 16 384 | 354.64 | 36.24 |
| 32 768 | 334.09 | 36.20 |
| 65 536 | **300.56** | **36.10** |

512x more context costs 1.3% of generation speed. Steady-state RSS goes from 2 716 MB at 32K to 2 775 MB at 64K — a 2x context increase for 59 MB.

The naive KV formula assumes every layer caches K and V:

```
2 × 32 layers × 4 kv_heads × 256 key_len × 2 bytes = 128 KB/token  →  8 GiB at 64K
```

That predicts an impossible 8 GiB. The model uses **hybrid attention** (`full_attention_interval = 4`), so only ~¼ of layers keep a full KV cache. **Compute KV cost from full-attention layer count, never from `block_count`.** Dense-attention arithmetic overestimates by ~4x on this architecture.

→ [`references/context-scaling.md`](references/context-scaling.md)

### 5. Your memory budget is smaller than the box says

On a "8 GB" Orin Nano:

| Figure | Value |
|---|---|
| Nominal | 8 GB |
| Present | 8 050 304 kB = **7.68 GiB** |
| Managed (usable by CUDA) | 7 665 092 kB = **7.31 GiB** |

The missing ~890 MB is 512 MB of device-tree carveout (`vpr-carveout`, `camdbg_carveout`, `fsi-carveout`, `rce-reservation`, `ramoops`, `framebuffer`) plus ~370 MB of kernel.

Two numbers that get misdiagnosed constantly:

- A single `cudaMalloc` caps at **4.85 GB** (measured: 1/2/4 GB succeed, 5 GB and 6.5 GB OOM). This is the UMA allocator, not a fragment limit.
- The "1 GB limit on Jetson" that people remember belongs to the **CMA/NvMap** path — `CmaTotal` 256 MB, `CmaFree` 3.3 MB — used by V4L2/ISP/display. GGML and CUDA never touch CMA, so CMA is never why your model fails to load.

### 6. Cage the process, don't hope

A 64K prefill peaked at **6 914 / 7 485 MB**, leaving 83 MB free, then settled back to 2.8 GB. During that peak an OOM took down an unrelated process (a gateway daemon was killed at 11:10:55; `dmesg` had been showing `all_unreclaimable? yes` with 33 MB free in two zones).

Always wrap inference in its own cgroup:

```bash
systemd-run --user --scope -p MemoryHigh=6.5G -p MemoryMax=7G \
  ./llama-bench -m <model> -p 65536 -n 64 -ngl 99 -t 6 -r 1
```

Now a runaway allocation kills the benchmark instead of your session. **Never compile and run inference simultaneously on 8 GB** — `cc1plus` + `nvcc` + a loaded model exceeds 7.3 GiB and the kernel OOM killer doesn't check which process matters.

→ [`references/memory-budget.md`](references/memory-budget.md)

### 7. KV quantization decides your context ceiling — and it's a runtime property, not a model property

Same board, same model, same weights, two engines, two ceilings:

| Engine | Model | KV bytes/token | KV quant | Largest context actually run |
|---|---|---|---|---|
| llama.cpp | Bonsai 9B (hybrid) | 32 KB | ✅ `-ctk q8_0` | **65 536** @ 36.10 t/s |
| MLC-LLM | Qwen3-1.7B | 112 KB | ❌ none | **8 192** @ 14.96 t/s (32K OOM) |

A 9B reaches 64K while a 1.7B cannot reach 32K. Two independent multipliers: hybrid attention (÷4 on KV bytes) and whether the runtime can quantize K/V at all.

**Weight quantization does not shrink KV.** Qwen3-1.7B drops from ~3.4 GB fp16 to 961 MB with `q4f16_1` — a working 3.5x weight saving — while KV stays pinned at 112 KB/token. Conflating the two is the single most common reason people conclude "4-bit models must have cheap long context."

Practical order of operations:

1. Compute `context × bytes/token` (dividing by `full_attention_interval`). If it exceeds ~60% of usable memory, it's structural.
2. Prefer an engine with KV quantization: `-ctk q8_0 -ctv q8_0 -fa 1` in llama.cpp-family. `-fa 1` is **required** for quantized KV to be efficient.
3. If locked to an engine without it, cap context instead of OOMing — compile-time `sliding_window_size` fixes KV at a constant size and only costs distant memory.
4. Never change weight quant level hoping it helps context. It moves KV by zero.

→ [`references/kv-quantization.md`](references/kv-quantization.md)

## Reference index

| File | Covers |
|---|---|
| [`references/throughput-bottleneck.md`](references/throughput-bottleneck.md) | Bottleneck diagnosis, the parameter matrix that does nothing, quant format selection, vocab tax, speculative decoding, single-stream vs aggregate reporting |
| [`references/memory-budget.md`](references/memory-budget.md) | Real vs nominal memory, `cudaMalloc` ceiling, CMA vs CUDA, cgroup caging, OOM precursors, desktop stack cost |
| [`references/emc-bandwidth.md`](references/emc-bandwidth.md) | Unlocking EMC 2133→3199 MHz, why `apt upgrade` can't do it, three independent readouts, bandwidth measurement paths, timing bugs that corrupt results |
| [`references/context-scaling.md`](references/context-scaling.md) | Measured context sweep, hybrid vs dense attention KV math, prefill vs generation cost model, what actually breaks at 64K |
| [`references/kv-quantization.md`](references/kv-quantization.md) | KV bytes/token math, which engines can quantize KV, llama.cpp flags, MLC absence evidence, `sliding_window_size` fallback, NvMap/CMA misdiagnosis |

## Tested platform

| Item | Value |
|---|---|
| Device | Jetson Orin Nano 8GB (P3767, P3768 carrier) |
| L4T | R39.2.1 |
| Ubuntu | 24.04.4 |
| CUDA | 13.2 |
| Compute capability | sm_87 |
| CPU | 6 cores |
| Kernel | 6.8.12-tegra |

Conclusions 1–3 and 6 are runtime-behaviour facts and transfer to other Orin SKUs. Numbers for 4 and 5 are board-specific — re-measure on NX/AGX before quoting them.

## Related

- [`mlc-llm-on-jetson`](https://github.com/zazuss/mlc-llm-on-jetson) — building MLC-LLM from source on Orin (no official JetPack wheel exists)
- [`llama-prism-on-jetson`](https://github.com/zazuss/llama-prism-on-jetson) — CUDA build of PrismML's llama.cpp fork for ternary models, with prebuilt binary

## License

MIT for text and scripts. Measurement data is yours to reuse.
