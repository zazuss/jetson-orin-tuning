---
name: jetson-orin-tuning
description: Use when tuning LLM inference speed or memory on Jetson Orin, or when throughput/memory looks wrong. Classify the bottleneck before changing flags.
---

# Jetson Orin tuning

> **中文速览** —— 这份文档是 Hermes 技能的定义文件，agent 加载这个 skill 时会读它。内容是在 Jetson Orin 上调 LLM 推理的七条结论速查：先判瓶颈、量化格式比体积重要、词表税、长上下文几乎免费、内存账、cgroup 关笼子、KV 量化决定上下文天花板。正文保持英文，术语全是命令和参数名，直接照着执行即可。


Measured performance notes for LLM inference on Jetson Orin. The organizing principle: **most tunable parameters are noise, and the ones that matter are structural.** Classify the bottleneck first, then act.

## When to use

- "Is 35 tok/s on an 8 GB Orin correct?" (yes, for 2B Q4 — see bottleneck table)
- "Something must be misconfigured" — prove the bottleneck class before touching anything
- Model won't load → memory budget, not CMA
- Long context too slow / not fitting → hybrid attention KV math
- "My 4-bit model can't hold long context" → KV isn't affected by weight quantization; check whether the engine can quantize KV at all
- Bandwidth numbers look inconsistent → measurement path and timing bugs
- Choosing a quant format → simple block quant beats K-quant on weak GPUs

## Decision procedure

### Step 1 — classify the bottleneck (always first)

Sample `tegrastats` **during generation**, never at idle:

| GPU | EMC | Diagnosis | Action |
|---|---|---|---|
| ≥90% | <60% | Compute-bound | No flag helps. Smaller model / cheaper quant. |
| <70% | ≥85% | Bandwidth-bound | Fewer weight bytes only |
| Both low | Both low | Launch overhead | Batch concurrent requests |

High GPU utilization is not success — it means the ALU is busy, possibly on dequantization.

### Step 2 — quant format before model size

```
MB × tok/s ≈ 40 000        (simple block quant, this hardware class)
```

- Prefer **Q4_0 / Q5_0** over K-quants. Q3_K_S can be **2× slower** than a larger Q4_0.
- Check vocab share (`n_vocab × n_embd`): >20% of params means speed will look wrong for the file size.

### Step 3 — memory, using real numbers

- Managed total **7.31 GiB**, not 8 GB (nominal 8 / present 7.68).
- Single contiguous `cudaMalloc` ceiling **4 576 MB**; cumulative multi-block **5 376 MB** (7 × 768 MB). llama.cpp allocates per-tensor, so 5 376 MB is the number that decides whether a model loads. Both figures move with current free memory — treat as ~4.4–4.6 GB / ~5.2–5.4 GB, not constants.
- CMA limits (256 MB) apply to V4L2/camera only — **never** the cause of a model load failure.
- Cage every run: `systemd-run --user --scope -p MemoryHigh=6.5G -p MemoryMax=7G …`
- Size for the **prefill peak** (6.9 GB at 64K), not steady state (2.8 GB).

### Step 4 — long context

- Generation is nearly flat with context on hybrid-attention models: **36.10 t/s at 64K → 36.08 t/s at 256K** (−0.05%, not the −18 t/s an earlier bandwidth extrapolation predicted).
- KV bytes = formula **÷ full-attention interval**, not `block_count`. `full_attention_interval = 4` → ~38 KB/token measured (32 KB by layer count), not 128 KB.
- Prefill is what costs: 64K ÷ 300 t/s ≈ 218 s to first token; 128K ÷ 214 t/s ≈ **612 s**. Prefill peak — not steady-state size — is what OOMs.
- **Set `-np 1` before touching KV quantization.** `llama-server` defaults to 4 slots, each reserving the full `n_ctx`: at 256K that is 16.53 t/s and an OOM-kill on request #2, versus 36.08 t/s and stable with `-np 1`.
- Concurrency cannot add throughput on a bandwidth-bound model — single-stream decode already uses ~67 of ~98 GB/s. Parallel sub-agents against a local model each run slower than serialized ones.

### Step 5 — can the engine quantize KV?

This bounds the context ceiling before any tuning. Measured on one board:

| Engine | KV/token | KV quant | Max context actually run |
|---|---|---|---|
| llama.cpp | 32 KB (hybrid model) | ✅ `-ctk q4_0 -ctv q4_0 -fa on -np 1` | **262 144** @ 36.1 t/s (39 MB free) |
| llama.cpp | 32 KB (hybrid model) | ✅ `-ctk q8_0 -ctv q8_0 -fa on -np 1` | 131 072 @ 36.7 t/s (360 MB free) |
| MLC-LLM | 112 KB | ❌ none | 8 192 @ 15.0 t/s; 32K OOMs |

- **Weight quant ≠ KV quant.** A 3.5x weight saving (`q4f16_1`: 3.4 GB → 961 MB) leaves KV untouched at 112 KB/token.
- No KV quant and context doesn't fit → cap it, don't fight it: compile-time `sliding_window_size` fixes KV (costs distant memory, not stability).
- With quantized KV, `-fa on` is mandatory for speed.

## Reference files

| File | Read it when |
|---|---|
| [`references/throughput-bottleneck.md`](references/throughput-bottleneck.md) | Diagnosing speed; the parameter matrix that does nothing; quant selection; vocab tax; speculative decoding; reporting single-stream vs aggregate |
| [`references/memory-budget.md`](references/memory-budget.md) | OOM, load failures, `cudaMalloc` ceiling, CMA confusion, cgroup caging, prefill peaks |
| [`references/emc-bandwidth.md`](references/emc-bandwidth.md) | EMC stuck at 2133; verifying a bandwidth unlock; measuring bandwidth correctly; event-timing bugs |
| [`references/context-scaling.md`](references/context-scaling.md) | Context length vs speed and memory; hybrid vs dense KV math; measured 128K / 256K ceilings; why `-np` beats KV quant |
| [`references/kv-quantization.md`](references/kv-quantization.md) | Can this engine quantize KV? KV bytes/token math; llama.cpp flags; MLC's absence; `sliding_window_size` fallback; NvMap/CMA misdiagnosis |

## Rules that save the most time

1. **Never quote a datasheet number as measured.** Every figure in this skill came from this hardware.
2. **Idle readings are meaningless** for EMC frequency and GPU utilization.
3. **Never compile and run inference simultaneously** on 8 GB — `cc1plus` + `nvcc` + a loaded model exceeds 7.31 GiB.
4. **`jetson_clocks` does not survive reboot.** Re-apply before every benchmark or measure 8–9% low.
5. **`-r 3` is wasteful at 64K** — each rep is ~4 minutes. Use `-r 1` above 16K; variance is already ±0.1.
6. **A measurement whose average is far below its own peak is broken**, not a hardware fact. Check for cumulative-vs-per-round timer mixing.

## Platform these numbers came from

| Item | Value |
|---|---|
| Device | Jetson Orin Nano 8GB (P3767) |
| L4T | R39.2.1 |
| CUDA | 13.2, sm_87 |
| Kernel | 6.8.12-tegra |
| Managed memory | 7.31 GiB |

Bottleneck classification and quant/vocab conclusions transfer across Orin SKUs. Absolute throughput and memory numbers are board-specific — re-measure on NX/AGX.
