# Context scaling: measured, and why the naive KV formula lies

> **中文速览** —— 这份文档是上下文长度的实测扫描（128 到 262144），解释为什么混合注意力模型上长上下文几乎免费、解码速度为什么不随上下文下降、`-np` 单路设置为什么比 KV 量化更关键，以及 KV 成本该怎么算（必须按 full-attention 的层数算，不能按 block_count）。KV 尺寸按实测斜率算：`q4_0` 是每 token 10.7 KB、fp16 约 38 KB，且 `q4_0` 只压 3.56 倍不是 4 倍（4 bit + 每 32 值一个 fp16 scale = 4.5 bit/值），所以教科书那种「q4 = 四分之一」的估法在 64K 上会低约 35%。还记录了真正会坏在哪里：不是 KV，是 prefill 的瞬时计算缓冲。


## 1. The measurement

Ternary-Bonsai-2 9B (`PTQ1_0`, 1.85 GiB, 8.95B params, 1.75 bpw) on a 7.3 GiB Orin Nano, `llama-bench`:

```bash
systemd-run --user --scope -p MemoryHigh=6.5G -p MemoryMax=7G \
  ./llama-bench -m Ternary-Bonsai-2-9B-PTQ1_0.gguf \
                -p <CTX> -n 64 -ngl 99 -t 6 -r 1
```

| Context | Prefill (t/s) | Generation (t/s) | Process RSS |
|---|---|---|---|
| 128 | 360.75 ± 12.78 | 36.57 ± 0.26 | — |
| 2 048 | 376.12 ± 0.11 | 36.46 ± 0.25 | — |
| 8 192 | 367.23 ± 0.04 | 36.59 ± 0.24 | — |
| 16 384 | 354.64 | 36.24 | 2 751 MB |
| 32 768 | 334.09 | 36.20 | 2 716 MB |
| 65 536 | **300.56** | **36.10** | 2 775 MB |
| 131 072 | **214.24** | 35.97 | 128K prefill, `q4_0` KV |

**Generation falls 1.3% across a 512× context increase; prefill falls 29% by 128K.** Doubling context from 32K to 64K costs 59 MB of steady-state RSS. Decode is flat because the per-token weight stream dominates; prefill degrades because attention over the prompt becomes a growing share of a matmul-bound workload.

`-r 1` for the ≥16K runs: each 64K prefill takes ~218 s (65536 ÷ 300.56) and each 128K prefill ~612 s (131072 ÷ 214.24), so repetitions would burn tens of minutes for a number already stable to ±0.1.

## 2. Why this is not what the standard formula predicts

The textbook per-token KV cost:

```
bytes/token = 2 (K and V) × n_layers × n_kv_heads × key_length × 2 (fp16)
```

For this model — `block_count = 32`, `head_count_kv = 4`, `key_length = 256`:

```
2 × 32 × 4 × 256 × 2 = 131 072 bytes = 128 KB/token
```

At 64K tokens that predicts **8 GiB of KV cache**. The machine has 7.31 GiB total. The formula says the run is impossible; the run completed in 6.5 minutes at 36 t/s.

### The missing metadata field

```
qwen35.attention.full_attention_interval = 4
```

This is a **hybrid-attention** architecture. Only one layer in four maintains a full KV cache; the rest use a recurrent/linear-attention state whose size does not grow with context. Effective KV cost:

```
bytes/token ≈ 128 KB / 4 = 32 KB/token   # layer-count arithmetic
bytes/token ≈ 38 KB/token                # what RSS actually showed
```

The layer-count figure is a lower bound: measured fp16 KV is ~38 KB/token, about 19% above it (allocation granularity and per-layer padding). Size from the measured number.

> Whether you can shrink this further depends on the **runtime**, not the model: llama.cpp supports `-ctk q8_0` / `q4_0` (halves or quarters it again), MLC-LLM has no KV quantization at all. See [`kv-quantization.md`](kv-quantization.md) — that property, plus the slot count, is why a 9B model reaches 256K here while a 1.7B model cannot reach 32K.

| Context | Naive estimate | fp16 (derived) | **`q4_0` (measured slope)** |
|---|---|---|---|
| 8 192 | 1.0 GiB | ~300 MB | **~88 MB** |
| 32 768 | 4.0 GiB | ~1.2 GiB | **~350 MB** |
| 65 536 | 8.0 GiB | ~2.4 GiB | **~710 MB** |
| 131 072 | 16.0 GiB | ~4.9 GiB (does not fit) | **~1.37 GiB** |
| 262 144 | 32.0 GiB | ~9.7 GiB (impossible) | **~2.68 GiB** |

The `q4_0` column is no longer arithmetic — it is the measured slope. Holding weights constant and varying only `-c`, process RSS came out at 2 676 MB (8K), 3 300 MB (64K), 3 999 MB (128K), 5 393 MB (256K). Successive deltas: 624 / 699 / 1 394 MB, i.e. **10.6–10.9 KB per token, dead linear**. The fp16 column is that slope × 3.56.

**`q4_0` is 3.56x, not 4x.** The format stores 4 bits plus one fp16 scale per 32 values = 4.5 bits/value, so 16 ÷ 4.5 = 3.56. An earlier version of this table applied the textbook 4x and came out ~35% low at 64K (`~520 MB` when the real figure is ~710 MB). If you are budgeting, use 10.7 KB/token for `q4_0` and 38 KB/token for fp16.

**Rule: compute KV cost from the count of full-attention layers, never from `block_count`.** On dense-attention models the two are equal, which is why the naive formula usually works — and why it produces a confident 4× overestimate on hybrid architectures like Qwen3.5, Jamba, MiniMax, and Phi-3.5-MoE derivatives.

Read the field before sizing anything:

```bash
python3 - <<'PY'
from gguf import GGUFReader
r = GGUFReader("model.gguf")
for k in r.fields:
    if any(s in k.lower() for s in ("attention", "head_count", "block_count", "key_length", "value_length")):
        print(k, r.fields[k].contents())
PY
```

`llama-gguf` on some builds prints nothing; a 40-line metadata reader is more reliable than fighting the tool.

## 3. Prefill vs generation are different problems

| | Scales with | Bound by |
|---|---|---|
| **Prefill** | prompt length × n_layers | Compute (matrix-matrix) |
| **Generation** | 1 token, all weights | Memory bandwidth |

At 64K the prefill rate drops 16.7% (360 → 300 t/s) because attention over 64K tokens becomes a growing share of a matmul-dominated workload. Generation is unaffected because each step still reads the same number of weight bytes — the KV cache it additionally reads (~38 KB/token × 65 536 = 2.4 GiB) is spread over 36 tokens/s, i.e. ~1.4 GB/s, negligible against the 67 GB/s weight stream.

**Practical consequence:** time-to-first-token on a long prompt is the number users actually feel. 64K ÷ 300 t/s ≈ 218 s before the first token. No generation-side optimization helps that.

## 4. What actually breaks at 64K

Not the KV cache — the **prefill compute buffer**:

| Phase | RSS |
|---|---|
| After load, idle | 2 775 MB |
| During 64K prefill | **6 914 MB** (83 MB free) |
| Generating | 2 775 MB |

The transient peak is proportional to prompt length. Lower `n_batch` / `n_ubatch` to shrink it, at the cost of prefill throughput. Always measure the peak, not the steady state, when deciding whether a context length fits.

## 5. Is 128K or 256K possible? — measured, yes to both

The model declares `context_length = 262144`. That is the training limit, and it turns out to also be reachable on an 8 GB board — provided you quantize the KV cache and **run a single slot**.

All rows below are `llama-server`, `PTQ1_0` weights, single slot (`-np 1`), flash attention on, measured 2026-10-25:

| Context | KV type | Steady RSS | Free RAM after | Decode | Verdict |
|---|---|---|---|---|---|
| 65 536 | fp16 | 2 775 MB | ~4.6 GB | 36.10 t/s | comfortable |
| 131 072 | `q4_0` | 4.46 GB | **1.4 GB** | 35.37 / 36.59 t/s | **recommended** |
| 131 072 | `q8_0` | 4.76 GB | 360 MB | 35.51 / 36.67 t/s | tight |
| 262 144 | `q4_0` | 5.70 GB | **39 MB** | 36.08 t/s | works, no headroom |

Three findings that invert what the naive arithmetic predicts:

**Decode speed does not care about context length.** 64K → 256K stays flat at ~36 t/s. The earlier prediction in this document ("128K would drop to about 18 t/s") was wrong. Decode is bound by the weight stream — 1.86 GB read per token — and the 24 non-full-attention layers keep a state that does not grow with context, so a longer KV does not add to that stream. Only *memory* grows.

**`-np` matters more than KV quantization.** `llama-server` defaults to 4 slots, and each slot reserves the full `n_ctx`. At 256K that means four 256K KV allocations:

| Slots | Context | Decode | Outcome |
|---|---|---|---|
| `-np 1` | 262 144 | **36.08 t/s** | works |
| `-np 4` (default) | 262 144 | **16.53 t/s** | first request succeeds, **second request OOM-kills the process** |

The 2.2x speed loss is not the extra memory — it is that a four-slot unified KV pool makes attention slower. Anyone who leaves the default `-np 4` and concludes "long context is slow on Jetson" has measured the wrong variable.

**Concurrency buys nothing on a bandwidth-bound model.** Single-request decode already consumes ~67 GB/s of the ~98 GB/s the board can move. Four slots do not quadruple throughput; they split the same ~36 t/s into four ~9 t/s streams. Parallelism here only helps if your requests are waiting on something other than the GPU — which is why a local 9B is a poor fit for spawning parallel sub-agents: they contend for the same bandwidth, and each runs slower than if they were serialized.

### What still binds: prefill, not decode

The one thing that gets worse is the transient prefill compute buffer, which scales with prompt length. At 64K it already peaks at 6 914 MB of 7 485 MB. A genuine 128K or 256K *fill* is therefore a memory event even when steady-state loading is fine — see §4. Load success is not the same as "I can push 256K tokens through it".

For an agent that actually fills the window, 128K with `q8_0` or `q4_0` is the sane ceiling on this board; 256K is a demo, not a working configuration.

**Do not quote the model's declared `context_length` as achievable without stating the KV type, the slot count, and the headroom left.** All three are load-bearing. The same 262 144 is either "works at 36 t/s" or "OOM-kills your service on request #2" depending entirely on `-np`.
