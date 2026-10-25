# Single-stream decode: classify the bottleneck first

> **中文速览** —— 这份文档讲怎么判断你卡在算力还是带宽上（tegrastats 看 GR3D 和 EMC 的占用组合），以及一堆大家爱调但其实没用的参数（线程数、batch、GPU 频率、投机解码深度）。还有量化格式怎么选（Q4_0 为什么比体积更小的 Q3_K_S 快）、词表税怎么算、单流和聚合吞吐的口径差别。


When asking "is this as fast as it should be?" or "something must be misconfigured" — **prove which class of bottleneck you're in before changing anything.** On an Orin Nano Super 8GB, a 2B Q4 model at 32–35 tok/s single-stream is correct, and there is published reference data to confirm it.

## 1. Classify: sample tegrastats during generation

Idle readings tell you nothing. Sample while tokens are actually being produced.

```bash
# minimal probe: run generation, sample tegrastats concurrently
tegrastats --interval 500 &
./llama-cli -m model.gguf -ngl 99 -fa on -n 400 -p "..." -st --no-display-prompt &
wait
```

| Observation | Diagnosis | Action |
|---|---|---|
| GPU ≥90%, EMC <60% | **Compute-bound** (dequant + ALU) | Tuning is useless. Cheaper quant format or smaller model. |
| GPU <70%, EMC ≥85% | Bandwidth-bound | Only fewer weight bytes helps |
| Both low | Serialization / kernel launch overhead | Batch concurrent requests instead of optimizing single-stream |

Measured: single-stream 2B Q4 gave **GR3D median 94% (peak 99%), EMC median 43%, CPU 5%, board 6.9 W** → compute-bound, first row.

**Do not read "GPU busy" as "running well."** High utilization with low bandwidth means the compute units are saturated with work that produces no output — usually dequantization.

## 2. Parameters that provably do nothing

Same model, same prompt, `-temp 0`, all landed in 31.3–32.2 tok/s:

- `-fa 0` (31.3)
- `GGML_CUDA_FORCE_MMQ=1` (31.9)
- `-ctk q8_0 -ctv q8_0` (31.7)
- `-ub 1` (32.2), `-ub 256` (32.0)
- `-t 1` (32.0)
- `--mlock`

Spread is under 3%. On a compute-bound workload these are noise. Run the classification table in §1 before touching flags.

## 3. Quantization format beats file size

`llama-cli -ngl 99 -fa on --temp 0 -n 150`, same prose prompt, Qwen3 family:

| Model / quant | Size | Single-stream tok/s |
|---|---|---|
| Qwen3-0.6B Q4_K_M | 378 MB | **64.4** |
| Qwen3-1.7B Q3_K_S | 827 MB | **22.4** |
| Qwen3-1.7B Q4_0 | 1008 MB | 40.8 |
| Qwen3.5-2B Q4_0 | 1159 MB | 34.7 |
| Qwen3.5-2B Q4_K_M | 1222 MB | 32.1 |

**Rule: on a weak GPU, prefer simple block quantization (Q4_0 / Q5_0) over K-quants.**

Mechanism: Q4_0 stores 32 weights per block with a single fp16 scale. Dequantization is one multiply-add that folds into the trailing integer accumulation. K-quant uses a 256-weight super-block with a 6-bit scale *and* a 6-bit min, subdivided into 8 sub-blocks, each requiring its own `min + scale × code` step. That cannot be folded, so the instruction count per weight multiplies. Result: Q4_0 is 8% faster than Q4_K_M, and Q3_K_S is **twice as slow** as a larger Q4_0.

**"Smaller is faster" is false for K-quants** — the single most common wrong intuition here.

## 4. Large vocabularies are a hidden per-token tax

`lm_head params = n_vocab × n_embd`. Every generated token streams that entire matrix to produce logits.

| Model | Params in lm_head | Share of total |
|---|---|---|
| Qwen3.5-2B | 248320 × 2048 = 508 M | **27%** |
| Llama-3.2-3B | 128256 × 3072 ≈ 392 M | 12% |
| SmolLM2-1.7B | 49152 × 2048 ≈ 101 M | 6% |

Consequence: a 1.19 GB Qwen3.5-2B (34.7 tok/s) is slower than a 1.9 GB Llama-3.2-3B (43.07 tok/s official). **If vocab share exceeds ~20%, expect anomalous speed for the file size.**

Where to read it: `n_vocab` and `n_embd` appear in `llama-cli`'s `print_info:` log, or parse the GGUF metadata directly.

## 5. Calibration constant: MB·tok/s

On this class of hardware, for simple block quantization:

```
MB × tok/s ≈ 40 000
```

50 tok/s therefore needs weights ≲ 800 MB *and* a small vocabulary. A 1B-class model at Q4 (0.7–0.8 GB) clears the bar; 1.7B at Q4 (1.0 GB) reaches only 40.8.

Use this to size a recommendation instead of guessing at model tiers.

## 6. Speculative decoding loses whenever compute is saturated

Prerequisites: **bandwidth-bound and compute-idle**. This machine is the opposite, so:

- **Draft and target must share a vocabulary.** Qwen3.5 (248320) vs Qwen3 (151936) simply cannot pair.
- This build ships with it off and fails silently: the log prints `"speculative":false` unless you pass an explicit `--spec-type draft-simple`.
- Even with matching vocabularies it loses: Qwen3-1.7B + Qwen3-0.6B draft went 40.7 → **24.2** (n-max 4) / 17.2 (n-max 8). The draft model steals compute from the target.
- ngram variants (`--spec-type ngram-simple`, `ngram-map-k4v`): 32.1 → 32.4. Zero.

**Conclusion: do not expect speculative decoding to speed up single-stream decoding on a small GPU.**

## 7. Report single-stream and aggregate separately

`llama-server` multi-slot measurement: 1 slot 32.1 / 2 slots 67.8 / 4 slots 113.3 / **8 slots 164.4** aggregate (GPU 89% avg / 99% peak, EMC 50%, 18.8 W).

Users asking "how fast is it for me" mean **single-stream**. Never present aggregate throughput as a single-request figure. Conversely, aggregate is the honest measure of the machine's real capability. Report both, labeled.

## 8. Reference numbers for "is someone else faster?"

NVIDIA Jetson AI Lab archived benchmarks, Orin Nano Super, **MLC engine with INT4** — different engine, different quantization, so the comparison is only valid at the order-of-magnitude level:

```
Llama 3.1 8B   19.14    Qwen2.5 7B   21.75    Llama 3.2 3B  43.07
Phi 3.5 3B     38.10    Gemma 2 2B   34.97    SmolLM2       64.50
```

**Gemma 2 2B at 34.97 official vs 34.7 measured here on a completely different engine is the strongest possible evidence that nothing is misconfigured.**

Fetching that page: it 301s to `/archive/benchmarks.html`, so `curl -L` is required, and the numbers only survive if you parse `<tr>` / `<t[dh]>` rather than flattening to text.

## 9. Environment pitfalls

- **Loading two models OOMs.** Each needs its own cgroup; see `memory-budget.md`.
- **`jetson_clocks` does not survive reboot.** Before any benchmark: `sudo nvpmodel -m 0 && sudo jetson_clocks`. Otherwise GPU idles at 918 MHz and EMC at 2133, costing 8–9%.
- **Generation time includes thinking tokens.** Qwen3.x emits reasoning output by default, so perceived latency is worse than tok/s suggests. `/no_think` may help — **not verified on this platform, do not report it as measured.**

## 10. Answering "something must be misconfigured"

Evidence first, conclusion second, choice left to the user:

1. All layers offloaded (`offloaded 25/25 layers`), CUDA backend active.
2. GPU ≥90% with EMC <60% during generation → compute-bound, not misconfigured.
3. The parameter matrix in §2 was tried.
4. Official reference numbers at the same tier agree.
5. **Then** offer paths: smaller model / simpler quant / accept the current speed. The speed-versus-quality tradeoff belongs to the user.
