# Memory budget on an "8 GB" Orin, and how to not get OOM-killed

> **中文速览** —— 这份文档算内存账：标称 8 GB 实际只剩 7.31 GiB 可用，单次 cudaMalloc 能拿多少、分块累积能拿多少、CMA 那 256 MB 为什么跟你无关、cgroup 怎么关笼子、OOM 之前有哪些前兆、桌面环境吃掉多少。判断某个模型能不能装下之前先看这个。


## 1. Nominal is not available

| Figure | Value | Source |
|---|---|---|
| Nominal | 8 GB | product spec |
| **Present** | 8 050 304 kB = **7.68 GiB** | `/proc/meminfo` `MemTotal` |
| **Managed** | 7 665 092 kB = **7.31 GiB** | `cat /sys/devices/system/memory/block_size_bytes`, or `nvidia-smi`-less route via `/proc/iomem` |

The ~890 MB gap is device-tree carveout plus kernel:

```
vpr-carveout         512 MB    (video protection region, camera pipeline)
camdbg_carveout      ~64 MB
fsi-carveout         ~8 MB
rce-reservation      ~1 MB
ramoops              ~1 MB
framebuffer          ~64 MB
kernel + initrd      ~370 MB
```

**Model sizing must use 7.31 GiB, not 8 GB.** A 6 GB model leaves 1.3 GiB for KV cache, CUDA context (~250 MB) and the runtime — which is not enough for meaningful context on most architectures.

## 2. `cudaMalloc` ceiling: not fixed — it tracks available memory

There is no constant "CUDA allocation limit" on this board. The probe (`cudaMalloc` + `cudaMemcpy` + free, bisected, run under `systemd-run --user --scope -p MemoryMax=infinity` so it inherits no cgroup limit) returns different numbers depending on what else is running:

| Environment | `MemAvailable` | Single contiguous | Cumulative (768 MB chunks) |
|---|---|---|---|
| Bonsai + searxng stopped | 4 739 MB | **4 608 MB** | **5 376 MB** (7 chunks) |
| Bonsai stopped only | 4 593 MB | **4 384 MB** | **4 608 MB** (6 chunks) |
| Inside an agent's cgroup (3.65 GiB cap) | — | 4 320 MB | 5 250 MB |

`cudaMemcpy` device↔host throughput is a stable **98 GB/s**.

Two rules follow:

- **Single contiguous ≈ `MemAvailable` − 200 MB**, i.e. ~4.3–4.5 GB in practice. The deficit is driver/fragmentation overhead.
- **Cumulative is quantised by the chunk size.** With 768 MB chunks the result jumps a whole chunk at a time (4 608 → 5 376), so the number you get is a step function of ambient memory, not a smooth curve. Peak observed: **5 376 MB**.

Three corrections that matter for sizing:

- **Chunking does not unlock "all of it".** It raised the ceiling from 4 608 to 5 376 MB (~17%) in the best case, not to the 7.31 GiB of managed memory. The remaining ~1.9 GiB is unavailable to CUDA however you slice the request.
- **llama.cpp allocates per-tensor, not in one block**, so the cumulative figure is the one that decides whether a GGUF loads.
- **27B does not fit, but by less than folklore says.** `PTQ1_0` weights are 5 671 MB against a best-case 5 376 MB cumulative ceiling — short by **295 MB** in the best case and ~1 063 MB on a day when you only reach 4 608 MB. Either way it is short before any KV cache, so the conclusion is unchanged; the size of the gap is not a constant and should not be quoted as one.
- **Run the probe outside your agent's cgroup.** Inside a Hermes background wrapper it reported 4 320 / 5 250 MB and would have been taken as the answer. That wrapper's `memory.max` was 3.65 GiB.

## 3. CMA is never your problem

There is a persistent myth that Jetson caps allocations around 1 GB. That limit belongs to the **CMA / NvMap** path:

```bash
grep Cma /proc/meminfo
# CmaTotal:     262144 kB   (256 MB)
# CmaFree:        3312 kB   (3.3 MB, nearly exhausted at idle)
```

CMA is used by V4L2, the ISP, display and camera buffers. Under pressure you see:

```
NvMapMemAllocInternalTagged: NvMapMemAlloc: failed: error 12
```

**CUDA and GGML never touch CMA.** So:

- Model fails to load → **not** a CMA problem. Check the 5 376 MB cumulative CUDA ceiling and the managed total.
- Camera/GStreamer fails to allocate → that *is* CMA.

Do not "fix" a model load failure by raising `cma=` on the kernel command line. Wrong layer entirely.

## 4. The desktop stack costs 256 MB

Measured PSS of the full GDM3 + GNOME stack on a headless-ish Orin: **256 MB**.

That is worth reclaiming before a long-context run:

```bash
sudo systemctl stop gdm3     # frees ~256 MB, kills the local display
```

For an SSH-only workflow this is free memory. It is not enough to change what fits on its own (256 MB out of 7.31 GiB), but combined with a 6.9 GB prefill peak it is the difference between completing and being killed.

## 5. Cage every inference process

A 64K prefill on a 9B ternary model peaked at **6 914 / 7 485 MB** with 83 MB free, then settled back to 2 775 MB. During that peak, an unrelated daemon was OOM-killed (`all_unreclaimable? yes`, 33 MB free in two zones). The kernel does not distinguish your inference job from your session.

```bash
systemd-run --user --scope -p MemoryHigh=6.5G -p MemoryMax=7G \
  ./llama-bench -m model.gguf -p 65536 -n 64 -ngl 99 -t 6 -r 1
```

- `MemoryHigh` throttles and forces reclaim — the process slows instead of the system thrashing.
- `MemoryMax` is the hard wall — the runaway job dies alone.

Both are per-cgroup, so `systemd-run --scope` on the inference command is sufficient. Verify after the fact:

```bash
systemctl --user show <scope>.scope -p MemoryPeak
journalctl --user -u <scope>.scope | grep -i oom
```

## 6. Never compile and infer at the same time

`cc1plus` on a large C++ TU plus `nvcc` easily reach 3–4 GB. Add a loaded 2.5 GB model and the total exceeds 7.31 GiB. The OOM killer picks by `oom_score_adj`, which for a shell-derived process is typically not in your favor.

Practical rule: **`-j4` for compilation on 8 GB, and nothing else running.** 6 cores × full parallel C++ compilation is what pushes it over.

## 7. OOM precursors worth watching

```bash
# before a big run
grep -E "MemAvailable|CmaFree" /proc/meminfo
tegrastats --interval 1000 | grep -E "RAM|SWAP"

# after a suspected OOM
dmesg -T | grep -iE "out of memory|oom-kill|all_unreclaimable"
```

`all_unreclaimable? yes` in `dmesg` means reclaim already failed — you are past the point where tuning helps; the allocation must be reduced.

## 8. Prefill peak vs steady state

The single most misleading number when sizing a run is the peak during prefill:

| Phase | RSS |
|---|---|
| Idle after load | 2 775 MB |
| During 64K prefill | 6 914 MB |
| After prefill, generating | 2 775 MB |

The peak comes from the prefill compute buffer, which is proportional to prompt length. It is transient but real — the cgroup will see it, and it is what triggers OOM.

**Size for the peak, not the steady state.** `n_batch` / `n_ubatch` reduce the prefill buffer at the cost of prefill throughput.
