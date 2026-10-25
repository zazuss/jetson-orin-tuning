# Unlocking EMC 2133 → 3199 MHz, and measuring bandwidth honestly

## 1. The problem

A Jetson Orin Nano **non-Super** BCT hard-locks EMC at 2133 MHz (measured ceiling **68.3 GB/s**). Flashing the BCT from `jetson-orin-nano-devkit-super.conf` (JetPack 7 / L4T R39.2.1) raises the ceiling to 3199 MHz (**102.4 GB/s**).

**No software path unlocks it.** Switching `nvpmodel` to 25W, writing sysfs, re-applying the DTB — all no-ops.

### Root cause (confirmed by measurement)

The EMC frequency table lives in the **QSPI BCT / BPMP device tree**, not in the rootfs.

`/etc/nvpmodel.conf` can only select from profiles the BPMP actually exposes. An old system (apt-upgraded to r36.4.7) has `25W → 3199` written in its nvpmodel table, **but the BPMP has no 3199 entry** — so `nvpmodel_clk_cap/emc` gets clamped to 2133. Changing power mode does nothing; raising voltages does nothing.

**Symptom: all four power profiles show configured values that have no effect, EMC pinned at 2133.**

**Why apt can't fix it:** the EMC table is not part of the rootfs. `apt upgrade` never writes QSPI.

Verification after a full flash:

```bash
grep TNSPEC /etc/nv_boot_control.conf
# expect: ...-jetson-orin-nano-devkit-super-...

tr -d '\0' < /proc/device-tree/model
# expect: ... Developer Kit Super

tr '\0' '\n' < /proc/device-tree/compatible | head
# expect: nvidia,p3768-0000+p3767-0005-super
```

### Before / after, same SM kernel, only EMC changed

| | System | EMC cap | Pin theoretical | SM kernel measured |
|---|---|---|---|---|
| Before | L4T r36.4.7 / kernel 5.15 | 2133 MHz | 68.3 GB/s | 59.5 GB/s (87%) |
| After | L4T r39.2.1 / kernel 6.8.12 | 3199 MHz | 102.4 GB/s | 90.0 GB/s (88%) |
| After | same, `cudaMemcpy` | 3199 MHz | 102.4 GB/s | **98.0 GB/s (96%)** |

**Efficiency is unchanged (87% → 88%): nothing changed except EMC frequency. Bandwidth +51% tracks EMC +50%.** That proportionality is the proof the unlock worked and that nothing else regressed.

## 2. Three independent readouts

Check all three; they must agree.

```bash
cat /sys/kernel/nvpmodel_clk_cap/emc            # 2133 = locked, 3199 or -1 = unlocked
cat /sys/class/devfreq/bwmgr/max_freq           # ceiling
sudo jetson_clocks --show | grep -i emc         # MaxFreq
cat /sys/class/devfreq/bwmgr/available_frequencies   # must contain 3199
```

Note: `/sys/kernel/nvpmodel_clk_cap/emc` exists on JetPack 7 and is world-readable (`-rw-rw-r--`). An earlier "the file doesn't exist" conclusion was wrong — `sudo -n cat` had failed silently on a machine without passwordless sudo. **When a sysfs read fails, suspect permissions before suspecting the path.**

## 3. You must apply load to judge

EMC is driven by the `bpmp-bwmgr` governor on demand. **At idle it always sits at 2133**, unlocked or not.

- CPU memory load does **not** drive it up. An OpenMP read/reduce loop sampled for 60 s stayed at 2133 throughout.
- **GPU streaming kernels do.** A read+write kernel over ≥1 GiB immediately pins it at 3199.

```bash
# GPU frequency node is by address, not by name
cat /sys/class/devfreq/17000000.gpu/cur_freq
```

`nvidia-smi` does not exist on Jetson; use devfreq.

## 4. Bandwidth benchmarks on sm_87

```
/usr/local/cuda/bin/nvcc -O3 -arch=sm_87 -o gpubw gpubw.cu
```

Kernel design that produces trustworthy numbers:

- `float4` streaming read + write
- working set ≥ 1 GiB per buffer (far above L2, so nothing is cached)
- `blocks = SM count × 8`
- warm up one round before timing
- **throughput = 2 × buffer size / round time** (read *and* write both consume bandwidth)

| Method | Path | Peak | % of 102 GB/s |
|---|---|---|---|
| SM streaming kernel (`float4` read+write) | SM issues loads | 90.0 GB/s | 88% |
| `cudaMemcpy`, single stream | copy engine / DMA | 96.5 GB/s | 95% |
| `cudaMemcpy`, 8 concurrent streams | copy engine / DMA | **97.9 GB/s** | **96%** |

**All three ran at EMC 3199 — the difference is only *who issues the requests*.** Do not mistake the SM kernel's 88% for the hardware ceiling. The SM path traverses L2, MMU and page tables, and write-allocate consumes a second copy of bandwidth; the copy engine path is more direct.

**Rule: quote "this machine's memory bandwidth" from concurrent `cudaMemcpy`; quote "what a GPU kernel actually obtains" from the SM kernel.** Never mix paths when comparing before/after.

## 5. Timing bugs that silently corrupt every number

Two real cases, both producing plausible-looking garbage.

### Bug 1 — cumulative timer used as per-iteration

```c
// WRONG: ms is cumulative from t0, but used as a single-round cost
cudaEventRecord(t0);
while (1) {
    kernel<<<...>>>();
    cudaDeviceSynchronize();
    cudaEventRecord(t1);
    cudaEventElapsedTime(&ms, t0, t1);       // cumulative!
    gbs = 2.0*bytes/(ms/1000.0)/1e9;         // shrinks every iteration
    total += gbs;
}
```

### Bug 2 — adjacent events measure ~0

Two events recorded back to back with no work between them yield a duration of essentially zero, so `if (ms >= secs) break;` **never fires** → infinite loop.

### Correct pattern

```c
cudaEventRecord(tstart);
while (1) {
    cudaEventRecord(t0);
    kernel<<<...>>>();
    cudaDeviceSynchronize();
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);        // per-round
    gbs = 2.0*bytes/(ms/1000.0)/1e9;
    cudaEventElapsedTime(&tot, tstart, t1);   // independent total
    if (tot/1000.0 >= secs) break;
}
```

**Symptom recognition:** a report saying "average 0.5 GB/s, peak 61.4 GB/s" means cumulative and per-round values were mixed. An average far below the peak is internally inconsistent — the measurement is wrong, not the hardware.

Always warm up: the first round includes page-fault and cold-start cost.

## 6. Other pitfalls

- `pkill -f <path>` matches the invoking shell's own command line and kills it. Use `pkill -x <name>`.
- Memory pressure programs need `gcc -O2 -fopenmp`; `-O2` alone gives `undefined reference to 'omp_get_max_threads'`.
- The `gpu` devfreq node is `/sys/class/devfreq/17000000.gpu/` — it is not discoverable by name.
- `jetson_clocks` does not persist across reboot. Re-apply before every benchmark or you measure 8–9% low.
