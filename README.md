# Jetson Orin 调优实测笔记

在 Jetson Orin（Nano 8GB / Super / NX / AGX）上做 LLM 推理的**可复现实测**笔记。这里每一个数字都来自 7.3 GiB Orin Nano 上的真实测量，没有一条是从数据手册抄来的，也没有一条是用未经验证的公式推出来的。

> **English**: Measured, reproducible performance notes for LLM inference on Jetson Orin. Every number came from a real measurement on a 7.3 GiB Orin Nano. The point is to stop you from tuning the wrong knob: on an 8 GB Orin, thread counts, batch size, GPU clocks and speculative-decoding depth are **noise**; the things that move the needle are structural — quantization layout, vocabulary size, memory bandwidth, which bound you're actually on, and whether the runtime can quantize its KV cache.

这个仓库的目的，是不让你去拧错的旋钮。在 8 GB 的 Orin 上，大家爱调的那些东西——线程数、batch size、GPU 频率、投机解码深度——基本全是**噪声**。真正影响结果的是结构性的东西：量化格式的排布方式、词表大小、内存带宽、你到底卡在算力还是带宽上，以及运行时能不能量化 KV cache。

## 七条结论，每条都有实测证据

### 1. 动手之前，先搞清楚你卡在哪个瓶颈

`tegrastats` 要在**生成过程中**采样，不能在空闲时采样：

| GPU 占用 | EMC 占用 | 诊断 | 什么真的有用 |
|---|---|---|---|
| ≥90% | <60% | **算力受限**（反量化/ALU） | 没有可调项。只能换更小的模型或更便宜（更快）的量化格式 |
| <70% | ≥85% | **带宽受限** | 只有减少权重的字节数有用 |
| 两者都低 | 两者都低 | 串行化 / kernel 启动开销 | 要靠并发请求批处理，单流调参没用 |

在 2B Q4 模型上实测：**GR3D 中位 94%，EMC 中位 43%，CPU 5%，整板 6.9 W** → 算力受限。在这个状态下，没有任何参数有意义。**GPU 占用高本身不是好消息**——GPU 忙而带宽闲，说明 ALU 正在烧周期做一些产不出 token 的工作。

→ [`references/throughput-bottleneck.md`](references/throughput-bottleneck.md)

### 2. 量化**格式**比体积更重要，而且“越小越快”是错的

同一个模型、同一个 prompt，参数 `-ngl 99 -fa on --temp 0 -n 150`：

| 模型 / 量化 | 体积 | 单流 tok/s |
|---|---|---|
| Qwen3-0.6B Q4_K_M | 378 MB | **64.4** |
| Qwen3-1.7B Q3_K_S | 827 MB | **22.4** ← 体积更小，速度只有一半 |
| Qwen3-1.7B Q4_0 | 1008 MB | 40.8 |
| Qwen3.5-2B Q4_0 | 1159 MB | 34.7 |
| Qwen3.5-2B Q4_K_M | 1222 MB | 32.1 |

**在弱 GPU 上，优先用简单的块量化（Q4_0/Q5_0），避开 K-quant。** Q4_0 每 32 个权重打包成一块，只带一个 fp16 的 scale——反量化就是一次乘加，能直接折进后面的整数累加里。K-quant 用 256 个权重的 super-block，同时带 6-bit 的 scale **和** 6-bit 的 min，还要再拆成 8 个子块，每个子块都要单独解出自己的 min+scale。每个权重摊到的指令数直接翻倍。这就是为什么 Q3_K_S 比一个**体积更大**的 Q4_0 慢 45%。

### 3. 大词表是一项隐藏的逐 token 税

`lm_head 参数量 = n_vocab × n_embd`，而每生成一个 token 都要把这整块权重流式读一遍来算 logits：

| 模型 | n_vocab × n_embd | 占参数总量 |
|---|---|---|
| Qwen3.5-2B | 248320 × 2048 = 508 M | **27%** |
| Llama-3.2-3B | 128256 × 3072 ≈ 392 M | 12% |
| SmolLM2-1.7B | 49152 × 2048 ≈ 101 M | 6% |

这就是为什么 1.19 GB 的 Qwen3.5-2B（34.7 tok/s）反而比 1.9 GB 的 Llama-3.2-3B（官方 43.07 tok/s）慢。如果词表占比超过约 20%，就要预期速度与文件体积严重不匹配。

### 4. 在混合注意力模型上，长上下文几乎免费

Ternary-Bonsai-2 9B（`PTQ1_0`），7.3 GiB Orin，`llama-bench`：

| 上下文 | prefill t/s | 生成 t/s |
|---|---|---|
| 128 | 360.75 | 36.57 |
| 2 048 | 376.12 | 36.46 |
| 8 192 | 367.23 | 36.59 |
| 16 384 | 354.64 | 36.24 |
| 32 768 | 334.09 | 36.20 |
| 65 536 | **300.56** | **36.10** |
| 131 072（+ `q4_0` KV） | **214.24** | **35.97** |

上下文翻 1024 倍，生成速度只掉 1.6%；但 128K 时 prefill 掉了 29%。稳态 RSS 从 32K 的 2 716 MB 到 64K 的 2 775 MB——上下文翻倍，只多 59 MB。**解码不受长度影响（受权重量限制），prefill 受影响（attention 占 matmul 的比例变大）。**

再往上测到 128K / 256K 也一样平：`llama-server` 单路下 128K 是 36.59 t/s、256K 是 36.08 t/s，与 64K 的 36.10 t/s 差 0.05%。**长度只吃内存，不吃速度**——代价全在 KV 常驻上（128K + `q4_0` 是 4.46 GB 常驻、256K + `q4_0` 是 5.70 GB 常驻，只剩 39 MB）。

真正的门槛是 prefill 时长：128K 一次塞满要 **612 秒（约 10 分钟）**才出第一个 token，256K 按同样的衰减要接近半小时。这是用户实际感受到的延迟，生成侧的优化一点帮不上。

按教科书的朴素 KV 公式，会假设每一层都缓存 K 和 V：

```
2 × 32 层 × 4 kv_heads × 256 key_len × 2 字节 = 128 KB/token  →  64K 时要 8 GiB
```

这算出来是 8 GiB，物理上不可能。实际模型用的是**混合注意力**（`full_attention_interval = 4`），只有约 ¼ 的层保留完整 KV cache。**KV 成本必须按 full-attention 的层数算，绝不能按 `block_count` 算。** 在这个架构上，按稠密注意力去算会高估约 4 倍。

→ [`references/context-scaling.md`](references/context-scaling.md)

### 5. 你能用的内存比机器标的少

在一台标称“8 GB”的 Orin Nano 上：

| 口径 | 数值 |
|---|---|
| 标称 | 8 GB |
| present | 8 050 304 kB = **7.68 GiB** |
| managed（CUDA 可用） | 7 665 092 kB = **7.31 GiB** |

消失的约 890 MB 里，512 MB 是设备树 carveout（`vpr-carveout`、`camdbg_carveout`、`fsi-carveout`、`rce-reservation`、`ramoops`、`framebuffer`），再加约 370 MB 的内核占用。

有两个数字经常被误诊：

- **单次 `cudaMalloc` 的上限约 4.4–4.6 GB**，实测值在 4320–4576 MB 之间浮动（跟随当时 free 内存变化），不是固定数。这个限制来自 UMA 分配器，不是碎片问题。
- **分块累积分配能拿到 5376 MB**（7 块 × 768 MB，第 8 块失败）。llama.cpp 是逐 tensor 分块分配的，所以真正卡它的是 5376 MB 而不是单次上限——这一点在判断“某个模型到底能不能装下”时很关键，用单次上限去估会低估约 800 MB。
- 大家记忆里那个“Jetson 有 1 GB 限制”属于 **CMA/NvMap** 那条路——`CmaTotal` 256 MB、`CmaFree` 3.3 MB——那是 V4L2/ISP/显示用的。GGML 和 CUDA 从不碰 CMA，所以 CMA 永远不是你模型加载失败的原因。

### 6. 把进程关进笼子，别指望运气

一次 64K prefill 峰值冲到 **6 914 / 7 485 MB**，只剩 83 MB 空闲，之后才回落到 2.8 GB。就在那个峰值上，OOM 带走了一个不相关的进程（一个网关守护进程在 11:10:55 被杀；`dmesg` 里此前一直是 `all_unreclaimable? yes`，两个 zone 各剩 33 MB）。

永远把推理包在它自己的 cgroup 里：

```bash
systemd-run --user --scope -p MemoryHigh=6.5G -p MemoryMax=7G \
  ./llama-bench -m <model> -p 65536 -n 64 -ngl 99 -t 6 -r 1
```

这样失控的内存分配会杀掉那个 benchmark，而不是你的会话。**在 8 GB 上永远不要同时编译和跑推理**——`cc1plus` + `nvcc` + 一个已加载的模型会超过 7.3 GiB，而内核的 OOM killer 并不检查哪个进程更重要。

→ [`references/memory-budget.md`](references/memory-budget.md)

### 7. KV 量化决定你的上下文天花板，而且它是运行时属性，不是模型属性

同一块板、同一个模型、同一份权重，两个引擎，两个天花板：

| 引擎 | 模型 | KV 字节/token | KV 量化 | 实际跑到的最大上下文 |
|---|---|---|---|---|
| llama.cpp | Bonsai 9B（混合注意力） | 32 KB | ✅ `-ctk q4_0 -ctv q4_0 -fa on -np 1` | **262 144** @ 36.08 t/s（剩 39 MB） |
| llama.cpp | Bonsai 9B（混合注意力） | 32 KB | ✅ `-ctk q8_0 -ctv q8_0 -fa on -np 1` | 131 072 @ 36.67 t/s（剩 360 MB） |
| MLC-LLM | Qwen3-1.7B | 112 KB | ❌ 无 | **8 192** @ 14.96 t/s（32K 直接 OOM） |

一个 9B 能跑到 256K，一个 1.7B 连 32K 都到不了。三个互相独立的倍数：混合注意力（KV 字节 ÷4）、运行时到底能不能量化 K/V、以及**槽位数**（`-np 4` 在 256K 下直接掉到 16.53 t/s 并 OOM）。

**权重量化不会缩小 KV。** Qwen3-1.7B 用 `q4f16_1` 从约 3.4 GB fp16 压到 961 MB——一个实打实的 3.5 倍权重节省——而 KV 纹丝不动地钉在 112 KB/token。把这两件事混为一谈，是人们得出“4-bit 模型的长上下文应该很便宜”这个错误结论最常见的原因。

实操顺序：

1. 先算 `上下文 × 字节/token`（记得除以 `full_attention_interval`）。如果超过可用内存的约 60%，那就是结构性问题。
2. 优先选支持 KV 量化的引擎：llama.cpp 系用 `-ctk q8_0 -ctv q8_0 -fa on`。`-fa on` 是**必需的**，否则量化 KV 拿不到效率。
3. 如果被锁在某个不支持 KV 量化的引擎上，就砍上下文，别去撞 OOM——编译期的 `sliding_window_size` 能把 KV 固定成常数大小，代价只是丢掉远期记忆。
4. 永远不要靠改权重量化等级去指望改善上下文。它对 KV 的影响是零。

→ [`references/kv-quantization.md`](references/kv-quantization.md)

## 参考文件索引

| 文件 | 内容 |
|---|---|
| [`references/throughput-bottleneck.md`](references/throughput-bottleneck.md) | 瓶颈诊断、那些毫无作用的参数矩阵、量化格式选择、词表税、投机解码、单流与聚合的口径问题 |
| [`references/memory-budget.md`](references/memory-budget.md) | 真实内存与标称内存、`cudaMalloc` 上限、CMA 与 CUDA 的区别、cgroup 关笼子、OOM 前兆、桌面栈开销 |
| [`references/emc-bandwidth.md`](references/emc-bandwidth.md) | EMC 2133→3199 MHz 解锁、为什么 `apt upgrade` 做不到、三条独立读数、带宽测量路径、会污染结果的时间测量 bug |
| [`references/context-scaling.md`](references/context-scaling.md) | 上下文实测扫描、混合注意力与稠密注意力的 KV 算法、prefill 与生成的成本模型、64K 时真正会坏在哪里 |
| [`references/kv-quantization.md`](references/kv-quantization.md) | KV 字节/token 公式、哪些引擎能量化 KV、llama.cpp 参数、MLC 缺失的证据、`sliding_window_size` 兜底、NvMap/CMA 误诊 |

## 测试平台

| 项 | 值 |
|---|---|
| 设备 | Jetson Orin Nano 8GB（P3767，P3768 载板） |
| L4T | R39.2.1 |
| Ubuntu | 24.04.4 |
| CUDA | 13.2 |
| 计算能力 | sm_87 |
| CPU | 6 核 |
| 内核 | 6.8.12-tegra |

第 1–3 条和第 6 条属于运行时行为事实，可以迁移到其他 Orin SKU。第 4、5 条的数字是这块板特有的——在 NX/AGX 上引用之前请重新测量。

## 相关仓库

- [`mlc-llm-on-jetson`](https://github.com/zazuss/mlc-llm-on-jetson)——在 Orin 上从源码编译 MLC-LLM（这个平台没有官方 JetPack wheel）
- [`llama-prism-on-jetson`](https://github.com/zazuss/llama-prism-on-jetson)——PrismML 的 llama.cpp fork 在 CUDA 下的构建，用于三值量化模型，附带预编译二进制

## License

文档与脚本 MIT。实测数据随便用。
