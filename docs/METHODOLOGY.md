# LLM Capacity Planning & Hardware Sizing — Methodology

As LLMs scale, infrastructure planning must move from heuristics to precise modelling. Because autoregressive decoding generates **one token at a time**, serving is inherently **memory-bound** — the bottleneck is memory *bandwidth*, not compute. This is the deterministic model the calculator uses.

Fixed constants: runtime reserve **2.5 GB**, weight overhead **×1.02**, MBU **0.55**, MLA latent **576**.

## 1. Hardware memory modelling

How much high-bandwidth memory (HBM) the inference engine (e.g. vLLM) may use. The `gpu_memory_utilization` factor caps it; a fixed runtime reserve avoids out-of-memory failures.

```
Usable VRAM per GPU = (Physical capacity × Utilisation) − Runtime reserve
```

Tensor Parallelism (TP) splits one replica across GPUs; their usable memory pools linearly:

```
Usable pod memory = Usable VRAM per GPU × TP size
```

## 2. Weights vs. dynamic cache

Pod memory splits between **static weights** and the **dynamic KV cache**. What's left after weights is the budget for concurrency:

```
Free KV space = Usable pod memory − (Total parameters × Bytes per parameter)
```

Bytes per parameter follow the quantisation: FP16 = 2, FP8/INT8 = 1, INT4/MXFP4 = 0.5, NVFP4 ≈ 0.625. For Mixture-of-Experts models, all experts must be resident, so weights use the **total** parameter count.

## 3. KV cache & concurrency

KV cache grows linearly with sequence length and batch size — the real limiter for long-context, high-concurrency serving. Per-token size depends on the attention geometry:

```
Bytes per token = 2 × layers × KV-heads × head-dim × Bytes per element
```

The factor 2 covers Key and Value tensors. **MLA** models (DeepSeek, Kimi) compress KV into a latent instead — `layers × 576 × bytes` — materially smaller. Per request:

```
KV per session (GB) = (Bytes per token × Active tokens) / 1024³
```

where *active tokens = context length × average utilisation*. The most sessions one pod can hold:

```
Max pod concurrency = floor( Free KV space / KV per session )
```

## 4. Decode roofline throughput

For every token generated, the weights and active KV cache are read from memory to the compute cores — so generation speed is bounded by achievable bandwidth (with an MBU penalty).

```
Data read per step        = Weight memory + (Active sequences × KV per session)
Aggregate throughput      = (Effective pod bandwidth / Data read per step) × Active sequences
Effective pod bandwidth   = TP size × per-GPU bandwidth × MBU
```

## 5. Worked example — Llama 3.3 70B

Host Llama 3.3 70B Instruct at FP8, 10 concurrent sessions, 128K context at 60% utilisation, on 2× H200 (TP2).

| Step | Calculation | Result |
|---|---|---|
| Usable memory | (141 × 0.90) − 2.5 = 124.4; × 2 GPUs | **248.8 GB** |
| Weights & free cache | weights (FP8) ≈ 72.0; 248.8 − 72.0 | **176.8 GB free** |
| KV per token | 2 × 80 × 8 × 128 × 1 = 163,840 B | **0.156 MB** |
| KV per session | 131,072 × 0.60 = 78,643 active tokens × 0.156 MB | **≈ 12.0 GB** |
| Concurrency | floor(176.8 / 12.0) = 14 ≥ 10 target | **1 pod (2 GPUs)** |
| Throughput | data/step = 72 + 10×12 = 192 GB; bw = 2×4.8 TB/s × 0.55 ≈ 5,280 GB/s; (5,280/192)×10 | **≈ 275 tok/s** |

## Notes

Outputs are first-order roofline estimates (throughput ±40%, TTFT ±50%). Real numbers depend on kernels, batching, prefix caching and speculative decoding — treat them as planning figures and validate against benchmarks before procurement. A model is **infeasible** when weights plus one request's KV can't fit even at the largest permitted TP size.
