# LLM Capacity Planning & Hardware Sizing — Methodology

As LLMs scale, infrastructure planning must move from heuristics to precise modelling. Because autoregressive decoding generates **one token at a time**, serving is inherently **memory-bound** — the bottleneck is memory *bandwidth*, not compute. This is the deterministic model the calculator uses.

Fixed constants: runtime reserve **2.5 GiB**, MBU **0.55**, MLA latent **576**, tight-fit threshold **10%**.

**Units.** Every memory figure here and in the app is **GiB = 2³⁰ bytes** — the unit `nvidia-smi` reports and the one `gpu_memory_utilization` applies against. Parameter counts are in billions (10⁹), so weights convert explicitly: `params × bytes/param × 10⁹ ÷ 2³⁰`. Bandwidth is the exception — `bw_tbs` is decimal TB/s (10¹² B/s) as vendors quote it, so the roofline takes memory to raw bytes before dividing. (A flat weight overhead of ×1.02 is retained only as a fallback for models with no embedding geometry — see §2.)

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
Free KV space = Usable pod memory − Weights
```

For Mixture-of-Experts models, all experts must be resident, so weights use the **total** parameter count — not the active one.

### Weights are not one flat bytes-per-parameter

A quantised checkpoint is not uniformly quantised. The **embedding table and the output head stay at 16-bit** in essentially every real INT4/FP8/MXFP4 release, because quantising them costs disproportionate quality. That tail is invisible at FP16 and dominant at 4-bit:

```
Weights = (Total params − Un-quantised tail) × Effective bytes/param
        + Un-quantised tail × 2

Un-quantised tail (params) = vocab_size × hidden_size × (tied embeddings ? 1 : 2)
```

For Llama-3.3-70B the tail is 2 × 128,256 × 8,192 = **2.10 B parameters = 3.9 GiB** — over 10% of an INT4 checkpoint, and far more than a flat ×1.02 factor allows for. Modelling it is the difference between predicting 32.9 GiB (wrong) and 37.1 GiB (what the published AWQ/GPTQ checkpoints actually weigh — ~40 × 10⁹ bytes).

**Effective** bytes per parameter also exceed the nominal bit-width, because low-bit formats store per-group scale metadata alongside the data:

| Quant | Nominal | Effective | Metadata |
|---|---|---|---|
| FP16 | 2 | 2 | — |
| FP8 / INT8 | 1 | 1 | per-tensor/channel scale — negligible |
| INT4 (grouped, g=128) | 0.5 | **0.52** | fp16 scale + int4 zero per 128 weights |
| MXFP4 (block=32) | 0.5 | **0.53125** | E8M0 scale per 32 weights |
| NVFP4 (block=16) | 0.5 | **0.5625** | E4M3 scale per 16 weights |
| GGUF Q8_0 | 1.0 | **1.06** | 8.5 real bits — whole-file average |
| GGUF Q4_K_M | 0.5 | **0.61** | 4.9 real bits — mixes Q4_K and Q6_K per tensor |
| GGUF IQ4_XS | 0.5 | **0.53125** | 4.25 real bits |

When a model in the catalog carries no `hidden_size`/`vocab_size`, the engine falls back to the legacy `total × bytes × 1.02` estimate and the sizing view labels the result approximate. Hugging Face import fills all three fields automatically from `config.json`.

### The GGUF trap

Q4_K_M is **not** 4 bits. It mixes Q4_K and Q6_K tensor by tensor and lands near **4.9 effective bits**; assuming 4.0 undercounts a 70B model by roughly 8 GB. The GGUF figures above are whole-*file* averages measured across a finished checkpoint, and GGUF quantises the embedding layers too — so those quants **skip** the un-quantised-tail term rather than paying it twice:

```
GGUF weights = total params × effective bytes/param        (no tail term)
other weights = (total − tail) × bytes/param + tail × 2
```

Cross-checks against published checkpoints: Llama-3.3-70B Q4_K_M → **43.1 GB**, Llama-3.1-8B Q4_K_M → **4.9 GB**. Both are pinned as acceptance vectors.

GGUF is llama.cpp / Ollama territory; vLLM's support for it is experimental. The GGUF quants exist here because the catalog now includes consumer cards, where they are the dominant format — but note that this tool's overhead model (a 2.5 GiB reserve plus a ~10% paged-server utilisation cap) is tuned for vLLM. llama.cpp runs leaner, so single-card GGUF plans will read more pessimistically here than they behave in practice.

### Prefill activations

The 2.5 GiB runtime reserve covers the CUDA context (~1–2 GiB) plus activation buffers at default settings. It is a **flat** figure: raising `--max-num-batched-tokens` for chunked prefill grows the activation peak beyond what this model accounts for. Treat long-prefill, large-batch configurations as under-reserved.

## 3. KV cache & concurrency

KV cache grows linearly with sequence length and batch size — the real limiter for long-context, high-concurrency serving. Per-token size depends on the attention geometry:

```
Bytes per token = 2 × layers × KV-heads × head-dim × Bytes per element
```

The factor 2 covers Key and Value tensors. **MLA** models (DeepSeek, Kimi) compress KV into a latent instead — `layers × 576 × bytes` — materially smaller. Per request:

```
KV per session (GiB) = (Bytes per token × Active tokens) / 1024³
```

where *active tokens = context length × average utilisation*. The most sessions one pod can hold:

```
Max pod concurrency = floor( Free KV space / KV per session )
```

### Local & global attention

Not every layer attends over the whole context. Many models alternate **full-attention** layers with **sliding-window** layers that only look back a fixed number of tokens. Windowed layers' KV stops growing once the sequence passes the window:

```
KV per request = perLayer × ( full × tokens + windowed × min(tokens, window) )
```

GPT-OSS-120B is the clearest case: 18 of its 36 layers are locally banded at 128 tokens. At 128K context that is **2.7 GiB** per request instead of **5.4 GiB** — and the difference between room for three concurrent requests on one H100 and a plan that looks dangerously tight with room for one. Gemma (one global layer every N) and Mistral v0.1 (every layer windowed) use the same idea with different patterns.

Because the windowed layers stop growing, the per-token KV figure the app reports is an *effective average* over the request rather than a constant marginal rate. It still reconciles exactly: per-token × active tokens = per-request. A model with no window declared is treated as all-full-attention — the safe direction to be wrong in.

`config.json` expresses this three ways, all of which the HF importer handles: `layer_types` (per-layer array), `sliding_window_pattern` (one global every N), or a bare `sliding_window` (every layer windowed).

## 4. Decode roofline throughput

For every token generated, the weights and active KV cache are read from memory to the compute cores — so generation speed is bounded by achievable bandwidth (with an MBU penalty).

```
Data read per step (bytes) = (Active weights + Active sequences × KV per session) × 2³⁰
Effective pod bandwidth    = TP size × per-GPU TB/s × 10¹² × MBU
Aggregate throughput       = (Effective pod bandwidth / Data read per step) × Active sequences
```

Note the explicit `× 2³⁰`: memory here is GiB but bandwidth is decimal, so both sides go to raw bytes before dividing. **Active weights** is not the same as total weights — only the output head streams every step; the embedding table is a per-token gather. For MoE models only the *active* parameters are read.

## 5. Worked example — Llama 3.3 70B

Host Llama 3.3 70B Instruct at FP8, 10 concurrent sessions, 128K context at 60% utilisation, on 2× H200 (TP2).

| Step | Calculation | Result |
|---|---|---|
| Usable memory | (141 × 0.90) − 2.5 = 124.4; × 2 GPUs | **248.8 GiB** |
| Un-quantised tail | 2 × 128,256 × 8,192 = 2.10 B params, held at fp16 | **3.9 GiB** |
| Weights & free cache | ((70.6 − 2.10) × 1.0 + 2.10 × 2) × 10⁹ ÷ 2³⁰ = 67.7; 248.8 − 67.7 | **181.1 GiB free** |
| KV per token | 2 × 80 × 8 × 128 × 1 = 163,840 B | **0.156 MiB** |
| KV per session | 131,072 × 0.60 = 78,643 active tokens × 0.156 MiB | **≈ 12.0 GiB** |
| Concurrency | floor(181.1 / 12.0) = 15 ≥ 10 target | **1 pod (2 GPUs)** |
| Pod headroom | (248.8 − 67.7 − 12.0) / 248.8 = 68% ≥ 10% | **fits (not tight)** |
| Throughput | data/step ≈ (66.7 + 10×12) × 2³⁰ ≈ 201×10⁹ B; bw = 2 × 4.8×10¹² × 0.55 ≈ 5.28×10¹² B/s → 38.1 ms/step | **≈ 262 tok/s** |

## 6. Fits / tight / infeasible

A plan is **infeasible** when weights plus *one* request's KV can't fit even at the largest permitted TP size. Between that and a comfortable fit sits a band worth naming:

```
Pod headroom = (Usable pod memory − Weights − KV per session) / Usable pod memory
Tight        = Pod headroom < 10%
```

A tight plan is arithmetically feasible but has no margin for the ±5% weight estimate, allocator fragmentation, or a prompt longer than the modelled average — it is the configuration that passes a spreadsheet and then OOMs on launch. Qwen3-32B at Q4_K_M on a single RTX 4090 is the canonical example: 18.6 GiB of weights against 19.1 GiB usable leaves **0.9% headroom** at 4K context and room for exactly one request. Push the context to 8K and it needs TP2, where it is comfortable again.

## Notes

Outputs are first-order roofline estimates (throughput ±40%, TTFT ±50%). Real numbers depend on kernels, batching, prefix caching and speculative decoding — treat them as planning figures and validate against benchmarks before procurement.

**Note on the ×1.02 fallback.** Models with no `hidden_size`/`vocab_size` still use the legacy flat factor. It is a worse estimate at low bit-widths, not a different unit — supply the embedding geometry and it disappears.
