# LLM Capacity Planning & Hardware Sizing — Methodology

As LLMs scale, infrastructure planning must move from heuristics to precise modelling. Because autoregressive decoding generates **one token at a time**, serving is inherently **memory-bound** — the bottleneck is memory *bandwidth*, not compute. This is the deterministic model the calculator uses.

Fixed constants: runtime reserve **2.5 GiB**, MBU **0.55**, MLA latent **576**, tight-fit threshold **10%**.

All memory is **GiB**; see §7 for why that matters. A flat weight overhead of ×1.02 survives only as a fallback for models with no embedding geometry (§2).

## 1. Hardware memory modelling

How much high-bandwidth memory (HBM) the inference engine (e.g. vLLM) may use. The `gpu_memory_utilization` factor caps it; a fixed runtime reserve avoids out-of-memory failures.

```
Usable VRAM per GPU = (Physical capacity × Utilisation) − Runtime reserve
```

Every GPU's HBM therefore divides into five parts that sum to the card, and the sizing view draws them that way:

| slice | what it is |
|---|---|
| **Weights** | this replica's shard of the model |
| **KV in use** | cache for the sessions the plan actually places |
| **KV free** | usable now — what more concurrency would consume |
| **Runtime reserve** | vLLM's own overhead: CUDA context, collectives, prefill activations |
| **Withheld by mem-util** | the `1 − gpu_memory_utilization` slice never handed to vLLM |

Only the last is recoverable by changing a flag, and only at the cost of the margin that keeps the server off an OOM. Collapsing the last three into one "reserve" figure — as this tool originally did — hides that distinction and makes a 141 GiB card look like it has 20 GiB of untouchable overhead when 14 of it is a slider position.

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

### Mixed-precision checkpoints

Frontier low-bit releases are rarely uniform. NVIDIA ModelOpt's GLM-5.2 NVFP4 card says *"only MoE expert linears are quantized"*; DeepSeek-V4 ships *"MoE experts FP4, remaining params FP8"*; Mistral's FP8 keeps *"vision tower / projector / lm_head in BF16"*. Of the 425 recipe variants surveyed, **32 explicitly name which tensors are quantised.**

A single bytes/param cannot express that, so a model may declare a **dense remainder** — attention, shared experts, router and dense MLP, excluding the embedding tail — and which precision that remainder keeps under a given quant:

```
weights = quantised_body × effective_bytes(quant)
        + dense_params  × effective_bytes(dense_quant)
        + tail          × 2
```

GLM-5.2's dense block is 16.5 B parameters. Sizing its NVFP4 checkpoint as uniform reads 0.75 of the published VRAM floor; keeping the dense block at 16-bit reads **0.80**, inside the healthy band. Quants with no declaration stay uniform — GLM-5.2's MXFP4 is deliberately left alone, because AMD's card says only "MoE weights quantized", which is not specific enough to model differently.

### Prefill activations

The runtime reserve is **not a constant, and not the same on every GPU of a replica.** A prefill chunk materialises activations for every token in it simultaneously, and tensor parallelism shards most of them:

```
elems/token/GPU = 3 × hidden            (replicated: residual in, residual out, one temp)
                + ( 3 × FFN_width + hidden ) / TP     (sharded: gate, up, their product, qkv)

reserve = max( 2.5 GiB , 1.5 GiB CUDA context + chunk × elems/token/GPU × 2 bytes )
```

Two properties matter. The **sharded term** dominates — gate and up are column-parallel, so a rank holds 1/TP of the widest tensors in the layer. The **replicated term does not shrink**, so the reserve converges on `3 × hidden × chunk × 2` rather than on zero: a TP64 plan still reserves something. Peak is **one layer's worth**, not the whole stack, because activations are freed as the forward pass advances — layer count does not appear.

`FFN_width` is the width one token traverses in one layer. For a dense model that is `intermediate_size`. For MoE it is `moe_intermediate_size × num_experts_per_tok` **plus the always-on shared experts**, because a token materialises activations in every routed expert it visits *and* in every shared one. Shared experts are not a rounding error: DeepSeek-V3's single shared expert against top-8 is an eighth of the total, and Kimi K3's two against top-16 the same again.

That gives DeepSeek-V3 `2048 × (8 + 1) = 18,432` per token — exactly the `intermediate_size` its own dense layers use, which is a good sign the model is right rather than a coincidence. Field spellings differ by family, so the import reads both `num_experts_per_tok` and Kimi's `num_experts_per_token`, and takes shared experts either as a count (`n_shared_experts` / `num_shared_experts`) or as a width (Qwen2-MoE's `shared_expert_intermediate_size`).

Every model in the seeded catalogue declares its width, read from its published `config.json`. A model that declares none falls back to **3.5 × hidden_size**, the SwiGLU convention — which over-reserves for MoE, whose per-token width is typically narrower, and is the safe direction to be wrong in. The ratios that result span 2.6× (DeepSeek-V3, Kimi K2) to 7.7× (Kimi K3, whose 16 routed and 2 shared experts of 3072 make it by far the widest per-token FFN in the catalogue), so no single ratio would have served.

Because the reserve depends on TP, it is evaluated **per candidate shard width inside the TP search**, not once up front. Hoisting it out would charge every candidate the TP1 activation peak and bias the selection toward wide shards for the wrong reason.

Llama-3.3-70B (hidden 8192, FFN 28,672) at a 32K chunk shows the spread: **8.75 GiB at TP1, 5.88 at TP2, 4.44 at TP4, 3.72 at TP8.** The flat `chunk × hidden × 12 bytes` model this replaces answered 4.50 GiB at every width — it under-reserved narrow shards by nearly 2× and over-reserved wide ones, which is exactly the error being removed.

The chunk at which the floor stops binding is worth stating outright, because otherwise it is only findable by bisecting the input by hand:

```
floor chunk = 1 GiB / ( elems/token/GPU × 2 bytes )
```

That is ~11.2K tokens for Llama-3.3-70B at TP4 and ~4.5K for Mistral-Small-24B at TP1, whose 32,768-wide FFN over hidden 5120 is the steepest ratio in the catalogue. The planner shows the figure for the shard width the plan actually chose, next to the derived reserve.

Two defaults are distinct, and the tool keeps them distinct. vLLM's own default chunk is **2048**, which its tuning guide presents as the inter-token-latency choice, and the emitted `vllm serve` command only names the flag when the plan differs from it. The planner's own starting point is **8192**, because vLLM recommends `> 8192` for throughput and throughput is what this tool sizes for. At that chunk every catalogue model is back on the 2.5 GiB floor once sharded eight ways; unsharded, the widest FFNs do leave it.

A reserve can now exhaust the card on its own — a large enough chunk on a small enough GPU leaves nothing for weights. That is reported as its own infeasibility, naming `--max-num-batched-tokens`, rather than being folded into the "weights do not fit" message that would send the reader to the wrong lever.

The multiples (3 replicated, 3 sharded at FFN width, 1 sharded at hidden) fold qkv, the gate/up projections, their SiLU product and the residual copies into whole numbers at 2-byte activations. It is a structural model, not a kernel-accurate one: it tracks what tensor parallelism does and does not divide, which a flat reserve cannot, but it does not model fusion, recomputation or an always-on shared expert. Models with no `hidden_size` fall back to the flat 2.5 GiB.

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

### Choosing the tensor-parallel size

The obvious rule — smallest TP that holds the weights plus one request — **over-recommends hardware.** A bigger shard leaves proportionally more room for KV and so packs far more sessions per replica, and total cost is `pods × TP`, not `TP`.

Llama-3.3-70B at FP8, 128K/60%, 64 concurrent, on H200:

| TP | free KV | sessions/pod | pods | **total GPUs** |
|---|---|---|---|---|
| 1 | 57 GiB | 4 | 16 | 16 |
| 2 | 181 GiB | 15 | 5 | 10 |
| 4 | 430 GiB | 35 | 2 | **8** |
| 8 | 927 GiB | 77 | 1 | **8** |

The engine evaluates every TP in the model's ladder and takes the cheapest total, breaking ties toward the **smaller** shard (same GPU count, less collective traffic). That makes this a cost/throughput objective; a latency-oriented planner would bias toward larger shards instead.

### What crossing a node boundary costs

TP is not free at any width, but the price jumps at the node boundary. A tensor-parallel group performs an **all-reduce on every layer, for every token** — inside a node that traffic rides NVLink at multiple TB/s; between nodes it rides InfiniBand or RoCE at a fraction of that, and it sits on the critical path of every forward pass. The throughput and TTFT figures here assume the collective is not the bottleneck, which stops being true once a replica spans nodes.

The sizing view draws this rather than asserting it: every GPU on one axis, grouped into node boxes, with each replica as a bar above them. A replica that fits inside a node is a bar inside one box; one that does not is a bar visibly spanning the gap where the fabric is drawn. Replicas are independent of each other — they share only the weights on storage and the router in front — so scaling *out* adds throughput without adding collective traffic, while scaling *up* past the node boundary adds both.

This is also why `tp_options` should be generous. Listing extra small sizes is harmless — infeasible ones are skipped — but listing too few silently forces more hardware than the model needs.

### Local & global attention

Not every layer attends over the whole context. Many models alternate **full-attention** layers with **sliding-window** layers that only look back a fixed number of tokens. Windowed layers' KV stops growing once the sequence passes the window:

```
KV per request = perLayer × ( full × tokens + windowed × min(tokens, window) )
```

GPT-OSS-120B is the clearest case: 18 of its 36 layers are locally banded at 128 tokens. At 128K context that is **2.7 GiB** per request instead of **5.4 GiB** — and the difference between room for three concurrent requests on one H100 and a plan that looks dangerously tight with room for one. Gemma (one global layer every N) and Mistral v0.1 (every layer windowed) use the same idea with different patterns.

Because the windowed layers stop growing, the per-token KV figure the app reports is an *effective average* over the request rather than a constant marginal rate. It still reconciles exactly: per-token × active tokens = per-request. A model with no window declared is treated as all-full-attention — the safe direction to be wrong in.

### Linear & recurrent layers

A third regime is spreading fast. Hybrid models replace most attention layers with a **recurrent** form — Kimi K3's KDA, Qwen3-Next, MiniMax — whose state is a fixed-size matrix per layer that does not grow with the sequence at all:

```
KV per request = perLayer × ( full × tokens + windowed × min(tokens, window) )
               + linear × constant_state
```

Qwen3.6-27B is 64 layers with only **16** cached (4.9 GiB of KV at 256K instead of 19.2), and Kimi K3 has 93 layers of which only **24** keep a token-indexed cache; the other 69 are KDA, costing a flat ~414 MB per request whether the context is 1K or 1M (state = `num_heads × head_dim² × 4` bytes, fp32). At its full 1M window that is **8.5 GiB** of KV instead of the 31.4 GiB an all-93-layer sizing would claim — the difference between fitting one 8×B300 node and not. The constant term dominates at short context and vanishes at long, so it is modelled rather than dropped.

Every layer must be accounted for: `full + windowed + linear = layers`, and validation rejects a split that leaves layers unexplained.

Vendors spell this incompatibly, and the importer handles each. Kimi K3 nests a `linear_attn_config` listing both layer sets; Qwen3.6 puts `linear_attention` in `layer_types` alongside flat `linear_num_value_heads` / `linear_key_head_dim` keys. The importer buckets `layer_types` by vocabulary — anything containing *linear*, *mamba* or *recurrent* is linear; *sliding* or *local* is windowed; anything unrecognised defaults to full attention, so a new spelling over-sizes rather than under-sizes.

`config.json` expresses the sliding-window case three ways, all of which the HF importer handles: `layer_types` (per-layer array), `sliding_window_pattern` (one global every N), or a bare `sliding_window` (every layer windowed).

## 4. Decode roofline throughput

For every token generated, the weights and active KV cache are read from memory to the compute cores — so generation speed is bounded by achievable bandwidth (with an MBU penalty).

```
Data read per step (bytes) = (Active weights + Active sequences × KV per session) × 2³⁰
Effective pod bandwidth    = TP size × per-GPU TB/s × 10¹² × MBU
Aggregate throughput       = (Effective pod bandwidth / Data read per step) × Active sequences
```

Note the explicit `× 2³⁰`: memory here is GiB but bandwidth is decimal, so both sides go to raw bytes before dividing. **Active weights** is not the same as total weights — only the output head streams every step; the embedding table is a per-token gather. For MoE models only the *active* parameters are read.

### Time to first token is a different problem

Decode is memory-bound; **prefill is not.** It runs the entire prompt through the network before emitting a token, so TTFT is bounded by arithmetic:

```
prefill FLOPs = 2 × active params × tokens                     (dense matmuls, linear)
              + 4 × hidden × ( full × tokens²                  (attention, QUADRATIC)
                             + windowed × tokens × window
                             + linear × tokens )

TTFT = max( prefill FLOPs / (TP × TFLOPS × speedup × MFU) , weight-streaming time )
```

The attention term is not a correction — it is usually the larger half. Llama-3.3-70B prefilling 78,643 tokens spends **16.2 PFLOPs on attention against 11.1 on the matmuls**. Long-context TTFT is an attention problem, which is why the layer regimes matter here as much as they do for KV: a sliding-window layer costs `tokens × window` rather than `tokens²`, making GPT-OSS-120B's prefill **38% cheaper** than the same shape withfull attention.

Prefill MFU is taken as 0.4 — a large dense GEMM reaches far better utilisation than decode, but nowhere near peak. Sub-16-bit formats get a 2× tensor-core speedup, capped there even for 4-bit: Blackwell does better, but the catalog does not track GPU generation and under-promising TTFT is the safe direction. A SKU with no `tflops_fp16` falls back to the weight-streaming floor, and the result is flagged as such rather than presented as a prefill estimate.

## 5. Worked example — Llama 3.3 70B

Host Llama 3.3 70B Instruct at FP8, 10 concurrent sessions, 128K context at 60% utilisation, on H200s at vLLM's default prefill chunk.

TP is not assumed — it is selected. At this concurrency TP1 needs 3 GPUs, TP2 needs 2, TP4 needs 4; TP2 wins (§3, *Choosing the tensor-parallel size*).

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

## 7. A note on units

Every memory figure here and in the app is **GiB = 2³⁰ bytes** — the unit `nvidia-smi` reports and the one `gpu_memory_utilization` is applied against. Parameter counts are quoted in billions (10⁹), so weights need an explicit conversion:

```
weights (GiB) = params_b × bytes/param × 10⁹ ÷ 2³⁰
```

Skipping that conversion — a common shortcut — makes weights read **7.4% larger** than they are relative to GPU capacity. Conservative, but wrong, and it compounds against a KV figure that *was* converted.

Bandwidth is the deliberate exception: `bw_tbs` is decimal TB/s (10¹² B/s) as vendors quote it, so the roofline converts memory to raw bytes before dividing rather than mixing the two scales.

## 8. Concurrency rubric

The rubric re-runs the **entire** sizing at each target concurrency — it is not a scaling of one result. That matters because TP selection depends on the target: the cheapest shard width at 1 concurrent session is rarely the cheapest at 256. A rubric row can therefore change TP, pods and the tight verdict all at once, which a linear extrapolation would hide.

Per-request decode rate falls as concurrency rises (more sessions share the pod's bandwidth) while aggregate throughput and GPU count rise.

## 9. Cost model

Costs are rental-rate arithmetic on an admin-set `$/GPU-hour` per SKU. Nothing is amortised, and power, networking and storage are out of scope.

```
Run rate ($/hr) = Σ over SKUs ( committed GPUs × $/GPU-hour )
$/month  = $/hr × 730          $/year = $/hr × 8760

$ per million tokens = ( GPUs × $/GPU-hour × 1,000,000 ) / ( tokens/sec × 3600 )
```

The per-million-token figure inherits the throughput estimate's ±40% band, so treat it as a comparison tool between configurations rather than a budget line. A configuration that halves GPU count but also halves throughput costs the same per token.

The same arithmetic runs on a single deployment, so the sizing view reports its own run rate, monthly figure and cost per million tokens without waiting for the plan to be added to a cluster.

## 10. Fleet reconciliation and the capacity gate

A plan is checked against a declared fleet **per SKU**, on integer bytes:

```
fleet_bytes     = fleet GPUs     × mem_gb × 2³⁰
committed_bytes = committed GPUs × mem_gb × 2³⁰
committed + available = fleet total        (invariant, always)
```

Two properties are deliberate. Commitments count **whole GPUs** — a replica occupying part of a card still retires the whole card, because vLLM does not share a GPU between deployments. And there is **no cross-SKU masking**: surplus H200s never offset an H100 shortage, so each SKU is reconciled independently and a plan is over-committed if *any* SKU is.

Integer bytes rather than floats keep the browser's live verdict and the server's authoritative one identical at the boundary.

## 11. Launch command

Every feasible plan emits the `vllm serve` command implied by its own numbers — TP size, `--max-model-len`, `--gpu-memory-utilization`, `--kv-cache-dtype`, `--max-num-batched-tokens` where it differs from default, and `--max-num-seqs`.

`--max-num-seqs` is the one worth understanding. It is set to the **pod's** KV budget, not the deployment target. Left at vLLM's default the scheduler admits more sequences than the cache can hold and preempts under load — which presents as a throughput problem and is really a sizing one.

The command describes **one replica**. A plan needing *n* pods needs *n* copies of it behind a load balancer; conflating the two is the classic way to under-provision.

## 12. Where catalog numbers come from

Model geometry is not guessed. Two sources, deliberately separated by what each is authoritative for:

| source | supplies |
|---|---|
| Hugging Face `config.json` | layers, attention geometry, embedding sizes, sliding-window and linear-attention splits |
| [recipes.vllm.ai](https://recipes.vllm.ai) | parameter counts, context length, shipped quantisations, TP sizes |

The second set is precisely what a `config.json` never carries, and what the importer would otherwise leave for an admin to type. A recipe has no authority over geometry and is not allowed to set it.

Recipes also publish a `vram_minimum_gb` per variant, which the import panel shows beside our own weight estimate. Their floor covers weights + KV + overhead, so a ratio near **0.85** is healthy; above 1.00 means one of the two figures is wrong. Across the 20 catalogue models checked against published recipes the median is 0.85, with one outlier flagged in the notes below.

## Notes

**TTFT was wrong until it was measured against arithmetic.** It was originally modelled as the time to stream the active weights once — which is the *decode* bound. For a 78,643-token prefill that produced **7 ms** where the compute-bound figure is **~8.6 seconds**, a factor of 1,200. The lesson generalises: a bound borrowed from the wrong phase is worse than no estimate, because it carries a plausible error bar.

Outputs are first-order roofline estimates (throughput ±40%, TTFT ±50%). Real numbers depend on kernels, batching, prefix caching and speculative decoding — treat them as planning figures and validate against benchmarks before procurement.

**Catalog geometry is sourced, not guessed.** Every seeded model's layer count, attention geometry and embedding sizes are taken from its published `config.json`. Three findings came out of that pass worth recording: GPT-OSS's 128-token window applies to exactly half its layers (18/36 and 12/24), and **GLM-5.2 is an MLA model, not GQA** — it ships as `GlmMoeDsaForCausalLM` with `kv_lora_rank: 512` + `qk_rope_head_dim: 64`, the same 576-element latent per layer DeepSeek uses. Sizing it as GQA 8×128 overstated its KV cache by ~3.6×, which is the difference between 8 GPUs and 24 for a 64-user deployment. And Kimi K3 keeps a token cache on only 24 of its 93 layers (§3, *Linear & recurrent layers*).

**One unexplained cross-check.** Llama-3.3-70B at NVFP4 computes to 1.02 of its published VRAM floor — physically impossible, so one of the two is wrong. That checkpoint appears to quantise the embedding tail as well, but no published card says so, and fitting a constant to a single data point is how the original NVFP4 error arose. It is left flagged rather than tuned away.

**Sparse attention is not a memory saving.** GLM-5.2's `index_topk: 2048` (DSA) selects which cached tokens each query attends to. It cuts attention *compute*; the KV cache still holds every token. This tool deliberately does not model it as a cache reduction — treating token-selection sparsity as eviction would under-size the deployment.

**Note on the ×1.02 fallback.** Models with no `hidden_size`/`vocab_size` still use the legacy flat factor. It is a worse estimate at low bit-widths, not a different unit — supply the embedding geometry and it disappears.
