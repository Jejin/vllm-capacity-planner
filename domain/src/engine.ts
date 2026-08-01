// Sizing engine — the calculation contract (addendum §A). Pure functions only (AD-1).
// Every constant is from addendum §A and verified against the §C acceptance vectors.
//
// UNITS — read this before touching any arithmetic here.
//   Every memory quantity in this engine (and every `*_gb` field it returns) is **GiB = 2^30
//   bytes**, matching `nvidia-smi`, `GpuSku.mem_gb`, and the integer-byte capacity gate in
//   reconcile.ts. The `_gb` field names are kept for API and saved-snapshot compatibility.
//   Parameter counts are in BILLIONS (1e9), so params → bytes → GiB needs an explicit
//   conversion: `paramBytesToGib()`. Do not multiply params_b by bytes/param and call the
//   result GiB — that is 1e9-byte GB and reads ~7.4% high against GPU capacity.
//   Bandwidth (`GpuSku.bw_tbs`) is DECIMAL TB/s (1e12 B/s) as vendors quote it, so the
//   roofline converts memory to bytes rather than mixing the two scales.

import type {
  Model,
  GpuSku,
  Quant,
  SizingInput,
  Sizing,
  FeasibleSizing,
} from './types.js';

/**
 * Per-GPU runtime reserve, GiB. Retained as the FLOOR of the reserve so that default-configured
 * plans size exactly as they did before activations were modelled.
 */
export const RUNTIME_GB = 2.5;
/** CUDA context + kernels + NCCL buffers, replicated on every card and independent of batch. */
export const CUDA_CONTEXT_GB = 1.5;
/** vLLM's default --max-num-batched-tokens; the prefill chunk when the caller does not say. */
export const DEFAULT_BATCHED_TOKENS = 2048;
/** Activations are held at 2-byte precision even when the weights are not. */
export const ACT_DTYPE_BYTES = 2;
/**
 * Activation elements per token that EVERY tensor-parallel rank materialises at full hidden
 * width: the residual coming in, the residual going out, and one norm/temporary buffer. These do
 * not shrink as the shard width grows, which is why a big-TP plan still reserves something.
 */
export const ACT_REPLICATED_HIDDEN = 3;
/**
 * Activation elements per token at INTERMEDIATE width, divided across ranks. gate, up and their
 * SiLU product, all produced by column-parallel projections, so each rank holds 1/tp of them.
 * This is the dominant term and the reason a TP-blind reserve is wrong in both directions.
 */
export const ACT_SHARDED_INTERMEDIATE = 3;
/** qkv and the attention output, also column-parallel; approximated at one hidden width. */
export const ACT_SHARDED_HIDDEN = 1;
/**
 * Assumed FFN width when a model does not declare `intermediate_size`. SwiGLU models cluster
 * near 3.5x hidden (Llama-3.3-70B is 28672 over 8192 = 3.5 exactly). MoE models are usually far
 * narrower per token, so this over-reserves for them — the safe direction to be wrong in.
 */
export const DEFAULT_INTERMEDIATE_RATIO = 3.5;
/**
 * Model FLOPs Utilisation achieved during prefill. Prefill is a big dense GEMM and reaches far
 * better utilisation than decode, but nowhere near peak once attention, norms and the scheduler
 * are counted. Distinct from MBU, which bounds the memory-bound decode phase.
 */
export const PREFILL_MFU = 0.4;
/**
 * Tensor-core speedup for sub-16-bit compute, relative to the SKU's FP16 figure. Capped at 2x
 * (FP8-class) even for 4-bit formats: Blackwell does better, but we do not track GPU generation
 * and under-promising TTFT is the safe direction.
 */
const LOW_PRECISION_SPEEDUP = 2;
/** Legacy flat overhead factor — used ONLY when a model carries no embedding geometry. */
export const WEIGHT_OVERHEAD = 1.02;
export const MBU = 0.55; // model bandwidth utilisation (decode roofline)
/**
 * Fallback MLA latent width, per layer per token, for a model that declares `mla: true` but
 * carries no `mla_latent_elems`. This is NOT a global sizing constant like MBU or the runtime
 * reserve — latent width is per-model geometry (`kv_lora_rank + qk_rope_head_dim`), and GQA
 * models have none at all. 576 is DeepSeek's 512 + 64, which Kimi and GLM-5.2 also ship, so it
 * is the safe default for an under-specified MLA entry rather than a universal truth.
 */
export const DEFAULT_MLA_LATENT_ELEMS = 576;
export const FP16_BYTES = 2; // the un-quantised tail (embeddings / lm_head) stays 16-bit
/** A feasible plan with less than this fraction of pod HBM free is reported as "tight". */
export const TIGHT_HEADROOM = 0.1;
/** 1 GiB in bytes — the single memory unit for the whole engine. */
export const GIB = 2 ** 30;
/** Parameter counts are quoted in billions; this is that scale in raw params. */
const BILLION = 1e9;
/** Decimal bytes/sec from a vendor-quoted TB/s figure. */
const TBS_TO_BYTES_PER_SEC = 1e12;

/** (billions of params) × (bytes per param) → GiB. The conversion the old code omitted. */
export function paramBytesToGib(paramsB: number, bytesPerParam: number): number {
  return (paramsB * BILLION * bytesPerParam) / GIB;
}

/**
 * EFFECTIVE bytes per parameter by quantisation — data bits PLUS the per-group metadata
 * that low-bit formats must store alongside them. Ideal bit-width alone under-counts:
 *   INT4  grouped g=128 : 0.5 + fp16 scale (2 B/128) + int4 zero (0.5 B/128) ≈ 0.52
 *   MXFP4 block=32      : 0.5 + E8M0 scale (1 B/32)                          ≈ 0.53125
 *   NVFP4 block=16      : 0.5 + E4M3 scale (1 B/16)                          ≈ 0.5625
 * FP8/INT8 use per-tensor or per-channel scales — negligible, so they stay at 1.0.
 */
export const QB: Record<Quant, number> = {
  FP16: 2,
  FP8: 1,
  INT8: 1,
  INT4: 0.52,
  MXFP4: 0.53125,
  NVFP4: 0.5625,
  // GGUF k-quants: the advertised bit-width is not what you get. Q4_K_M mixes Q4_K and Q6_K
  // per tensor and lands at ~4.9 effective bits; assuming 4.0 undercounts a 70B by ~8 GB.
  Q8_0: 1.06, // 8.5 bits
  Q4_K_M: 0.61, // 4.9 bits
  IQ4_XS: 0.53125, // 4.25 bits
};

/**
 * Quants whose bytes/param is a whole-FILE average, measured across the finished checkpoint.
 * GGUF quantises the embedding layers too (typically to Q6_K/Q8_0) and the published figure
 * already absorbs that, so these must NOT also pay the un-quantised-tail term — doing both
 * double-counts the embeddings.
 */
export const WHOLE_FILE_QUANTS: ReadonlySet<Quant> = new Set<Quant>(['Q8_0', 'Q4_K_M', 'IQ4_XS']);

/**
 * Per-GPU runtime reserve, split into the part that is fixed and the part that is not.
 *
 * A flat reserve is fine at default settings and wrong as soon as `--max-num-batched-tokens` is
 * raised: prefill materialises activations for the whole chunk at once, so the peak scales with
 * chunk x hidden_size. Sizing a long-prefill, large-chunk deployment against a flat 2.5 GiB
 * silently under-reserves and the server OOMs during warmup rather than under load.
 *
 * The floor is kept at RUNTIME_GB so nothing re-sizes for callers who never touch the chunk.
 */
export function runtimeReserveGib(
  model: Model,
  batchedTokens?: number,
  tp = 1,
): { total: number; activations: number } {
  const chunk = batchedTokens && batchedTokens > 0 ? batchedTokens : DEFAULT_BATCHED_TOKENS;
  const hidden = model.hidden_size;
  // Without hidden_size we cannot scale anything; fall back to the flat historical reserve.
  if (!hidden) return { total: RUNTIME_GB, activations: 0 };
  const activations = (chunk * activationElemsPerToken(model, tp) * ACT_DTYPE_BYTES) / GIB;
  return { total: Math.max(RUNTIME_GB, CUDA_CONTEXT_GB + activations), activations };
}

/**
 * Activation elements per token, per GPU, at a given tensor-parallel width.
 *
 * Tensor parallelism shards the FFN: gate and up are column-parallel, so a rank holds 1/tp of the
 * intermediate-width tensors, while the residual stream stays replicated at full width. A reserve
 * that ignores this is wrong in BOTH directions — it over-reserves at TP16, where the dominant
 * term has been divided by 16, and under-reserves at TP1, where nothing has been divided at all.
 *
 * `intermediate_size` is the FFN width one token traverses in one layer; for MoE that is the
 * per-expert width times the experts a token is routed to. Peak is one layer's worth, not the
 * whole stack, because activations are freed as the forward pass advances — which is why layer
 * count does not appear here.
 */
export function activationElemsPerToken(model: Model, tp = 1): number {
  const hidden = model.hidden_size;
  if (!hidden) return 0;
  const intermediate = model.intermediate_size ?? DEFAULT_INTERMEDIATE_RATIO * hidden;
  const width = Math.max(1, tp);
  return (
    ACT_REPLICATED_HIDDEN * hidden +
    (ACT_SHARDED_INTERMEDIATE * intermediate + ACT_SHARDED_HIDDEN * hidden) / width
  );
}

/**
 * The prefill chunk at which the reserve stops being the flat floor and starts tracking
 * activations — below it, raising `--max-num-batched-tokens` costs no memory at all because the
 * floor already covers the activations.
 *
 * Worth stating rather than leaving to be discovered by experiment: the breakeven scales as
 * 1/hidden_size, so it is ~10.9K tokens on Llama-3.3-70B (hidden 8192) and ~31K on GPT-OSS-120B
 * (hidden 2880). Null when the model carries no embedding geometry, where the reserve is flat
 * at every chunk.
 */
export function reserveFloorChunk(model: Model, tp = 1): number | null {
  const hidden = model.hidden_size;
  if (!hidden) return null;
  const perToken = activationElemsPerToken(model, tp) * ACT_DTYPE_BYTES;
  return Math.ceil(((RUNTIME_GB - CUDA_CONTEXT_GB) * GIB) / perToken);
}

/**
 * The latent width one MLA layer caches per token — the model's own geometry, falling back to
 * DeepSeek's 576 when the entry does not state it. Meaningless for a GQA model, which caches
 * `2 x kv_heads x head_dim` instead; callers should gate on `model.mla`.
 */
export function mlaLatentElems(model: Model): number {
  return model.mla_latent_elems ?? DEFAULT_MLA_LATENT_ELEMS;
}

/** KV bytes per token for ONE layer — GQA vs MLA (addendum §A). */
export function kvPerLayerPerTokenBytes(model: Model, kvDtypeBytes: number): number {
  return model.mla
    ? mlaLatentElems(model) * kvDtypeBytes
    : 2 * model.kv_heads * model.head_dim * kvDtypeBytes;
}

/** Nominal KV bytes per token across all layers, as if every layer were full-context. */
export function kvPerTokenBytes(model: Model, kvDtypeBytes: number): number {
  return model.layers * kvPerLayerPerTokenBytes(model, kvDtypeBytes);
}

/**
 * How a model's layers divide across the three attention regimes.
 *   full    — KV grows with the sequence
 *   windowed— KV grows to `sliding_window` tokens, then stops
 *   linear  — recurrent state, CONSTANT in sequence length (no per-token cache at all)
 * Undeclared models are all-full, which is the safe direction to be wrong in.
 */
export function layerSplit(model: Model): { full: number; windowed: number; linear: number } {
  const linear = model.linear_attention_layers ?? 0;
  const cached = model.layers - linear; // layers that keep a token-indexed cache
  const full = model.full_attention_layers ?? cached;
  const windowed = Math.max(0, cached - full);
  return { full: Math.max(0, Math.min(full, cached)), windowed, linear };
}

/** True when some layers are sliding-window or linear, so KV is below the all-full figure. */
export function hasWindowedLayers(model: Model): boolean {
  const { windowed, linear } = layerSplit(model);
  return (!!model.sliding_window && windowed > 0) || linear > 0;
}

/**
 * KV bytes for ONE request holding `activeTokens` of context.
 *
 *   full × perLayer × tokens
 * + windowed × perLayer × min(tokens, window)
 * + linear × constant state
 *
 * None of this is a refinement. GPT-OSS-120B at 128K halves (18 of 36 layers banded at 128
 * tokens); Kimi K3 falls ~3.9x (only 24 of 93 layers keep a token cache at all — the other 69
 * are KDA, whose state does not grow). Sizing those as full attention manufactures cache that
 * will never exist, and can turn a comfortable plan into a false "tight" verdict.
 */
export function kvPerRequestBytes(model: Model, kvDtypeBytes: number, activeTokens: number): number {
  const perLayer = kvPerLayerPerTokenBytes(model, kvDtypeBytes);
  const { full, windowed, linear } = layerSplit(model);
  const windowTokens = model.sliding_window
    ? Math.min(activeTokens, model.sliding_window)
    : activeTokens;
  const linearState = linear * (model.linear_state_bytes_per_layer ?? 0);
  return perLayer * (full * activeTokens + windowed * windowTokens) + linearState;
}

/**
 * Parameters (in billions) that stay at 16-bit regardless of the body quantisation:
 * the embedding table plus the output head (one shared table when tied).
 * Returns null when the model carries no embedding geometry — callers then fall back
 * to the legacy flat overhead factor.
 */
export function unquantisedParamsB(model: Model): number | null {
  if (!model.vocab_size || !model.hidden_size) return null;
  const tables = model.tied_embeddings ? 1 : 2;
  return (tables * model.vocab_size * model.hidden_size) / 1e9;
}

/** Output-head parameters (billions) — read in full on every decode step, unlike the embedding gather. */
function outputHeadParamsB(model: Model): number {
  if (!model.vocab_size || !model.hidden_size) return 0;
  return (model.vocab_size * model.hidden_size) / 1e9;
}

/**
 * Weights memory in GiB (all experts HBM-resident for MoE).
 *
 * weights = quantised_body × effective_bytes(quant) + unquantised_tail × 2
 *
 * The tail term is what a naive `params × bytes_per_param` misses. For Llama-3.3-70B the
 * fp16 embedding + lm_head pair is 2.1 B params = 3.9 GiB — >10% of an INT4 checkpoint,
 * far more than the 2% a flat factor allows for.
 */
export function weightsGb(model: Model, quant: Quant): number {
  // GGUF figures already include the (quantised) embedding layers — applying the fp16 tail
  // on top would count the embeddings twice.
  if (WHOLE_FILE_QUANTS.has(quant)) return paramBytesToGib(model.total_params_b, QB[quant]);
  const tail = unquantisedParamsB(model);
  if (tail === null) return paramBytesToGib(model.total_params_b * WEIGHT_OVERHEAD, QB[quant]);

  // Mixed-precision checkpoint: only part of the body is at `quant`, the declared dense
  // remainder stays at a higher precision. Three buckets instead of two.
  const denseQuant = model.mixed_precision?.[quant];
  const dense = model.dense_params_b ?? 0;
  if (denseQuant && dense > 0) {
    const quantised = Math.max(0, model.total_params_b - tail - dense);
    return (
      paramBytesToGib(quantised, QB[quant]) +
      paramBytesToGib(dense, QB[denseQuant]) +
      paramBytesToGib(tail, FP16_BYTES)
    );
  }

  const body = Math.max(0, model.total_params_b - tail);
  return paramBytesToGib(body, QB[quant]) + paramBytesToGib(tail, FP16_BYTES);
}

/** GiB streamed from HBM per decode step: active body at `quant` + the 16-bit output head. */
export function activeWeightsGb(model: Model, quant: Quant): number {
  if (WHOLE_FILE_QUANTS.has(quant)) return paramBytesToGib(model.active_params_b, QB[quant]);
  const head = outputHeadParamsB(model);
  if (head === 0) return paramBytesToGib(model.active_params_b * WEIGHT_OVERHEAD, QB[quant]);
  const body = Math.max(0, model.active_params_b - head);
  return paramBytesToGib(body, QB[quant]) + paramBytesToGib(head, FP16_BYTES);
}

/**
 * FLOPs to prefill one request of `tokens`, which is what TTFT is actually bounded by.
 *
 *   dense matmuls : 2 x active_params x tokens          (linear in sequence length)
 *   attention     : 4 x hidden x per-layer sequence work (QUADRATIC on full-attention layers)
 *
 * The attention term is not a rounding error. Llama-3.3-70B prefilling 78k tokens spends 16.2
 * PFLOPs on attention against 11.1 on the matmuls — long-context TTFT is an attention problem.
 * That also makes the layer regimes matter enormously: a sliding-window layer costs
 * tokens x window instead of tokens^2, and a linear layer is linear in tokens.
 */
export function prefillFlops(model: Model, tokens: number): number {
  const dense = 2 * model.active_params_b * 1e9 * tokens;
  const hidden = model.hidden_size;
  if (!hidden) return dense; // no geometry to size attention with; matmuls only
  const { full, windowed, linear } = layerSplit(model);
  const windowTokens = model.sliding_window ? Math.min(tokens, model.sliding_window) : tokens;
  const seqWork = full * tokens * tokens + windowed * tokens * windowTokens + linear * tokens;
  return dense + 4 * hidden * seqWork;
}

/**
 * Compute a full sizing (FR-10). Pure function of (model, gpu, input).
 * The same inputs against the same geometry recompute identically (AD-2a, FR-10).
 */
export function computeSizing(model: Model, gpu: GpuSku, input: SizingInput): Sizing {
  const {
    quant,
    kv_dtype_bytes,
    selected_ctx,
    avg_context_utilisation,
    target_concurrency,
    mem_util_fraction,
    gpus_per_node,
  } = input;

  const weights_gb = weightsGb(model, quant);
  const active_tokens = selected_ctx * avg_context_utilisation;
  const kv_per_request_gb = kvPerRequestBytes(model, kv_dtype_bytes, active_tokens) / GIB;
  // Reported per-token rate is the EFFECTIVE average, so kv_per_token × active_tokens always
  // reconciles with kv_per_request. For an all-full-attention model it equals the nominal rate;
  // with windowed layers it is lower, because most layers stopped growing at the window.
  const kv_per_token_gb = active_tokens > 0 ? kv_per_request_gb / active_tokens : 0;

  // TP selection: the TP size that needs the FEWEST TOTAL GPUs for the target concurrency.
  //
  // Picking the smallest TP that merely holds one request (the obvious rule) over-recommends
  // hardware, because a bigger shard leaves proportionally more room for KV and packs far more
  // sessions per replica. Llama-3.3-70B FP8 at 128K/conc-64 on H200: TP1 needs 16 GPUs, TP2
  // needs 10, TP4 and TP8 need 8. The smallest-that-fits rule answers 10; the true minimum is 8.
  //
  // Ties break toward the SMALLER shard — same GPU count, less collective traffic. That makes
  // this a cost/throughput objective; a latency-oriented planner would bias the other way.
  //
  // The reserve is evaluated PER CANDIDATE, not once up front: prefill activations shard with the
  // FFN, so usable memory per GPU is itself a function of the shard width. Hoisting it out would
  // charge every candidate the TP1 activation peak and bias the search toward wide shards for the
  // wrong reason.
  let tp: number | null = null;
  let free_gb = 0;
  let best_gpus = Infinity;
  let reserve = runtimeReserveGib(model, input.max_num_batched_tokens, 1);
  let usable_gb = gpu.mem_gb * mem_util_fraction - reserve.total;
  let any_room = false;
  for (const t of [...model.tp_options].sort((a, b) => a - b)) {
    const res = runtimeReserveGib(model, input.max_num_batched_tokens, t);
    const usable = gpu.mem_gb * mem_util_fraction - res.total;
    if (usable <= 0) continue; // the reserve alone exhausts the card at this shard width
    any_room = true;
    const f = t * usable - weights_gb;
    if (f < kv_per_request_gb) continue; // cannot hold weights + one request
    const conc = Math.max(1, Math.floor(f / kv_per_request_gb));
    const total_gpus = Math.ceil(Math.max(1, target_concurrency) / conc) * t;
    if (total_gpus < best_gpus) {
      best_gpus = total_gpus;
      tp = t;
      free_gb = f;
      reserve = res;
      usable_gb = usable;
    }
  }

  if (tp === null) {
    const largest = Math.max(...model.tp_options);
    // Two different failures, and conflating them sends the reader to the wrong lever: no room
    // for the weights is a quant/context/SKU problem, whereas a reserve that eats the whole card
    // is a prefill-chunk problem the caller can fix without changing the model at all.
    if (!any_room) {
      const worst = runtimeReserveGib(model, input.max_num_batched_tokens, largest);
      return {
        ok: false,
        reason:
          `The runtime reserve alone (${worst.total.toFixed(1)} GiB, of which ` +
          `${worst.activations.toFixed(1)} GiB is prefill activations) exceeds the ` +
          `${(gpu.mem_gb * mem_util_fraction).toFixed(1)} GiB this plan hands to vLLM. ` +
          `Lower --max-num-batched-tokens.`,
        weights_gb,
        kv_per_request_gb,
      };
    }
    return {
      ok: false,
      reason:
        `Weights + one request of KV do not fit even at TP ${largest}. ` +
        `Use a smaller quant, shorter context, or a larger-memory GPU.`,
      weights_gb,
      kv_per_request_gb,
    };
  }

  const concurrency_per_pod = Math.max(1, Math.floor(free_gb / kv_per_request_gb));
  const pods = Math.ceil(Math.max(1, target_concurrency) / concurrency_per_pod);
  const gpus = pods * tp;
  const nodes = Math.ceil(gpus / gpus_per_node);
  const multi_node = tp > gpus_per_node;

  // "Tight" band: it fits, but with no margin for modelling error. Measured on the pod as a
  // whole (weights are sharded across TP; one request of KV is the minimum working set).
  const pod_capacity_gb = tp * usable_gb;
  const headroom_fraction =
    (pod_capacity_gb - weights_gb - kv_per_request_gb) / pod_capacity_gb;
  const tight = headroom_fraction < TIGHT_HEADROOM;

  // Indicative decode throughput (±40%), addendum §A roofline. Memory is GiB and bandwidth is
  // decimal TB/s, so both sides are taken to raw bytes before dividing — mixing the scales here
  // silently inflated the KV term by 7.4% against the weight term.
  const active_gib = activeWeightsGb(model, quant);
  const active_per_replica = Math.min(
    concurrency_per_pod,
    Math.ceil(target_concurrency / pods),
  );
  const pod_bytes_per_sec = tp * gpu.bw_tbs * TBS_TO_BYTES_PER_SEC * MBU;
  const step_bytes = (active_gib + active_per_replica * kv_per_request_gb) * GIB;
  const step_time_sec = step_bytes / pod_bytes_per_sec;
  const throughput_tokens_per_sec = Math.round(
    (pods * active_per_replica) / step_time_sec,
  );
  // per-request decode rate = one token per step, from that request's share of the batch.
  const decode_tps_per_request = active_per_replica > 0 ? Math.round(1 / step_time_sec) : 0;
  // Indicative TTFT. Prefill is COMPUTE-bound: it must run the whole prompt through the network
  // before emitting a token. Sizing it as "stream the weights once" — the decode bound — under-
  // states a 78k-token prefill by three orders of magnitude. The weight-streaming time survives
  // only as a floor, for the short prompts where it genuinely dominates.
  const prefill_flops = prefillFlops(model, active_tokens);
  const weight_stream_sec = (active_gib * GIB) / pod_bytes_per_sec;
  let ttft_sec = weight_stream_sec;
  let ttft_compute_bound = false;
  if (gpu.tflops_fp16 && gpu.tflops_fp16 > 0) {
    const speedup = QB[quant] <= 1 ? LOW_PRECISION_SPEEDUP : 1;
    const pod_flops_per_sec = tp * gpu.tflops_fp16 * 1e12 * speedup * PREFILL_MFU;
    const compute_sec = prefill_flops / pod_flops_per_sec;
    ttft_compute_bound = compute_sec > weight_stream_sec;
    ttft_sec = Math.max(compute_sec, weight_stream_sec);
  }
  const ttft_ms = Math.round(ttft_sec * 1000);

  const result: FeasibleSizing = {
    ok: true,
    tp,
    weights_gb,
    kv_per_token_gb,
    kv_per_request_gb,
    usable_gb,
    free_gb,
    concurrency_per_pod,
    pods,
    gpus,
    nodes,
    multi_node,
    headroom_fraction,
    tight,
    weights_estimated: !WHOLE_FILE_QUANTS.has(quant) && unquantisedParamsB(model) === null,
    kv_windowed: hasWindowedLayers(model),
    runtime_reserve_gb: reserve.total,
    activation_gb: reserve.activations,
    throughput_tokens_per_sec,
    decode_tps_per_request,
    ttft_ms,
    ttft_compute_bound,
    prefill_pflops: prefill_flops / 1e15,
    step_time_ms: Math.round(step_time_sec * 1000 * 100) / 100,
    committed_bytes_per_gpu: committedBytesPerGpu(gpu),
  };
  return result;
}

/**
 * Physical whole-GPU HBM in integer bytes (addendum §G) — `mem_gb` is GiB, matching
 * nvidia-smi. The capacity gate (AD-10) compares committed vs available on integer bytes
 * so client and server agree exactly (AD-2c).
 */
export function committedBytesPerGpu(gpu: GpuSku): bigint {
  return BigInt(gpu.mem_gb) * BigInt(GIB);
}
