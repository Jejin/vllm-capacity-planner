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

export const RUNTIME_GB = 2.5; // per-GPU runtime reserve, GiB (CUDA graphs, activation buffers)
/** Legacy flat overhead factor — used ONLY when a model carries no embedding geometry. */
export const WEIGHT_OVERHEAD = 1.02;
export const MBU = 0.55; // model bandwidth utilisation (decode roofline)
export const MLA_ELEMS_PER_LAYER = 576; // MLA latent size per layer/token
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
};

/** KV cache bytes per token — GQA vs MLA (addendum §A). */
export function kvPerTokenBytes(model: Model, kvDtypeBytes: number): number {
  return model.mla
    ? model.layers * MLA_ELEMS_PER_LAYER * kvDtypeBytes
    : 2 * model.layers * model.kv_heads * model.head_dim * kvDtypeBytes;
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
  const tail = unquantisedParamsB(model);
  if (tail === null) return paramBytesToGib(model.total_params_b * WEIGHT_OVERHEAD, QB[quant]);
  const body = Math.max(0, model.total_params_b - tail);
  return paramBytesToGib(body, QB[quant]) + paramBytesToGib(tail, FP16_BYTES);
}

/** GiB streamed from HBM per decode step: active body at `quant` + the 16-bit output head. */
export function activeWeightsGb(model: Model, quant: Quant): number {
  const head = outputHeadParamsB(model);
  if (head === 0) return paramBytesToGib(model.active_params_b * WEIGHT_OVERHEAD, QB[quant]);
  const body = Math.max(0, model.active_params_b - head);
  return paramBytesToGib(body, QB[quant]) + paramBytesToGib(head, FP16_BYTES);
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
  const kv_per_token_gb = kvPerTokenBytes(model, kv_dtype_bytes) / GIB;
  const kv_per_request_gb = kv_per_token_gb * selected_ctx * avg_context_utilisation;
  const usable_gb = gpu.mem_gb * mem_util_fraction - RUNTIME_GB;

  // TP selection: smallest tp such that tp*usable - weights >= kv_per_request (FR-13 if none).
  let tp: number | null = null;
  let free_gb = 0;
  for (const t of [...model.tp_options].sort((a, b) => a - b)) {
    const f = t * usable_gb - weights_gb;
    if (f >= kv_per_request_gb) {
      tp = t;
      free_gb = f;
      break;
    }
  }

  if (tp === null) {
    const largest = Math.max(...model.tp_options);
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
  // Indicative TTFT (prefill): bandwidth floor to stream the active weights once before the
  // first token. Compute-bound prefill needs FLOPS for precision (out of Phase-1 scope) — ±50%.
  const ttft_ms = Math.round(((active_gib * GIB) / pod_bytes_per_sec) * 1000);

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
    weights_estimated: unquantisedParamsB(model) === null,
    throughput_tokens_per_sec,
    decode_tps_per_request,
    ttft_ms,
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
