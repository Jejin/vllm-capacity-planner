// Sizing engine — the calculation contract (addendum §A). Pure functions only (AD-1).
// Every constant is from addendum §A and verified against the §C acceptance vectors.

import type {
  Model,
  GpuSku,
  Quant,
  SizingInput,
  Sizing,
  FeasibleSizing,
} from './types.js';

export const RUNTIME_GB = 2.5; // per-GPU runtime reserve (CUDA graphs, activation buffers)
export const WEIGHT_OVERHEAD = 1.02; // non-expert overhead factor
export const MBU = 0.55; // model bandwidth utilisation (decode roofline)
export const MLA_ELEMS_PER_LAYER = 576; // MLA latent size per layer/token
const GB = 2 ** 30;

/** Bytes per parameter by quantisation (addendum §A). */
export const QB: Record<Quant, number> = {
  FP16: 2,
  FP8: 1,
  INT8: 1,
  INT4: 0.5,
  MXFP4: 0.5,
  NVFP4: 0.625,
};

/** KV cache bytes per token — GQA vs MLA (addendum §A). */
export function kvPerTokenBytes(model: Model, kvDtypeBytes: number): number {
  return model.mla
    ? model.layers * MLA_ELEMS_PER_LAYER * kvDtypeBytes
    : 2 * model.layers * model.kv_heads * model.head_dim * kvDtypeBytes;
}

/** Weights memory in GB (all experts HBM-resident for MoE). */
export function weightsGb(model: Model, quant: Quant): number {
  return model.total_params_b * QB[quant] * WEIGHT_OVERHEAD;
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
  const kv_per_token_gb = kvPerTokenBytes(model, kv_dtype_bytes) / GB;
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

  // Indicative decode throughput (±40%), addendum §A roofline.
  const active_bytes = model.active_params_b * QB[quant] * WEIGHT_OVERHEAD;
  const active_per_replica = Math.min(
    concurrency_per_pod,
    Math.ceil(target_concurrency / pods),
  );
  const step_time_sec =
    (active_bytes + active_per_replica * kv_per_request_gb) /
    (tp * gpu.bw_tbs * 1000 * MBU);
  const throughput_tokens_per_sec = Math.round(
    (pods * active_per_replica) / step_time_sec,
  );
  // per-request decode rate = one token per step, from that request's share of the batch.
  const decode_tps_per_request = active_per_replica > 0 ? Math.round(1 / step_time_sec) : 0;
  // Indicative TTFT (prefill): bandwidth floor to stream the active weights once before the
  // first token. Compute-bound prefill needs FLOPS for precision (out of Phase-1 scope) — ±50%.
  const ttft_ms = Math.round((active_bytes / (tp * gpu.bw_tbs * 1000 * MBU)) * 1000);

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
    throughput_tokens_per_sec,
    decode_tps_per_request,
    ttft_ms,
    step_time_ms: Math.round(step_time_sec * 1000 * 100) / 100,
    committed_bytes_per_gpu: committedBytesPerGpu(gpu),
  };
  return result;
}

/**
 * Physical whole-GPU HBM in integer bytes (addendum §G). The capacity gate (AD-10)
 * compares committed vs available on integer bytes so client and server agree exactly (AD-2c).
 */
export function committedBytesPerGpu(gpu: GpuSku): bigint {
  return BigInt(gpu.mem_gb) * BigInt(GB);
}
