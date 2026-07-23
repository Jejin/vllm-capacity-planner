// Canonical domain types — addendum §F field names (AD-13). No framework imports (AD-1).

export const QUANTS = ['FP16', 'FP8', 'INT8', 'INT4', 'MXFP4', 'NVFP4'] as const;
export type Quant = (typeof QUANTS)[number];

/** A servable model, by sizing-relevant geometry (addendum §F.1). */
export interface Model {
  id: string;
  name: string;
  total_params_b: number;
  active_params_b: number;
  layers: number;
  kv_heads: number; // mla=false => >0 ; mla=true => 0 (unused)
  head_dim: number; // mla=false => >0 ; mla=true => 0 (unused)
  mla: boolean;
  max_ctx: number;
  tp_options: number[];
  quants: Quant[];
}

/** A GPU type (addendum §F.2). */
export interface GpuSku {
  id: string;
  name: string;
  mem_gb: number;
  bw_tbs: number; // per-GPU HBM aggregate bandwidth (TB/s)
  price_per_gpu_hour?: number; // admin-set rental rate ($/GPU-hour) for cost estimates
}

/** The nine sizing inputs (addendum §F.3 / FR-9). */
export interface SizingInput {
  quant: Quant;
  kv_dtype_bytes: number; // 2 = FP16/BF16, 1 = FP8
  selected_ctx: number; // constrained <= model.max_ctx
  avg_context_utilisation: number; // 0 < v <= 1
  target_concurrency: number;
  mem_util_fraction: number; // 0 < v <= 1
  gpus_per_node: number;
}

export interface FeasibleSizing {
  ok: true;
  tp: number;
  weights_gb: number;
  kv_per_token_gb: number;
  kv_per_request_gb: number;
  usable_gb: number; // per-GPU usable HBM
  free_gb: number; // free KV budget per replica
  concurrency_per_pod: number;
  pods: number;
  gpus: number;
  nodes: number;
  multi_node: boolean;
  throughput_tokens_per_sec: number; // aggregate decode throughput across the deployment, ±40%
  decode_tps_per_request: number; // per-request decode tokens/sec (1 / step time), ±40%
  ttft_ms: number; // indicative time-to-first-token (prefill, bandwidth floor), ±50%
  step_time_ms: number; // per-decode-step time for the in-flight batch
  /** integer bytes per GPU committed by this replica's weights+KV — used by the capacity gate (AD-2c). */
  committed_bytes_per_gpu: bigint;
}

export interface InfeasibleSizing {
  ok: false;
  reason: string;
  weights_gb: number;
  kv_per_request_gb: number;
}

export type Sizing = FeasibleSizing | InfeasibleSizing;

/** A model + GPU pair with just the geometry the engine needs (what a saved config snapshots, AD-4). */
export interface CatalogGeometry {
  model: Model;
  gpu: GpuSku;
}
