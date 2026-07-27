// Canonical domain types — addendum §F field names (AD-13). No framework imports (AD-1).

// GPU-served formats first, then the GGUF family (llama.cpp / Ollama; vLLM's GGUF support is
// experimental). GGUF bytes/param are whole-FILE averages that already include the embedding
// layers, so they skip the un-quantised-tail term — see WHOLE_FILE_QUANTS in engine.ts.
export const QUANTS = ['FP16', 'FP8', 'INT8', 'INT4', 'MXFP4', 'NVFP4', 'Q8_0', 'Q4_K_M', 'IQ4_XS'] as const;
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
  // --- embedding geometry (optional; drives the un-quantised-tail weight term) ---
  // Real checkpoints keep the embedding table and lm_head at 16-bit even when the
  // transformer body is quantised. At INT4/MXFP4 that tail is a double-digit share of
  // the footprint, so omitting it under-sizes the deployment. All three come straight
  // from HF config.json (hidden_size / vocab_size / tie_word_embeddings). When absent
  // the engine falls back to the legacy flat WEIGHT_OVERHEAD factor.
  hidden_size?: number;
  vocab_size?: number;
  tied_embeddings?: boolean; // true => one shared table, false/undefined => embedding + lm_head
  // --- local/global attention (optional; caps KV on the windowed layers) ---
  // Many models (GPT-OSS, Gemma, Mistral-v0.1) run most layers over a fixed sliding window
  // instead of the full context. Those layers' KV stops growing at the window, cutting
  // long-context KV several-fold. Absent => every layer treated as full-context.
  // From config.json: `sliding_window` + `layer_types` / `sliding_window_pattern`.
  sliding_window?: number; // window length in tokens
  full_attention_layers?: number; // layers using full context; the rest use the window
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
  /** Fraction of pod HBM still free once weights + ONE request of KV are placed (0..1). */
  headroom_fraction: number;
  /** headroom_fraction < TIGHT_HEADROOM — it fits, but with no margin for modelling error. */
  tight: boolean;
  /** True when the weight estimate used the legacy flat overhead (no embedding geometry on the model). */
  weights_estimated: boolean;
  /** True when some layers are sliding-window, so KV is below the all-full-attention figure. */
  kv_windowed: boolean;
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
