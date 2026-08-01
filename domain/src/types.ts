// Canonical domain types — addendum §F field names (AD-13). No framework imports (AD-1).

// GPU-served formats first, then the GGUF family (llama.cpp / Ollama; vLLM's GGUF support is
// experimental). GGUF bytes/param are whole-FILE averages that already include the embedding
// layers, so they skip the un-quantised-tail term — see WHOLE_FILE_QUANTS in engine.ts.
export const QUANTS = ['FP16', 'FP8', 'INT8', 'INT4', 'MXFP4', 'NVFP4', 'Q8_0', 'Q4_K_M', 'IQ4_XS'] as const;
export type Quant = (typeof QUANTS)[number];

/**
 * How a given precision is actually obtained at launch. A bytes-per-parameter figure is enough
 * to size memory and not enough to start a server: INT4 is a different repository, FP8 is
 * sometimes a repository and sometimes a runtime flag, and MXFP4 is baked into the checkpoint.
 *   checkpoint — the artifact already carries this precision; passing --quantization as well
 *                conflicts with the checkpoint's own metadata
 *   online     — an unquantised checkpoint plus a vLLM --quantization method applied at load
 *   none       — the checkpoint's native dtype already matches the estimate
 */
export type QuantSource = 'checkpoint' | 'online' | 'none';

/** The deployment path for ONE precision of one model (§4.2 of the implementation handoff). */
export interface DeploymentVariant {
  source: QuantSource;
  /** Repository carrying this precision. Absent => the model's base `hf_id`. */
  hf_id?: string;
  /** vLLM `--quantization` value. Required for `online`; must be absent otherwise. */
  method?: string;
  /** Pinned tag or commit, where reproducibility matters. */
  revision?: string;
  /**
   * Where this artifact's FP8 KV cache scaling factors come from (§4.3). Only consulted when a
   * plan selects a 1-byte KV cache. Absent => `unknown`, which reads as a warning rather than
   * as approval: vLLM's default with no scales present is to set them all to 1.0.
   */
  kv_scale_source?: 'checkpoint' | 'calibrated' | 'runtime' | 'none';
}

/** A servable model, by sizing-relevant geometry (addendum §F.1). */
export interface Model {
  id: string;
  /** Display label. UI only — it must never reach a command line. */
  name: string;
  // --- deployment identity ---
  // The repository a launch command resolves. `name` is a marketing label ("Llama 3.3 70B
  // Instruct"); passing it to `vllm serve` fails at argument parsing, never mind resolution.
  // Absent, no command can be generated at all — which is the honest outcome, since there is
  // nothing to serve.
  hf_id?: string;
  revision?: string;
  /**
   * Per-precision deployment path. A quant with no entry has no known way to launch, so the
   * command is blocked rather than guessed: sizing a plan at INT4 and emitting a command that
   * silently launches BF16 is worse than emitting nothing.
   */
  deployments?: Partial<Record<Quant, DeploymentVariant>>;
  total_params_b: number;
  active_params_b: number;
  layers: number;
  kv_heads: number; // mla=false => >0 ; mla=true => 0 (unused)
  head_dim: number; // mla=false => >0 ; mla=true => 0 (unused)
  mla: boolean;
  max_ctx: number;
  tp_options: number[];
  quants: Quant[];
  // --- MLA latent geometry (optional; MLA models only) ---
  // An MLA model caches ONE compressed latent per layer per token instead of a K and a V
  // tensor, so its per-layer KV width is `kv_lora_rank + qk_rope_head_dim` from config.json.
  // That is model geometry, not a planner constant: it belongs to the checkpoint the same way
  // kv_heads x head_dim belongs to a GQA one, and a GQA model has no latent at all.
  // Every MLA checkpoint currently catalogued happens to use DeepSeek's 512 + 64 = 576 (Kimi
  // and GLM-5.2 inherited the shape), and DEFAULT_MLA_LATENT_ELEMS is that fallback when a
  // model declares mla but not its width. Leave unset on GQA models.
  mla_latent_elems?: number;
  // --- embedding geometry (optional; drives the un-quantised-tail weight term) ---
  // Real checkpoints keep the embedding table and lm_head at 16-bit even when the
  // transformer body is quantised. At INT4/MXFP4 that tail is a double-digit share of
  // the footprint, so omitting it under-sizes the deployment. All three come straight
  // from HF config.json (hidden_size / vocab_size / tie_word_embeddings). When absent
  // the engine falls back to the legacy flat WEIGHT_OVERHEAD factor.
  hidden_size?: number;
  vocab_size?: number;
  tied_embeddings?: boolean; // true => one shared table, false/undefined => embedding + lm_head
  // --- FFN width (optional; drives the sharded half of the prefill activation reserve) ---
  // The FFN width ONE token passes through in ONE layer:
  //   dense : config.json `intermediate_size`
  //   MoE   : `moe_intermediate_size` x `num_experts_per_tok`, because a routed token
  //           materialises activations inside every expert it is sent to
  // Absent, the engine assumes DEFAULT_INTERMEDIATE_RATIO x hidden_size, which over-reserves for
  // MoE (whose per-token width is typically far narrower than a dense FFN of the same hidden).
  intermediate_size?: number;
  // --- local/global attention (optional; caps KV on the windowed layers) ---
  // Many models (GPT-OSS, Gemma, Mistral-v0.1) run most layers over a fixed sliding window
  // instead of the full context. Those layers' KV stops growing at the window, cutting
  // long-context KV several-fold. Absent => every layer treated as full-context.
  // From config.json: `sliding_window` + `layer_types` / `sliding_window_pattern`.
  sliding_window?: number; // window length in tokens
  full_attention_layers?: number; // layers using full context; the rest are windowed or linear
  // --- linear / recurrent attention (optional) ---
  // Hybrid models (Kimi K3's KDA, Qwen3-Next, MiniMax) replace most attention layers with a
  // recurrent form whose state is CONSTANT in sequence length — it never accumulates a
  // per-token cache. Sizing those layers as full attention massively overstates KV.
  // Layer split is: full_attention_layers + linear_attention_layers + windowed = layers.
  linear_attention_layers?: number;
  linear_state_bytes_per_layer?: number; // fixed state per layer per request, in bytes
  // --- mixed-precision checkpoints (optional) ---
  // Frontier low-bit checkpoints are rarely uniform. NVIDIA ModelOpt's GLM-5.2 NVFP4 quantises
  // "only MoE expert linears"; DeepSeek-V4 ships "MoE experts FP4, remaining params FP8". A
  // single bytes/param cannot express that, so a model may declare how much of it is NOT in the
  // quantisable block, and which precision that remainder sits at for a given quant.
  //   dense_params_b: attention + shared experts + router + dense MLP, EXCLUDING the embedding
  //                   tail (that is already handled by hidden_size/vocab_size).
  //   mixed_precision: quant -> precision the dense remainder keeps under that quant.
  dense_params_b?: number;
  mixed_precision?: Partial<Record<Quant, Quant>>;
}

/**
 * GPU architecture, as far as kernel support is concerned. Not a marketing generation: consumer
 * Blackwell (GB202, SM 12.0) and datacentre Blackwell (SM 10.0) are separate entries because they
 * differ in NVLink and in which kernels vLLM selects.
 */
export const GPU_ARCHES = ['ampere', 'ada', 'hopper', 'blackwell', 'blackwell-consumer', 'cdna3', 'cdna4'] as const;
export type GpuArch = (typeof GPU_ARCHES)[number];

/** A GPU type (addendum §F.2). */
export interface GpuSku {
  id: string;
  name: string;
  mem_gb: number;
  bw_tbs: number; // per-GPU HBM aggregate bandwidth (TB/s)
  /**
   * What kernels this card can run. Drives the runtime-support verdict, which is a separate
   * question from whether the plan fits in HBM. Absent => support reports `unverified`.
   */
  arch?: GpuArch;
  /**
   * Dense (non-sparse) FP16 tensor throughput in TFLOPS. Drives the prefill/TTFT estimate,
   * which is compute-bound — bandwidth alone under-states it by orders of magnitude at long
   * context. Absent => TTFT falls back to the bandwidth floor and is flagged as such.
   */
  tflops_fp16?: number;
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
  /** vLLM --max-num-batched-tokens: the prefill chunk. Drives the activation reserve. */
  max_num_batched_tokens?: number;
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
  /** Per-GPU runtime reserve actually applied (GiB): CUDA context floor + prefill activations. */
  runtime_reserve_gb: number;
  /** Prefill activation component of that reserve (GiB) — scales with the batched-token chunk. */
  activation_gb: number;
  throughput_tokens_per_sec: number; // aggregate decode throughput across the deployment, ±40%
  decode_tps_per_request: number; // per-request decode tokens/sec (1 / step time), ±40%
  ttft_ms: number; // indicative time-to-first-token (prefill), ±50%
  /** True when TTFT is set by prefill arithmetic rather than by streaming the weights once. */
  ttft_compute_bound: boolean;
  /** Total prefill work for one request, in PFLOPs — the thing TTFT is actually bounded by. */
  prefill_pflops: number;
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
