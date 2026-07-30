// Pinned Hugging Face config.json → §F Model mapping (AD-11). One place, so every caller maps
// identically. Params/tp/quants aren't in config.json → the admin completes them before commit.
import type { Model, Quant } from './types.js';
import { computeSizing } from './engine.js';
import type { GpuSku, SizingInput, Sizing } from './types.js';

/** HF `architectures` / `model_type` values that use Multi-head Latent Attention. */
const MLA_ARCHITECTURES = ['deepseek', 'kimi', 'mla'];

/**
 * Structural MLA detection. Name matching alone misses models whose architecture string
 * doesn't say "mla" — GLM-5.2 ships as `GlmMoeDsaForCausalLM` yet is squarely MLA. The
 * reliable signal is the compressed KV projection: `kv_lora_rank` only exists on MLA models,
 * and `kv_lora_rank + qk_rope_head_dim` is the per-layer latent width (576 for DeepSeek,
 * Kimi and GLM-5.2 alike).
 */
/**
 * Linear-attention layer count and per-layer state, for hybrid models.
 * Kimi K3 lists both sides explicitly (`full_attn_layers` / `kda_layers`); the recurrent state
 * is one [head_dim x head_dim] matrix per head, held at fp32 for numerical stability.
 */
export function linearAttention(cfg: HfConfig, layers: number | undefined):
  { linear_attention_layers: number; linear_state_bytes_per_layer: number; full_attention_layers: number } | null {
  // --- Kimi spelling: nested linear_attn_config enumerating both layer sets ---
  const lac = cfg.linear_attn_config;
  if (lac) {
    const full = lac.full_attn_layers?.length;
    const linear = lac.kda_layers?.length ?? (full != null && layers != null ? layers - full : undefined);
    if (full != null && linear != null && linear > 0 && lac.num_heads && lac.head_dim) {
      return {
        full_attention_layers: full,
        linear_attention_layers: linear,
        // recurrent state is one [head_dim x head_dim] matrix per head, fp32 for stability
        linear_state_bytes_per_layer: lac.num_heads * lac.head_dim * lac.head_dim * 4,
      };
    }
  }

  // --- Qwen spelling: 'linear_attention' in layer_types + flat linear_* dimensions ---
  if (Array.isArray(cfg.layer_types) && cfg.layer_types.length > 0) {
    const { full, linear } = bucketLayerTypes(cfg.layer_types);
    const vHeads = cfg.linear_num_value_heads;
    const kDim = cfg.linear_key_head_dim;
    const vDim = cfg.linear_value_head_dim;
    if (linear > 0 && vHeads && kDim && vDim) {
      return {
        full_attention_layers: full,
        linear_attention_layers: linear,
        // gated delta-net state: [v_heads x k_head_dim x v_head_dim], fp32
        linear_state_bytes_per_layer: vHeads * kDim * vDim * 4,
      };
    }
  }
  return null;
}

export function detectMla(cfg: HfConfig): boolean {
  const arch = [cfg.model_type ?? '', ...(cfg.architectures ?? [])].join(' ').toLowerCase();
  return MLA_ARCHITECTURES.some((a) => arch.includes(a)) || cfg.kv_lora_rank != null;
}

export interface HfConfig {
  architectures?: string[];
  model_type?: string;
  num_hidden_layers?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  head_dim?: number;
  hidden_size?: number;
  vocab_size?: number;
  tie_word_embeddings?: boolean;
  /** Dense FFN width. On an MoE config this is the width of the DENSE layers only. */
  intermediate_size?: number;
  /** MoE: per-expert FFN width. Narrower than intermediate_size, often by an order of magnitude. */
  moe_intermediate_size?: number;
  /** MoE: experts each token is routed to (top-k). */
  num_experts_per_tok?: number;
  /** Kimi's spelling of the same field. */
  num_experts_per_token?: number;
  /** DeepSeek/GLM: COUNT of always-on shared experts, each moe_intermediate_size wide. */
  n_shared_experts?: number;
  /** Kimi's spelling of the same count. */
  num_shared_experts?: number;
  /** Qwen2-MoE style: the shared expert's WIDTH directly, rather than a count. */
  shared_expert_intermediate_size?: number;
  max_position_embeddings?: number;
  sliding_window?: number | null;
  /** MLA marker: compressed KV projection rank. Present only on latent-attention models. */
  kv_lora_rank?: number;
  qk_rope_head_dim?: number;
  /** Per-layer attention kinds, newer transformers configs: 'full_attention' | 'sliding_attention'. */
  layer_types?: string[];
  /** Gemma-style: one global layer every N (so N-1 of every N are windowed). */
  sliding_window_pattern?: number;
  /** Hybrid linear attention, Kimi spelling: nested config listing both layer sets. */
  linear_attn_config?: {
    full_attn_layers?: number[];
    kda_layers?: number[];
    num_heads?: number;
    head_dim?: number;
  };
  /** Hybrid linear attention, Qwen spelling: flat keys + 'linear_attention' in layer_types. */
  linear_num_key_heads?: number;
  linear_num_value_heads?: number;
  linear_key_head_dim?: number;
  linear_value_head_dim?: number;
  full_attention_interval?: number;
  [k: string]: unknown;
}

/**
 * Resolve how many layers keep full context, across the three ways configs express it.
 * Returns null when the model has no usable sliding-window declaration.
 */
/**
 * Bucket a `layer_types` array into the three regimes. Configs use different vocabularies —
 * GPT-OSS says 'sliding_attention', Qwen3.6 says 'linear_attention' — and anything that is not
 * recognisably windowed or linear is treated as full attention, which is the safe default.
 */
export function bucketLayerTypes(types: string[]): { full: number; sliding: number; linear: number } {
  let full = 0, sliding = 0, linear = 0;
  for (const t of types) {
    const k = t.toLowerCase();
    if (k.includes('linear') || k.includes('mamba') || k.includes('recurrent')) linear++;
    else if (k.includes('sliding') || k.includes('local')) sliding++;
    else full++;
  }
  return { full, sliding, linear };
}

export function fullAttentionLayers(cfg: HfConfig, layers: number | undefined): number | null {
  if (Array.isArray(cfg.layer_types) && cfg.layer_types.length > 0) {
    return bucketLayerTypes(cfg.layer_types).full;
  }
  if (!cfg.sliding_window) return null; // no window at all => every layer is full-context
  if (cfg.sliding_window_pattern && cfg.sliding_window_pattern > 0 && layers) {
    return Math.ceil(layers / cfg.sliding_window_pattern); // 1 global every N
  }
  return 0; // Mistral-v0.1 style: a window with no pattern means every layer is windowed
}

export interface HfMapResult {
  /** Best-effort §F fields mapped from config.json (partial — the admin completes the rest). */
  mapped: Partial<Model>;
  /** §F fields the card does not carry — the admin MUST supply these before commit. */
  missing: (keyof Model)[];
  detectedMla: boolean;
}

function slug(id: string): string {
  return id.split('/').pop()!.toLowerCase().replace(/[^a-z0-9-]/g, '-').slice(0, 64);
}

/**
 * FFN width one token traverses in one layer.
 *
 * On an MoE config, `intermediate_size` describes the dense layers and is the WRONG number to
 * reserve activations against — a token visits `num_experts_per_tok` routed experts of
 * `moe_intermediate_size` each. Prefer the MoE fields whenever the config carries them.
 *
 * Shared experts ARE counted. They run on every token, not a routed subset, so they add their
 * full width to every token's path: at DeepSeek-V3's 1 shared against top-8 that is an eighth of
 * the total, and at Kimi K3's 2 against top-16 the same again — not the rounding error an earlier
 * version of this assumed. Counting them gives DeepSeek-V3 9 x 2048 = 18432, which is exactly the
 * `intermediate_size` its own dense layers use; that the two agree is a good sign the model is
 * right rather than a coincidence.
 *
 * Field spellings differ between families, so both are read: `num_experts_per_tok` and Kimi's
 * `num_experts_per_token`; `n_shared_experts` / `num_shared_experts` as a COUNT, or Qwen2-MoE's
 * `shared_expert_intermediate_size` as a width.
 */
export function perTokenFfnWidth(cfg: HfConfig): number | undefined {
  const expert = cfg.moe_intermediate_size;
  const topK = cfg.num_experts_per_tok ?? cfg.num_experts_per_token;
  if (!expert || !topK) return cfg.intermediate_size;
  const sharedCount = cfg.n_shared_experts ?? cfg.num_shared_experts ?? 0;
  const shared = cfg.shared_expert_intermediate_size ?? sharedCount * expert;
  return expert * topK + shared;
}

/** Map a HF model id + its config.json to a partial §F Model (AD-11). */
export function hfConfigToModel(id: string, cfg: HfConfig): HfMapResult {
  const mla = detectMla(cfg);
  const heads = cfg.num_attention_heads;
  const head_dim = cfg.head_dim ?? (cfg.hidden_size && heads ? Math.round(cfg.hidden_size / heads) : undefined);
  const kv_heads = cfg.num_key_value_heads ?? heads; // GQA-correct: KV heads, fall back to attention heads

  const mapped: Partial<Model> = {
    id: slug(id),
    name: id,
    layers: cfg.num_hidden_layers,
    mla,
    kv_heads: mla ? 0 : kv_heads,
    head_dim: mla ? 0 : head_dim,
    max_ctx: cfg.max_position_embeddings,
    // embedding geometry — config.json carries all three, so low-bit weight estimates
    // get the 16-bit embedding/lm_head tail for free on import
    hidden_size: cfg.hidden_size,
    vocab_size: cfg.vocab_size,
    tied_embeddings: cfg.tie_word_embeddings ?? false,
    // FFN width per token per layer — what the prefill activation reserve shards
    intermediate_size: perTokenFfnWidth(cfg),
  };
  // local/global attention — only set when the config actually declares a window, since
  // an unset pair means "treat every layer as full-context"
  const fullAttn = fullAttentionLayers(cfg, cfg.num_hidden_layers);
  if (cfg.sliding_window && fullAttn !== null) {
    mapped.sliding_window = cfg.sliding_window;
    mapped.full_attention_layers = fullAttn;
  }
  // hybrid linear attention — overrides the full-layer count, since linear_attn_config
  // enumerates the split directly
  const lin = linearAttention(cfg, cfg.num_hidden_layers);
  if (lin) {
    mapped.full_attention_layers = lin.full_attention_layers;
    mapped.linear_attention_layers = lin.linear_attention_layers;
    mapped.linear_state_bytes_per_layer = lin.linear_state_bytes_per_layer;
  }
  // params + tp + quants are never in config.json — admin-supplied
  const missing: (keyof Model)[] = ['total_params_b', 'active_params_b', 'tp_options', 'quants'];
  if (!mla && (!kv_heads || !head_dim)) missing.push('kv_heads', 'head_dim');
  if (!cfg.num_hidden_layers) missing.push('layers');
  if (!cfg.max_position_embeddings) missing.push('max_ctx');
  return { mapped, missing: [...new Set(missing)], detectedMla: mla };
}

/** Concurrency rubric: sweep target concurrency, returning the sizing metrics at each level. */
export interface SweepRow {
  concurrency: number;
  feasible: boolean;
  tight: boolean; // feasible, but <10% pod headroom — see engine.TIGHT_HEADROOM
  gpus: number;
  pods: number;
  tp: number;
  decode_tps_per_request: number;
  throughput_tokens_per_sec: number;
  ttft_ms: number;
}
export function concurrencySweep(
  model: Model,
  gpu: GpuSku,
  input: Omit<SizingInput, 'target_concurrency'>,
  concurrencies: number[],
): SweepRow[] {
  return concurrencies.map((c) => {
    const r: Sizing = computeSizing(model, gpu, { ...input, target_concurrency: c });
    if (!r.ok) return { concurrency: c, feasible: false, tight: false, gpus: 0, pods: 0, tp: 0, decode_tps_per_request: 0, throughput_tokens_per_sec: 0, ttft_ms: 0 };
    return {
      concurrency: c, feasible: true, tight: r.tight, gpus: r.gpus, pods: r.pods, tp: r.tp,
      decode_tps_per_request: r.decode_tps_per_request, throughput_tokens_per_sec: r.throughput_tokens_per_sec, ttft_ms: r.ttft_ms,
    };
  });
}

export const HF_QUANT_HINT: Quant[] = ['FP16', 'FP8', 'INT4'];
