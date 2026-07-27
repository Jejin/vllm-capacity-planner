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
  max_position_embeddings?: number;
  sliding_window?: number | null;
  /** MLA marker: compressed KV projection rank. Present only on latent-attention models. */
  kv_lora_rank?: number;
  qk_rope_head_dim?: number;
  /** Per-layer attention kinds, newer transformers configs: 'full_attention' | 'sliding_attention'. */
  layer_types?: string[];
  /** Gemma-style: one global layer every N (so N-1 of every N are windowed). */
  sliding_window_pattern?: number;
  [k: string]: unknown;
}

/**
 * Resolve how many layers keep full context, across the three ways configs express it.
 * Returns null when the model has no usable sliding-window declaration.
 */
export function fullAttentionLayers(cfg: HfConfig, layers: number | undefined): number | null {
  if (Array.isArray(cfg.layer_types) && cfg.layer_types.length > 0) {
    return cfg.layer_types.filter((t) => t === 'full_attention').length;
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
  };
  // local/global attention — only set when the config actually declares a window, since
  // an unset pair means "treat every layer as full-context"
  const fullAttn = fullAttentionLayers(cfg, cfg.num_hidden_layers);
  if (cfg.sliding_window && fullAttn !== null) {
    mapped.sliding_window = cfg.sliding_window;
    mapped.full_attention_layers = fullAttn;
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
