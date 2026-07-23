// Pinned Hugging Face config.json → §F Model mapping (AD-11). One place, so every caller maps
// identically. Params/tp/quants aren't in config.json → the admin completes them before commit.
import type { Model, Quant } from './types.js';
import { computeSizing } from './engine.js';
import type { GpuSku, SizingInput, Sizing } from './types.js';

/** HF `architectures` / `model_type` values that use Multi-head Latent Attention. */
const MLA_ARCHITECTURES = ['deepseek', 'kimi', 'mla'];

export interface HfConfig {
  architectures?: string[];
  model_type?: string;
  num_hidden_layers?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  head_dim?: number;
  hidden_size?: number;
  max_position_embeddings?: number;
  [k: string]: unknown;
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
  const arch = [cfg.model_type ?? '', ...(cfg.architectures ?? [])].join(' ').toLowerCase();
  const mla = MLA_ARCHITECTURES.some((a) => arch.includes(a));
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
  };
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
    if (!r.ok) return { concurrency: c, feasible: false, gpus: 0, pods: 0, tp: 0, decode_tps_per_request: 0, throughput_tokens_per_sec: 0, ttft_ms: 0 };
    return {
      concurrency: c, feasible: true, gpus: r.gpus, pods: r.pods, tp: r.tp,
      decode_tps_per_request: r.decode_tps_per_request, throughput_tokens_per_sec: r.throughput_tokens_per_sec, ttft_ms: r.ttft_ms,
    };
  });
}

export const HF_QUANT_HINT: Quant[] = ['FP16', 'FP8', 'INT4'];
