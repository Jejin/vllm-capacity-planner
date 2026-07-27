// vLLM recipes (recipes.vllm.ai) → §F Model mapping.
//
// This is the complement to hf.ts, not a replacement. A Hugging Face config.json carries the
// GEOMETRY (layers, heads, embedding sizes, attention regime) but never the commercial facts:
// parameter counts, which quantised checkpoints exist, or what TP sizes people actually run.
// The recipe JSON carries exactly those four — which is precisely the set hf.ts reports as
// `missing`. Import both and a model arrives complete.

import type { Model, Quant } from './types.js';

/** The subset of a recipe document we consume. Everything is optional — recipes vary. */
export interface RecipeDoc {
  hf_id?: string;
  meta?: { title?: string; provider?: string; description?: string; date_updated?: string };
  model?: {
    parameter_count?: string; // "36B", "2.8T", "1T"
    active_parameters?: string; // same, or prose for exotic MoE
    context_length?: number;
    min_vllm_version?: string;
  };
  variants?: Record<string, RecipeVariant>;
}

export interface RecipeVariant {
  precision?: string;
  vram_minimum_gb?: number;
  description?: string;
  extra_args?: string[];
}

/** Recipe precision labels → our Quant. Unknown labels are skipped rather than guessed. */
const PRECISION_TO_QUANT: Record<string, Quant> = {
  bf16: 'FP16',
  fp16: 'FP16',
  fp8: 'FP8',
  mxfp8: 'FP8',
  int8: 'INT8',
  int4: 'INT4',
  awq: 'INT4',
  gptq: 'INT4',
  nvfp4: 'NVFP4',
  fp4: 'NVFP4',
  mxfp4: 'MXFP4',
};

/**
 * Parse a recipe parameter count ("36B", "2.8T", "480B") to billions.
 * Returns null for prose the recipes sometimes put in `active_parameters`, e.g.
 * "16 experts/token + shared (of 896 routed)" — better to leave it for the admin than guess.
 */
export function parseParamCount(s: string | undefined): number | null {
  if (!s) return null;
  const m = /^\s*([\d.]+)\s*([BTM])\b/i.exec(s);
  if (!m) return null;
  const v = Number(m[1]);
  if (!Number.isFinite(v) || v <= 0) return null;
  const unit = m[2].toUpperCase();
  return unit === 'T' ? v * 1000 : unit === 'M' ? v / 1000 : v;
}

/** Pull `--tensor-parallel-size N` out of a variant's extra args. */
export function tpFromArgs(args: string[] | undefined): number | null {
  if (!args) return null;
  const i = args.indexOf('--tensor-parallel-size');
  if (i < 0 || i + 1 >= args.length) return null;
  const n = Number(args[i + 1]);
  return Number.isInteger(n) && n > 0 ? n : null;
}

export interface RecipeMapResult {
  /** §F fields the recipe can supply. */
  mapped: Partial<Model>;
  /** Which fields actually came from the recipe (for UI highlighting). */
  filled: (keyof Model)[];
  /** Per-precision VRAM floors the recipe states — a cross-check against our own estimate. */
  vram_minimums: { quant: Quant; precision: string; vram_gb: number; description?: string }[];
  min_vllm_version?: string;
  /** Precisions we saw but have no Quant for (e.g. mixed "int2/4/8" checkpoints). */
  unmapped_precisions: string[];
}

/**
 * Map a recipe document to the §F fields it can supply.
 *
 * Deliberately does NOT touch geometry (layers / heads / embedding / attention regime) — that
 * comes from config.json via hf.ts, and a recipe has no authority over it.
 */
export function recipeToModel(doc: RecipeDoc): RecipeMapResult {
  const mapped: Partial<Model> = {};
  const filled: (keyof Model)[] = [];
  const vram_minimums: RecipeMapResult['vram_minimums'] = [];
  const unmapped = new Set<string>();

  const total = parseParamCount(doc.model?.parameter_count);
  if (total != null) {
    mapped.total_params_b = total;
    filled.push('total_params_b');
  }
  // Dense models often omit active_parameters; it equals the total.
  const active = parseParamCount(doc.model?.active_parameters);
  if (active != null && (total == null || active <= total)) {
    mapped.active_params_b = active;
    filled.push('active_params_b');
  } else if (total != null && !doc.model?.active_parameters) {
    mapped.active_params_b = total;
    filled.push('active_params_b');
  }

  if (doc.model?.context_length && doc.model.context_length > 0) {
    mapped.max_ctx = doc.model.context_length;
    filled.push('max_ctx');
  }

  const quants: Quant[] = [];
  const tps = new Set<number>();
  for (const [, v] of Object.entries(doc.variants ?? {})) {
    const label = (v.precision ?? '').toLowerCase().trim();
    const q = PRECISION_TO_QUANT[label];
    if (q) {
      if (!quants.includes(q)) quants.push(q);
      if (typeof v.vram_minimum_gb === 'number' && v.vram_minimum_gb > 0) {
        vram_minimums.push({ quant: q, precision: label, vram_gb: v.vram_minimum_gb, description: v.description });
      }
    } else if (label) {
      unmapped.add(label);
    }
    const tp = tpFromArgs(v.extra_args);
    if (tp) tps.add(tp);
  }
  if (quants.length) {
    mapped.quants = quants;
    filled.push('quants');
  }
  if (tps.size) {
    mapped.tp_options = [...tps].sort((a, b) => a - b);
    filled.push('tp_options');
  }

  if (doc.hf_id) {
    mapped.name = doc.meta?.title || doc.hf_id;
    mapped.id = doc.hf_id.split('/').pop()!.toLowerCase().replace(/[^a-z0-9-]/g, '-').slice(0, 64);
  }

  return {
    mapped,
    filled,
    vram_minimums,
    min_vllm_version: doc.model?.min_vllm_version,
    unmapped_precisions: [...unmapped],
  };
}

/**
 * Compare our computed weight figure against a recipe's stated VRAM floor.
 * Their floor covers weights + KV + runtime overhead, so ours should sit below it — typically
 * around 85%. Anything above 100% means one of the two is wrong and is worth surfacing.
 */
export function crossCheckVram(ourWeightsGib: number, theirFloorGb: number): {
  ours_gb: number;
  ratio: number;
  verdict: 'plausible' | 'tight' | 'over_floor';
} {
  const ours_gb = (ourWeightsGib * 2 ** 30) / 1e9;
  const ratio = ours_gb / theirFloorGb;
  return {
    ours_gb,
    ratio,
    verdict: ratio > 1 ? 'over_floor' : ratio > 0.95 ? 'tight' : 'plausible',
  };
}
