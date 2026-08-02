// Measured vLLM memory profile (§5.1 of the implementation handoff).
//
// Everything else in this engine is a heuristic standing in for a number the runtime already
// knows. vLLM profiles itself at startup and reports exactly how much KV cache it ended up with
// — so where that figure exists, guessing at it is a choice rather than a necessity.
//
// The rule from the handoff, and the one thing that matters here: measured
// `available_kv_cache_bytes` becomes AUTHORITATIVE for concurrency. The estimate is preserved
// beside it and the variance is shown. The two are never blended — an average of a measurement
// and a guess is neither, and it hides which one moved.
//
// What a profile does NOT do is make the throughput figure measured. It calibrates memory. A
// tok/s number still comes from the roofline, and calling it measured because a memory profile
// was imported would be exactly the overclaim this module exists to avoid.

import { GIB } from './engine.js';
import type { GpuSku, Model } from './types.js';

/**
 * What vLLM's GPU worker reports after its startup profiling run. Only two fields are required:
 * the KV capacity (the point of the exercise) and the total, which is what makes the figure
 * checkable against the card rather than taken on faith.
 */
export interface MeasuredProfile {
  /** Bytes vLLM saw on the device. */
  total_memory_bytes: number;
  /** Bytes the weights occupy, per GPU. */
  model_memory_bytes?: number;
  peak_activation_bytes?: number;
  non_framework_bytes?: number;
  graph_memory_bytes?: number;
  /** The number that replaces the estimate: KV bytes actually available, per GPU. */
  available_kv_cache_bytes: number;
  /** Provenance — a profile from a different shape must not silently re-size a plan. */
  runtime_version?: string;
  gpu_sku_id?: string;
  driver_version?: string;
  tensor_parallel_size?: number;
  gpu_memory_utilization?: number;
}

export type MeasuredStatus = 'applied' | 'mismatch' | 'absent';

export interface MeasuredReconciliation {
  status: MeasuredStatus;
  /** Free KV per replica the estimate produced, always preserved. */
  estimated_free_gb: number;
  /** Free KV per replica the profile reports, when it applies. */
  measured_free_gb: number | null;
  /** measured / estimated − 1, positive when the estimate was conservative. */
  variance: number | null;
  /** Why a profile was not applied, when it was not. */
  reason: string | null;
}

/**
 * Decide whether a profile describes the plan in front of us.
 *
 * A profile is a measurement of one specific deployment shape. Applied to a different TP width
 * or a different card it is not conservative or optimistic, it is unrelated — so a mismatch
 * refuses rather than scaling, and says which field disagreed.
 */
export function reconcileMeasured(
  profile: MeasuredProfile | undefined,
  gpu: GpuSku,
  tp: number,
  estimatedFreeGb: number,
): MeasuredReconciliation {
  const base = { estimated_free_gb: estimatedFreeGb, measured_free_gb: null, variance: null };
  if (!profile) return { ...base, status: 'absent', reason: null };

  if (profile.gpu_sku_id && profile.gpu_sku_id !== gpu.id) {
    return { ...base, status: 'mismatch', reason: `Profile was taken on ${profile.gpu_sku_id}, this plan is on ${gpu.id}. A measurement of another card is not evidence about this one.` };
  }
  if (profile.tensor_parallel_size && profile.tensor_parallel_size !== tp) {
    return { ...base, status: 'mismatch', reason: `Profile was taken at TP ${profile.tensor_parallel_size}, this plan selects TP ${tp}. KV capacity per GPU does not carry across shard widths.` };
  }
  if (!(profile.available_kv_cache_bytes > 0)) {
    return { ...base, status: 'mismatch', reason: 'Profile reports no available KV cache bytes, which is the one field this replaces.' };
  }
  if (profile.total_memory_bytes > 0 && profile.available_kv_cache_bytes > profile.total_memory_bytes) {
    return { ...base, status: 'mismatch', reason: 'Profile reports more KV cache than total device memory — the figures are not from the same run.' };
  }

  // vLLM reports per GPU; a replica's KV budget is that across the shard.
  const measured = (profile.available_kv_cache_bytes * tp) / GIB;
  return {
    status: 'applied',
    estimated_free_gb: estimatedFreeGb,
    measured_free_gb: measured,
    variance: estimatedFreeGb > 0 ? measured / estimatedFreeGb - 1 : null,
    reason: null,
  };
}

/**
 * Parse a profile out of pasted text — either the JSON object itself, or a vLLM startup log
 * line, which is how anyone would actually come by these numbers.
 *
 * vLLM logs, in the form this recognises:
 *   GPU KV cache size: 1,234,567,890 bytes
 *   Available KV cache memory: 45.67 GiB
 *   model weights take 65.43 GiB; ... PyTorch activation peak memory takes 1.23 GiB
 */
export function parseProfile(text: string): { profile: MeasuredProfile | null; error: string | null } {
  const trimmed = text.trim();
  if (!trimmed) return { profile: null, error: null };

  if (trimmed.startsWith('{')) {
    try {
      const raw = JSON.parse(trimmed) as Partial<MeasuredProfile>;
      if (!(Number(raw.available_kv_cache_bytes) > 0)) {
        return { profile: null, error: 'JSON parsed, but available_kv_cache_bytes is missing or not positive — that is the field this replaces.' };
      }
      return {
        profile: {
          ...raw,
          total_memory_bytes: Number(raw.total_memory_bytes ?? 0),
          available_kv_cache_bytes: Number(raw.available_kv_cache_bytes),
        },
        error: null,
      };
    } catch {
      return { profile: null, error: 'Not valid JSON. Paste the profile object, or the vLLM startup log lines.' };
    }
  }

  const num = (re: RegExp): number | null => {
    const m = re.exec(trimmed);
    if (!m) return null;
    const v = Number(m[1].replace(/,/g, ''));
    if (!Number.isFinite(v)) return null;
    return /GiB/i.test(m[0]) ? v * GIB : v;
  };
  const kv =
    num(/Available KV cache memory:?\s*([\d.,]+)\s*GiB/i) ??
    num(/GPU KV cache size:?\s*([\d.,]+)\s*bytes/i);
  if (kv === null) {
    return { profile: null, error: 'No KV cache figure found. Expected "Available KV cache memory: N GiB" or "GPU KV cache size: N bytes".' };
  }
  const weights = num(/model weights take\s*([\d.,]+)\s*GiB/i);
  const activation = num(/activation peak memory takes\s*([\d.,]+)\s*GiB/i);
  const nonTorch = num(/non[- ]torch memory takes\s*([\d.,]+)\s*GiB/i);
  return {
    profile: {
      total_memory_bytes: 0,
      available_kv_cache_bytes: kv,
      ...(weights !== null ? { model_memory_bytes: weights } : {}),
      ...(activation !== null ? { peak_activation_bytes: activation } : {}),
      ...(nonTorch !== null ? { non_framework_bytes: nonTorch } : {}),
    },
    error: null,
  };
}

/** How far the heuristic was off, phrased for a reader rather than as a signed float. */
export function varianceLabel(r: MeasuredReconciliation): string | null {
  if (r.status !== 'applied' || r.variance === null) return null;
  const pct = Math.abs(r.variance * 100);
  if (pct < 1) return 'the estimate was within 1% of the measurement';
  return r.variance > 0
    ? `the estimate was ${pct.toFixed(0)}% conservative — vLLM found more KV room than modelled`
    : `the estimate was ${pct.toFixed(0)}% optimistic — vLLM found less KV room than modelled`;
}

/** Unused-import guard: Model is part of the public shape callers pass around. */
export type MeasuredFor = { model: Model; profile: MeasuredProfile };
