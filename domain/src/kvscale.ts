// FP8 KV cache scale provenance (§4.3 of the implementation handoff).
//
// Selecting an FP8 KV cache halves the cache and the planner sizes every plan accordingly. What
// it does not decide is where the K and V scaling factors come from — and vLLM's default is
// unambiguous: with `calculate_kv_scales=False` and no scales in the checkpoint, "all
// quantization scales are set to 1.0". A unit scale is not a calibration; it is the absence of
// one, and it costs accuracy silently.
//
// This is not a kernel-support question (see compat.ts) and not a memory question. It is the one
// place where a setting the planner treats as free — FP8 KV is the default here — carries a
// quality cost that nothing else on the page would reveal.
//
// VERIFIED 2026-08-02: the safetensors index of all 31 artifacts in the seeded catalogue was
// checked for `k_scale`, `v_scale` and the older single `kv_scale` tensor. NOT ONE carries them.
// The method was validated against two checkpoints that do (RedHatAI/Meta-Llama-3-8B-Instruct-
// FP8-KV, 32 kv_scale tensors and kv_cache_scheme "static"; nvidia/Llama-3.1-8B-Instruct-FP8,
// 32 k_scale + 32 v_scale), so a zero here means absent rather than differently named.

import type { DeploymentVariant, Model, Quant } from './types.js';

/**
 * Where a deployment's K/V scaling factors come from.
 *   checkpoint — the artifact ships k_scale/v_scale tensors
 *   calibrated — those scales were produced against a calibration dataset (llm-compressor)
 *   runtime    — the operator passes `calculate_kv_scales`; estimated from one warm-up batch
 *   none       — verified absent: vLLM will use 1.0
 *   unknown    — nobody has checked this artifact
 */
export type KvScaleSource = NonNullable<DeploymentVariant['kv_scale_source']> | 'unknown';

export interface KvScaleVerdict {
  /** False when the plan keeps a 16-bit KV cache, where none of this applies. */
  applies: boolean;
  source: KvScaleSource;
  level: 'ok' | 'caution' | 'warn';
  headline: string;
  detail: string;
  remedy?: string;
}

const NOT_APPLICABLE: KvScaleVerdict = {
  applies: false,
  source: 'unknown',
  level: 'ok',
  headline: 'KV cache is 16-bit',
  detail: 'No KV quantisation, so no scaling factors are involved.',
};

/**
 * Resolve the scale story for a plan. `quant` selects which deployment artifact is in play —
 * scales live in the checkpoint, so the answer depends on which repository is being served, not
 * only on which model it is.
 */
export function kvScalePolicy(model: Model, quant: Quant, kvDtypeBytes: number): KvScaleVerdict {
  if (kvDtypeBytes !== 1) return NOT_APPLICABLE;
  const source: KvScaleSource = model.deployments?.[quant]?.kv_scale_source ?? 'unknown';

  switch (source) {
    case 'calibrated':
      return {
        applies: true, source, level: 'ok',
        headline: 'Dataset-calibrated K/V scales',
        detail: 'The artifact carries scaling factors estimated against a calibration dataset — the highest-assurance option, and what the halved cache in this plan assumes.',
      };
    case 'checkpoint':
      return {
        applies: true, source, level: 'ok',
        headline: 'Checkpoint-provided K/V scales',
        detail: 'The artifact ships k_scale/v_scale tensors, so vLLM quantises the cache against real scales rather than unit ones.',
      };
    case 'runtime':
      return {
        applies: true, source, level: 'caution',
        headline: 'Warm-up-calculated K/V scales',
        detail: 'Scales are estimated from a single batch of random tokens during warm-up and then fixed. Better than unit scales, and lower assurance than a dataset calibration — the sample is one batch and it is not your traffic.',
        remedy: 'A checkpoint with calibrated scales removes the guesswork; llm-compressor produces one.',
      };
    case 'none':
      return {
        applies: true, source, level: 'warn',
        headline: 'Uncalibrated FP8 KV cache',
        detail: 'This artifact ships no K/V scaling factors, so vLLM sets every quantisation scale to 1.0. The cache really is half the size — the memory plan is unaffected — but the accuracy cost is unbounded and invisible at launch, because nothing fails.',
        remedy: 'Pass calculate_kv_scales to estimate scales at warm-up, serve a checkpoint with calibrated scales, or keep the KV cache at 16 bits and re-size.',
      };
    default:
      return {
        applies: true, source: 'unknown', level: 'warn',
        headline: 'K/V scale source unverified',
        detail: 'Nothing is recorded about whether this artifact carries K/V scaling factors. If it does not, vLLM will use 1.0 scales and quantise the cache against them without complaint.',
        remedy: 'Check the artifact for k_scale/v_scale tensors and record the result on its deployment entry.',
      };
  }
}
