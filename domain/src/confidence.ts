// Confidence tiers for the performance figures (§6.1 of the implementation handoff).
//
// The complaint this answers: "the fixed ±40%/±50% bands do not react to runtime fallback,
// topology, calibration provenance, or measured evidence." Two of those now suppress the figure
// outright rather than banding it, which leaves a narrower question — what does a number that IS
// shown actually rest on?
//
// What this deliberately does NOT do is invent per-factor band widths. Widening ±40% to ±55%
// because a model is MoE would be a fabricated number dressed as rigour, and this codebase has
// twice chosen to state an omission rather than paper over it (collective latency, GGUF
// suppression). So the band stays as published and the ASSUMPTIONS become explicit and
// per-plan: which ones apply to this configuration, and which way each one biases the figure.
//
// Direction is derivable without inventing magnitudes, and it is the part a reader can act on.
// "Real routers are skewed, so the true throughput is probably higher than this" is a useful
// sentence. "±47%" is not.

import { expertCoverage } from './engine.js';
import type { FeasibleSizing, GpuSku, Model, SizingInput } from './types.js';

export type ConfidenceTier = 'measured' | 'calibrated' | 'estimated' | 'withheld';

/** Which way an unmodelled effect pushes the figure, relative to what the plan reports. */
export type Bias = 'reads high' | 'reads low' | 'unknown';

export interface Assumption {
  label: string;
  detail: string;
  bias: Bias;
}

export interface Confidence {
  tier: ConfidenceTier;
  /** The published band, or null when the figure is withheld. */
  band: string | null;
  /** Why the tier is what it is. */
  basis: string;
  /** Named, per-plan — not a static list. */
  assumptions: Assumption[];
}

const WITHHELD = (why: string): Confidence => ({
  tier: 'withheld',
  band: null,
  basis: why,
  assumptions: [],
});

/**
 * The tier every figure sits at today.
 *
 * `measured` and `calibrated` are in the type because they are the destination, not because
 * anything reaches them: both require benchmark evidence this planner cannot yet import. Naming
 * a tier the tool can never currently return would be worse than admitting the ceiling — so
 * `basis` says so in as many words rather than leaving "estimated" to look like a grade.
 */
const ANALYTICAL =
  'Analytical model only — no benchmark for this model, artifact, GPU, TP width and workload ' +
  'has been imported, so nothing here is calibrated against a measurement.';

/** Assumptions behind the decode throughput figure, for this plan specifically. */
export function throughputConfidence(
  model: Model,
  gpu: GpuSku,
  input: Pick<SizingInput, 'quant'>,
  sizing: FeasibleSizing,
  batchTokens: number,
): Confidence {
  if (sizing.throughput_suppressed) return WITHHELD(sizing.throughput_suppressed);
  const assumptions: Assumption[] = [
    {
      label: 'MBU 0.55',
      detail: 'A single bandwidth-utilisation constant stands in for kernels, batching and scheduler overhead on every GPU in the catalogue.',
      bias: 'unknown',
    },
  ];

  const coverage = expertCoverage(model, batchTokens);
  if (model.num_experts && coverage > 0.02 && coverage < 0.98) {
    assumptions.push({
      label: 'uniform expert routing',
      detail: `This batch is modelled as touching ${(coverage * 100).toFixed(0)}% of ${model.num_experts} experts. Real routers are skewed, which concentrates tokens on fewer experts and reads less weight per step.`,
      bias: 'reads low',
    });
  }
  if (sizing.tp > 1) {
    assumptions.push({
      label: 'collective latency unmodelled',
      detail: `The all-reduce is priced by bandwidth only. At decode batch sizes the messages are small and there are ${2 * model.layers} launches per step, so the fixed per-collective cost is real and not counted here.`,
      bias: 'reads high',
    });
  }
  if (sizing.weights_estimated) {
    assumptions.push({
      label: 'flat weight overhead',
      detail: 'This model carries no embedding geometry, so its weights use a flat ×1.02 factor rather than a sized un-quantised tail.',
      bias: 'unknown',
    });
  }
  assumptions.push({
    label: 'no prefix cache or speculative decoding',
    detail: 'Neither is modelled. Both raise real throughput on workloads that benefit, and neither is free to assume.',
    bias: 'reads low',
  });

  return { tier: 'estimated', band: '±40%', basis: ANALYTICAL, assumptions };
}

/** Assumptions behind the TTFT figure, for this plan specifically. */
export function ttftConfidence(
  model: Model,
  gpu: GpuSku,
  input: Pick<SizingInput, 'quant'>,
  sizing: FeasibleSizing,
): Confidence {
  if (sizing.ttft_suppressed) return WITHHELD(sizing.ttft_suppressed);
  const assumptions: Assumption[] = [];

  if (!gpu.tflops_fp16) {
    assumptions.push({
      label: 'no FLOPS figure for this SKU',
      detail: 'TTFT falls back to a weight-streaming floor, which is a decode bound borrowed for a compute-bound phase. It is a floor, not an estimate.',
      bias: 'reads low',
    });
  } else {
    assumptions.push({
      label: 'prefill MFU 0.4',
      detail: 'One utilisation constant for every GPU and every model shape.',
      bias: 'unknown',
    });
    if (sizing.tp > 1) {
      assumptions.push({
        label: 'collective latency unmodelled',
        detail: `The prefill all-reduce is priced by bandwidth, which dominates at prompt sizes — but the ${2 * model.layers} launches still cost something that is not counted.`,
        bias: 'reads high',
      });
    }
  }
  assumptions.push({
    label: 'low-precision speedup capped at 2×',
    detail: 'Sub-16-bit prefill is assumed at most twice the FP16 rate. Blackwell exceeds that, and the catalogue does not track generation, so TTFT is deliberately not sharpened for it.',
    bias: 'reads high',
  });
  assumptions.push({
    label: 'one prompt, no queue',
    detail: 'This is the prefill time for a single request on an idle replica. Concurrent prefills share the same compute and queue behind each other.',
    bias: 'reads low',
  });

  return { tier: 'estimated', band: '±50%', basis: ANALYTICAL, assumptions };
}
