// Runtime support — the second half of feasibility (§4.4 of the implementation handoff).
//
// "Fits in HBM" and "will actually run" are different questions, and this tool only ever
// answered the first. A B300 plan at INT4 and an A100 plan at NVFP4 both size cleanly; one of
// them runs on native kernels and the other quietly falls back to a weight-only path that makes
// the throughput estimate fiction.
//
// The three-way verdict below is deliberate, and differs from the handoff's binary
// supported/unsupported. vLLM rarely refuses a format outright — it selects a kernel from the
// backends available on the platform and falls back to weight-only execution with a warning when
// there is no native one. That fallback keeps the MEMORY estimate intact (the weights really are
// 4-bit) while invalidating the COMPUTE estimate (activations stay 16-bit and the tensor cores
// never see the low-precision path). Collapsing that into "unsupported" would be wrong, and
// collapsing it into "supported" is what this tool did before.
//
// Unknown combinations report `unverified` rather than failing closed. Failing closed on unknown
// would block the entire catalogue the moment a GPU generation is added, which turns the safety
// property into a reason to delete the check.

import type { GpuArch, GpuSku, Model, Quant, SizingInput } from './types.js';

/** How well a (format, architecture) pair is actually served. */
export type SupportLevel = 'supported' | 'degraded' | 'unsupported' | 'unverified';

/** Ordering for "worst wins" when several constraints apply at once. */
const SEVERITY: Record<SupportLevel, number> = { supported: 0, unverified: 1, degraded: 2, unsupported: 3 };

export interface SupportFinding {
  level: SupportLevel;
  /** What the constraint is, in one sentence. */
  detail: string;
  /** The nearest thing that does work, where one exists (§4.4). */
  alternative?: string;
}

export interface RuntimeSupport {
  level: SupportLevel;
  /** Short label for the badge. */
  headline: string;
  findings: SupportFinding[];
  /** Where the rules came from, and when they were last checked. */
  source: typeof COMPAT_SOURCE;
}

/**
 * Provenance for the table below. Every rule traces to one of these; a rule with no source does
 * not belong here, it belongs in `unverified`.
 */
export const COMPAT_SOURCE = {
  runtime: 'vLLM',
  docs_snapshot: '2026-08-01',
  urls: [
    'https://docs.vllm.ai/en/latest/features/quantization/',
    'https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/',
    'https://docs.vllm.ai/en/latest/features/quantization/modelopt/',
    'https://docs.vllm.ai/en/latest/features/quantization/online/',
    'https://docs.vllm.ai/en/latest/features/quantization/quark/',
  ],
} as const;

type Rule = { level: SupportLevel; detail: string; alternative?: string };

/**
 * Quantisation × architecture. Absent cell ⇒ `unverified`, which is the honest answer for the
 * combinations vLLM's own hardware matrix does not cover (it stops at Hopper for several rows).
 *
 * Sourced, cell by cell:
 *  - FP8 W8A8 "is supported on NVIDIA GPUs with compute capability >= 8.9 (Ada Lovelace,
 *    Hopper)"; "Turing/Ampere GPUs are supported for W8A16 (weight-only FP8) utilizing Marlin
 *    kernels". AMD carries a ✅ for llm-compressor FP8 in the support matrix.
 *  - AWQ and GPTQ both carry ❌ for AMD GPUs in that matrix.
 *  - NVFP4: "On GPUs without a supported native FP4 GEMM kernel, vLLM falls back to weight-only
 *    (W4A16) execution via Marlin and logs a warning."
 *  - MXFP4: the matrix footnote is "Turing does not support Marlin MXFP4"; Quark documents MX
 *    *simulation* on MI300/MI325/MI250, which is not native execution.
 *  - GGUF carries ✅ across NVIDIA and AMD, but is not vLLM's native path and this planner's
 *    overhead model is tuned for vLLM rather than llama.cpp (see §2 of the methodology).
 */
const QUANT_RULES: Partial<Record<Quant, Partial<Record<GpuArch, Rule>>>> = {
  FP8: {
    ampere: {
      level: 'degraded',
      detail:
        'FP8 W8A8 needs compute capability 8.9 or newer. On Ampere the weights load at FP8 but run ' +
        'weight-only (W8A16) through Marlin kernels, so the memory saving is real and the compute ' +
        'speed-up is not — the throughput and TTFT figures here are optimistic.',
      alternative: 'An Ada, Hopper or Blackwell card gets native FP8; INT8 runs natively on Ampere.',
    },
    ada: { level: 'supported', detail: 'Native FP8 W8A8 (compute capability 8.9).' },
    hopper: { level: 'supported', detail: 'Native FP8 W8A8 (compute capability 9.0).' },
    blackwell: { level: 'supported', detail: 'Native FP8 W8A8.' },
    'blackwell-consumer': { level: 'supported', detail: 'Native FP8 W8A8.' },
    cdna3: { level: 'supported', detail: 'FP8 W8A8 is supported on AMD GPUs.' },
    cdna4: { level: 'supported', detail: 'FP8 W8A8 is supported on AMD GPUs.' },
  },
  INT8: {
    ampere: { level: 'supported', detail: 'INT8 W8A8 is supported from Turing onwards.' },
    ada: { level: 'supported', detail: 'INT8 W8A8 is supported from Turing onwards.' },
    hopper: { level: 'supported', detail: 'INT8 W8A8 is supported from Turing onwards.' },
    cdna3: { level: 'unsupported', detail: 'INT8 W8A8 is not supported on AMD GPUs.', alternative: 'FP8 runs natively on MI300X and newer.' },
    cdna4: { level: 'unsupported', detail: 'INT8 W8A8 is not supported on AMD GPUs.', alternative: 'FP8 runs natively on MI300X and newer.' },
  },
  INT4: {
    ampere: { level: 'supported', detail: 'AWQ/GPTQ 4-bit weight-only is supported from Turing onwards.' },
    ada: { level: 'supported', detail: 'AWQ/GPTQ 4-bit weight-only is supported from Turing onwards.' },
    hopper: { level: 'supported', detail: 'AWQ/GPTQ 4-bit weight-only is supported from Turing onwards.' },
    cdna3: { level: 'unsupported', detail: 'AWQ and GPTQ are both unsupported on AMD GPUs.', alternative: 'FP8 is the low-precision path on Instinct cards.' },
    cdna4: { level: 'unsupported', detail: 'AWQ and GPTQ are both unsupported on AMD GPUs.', alternative: 'FP8 is the low-precision path on Instinct cards.' },
  },
  NVFP4: {
    blackwell: { level: 'supported', detail: 'Native FP4 GEMM on Blackwell tensor cores.' },
    'blackwell-consumer': { level: 'supported', detail: 'Native FP4 GEMM on Blackwell tensor cores.' },
    ampere: NVFP4_FALLBACK('Ampere'),
    ada: NVFP4_FALLBACK('Ada'),
    hopper: NVFP4_FALLBACK('Hopper'),
    cdna3: NVFP4_FALLBACK('CDNA3'),
    cdna4: NVFP4_FALLBACK('CDNA4'),
  },
  MXFP4: {
    hopper: { level: 'supported', detail: 'MXFP4 checkpoints run on Hopper and newer.' },
    blackwell: { level: 'supported', detail: 'Native MX support on Blackwell.' },
    'blackwell-consumer': { level: 'supported', detail: 'Native MX support on Blackwell.' },
    ampere: {
      level: 'degraded',
      detail: 'No native MX path before Hopper — MXFP4 runs weight-only through Marlin, so the compute estimate is optimistic.',
      alternative: 'Hopper or Blackwell for native MXFP4; INT4 runs natively here.',
    },
    ada: {
      level: 'degraded',
      detail: 'No native MX path before Hopper — MXFP4 runs weight-only through Marlin, so the compute estimate is optimistic.',
      alternative: 'Hopper or Blackwell for native MXFP4; INT4 runs natively here.',
    },
    cdna3: {
      level: 'degraded',
      detail: 'MI300/MI325 do not implement OCP MX natively; the format is simulated, which is a correctness path rather than a performance one.',
      alternative: 'MI355X implements MX natively.',
    },
    cdna4: { level: 'supported', detail: 'MI355X implements OCP MX natively.' },
  },
};

/** The same fallback story on every architecture without an FP4 GEMM — written once. */
function NVFP4_FALLBACK(arch: string): Rule {
  return {
    level: 'degraded',
    detail:
      `${arch} has no native FP4 GEMM kernel, so vLLM falls back to weight-only (W4A16) execution ` +
      'via Marlin and logs a warning. The weights really are 4-bit — the memory plan holds — but ' +
      'activations stay 16-bit, so the throughput and TTFT figures here are optimistic.',
    alternative: 'A Blackwell card runs NVFP4 on native tensor cores.',
  };
}

/** GGUF quants: not vLLM's native territory, whatever the support matrix says. */
const GGUF_RULE: Rule = {
  level: 'degraded',
  detail:
    'GGUF is llama.cpp / Ollama territory and vLLM\'s support for it is experimental. This planner\'s ' +
    'overhead model (a 2.5 GiB reserve plus a paged-server utilisation cap) is tuned for vLLM, so a ' +
    'GGUF plan reads more pessimistically here than it behaves under llama.cpp.',
  alternative: 'INT4 or FP8 for a vLLM-native low-precision path.',
};

const GGUF_QUANTS: ReadonlySet<Quant> = new Set<Quant>(['Q8_0', 'Q4_K_M', 'IQ4_XS']);

/** Consumer cards have no NVLink, so a tensor-parallel group talks over PCIe. */
const CONSUMER_ARCHES: ReadonlySet<GpuArch> = new Set<GpuArch>(['blackwell-consumer']);

/**
 * Whether the planned deployment has a runtime path, and how good it is.
 *
 * Memory feasibility is computed separately and is NOT consulted here: a plan can fit perfectly
 * and still have no way to run, which is exactly the gap this exists to show.
 */
export function runtimeSupport(
  model: Model,
  gpu: GpuSku,
  input: Pick<SizingInput, 'quant' | 'gpus_per_node'>,
  tp = 1,
): RuntimeSupport {
  const findings: SupportFinding[] = [];
  const arch = gpu.arch;

  // --- format × architecture ---
  if (!arch) {
    findings.push({
      level: 'unverified',
      detail: `${gpu.name} has no architecture recorded, so no kernel support can be checked for ${input.quant}.`,
      alternative: 'Set the architecture on this GPU SKU in the catalogue.',
    });
  } else if (GGUF_QUANTS.has(input.quant)) {
    findings.push({ ...GGUF_RULE });
  } else {
    const rule = QUANT_RULES[input.quant]?.[arch];
    if (rule) findings.push({ ...rule });
    else if (input.quant !== 'FP16') {
      findings.push({
        level: 'unverified',
        detail: `vLLM's published support matrix does not cover ${input.quant} on ${arch}. It may work; it has not been checked.`,
      });
    }
  }

  // --- topology ---
  // A replica wider than a node is a different runtime shape, not just a bigger one.
  if (tp > input.gpus_per_node) {
    findings.push({
      level: 'degraded',
      detail:
        `TP ${tp} exceeds ${input.gpus_per_node} GPUs per node, so this replica spans nodes. That needs a Ray ` +
        'cluster and a fast fabric, and every layer\'s all-reduce crosses it — the throughput figures here ' +
        'assume the collective is not the bottleneck, which stops being true at this width.',
      alternative: 'A larger-memory GPU, or a smaller context, to keep the replica inside one node.',
    });
  } else if (arch && CONSUMER_ARCHES.has(arch) && tp > 1) {
    findings.push({
      level: 'degraded',
      detail:
        `${gpu.name} has no NVLink, so a TP ${tp} group runs its per-layer all-reduce over PCIe. It works, ` +
        'and it does not reach the collective bandwidth the throughput roofline assumes.',
      alternative: 'A single card that holds the model, or a datacentre SKU with NVLink.',
    });
  }

  const level = findings.reduce<SupportLevel>(
    (worst, f) => (SEVERITY[f.level] > SEVERITY[worst] ? f.level : worst),
    'supported',
  );
  return { level, headline: HEADLINE[level], findings, source: COMPAT_SOURCE };
}

const HEADLINE: Record<SupportLevel, string> = {
  supported: 'Runtime supported',
  degraded: 'Runs, but not as modelled',
  unsupported: 'No runtime path',
  unverified: 'Runtime unverified',
};
