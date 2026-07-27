// Addendum §C sizing acceptance vectors (AC-1…AC-5) — the PRD §18.2 gate (NFR-M-3).
// Tolerances: integer counts EXACT; memory within ±5%; concurrency/pod within ±10%. Failures block release.
//
// ALL memory figures below are GiB (2^30 bytes) — the engine's single unit, matching nvidia-smi
// and GpuSku.mem_gb. Vectors were re-pinned when the engine stopped mixing 1e9-byte GB (weights)
// with 2^30-byte GiB (KV, capacity), which had inflated weights ~7.4% against GPU capacity.
import { describe, it, expect } from 'vitest';
import { computeSizing, weightsGb, activeWeightsGb, unquantisedParamsB, paramBytesToGib, kvPerTokenBytes, kvPerRequestBytes, layerSplit, GIB, QB, TIGHT_HEADROOM, WEIGHT_OVERHEAD, RUNTIME_GB } from '../engine.js';
import { seedCatalog } from '../seed.js';
import { modelSchema, gpuSkuSchema } from '../schema.js';
import { reconcile, headroomCheck } from '../reconcile.js';
import type { FeasibleSizing } from '../types.js';

const { models, gpus } = seedCatalog();
const model = (id: string) => models.find((m) => m.id === id)!;
const gpu = (id: string) => gpus.find((g) => g.id === id)!;

const near = (actual: number, expected: number, pct: number) =>
  Math.abs(actual - expected) <= Math.abs(expected) * pct;

describe('§C sizing acceptance vectors (PRD §18.2)', () => {
  it('AC-1 — Llama 3.3 70B FP8/KV-FP8 128K 60% conc64 H200 → TP4, 8 GPUs, 1 node', () => {
    const r = computeSizing(model('llama33-70b'), gpu('h200'), {
      quant: 'FP8', kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8,
    }) as FeasibleSizing;
    expect(r.ok).toBe(true);
    expect(r.tp).toBe(4); // TP2 also fits one request but needs 10 GPUs; TP4 needs 8
    expect(near(r.weights_gb, 67.7, 0.05)).toBe(true);
    expect(near(r.kv_per_token_gb * 1024, 0.156, 0.05)).toBe(true); // ~0.156 MiB/token
    expect(near(r.kv_per_request_gb, 12.0, 0.05)).toBe(true);
    expect(near(r.concurrency_per_pod, 35, 0.1)).toBe(true);
    expect(r.pods).toBe(2);
    expect(r.gpus).toBe(8);
    expect(r.nodes).toBe(1);
  });

  it('AC-2 — GLM-5.2 743B FP8/KV-FP8 128K 60% conc64 H200 → TP8, 8 GPUs (one node)', () => {
    const r = computeSizing(model('glm52'), gpu('h200'), {
      quant: 'FP8', kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8,
    }) as FeasibleSizing;
    expect(r.ok).toBe(true);
    expect(r.tp).toBe(8);
    expect(near(r.weights_gb, 694, 0.05)).toBe(true); // 16-bit embedding tail split out, in GiB
    expect(near(r.free_gb, 301, 0.05)).toBe(true); // free KV budget/replica
    // MLA (kv_lora_rank 512 + qk_rope_head_dim 64 = 576/layer), NOT the GQA 8x128 first assumed
    expect(near(r.kv_per_request_gb, 3.29, 0.05)).toBe(true);
    expect(near(r.concurrency_per_pod, 91, 0.1)).toBe(true);
    expect(r.pods).toBe(1);
    expect(r.gpus).toBe(8);
    expect(r.nodes).toBe(1); // matches the vLLM recipe's "8xH200 single-node FP8"
  });

  it('AC-2b — GLM-5.2 at its full 1M context still fits one node', () => {
    const r = computeSizing(model('glm52'), gpu('b200'), {
      quant: 'FP8', kv_dtype_bytes: 1, selected_ctx: 1048576, avg_context_utilisation: 0.6,
      target_concurrency: 8, mem_util_fraction: 0.9, gpus_per_node: 8,
    }) as FeasibleSizing;
    expect(r.ok).toBe(true);
    expect(r.tp).toBe(8);
    expect(r.gpus).toBe(8);
    expect(r.nodes).toBe(1); // the recipe's "8xB200 for the full 1M context"
    expect(near(r.kv_per_request_gb, 26.3, 0.05)).toBe(true); // 1M ctx MLA KV is still 26 GiB/request
  });

  it('AC-2c — GLM-5.2 is MLA: the old GQA 8x128 guess overstated its KV ~3.6x', () => {
    const m = model('glm52');
    expect(m.mla).toBe(true);
    expect(m.kv_heads).toBe(0);
    expect(m.head_dim).toBe(0);
    const tokens = 131072 * 0.6;
    const mlaKv = kvPerRequestBytes(m, 1, tokens);
    const gqaGuess = 2 * m.layers * 8 * 128 * 1 * tokens; // what the seed used to claim
    expect(gqaGuess / mlaKv).toBeCloseTo(3.56, 1);
  });

  it('AC-3 — DeepSeek-V3 671B (MLA) FP8/KV-FP8 128K 60% conc64 H200 → TP8, 8 GPUs, conc/pod ≥100', () => {
    const r = computeSizing(model('dsv3'), gpu('h200'), {
      quant: 'FP8', kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8,
    }) as FeasibleSizing;
    expect(r.ok).toBe(true);
    expect(r.tp).toBe(8);
    expect(near(r.weights_gb, 627, 0.05)).toBe(true);
    expect(near(r.kv_per_request_gb, 2.6, 0.05)).toBe(true); // MLA — materially smaller
    expect(r.concurrency_per_pod).toBeGreaterThanOrEqual(100);
    expect(r.pods).toBe(1);
    expect(r.gpus).toBe(8);
  });

  it('AC-4 — Llama 3.1 8B FP8/KV-FP8 128K 60% conc64 H100 → TP1, pods ≤6, GPUs ≤6', () => {
    const r = computeSizing(model('llama31-8b'), gpu('h100'), {
      quant: 'FP8', kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8,
    }) as FeasibleSizing;
    expect(r.ok).toBe(true);
    expect(r.tp).toBe(1);
    expect(r.pods).toBeLessThanOrEqual(6);
    expect(r.gpus).toBeLessThanOrEqual(6);
  });

  it('AC-5 — Kimi K2 1T (MLA) FP16/KV-FP16 H100 → infeasible (weights exceed TP16)', () => {
    const r = computeSizing(model('kimi-k2'), gpu('h100'), {
      quant: 'FP16', kv_dtype_bytes: 2, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8,
    });
    expect(r.ok).toBe(false);
  });
});

// TP selection minimises TOTAL GPUs, not shard size. Picking the smallest TP that merely holds
// one request over-recommends hardware, because a bigger shard packs far more sessions per pod.
describe('§C TP selection minimises total GPUs', () => {
  it('AC-38 — every feasible TP is considered; the cheapest total wins', () => {
    const m = model('llama33-70b'), g = gpu('h200');
    const input = {
      quant: 'FP8' as const, kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8,
    };
    const chosen = computeSizing(m, g, input) as FeasibleSizing;
    // brute-force the same objective over the model's TP ladder
    const usable = g.mem_gb * 0.9 - RUNTIME_GB;
    const w = weightsGb(m, 'FP8');
    const kv = kvPerRequestBytes(m, 1, 131072 * 0.6) / GIB;
    let bestGpus = Infinity;
    for (const t of m.tp_options) {
      const free = t * usable - w;
      if (free < kv) continue;
      bestGpus = Math.min(bestGpus, Math.ceil(64 / Math.floor(free / kv)) * t);
    }
    expect(chosen.gpus).toBe(bestGpus);
    expect(chosen.gpus).toBe(8); // the old smallest-that-fits rule answered 10
  });

  it('AC-39 — ties break toward the smaller shard (less collective traffic)', () => {
    // TP4 and TP8 both reach 8 GPUs for this workload; TP4 must win
    const r = computeSizing(model('llama33-70b'), gpu('h200'), {
      quant: 'FP8', kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8,
    }) as FeasibleSizing;
    expect(r.tp).toBe(4);
    expect(r.multi_node).toBe(false);
  });

  it('AC-40 — a wider TP ladder never costs more GPUs than a narrow one', () => {
    const g = gpu('h200');
    const input = {
      quant: 'FP8' as const, kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8,
    };
    const wide = computeSizing(model('llama33-70b'), g, input) as FeasibleSizing;
    const narrow = computeSizing({ ...model('llama33-70b'), tp_options: [2] }, g, input) as FeasibleSizing;
    expect(wide.gpus).toBeLessThanOrEqual(narrow.gpus);
  });

  it('AC-41 — DeepSeek-V3 INT4 lands on 4 B200s, matching the published recipe', () => {
    const r = computeSizing(model('dsv3'), gpu('b200'), {
      quant: 'INT4', kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8,
    }) as FeasibleSizing;
    expect(r.tp).toBe(4);
    expect(r.gpus).toBe(4); // tp_options used to start at 8, forcing twice the hardware
  });
});

// The un-quantised tail (16-bit embedding + lm_head) is invisible at FP16 and dominant at 4-bit.
// A flat overhead factor cannot express it — these vectors pin the difference against real
// published checkpoint sizes. Tolerance: GB within ±5%, as above.
describe('§C low-bit weight vectors (un-quantised embedding/lm_head tail)', () => {
  it('AC-6 — Llama 3.3 70B INT4 ≈ 37.1 GiB, not the 32.9 GiB a naive params×0.5 gives', () => {
    const m = model('llama33-70b');
    // 2 × 128256 vocab × 8192 hidden = 2.10 B params held at fp16 = 3.91 GiB
    expect(near(unquantisedParamsB(m)!, 2.101, 0.01)).toBe(true);
    const w = weightsGb(m, 'INT4');
    // published AWQ/GPTQ 4-bit checkpoints weigh ~40e9 bytes = 37.3 GiB
    expect(near(w, 37.1, 0.05)).toBe(true);
    expect(near(w, (40e9) / 2 ** 30, 0.05)).toBe(true);
    const naive = paramBytesToGib(m.total_params_b, 0.5);
    expect(w).toBeGreaterThan(paramBytesToGib(m.total_params_b * WEIGHT_OVERHEAD, 0.5)); // beats the flat factor
    expect(w / naive).toBeGreaterThan(1.1); // tail is >10% of the footprint
  });

  it('AC-7 — GPT-OSS 120B MXFP4 ≈ 59.5 GiB, within 5% of the published ~60.8 GiB checkpoint', () => {
    const w = weightsGb(model('gptoss-120b'), 'MXFP4');
    expect(near(w, 59.5, 0.05)).toBe(true);
    expect(near(w, 60.8, 0.05)).toBe(true); // same unit — no conversion needed
  });

  it('AC-8 — Qwen3-30B-A3B INT4: a small-hidden MoE still carries a 1.2 GiB fp16 tail', () => {
    const m = model('qwen3-30a3');
    expect(near(unquantisedParamsB(m)!, 0.6223, 0.01)).toBe(true);
    expect(near(weightsGb(m, 'INT4'), 15.6, 0.05)).toBe(true);
  });

  it('AC-9 — FP16 is unaffected by the tail term (nothing is quantised)', () => {
    const m = model('llama33-70b');
    expect(weightsGb(m, 'FP16')).toBeCloseTo(paramBytesToGib(m.total_params_b, 2), 6);
  });

  it('AC-10 — effective bytes/param include per-group metadata, not just the data bits', () => {
    expect(QB.INT4).toBeGreaterThan(0.5); // fp16 scale + int4 zero per 128-element group
    expect(QB.MXFP4).toBeGreaterThan(0.5); // E8M0 scale per 32-element block
    expect(QB.NVFP4).toBeGreaterThan(QB.MXFP4); // E4M3 scale per 16-element block — denser metadata
    expect(QB.FP8).toBe(1);
  });

  it('AC-11 — a model with no embedding geometry falls back to the legacy flat factor', () => {
    const legacy = { ...model('llama33-70b'), hidden_size: undefined, vocab_size: undefined };
    expect(unquantisedParamsB(legacy)).toBeNull();
    expect(weightsGb(legacy, 'INT4')).toBeCloseTo(paramBytesToGib(70.6 * WEIGHT_OVERHEAD, QB.INT4), 6);
    const r = computeSizing(legacy, gpu('h200'), {
      quant: 'INT4', kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8,
    }) as FeasibleSizing;
    expect(r.ok).toBe(true);
    expect(r.weights_estimated).toBe(true); // surfaced to the UI as a caveat
  });
});

// Unit discipline. Weights, KV, usable HBM and the capacity gate must all be the SAME unit
// (GiB = 2^30 bytes), or every comparison between them carries a silent 7.4% error.
describe('§C unit consistency (GiB throughout)', () => {
  it('AC-15 — weights are GiB: params × bytes/param converted through 2^30, not left as 1e9-byte GB', () => {
    const m = model('llama33-70b');
    const rawBytes = m.total_params_b * 1e9 * 2; // FP16, no tail adjustment needed
    expect(weightsGb(m, 'FP16')).toBeCloseTo(rawBytes / GIB, 6);
    // the old bug: reporting rawBytes/1e9 instead, i.e. 7.37% high
    expect(weightsGb(m, 'FP16')).toBeLessThan(rawBytes / 1e9);
    expect((rawBytes / 1e9) / weightsGb(m, 'FP16')).toBeCloseTo(GIB / 1e9, 6);
  });

  it('AC-16 — weights, KV and usable HBM are directly comparable', () => {
    const m = model('llama33-70b');
    const g = gpu('h200');
    const r = computeSizing(m, g, {
      quant: 'FP8', kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8,
    }) as FeasibleSizing;
    // usable HBM derives from mem_gb, which the capacity gate treats as GiB
    expect(r.usable_gb).toBeCloseTo(g.mem_gb * 0.9 - RUNTIME_GB, 9);
    expect(Number(r.committed_bytes_per_gpu) / GIB).toBeCloseTo(g.mem_gb, 9);
    // KV comes straight from a byte count over 2^30
    expect(r.kv_per_token_gb).toBeCloseTo(kvPerTokenBytes(m, 1) / GIB, 12);
    // and the free-space arithmetic that mixes all three is therefore exact
    expect(r.free_gb).toBeCloseTo(r.tp * r.usable_gb - r.weights_gb, 9);
    expect(r.concurrency_per_pod).toBe(Math.floor(r.free_gb / r.kv_per_request_gb));
  });

  it('AC-17 — the roofline divides GiB-derived bytes by decimal TB/s bandwidth', () => {
    const m = model('llama33-70b');
    const g = gpu('h200');
    const r = computeSizing(m, g, {
      quant: 'FP8', kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 10, mem_util_fraction: 0.9, gpus_per_node: 8,
    }) as FeasibleSizing;
    // dense model, but the decode stream reads only the lm_head of the tail (the embedding
    // table is a per-token gather), so this is activeWeightsGb, not weightsGb
    const activeGib = activeWeightsGb(m, 'FP8');
    const stepBytes = (activeGib + 10 * r.kv_per_request_gb) * GIB;
    const bytesPerSec = r.tp * g.bw_tbs * 1e12 * 0.55;
    expect(r.step_time_ms).toBeCloseTo(Math.round((stepBytes / bytesPerSec) * 1e5) / 100, 2);
    expect(r.ttft_ms).toBe(Math.round(((activeGib * GIB) / bytesPerSec) * 1000));
  });
});

// "Tight" = it fits, but with <10% of pod HBM free once weights + one request of KV are placed.
describe('§C tight-fit band (fits / tight / infeasible)', () => {
  it('AC-12 — Qwen3-32B Q4_K_M on ONE RTX 4090 at 4K is feasible but tight', () => {
    const r = computeSizing(model('qwen3-32b'), gpu('rtx4090'), {
      quant: 'Q4_K_M', kv_dtype_bytes: 1, selected_ctx: 4096, avg_context_utilisation: 0.6,
      target_concurrency: 1, mem_util_fraction: 0.9, gpus_per_node: 1,
    }) as FeasibleSizing;
    expect(r.ok).toBe(true);
    expect(r.tp).toBe(1);
    expect(r.tight).toBe(true);
    expect(near(r.headroom_fraction, 0.009, 0.3)).toBe(true); // 18.6 GiB weights in 19.1 GiB usable
    expect(r.headroom_fraction).toBeLessThan(TIGHT_HEADROOM);
    expect(r.headroom_fraction).toBeGreaterThan(0); // still fits — not infeasible
    expect(r.concurrency_per_pod).toBe(1); // no room to batch a second request
  });

  it('AC-12b — the same model at 8K jumps to TP2 and stops being tight', () => {
    const r = computeSizing(model('qwen3-32b'), gpu('rtx4090'), {
      quant: 'Q4_K_M', kv_dtype_bytes: 1, selected_ctx: 8192, avg_context_utilisation: 0.6,
      target_concurrency: 1, mem_util_fraction: 0.9, gpus_per_node: 1,
    }) as FeasibleSizing;
    expect(r.tp).toBe(2);
    expect(r.tight).toBe(false);
  });

  it('AC-13 — GPT-OSS 120B on an H200 has real headroom and is not tight', () => {
    const r = computeSizing(model('gptoss-120b'), gpu('h200'), {
      quant: 'MXFP4', kv_dtype_bytes: 2, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8,
    }) as FeasibleSizing;
    expect(r.ok).toBe(true);
    expect(r.tight).toBe(false);
    expect(r.headroom_fraction).toBeGreaterThanOrEqual(TIGHT_HEADROOM);
  });

  it('AC-14 — headroom_fraction is exactly the pod HBM left after weights + one request', () => {
    const r = computeSizing(model('llama33-70b'), gpu('h200'), {
      quant: 'FP8', kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8,
    }) as FeasibleSizing;
    const pod = r.tp * r.usable_gb;
    expect(r.headroom_fraction).toBeCloseTo((pod - r.weights_gb - r.kv_per_request_gb) / pod, 9);
    expect(r.tight).toBe(false);
  });
});

// Local/global attention. Sliding-window layers stop accumulating KV at the window, so
// long-context KV is far below the all-full-attention figure. Getting this wrong inflates KV
// several-fold and can manufacture a false "tight" verdict.
describe('§C sliding-window KV', () => {
  it('AC-18 — GPT-OSS 120B: 18 of 36 layers banded at 128 tokens halves 128K KV', () => {
    const m = model('gptoss-120b');
    expect(m.sliding_window).toBe(128);
    expect(m.full_attention_layers).toBe(18);
    const activeTokens = 131072 * 0.6;
    const windowed = kvPerRequestBytes(m, 2, activeTokens) / GIB;
    const allFull = (kvPerTokenBytes(m, 2) * activeTokens) / GIB;
    expect(near(windowed, 2.704, 0.02)).toBe(true);
    expect(near(allFull, 5.4, 0.02)).toBe(true);
    expect(windowed / allFull).toBeCloseTo(0.5, 2); // exactly half the layers grow
  });

  it('AC-19 — modelling the window flips the verdict on one H100 from tight to comfortable', () => {
    const input = {
      quant: 'MXFP4' as const, kv_dtype_bytes: 2, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 1, mem_util_fraction: 0.9, gpus_per_node: 8,
    };
    const r = computeSizing(model('gptoss-120b'), gpu('h100'), input) as FeasibleSizing;
    expect(r.kv_windowed).toBe(true);
    expect(r.tight).toBe(false); // was tight at 6.7% when all 36 layers were treated as full
    expect(r.headroom_fraction).toBeGreaterThan(TIGHT_HEADROOM);
    expect(r.concurrency_per_pod).toBe(3); // was 1

    // same model with the window stripped — the old, wrong answer
    const naive = { ...model('gptoss-120b'), sliding_window: undefined, full_attention_layers: undefined };
    const rn = computeSizing(naive, gpu('h100'), input) as FeasibleSizing;
    expect(rn.kv_windowed).toBe(false);
    expect(rn.tight).toBe(true);
    expect(rn.kv_per_request_gb / r.kv_per_request_gb).toBeCloseTo(2, 1);
  });

  it('AC-20 — the window only binds past its length; short contexts are unaffected', () => {
    const m = model('gptoss-120b');
    // 100 active tokens < the 128-token window, so every layer is still accumulating
    expect(kvPerRequestBytes(m, 2, 100)).toBe(kvPerTokenBytes(m, 2) * 100);
    // well past it, only the full-attention layers keep growing
    expect(kvPerRequestBytes(m, 2, 100_000)).toBeLessThan(kvPerTokenBytes(m, 2) * 100_000);
  });

  it('AC-21 — reported kv_per_token reconciles with kv_per_request under windowing', () => {
    const r = computeSizing(model('gptoss-120b'), gpu('h200'), {
      quant: 'MXFP4', kv_dtype_bytes: 2, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 8, mem_util_fraction: 0.9, gpus_per_node: 8,
    }) as FeasibleSizing;
    expect(r.kv_per_token_gb * (131072 * 0.6)).toBeCloseTo(r.kv_per_request_gb, 9);
    expect(r.kv_per_token_gb).toBeLessThan(kvPerTokenBytes(model('gptoss-120b'), 2) / GIB);
  });

  it('AC-22 — a model with no window declared is unchanged (every layer full-context)', () => {
    const m = model('llama33-70b');
    expect(m.sliding_window).toBeUndefined();
    expect(kvPerRequestBytes(m, 1, 78643.2)).toBeCloseTo(kvPerTokenBytes(m, 1) * 78643.2, 6);
  });
});

// Hybrid linear attention. Recurrent layers (Kimi K3's KDA, Qwen3-Next, MiniMax) hold a
// CONSTANT state per request — no per-token cache at all. Sizing them as full attention
// invents cache that will never exist.
describe('§C linear-attention layers (constant state)', () => {
  it('AC-28 — Kimi K3 splits 24 full MLA / 69 linear across 93 layers', () => {
    const m = model('kimi-k3');
    expect(m.layers).toBe(93);
    const s = layerSplit(m);
    expect(s).toEqual({ full: 24, windowed: 0, linear: 69 });
    expect(s.full + s.windowed + s.linear).toBe(m.layers); // every layer accounted for
  });

  it('AC-29 — only the 24 cached layers grow: 8.5 GiB at 1M, not 31.4', () => {
    const m = model('kimi-k3');
    const tokens = 1048576 * 0.6;
    const real = kvPerRequestBytes(m, 1, tokens) / GIB;
    const allFull = (kvPerTokenBytes(m, 1) * tokens) / GIB; // if all 93 layers were MLA
    expect(near(real, 8.5, 0.05)).toBe(true);
    expect(near(allFull, 31.4, 0.05)).toBe(true);
    expect(allFull / real).toBeGreaterThan(3.5);
  });

  it('AC-30 — the linear state is genuinely constant in context length', () => {
    const m = model('kimi-k3');
    const stateGib = (69 * 6291456) / GIB; // 69 layers x 6.29 MB
    // subtract the growing MLA term at two very different contexts; the remainder must match
    const at = (tok: number) => kvPerRequestBytes(m, 1, tok) / GIB - (24 * 576 * tok) / GIB;
    expect(at(1000)).toBeCloseTo(stateGib, 9);
    expect(at(600_000)).toBeCloseTo(stateGib, 9);
    expect(near(stateGib, 0.404, 0.02)).toBe(true); // ~414 MB, flat
  });

  it('AC-31 — Kimi K3 MXFP4 fits one 8x B300 node at its full 1M context', () => {
    const r = computeSizing(model('kimi-k3'), gpu('b300'), {
      quant: 'MXFP4', kv_dtype_bytes: 1, selected_ctx: 1048576, avg_context_utilisation: 0.6,
      target_concurrency: 8, mem_util_fraction: 0.9, gpus_per_node: 8,
    }) as FeasibleSizing;
    expect(r.ok).toBe(true);
    expect(r.tp).toBe(8);
    expect(r.gpus).toBe(8);
    expect(r.nodes).toBe(1); // the vLLM recipe's "at least 8x GB300"
    expect(near(r.weights_gb, 1389, 0.05)).toBe(true);
    expect(r.kv_windowed).toBe(true);
  });

  it('AC-34 — Qwen3.6-27B: 16 of 64 layers cached, gated delta-net for the rest', () => {
    const m = model('qwen36-27b');
    expect(layerSplit(m)).toEqual({ full: 16, windowed: 0, linear: 48 });
    const tokens = 262144 * 0.6;
    const real = kvPerRequestBytes(m, 1, tokens) / GIB;
    const allFull = (kvPerTokenBytes(m, 1) * tokens) / GIB;
    expect(near(real, 4.94, 0.05)).toBe(true);
    expect(near(allFull, 19.2, 0.05)).toBe(true);
    expect(allFull / real).toBeGreaterThan(3.8); // 64/16 on the cached term
  });

  it('AC-35 — its 248K vocab makes the fp16 tail ~30% of an INT4 checkpoint', () => {
    const m = model('qwen36-27b');
    expect(near(unquantisedParamsB(m)!, 2.543, 0.01)).toBe(true); // 2 x 248320 x 5120
    const w = weightsGb(m, 'INT4');
    const bodyOnly = paramBytesToGib(m.total_params_b, QB.INT4);
    expect(near(w, 16.6, 0.05)).toBe(true);
    expect(w / bodyOnly).toBeGreaterThan(1.25); // tail dominates on a big-vocab small model
  });

  it('AC-36 — FP8 serves on a single H100, matching the recipe', () => {
    const r = computeSizing(model('qwen36-27b'), gpu('h100'), {
      quant: 'FP8', kv_dtype_bytes: 1, selected_ctx: 262144, avg_context_utilisation: 0.6,
      target_concurrency: 8, mem_util_fraction: 0.9, gpus_per_node: 8,
    }) as FeasibleSizing;
    expect(r.ok).toBe(true);
    expect(r.tp).toBe(1);
    expect(r.gpus).toBe(1); // "FP8: single GPU"
    expect(r.kv_windowed).toBe(true);
  });

  it('AC-37 — INT4 fits a single 24 GB card at moderate context', () => {
    const r = computeSizing(model('qwen36-27b'), gpu('rtx4090'), {
      quant: 'INT4', kv_dtype_bytes: 1, selected_ctx: 32768, avg_context_utilisation: 0.6,
      target_concurrency: 1, mem_util_fraction: 0.9, gpus_per_node: 1,
    }) as FeasibleSizing;
    expect(r.ok).toBe(true);
    expect(r.tp).toBe(1); // "Int4: single 24GB GPU"
  });

  it('AC-32 — a model with no linear layers is unchanged', () => {
    const m = model('kimi-k2');
    expect(layerSplit(m)).toEqual({ full: 61, windowed: 0, linear: 0 });
    expect(kvPerRequestBytes(m, 1, 78643.2)).toBeCloseTo(kvPerTokenBytes(m, 1) * 78643.2, 6);
  });

  it('AC-33 — validation rejects a layer split that does not account for every layer', () => {
    const m = model('kimi-k3');
    // linear layers with no state size would be counted as free
    expect(modelSchema.safeParse({ ...m, linear_state_bytes_per_layer: undefined }).success).toBe(false);
    // full + linear over-subscribing the layer count
    expect(modelSchema.safeParse({ ...m, full_attention_layers: 40 }).success).toBe(false);
    // leftover layers with no window to size them
    expect(modelSchema.safeParse({ ...m, linear_attention_layers: 40, full_attention_layers: 24 }).success).toBe(false);
    expect(modelSchema.safeParse(m).success).toBe(true);
  });
});

// GGUF k-quants advertise a bit-width they do not deliver. These pin the published
// effective figures, cross-checked against real checkpoint sizes.
describe('§C GGUF whole-file quants', () => {
  it('AC-23 — Q4_K_M is 4.9 effective bits, not 4.0', () => {
    expect(QB.Q4_K_M).toBeCloseTo(0.61, 3);
    expect(QB.Q4_K_M / 0.5).toBeGreaterThan(1.2); // >20% above the nominal 4-bit assumption
    expect(QB.Q8_0).toBeGreaterThan(1); // 8.5 bits, not 8.0
    expect(QB.IQ4_XS).toBeCloseTo(4.25 / 8, 6);
  });

  it('AC-24 — Llama 3.3 70B Q4_K_M = 43.1 GB, matching the published checkpoint', () => {
    const gib = weightsGb(model('llama33-70b'), 'Q4_K_M');
    expect(near((gib * GIB) / 1e9, 43.1, 0.02)).toBe(true);
    // assuming a flat 0.5 B/param would undercount by ~8 GB — the documented GGUF trap
    const naive = (paramBytesToGib(70.6, 0.5) * GIB) / 1e9;
    expect((gib * GIB) / 1e9 - naive).toBeGreaterThan(7);
  });

  it('AC-25 — Llama 3.1 8B Q4_K_M = 4.9 GB', () => {
    const gib = weightsGb(model('llama31-8b'), 'Q4_K_M');
    expect(near((gib * GIB) / 1e9, 4.91, 0.02)).toBe(true);
  });

  it('AC-26 — GGUF figures are whole-file: no fp16 embedding tail added on top', () => {
    const m = model('llama33-70b');
    // whole-file quants ignore the tail entirely — total x bytes/param, nothing else
    expect(weightsGb(m, 'Q4_K_M')).toBeCloseTo(paramBytesToGib(m.total_params_b, QB.Q4_K_M), 9);
    // and a model with NO embedding geometry gives the identical answer (no fallback factor)
    const bare = { ...m, hidden_size: undefined, vocab_size: undefined };
    expect(weightsGb(bare, 'Q4_K_M')).toBeCloseTo(weightsGb(m, 'Q4_K_M'), 9);
    // ...unlike INT4, where the tail matters
    expect(weightsGb(bare, 'INT4')).not.toBeCloseTo(weightsGb(m, 'INT4'), 3);
  });

  it('AC-27 — GGUF sizing is never flagged as an estimate (the figure is measured)', () => {
    const r = computeSizing(model('llama33-70b'), gpu('rtx4090'), {
      quant: 'Q4_K_M', kv_dtype_bytes: 1, selected_ctx: 8192, avg_context_utilisation: 0.6,
      target_concurrency: 1, mem_util_fraction: 0.9, gpus_per_node: 1,
    }) as FeasibleSizing;
    expect(r.ok).toBe(true);
    expect(r.weights_estimated).toBe(false);
  });
});

describe('extended GPU catalog', () => {
  it('covers AMD Instinct and workstation SKUs alongside NVIDIA datacenter', () => {
    const ids = new Set(gpus.map((g) => g.id));
    for (const id of ['mi300x', 'mi325x', 'mi355x', 'rtxpro6000', 'rtx5090', 'rtx4090', 'l4']) {
      expect(ids.has(id)).toBe(true);
    }
    expect(gpu('mi325x').mem_gb).toBe(256);
    expect(gpus.every((g) => g.bw_tbs > 0 && g.mem_gb > 0)).toBe(true);
    expect(new Set(gpus.map((g) => g.id)).size).toBe(gpus.length); // unique ids
  });

  it('a 24 GB consumer card serves an 8B INT4 model on one GPU', () => {
    const r = computeSizing(model('llama31-8b'), gpu('rtx4090'), {
      quant: 'INT4', kv_dtype_bytes: 1, selected_ctx: 8192, avg_context_utilisation: 0.6,
      target_concurrency: 4, mem_util_fraction: 0.9, gpus_per_node: 1,
    }) as FeasibleSizing;
    expect(r.ok).toBe(true);
    expect(r.tp).toBe(1);
    expect(near(r.weights_gb, 5.34, 0.05)).toBe(true); // 3.4 GiB INT4 body + 1.96 GiB fp16 tail
  });

  it('a 24 GB consumer card cannot hold a 70B at FP16/128K even at TP8', () => {
    expect(computeSizing(model('llama33-70b'), gpu('rtx4090'), {
      quant: 'FP16', kv_dtype_bytes: 2, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 4, mem_util_fraction: 0.9, gpus_per_node: 1,
    }).ok).toBe(false);
  });
});

describe('reproducibility (AD-2a / FR-10)', () => {
  it('same inputs + same geometry recompute identically', () => {
    const input = { quant: 'FP8' as const, kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6, target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8 };
    const a = computeSizing(model('llama33-70b'), gpu('h200'), input);
    const b = computeSizing(model('llama33-70b'), gpu('h200'), input);
    expect(JSON.stringify(a, (_k, v) => (typeof v === 'bigint' ? v.toString() : v)))
      .toBe(JSON.stringify(b, (_k, v) => (typeof v === 'bigint' ? v.toString() : v)));
  });
});

describe('§F validation (AD-14)', () => {
  it('accepts every seeded model', () => {
    for (const m of models) expect(modelSchema.safeParse(m).success).toBe(true);
  });
  it('accepts every seeded GPU SKU', () => {
    for (const g of gpus) expect(gpuSkuSchema.safeParse(g).success).toBe(true);
  });
  it('rejects a GQA model with kv_heads=0 (divide-by-zero guard)', () => {
    const bad = { ...model('llama33-70b'), kv_heads: 0 };
    expect(modelSchema.safeParse(bad).success).toBe(false);
  });
  it('rejects active_params_b > total_params_b', () => {
    const bad = { ...model('llama33-70b'), active_params_b: 999 };
    expect(modelSchema.safeParse(bad).success).toBe(false);
  });
  it('rejects max_ctx over 8,388,608', () => {
    const bad = { ...model('llama33-70b'), max_ctx: 9_000_000 };
    expect(modelSchema.safeParse(bad).success).toBe(false);
  });
});

describe('reconciliation invariant + hard-block (FR-19..21, AD-10)', () => {
  const gpuById = new Map(gpus.map((g) => [g.id, g]));
  it('committed + available = fleet total per SKU; flags over-commitment; no cross-SKU masking', () => {
    const fleet = [
      { gpu_sku_id: 'h200', gpus_per_node: 8, node_count: 5 }, // 40 H200
      { gpu_sku_id: 'h100', gpus_per_node: 8, node_count: 2 }, // 16 H100
    ];
    const commitments = [
      { gpu_sku_id: 'h200', gpus: 28 }, // 70%
      { gpu_sku_id: 'h100', gpus: 18 }, // 112.5% — over
    ];
    const rows = reconcile(fleet, commitments, gpuById);
    for (const r of rows) expect(r.committed_bytes + r.available_bytes).toBe(r.fleet_bytes);
    const h200 = rows.find((r) => r.gpu_sku_id === 'h200')!;
    const h100 = rows.find((r) => r.gpu_sku_id === 'h100')!;
    expect(h200.over_committed).toBe(false);
    expect(h100.over_committed).toBe(true); // H200 surplus never masks H100 shortage
    // headroom check verdicts (FR-21)
    expect(headroomCheck('h200', 10, rows).verdict).toBe('fit');
    expect(headroomCheck('h100', 4, rows).verdict).toBe('shortage');
    expect(headroomCheck('b200', 4, rows).verdict).toBe('sku_absent');
  });
});
