// Addendum §C sizing acceptance vectors (AC-1…AC-5) — the PRD §18.2 gate (NFR-M-3).
// Tolerances: integer counts EXACT; memory within ±5%; concurrency/pod within ±10%. Failures block release.
//
// ALL memory figures below are GiB (2^30 bytes) — the engine's single unit, matching nvidia-smi
// and GpuSku.mem_gb. Vectors were re-pinned when the engine stopped mixing 1e9-byte GB (weights)
// with 2^30-byte GiB (KV, capacity), which had inflated weights ~7.4% against GPU capacity.
import { describe, it, expect } from 'vitest';
import { computeSizing, weightsGb, activeWeightsGb, unquantisedParamsB, paramBytesToGib, kvPerTokenBytes, GIB, QB, TIGHT_HEADROOM, WEIGHT_OVERHEAD, RUNTIME_GB } from '../engine.js';
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
  it('AC-1 — Llama 3.3 70B FP8/KV-FP8 128K 60% conc64 H200 → TP2, 10 GPUs, 2 nodes', () => {
    const r = computeSizing(model('llama33-70b'), gpu('h200'), {
      quant: 'FP8', kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8,
    }) as FeasibleSizing;
    expect(r.ok).toBe(true);
    expect(r.tp).toBe(2);
    expect(near(r.weights_gb, 67.7, 0.05)).toBe(true);
    expect(near(r.kv_per_token_gb * 1024, 0.156, 0.05)).toBe(true); // ~0.156 MiB/token
    expect(near(r.kv_per_request_gb, 12.0, 0.05)).toBe(true);
    expect(near(r.concurrency_per_pod, 15, 0.1)).toBe(true);
    expect(r.pods).toBe(5);
    expect(r.gpus).toBe(10);
    expect(r.nodes).toBe(2);
  });

  it('AC-2 — GLM-5.2 744B FP8/KV-FP8 128K 60% conc64 H200 → TP8, 24 GPUs', () => {
    const r = computeSizing(model('glm52'), gpu('h200'), {
      quant: 'FP8', kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8,
    }) as FeasibleSizing;
    expect(r.ok).toBe(true);
    expect(r.tp).toBe(8);
    expect(near(r.weights_gb, 694, 0.05)).toBe(true); // 16-bit embedding tail split out, in GiB
    expect(near(r.free_gb, 301, 0.05)).toBe(true); // free KV budget/replica
    expect(near(r.kv_per_request_gb, 11.7, 0.05)).toBe(true);
    expect(near(r.concurrency_per_pod, 25, 0.1)).toBe(true);
    expect(r.pods).toBe(3);
    expect(r.gpus).toBe(24);
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
  it('AC-12 — GPT-OSS 120B MXFP4 on ONE H100 at 128K is feasible but tight', () => {
    const r = computeSizing(model('gptoss-120b'), gpu('h100'), {
      quant: 'MXFP4', kv_dtype_bytes: 2, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 1, mem_util_fraction: 0.9, gpus_per_node: 8,
    }) as FeasibleSizing;
    expect(r.ok).toBe(true);
    expect(r.tp).toBe(1);
    expect(r.tight).toBe(true);
    expect(near(r.headroom_fraction, 0.067, 0.1)).toBe(true); // ~59.5 GiB weights in 69.5 GiB usable
    expect(r.headroom_fraction).toBeLessThan(TIGHT_HEADROOM);
    expect(r.headroom_fraction).toBeGreaterThan(0); // still fits — not infeasible
    expect(r.concurrency_per_pod).toBe(1); // no room to batch a second request
  });

  it('AC-13 — the same model on an H200 has real headroom and is not tight', () => {
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
