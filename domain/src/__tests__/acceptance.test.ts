// Addendum §C sizing acceptance vectors (AC-1…AC-5) — the PRD §18.2 gate (NFR-M-3).
// Tolerances: integer counts EXACT; GB within ±5%; concurrency/pod within ±10%. Failures block release.
import { describe, it, expect } from 'vitest';
import { computeSizing } from '../engine.js';
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
    expect(near(r.weights_gb, 72, 0.05)).toBe(true);
    expect(near(r.kv_per_token_gb * 1024, 0.156, 0.05)).toBe(true); // ~0.156 MB/token
    expect(near(r.kv_per_request_gb, 12.0, 0.05)).toBe(true);
    expect(near(r.concurrency_per_pod, 14, 0.1)).toBe(true);
    expect(r.pods).toBe(5);
    expect(r.gpus).toBe(10);
    expect(r.nodes).toBe(2);
  });

  it('AC-2 — GLM-5.2 744B FP8/KV-FP8 128K 60% conc64 H200 → TP8, 32 GPUs', () => {
    const r = computeSizing(model('glm52'), gpu('h200'), {
      quant: 'FP8', kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8,
    }) as FeasibleSizing;
    expect(r.ok).toBe(true);
    expect(r.tp).toBe(8);
    expect(near(r.weights_gb, 759, 0.05)).toBe(true);
    expect(near(r.free_gb, 236, 0.05)).toBe(true); // free KV budget/replica
    expect(near(r.kv_per_request_gb, 11.7, 0.05)).toBe(true);
    expect(near(r.concurrency_per_pod, 20, 0.1)).toBe(true);
    expect(r.pods).toBe(4);
    expect(r.gpus).toBe(32);
  });

  it('AC-3 — DeepSeek-V3 671B (MLA) FP8/KV-FP8 128K 60% conc64 H200 → TP8, 8 GPUs, conc/pod ≥100', () => {
    const r = computeSizing(model('dsv3'), gpu('h200'), {
      quant: 'FP8', kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8,
    }) as FeasibleSizing;
    expect(r.ok).toBe(true);
    expect(r.tp).toBe(8);
    expect(near(r.weights_gb, 684, 0.05)).toBe(true);
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
