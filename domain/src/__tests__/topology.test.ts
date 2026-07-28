import { describe, it, expect } from 'vitest';
import { topologyLayout } from '../topology.js';
import { computeSizing } from '../engine.js';
import { seedCatalog } from '../seed.js';
import type { FeasibleSizing } from '../types.js';

const { models, gpus } = seedCatalog();
const M = (i: string) => models.find((x) => x.id === i)!;
const G = (i: string) => gpus.find((x) => x.id === i)!;
const size = (m: string, g: string, o: any) =>
  computeSizing(M(m), G(g), {
    kv_dtype_bytes: 1, avg_context_utilisation: 0.6, mem_util_fraction: 0.9, ...o,
  }) as FeasibleSizing;

describe('topology layout', () => {
  it('places every GPU of every node, marking the unused ones', () => {
    const r = size('llama33-70b', 'h200', { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64, gpus_per_node: 8 });
    const t = topologyLayout(r, 8);
    expect(t.gpus).toHaveLength(t.shown * 8);
    expect(t.gpus.filter((g) => g.used)).toHaveLength(r.gpus);
    expect(t.gpus.filter((g) => !g.used).every((g) => g.pod === null)).toBe(true);
  });

  it('assigns each replica a contiguous run of GPUs', () => {
    const r = size('llama33-70b', 'h200', { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64, gpus_per_node: 8 });
    const t = topologyLayout(r, 8);
    for (const pod of t.pods) {
      const mine = t.gpus.filter((g) => g.pod === pod.p).map((g) => g.g);
      expect(mine).toHaveLength(r.tp);
      expect(Math.max(...mine) - Math.min(...mine)).toBe(r.tp - 1); // no gaps
    }
  });

  it('flags a replica that straddles a node boundary — and only then', () => {
    const single = size('llama33-70b', 'h200', { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64, gpus_per_node: 8 });
    expect(single.multi_node).toBe(false);
    expect(topologyLayout(single, 8).pods.some((p) => p.spans)).toBe(false);

    const spanning = size('kimi-k3', 'h200', { quant: 'MXFP4', selected_ctx: 1048576, target_concurrency: 8, gpus_per_node: 8 });
    expect(spanning.multi_node).toBe(true);
    const t = topologyLayout(spanning, 8);
    expect(t.pods.some((p) => p.spans)).toBe(true);
    expect(t.multi).toBe(true); // a fabric is drawn
  });

  it('a narrow bar labels outside itself rather than overflowing', () => {
    const r = size('gptoss-120b', 'h200', { quant: 'MXFP4', selected_ctx: 131072, target_concurrency: 8, gpus_per_node: 8 });
    expect(r.tp).toBe(1);
    const t = topologyLayout(r, 8);
    expect(t.pods[0].x1 - t.pods[0].x0).toBe(t.cell); // one cell wide
    expect(t.pods[0].labelInside).toBe(false);
  });

  it('drops the GPU-count caption when node boxes are too narrow for it', () => {
    const r = size('llama33-70b', 'h100', { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64, gpus_per_node: 2 });
    expect(topologyLayout(r, 2).nodeLabelFull).toBe(false);
    expect(topologyLayout(r, 8).nodeLabelFull).toBe(true);
  });

  it('caps the node boxes drawn and says how many were left out', () => {
    const r = size('llama33-70b', 'h100', { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64, gpus_per_node: 2 });
    expect(r.nodes).toBeGreaterThan(4);
    const t = topologyLayout(r, 2);
    expect(t.shown).toBe(4);
    expect(t.truncated).toBe(true);
    expect(t.hiddenNodes).toBe(r.nodes - 4);
  });

  it('leaves slack past the drawing area so right-hand labels are not clipped', () => {
    const r = size('llama33-70b', 'h200', { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64, gpus_per_node: 8 });
    const t = topologyLayout(r, 8);
    expect(t.width).toBeGreaterThan(t.contentW);
    // every drawn element stays inside the viewBox
    for (const g of t.gpus) expect(g.x + t.cell).toBeLessThanOrEqual(t.width);
    for (const p of t.pods) expect(p.x1).toBeLessThanOrEqual(t.width);
  });

  it('stacks the fabric only when there is more than one node', () => {
    const one = topologyLayout(size('gptoss-120b', 'h200', { quant: 'MXFP4', selected_ctx: 131072, target_concurrency: 8, gpus_per_node: 8 }), 8);
    expect(one.multi).toBe(false);
    expect(one.storeY - one.switchY).toBeLessThan(56); // no switch band reserved
  });
});
