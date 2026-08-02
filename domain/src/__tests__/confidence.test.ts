// Confidence tiers (§6.1). The point of this module is that the assumptions are PER-PLAN, so
// the tests are mostly about which ones appear and which do not.
import { describe, it, expect } from 'vitest';
import { throughputConfidence, ttftConfidence } from '../confidence.js';
import { computeSizing } from '../engine.js';
import { seedCatalog } from '../seed.js';
import type { FeasibleSizing } from '../types.js';

const { models, gpus } = seedCatalog();
const M = (i: string) => models.find((x) => x.id === i)!;
const G = (i: string) => gpus.find((x) => x.id === i)!;
const base = {
  quant: 'FP8' as const, kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6,
  target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8,
};
const size = (mid: string, gid: string, over: any = {}) =>
  computeSizing(M(mid), G(gid), { ...base, ...over }) as FeasibleSizing;
const labels = (c: { assumptions: { label: string }[] }) => c.assumptions.map((a) => a.label);

describe('confidence tiers', () => {
  it('a withheld figure reports the tier, not a band', () => {
    const r = size('glm52', 'h100', { selected_ctx: 1_048_576, target_concurrency: 8 });
    expect(r.throughput_suppressed).toBeTruthy();
    const c = throughputConfidence(M('glm52'), G('h100'), base, r, 8);
    expect(c.tier).toBe('withheld');
    expect(c.band).toBeNull();
    expect(c.assumptions).toEqual([]); // nothing to qualify — there is no number
    expect(ttftConfidence(M('glm52'), G('h100'), base, r).tier).toBe('withheld');
  });

  it('everything else is `estimated`, and says why it cannot be better', () => {
    const r = size('llama33-70b', 'h200');
    const c = throughputConfidence(M('llama33-70b'), G('h200'), base, r, 17);
    expect(c.tier).toBe('estimated');
    expect(c.band).toBe('±40%');
    // measured/calibrated exist in the type as a destination; nothing reaches them yet, and the
    // basis says so rather than letting "estimated" read like a grade
    expect(c.basis).toMatch(/no benchmark/i);
  });

  it('MoE routing skew is named for an MoE plan and absent for a dense one', () => {
    const moe = size('dsv3', 'h200');
    const cm = throughputConfidence(M('dsv3'), G('h200'), base, moe, 64);
    const routing = cm.assumptions.find((a) => a.label === 'uniform expert routing')!;
    expect(routing).toBeTruthy();
    // skew touches FEWER experts, so the modelled traffic is too high and throughput reads low
    expect(routing.bias).toBe('reads low');
    expect(routing.detail).toMatch(/\d+% of 256 experts/);

    const dense = size('llama33-70b', 'h200');
    expect(labels(throughputConfidence(M('llama33-70b'), G('h200'), base, dense, 17)))
      .not.toContain('uniform expert routing');
  });

  it('the unmodelled collective latency is named only when there is a collective', () => {
    const tp1 = size('llama31-8b', 'h200', { target_concurrency: 1 });
    expect(tp1.tp).toBe(1);
    expect(labels(throughputConfidence(M('llama31-8b'), G('h200'), base, tp1, 1)))
      .not.toContain('collective latency unmodelled');

    const wide = size('llama33-70b', 'h200');
    expect(wide.tp).toBeGreaterThan(1);
    const c = throughputConfidence(M('llama33-70b'), G('h200'), base, wide, 17);
    const coll = c.assumptions.find((a) => a.label === 'collective latency unmodelled')!;
    expect(coll.bias).toBe('reads high'); // real is slower than modelled
    expect(coll.detail).toMatch(/160 launches per step/); // 2 x 80 layers
  });

  it('TTFT names the weight-streaming floor only when the SKU has no FLOPS figure', () => {
    const noFlops = { ...G('h200'), tflops_fp16: undefined };
    const r = computeSizing(M('llama33-70b'), noFlops, base) as FeasibleSizing;
    const c = ttftConfidence(M('llama33-70b'), noFlops, base, r);
    expect(labels(c)).toContain('no FLOPS figure for this SKU');
    expect(labels(c)).not.toContain('prefill MFU 0.4');

    const ok = ttftConfidence(M('llama33-70b'), G('h200'), base, size('llama33-70b', 'h200'));
    expect(labels(ok)).toContain('prefill MFU 0.4');
    expect(labels(ok)).not.toContain('no FLOPS figure for this SKU');
  });

  it('every assumption declares which way it biases the figure', () => {
    const r = size('dsv3', 'h200');
    for (const c of [throughputConfidence(M('dsv3'), G('h200'), base, r, 64), ttftConfidence(M('dsv3'), G('h200'), base, r)]) {
      expect(c.assumptions.length).toBeGreaterThan(0);
      for (const a of c.assumptions) {
        expect(['reads high', 'reads low', 'unknown']).toContain(a.bias);
        expect(a.detail.length).toBeGreaterThan(20); // a label alone is not an explanation
      }
    }
  });
});
