// Workload shape (§7): one "average utilisation" figure cannot size memory and time latency at
// once. Given a distribution they separate — KV at P95, prefill at P50.
import { describe, it, expect } from 'vitest';
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

describe('workload distribution', () => {
  it('without one, both questions use the single utilisation figure — nothing re-sizes', () => {
    const r = computeSizing(M('llama33-70b'), G('h200'), base) as FeasibleSizing;
    const expected = 131072 * 0.6;
    expect(r.kv_tokens).toBeCloseTo(expected, 6);
    expect(r.prefill_tokens).toBeCloseTo(expected, 6);
  });

  it('with one, KV is sized at P95 and prefill is timed at P50', () => {
    const workload = { prompt_p50: 4_000, prompt_p95: 60_000, output_p50: 500, output_p95: 8_000 };
    const r = computeSizing(M('llama33-70b'), G('h200'), { ...base, workload }) as FeasibleSizing;
    expect(r.kv_tokens).toBe(68_000); // 60k prompt + 8k output — the cache must hold the tail
    expect(r.prefill_tokens).toBe(4_000); // the typical request is what TTFT should describe
  });

  it('the split moves the two numbers in opposite directions, which is the point', () => {
    // A workload whose tail is long but whose typical request is short: more memory per
    // session than the flat 60% assumption, and a far quicker first token.
    const flat = computeSizing(M('llama33-70b'), G('h200'), base) as FeasibleSizing;
    const shaped = computeSizing(M('llama33-70b'), G('h200'), {
      ...base,
      workload: { prompt_p50: 2_000, prompt_p95: 120_000, output_p50: 400, output_p95: 4_000 },
    }) as FeasibleSizing;

    expect(shaped.kv_per_request_gb).toBeGreaterThan(flat.kv_per_request_gb); // sized for the tail
    expect(shaped.concurrency_per_pod).toBeLessThan(flat.concurrency_per_pod);
    expect(shaped.ttft_ms).toBeLessThan(flat.ttft_ms); // timed for the typical request
  });

  it('a workload with no tail behaves like a flat one at the same length', () => {
    const t = 40_000;
    const shaped = computeSizing(M('llama33-70b'), G('h200'), {
      ...base, workload: { prompt_p50: t, prompt_p95: t, output_p50: 0, output_p95: 0 },
    }) as FeasibleSizing;
    const flat = computeSizing(M('llama33-70b'), G('h200'), {
      ...base, selected_ctx: 131072, avg_context_utilisation: t / 131072,
    }) as FeasibleSizing;
    expect(shaped.kv_per_request_gb).toBeCloseTo(flat.kv_per_request_gb, 6);
    expect(shaped.ttft_ms).toBe(flat.ttft_ms);
  });
});
