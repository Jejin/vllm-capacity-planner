// Measured vLLM profile (§5.1). The rule under test: a measurement REPLACES the estimate where
// it applies, is refused where it does not, and is never blended with it.
import { describe, it, expect } from 'vitest';
import { reconcileMeasured, parseProfile, varianceLabel } from '../measured.js';
import { computeSizing, GIB } from '../engine.js';
import { seedCatalog } from '../seed.js';
import type { FeasibleSizing, MeasuredProfile } from '../types.js';

const { models, gpus } = seedCatalog();
const M = (i: string) => models.find((x) => x.id === i)!;
const G = (i: string) => gpus.find((x) => x.id === i)!;
const base = {
  quant: 'FP8' as const, kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6,
  target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8,
};

describe('measured memory profile', () => {
  const plain = computeSizing(M('llama33-70b'), G('h200'), base) as FeasibleSizing;

  it('absent by default, and the estimate stands', () => {
    expect(plain.measured.status).toBe('absent');
    expect(plain.measured.measured_free_gb).toBeNull();
    expect(plain.free_gb).toBeCloseTo(plain.measured.estimated_free_gb, 6);
  });

  it('a matching profile replaces the estimate and reports the variance', () => {
    // 10% more room than modelled, per GPU, across the plan's own TP width
    const perGpu = ((plain.free_gb / plain.tp) * 1.1 * GIB);
    const profile: MeasuredProfile = {
      total_memory_bytes: 141 * GIB,
      available_kv_cache_bytes: perGpu,
      gpu_sku_id: 'h200',
      tensor_parallel_size: plain.tp,
    };
    const r = computeSizing(M('llama33-70b'), G('h200'), { ...base, measured: profile }) as FeasibleSizing;
    expect(r.measured.status).toBe('applied');
    expect(r.measured.variance!).toBeCloseTo(0.1, 2);
    // the estimate is preserved, not overwritten in place
    expect(r.measured.estimated_free_gb).toBeCloseTo(plain.free_gb, 4);
    // and it is REPLACED, not averaged: 10% more room, not 5%
    expect(r.free_gb / plain.free_gb).toBeCloseTo(1.1, 3);
    expect(r.concurrency_per_pod).toBeGreaterThan(plain.concurrency_per_pod);
    expect(varianceLabel(r.measured)).toMatch(/conservative/);
  });

  it('refuses a profile from a different shape rather than scaling it', () => {
    const wrongTp = reconcileMeasured(
      { total_memory_bytes: 141 * GIB, available_kv_cache_bytes: 40 * GIB, tensor_parallel_size: 8 },
      G('h200'), 2, 100,
    );
    expect(wrongTp.status).toBe('mismatch');
    expect(wrongTp.reason).toMatch(/TP 8.*TP 2/);
    expect(wrongTp.measured_free_gb).toBeNull(); // nothing applied

    const wrongGpu = reconcileMeasured(
      { total_memory_bytes: 80 * GIB, available_kv_cache_bytes: 20 * GIB, gpu_sku_id: 'h100' },
      G('h200'), 2, 100,
    );
    expect(wrongGpu.status).toBe('mismatch');
    expect(wrongGpu.reason).toMatch(/another card/);
  });

  it('rejects internally inconsistent figures', () => {
    const impossible = reconcileMeasured(
      { total_memory_bytes: 10 * GIB, available_kv_cache_bytes: 40 * GIB },
      G('h200'), 1, 100,
    );
    expect(impossible.status).toBe('mismatch');
    expect(impossible.reason).toMatch(/not from the same run/);
  });

  it('parses the vLLM startup log, which is how anyone actually has these numbers', () => {
    const log = `
      INFO 08-02 09:14:22 gpu_worker.py:255] Available KV cache memory: 45.67 GiB
      INFO 08-02 09:14:22 gpu_worker.py:250] model weights take 65.43 GiB; non-torch memory takes 1.10 GiB;
      PyTorch activation peak memory takes 1.23 GiB
    `;
    const { profile, error } = parseProfile(log);
    expect(error).toBeNull();
    expect(profile!.available_kv_cache_bytes).toBeCloseTo(45.67 * GIB, 0);
    expect(profile!.model_memory_bytes).toBeCloseTo(65.43 * GIB, 0);
    expect(profile!.peak_activation_bytes).toBeCloseTo(1.23 * GIB, 0);
  });

  it('parses the JSON form, and says what is wrong when it cannot', () => {
    const ok = parseProfile('{"available_kv_cache_bytes": 1234567890, "total_memory_bytes": 151000000000}');
    expect(ok.profile!.available_kv_cache_bytes).toBe(1234567890);

    expect(parseProfile('{"total_memory_bytes": 1}').error).toMatch(/available_kv_cache_bytes/);
    expect(parseProfile('{oops').error).toMatch(/Not valid JSON/);
    expect(parseProfile('nothing useful here').error).toMatch(/No KV cache figure/);
    expect(parseProfile('   ').profile).toBeNull(); // empty is not an error
  });
});
