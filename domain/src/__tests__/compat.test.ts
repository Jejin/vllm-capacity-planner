// Runtime support (§4.4) — "fits in HBM" and "will actually run" are separate verdicts.
import { describe, it, expect } from 'vitest';
import { runtimeSupport, COMPAT_SOURCE } from '../compat.js';
import { computeSizing } from '../engine.js';
import { seedCatalog } from '../seed.js';
import type { FeasibleSizing } from '../types.js';

const { models, gpus } = seedCatalog();
const M = (i: string) => models.find((x) => x.id === i)!;
const G = (i: string) => gpus.find((x) => x.id === i)!;
const at = (q: any, perNode = 8) => ({ quant: q, gpus_per_node: perNode });

describe('runtime support verdict', () => {
  it('every seeded GPU declares an architecture, so nothing reports unverified by accident', () => {
    for (const g of gpus) expect(g.arch, `${g.id} has no arch`).toBeTruthy();
  });

  it('native paths are clean: FP8 on Hopper, NVFP4 on Blackwell, MXFP4 on MI355X', () => {
    expect(runtimeSupport(M('llama33-70b'), G('h200'), at('FP8')).level).toBe('supported');
    expect(runtimeSupport(M('glm52'), G('b300'), at('NVFP4')).level).toBe('supported');
    expect(runtimeSupport(M('gptoss-120b'), G('mi355x'), at('MXFP4')).level).toBe('supported');
  });

  it('a weight-only fallback is degraded, not unsupported — the memory plan still holds', () => {
    // vLLM does not refuse NVFP4 on Hopper; it drops to W4A16 via Marlin and warns. The weights
    // really are 4-bit, so sizing is unaffected — it is the throughput estimate that breaks.
    const s = runtimeSupport(M('glm52'), G('h200'), at('NVFP4'));
    expect(s.level).toBe('degraded');
    expect(s.findings[0].detail).toMatch(/weight-only \(W4A16\)/);
    expect(s.findings[0].detail).toMatch(/memory plan holds/);
    expect(s.findings[0].alternative).toMatch(/Blackwell/);

    // and the plan it describes is still perfectly feasible
    const plan = computeSizing(M('glm52'), G('h200'), {
      quant: 'NVFP4', kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 8, mem_util_fraction: 0.9, gpus_per_node: 8,
    }) as FeasibleSizing;
    expect(plan.ok).toBe(true);
  });

  it('FP8 on Ampere is degraded — the compute path is not what the estimate assumes', () => {
    const s = runtimeSupport(M('llama33-70b'), G('a100s'), at('FP8'));
    expect(s.level).toBe('degraded');
    expect(s.findings[0].detail).toMatch(/W8A16/);
    expect(s.findings[0].alternative).toMatch(/INT8/);
  });

  it('AWQ/GPTQ-style INT4 and INT8 have no AMD path at all', () => {
    for (const q of ['INT4', 'INT8'] as const) {
      const s = runtimeSupport(M('qwen3-32b'), G('mi300x'), at(q));
      expect(s.level).toBe('unsupported');
      expect(s.findings[0].alternative).toMatch(/FP8/);
    }
  });

  it('an uncovered combination says so instead of guessing either way', () => {
    // vLLM's published matrix stops at Hopper for the INT8/INT4 rows, so Blackwell is genuinely
    // unknown. Reporting "supported" would be a fabrication and "unsupported" would be a lie.
    const s = runtimeSupport(M('qwen3-32b'), G('b300'), at('INT4'));
    expect(s.level).toBe('unverified');
    expect(s.findings[0].detail).toMatch(/does not cover/);
  });

  it('a GPU with no architecture is unverified, not assumed good', () => {
    const unknown = { ...G('h200'), arch: undefined };
    const s = runtimeSupport(M('llama33-70b'), unknown, at('FP8'));
    expect(s.level).toBe('unverified');
    expect(s.findings[0].alternative).toMatch(/catalogue/);
  });

  it('topology is part of the verdict: node-spanning TP and PCIe-only consumer cards', () => {
    // TP16 on 8-GPU nodes: the format is fine, the shape is not
    const span = runtimeSupport(M('dsv3'), G('h200'), at('FP8', 8), 16);
    expect(span.level).toBe('degraded');
    expect(span.findings.some((f) => /spans nodes/.test(f.detail))).toBe(true);

    // consumer Blackwell has no NVLink, so the all-reduce rides PCIe
    const pcie = runtimeSupport(M('qwen3-32b'), G('rtx5090'), at('FP8', 4), 4);
    expect(pcie.findings.some((f) => /PCIe/.test(f.detail))).toBe(true);
    // ...and a single card carries no collective at all
    const one = runtimeSupport(M('qwen3-32b'), G('rtx5090'), at('FP8', 1), 1);
    expect(one.findings.some((f) => /PCIe/.test(f.detail))).toBe(false);
  });

  it('the worst constraint decides the verdict, and every constraint is still listed', () => {
    // INT8 on AMD (unsupported) plus a node-spanning replica (degraded)
    const s = runtimeSupport(M('dsv3'), G('mi300x'), at('INT8', 8), 16);
    expect(s.level).toBe('unsupported');
    expect(s.findings.length).toBe(2);
    expect(s.findings.map((f) => f.level).sort()).toEqual(['degraded', 'unsupported']);
  });

  it('carries its provenance, so a stale rule is visible rather than folklore', () => {
    const s = runtimeSupport(M('llama33-70b'), G('h200'), at('FP8'));
    expect(s.source.runtime).toBe('vLLM');
    expect(s.source.docs_snapshot).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(COMPAT_SOURCE.urls.length).toBeGreaterThan(0);
  });
});
