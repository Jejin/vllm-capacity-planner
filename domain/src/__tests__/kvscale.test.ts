// FP8 KV scale provenance (§4.3). The planner defaults to a 1-byte KV cache, so this is the
// verdict on the DEFAULT configuration of every catalogued model — not an edge case.
import { describe, it, expect } from 'vitest';
import { kvScalePolicy } from '../kvscale.js';
import { serveCommand } from '../serve.js';
import { computeSizing } from '../engine.js';
import { seedCatalog } from '../seed.js';
import type { FeasibleSizing, Model } from '../types.js';

const { models, gpus } = seedCatalog();
const M = (i: string) => models.find((x) => x.id === i)!;
const G = (i: string) => gpus.find((x) => x.id === i)!;

describe('FP8 KV cache scale policy', () => {
  it('does not apply to a 16-bit KV cache', () => {
    const v = kvScalePolicy(M('llama33-70b'), 'FP8', 2);
    expect(v.applies).toBe(false);
    expect(v.level).toBe('ok');
  });

  it('every seeded artifact is recorded as shipping no K/V scales', () => {
    // Measured, not assumed: the safetensors index of all 31 catalogued artifacts was checked
    // for k_scale / v_scale / kv_scale, with the probe validated against checkpoints that do
    // carry them. A future artifact that ships scales must say so on its own entry.
    for (const m of models) {
      for (const [q, v] of Object.entries(m.deployments ?? {})) {
        expect(v!.kv_scale_source, `${m.id}:${q}`).toBe('none');
      }
    }
  });

  it('an artifact with no scales warns, and names the three ways out', () => {
    const v = kvScalePolicy(M('llama33-70b'), 'FP8', 1);
    expect(v.applies).toBe(true);
    expect(v.level).toBe('warn');
    expect(v.detail).toMatch(/set(s)? every quantisation scale to 1\.0/);
    // the memory plan is NOT what is in doubt here
    expect(v.detail).toMatch(/memory plan is unaffected/);
    expect(v.remedy).toMatch(/calculate_kv_scales/);
    expect(v.remedy).toMatch(/16 bits/);
  });

  it('an unrecorded artifact warns too — silence is not approval', () => {
    const bare: Model = { ...M('llama33-70b'), deployments: { FP8: { source: 'none' } } };
    const v = kvScalePolicy(bare, 'FP8', 1);
    expect(v.source).toBe('unknown');
    expect(v.level).toBe('warn');
  });

  it('grades the calibrated, checkpoint and warm-up paths differently', () => {
    const at = (src: any) => kvScalePolicy(
      { ...M('llama33-70b'), deployments: { FP8: { source: 'none', kv_scale_source: src } } } as Model,
      'FP8', 1,
    );
    expect(at('calibrated').level).toBe('ok');
    expect(at('checkpoint').level).toBe('ok');
    // warm-up scales are better than unit ones and worse than a real calibration
    const warm = at('runtime');
    expect(warm.level).toBe('caution');
    expect(warm.detail).toMatch(/single batch of random tokens/);
  });

  it('the verdict follows the artifact, so it can differ per precision', () => {
    const m: Model = {
      ...M('llama33-70b'),
      deployments: {
        FP8: { source: 'checkpoint', hf_id: 'owner/fp8', kv_scale_source: 'calibrated' },
        INT4: { source: 'checkpoint', hf_id: 'owner/int4', kv_scale_source: 'none' },
      },
    };
    expect(kvScalePolicy(m, 'FP8', 1).level).toBe('ok');
    expect(kvScalePolicy(m, 'INT4', 1).level).toBe('warn');
  });

  it('the launch command asks for warm-up scales only when that is the declared source', () => {
    const input = {
      quant: 'FP8' as const, kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8,
    };
    const s = computeSizing(M('llama33-70b'), G('h200'), input) as FeasibleSizing;

    // seeded artifact: no scales, so nothing is silently added — the note carries the warning
    const asIs = serveCommand(M('llama33-70b'), input, s);
    expect(asIs.argv).not.toContain('--calculate-kv-scales');
    expect(asIs.notes.join(' ')).toMatch(/Uncalibrated FP8 KV cache/);

    // declared runtime source: the flag is what makes that true, so it is passed
    const runtime: Model = {
      ...M('llama33-70b'),
      deployments: { FP8: { source: 'checkpoint', hf_id: 'owner/fp8', kv_scale_source: 'runtime' } },
    };
    expect(serveCommand(runtime, input, s).argv).toContain('--calculate-kv-scales');

    // and a 16-bit cache never mentions scales at all
    const in16 = { ...input, kv_dtype_bytes: 2 };
    const s16 = computeSizing(M('llama33-70b'), G('h200'), in16) as FeasibleSizing;
    expect(serveCommand(M('llama33-70b'), in16, s16).notes.join(' ')).not.toMatch(/scale/i);
  });
});
