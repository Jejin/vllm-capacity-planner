import { describe, it, expect } from 'vitest';
import { recipeToModel, parseParamCount, tpFromArgs, crossCheckVram } from '../recipes.js';
import { modelSchema } from '../schema.js';
import { hfConfigToModel } from '../hf.js';
import { weightsGb } from '../engine.js';
import { seedCatalog } from '../seed.js';

describe('recipe parsing', () => {
  it('parses parameter counts across units', () => {
    expect(parseParamCount('36B')).toBe(36);
    expect(parseParamCount('2.8T')).toBe(2800);
    expect(parseParamCount('1T')).toBe(1000);
    expect(parseParamCount('743B')).toBe(743);
    expect(parseParamCount('500M')).toBe(0.5);
  });
  it('refuses prose rather than guessing', () => {
    // Kimi K3's active_parameters is literally this
    expect(parseParamCount('16 experts/token + shared (of 896 routed)')).toBeNull();
    expect(parseParamCount(undefined)).toBeNull();
    expect(parseParamCount('')).toBeNull();
  });
  it('extracts tensor-parallel size from variant args', () => {
    expect(tpFromArgs(['--tensor-parallel-size', '8'])).toBe(8);
    expect(tpFromArgs(['--foo', '--tensor-parallel-size', '4', '--bar'])).toBe(4);
    expect(tpFromArgs(['--tensor-parallel-size'])).toBeNull();
    expect(tpFromArgs([])).toBeNull();
  });
});

describe('recipe → §F mapping', () => {
  const glm52 = {
    hf_id: 'zai-org/GLM-5.2',
    meta: { title: 'GLM-5.2' },
    model: { parameter_count: '743B', active_parameters: '39B', context_length: 1048576, min_vllm_version: '0.17.0' },
    variants: {
      default: { precision: 'fp8', vram_minimum_gb: 893, extra_args: ['--tensor-parallel-size', '8'] },
      mxfp4: { precision: 'mxfp4', vram_minimum_gb: 446 },
      nvfp4: { precision: 'nvfp4', vram_minimum_gb: 558 },
      bf16: { precision: 'bf16', vram_minimum_gb: 1786, extra_args: ['--tensor-parallel-size', '16'] },
    },
  };

  it('supplies exactly the fields a HF config.json cannot', () => {
    const r = recipeToModel(glm52);
    expect(r.mapped.total_params_b).toBe(743);
    expect(r.mapped.active_params_b).toBe(39);
    expect(r.mapped.max_ctx).toBe(1048576);
    expect(r.mapped.quants).toEqual(['FP8', 'MXFP4', 'NVFP4', 'FP16']);
    expect(r.mapped.tp_options).toEqual([8, 16]);
    expect(r.min_vllm_version).toBe('0.17.0');
  });

  it('never touches geometry — that is config.json territory', () => {
    const r = recipeToModel(glm52);
    for (const k of ['layers', 'kv_heads', 'head_dim', 'mla', 'hidden_size', 'vocab_size', 'sliding_window'] as const) {
      expect(r.mapped[k]).toBeUndefined();
    }
  });

  it('carries the stated VRAM floors for cross-checking', () => {
    const r = recipeToModel(glm52);
    expect(r.vram_minimums).toHaveLength(4);
    expect(r.vram_minimums.find((v) => v.quant === 'FP8')!.vram_gb).toBe(893);
  });

  it('skips precisions it has no quant for, and reports them', () => {
    const r = recipeToModel({ variants: { a: { precision: 'int2/4/8' }, b: { precision: 'fp8' } } });
    expect(r.mapped.quants).toEqual(['FP8']);
    expect(r.unmapped_precisions).toEqual(['int2/4/8']);
  });

  it('defaults active to total for dense models that omit it', () => {
    const r = recipeToModel({ model: { parameter_count: '27B' } });
    expect(r.mapped.total_params_b).toBe(27);
    expect(r.mapped.active_params_b).toBe(27);
  });

  it('ignores an active count larger than the total', () => {
    const r = recipeToModel({ model: { parameter_count: '27B', active_parameters: '99B' } });
    expect(r.mapped.active_params_b).toBeUndefined();
  });
});

describe('recipe + config.json compose into a complete model', () => {
  it('the union passes §F validation with no admin input', () => {
    // geometry from config.json...
    const { mapped: geo, missing } = hfConfigToModel('zai-org/GLM-5.2', {
      architectures: ['GlmMoeDsaForCausalLM'], model_type: 'glm_moe_dsa',
      num_hidden_layers: 78, num_attention_heads: 64, num_key_value_heads: 64, head_dim: 192,
      hidden_size: 6144, vocab_size: 154880, tie_word_embeddings: false,
      kv_lora_rank: 512, qk_rope_head_dim: 64, max_position_embeddings: 1048576,
    });
    // ...commercials from the recipe
    const { mapped: rec } = recipeToModel({
      hf_id: 'zai-org/GLM-5.2',
      model: { parameter_count: '743B', active_parameters: '39B', context_length: 1048576 },
      variants: { default: { precision: 'fp8', extra_args: ['--tensor-parallel-size', '8'] } },
    });
    const merged = { ...geo, ...rec };
    // every field hf.ts flagged as missing is now supplied
    for (const k of missing) expect((merged as any)[k]).toBeDefined();
    expect(modelSchema.safeParse(merged).success).toBe(true);
  });
});

describe('VRAM cross-check', () => {
  const { models } = seedCatalog();
  it('our GLM-5.2 FP8 weights sit plausibly under the published floor', () => {
    const m = models.find((x) => x.id === 'glm52')!;
    const c = crossCheckVram(weightsGb(m, 'FP8'), 893);
    expect(c.verdict).toBe('plausible');
    expect(c.ratio).toBeGreaterThan(0.7);
    expect(c.ratio).toBeLessThan(0.95);
  });
  it('flags an estimate that exceeds the stated floor', () => {
    expect(crossCheckVram(1000, 500).verdict).toBe('over_floor');
    expect(crossCheckVram(400, 450).verdict).toBe('tight');
  });
});
