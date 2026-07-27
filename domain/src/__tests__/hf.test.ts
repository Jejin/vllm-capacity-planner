import { describe, it, expect } from 'vitest';
import { hfConfigToModel, concurrencySweep } from '../hf.js';
import { seedCatalog } from '../seed.js';
import { modelSchema } from '../schema.js';

describe('HF config.json → §F mapping (AD-11)', () => {
  it('maps a GQA model (Qwen-like) with GQA-correct kv_heads', () => {
    const { mapped, missing, detectedMla } = hfConfigToModel('Qwen/Qwen2.5-72B-Instruct', {
      architectures: ['Qwen2ForCausalLM'], model_type: 'qwen2',
      num_hidden_layers: 80, num_attention_heads: 64, num_key_value_heads: 8, head_dim: 128,
      max_position_embeddings: 131072,
    });
    expect(detectedMla).toBe(false);
    expect(mapped.layers).toBe(80);
    expect(mapped.kv_heads).toBe(8); // KV heads, not the 64 attention heads
    expect(mapped.head_dim).toBe(128);
    expect(mapped.max_ctx).toBe(131072);
    // params, tp, quants are never in config.json → admin must complete
    expect(missing).toContain('total_params_b');
    expect(missing).toContain('tp_options');
    expect(missing).toContain('quants');
  });

  it('detects MLA from architecture and zeroes kv geometry', () => {
    const { mapped, detectedMla } = hfConfigToModel('deepseek-ai/DeepSeek-V3', {
      architectures: ['DeepseekV3ForCausalLM'], model_type: 'deepseek_v3',
      num_hidden_layers: 61, num_attention_heads: 128, max_position_embeddings: 131072,
    });
    expect(detectedMla).toBe(true);
    expect(mapped.kv_heads).toBe(0);
    expect(mapped.head_dim).toBe(0);
  });

  it('derives head_dim from hidden_size/heads when absent', () => {
    const { mapped } = hfConfigToModel('x/y', { num_hidden_layers: 32, num_attention_heads: 32, num_key_value_heads: 8, hidden_size: 4096, max_position_embeddings: 8192 });
    expect(mapped.head_dim).toBe(128); // 4096/32
  });

  it('maps the embedding geometry that sizes the 16-bit tail', () => {
    const { mapped } = hfConfigToModel('meta-llama/Llama-3.3-70B-Instruct', {
      num_hidden_layers: 80, num_attention_heads: 64, num_key_value_heads: 8, head_dim: 128,
      hidden_size: 8192, vocab_size: 128256, tie_word_embeddings: false, max_position_embeddings: 131072,
    });
    expect(mapped.hidden_size).toBe(8192);
    expect(mapped.vocab_size).toBe(128256);
    expect(mapped.tied_embeddings).toBe(false);
  });

  it('honours tie_word_embeddings (one shared table, not two)', () => {
    const { mapped } = hfConfigToModel('Qwen/Qwen3-1.7B', {
      num_hidden_layers: 28, num_attention_heads: 16, num_key_value_heads: 8, head_dim: 128,
      hidden_size: 2048, vocab_size: 151936, tie_word_embeddings: true, max_position_embeddings: 32768,
    });
    expect(mapped.tied_embeddings).toBe(true);
  });

  it('a completed mapping passes §F validation', () => {
    const { mapped } = hfConfigToModel('Qwen/Qwen2.5-72B', { num_hidden_layers: 80, num_attention_heads: 64, num_key_value_heads: 8, head_dim: 128, max_position_embeddings: 131072 });
    const completed = { ...mapped, total_params_b: 72.7, active_params_b: 72.7, tp_options: [2, 4, 8], quants: ['FP16', 'FP8'] };
    expect(modelSchema.safeParse(completed).success).toBe(true);
  });
});

describe('concurrency rubric sweep', () => {
  it('returns a row per concurrency with rising aggregate throughput', () => {
    const { models, gpus } = seedCatalog();
    const m = models.find((x) => x.id === 'llama33-70b')!;
    const g = gpus.find((x) => x.id === 'h200')!;
    const base = { quant: 'FP8' as const, kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6, mem_util_fraction: 0.9, gpus_per_node: 8 };
    const rows = concurrencySweep(m, g, base, [1, 16, 64, 256]);
    expect(rows).toHaveLength(4);
    expect(rows.every((r) => r.feasible)).toBe(true);
    expect(rows[3].gpus).toBeGreaterThanOrEqual(rows[0].gpus); // more concurrency → ≥ GPUs
    expect(rows[3].throughput_tokens_per_sec).toBeGreaterThan(rows[0].throughput_tokens_per_sec);
    expect(rows.every((r) => r.tight === false)).toBe(true); // 70B FP8 on H200 TP2 has ample headroom
  });

  it('carries the tight verdict per row', () => {
    const { models, gpus } = seedCatalog();
    const rows = concurrencySweep(
      models.find((x) => x.id === 'gptoss-120b')!,
      gpus.find((x) => x.id === 'h100')!,
      { quant: 'MXFP4' as const, kv_dtype_bytes: 2, selected_ctx: 131072, avg_context_utilisation: 0.6, mem_util_fraction: 0.9, gpus_per_node: 8 },
      [1, 4],
    );
    expect(rows.every((r) => r.feasible && r.tight)).toBe(true); // TP1 at 128K leaves ~7% headroom
  });
});
