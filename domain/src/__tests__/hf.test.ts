import { describe, it, expect } from 'vitest';
import { hfConfigToModel, concurrencySweep, bucketLayerTypes } from '../hf.js';
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

  it('detects MLA structurally from kv_lora_rank, not just the architecture name', () => {
    // GLM-5.2 ships as GlmMoeDsaForCausalLM — no "mla" anywhere in the name, but the
    // compressed KV projection makes it unambiguous
    const { mapped, detectedMla } = hfConfigToModel('zai-org/GLM-5.2', {
      architectures: ['GlmMoeDsaForCausalLM'], model_type: 'glm_moe_dsa',
      num_hidden_layers: 78, num_attention_heads: 64, num_key_value_heads: 64, head_dim: 192,
      hidden_size: 6144, vocab_size: 154880, tie_word_embeddings: false,
      kv_lora_rank: 512, qk_rope_head_dim: 64, max_position_embeddings: 1048576,
    });
    expect(detectedMla).toBe(true);
    expect(mapped.kv_heads).toBe(0); // the 64 KV heads in the config are NOT the cache geometry
    expect(mapped.head_dim).toBe(0);
    expect(mapped.layers).toBe(78);
    expect(mapped.hidden_size).toBe(6144);
    expect(mapped.vocab_size).toBe(154880);
  });

  it('does not mistake a plain GQA model for MLA', () => {
    const { detectedMla } = hfConfigToModel('zai-org/GLM-4.5', {
      architectures: ['Glm4MoeForCausalLM'], model_type: 'glm4_moe',
      num_hidden_layers: 92, num_attention_heads: 96, num_key_value_heads: 8, head_dim: 128,
      hidden_size: 5120, vocab_size: 151552, max_position_embeddings: 131072,
    });
    expect(detectedMla).toBe(false); // no kv_lora_rank => real GQA
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

  it('maps sliding-window layers from layer_types (GPT-OSS style)', () => {
    const layer_types = Array.from({ length: 36 }, (_, i) => (i % 2 === 0 ? 'sliding_attention' : 'full_attention'));
    const { mapped } = hfConfigToModel('openai/gpt-oss-120b', {
      num_hidden_layers: 36, num_attention_heads: 64, num_key_value_heads: 8, head_dim: 64,
      hidden_size: 2880, vocab_size: 201088, sliding_window: 128, layer_types,
      max_position_embeddings: 131072,
    });
    expect(mapped.sliding_window).toBe(128);
    expect(mapped.full_attention_layers).toBe(18);
  });

  it('maps sliding_window_pattern (Gemma style: one global every N)', () => {
    const { mapped } = hfConfigToModel('google/gemma-2-27b', {
      num_hidden_layers: 46, num_attention_heads: 32, num_key_value_heads: 16, head_dim: 128,
      hidden_size: 4608, vocab_size: 256000, sliding_window: 4096, sliding_window_pattern: 2,
      max_position_embeddings: 8192,
    });
    expect(mapped.sliding_window).toBe(4096);
    expect(mapped.full_attention_layers).toBe(23); // 46 / 2
  });

  it('a bare sliding_window with no pattern windows every layer (Mistral v0.1 style)', () => {
    const { mapped } = hfConfigToModel('mistralai/Mistral-7B-v0.1', {
      num_hidden_layers: 32, num_attention_heads: 32, num_key_value_heads: 8, head_dim: 128,
      hidden_size: 4096, vocab_size: 32000, sliding_window: 4096, max_position_embeddings: 32768,
    });
    expect(mapped.full_attention_layers).toBe(0);
  });

  it('maps hybrid linear attention from linear_attn_config (Kimi K3 style)', () => {
    const { mapped } = hfConfigToModel('moonshotai/Kimi-K3', {
      architectures: ['KimiLinearForCausalLM'], model_type: 'kimi_linear',
      num_hidden_layers: 93, num_attention_heads: 96, num_key_value_heads: 96,
      hidden_size: 7168, vocab_size: 163840, kv_lora_rank: 512, qk_rope_head_dim: 64,
      max_position_embeddings: 1048576,
      linear_attn_config: {
        full_attn_layers: Array.from({ length: 24 }, (_, i) => (i + 1) * 4),
        kda_layers: Array.from({ length: 69 }, (_, i) => i + 1),
        num_heads: 96, head_dim: 128,
      },
    });
    expect(mapped.full_attention_layers).toBe(24);
    expect(mapped.linear_attention_layers).toBe(69);
    expect(mapped.linear_state_bytes_per_layer).toBe(96 * 128 * 128 * 4); // fp32 recurrent state
    expect(mapped.mla).toBe(true); // kv_lora_rank still marks the 24 cached layers as MLA
  });

  it('maps the Qwen spelling: linear_attention in layer_types + flat linear_* dims', () => {
    // Qwen3.6 declares hybrid attention completely differently from Kimi — no linear_attn_config,
    // and 'linear_attention' rather than 'sliding_attention' in layer_types
    const layer_types = Array.from({ length: 64 }, (_, i) => ((i + 1) % 4 === 0 ? 'full_attention' : 'linear_attention'));
    const { mapped } = hfConfigToModel('Qwen/Qwen3.6-27B', {
      architectures: ['Qwen3_5ForConditionalGeneration'], model_type: 'qwen3_5_text',
      num_hidden_layers: 64, num_attention_heads: 24, num_key_value_heads: 4, head_dim: 256,
      hidden_size: 5120, vocab_size: 248320, max_position_embeddings: 262144, layer_types,
      linear_num_key_heads: 16, linear_num_value_heads: 48,
      linear_key_head_dim: 128, linear_value_head_dim: 128, full_attention_interval: 4,
    });
    expect(mapped.full_attention_layers).toBe(16);
    expect(mapped.linear_attention_layers).toBe(48);
    expect(mapped.linear_state_bytes_per_layer).toBe(48 * 128 * 128 * 4); // v_heads x k_dim x v_dim, fp32
    expect(mapped.mla).toBe(false); // real GQA on the 16 cached layers
    expect(mapped.kv_heads).toBe(4);
    expect(mapped.head_dim).toBe(256);
  });

  it('buckets layer_types by vocabulary, defaulting unknown kinds to full attention', () => {
    expect(bucketLayerTypes(['full_attention', 'sliding_attention', 'linear_attention']))
      .toEqual({ full: 1, sliding: 1, linear: 1 });
    expect(bucketLayerTypes(['mamba', 'recurrent', 'local_attention']))
      .toEqual({ full: 0, sliding: 1, linear: 2 });
    expect(bucketLayerTypes(['something_new'])).toEqual({ full: 1, sliding: 0, linear: 0 });
  });

  it('leaves linear fields unset for a normal model', () => {
    const { mapped } = hfConfigToModel('meta-llama/Llama-3.3-70B-Instruct', {
      num_hidden_layers: 80, num_attention_heads: 64, num_key_value_heads: 8, head_dim: 128,
      hidden_size: 8192, vocab_size: 128256, max_position_embeddings: 131072,
    });
    expect(mapped.linear_attention_layers).toBeUndefined();
    expect(mapped.linear_state_bytes_per_layer).toBeUndefined();
  });

  it('leaves window fields unset when the config declares no window', () => {
    const { mapped } = hfConfigToModel('meta-llama/Llama-3.3-70B-Instruct', {
      num_hidden_layers: 80, num_attention_heads: 64, num_key_value_heads: 8, head_dim: 128,
      hidden_size: 8192, vocab_size: 128256, max_position_embeddings: 131072,
    });
    expect(mapped.sliding_window).toBeUndefined();
    expect(mapped.full_attention_layers).toBeUndefined();
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
      models.find((x) => x.id === 'qwen3-32b')!,
      gpus.find((x) => x.id === 'rtx4090')!,
      { quant: 'Q4_K_M' as const, kv_dtype_bytes: 1, selected_ctx: 4096, avg_context_utilisation: 0.6, mem_util_fraction: 0.9, gpus_per_node: 1 },
      [1, 4],
    );
    // 18.6 GiB of weights in 19.1 GiB usable. At concurrency 1 a single tight GPU is cheapest;
    // by concurrency 4 the engine prefers TP2 (2 GPUs) over 4 tight TP1 pods, so the verdict
    // varies across the sweep exactly as the GPU-minimising rule implies.
    expect(rows[0].feasible && rows[0].tight).toBe(true);
    expect(rows[0].tp).toBe(1);
    expect(rows[0].gpus).toBe(1);
    expect(rows[1].tp).toBe(2);
    expect(rows[1].gpus).toBe(2); // not 4 tight single-GPU pods
    expect(rows[1].tight).toBe(false);

    // the flag does vary with context: at 8K the same model needs TP2, which is roomy
    const wider = concurrencySweep(
      models.find((x) => x.id === 'qwen3-32b')!,
      gpus.find((x) => x.id === 'rtx4090')!,
      { quant: 'Q4_K_M' as const, kv_dtype_bytes: 1, selected_ctx: 8192, avg_context_utilisation: 0.6, mem_util_fraction: 0.9, gpus_per_node: 1 },
      [1],
    );
    expect(wider[0].tp).toBe(2);
    expect(wider[0].tight).toBe(false);
  });
});
