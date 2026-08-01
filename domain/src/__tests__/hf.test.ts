import { describe, it, expect } from 'vitest';
import { hfConfigToModel, concurrencySweep, bucketLayerTypes, perTokenFfnWidth } from '../hf.js';
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

  it('reads the MLA latent width out of the config rather than assuming 576', () => {
    // kv_lora_rank alone under-counts the cache by the rope dimension that rides beside it
    const { mapped } = hfConfigToModel('deepseek-ai/DeepSeek-V3', {
      architectures: ['DeepseekV3ForCausalLM'], model_type: 'deepseek_v3',
      num_hidden_layers: 61, num_attention_heads: 128, hidden_size: 7168, vocab_size: 129280,
      kv_lora_rank: 512, qk_rope_head_dim: 64, max_position_embeddings: 131072,
    });
    expect(mapped.mla_latent_elems).toBe(576);

    // a hypothetical wider rank must come through as itself, not as DeepSeek's number
    const { mapped: wide } = hfConfigToModel('acme/WideLatent', {
      architectures: ['AcmeMlaForCausalLM'], num_hidden_layers: 48, num_attention_heads: 64,
      hidden_size: 8192, vocab_size: 128000, kv_lora_rank: 1024, qk_rope_head_dim: 128,
      max_position_embeddings: 131072,
    });
    expect(wide.mla_latent_elems).toBe(1152);
  });

  it('leaves the latent unmapped when either half is missing, and on GQA models', () => {
    // half the geometry would size the cache short; better to fall back to the documented default
    const { mapped: half } = hfConfigToModel('acme/HalfConfig', {
      architectures: ['DeepseekV3ForCausalLM'], num_hidden_layers: 61, num_attention_heads: 128,
      hidden_size: 7168, vocab_size: 129280, kv_lora_rank: 512, max_position_embeddings: 131072,
    });
    expect(half.mla).toBe(true);
    expect(half.mla_latent_elems).toBeUndefined();

    const { mapped: gqa } = hfConfigToModel('meta-llama/Llama-3.3-70B-Instruct', {
      num_hidden_layers: 80, num_attention_heads: 64, num_key_value_heads: 8, head_dim: 128,
      hidden_size: 8192, vocab_size: 128256, max_position_embeddings: 131072,
    });
    expect(gqa.mla_latent_elems).toBeUndefined();
  });

  it('unwraps a multimodal config to the language model inside it', () => {
    // Kimi K3 ships as KimiK3ForConditionalGeneration: the top level is the vision tower and
    // everything that sizes a deployment lives under text_config. Read from the top level this
    // mapped six fields and still reported partial success — a form that looks mostly filled in.
    const { mapped, missing } = hfConfigToModel('moonshotai/Kimi-K3', {
      architectures: ['KimiK3ForConditionalGeneration'], model_type: 'kimi_k3',
      tie_word_embeddings: false,
      text_config: {
        architectures: ['KimiLinearForCausalLM'], model_type: 'kimi_linear',
        num_hidden_layers: 93, hidden_size: 7168, vocab_size: 163840,
        max_position_embeddings: 1048576, kv_lora_rank: 512, qk_rope_head_dim: 64,
        num_attention_heads: 96, num_key_value_heads: 96,
        moe_intermediate_size: 3072, num_experts: 896, num_experts_per_token: 16,
        num_shared_experts: 2,
        linear_attn_config: {
          full_attn_layers: Array.from({ length: 24 }, (_, i) => (i + 1) * 4),
          kda_layers: Array.from({ length: 69 }, (_, i) => i + 1),
          num_heads: 96, head_dim: 128,
        },
      },
    });
    expect(mapped.layers).toBe(93);
    expect(mapped.hidden_size).toBe(7168);
    expect(mapped.max_ctx).toBe(1048576);
    expect(mapped.mla_latent_elems).toBe(576);
    expect(mapped.num_experts).toBe(896);
    expect(mapped.experts_per_token).toBe(16);
    expect(mapped.linear_attention_layers).toBe(69);
    // 3072 per expert x (16 routed + 2 shared) — the width one token actually traverses
    expect(mapped.intermediate_size).toBe(55296);
    expect(missing).toEqual(['total_params_b', 'active_params_b', 'tp_options', 'quants']);
  });

  it('leaves a plain config alone rather than hunting for a nested one', () => {
    const { mapped } = hfConfigToModel('meta-llama/Llama-3.3-70B-Instruct', {
      num_hidden_layers: 80, num_attention_heads: 64, num_key_value_heads: 8, head_dim: 128,
      hidden_size: 8192, vocab_size: 128256, max_position_embeddings: 131072,
      // a vision block with no transformer of its own must not be mistaken for the text model
      text_config: { hidden_size: 1280 },
    });
    expect(mapped.hidden_size).toBe(8192);
    expect(mapped.layers).toBe(80);
  });

  it('maps MoE routing, however the family spells the expert count', () => {
    const one = hfConfigToModel('deepseek-ai/DeepSeek-V3', {
      model_type: 'deepseek_v3', num_hidden_layers: 61, num_attention_heads: 128,
      hidden_size: 7168, vocab_size: 129280, max_position_embeddings: 131072,
      kv_lora_rank: 512, qk_rope_head_dim: 64,
      n_routed_experts: 256, num_experts_per_tok: 8,
    }).mapped;
    expect(one.num_experts).toBe(256);
    expect(one.experts_per_token).toBe(8);

    const two = hfConfigToModel('openai/gpt-oss-120b', {
      num_hidden_layers: 36, num_attention_heads: 64, num_key_value_heads: 8, head_dim: 64,
      hidden_size: 2880, vocab_size: 201088, max_position_embeddings: 131072,
      num_local_experts: 128, num_experts_per_tok: 4,
    }).mapped;
    expect(two.num_experts).toBe(128);
    expect(two.experts_per_token).toBe(4);
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

  // The reserve shards the FFN, so the import has to supply the width a token actually traverses.
  it('maps the dense FFN width straight through', () => {
    const { mapped } = hfConfigToModel('meta-llama/Llama-3.3-70B-Instruct', {
      num_hidden_layers: 80, num_attention_heads: 64, num_key_value_heads: 8, head_dim: 128,
      hidden_size: 8192, intermediate_size: 28672, vocab_size: 128256, max_position_embeddings: 131072,
    });
    expect(mapped.intermediate_size).toBe(28672);
  });

  it('prefers the MoE fields, because intermediate_size describes the dense layers', () => {
    // DeepSeek-V3's published config: 8 routed experts of 2048 plus 1 always-on shared expert
    const cfg = {
      num_hidden_layers: 61, num_attention_heads: 128, hidden_size: 7168, kv_lora_rank: 512,
      intermediate_size: 18432, moe_intermediate_size: 2048, num_experts_per_tok: 8,
      n_shared_experts: 1, vocab_size: 129280, max_position_embeddings: 131072,
    };
    expect(perTokenFfnWidth(cfg)).toBe(18432); // (8 + 1) x 2048
    expect(hfConfigToModel('deepseek-ai/DeepSeek-V3', cfg).mapped.intermediate_size).toBe(18432);
    // the shared expert is an eighth of the total here, not a rounding error
    expect(perTokenFfnWidth({ ...cfg, n_shared_experts: 0 })).toBe(16384);
    // an incomplete MoE pair falls back rather than guessing a top-k
    expect(perTokenFfnWidth({ ...cfg, num_experts_per_tok: undefined })).toBe(18432);
    expect(perTokenFfnWidth({ hidden_size: 4096 })).toBeUndefined();
  });

  it('reads the field spellings the MoE families actually ship', () => {
    // Kimi K3: num_experts_per_token / num_shared_experts
    expect(perTokenFfnWidth({
      hidden_size: 7168, moe_intermediate_size: 3072, num_experts_per_token: 16, num_shared_experts: 2,
    })).toBe(55296); // (16 + 2) x 3072
    // Qwen2-MoE: the shared expert as a WIDTH rather than a count
    expect(perTokenFfnWidth({
      hidden_size: 3584, moe_intermediate_size: 2560, num_experts_per_tok: 4,
      shared_expert_intermediate_size: 20480,
    })).toBe(30720); // 4 x 2560 + 20480
    // Qwen3-MoE dropped the shared expert entirely
    expect(perTokenFfnWidth({
      hidden_size: 4096, moe_intermediate_size: 1536, num_experts_per_tok: 8,
    })).toBe(12288);
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
