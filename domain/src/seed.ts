// tp_options ladder: every TP size from the smallest that can hold the model's SMALLEST
// supported quant on the largest catalogue GPU, up to 16. Listing too few sizes made the
// engine over-recommend hardware (GLM-4.5 FP8 fits TP4 but only TP8 was offered); listing
// extra small sizes is harmless, since infeasible ones are skipped during selection.
// Canonical seeded catalog (addendum §B). ONE source of truth — consumed by the DB
// migration seed AND the FR-23 reset (AD-15). [VERIFY]: all GPU bandwidths were self-flagged
// approximate in the addendum (OQ-2) — confirm before prod. Model geometry is now sourced.

import type { Model, GpuSku } from './types.js';

// hidden_size / vocab_size / tied_embeddings come from each model's HF config.json and size the
// 16-bit embedding + lm_head tail that survives quantisation (engine.weightsGb).
//
// glm45 / glm52: VERIFIED 2026-07-28 against the published configs.
//   GLM-4.5 (glm4_moe) was already correct: 92 layers, GQA 8x128, hidden 5120, vocab 151552.
//   GLM-5.2 (glm_moe_dsa) was WRONG and is corrected here — it is an MLA model, not GQA:
//   kv_lora_rank 512 + qk_rope_head_dim 64 = the same 576 latent elements/layer DeepSeek uses.
//   The old GQA 8x128 guess overstated its KV cache by ~3.6x. Also corrected: hidden 5120->6144,
//   vocab 151552->154880, params 744/40 -> 743/39 (per the vLLM recipe), TP {8,16} -> {4,8,16}
//   and the BF16 + AMD Quark MXFP4 checkpoints added.
//   kimi-k2 / kimi-k3: VERIFIED 2026-07-28. K2 was already correct (61 layers, hidden 7168,
//   vocab 163840, MLA 576/layer). K3 is new — see the note on its entry below.
//   dense_params_b (attention + shared experts + router + dense MLP, excluding the embedding
//   tail) is computed from each config and reconciles with the published totals to within 0.03%
//   for GLM-5.2 and Kimi-K2. It only changes sizing for quants listed in mixed_precision.
//   glm52 declares NVFP4 mixed because NVIDIA's ModelOpt card says "only MoE expert linears are
//   quantized to NVFP4; shared experts, attention ..." stay higher. MXFP4 is left uniform: AMD's
//   card says only "MoE weights quantized", which is not specific enough to model differently.
//   NOTE: `index_topk: 2048` (DSA sparse attention) selects which tokens each query attends to.
//   It reduces attention COMPUTE, not cache residency — the KV cache still holds every token —
//   so it is deliberately NOT modelled as a memory saving.
// sliding_window/full_attention_layers: VERIFIED 2026-07-28 against the published
// config.json for both GPT-OSS checkpoints — `layer_types` strictly alternates
// sliding_attention/full_attention with `sliding_window: 128`:
//   gpt-oss-120b: 36 layers = 18 full + 18 sliding
//   gpt-oss-20b : 24 layers = 12 full + 12 sliding
// Their head/embedding geometry was confirmed in the same pass (kv_heads 8, head_dim 64,
// hidden_size 2880, vocab_size 201088, tie_word_embeddings false).
export const SEED_MODELS: Model[] = [
  { id: 'llama31-8b', name: 'Llama 3.1 8B Instruct', total_params_b: 8.03, active_params_b: 8.03, layers: 32, kv_heads: 8, head_dim: 128, mla: false, max_ctx: 131072, tp_options: [1, 2, 4], quants: ['FP16', 'FP8', 'INT4', 'Q8_0', 'Q4_K_M'], hidden_size: 4096, intermediate_size: 14336, vocab_size: 128256, tied_embeddings: false },
  { id: 'gptoss-20b', name: 'GPT-OSS 20B (MoE 3.6B act)', total_params_b: 21, active_params_b: 3.6, layers: 24, kv_heads: 8, head_dim: 64, mla: false, max_ctx: 131072, tp_options: [1, 2, 4], quants: ['MXFP4'], hidden_size: 2880, intermediate_size: 11520, vocab_size: 201088, tied_embeddings: false, sliding_window: 128, full_attention_layers: 12 },
  { id: 'mistral-s24', name: 'Mistral Small 3.2 24B', total_params_b: 24, active_params_b: 24, layers: 40, kv_heads: 8, head_dim: 128, mla: false, max_ctx: 131072, tp_options: [1, 2, 4], quants: ['FP16', 'FP8', 'INT4', 'Q4_K_M'], hidden_size: 5120, intermediate_size: 32768, vocab_size: 131072, tied_embeddings: false },
  { id: 'qwen3-30a3', name: 'Qwen3-30B-A3B / Coder (MoE)', total_params_b: 30.5, active_params_b: 3.3, layers: 48, kv_heads: 4, head_dim: 128, mla: false, max_ctx: 262144, tp_options: [1, 2, 4], quants: ['FP16', 'FP8', 'INT4'], hidden_size: 2048, intermediate_size: 6144, vocab_size: 151936, tied_embeddings: false },
  // Qwen3.6-27B — hybrid gated delta-net. 16 of 64 layers keep a token cache (full_attention_interval 4);
  // the other 48 are linear, with a [48 v-heads x 128 x 128] fp32 state = 3.15 MB/layer (151 MB flat).
  // Note the 248,320-token vocab: at INT4 the fp16 embedding tail is ~5.1 GB, ~30% of the checkpoint.
  { id: 'qwen36-27b', name: 'Qwen3.6-27B (hybrid GDN)', total_params_b: 27, active_params_b: 27, layers: 64, kv_heads: 4, head_dim: 256, mla: false, max_ctx: 262144, tp_options: [1, 2, 4], quants: ['FP16', 'FP8', 'INT4', 'NVFP4'], hidden_size: 5120, vocab_size: 248320, tied_embeddings: false, full_attention_layers: 16, linear_attention_layers: 48, linear_state_bytes_per_layer: 3145728 },
  { id: 'qwen3-32b', name: 'Qwen3-32B (dense)', total_params_b: 32.8, active_params_b: 32.8, layers: 64, kv_heads: 8, head_dim: 128, mla: false, max_ctx: 131072, tp_options: [1, 2, 4], quants: ['FP16', 'FP8', 'INT4', 'Q4_K_M'], hidden_size: 5120, intermediate_size: 25600, vocab_size: 151936, tied_embeddings: false },
  { id: 'llama33-70b', name: 'Llama 3.3 70B Instruct', total_params_b: 70.6, active_params_b: 70.6, layers: 80, kv_heads: 8, head_dim: 128, mla: false, max_ctx: 131072, tp_options: [1, 2, 4, 8], quants: ['FP16', 'FP8', 'INT4', 'Q4_K_M', 'IQ4_XS'], hidden_size: 8192, intermediate_size: 28672, vocab_size: 128256, tied_embeddings: false },
  { id: 'qwen25-72b', name: 'Qwen2.5-72B Instruct', total_params_b: 72.7, active_params_b: 72.7, layers: 80, kv_heads: 8, head_dim: 128, mla: false, max_ctx: 131072, tp_options: [1, 2, 4, 8], quants: ['FP16', 'FP8', 'INT4'], hidden_size: 8192, intermediate_size: 29568, vocab_size: 152064, tied_embeddings: false },
  { id: 'gptoss-120b', name: 'GPT-OSS 120B (MoE 5.1B act)', total_params_b: 117, active_params_b: 5.1, layers: 36, kv_heads: 8, head_dim: 64, mla: false, max_ctx: 131072, tp_options: [1, 2, 4, 8], quants: ['MXFP4'], hidden_size: 2880, intermediate_size: 11520, vocab_size: 201088, tied_embeddings: false, sliding_window: 128, full_attention_layers: 18 },
  { id: 'qwen3-235b', name: 'Qwen3-235B-A22B (MoE)', total_params_b: 235, active_params_b: 22, layers: 94, kv_heads: 4, head_dim: 128, mla: false, max_ctx: 262144, tp_options: [1, 2, 4, 8, 16], quants: ['FP16', 'FP8', 'INT4'], hidden_size: 4096, intermediate_size: 12288, vocab_size: 151936, tied_embeddings: false },
  { id: 'glm45', name: 'GLM-4.5 355B-A32B (MoE)', total_params_b: 355, active_params_b: 32, layers: 92, kv_heads: 8, head_dim: 128, mla: false, max_ctx: 131072, tp_options: [1, 2, 4, 8, 16], quants: ['FP16', 'FP8', 'INT4'], hidden_size: 5120, vocab_size: 151552, tied_embeddings: false, dense_params_b: 15.3 },
  { id: 'glm52', name: 'GLM-5.2 743B-A39B (MoE·MLA·DSA)', total_params_b: 743, active_params_b: 39, layers: 78, kv_heads: 0, head_dim: 0, mla: true, max_ctx: 1048576, tp_options: [2, 4, 8, 16], quants: ['FP16', 'FP8', 'MXFP4', 'NVFP4'], hidden_size: 6144, vocab_size: 154880, tied_embeddings: false, dense_params_b: 16.5, mixed_precision: { NVFP4: 'FP16' } },
  { id: 'dsv3', name: 'DeepSeek-V3 / R1 671B (MLA)', total_params_b: 671, active_params_b: 37, layers: 61, kv_heads: 0, head_dim: 0, mla: true, max_ctx: 131072, tp_options: [2, 4, 8, 16], quants: ['FP8', 'INT4'], hidden_size: 7168, intermediate_size: 16384, vocab_size: 129280, tied_embeddings: false },
  { id: 'kimi-k2', name: 'Kimi K2 1T-A32B (MLA)', total_params_b: 1026, active_params_b: 32.5, layers: 61, kv_heads: 0, head_dim: 0, mla: true, max_ctx: 131072, tp_options: [4, 8, 16], quants: ['FP8', 'INT4'], hidden_size: 7168, intermediate_size: 16384, vocab_size: 163840, tied_embeddings: false, dense_params_b: 9.4 },
  // Kimi K3 — hybrid attention. Only 24 of 93 layers keep a token-indexed cache (full MLA);
  // the other 69 are KDA (Kimi Delta Attention), a recurrent form whose state is CONSTANT in
  // sequence length. linear_state_bytes_per_layer = num_heads 96 x head_dim 128 x 128 x 4 B
  // (fp32 recurrent state) = 6.29 MB/layer, so 69 layers cost a flat ~434 MB per request
  // regardless of context. Sizing all 93 layers as MLA would overstate KV ~3.9x at 1M.
  { id: 'kimi-k3', name: 'Kimi K3 2.8T-A60B (MoE·MLA+KDA)', total_params_b: 2800, active_params_b: 60, layers: 93, kv_heads: 0, head_dim: 0, mla: true, max_ctx: 1048576, tp_options: [8, 16], quants: ['MXFP4'], hidden_size: 7168, vocab_size: 163840, tied_embeddings: false, full_attention_layers: 24, linear_attention_layers: 69, linear_state_bytes_per_layer: 6291456, dense_params_b: 20.8 },
];

// tflops_fp16: DENSE (non-sparse) FP16 tensor throughput, driving the compute-bound TTFT
// estimate. [VERIFY] against vendor datasheets — like the bandwidths, these are indicative.
// price_per_gpu_hour: INDICATIVE market rental rates ($/GPU-hour) — [VERIFY] against your
// contracts. Admin-editable; the cost estimate uses these.
export const SEED_GPUS: GpuSku[] = [
  // --- NVIDIA datacenter ---
  { id: 'l4', name: 'L4 24 GB', mem_gb: 24, bw_tbs: 0.3, tflops_fp16: 121, price_per_gpu_hour: 0.35 },
  { id: 'l40s', name: 'L40S 48 GB', mem_gb: 48, bw_tbs: 0.86, tflops_fp16: 362, price_per_gpu_hour: 0.8 },
  { id: 'a100p', name: 'A100 80 GB PCIe', mem_gb: 80, bw_tbs: 1.94, tflops_fp16: 312, price_per_gpu_hour: 1.5 },
  { id: 'a100s', name: 'A100 80 GB SXM', mem_gb: 80, bw_tbs: 2.04, tflops_fp16: 312, price_per_gpu_hour: 1.8 },
  { id: 'h100', name: 'H100 80 GB SXM', mem_gb: 80, bw_tbs: 3.35, tflops_fp16: 989, price_per_gpu_hour: 2.9 },
  { id: 'h100n', name: 'H100 NVL 94 GB', mem_gb: 94, bw_tbs: 3.9, tflops_fp16: 835, price_per_gpu_hour: 3.2 },
  { id: 'h200', name: 'H200 141 GB (SXM/NVL)', mem_gb: 141, bw_tbs: 4.8, tflops_fp16: 989, price_per_gpu_hour: 4.5 },
  { id: 'b200', name: 'B200 180 GB SXM', mem_gb: 180, bw_tbs: 8.0, tflops_fp16: 2250, price_per_gpu_hour: 6.5 },
  { id: 'b300', name: 'B300 288 GB (Blackwell Ultra)', mem_gb: 288, bw_tbs: 8.0, tflops_fp16: 2500, price_per_gpu_hour: 8.5 },
  // --- AMD Instinct (ROCm vLLM) ---
  { id: 'mi300x', name: 'MI300X 192 GB', mem_gb: 192, bw_tbs: 5.3, tflops_fp16: 1307, price_per_gpu_hour: 2.0 },
  { id: 'mi325x', name: 'MI325X 256 GB', mem_gb: 256, bw_tbs: 6.0, tflops_fp16: 1307, price_per_gpu_hour: 2.5 },
  { id: 'mi355x', name: 'MI355X 288 GB', mem_gb: 288, bw_tbs: 8.0, tflops_fp16: 2300, price_per_gpu_hour: 4.0 },
  // --- Workstation / consumer (single-box self-hosting; no NVLink, TP over PCIe) ---
  { id: 'rtxpro6000', name: 'RTX PRO 6000 Blackwell 96 GB', mem_gb: 96, bw_tbs: 1.79, tflops_fp16: 503, price_per_gpu_hour: 1.8 },
  { id: 'rtx5090', name: 'RTX 5090 32 GB', mem_gb: 32, bw_tbs: 1.79, tflops_fp16: 210, price_per_gpu_hour: 0.7 },
  { id: 'rtx4090', name: 'RTX 4090 24 GB', mem_gb: 24, bw_tbs: 1.01, tflops_fp16: 165, price_per_gpu_hour: 0.4 },
];

export function seedCatalog(): { models: Model[]; gpus: GpuSku[] } {
  // deep clone so callers can't mutate the canonical seed
  return {
    models: SEED_MODELS.map((m) => ({ ...m, tp_options: [...m.tp_options], quants: [...m.quants] })),
    gpus: SEED_GPUS.map((g) => ({ ...g })),
  };
}
