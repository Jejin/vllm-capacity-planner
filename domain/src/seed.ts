// Canonical seeded catalog (addendum §B). ONE source of truth — consumed by the DB
// migration seed AND the FR-23 reset (AD-15). [VERIFY] flags: glm45 layer/head cfg and
// all GPU bandwidths were self-flagged approximate in the addendum (OQ-2) — confirm before prod.

import type { Model, GpuSku } from './types.js';

// hidden_size / vocab_size / tied_embeddings come from each model's HF config.json and size the
// 16-bit embedding + lm_head tail that survives quantisation (engine.weightsGb). [VERIFY] the
// glm52 pair — that entry's geometry was already flagged approximate upstream.
// sliding_window/full_attention_layers: GPT-OSS alternates dense and locally-banded (128-token)
// attention, so half its layers stop growing at 128 tokens. [VERIFY] the split against config.json
// `layer_types` before relying on it for procurement.
export const SEED_MODELS: Model[] = [
  { id: 'llama31-8b', name: 'Llama 3.1 8B Instruct', total_params_b: 8.03, active_params_b: 8.03, layers: 32, kv_heads: 8, head_dim: 128, mla: false, max_ctx: 131072, tp_options: [1, 2], quants: ['FP16', 'FP8', 'INT4', 'Q8_0', 'Q4_K_M'], hidden_size: 4096, vocab_size: 128256, tied_embeddings: false },
  { id: 'gptoss-20b', name: 'GPT-OSS 20B (MoE 3.6B act)', total_params_b: 21, active_params_b: 3.6, layers: 24, kv_heads: 8, head_dim: 64, mla: false, max_ctx: 131072, tp_options: [1, 2], quants: ['MXFP4'], hidden_size: 2880, vocab_size: 201088, tied_embeddings: false, sliding_window: 128, full_attention_layers: 12 },
  { id: 'mistral-s24', name: 'Mistral Small 3.2 24B', total_params_b: 24, active_params_b: 24, layers: 40, kv_heads: 8, head_dim: 128, mla: false, max_ctx: 131072, tp_options: [1, 2, 4], quants: ['FP16', 'FP8', 'INT4', 'Q4_K_M'], hidden_size: 5120, vocab_size: 131072, tied_embeddings: false },
  { id: 'qwen3-30a3', name: 'Qwen3-30B-A3B / Coder (MoE)', total_params_b: 30.5, active_params_b: 3.3, layers: 48, kv_heads: 4, head_dim: 128, mla: false, max_ctx: 262144, tp_options: [1, 2, 4], quants: ['FP16', 'FP8', 'INT4'], hidden_size: 2048, vocab_size: 151936, tied_embeddings: false },
  { id: 'qwen3-32b', name: 'Qwen3-32B (dense)', total_params_b: 32.8, active_params_b: 32.8, layers: 64, kv_heads: 8, head_dim: 128, mla: false, max_ctx: 131072, tp_options: [1, 2, 4], quants: ['FP16', 'FP8', 'INT4', 'Q4_K_M'], hidden_size: 5120, vocab_size: 151936, tied_embeddings: false },
  { id: 'llama33-70b', name: 'Llama 3.3 70B Instruct', total_params_b: 70.6, active_params_b: 70.6, layers: 80, kv_heads: 8, head_dim: 128, mla: false, max_ctx: 131072, tp_options: [2, 4, 8], quants: ['FP16', 'FP8', 'INT4', 'Q4_K_M', 'IQ4_XS'], hidden_size: 8192, vocab_size: 128256, tied_embeddings: false },
  { id: 'qwen25-72b', name: 'Qwen2.5-72B Instruct', total_params_b: 72.7, active_params_b: 72.7, layers: 80, kv_heads: 8, head_dim: 128, mla: false, max_ctx: 131072, tp_options: [2, 4, 8], quants: ['FP16', 'FP8', 'INT4'], hidden_size: 8192, vocab_size: 152064, tied_embeddings: false },
  { id: 'gptoss-120b', name: 'GPT-OSS 120B (MoE 5.1B act)', total_params_b: 117, active_params_b: 5.1, layers: 36, kv_heads: 8, head_dim: 64, mla: false, max_ctx: 131072, tp_options: [1, 2, 4], quants: ['MXFP4'], hidden_size: 2880, vocab_size: 201088, tied_embeddings: false, sliding_window: 128, full_attention_layers: 18 },
  { id: 'qwen3-235b', name: 'Qwen3-235B-A22B (MoE)', total_params_b: 235, active_params_b: 22, layers: 94, kv_heads: 4, head_dim: 128, mla: false, max_ctx: 262144, tp_options: [4, 8, 16], quants: ['FP16', 'FP8', 'INT4'], hidden_size: 4096, vocab_size: 151936, tied_embeddings: false },
  { id: 'glm45', name: 'GLM-4.5 355B-A32B (MoE)', total_params_b: 355, active_params_b: 32, layers: 92, kv_heads: 8, head_dim: 128, mla: false, max_ctx: 131072, tp_options: [8, 16], quants: ['FP16', 'FP8', 'INT4'], hidden_size: 5120, vocab_size: 151552, tied_embeddings: false },
  { id: 'glm52', name: 'GLM-5.2 744B-A40B (MoE·DSA)', total_params_b: 744, active_params_b: 40, layers: 78, kv_heads: 8, head_dim: 128, mla: false, max_ctx: 1048576, tp_options: [8, 16], quants: ['FP8', 'NVFP4'], hidden_size: 5120, vocab_size: 151552, tied_embeddings: false },
  { id: 'dsv3', name: 'DeepSeek-V3 / R1 671B (MLA)', total_params_b: 671, active_params_b: 37, layers: 61, kv_heads: 0, head_dim: 0, mla: true, max_ctx: 131072, tp_options: [8, 16], quants: ['FP8', 'INT4'], hidden_size: 7168, vocab_size: 129280, tied_embeddings: false },
  { id: 'kimi-k2', name: 'Kimi K2 1T-A32B (MLA)', total_params_b: 1026, active_params_b: 32.5, layers: 61, kv_heads: 0, head_dim: 0, mla: true, max_ctx: 131072, tp_options: [16], quants: ['FP8', 'INT4'], hidden_size: 7168, vocab_size: 163840, tied_embeddings: false },
];

// price_per_gpu_hour: INDICATIVE market rental rates ($/GPU-hour) — [VERIFY] against your
// contracts. Admin-editable; the cost estimate uses these.
export const SEED_GPUS: GpuSku[] = [
  // --- NVIDIA datacenter ---
  { id: 'l4', name: 'L4 24 GB', mem_gb: 24, bw_tbs: 0.3, price_per_gpu_hour: 0.35 },
  { id: 'l40s', name: 'L40S 48 GB', mem_gb: 48, bw_tbs: 0.86, price_per_gpu_hour: 0.8 },
  { id: 'a100p', name: 'A100 80 GB PCIe', mem_gb: 80, bw_tbs: 1.94, price_per_gpu_hour: 1.5 },
  { id: 'a100s', name: 'A100 80 GB SXM', mem_gb: 80, bw_tbs: 2.04, price_per_gpu_hour: 1.8 },
  { id: 'h100', name: 'H100 80 GB SXM', mem_gb: 80, bw_tbs: 3.35, price_per_gpu_hour: 2.9 },
  { id: 'h100n', name: 'H100 NVL 94 GB', mem_gb: 94, bw_tbs: 3.9, price_per_gpu_hour: 3.2 },
  { id: 'h200', name: 'H200 141 GB (SXM/NVL)', mem_gb: 141, bw_tbs: 4.8, price_per_gpu_hour: 4.5 },
  { id: 'b200', name: 'B200 180 GB SXM', mem_gb: 180, bw_tbs: 8.0, price_per_gpu_hour: 6.5 },
  { id: 'b300', name: 'B300 288 GB (Blackwell Ultra)', mem_gb: 288, bw_tbs: 8.0, price_per_gpu_hour: 8.5 },
  // --- AMD Instinct (ROCm vLLM) ---
  { id: 'mi300x', name: 'MI300X 192 GB', mem_gb: 192, bw_tbs: 5.3, price_per_gpu_hour: 2.0 },
  { id: 'mi325x', name: 'MI325X 256 GB', mem_gb: 256, bw_tbs: 6.0, price_per_gpu_hour: 2.5 },
  { id: 'mi355x', name: 'MI355X 288 GB', mem_gb: 288, bw_tbs: 8.0, price_per_gpu_hour: 4.0 },
  // --- Workstation / consumer (single-box self-hosting; no NVLink, TP over PCIe) ---
  { id: 'rtxpro6000', name: 'RTX PRO 6000 Blackwell 96 GB', mem_gb: 96, bw_tbs: 1.79, price_per_gpu_hour: 1.8 },
  { id: 'rtx5090', name: 'RTX 5090 32 GB', mem_gb: 32, bw_tbs: 1.79, price_per_gpu_hour: 0.7 },
  { id: 'rtx4090', name: 'RTX 4090 24 GB', mem_gb: 24, bw_tbs: 1.01, price_per_gpu_hour: 0.4 },
];

export function seedCatalog(): { models: Model[]; gpus: GpuSku[] } {
  // deep clone so callers can't mutate the canonical seed
  return {
    models: SEED_MODELS.map((m) => ({ ...m, tp_options: [...m.tp_options], quants: [...m.quants] })),
    gpus: SEED_GPUS.map((g) => ({ ...g })),
  };
}
