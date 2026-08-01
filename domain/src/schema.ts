// §F validation — expressed ONCE, shared by client (UX) and server (authoritative) (AD-14, NFR-S-4).
import { z } from 'zod';
import { GPU_ARCHES, QUANTS } from './types.js';

export const quantSchema = z.enum(QUANTS);

/**
 * `owner/name` — what `vllm serve` takes as its positional argument. Anything with a space in it
 * is not an artifact id, it is a display name, and the shell will split it into three arguments.
 */
export const HF_ID_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/;
const hfIdSchema = z.string().max(128).regex(HF_ID_RE, 'must be a Hugging Face id of the form owner/name');

export const deploymentVariantSchema = z.object({
  source: z.enum(['checkpoint', 'online', 'none']),
  hf_id: hfIdSchema.optional(),
  method: z.string().min(1).max(64).optional(),
  revision: z.string().min(1).max(64).optional(),
  kv_scale_source: z.enum(['checkpoint', 'calibrated', 'runtime', 'none']).optional(),
});

/**
 * Model entity (§F.1). The mla-conditional rule is the critical structural check
 * (a GQA model with kv_heads=0 divides by zero in KV-per-token).
 */
export const modelSchema = z
  .object({
    id: z.string().min(1).max(64),
    name: z.string().min(1).max(128),
    // Deployment identity, kept separate from the display name so the label can never be
    // passed to a runtime (§4.1). Optional for backward compatibility; without it no launch
    // command is generated.
    hf_id: hfIdSchema.optional(),
    revision: z.string().min(1).max(64).optional(),
    // NB: z.record(enum, ...) is EXHAUSTIVE in Zod 4 — keys are checked in the refinement below.
    deployments: z.record(z.string(), deploymentVariantSchema).optional(),
    total_params_b: z.number().positive(),
    active_params_b: z.number().positive(),
    layers: z.number().int().positive(),
    kv_heads: z.number().int().min(0),
    head_dim: z.number().int().min(0),
    mla: z.boolean(),
    // Per-layer MLA latent width (kv_lora_rank + qk_rope_head_dim). MLA models only —
    // it is checkpoint geometry, and a GQA model has no latent to describe.
    mla_latent_elems: z.number().int().positive().optional(),
    max_ctx: z.number().int().positive().max(8_388_608),
    tp_options: z.array(z.number().int().positive()).min(1),
    quants: z.array(quantSchema).min(1),
    // Embedding geometry — optional for backward compatibility with catalogs written before
    // the un-quantised-tail weight term; supplying both sharpens low-bit weight estimates.
    hidden_size: z.number().int().positive().optional(),
    vocab_size: z.number().int().positive().optional(),
    tied_embeddings: z.boolean().optional(),
    // FFN width one token traverses per layer (MoE: per-expert width x experts per token).
    // Drives the sharded half of the prefill activation reserve.
    intermediate_size: z.number().int().positive().optional(),
    // Local/global attention — optional; both required together to cap KV on windowed layers.
    sliding_window: z.number().int().positive().optional(),
    full_attention_layers: z.number().int().min(0).optional(),
    // Linear/recurrent attention — constant state per request, no per-token cache.
    linear_attention_layers: z.number().int().min(0).optional(),
    linear_state_bytes_per_layer: z.number().int().min(0).optional(),
    // Mixed-precision checkpoints: the dense remainder and what it keeps under each quant.
    dense_params_b: z.number().positive().optional(),
    // NB: z.record(enum, ...) is EXHAUSTIVE in Zod 4 — it would demand every quant as a key.
    // Keys are validated against QUANTS in the refinement below instead.
    mixed_precision: z.record(z.string(), quantSchema).optional(),
  })
  .superRefine((m, ctx) => {
    if (m.active_params_b > m.total_params_b) {
      ctx.addIssue({ code: 'custom', path: ['active_params_b'], message: 'active_params_b must be ≤ total_params_b' });
    }
    // mla-conditional (§F): mla=false ⇒ kv_heads>0 ∧ head_dim>0 ; mla=true ⇒ both = 0
    if (!m.mla) {
      if (m.kv_heads <= 0) ctx.addIssue({ code: 'custom', path: ['kv_heads'], message: 'GQA model (mla=false) requires kv_heads > 0' });
      if (m.head_dim <= 0) ctx.addIssue({ code: 'custom', path: ['head_dim'], message: 'GQA model (mla=false) requires head_dim > 0' });
      // A latent width on a GQA entry is silently ignored by the engine — reject it rather
      // than let a catalogue carry geometry that reads as if it were being used.
      if (m.mla_latent_elems != null) {
        ctx.addIssue({ code: 'custom', path: ['mla_latent_elems'], message: 'GQA model (mla=false) has no latent — mla_latent_elems applies to MLA models only' });
      }
    } else {
      if (m.kv_heads !== 0) ctx.addIssue({ code: 'custom', path: ['kv_heads'], message: 'MLA model (mla=true) must have kv_heads = 0 (unused)' });
      if (m.head_dim !== 0) ctx.addIssue({ code: 'custom', path: ['head_dim'], message: 'MLA model (mla=true) must have head_dim = 0 (unused)' });
    }
    // Embedding geometry is all-or-nothing: one half alone can't size the un-quantised tail.
    if ((m.hidden_size == null) !== (m.vocab_size == null)) {
      ctx.addIssue({ code: 'custom', path: ['vocab_size'], message: 'hidden_size and vocab_size must be supplied together (they size the 16-bit embedding/lm_head tail)' });
    }
    if (m.vocab_size != null && m.hidden_size != null) {
      const tailB = ((m.tied_embeddings ? 1 : 2) * m.vocab_size * m.hidden_size) / 1e9;
      if (tailB >= m.total_params_b) {
        ctx.addIssue({ code: 'custom', path: ['vocab_size'], message: 'embedding tail must be smaller than total_params_b — check hidden_size / vocab_size' });
      }
    }
    // Attention-regime split must account for every layer: full + windowed + linear = layers.
    const linear = m.linear_attention_layers ?? 0;
    const cached = m.layers - linear;
    if (linear > m.layers) {
      ctx.addIssue({ code: 'custom', path: ['linear_attention_layers'], message: 'linear_attention_layers must be ≤ layers' });
    }
    if (m.full_attention_layers != null && m.full_attention_layers > cached) {
      ctx.addIssue({ code: 'custom', path: ['full_attention_layers'], message: 'full_attention_layers + linear_attention_layers must be ≤ layers' });
    }
    // Any layer that is neither full nor linear is windowed, which needs a window length.
    const windowed = cached - (m.full_attention_layers ?? cached);
    if (windowed > 0 && m.sliding_window == null) {
      ctx.addIssue({ code: 'custom', path: ['sliding_window'], message: 'layers are left over after full + linear, so sliding_window is required to size them' });
    }
    if (m.sliding_window != null && m.full_attention_layers == null) {
      ctx.addIssue({ code: 'custom', path: ['full_attention_layers'], message: 'sliding_window needs full_attention_layers to know how the layers split' });
    }
    // A linear layer with no state size would be counted as free, understating memory.
    if (linear > 0 && m.linear_state_bytes_per_layer == null) {
      ctx.addIssue({ code: 'custom', path: ['linear_state_bytes_per_layer'], message: 'linear_attention_layers requires linear_state_bytes_per_layer (its constant per-request state)' });
    }
    // Mixed precision needs a dense split to apply to, and the split must leave a body to quantise.
    for (const k of Object.keys(m.mixed_precision ?? {})) {
      if (!(QUANTS as readonly string[]).includes(k)) {
        ctx.addIssue({ code: 'custom', path: ['mixed_precision', k], message: `"${k}" is not a known quantisation` });
      }
    }
    if (m.mixed_precision && Object.keys(m.mixed_precision).length > 0 && m.dense_params_b == null) {
      ctx.addIssue({ code: 'custom', path: ['dense_params_b'], message: 'mixed_precision requires dense_params_b — the parameters that stay at the higher precision' });
    }
    // Deployment variants: keys must be real quants the model actually offers, and the
    // source must agree with whether a --quantization method is present. A checkpoint that
    // also passes the flag conflicts with its own metadata; an online path without one
    // silently launches the base precision the plan was not sized for.
    for (const [k, v] of Object.entries(m.deployments ?? {})) {
      const path = ['deployments', k];
      if (!(QUANTS as readonly string[]).includes(k)) {
        ctx.addIssue({ code: 'custom', path, message: `"${k}" is not a known quantisation` });
        continue;
      }
      if (!m.quants.includes(k as (typeof QUANTS)[number])) {
        ctx.addIssue({ code: 'custom', path, message: `deployment declared for ${k}, which is not in this model's quants` });
      }
      if (v.source === 'online' && !v.method) {
        ctx.addIssue({ code: 'custom', path: [...path, 'method'], message: 'online quantisation requires a --quantization method' });
      }
      if (v.source !== 'online' && v.method) {
        ctx.addIssue({ code: 'custom', path: [...path, 'method'], message: `a ${v.source} artifact must not also pass --quantization — the flag conflicts with the checkpoint's own metadata` });
      }
      if (v.source === 'checkpoint' && !v.hf_id && !m.hf_id) {
        ctx.addIssue({ code: 'custom', path: [...path, 'hf_id'], message: 'checkpoint source needs an artifact id, on the variant or the model' });
      }
    }
    if (m.dense_params_b != null && m.dense_params_b >= m.total_params_b) {
      ctx.addIssue({ code: 'custom', path: ['dense_params_b'], message: 'dense_params_b must be smaller than total_params_b' });
    }
  });

/** GPU SKU entity (§F.2). */
export const gpuSkuSchema = z.object({
  id: z.string().min(1).max(64),
  name: z.string().min(1).max(128),
  mem_gb: z.number().int().positive(),
  bw_tbs: z.number().positive(),
  // Kernel-support architecture. Optional for backward compatibility; without it the runtime
  // verdict is `unverified` rather than assumed good.
  arch: z.enum(GPU_ARCHES).optional(),
  // Dense FP16/BF16 TFLOPS. Drives the compute-bound TTFT estimate; without it the engine falls
  // back to a weight-streaming floor. Omitted here, Zod stripped it on every upsert and catalog
  // import, so editing a seeded SKU silently downgraded its own TTFT to that floor.
  tflops_fp16: z.number().positive().optional(),
  price_per_gpu_hour: z.number().nonnegative().optional(), // $/GPU-hour for cost estimates
});

/** Sizing input constraints (§F.3, FR-9). selected_ctx ≤ max_ctx is checked against the model at the call site. */
export const sizingInputSchema = z.object({
  quant: quantSchema,
  kv_dtype_bytes: z.number().positive(),
  selected_ctx: z.number().int().positive(),
  avg_context_utilisation: z.number().gt(0).max(1),
  target_concurrency: z.number().int().positive(),
  mem_util_fraction: z.number().gt(0).max(1),
  gpus_per_node: z.number().int().positive(),
});

export const catalogSchema = z.object({
  models: z.array(modelSchema).min(1), // non-empty guard (FR-4/FR-8)
  gpus: z.array(gpuSkuSchema).min(1),
});

export type ValidationIssue = { path: string; message: string };

/** Uniform error-envelope shape (AD-14): { error: { code, message, fields?[] } }. */
export function toFields(err: z.ZodError): ValidationIssue[] {
  return err.issues.map((i) => ({ path: i.path.join('.'), message: i.message }));
}
