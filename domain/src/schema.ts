// §F validation — expressed ONCE, shared by client (UX) and server (authoritative) (AD-14, NFR-S-4).
import { z } from 'zod';
import { QUANTS } from './types.js';

export const quantSchema = z.enum(QUANTS);

/**
 * Model entity (§F.1). The mla-conditional rule is the critical structural check
 * (a GQA model with kv_heads=0 divides by zero in KV-per-token).
 */
export const modelSchema = z
  .object({
    id: z.string().min(1).max(64),
    name: z.string().min(1).max(128),
    total_params_b: z.number().positive(),
    active_params_b: z.number().positive(),
    layers: z.number().int().positive(),
    kv_heads: z.number().int().min(0),
    head_dim: z.number().int().min(0),
    mla: z.boolean(),
    max_ctx: z.number().int().positive().max(8_388_608),
    tp_options: z.array(z.number().int().positive()).min(1),
    quants: z.array(quantSchema).min(1),
    // Embedding geometry — optional for backward compatibility with catalogs written before
    // the un-quantised-tail weight term; supplying both sharpens low-bit weight estimates.
    hidden_size: z.number().int().positive().optional(),
    vocab_size: z.number().int().positive().optional(),
    tied_embeddings: z.boolean().optional(),
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
