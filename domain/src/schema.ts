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
