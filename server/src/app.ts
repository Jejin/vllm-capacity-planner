// Fastify app factory (AD-6 stateless). Thin HTTP adapters over the domain core.
import Fastify, { type FastifyInstance, type FastifyReply, type FastifyRequest } from 'fastify';
import {
  computeSizing, reconcile, headroomCheck, modelSchema, gpuSkuSchema, sizingInputSchema,
  catalogSchema, toFields, hfConfigToModel, type Model, type GpuSku,
} from '@vcp/domain';
import { z } from 'zod';
import { existsSync } from 'node:fs';
import { Store } from './store.js';

export type Role = 'admin' | 'user';
export interface Identity { sub: string; role: Role; name?: string }

// Uniform error envelope (AD-14).
type ErrCode = 'unauthenticated' | 'forbidden' | 'not_found' | 'validation' | 'conflict' | 'bad_request';
function fail(reply: FastifyReply, status: number, code: ErrCode, message: string, fields?: { path: string; message: string }[]) {
  return reply.status(status).send({ error: { code, message, ...(fields ? { fields } : {}) } });
}

/**
 * DEV AUTH SHIM — stands in for OIDC token validation (AD-8/AD-21, Epic 1.3 / Epic 7.3).
 * In production this is replaced by JWKS/alg/iss/aud/exp validation of the bearer token;
 * role derives from IdP claims. Here identity comes from x-dev-sub / x-dev-role headers.
 * [NOT PRODUCTION AUTH] — clearly marked so it is never mistaken for the real gate.
 */
function identityFrom(req: FastifyRequest): Identity | null {
  const sub = req.headers['x-dev-sub'];
  const role = req.headers['x-dev-role'];
  if (typeof sub !== 'string' || !sub) return null;
  return { sub, role: role === 'admin' ? 'admin' : 'user' };
}

export function buildApp(opts: { dbPath?: string; now?: () => string; webDist?: string } = {}): FastifyInstance {
  const app = Fastify({ logger: false });
  // Tolerate empty JSON bodies (e.g. DELETE with content-type: application/json) → undefined, not 400.
  app.addContentTypeParser('application/json', { parseAs: 'string' }, (_req, body, done) => {
    if (!body || (body as string).trim() === '') return done(null, undefined);
    try { done(null, JSON.parse(body as string)); } catch (err) { done(err as Error); }
  });
  const store = new Store(opts.dbPath ?? ':memory:');
  const now = opts.now ?? (() => new Date().toISOString());
  store.seedIfEmpty(now());

  // Never cache HTML (index.html) so a new deploy's hashed bundles are always loaded.
  app.addHook('onSend', async (_req, reply) => {
    const ct = reply.getHeader('content-type');
    if (typeof ct === 'string' && ct.includes('text/html')) reply.header('cache-control', 'no-store, must-revalidate');
  });

  // authN: every protected route requires an identity (Epic 1.3 / FR-25 base)
  const authed = (handler: (req: FastifyRequest, reply: FastifyReply, id: Identity) => unknown) =>
    async (req: FastifyRequest, reply: FastifyReply) => {
      const id = identityFrom(req);
      if (!id) return fail(reply, 401, 'unauthenticated', 'Sign-in required.');
      return handler(req, reply, id);
    };
  // authZ: admin-only (verb-independent) (AD-9 / FR-29)
  const adminOnly = (handler: (req: FastifyRequest, reply: FastifyReply, id: Identity) => unknown) =>
    authed((req, reply, id) => {
      if (id.role !== 'admin') return fail(reply, 403, 'forbidden', 'Admin role required.');
      return handler(req, reply, id);
    });

  app.get('/healthz', async () => ({ status: 'ok' }));
  app.get('/readyz', async () => ({ status: 'ready', models: store.countModels() }));

  // ── Catalog reads (any authenticated user) — FR-1, FR-7 ──
  app.get('/api/v1/catalog', authed(async (req, reply, id) => {
    store.audit({ actor_sub: id.sub, action: 'catalog.read' }, now());
    return reply.send({ models: store.listModels(), gpus: store.listGpus() });
  }));
  app.get('/api/v1/catalog/export', authed(async (_req, reply) =>
    reply.send({ models: store.listModels(), gpus: store.listGpus() })));

  // ── Model curation (admin) — FR-2/3/4/6 ──
  app.post('/api/v1/models', adminOnly(async (req, reply, id) => {
    const parsed = modelSchema.safeParse(req.body);
    if (!parsed.success) return fail(reply, 422, 'validation', 'Model failed validation.', toFields(parsed.error));
    store.upsertModel(parsed.data as Model, now(), id.sub);
    store.audit({ actor_sub: id.sub, action: 'model.create', detail: parsed.data.id }, now());
    return reply.status(201).send(parsed.data);
  }));
  app.put('/api/v1/models/:mid', adminOnly(async (req, reply, id) => {
    const { mid } = req.params as { mid: string };
    if (!store.getModel(mid)) return fail(reply, 404, 'not_found', 'Model not found.');
    const parsed = modelSchema.safeParse(req.body);
    if (!parsed.success) return fail(reply, 422, 'validation', 'Model failed validation.', toFields(parsed.error));
    store.upsertModel(parsed.data as Model, now(), id.sub);
    store.audit({ actor_sub: id.sub, action: 'model.edit', detail: mid }, now());
    return reply.send(parsed.data);
  }));
  app.delete('/api/v1/models/:mid', adminOnly(async (req, reply, id) => {
    const { mid } = req.params as { mid: string };
    if (!store.getModel(mid)) return fail(reply, 404, 'not_found', 'Model not found.');
    if (store.countModels() <= 1) return fail(reply, 409, 'conflict', 'Cannot delete the last remaining model.');
    store.deleteModel(mid);
    store.audit({ actor_sub: id.sub, action: 'model.delete', detail: mid }, now());
    return reply.status(204).send();
  }));

  // ── GPU curation (admin) — FR-7/8 ──
  app.post('/api/v1/gpus', adminOnly(async (req, reply, id) => {
    const parsed = gpuSkuSchema.safeParse(req.body);
    if (!parsed.success) return fail(reply, 422, 'validation', 'GPU SKU failed validation.', toFields(parsed.error));
    store.upsertGpu(parsed.data as GpuSku, now(), id.sub);
    return reply.status(201).send(parsed.data);
  }));
  app.delete('/api/v1/gpus/:gid', adminOnly(async (req, reply, id) => {
    const { gid } = req.params as { gid: string };
    if (store.countGpus() <= 1) return fail(reply, 409, 'conflict', 'Cannot delete the last remaining GPU SKU.');
    store.deleteGpu(gid);
    store.audit({ actor_sub: id.sub, action: 'gpu.delete', detail: gid }, now());
    return reply.status(204).send();
  }));

  // ── Import / reset (admin) — FR-22/23 ──
  app.post('/api/v1/catalog/import', adminOnly(async (req, reply, id) => {
    const parsed = catalogSchema.safeParse(req.body);
    if (!parsed.success) return fail(reply, 422, 'validation', 'Malformed catalog document.', toFields(parsed.error));
    store.replaceCatalog(parsed.data.models as Model[], parsed.data.gpus as GpuSku[], now(), id.sub);
    store.audit({ actor_sub: id.sub, action: 'catalog.import' }, now());
    return reply.send({ models: store.countModels(), gpus: store.countGpus() });
  }));
  app.post('/api/v1/catalog/reset', adminOnly(async (_req, reply, id) => {
    const { seedCatalog } = await import('@vcp/domain');
    const { models, gpus } = seedCatalog();
    store.replaceCatalog(models, gpus, now(), id.sub);
    store.audit({ actor_sub: id.sub, action: 'catalog.reset' }, now());
    return reply.send({ models: store.countModels(), gpus: store.countGpus() });
  }));

  // ── Sizing (any) — FR-9/10 (server also computes authoritatively; AD-2) ──
  const sizingReq = z.object({ model_id: z.string(), gpu_id: z.string(), input: sizingInputSchema });
  app.post('/api/v1/sizing', authed(async (req, reply) => {
    const parsed = sizingReq.safeParse(req.body);
    if (!parsed.success) return fail(reply, 422, 'validation', 'Invalid sizing request.', toFields(parsed.error));
    const model = store.getModel(parsed.data.model_id);
    const gpu = store.listGpus().find((g) => g.id === parsed.data.gpu_id);
    if (!model || !gpu) return fail(reply, 404, 'not_found', 'Model or GPU not found.');
    if (parsed.data.input.selected_ctx > model.max_ctx)
      return fail(reply, 422, 'validation', 'selected_ctx exceeds model.max_ctx.', [{ path: 'input.selected_ctx', message: `must be ≤ ${model.max_ctx}` }]);
    const result = computeSizing(model, gpu, parsed.data.input);
    return reply.send(JSON.parse(JSON.stringify(result, (_k, v) => (typeof v === 'bigint' ? Number(v) : v))));
  }));

  // ── Reconciliation (any) — FR-19..21 ──
  const reconReq = z.object({
    fleet: z.array(z.object({ gpu_sku_id: z.string(), gpus_per_node: z.number().int().positive(), node_count: z.number().int().positive() })),
    commitments: z.array(z.object({ gpu_sku_id: z.string(), gpus: z.number().int().nonnegative() })),
    candidate: z.object({ gpu_sku_id: z.string(), gpus: z.number().int().positive() }).optional(),
  });
  app.post('/api/v1/reconcile', authed(async (req, reply) => {
    const parsed = reconReq.safeParse(req.body);
    if (!parsed.success) return fail(reply, 422, 'validation', 'Invalid reconcile request.', toFields(parsed.error));
    const gpuById = new Map(store.listGpus().map((g) => [g.id, g]));
    const rows = reconcile(parsed.data.fleet, parsed.data.commitments, gpuById)
      .map((r) => ({ ...r, fleet_bytes: Number(r.fleet_bytes), committed_bytes: Number(r.committed_bytes), available_bytes: Number(r.available_bytes) }));
    const over = rows.some((r) => r.over_committed);
    let headroom = undefined as unknown;
    if (parsed.data.candidate) {
      const full = reconcile(parsed.data.fleet, parsed.data.commitments, gpuById);
      headroom = headroomCheck(parsed.data.candidate.gpu_sku_id, parsed.data.candidate.gpus, full);
    }
    return reply.send({ per_sku: rows, over_committed: over, headroom });
  }));

  // ── Hugging Face import (admin) — FR-30/31/32 ──
  // Fetch config.json, map to §F (AD-11). Egress hardened to Hugging Face hosts (AD-19); in
  // production this routes through the allowlisted proxy (AD-12/18). Human-confirmed commit only.
  app.post('/api/v1/huggingface/fetch', adminOnly(async (req, reply, id) => {
    const parsed = z.object({ model_id: z.string().min(1).max(200) }).safeParse(req.body);
    if (!parsed.success) return fail(reply, 422, 'validation', 'model_id required.');
    const mid = parsed.data.model_id.trim().replace(/^https?:\/\/huggingface\.co\//, '').replace(/\/+$/, '');
    if (!/^[\w.-]+\/[\w.-]+$/.test(mid)) return fail(reply, 400, 'bad_request', 'Expected an owner/model id, e.g. Qwen/Qwen2.5-72B-Instruct.');
    try {
      const res = await fetchHf(`https://huggingface.co/${mid}/resolve/main/config.json`);
      if (res.status === 401 || res.status === 403) return fail(reply, 502, 'bad_request', 'This model is gated on Hugging Face (needs an access token). Try a public model.');
      if (!res.ok) return fail(reply, 502, 'not_found', `Hugging Face returned ${res.status} for "${mid}".`);
      const cfg = (await res.json()) as Record<string, unknown>;
      const { mapped, missing, detectedMla } = hfConfigToModel(mid, cfg as any);
      store.audit({ actor_sub: id.sub, action: 'hf.fetch', detail: mid }, now()); // FR-32 audited egress
      return reply.send({
        model_id: mid, mapped, missing, detectedMla,
        card: { architectures: cfg.architectures, model_type: cfg.model_type, hidden_size: cfg.hidden_size, num_hidden_layers: cfg.num_hidden_layers, num_attention_heads: cfg.num_attention_heads, num_key_value_heads: cfg.num_key_value_heads, max_position_embeddings: cfg.max_position_embeddings },
      });
    } catch {
      return fail(reply, 502, 'not_found', 'Could not reach Hugging Face (egress blocked or timed out).');
    }
  }));

  // ── Saved configurations (per-user, owner-scoped) — FR-27/28/29 ──
  const saveReq = z.object({ name: z.string().min(1).max(120), snapshot: z.record(z.string(), z.unknown()) });
  app.post('/api/v1/configs', authed(async (req, reply, id) => {
    const parsed = saveReq.safeParse(req.body);
    if (!parsed.success) return fail(reply, 422, 'validation', 'Invalid configuration.', toFields(parsed.error));
    const ts = now();
    const row = { id: cryptoId(), owner_sub: id.sub, name: parsed.data.name, schema_version: 1, snapshot: parsed.data.snapshot, created_at: ts, updated_at: ts };
    store.insertConfig(row);
    return reply.status(201).send({ id: row.id, name: row.name });
  }));
  app.get('/api/v1/configs', authed(async (_req, reply, id) => reply.send(store.listConfigsByOwner(id.sub))));
  const ownedConfig = (handler: (req: FastifyRequest, reply: FastifyReply, id: Identity, row: import('./store.js').SavedConfigurationRow) => unknown) =>
    authed((req, reply, id) => {
      const row = store.getConfig((req.params as { cid: string }).cid);
      if (!row) return fail(reply, 404, 'not_found', 'Configuration not found.');
      if (row.owner_sub !== id.sub && id.role !== 'admin') return fail(reply, 403, 'forbidden', 'Not your configuration.'); // object-level (FR-29)
      return handler(req, reply, id, row);
    });
  app.get('/api/v1/configs/:cid', ownedConfig(async (_req, reply, _id, row) => reply.send(row)));
  app.patch('/api/v1/configs/:cid', ownedConfig(async (req, reply, _id, row) => {
    const name = (req.body as { name?: string })?.name;
    if (!name) return fail(reply, 400, 'bad_request', 'name required.');
    store.renameConfig(row.id, name, now());
    return reply.send({ id: row.id, name });
  }));
  app.delete('/api/v1/configs/:cid', ownedConfig(async (_req, reply, _id, row) => { store.deleteConfig(row.id); return reply.status(204).send(); }));

  // Serve the built SPA (single pod serves API + web). SPA fallback for client routing.
  if (opts.webDist && existsSync(opts.webDist)) {
    {
      // Hashed assets are immutable & cacheable; index.html must NOT be cached so new deploys
      // (which reference new hashed bundles) are always picked up by the browser.
      app.register(import('@fastify/static'), {
        root: opts.webDist,
        prefix: '/',
        setHeaders: (res, path) => {
          if (path.endsWith('index.html')) res.setHeader('cache-control', 'no-store, must-revalidate');
          else if (path.includes('/assets/')) res.setHeader('cache-control', 'public, max-age=31536000, immutable');
        },
      });
      app.setNotFoundHandler((req, reply) => {
        if (req.method === 'GET' && !req.url.startsWith('/api') && !req.url.startsWith('/healthz') && !req.url.startsWith('/readyz')) {
          return reply.header('cache-control', 'no-store, must-revalidate').sendFile('index.html');
        }
        return fail(reply, 404, 'not_found', 'Not found.');
      });
    }
  }

  app.decorate('store', store);
  return app;
}

function cryptoId(): string {
  return 'cfg_' + Array.from(crypto.getRandomValues(new Uint8Array(9)), (b) => b.toString(16).padStart(2, '0')).join('');
}

/** Fetch from Hugging Face with SSRF-ish hardening (AD-19): timeout + final host must be HF. */
async function fetchHf(url: string): Promise<Response> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 12_000);
  try {
    const res = await fetch(url, { signal: ctrl.signal, redirect: 'follow', headers: { 'user-agent': 'vllm-capacity-planner' } });
    const host = new URL(res.url).hostname;
    if (!/(^|\.)huggingface\.co$|(^|\.)hf\.co$/.test(host)) throw new Error('redirected off Hugging Face');
    return res;
  } finally {
    clearTimeout(t);
  }
}
