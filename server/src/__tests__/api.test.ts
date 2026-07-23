// Server integration tests (Fastify inject) — covers Epics 1/2/3/4/5 story acceptance criteria.
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { buildApp } from '../app.js';
import type { FastifyInstance } from 'fastify';

let app: FastifyInstance;
const admin = { 'x-dev-sub': 'u-admin', 'x-dev-role': 'admin' };
const user = { 'x-dev-sub': 'u-rana', 'x-dev-role': 'user' };
const user2 = { 'x-dev-sub': 'u-karim', 'x-dev-role': 'user' };

beforeEach(() => { app = buildApp(); });
afterEach(async () => { await app.close(); });

describe('Epic 1 — Access & seeded catalog', () => {
  it('rejects unauthenticated requests (FR-25)', async () => {
    const r = await app.inject({ method: 'GET', url: '/api/v1/catalog' });
    expect(r.statusCode).toBe(401);
  });
  it('lists the 13 seeded models + 7 GPUs (FR-1/7/5/8)', async () => {
    const r = await app.inject({ method: 'GET', url: '/api/v1/catalog', headers: user });
    expect(r.statusCode).toBe(200);
    const body = r.json();
    expect(body.models).toHaveLength(13);
    expect(body.gpus).toHaveLength(7);
  });
  it('healthz/readyz respond', async () => {
    expect((await app.inject({ url: '/healthz' })).statusCode).toBe(200);
    expect((await app.inject({ url: '/readyz' })).json().models).toBe(13);
  });
});

describe('Epic 3 — Catalog curation + RBAC (FR-2/3/4/29)', () => {
  const newModel = { id: 'test-x', name: 'Test X', total_params_b: 10, active_params_b: 10, layers: 32, kv_heads: 8, head_dim: 128, mla: false, max_ctx: 131072, tp_options: [1, 2], quants: ['FP16'] };
  it('standard user CANNOT create a model — server-side (FR-29)', async () => {
    const r = await app.inject({ method: 'POST', url: '/api/v1/models', headers: user, payload: newModel });
    expect(r.statusCode).toBe(403);
  });
  it('admin can create a valid model (FR-2)', async () => {
    const r = await app.inject({ method: 'POST', url: '/api/v1/models', headers: admin, payload: newModel });
    expect(r.statusCode).toBe(201);
    const list = (await app.inject({ url: '/api/v1/catalog', headers: user })).json();
    expect(list.models.find((m: any) => m.id === 'test-x')).toBeTruthy();
  });
  it('rejects a GQA model with kv_heads=0 with field-level message (FR-2 / AD-14)', async () => {
    const r = await app.inject({ method: 'POST', url: '/api/v1/models', headers: admin, payload: { ...newModel, kv_heads: 0 } });
    expect(r.statusCode).toBe(422);
    expect(r.json().error.code).toBe('validation');
    expect(r.json().error.fields.some((f: any) => f.path === 'kv_heads')).toBe(true);
  });
  it('blocks deleting the last model (FR-4)', async () => {
    // delete down to one
    const ids = (await app.inject({ url: '/api/v1/catalog', headers: admin })).json().models.map((m: any) => m.id);
    for (const id of ids.slice(0, ids.length - 1)) {
      await app.inject({ method: 'DELETE', url: `/api/v1/models/${id}`, headers: admin });
    }
    const last = ids[ids.length - 1];
    const r = await app.inject({ method: 'DELETE', url: `/api/v1/models/${last}`, headers: admin });
    expect(r.statusCode).toBe(409);
  });
  it('reset restores the 13+7 seed (FR-23)', async () => {
    await app.inject({ method: 'DELETE', url: '/api/v1/models/glm52', headers: admin });
    await app.inject({ method: 'POST', url: '/api/v1/catalog/reset', headers: admin });
    expect((await app.inject({ url: '/readyz' })).json().models).toBe(13);
  });
  it('export → import round-trips (FR-22); import denied to standard users', async () => {
    const exported = (await app.inject({ url: '/api/v1/catalog/export', headers: user })).json();
    expect((await app.inject({ method: 'POST', url: '/api/v1/catalog/import', headers: user, payload: exported })).statusCode).toBe(403);
    const imp = await app.inject({ method: 'POST', url: '/api/v1/catalog/import', headers: admin, payload: exported });
    expect(imp.json()).toEqual({ models: 13, gpus: 7 });
  });
});

describe('Epic 2 — Sizing endpoint (FR-9/10)', () => {
  it('computes AC-1 (Llama 70B → 10 GPUs, TP2) authoritatively', async () => {
    const r = await app.inject({ method: 'POST', url: '/api/v1/sizing', headers: user, payload: {
      model_id: 'llama33-70b', gpu_id: 'h200',
      input: { quant: 'FP8', kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6, target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8 },
    } });
    expect(r.statusCode).toBe(200);
    const s = r.json();
    expect(s.tp).toBe(2); expect(s.gpus).toBe(10); expect(s.nodes).toBe(2);
  });
  it('rejects selected_ctx > max_ctx (FR-9)', async () => {
    const r = await app.inject({ method: 'POST', url: '/api/v1/sizing', headers: user, payload: {
      model_id: 'llama33-70b', gpu_id: 'h200',
      input: { quant: 'FP8', kv_dtype_bytes: 1, selected_ctx: 999999999, avg_context_utilisation: 0.6, target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8 },
    } });
    expect(r.statusCode).toBe(422);
  });
});

describe('Epic 4 — Reconciliation hard-block + headroom (FR-19..21)', () => {
  it('flags per-SKU over-commitment and returns headroom verdicts', async () => {
    const r = await app.inject({ method: 'POST', url: '/api/v1/reconcile', headers: user, payload: {
      fleet: [{ gpu_sku_id: 'h200', gpus_per_node: 8, node_count: 5 }, { gpu_sku_id: 'h100', gpus_per_node: 8, node_count: 2 }],
      commitments: [{ gpu_sku_id: 'h200', gpus: 28 }, { gpu_sku_id: 'h100', gpus: 18 }],
      candidate: { gpu_sku_id: 'h100', gpus: 4 },
    } });
    const b = r.json();
    expect(b.over_committed).toBe(true);
    expect(b.headroom.verdict).toBe('shortage');
    const h100 = b.per_sku.find((x: any) => x.gpu_sku_id === 'h100');
    expect(h100.over_committed).toBe(true);
  });
});

describe('Epic 5 — Saved configs: owner-scoped + snapshot (FR-27/28/29)', () => {
  it('saves and lists only the owner\'s configs; blocks cross-user access (FR-29)', async () => {
    const snapshot = { fleet: [{ gpu_sku_id: 'h200', gpus_per_node: 8, node_count: 5 }], plan: [], geometry: {} };
    const created = await app.inject({ method: 'POST', url: '/api/v1/configs', headers: user, payload: { name: 'RFP-Acme', snapshot } });
    expect(created.statusCode).toBe(201);
    const cid = created.json().id;
    // owner sees it
    expect((await app.inject({ url: '/api/v1/configs', headers: user })).json()).toHaveLength(1);
    // other user does NOT see it and cannot fetch it by id (IDOR/BOLA)
    expect((await app.inject({ url: '/api/v1/configs', headers: user2 })).json()).toHaveLength(0);
    expect((await app.inject({ url: `/api/v1/configs/${cid}`, headers: user2 })).statusCode).toBe(403);
    // owner can fetch, and the geometry snapshot is embedded (AD-4)
    const got = (await app.inject({ url: `/api/v1/configs/${cid}`, headers: user })).json();
    expect(got.snapshot.fleet[0].gpu_sku_id).toBe('h200');
  });
});
