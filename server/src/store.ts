// Persistence port (AD-5) — pure-JS, file-backed store (no native deps → runs in any stock
// Node image via hostPath). PostgreSQL (AD-7) is the horizontal-scale target and drops in behind
// the same interface; this file store fits a single-pod deployment (low write volume).
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { seedCatalog, type Model, type GpuSku } from '@vcp/domain';

export interface SavedConfigurationRow {
  id: string; owner_sub: string; name: string; schema_version: number;
  snapshot: unknown; created_at: string; updated_at: string;
}
export interface AuditEntry { actor_sub: string; action: string; detail?: string }

interface Persisted {
  models: Record<string, { data: Model; updated_at: string; updated_by: string }>;
  gpus: Record<string, { data: GpuSku; updated_at: string; updated_by: string }>;
  configs: Record<string, SavedConfigurationRow>;
  audit: { ts: string; actor_sub: string; action: string; detail?: string }[];
}

export class Store {
  private data: Persisted = { models: {}, gpus: {}, configs: {}, audit: [] };
  private path?: string;

  /** path=undefined → in-memory (tests); a path → JSON-file persistence loaded on boot. */
  constructor(path?: string) {
    this.path = path && path !== ':memory:' ? path : undefined;
    if (this.path && existsSync(this.path)) {
      this.data = JSON.parse(readFileSync(this.path, 'utf8'));
    } else if (this.path) {
      mkdirSync(dirname(this.path), { recursive: true });
    }
  }

  private flush() {
    if (!this.path) return;
    const tmp = `${this.path}.tmp`;
    writeFileSync(tmp, JSON.stringify(this.data));
    renameSync(tmp, this.path); // atomic
  }

  seedIfEmpty(now: string) {
    if (Object.keys(this.data.models).length > 0) return;
    const { models, gpus } = seedCatalog();
    for (const m of models) this.data.models[m.id] = { data: m, updated_at: now, updated_by: 'seed' };
    for (const g of gpus) this.data.gpus[g.id] = { data: g, updated_at: now, updated_by: 'seed' };
    this.flush();
  }

  listModels(): Model[] { return Object.keys(this.data.models).sort().map((id) => this.data.models[id].data); }
  listGpus(): GpuSku[] { return Object.keys(this.data.gpus).sort().map((id) => this.data.gpus[id].data); }
  getModel(id: string): Model | undefined { return this.data.models[id]?.data; }
  countModels() { return Object.keys(this.data.models).length; }
  countGpus() { return Object.keys(this.data.gpus).length; }

  upsertModel(m: Model, now: string, by: string) { this.data.models[m.id] = { data: m, updated_at: now, updated_by: by }; this.flush(); }
  deleteModel(id: string) { delete this.data.models[id]; this.flush(); }
  upsertGpu(g: GpuSku, now: string, by: string) { this.data.gpus[g.id] = { data: g, updated_at: now, updated_by: by }; this.flush(); }
  deleteGpu(id: string) { delete this.data.gpus[id]; this.flush(); }
  modelUpdatedAt(id: string) {
    const r = this.data.models[id];
    return r ? { updated_at: r.updated_at, updated_by: r.updated_by } : undefined;
  }

  replaceCatalog(models: Model[], gpus: GpuSku[], now: string, by: string) {
    this.data.models = {}; this.data.gpus = {};
    for (const m of models) this.data.models[m.id] = { data: m, updated_at: now, updated_by: by };
    for (const g of gpus) this.data.gpus[g.id] = { data: g, updated_at: now, updated_by: by };
    this.flush();
  }

  insertConfig(row: SavedConfigurationRow) { this.data.configs[row.id] = row; this.flush(); }
  listConfigsByOwner(sub: string): Omit<SavedConfigurationRow, 'snapshot'>[] {
    return Object.values(this.data.configs)
      .filter((c) => c.owner_sub === sub)
      .sort((a, b) => b.updated_at.localeCompare(a.updated_at))
      .map(({ snapshot, ...rest }) => rest);
  }
  getConfig(id: string): SavedConfigurationRow | undefined { return this.data.configs[id]; }
  renameConfig(id: string, name: string, now: string) { const c = this.data.configs[id]; if (c) { c.name = name; c.updated_at = now; this.flush(); } }
  deleteConfig(id: string) { delete this.data.configs[id]; this.flush(); }

  audit(e: AuditEntry, now: string) { this.data.audit.push({ ts: now, ...e }); this.flush(); }
  auditCount() { return this.data.audit.length; }
  close() { /* no-op for file store */ }
}
