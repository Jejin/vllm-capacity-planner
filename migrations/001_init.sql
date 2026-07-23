-- Forward-only initial schema (AD-7 PostgreSQL prod target). Mirrors server/src/store.ts.
CREATE TABLE IF NOT EXISTS model      (id TEXT PRIMARY KEY, data JSONB NOT NULL, updated_at TIMESTAMPTZ NOT NULL, updated_by TEXT);
CREATE TABLE IF NOT EXISTS gpu_sku    (id TEXT PRIMARY KEY, data JSONB NOT NULL, updated_at TIMESTAMPTZ NOT NULL, updated_by TEXT);
CREATE TABLE IF NOT EXISTS app_user   (sub TEXT PRIMARY KEY, display_name TEXT, role TEXT NOT NULL CHECK (role IN ('admin','user')));
CREATE TABLE IF NOT EXISTS saved_configuration (
  id TEXT PRIMARY KEY, owner_sub TEXT NOT NULL, name TEXT NOT NULL,
  schema_version INT NOT NULL, snapshot JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL, updated_at TIMESTAMPTZ NOT NULL);
CREATE INDEX IF NOT EXISTS idx_config_owner ON saved_configuration(owner_sub);
-- append-only, tamper-evident audit (AD-20 / Epic 7.2): no UPDATE/DELETE grants in prod
CREATE TABLE IF NOT EXISTS audit_log (
  id BIGSERIAL PRIMARY KEY, ts TIMESTAMPTZ NOT NULL,
  actor_sub TEXT NOT NULL, action TEXT NOT NULL, detail TEXT);
