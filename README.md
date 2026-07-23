# vLLM Capacity Planner

A self-hostable web tool for **sizing LLM inference deployments on GPUs** — how many GPUs, which tensor-parallel topology, how much KV-cache budget, how many concurrent sessions, expected throughput, and cost. It turns "serve model X at context Y for Z concurrent users" into a defensible, reproducible GPU/pod/node answer in seconds.

The sizing math is a deterministic memory-bandwidth model (the same roofline vLLM-style serving is bound by). See the in-app **Methodology** tab or [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Features

- **Deployment sizing** — pick a model, quantisation, KV dtype, context, concurrency and GPU SKU → TP size, weights/KV memory, concurrency-per-pod, pods, GPUs, nodes, TTFT, and throughput.
- **Visuals** — per-GPU HBM allocation (weights / KV / reserve), tensor-parallel topology across GPUs & nodes, infeasibility & multi-node signals.
- **Concurrency rubric** — sweep target concurrency and read off GPUs, per-request and aggregate tokens/sec.
- **Fleet + Cluster** — define a mixed-SKU GPU fleet, add sized models, see **utilisation vs free space** per SKU, and get hard-blocked from over-committing the hardware.
- **Saved configurations** — save a fleet + plan as a named, reloadable scenario.
- **Model catalog** — a browsable model-card catalog with admin CRUD, plus **import model geometry from Hugging Face** (fetches `config.json`, maps it to the model schema for review + commit).
- **Cost estimation** — set a $/GPU-hour per SKU and get cluster run-rate ($/hr·mo·yr), per-SKU line items, and **cost per million tokens**; export the estimate as CSV or JSON.
- **Light / dark**, keyboard-friendly, single-file-store persistence.

## Quick start (local)

Requires **Node.js 24+**.

```bash
git clone https://github.com/YOUR_USER/vllm-capacity-planner.git
cd vllm-capacity-planner
npm install
npm run build -w web          # build the SPA
npm start -w server           # serve API + SPA on http://localhost:8080
```

Or for web hot-reload during development, run the API and the Vite dev server in two terminals:

```bash
npm start -w server           # API on :8080
npm run dev -w web            # SPA on :5173, proxying /api to :8080
```

Data (the catalog + saved configs) persists to `server/data/catalog.json` by default (override with `DATA_FILE`).

## Testing

```bash
npm test            # domain acceptance vectors + server integration tests
```

The **sizing acceptance vectors** (`domain/src/__tests__/acceptance.test.ts`) pin the engine's outputs for known model/GPU combinations and run in CI.

## Architecture

A small TypeScript monorepo with a **shared, isomorphic domain core** — the sizing engine and validation schemas live once and run in the browser (live, <100 ms), on the server (authoritative), and in CI.

| Package | What |
|---|---|
| `domain/` | Pure TS: sizing engine, Zod validation, seed catalog, Hugging Face mapping, reconciliation. No I/O. |
| `web/`    | Svelte 5 + Vite SPA. Runs the engine client-side for instant feedback. |
| `server/` | Fastify API: catalog CRUD, sizing, reconciliation, saved configs, HF import; serves the built SPA. |
| `migrations/` | PostgreSQL schema (optional prod store — see below). |
| `deploy/` | Dockerfile + a generic Kubernetes example. |

**Persistence:** ships with a zero-dependency JSON file store (great for single-instance / local use). For horizontal scaling, swap in PostgreSQL behind the same `Store` interface (`migrations/001_init.sql` is the schema).

**Access model:** the app has an Admin / Standard-user role toggle enforced server-side. For a self-hosted single-tenant instance this is fine as-is; **for multi-user deployments, put it behind your own auth/SSO proxy** — the role is currently taken from a request header (see `server/src/app.ts`), a deliberate shim, not production authentication.

## Deploy

Build the image and run it anywhere:

```bash
docker build -t vllm-capacity-planner -f deploy/Dockerfile .
docker run -p 8080:8080 -v $PWD/data:/data vllm-capacity-planner
```

For Kubernetes, `deploy/k8s.yaml` is a generic example (set your image + Ingress host).

## Accuracy & caveats

Outputs are **first-order roofline estimates** — throughput ±40%, TTFT ±50%. They're planning figures, not commitments; validate against real benchmarks before procurement. The seeded model geometry and GPU bandwidths/prices are indicative — verify against authoritative sources for your models and hardware.

## License

MIT — see [LICENSE](LICENSE).
