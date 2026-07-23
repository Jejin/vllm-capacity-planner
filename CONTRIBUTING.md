# Contributing

Thanks for your interest in improving **vLLM Capacity Planner**! Contributions of all kinds are welcome — bug fixes, new GPU/model catalog entries, UI polish, and improvements to the sizing model.

## Getting started

Requires **Node.js 20+**.

```bash
git clone https://github.com/YOUR_USER/vllm-capacity-planner.git
cd vllm-capacity-planner
npm install
npm test                      # domain + server tests must pass
npm start -w server           # API + SPA on :8080
npm run dev -w web            # (optional) SPA hot-reload on :5173
```

## Project layout

| Package | What lives here |
|---|---|
| `domain/` | The pure, framework-free core: the sizing **engine**, Zod **validation schemas**, the seed **catalog**, the Hugging Face mapping, and reconciliation. **No I/O.** This is imported by the browser, the server, and the tests. |
| `web/`    | Svelte 5 + Vite SPA. Runs `domain` client-side for live feedback. |
| `server/` | Fastify API (catalog, sizing, reconciliation, saved configs, HF import) that also serves the built SPA. |

## Where changes go

- **Sizing math / a formula change** → `domain/src/engine.ts`. **You must** add or update an acceptance vector in `domain/src/__tests__/acceptance.test.ts` and keep it green. The engine is a pure function of `(inputs, model, gpu)` — no I/O, no globals.
- **A model or GPU in the default catalog** → `domain/src/seed.ts` (values are indicative; cite a source in the PR).
- **Validation rules** → `domain/src/schema.ts` (shared by client + server).
- **A new API endpoint** → `server/src/app.ts` (+ an integration test in `server/src/__tests__/api.test.ts`).
- **UI** → `web/src/App.svelte`.

## Guidelines

- **Keep the engine pure and tested.** Every sizing change ships with a test.
- **One source of truth.** Don't duplicate a formula or a validation rule between client and server — put it in `domain/`.
- **Estimates are estimates.** The model is a first-order roofline; if you tighten it, say what it now assumes and keep the ± caveats honest.
- Run `npm test` and `npm run build -w web` before opening a PR. CI runs both.

## Reporting issues

Please include: the model + GPU + inputs you used, what the calculator produced, and what you expected (with a source if it's an accuracy claim).

By contributing, you agree that your contributions are licensed under the project's [MIT License](LICENSE).
