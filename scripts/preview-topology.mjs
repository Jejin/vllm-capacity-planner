#!/usr/bin/env node
// Render the deployment topology diagram for a spread of scenarios, headlessly, so its layout
// can be eyeballed without starting a dev server and clicking through the app.
//
// Three layout bugs shipped past a clean build, a clean type-check and a full test suite — an
// off-canvas label on a one-cell replica bar, a clipped node caption on narrow nodes, and a
// storage box overflowed by a four-digit weight figure. None were detectable except by looking.
//
// Nothing here re-implements the diagram: the SVG comes from the same `topologySvg` the app
// renders, and the CSS is read out of App.svelte. A preview that duplicates either stops being
// evidence about the real component the moment one of them changes.

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { topologyLayout, topologySvg, computeSizing, seedCatalog } from '../domain/dist/index.js';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const outDir = join(root, '.preview');
const { models, gpus } = seedCatalog();
const M = (id) => models.find((m) => m.id === id);
const G = (id) => gpus.find((g) => g.id === id);

// Chosen to cover the layout edges, not to look impressive: the one-cell bar, the narrow node,
// the node-spanning replica, the four-digit storage caption, and the truncation path.
const SCENARIOS = [
  ['TP1 — one cell, label must sit outside the bar', 'gptoss-120b', 'h200', 8,
    { quant: 'MXFP4', selected_ctx: 131072, target_concurrency: 8 }],
  ['Two replicas inside one node', 'llama33-70b', 'h200', 8,
    { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64 }],
  ['SPANS NODES — TP16 on 8-GPU nodes, four-digit weights', 'kimi-k3', 'h200', 8,
    { quant: 'MXFP4', selected_ctx: 1048576, target_concurrency: 8 }],
  ['Narrow nodes — 2 GPU/node, more nodes than are drawn', 'llama33-70b', 'h100', 2,
    { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64 }],
  ['Wide replica — TP8 across a full node', 'glm45', 'h200', 8,
    { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64 }],
];

/** Pull the theme variables and topology rules straight out of the component. */
function styleFromApp() {
  const app = readFileSync(join(root, 'web/src/App.svelte'), 'utf8');
  // `:global(...)` is Svelte syntax and invalid in a plain stylesheet — left in, the browser
  // drops the whole rule, every var() resolves to nothing and the diagram renders solid black.
  const vars = [...app.matchAll(/:global\((:root(?:\[data-theme=dark\])?)\)(\{[^}]*\})/g)]
    .map((m) => m[1] + m[2]);
  const topo = [...app.matchAll(/^\s*\.topowrap[^\n]*$/gm)]
    .map((m) => m[0].trim().replace(/\.topowrap :global\(([^)]*)\)/g, '$1'));
  if (!vars.length || !topo.length) {
    throw new Error('could not read theme vars / .topowrap rules out of App.svelte — did the CSS move?');
  }
  return [...vars, ...topo].join('\n');
}

function page(sections, theme) {
  // the theme attribute goes on <html>, matching how the app toggles it — the override
  // selector is :root[data-theme=dark], so putting it on <body> silently does nothing
  return `<!doctype html><html${theme === 'dark' ? ' data-theme="dark"' : ''}><head><meta charset="utf-8">
<style>
${styleFromApp()}
body{background:var(--bg);color:var(--ink);font-family:Manrope,system-ui,sans-serif;margin:0;padding:20px}
h2{font-size:12px;color:var(--ink2);margin:20px 0 6px;font-weight:700}
h2 small{color:var(--ink3);font-weight:600}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px}
.bad{color:var(--err);font-weight:700}
</style></head><body>${sections}</body></html>`;
}

function render() {
  let out = '';
  for (const [label, mid, gid, perNode, input] of SCENARIOS) {
    const m = M(mid);
    const r = computeSizing(m, G(gid), {
      kv_dtype_bytes: 1, avg_context_utilisation: 0.6, mem_util_fraction: 0.9,
      gpus_per_node: perNode, ...input,
    });
    if (!r.ok) {
      out += `<h2>${label}</h2><div class="panel"><span class="bad">infeasible — ${r.reason}</span></div>`;
      continue;
    }
    const t = topologyLayout(r, perNode);
    const svg = topologySvg(t, {
      tp: r.tp,
      perNode,
      multiNode: r.multi_node,
      storeLabel: `shared weights · ${r.weights_gb.toFixed(1)} GiB per replica`,
      desc: `${r.pods} replicas of TP${r.tp} on ${r.nodes} nodes.`,
      titleId: `t${mid}${perNode}`,
      descId: `d${mid}${perNode}`,
    });
    out += `<h2>${label}<br><small>${mid} · TP${r.tp} · ${r.pods} replicas · ${r.gpus} GPUs · ${r.nodes} nodes${t.truncated ? ` (drawing ${t.shown})` : ''}</small></h2>` +
      `<div class="panel"><div class="topowrap">${svg}</div></div>`;
  }
  return out;
}

if (!existsSync(outDir)) mkdirSync(outDir);
const sections = render();
for (const theme of ['light', 'dark']) {
  writeFileSync(join(outDir, `topology-${theme}.html`), page(sections, theme));
}
console.log(`Wrote ${SCENARIOS.length} scenarios → .preview/topology-{light,dark}.html`);

// Screenshot when a headless browser is available; the HTML alone is still useful without one.
const chrome = [
  process.env.CHROME_PATH,
  join(process.env.HOME ?? '', '.cache/ms-playwright/chromium-1228/chrome-linux/chrome'),
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
].find((p) => p && existsSync(p));

if (!chrome) {
  console.log('No headless Chrome found — open the HTML directly, or set CHROME_PATH.');
  process.exit(0);
}
for (const theme of ['light', 'dark']) {
  execFileSync(chrome, [
    '--headless', '--no-sandbox', '--disable-gpu', '--hide-scrollbars',
    '--window-size=1200,1700',
    `--screenshot=${join(outDir, `topology-${theme}.png`)}`,
    join(outDir, `topology-${theme}.html`),
  ], { stdio: 'ignore' });
}
console.log('Screenshots → .preview/topology-{light,dark}.png');
