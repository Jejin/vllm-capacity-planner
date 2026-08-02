#!/usr/bin/env node
// The methodology exists twice — as docs/METHODOLOGY.md and as the in-app Methodology tab —
// and the two drifted apart once already: a section that was §7 in the app was a §3 subsection
// in the doc, "The GGUF trap" was a heading in one and an inline note in the other, and a
// cross-reference pointed at a section that had moved. Neither file shows that on its own.
//
// This asserts they carry the same headings in the same order. It does not compare prose —
// only that every section documented in one place exists in the other, which is the failure
// mode that actually occurred.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const DOC = 'docs/METHODOLOGY.md';
const APP = 'web/src/App.svelte';

// Headings that are intentionally unique to one surface: the doc closes with caveats, the app
// closes with an FAQ. Anything else appearing on one side only is drift.
const ALLOWED_DOC_ONLY = new Set(['notes']);
const ALLOWED_APP_ONLY = new Set(['frequently asked']);

const decode = (s) =>
  s.replace(/&amp;/g, '&').replace(/&nbsp;/g, ' ').replace(/&lt;/g, '<').replace(/&gt;/g, '>');

/** Strip section numbering and separator style so "6. Fits / tight" == "6 · Fits · tight". */
function normalise(raw) {
  return decode(raw)
    .replace(/^\s*\d+\s*[.·]\s*/, '') // leading "7. " or "7 · "
    .toLowerCase()
    .replace(/[·/]/g, '|') // the two separator styles in use
    .replace(/\s+/g, ' ')
    .trim();
}

function docHeadings(src) {
  const out = [];
  for (const line of src.split('\n')) {
    const m = /^(#{2,3})\s+(.*\S)\s*$/.exec(line);
    if (m) out.push({ level: m[1].length, text: m[2], norm: normalise(m[2]) });
  }
  return out;
}

function appHeadings(src) {
  const out = [];
  const re = /class="(dh|dh3)"\s*>([^<]+)</g;
  let m;
  while ((m = re.exec(src)) !== null) {
    const text = decode(m[2]).trim();
    out.push({ level: m[1] === 'dh' ? 2 : 3, text, norm: normalise(text) });
  }
  return out;
}

/** Top-level sections must be numbered 1..n with no gaps or repeats, in document order. */
function checkNumbering(headings, label, errors) {
  const nums = [];
  for (const h of headings) {
    if (h.level !== 2) continue;
    const m = /^\s*(\d+)\s*[.·]/.exec(decode(h.text));
    if (m) nums.push({ n: Number(m[1]), text: h.text });
  }
  nums.forEach((cur, i) => {
    if (cur.n !== i + 1) {
      errors.push(`${label}: section "${cur.text}" is numbered ${cur.n} but appears in position ${i + 1}`);
    }
  });
}

/**
 * Values the prose quotes that the ENGINE owns. Headings matching is not enough: every stale
 * block this check ever missed sat under a heading that matched perfectly. If a constant moves
 * in the engine and the prose still quotes the old number, that is drift the reader cannot see.
 *
 * Each entry finds the constant's NAME in the prose and requires its current value to appear
 * near it. Deliberately fuzzy about phrasing — the two surfaces word things differently — and
 * strict about the number.
 */
/**
 * Read the constants out of the engine SOURCE, not its build.
 *
 * The first version imported `domain/dist`, passed locally against a stale build, and failed in
 * CI where nothing had run `tsc` — vitest reads `src`. A docs lint that only works after a
 * build is a docs lint that runs in the wrong places, so this parses the declarations directly
 * and has no ordering dependency at all.
 */
function engineConstants() {
  const src = readFileSync(join(root, 'domain/src/engine.ts'), 'utf8');
  const out = {};
  const re = /^export const ([A-Z][A-Z0-9_]*)\s*(?::\s*[^=]+)?=\s*([\d.]+);/gm;
  let m;
  while ((m = re.exec(src)) !== null) out[m[1]] = Number(m[2]);
  return out;
}
const K = engineConstants();
for (const need of ['RUNTIME_GB', 'CUDA_CONTEXT_GB', 'MBU', 'PREFILL_MFU', 'TIGHT_HEADROOM', 'DEFAULT_MLA_LATENT_ELEMS', 'DEFAULT_BATCHED_TOKENS', 'ALL_REDUCES_PER_LAYER']) {
  if (K[need] === undefined) {
    console.error(`methodology-parity: could not read ${need} from domain/src/engine.ts — did the declaration style change?`);
    process.exit(1);
  }
}

const CONSTANTS = [
  { name: /runtime reserve/gi, value: String(K.RUNTIME_GB), label: 'RUNTIME_GB' },
  { name: /CUDA context/gi, value: String(K.CUDA_CONTEXT_GB), label: 'CUDA_CONTEXT_GB' },
  { name: /\bMBU\b/g, value: String(K.MBU), label: 'MBU' },
  { name: /prefill MFU/gi, value: String(K.PREFILL_MFU), label: 'PREFILL_MFU' },
  { name: /tight-fit threshold/gi, value: `${K.TIGHT_HEADROOM * 100}%`, label: 'TIGHT_HEADROOM' },
  { name: /512 \+ 64/g, value: String(K.DEFAULT_MLA_LATENT_ELEMS), label: 'DEFAULT_MLA_LATENT_ELEMS' },
  { name: /vLLM's own default chunk|vLLM's default chunk/gi, value: String(K.DEFAULT_BATCHED_TOKENS), label: 'DEFAULT_BATCHED_TOKENS' },
  { name: /all-reduces? (?:on|per|twice per) (?:every |each )?layer/gi, value: String(K.ALL_REDUCES_PER_LAYER), label: 'ALL_REDUCES_PER_LAYER', valueAliases: ['two', 'twice'] },
];
const WINDOW = 220;

/**
 * Claims that were true once and are not any more.
 *
 * The MoE contradiction is why this exists: "for MoE models only the active parameters are
 * read" sat directly above the section explaining that a decode step reads the whole expert
 * union, and every heading-based check passed. A retired claim is cheap to record at the moment
 * it stops being true, and impossible to notice six sections away.
 */
const RETIRED = [
  { re: /only the (?:<em>)?active(?:<\/em>)? parameters are read/i, why: 'MoE decode reads the expert union at batch > 1 (§4)' },
  { re: /falls back to the catalogue (?:model )?name/i, why: 'the launch command refuses to serve a display name (§11)' },
  { re: /MLA latent \*?\*?576/i, why: 'MLA latent width is per-model geometry, not a fixed constant (§3)' },
  { re: /TTFT (?:are|is) unaffected/i, why: 'prefill all-reduces too, so TTFT is withheld across nodes with throughput (§3)' },
  { re: /the throughput and TTFT figures here are optimistic/i, why: 'those figures are withheld on a fallback kernel, not shown as optimistic (§6)' },
];

function checkProse(src, label, errors) {
  for (const c of CONSTANTS) {
    c.name.lastIndex = 0;
    const windows = [];
    let m;
    while ((m = c.name.exec(src)) !== null) {
      windows.push(src.slice(Math.max(0, m.index - WINDOW), m.index + WINDOW));
    }
    if (windows.length === 0) continue; // the prose does not mention it; not this check's business
    const wanted = [c.value, ...(c.valueAliases ?? [])];
    if (!windows.some((w) => wanted.some((v) => w.includes(v)))) {
      errors.push(
        `${label}: mentions ${c.label} but never its current value "${c.value}" nearby — ` +
        'the engine moved and the prose did not',
      );
    }
  }
  for (const r of RETIRED) {
    if (r.re.test(src)) errors.push(`${label}: retired claim still present — ${r.why}`);
  }
}

const doc = docHeadings(readFileSync(join(root, DOC), 'utf8'));
const app = appHeadings(readFileSync(join(root, APP), 'utf8'));
const errors = [];

if (doc.length === 0) errors.push(`${DOC}: no headings found — did the format change?`);
if (app.length === 0) errors.push(`${APP}: no .dh/.dh3 headings found — did the class names change?`);

const docSet = new Set(doc.map((h) => h.norm));
const appSet = new Set(app.map((h) => h.norm));

for (const h of doc) {
  if (!appSet.has(h.norm) && !ALLOWED_DOC_ONLY.has(h.norm)) {
    errors.push(`in ${DOC} but not the app tab: "${h.text}"`);
  }
}
for (const h of app) {
  if (!docSet.has(h.norm) && !ALLOWED_APP_ONLY.has(h.norm)) {
    errors.push(`in the app tab but not ${DOC}: "${h.text}"`);
  }
}

// Shared headings must also appear in the same order, so a section cannot silently migrate
// between parents (which is exactly what happened with local/global attention).
const shared = (list) => list.map((h) => h.norm).filter((n) => docSet.has(n) && appSet.has(n));
const docOrder = shared(doc);
const appOrder = shared(app);
for (let i = 0; i < Math.min(docOrder.length, appOrder.length); i++) {
  if (docOrder[i] !== appOrder[i]) {
    errors.push(`order diverges at position ${i + 1}: ${DOC} has "${docOrder[i]}", app has "${appOrder[i]}"`);
    break;
  }
}

checkNumbering(doc, DOC, errors);
checkNumbering(app, APP, errors);

checkProse(readFileSync(join(root, DOC), 'utf8'), DOC, errors);
// Only the Methodology tab's markup — a `<script>` comment mentioning a constant is not a
// claim to the reader, and the first run of this check tripped on exactly that.
const appSrc = readFileSync(join(root, APP), 'utf8');
const tabStart = appSrc.indexOf("tab === 'methodology'");
if (tabStart < 0) {
  errors.push(`${APP}: could not locate the methodology tab — did the markup change?`);
} else {
  checkProse(appSrc.slice(tabStart), `${APP} (methodology tab)`, errors);
}

if (errors.length) {
  console.error('Methodology parity check FAILED:\n');
  for (const e of errors) console.error(`  • ${e}`);
  console.error(`\n${DOC} and the in-app Methodology tab must document the same sections in the same order,`);
  console.error('and must not quote constants the engine has moved or claims it has retired.');
  process.exit(1);
}

console.log(
  `Methodology parity OK — ${doc.length} headings in ${DOC}, ${app.length} in the app tab, same order; ` +
  `${CONSTANTS.length} engine constants and ${RETIRED.length} retired claims checked in both.`,
);
