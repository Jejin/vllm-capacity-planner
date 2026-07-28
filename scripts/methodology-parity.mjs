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

if (errors.length) {
  console.error('Methodology parity check FAILED:\n');
  for (const e of errors) console.error(`  • ${e}`);
  console.error(`\n${DOC} and the in-app Methodology tab must document the same sections in the same order.`);
  process.exit(1);
}

console.log(`Methodology parity OK — ${doc.length} headings in ${DOC}, ${app.length} in the app tab, same order.`);
