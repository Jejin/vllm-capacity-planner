// The rendered page must agree with the engine that produced it.
//
// Two labels once read "KV per request (128K x 60%)" and "Prefill work (78,643 tokens)" while
// displaying P95 and P50 values — the arithmetic was right, the sentence beside it was wrong,
// and 209 domain tests had nothing to say about it because none of them look at the page.
//
// These mount the real component and compare what it SHOWS against what `computeSizing` returns
// for the same inputs. Text, not pixels: no fonts, no baselines, no browser.

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import { computeSizing, seedCatalog } from '@vcp/domain';
import type { FeasibleSizing } from '@vcp/domain';
import App from '../App.svelte';

const { models, gpus } = seedCatalog();
let host: HTMLElement;
let app: Record<string, unknown>;

/** Drive a bound control the way a user would, so Svelte's bindings actually fire. */
function set(el: Element, value: string) {
  const proto = el instanceof HTMLSelectElement ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(proto, 'value')!.set!.call(el, value);
  el.dispatchEvent(new Event('input', { bubbles: true }));
  el.dispatchEvent(new Event('change', { bubbles: true }));
  flushSync();
}

const labelled = (text: string): HTMLElement | undefined =>
  [...host.querySelectorAll('label')].find((l) => l.textContent?.trim().startsWith(text)) ?? undefined;
const control = (text: string): Element => {
  const el = labelled(text)?.querySelector('input, select');
  if (!el) throw new Error(`no control labelled "${text}"`);
  return el;
};
/** The breakdown row whose description starts with `text`, as one flat string. */
const row = (text: string): string => {
  const el = [...host.querySelectorAll('.li')].find((d) => d.querySelector('span')?.textContent?.trim().startsWith(text));
  if (!el) throw new Error(`no row "${text}"`);
  return el.textContent!.replace(/\s+/g, ' ').trim();
};
const kpi = (label: string): string => {
  const el = [...host.querySelectorAll('.kpi')].find((d) => d.querySelector('.l')?.textContent?.trim().toLowerCase() === label.toLowerCase());
  if (!el) throw new Error(`no KPI "${label}"`);
  return el.textContent!.replace(/\s+/g, ' ').trim();
};

beforeEach(() => {
  // No server in a unit test. A 404 rather than a throw, because that is what production
  // actually does — only the SPA is deployed, so every /api route 404s there too.
  vi.stubGlobal('fetch', vi.fn(async () => new Response(null, { status: 404 })));
  host = document.createElement('div');
  document.body.appendChild(host);
  app = mount(App, { target: host });
  flushSync();
});
afterEach(() => {
  unmount(app);
  host.remove();
  vi.unstubAllGlobals();
});

describe('the page agrees with the engine', () => {
  it('renders a plan for the default inputs', () => {
    expect(host.textContent).toContain('Deployment inputs');
    expect(kpi('GPUs')).toMatch(/\d/);
  });

  it('a workload changes both the value AND the label that describes it', () => {
    // the exact regression: the label kept describing context utilisation
    expect(row('KV per request')).toContain('×'); // "128K × 60%"
    for (const [lbl, v] of [['Prompt P50', '2000'], ['Prompt P95', '60000'], ['Output P50', '400'], ['Output P95', '4000']] as const) {
      set(control(lbl), v);
    }
    const kv = row('KV per request');
    expect(kv).toContain('P95');
    expect(kv).toContain('64,000'); // 60k + 4k, not 128K × 60%
    expect(kv).not.toContain('60%');

    const prefill = row('Prefill work');
    expect(prefill).toContain('2,000'); // P50, not the 78,643 the flat path uses
    expect(prefill).toContain('P50');
  });

  it('the KV figure it prints is the one the engine computed', () => {
    const m = models.find((x) => x.id === 'llama33-70b')!;
    const g = gpus.find((x) => x.id === 'h200')!;
    set(control('Model'), m.id);
    set(control('GPU SKU'), g.id);
    const r = computeSizing(m, g, {
      quant: 'FP8', kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6,
      target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8, max_num_batched_tokens: 8192,
    }) as FeasibleSizing;
    expect(r.ok).toBe(true);
    expect(row('KV per request')).toContain(r.kv_per_request_gb.toFixed(1));
    expect(kpi('Pods')).toContain(String(r.pods));
    expect(kpi('GPUs')).toContain(String(r.gpus));
  });

  it('a withheld figure prints no number anywhere it appears', () => {
    // DeepSeek-V3 on H100s selects TP16, which spans nodes with no fabric declared
    set(control('Model'), 'dsv3');
    set(control('GPU SKU'), 'h100');
    const shown = host.textContent!;
    expect(shown).toContain('Throughput and TTFT withheld');

    // the tiles must show an em dash rather than the number the engine still carries
    expect(kpi('Tokens/s')).toContain('—');
    expect(kpi('Tokens/s')).not.toMatch(/\d,\d{3}/);
    expect(kpi('Time to first token')).toContain('—');

    // and the figures derived from it must not reintroduce it
    expect(row('Aggregate throughput')).toContain('—');
    expect(row('Cost per million tokens')).toContain('—');
    expect(row('Decode throughput / request')).toContain('—');
  });

  it('a non-native kernel withholds the same figures for a different reason', () => {
    set(control('Model'), 'gptoss-120b');
    set(control('GPU SKU'), 'a100s'); // MXFP4 has no native path before Hopper
    expect(host.textContent).toContain('no native kernel');
    expect(kpi('Tokens/s')).toContain('—');
    // memory sizing is explicitly NOT withheld — that separation is the whole design
    expect(kpi('KV cache')).toMatch(/\d/);
    expect(kpi('GPUs')).toMatch(/\d/);
  });

  it('names the assumptions behind a figure it does show', () => {
    set(control('Model'), 'dsv3');
    set(control('GPU SKU'), 'b300'); // native FP8, single node — a figure is given
    expect(kpi('Tokens/s')).not.toContain('—');
    const shown = host.textContent!;
    expect(shown).toContain('What these figures rest on');
    expect(shown).toContain('uniform expert routing'); // MoE-specific, per-plan
    expect(shown).toContain('collective latency unmodelled');
  });
});
