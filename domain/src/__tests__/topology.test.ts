import { describe, it, expect } from 'vitest';
import { topologyLayout, topologySvg, escapeSvgText, truncLabels } from '../topology.js';
import { computeSizing } from '../engine.js';
import { seedCatalog } from '../seed.js';
import type { FeasibleSizing } from '../types.js';

const { models, gpus } = seedCatalog();
const M = (i: string) => models.find((x) => x.id === i)!;
const G = (i: string) => gpus.find((x) => x.id === i)!;
const size = (m: string, g: string, o: any) =>
  computeSizing(M(m), G(g), {
    kv_dtype_bytes: 1, avg_context_utilisation: 0.6, mem_util_fraction: 0.9, ...o,
  }) as FeasibleSizing;

describe('topology layout', () => {
  it('places every GPU of every node, marking the unused ones', () => {
    const r = size('llama33-70b', 'h200', { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64, gpus_per_node: 8 });
    const t = topologyLayout(r, 8);
    expect(t.gpus).toHaveLength(t.shown * 8);
    expect(t.gpus.filter((g) => g.used)).toHaveLength(r.gpus);
    expect(t.gpus.filter((g) => !g.used).every((g) => g.pod === null)).toBe(true);
  });

  it('assigns each replica a contiguous run of GPUs', () => {
    const r = size('llama33-70b', 'h200', { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64, gpus_per_node: 8 });
    const t = topologyLayout(r, 8);
    for (const pod of t.pods) {
      const mine = t.gpus.filter((g) => g.pod === pod.p).map((g) => g.g);
      expect(mine).toHaveLength(r.tp);
      expect(Math.max(...mine) - Math.min(...mine)).toBe(r.tp - 1); // no gaps
    }
  });

  it('flags a replica that straddles a node boundary — and only then', () => {
    const single = size('llama33-70b', 'h200', { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64, gpus_per_node: 8 });
    expect(single.multi_node).toBe(false);
    expect(topologyLayout(single, 8).pods.some((p) => p.spans)).toBe(false);

    const spanning = size('kimi-k3', 'h200', { quant: 'MXFP4', selected_ctx: 1048576, target_concurrency: 8, gpus_per_node: 8 });
    expect(spanning.multi_node).toBe(true);
    const t = topologyLayout(spanning, 8);
    expect(t.pods.some((p) => p.spans)).toBe(true);
    expect(t.multi).toBe(true); // a fabric is drawn
  });

  it('a narrow bar labels outside itself rather than overflowing', () => {
    const r = size('gptoss-120b', 'h200', { quant: 'MXFP4', selected_ctx: 131072, target_concurrency: 8, gpus_per_node: 8 });
    expect(r.tp).toBe(1);
    const t = topologyLayout(r, 8);
    expect(t.pods[0].x1 - t.pods[0].x0).toBe(t.cell); // one cell wide
    expect(t.pods[0].labelInside).toBe(false);
  });

  it('drops the GPU-count caption when node boxes are too narrow for it', () => {
    const r = size('llama33-70b', 'h100', { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64, gpus_per_node: 2 });
    expect(topologyLayout(r, 2).nodeLabelFull).toBe(false);
    expect(topologyLayout(r, 8).nodeLabelFull).toBe(true);
  });

  it('caps the node boxes drawn and says how many were left out', () => {
    const r = size('llama33-70b', 'h100', { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64, gpus_per_node: 2 });
    expect(r.nodes).toBeGreaterThan(4);
    const t = topologyLayout(r, 2);
    expect(t.shown).toBe(4);
    expect(t.truncated).toBe(true);
    expect(t.hiddenNodes).toBe(r.nodes - 4);
  });

  // The heading counts the whole deployment, so the drawing has to account for the replicas it
  // leaves out too — "8 replicas on 16 nodes" over a picture of two replicas reads as a bug.
  it('counts the replicas left out, not just the nodes', () => {
    const r = size('llama33-70b', 'h100', { quant: 'FP8', selected_ctx: 131072, target_concurrency: 512, gpus_per_node: 6 });
    expect(r.nodes).toBeGreaterThan(4);
    const t = topologyLayout(r, 6);
    expect(t.shownPods).toBe(t.pods.length);
    expect(t.shownPods).toBeLessThan(r.pods);
    expect(t.shownPods + t.hiddenPods).toBe(r.pods);
    expect(t.truncX).not.toBeNull();
    expect(t.truncX!).toBeGreaterThanOrEqual(t.contentW);
  });

  it('claims no hidden replicas when every replica is drawn', () => {
    const r = size('llama33-70b', 'h200', { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64, gpus_per_node: 8 });
    const t = topologyLayout(r, 8);
    expect(t.truncated).toBe(false);
    expect(t.hiddenPods).toBe(0);
    expect(t.hiddenNodes).toBe(0);
    expect(t.truncX).toBeNull();
    expect(t.pods.every((p) => p.gpusShown === r.tp)).toBe(true);
  });

  it('records how much of a replica the node cap actually drew', () => {
    // 3 GPUs/node with TP8: four node boxes hold 12 GPUs, so the second bar stops mid-replica.
    const r = size('llama33-70b', 'h100', { quant: 'FP8', selected_ctx: 131072, target_concurrency: 512, gpus_per_node: 3 });
    const t = topologyLayout(r, 3);
    expect(t.truncated).toBe(true);
    const last = t.pods[t.pods.length - 1];
    expect(last.gpusShown).toBeLessThan(r.tp);
    expect(t.pods.slice(0, -1).every((p) => p.gpusShown === r.tp)).toBe(true);
  });

  it('reserves viewBox width for the truncation marker', () => {
    const r = size('llama33-70b', 'h100', { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64, gpus_per_node: 2 });
    const t = topologyLayout(r, 2);
    const longest = Math.max(...truncLabels(t.hiddenNodes, t.hiddenPods).map((s) => s.length));
    expect(t.width - t.truncX!).toBeGreaterThan(longest * 6.2);
  });

  it('leaves slack past the drawing area so right-hand labels are not clipped', () => {
    const r = size('llama33-70b', 'h200', { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64, gpus_per_node: 8 });
    const t = topologyLayout(r, 8);
    expect(t.width).toBeGreaterThan(t.contentW);
    // every drawn element stays inside the viewBox
    for (const g of t.gpus) expect(g.x + t.cell).toBeLessThanOrEqual(t.width);
    for (const p of t.pods) expect(p.x1).toBeLessThanOrEqual(t.width);
  });

  it('stacks the fabric only when there is more than one node', () => {
    const one = topologyLayout(size('gptoss-120b', 'h200', { quant: 'MXFP4', selected_ctx: 131072, target_concurrency: 8, gpus_per_node: 8 }), 8);
    expect(one.multi).toBe(false);
    expect(one.storeY - one.switchY).toBeLessThan(56); // no switch band reserved
  });
});

describe('topology SVG rendering', () => {
  const render = (m: string, g: string, per: number, o: any, over: any = {}) => {
    const r = size(m, g, { gpus_per_node: per, ...o });
    const t = topologyLayout(r, per);
    return topologySvg(t, {
      tp: r.tp, perNode: per, multiNode: r.multi_node,
      storeLabel: `shared weights · ${r.weights_gb.toFixed(1)} GiB per replica`,
      desc: 'test', ...over,
    });
  };

  it('marks a node-spanning replica and heats the fabric — only when it spans', () => {
    const spanning = render('kimi-k3', 'h200', 8, { quant: 'MXFP4', selected_ctx: 1048576, target_concurrency: 8 });
    expect(spanning).toContain('tbox pod spanning');
    expect(spanning).toContain('tbox sw hot');
    expect(spanning).toContain('tlink fabric');
    expect(spanning).toContain('⚠');

    const contained = render('llama33-70b', 'h200', 8, { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64 });
    expect(contained).not.toContain('spanning');
    expect(contained).not.toContain('sw hot');
    expect(contained).not.toContain('⚠');
  });

  it('draws what it left out, naming replicas and nodes', () => {
    const truncated = render('llama33-70b', 'h100', 6, { quant: 'FP8', selected_ctx: 131072, target_concurrency: 512 });
    expect(truncated).toContain('more replicas');
    expect(truncated).toContain('more nodes');
    expect(truncated).toContain('⋯');

    const whole = render('llama33-70b', 'h200', 8, { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64 });
    expect(whole).not.toContain('more replica');
    expect(whole).not.toContain('⋯');
  });

  it('leaves a half-drawn replica bar open on the right instead of closing it', () => {
    const cut = render('llama33-70b', 'h100', 3, { quant: 'FP8', selected_ctx: 131072, target_concurrency: 512 });
    expect(cut).toMatch(/class="tbox pod(?: spanning)? cut"/);
    expect(cut).toMatch(/<path d="M [\d.]+ 48 H/); // open-ended, not a closed rect
    const whole = render('llama33-70b', 'h200', 8, { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64 });
    expect(whole).not.toContain('cut');
    expect(whole).not.toContain('<path');
  });

  it('omits the fabric entirely on a single node', () => {
    const one = render('gptoss-120b', 'h200', 8, { quant: 'MXFP4', selected_ctx: 131072, target_concurrency: 8 });
    expect(one).not.toContain('InfiniBand');
    expect(one).toContain('shared weights');
  });

  it('escapes text — catalogue labels are admin-editable', () => {
    expect(escapeSvgText('a & b')).toBe('a &amp; b');
    expect(escapeSvgText('<script>x</script>')).toBe('&lt;script&gt;x&lt;/script&gt;');
    const svg = render('llama33-70b', 'h200', 8, { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64 },
      { storeLabel: '<script>alert(1)</script>', desc: 'a & b' });
    expect(svg).not.toContain('<script>');
    expect(svg).toContain('&lt;script&gt;');
    expect(svg).toContain('a &amp; b');
  });

  it('carries an accessible name and a prose description', () => {
    const svg = render('llama33-70b', 'h200', 8, { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64 });
    expect(svg).toContain('role="img"');
    expect(svg).toContain('<title');
    expect(svg).toContain('<desc');
    expect(svg).toContain('aria-labelledby');
  });

  it('is well-formed and sized to the layout', () => {
    const r = size('llama33-70b', 'h200', { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64, gpus_per_node: 8 });
    const t = topologyLayout(r, 8);
    const svg = topologySvg(t, { tp: r.tp, perNode: 8, multiNode: r.multi_node, storeLabel: 'x', desc: 'y' });
    expect(svg.startsWith('<svg')).toBe(true);
    expect(svg.endsWith('</svg>')).toBe(true);
    expect(svg).toContain(`viewBox="0 0 ${t.width} ${t.height}"`);
    expect((svg.match(/<rect/g) ?? []).length).toBeGreaterThanOrEqual(t.gpus.length);
  });
});
