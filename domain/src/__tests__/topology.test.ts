import { describe, it, expect } from 'vitest';
import { topologyLayout, topologySvg, escapeSvgText, truncLabels, podLabel } from '../topology.js';
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

  // The layout reads left to right and stacks the nodes, so a wide plan grows down the page
  // rather than off the side of the panel.
  it('runs router → replicas → nodes across, and stacks the nodes down', () => {
    const r = size('kimi-k3', 'h200', { quant: 'MXFP4', selected_ctx: 1048576, target_concurrency: 8, gpus_per_node: 8 });
    const t = topologyLayout(r, 8);
    expect(t.routerW).toBeLessThanOrEqual(t.podX);
    expect(t.podX + t.podW).toBeLessThanOrEqual(t.nodeX);
    expect(t.nodes.length).toBeGreaterThan(1);
    for (let i = 1; i < t.nodes.length; i++) {
      expect(t.nodes[i].y).toBeGreaterThan(t.nodes[i - 1].y); // stacked, not side by side
    }
    // every node's GPUs share one x range — the columns line up down the stack
    const xs = (n: number) => t.gpus.filter((g) => g.n === n).map((g) => g.x).join(',');
    expect(xs(1)).toBe(xs(0));
  });

  it('keeps a wide plan inside a panel that used to need a scrollbar', () => {
    const r = size('glm52', 'h200', { quant: 'FP8', selected_ctx: 1048576, target_concurrency: 256, gpus_per_node: 8, avg_context_utilisation: 0.8 });
    expect(r.tp).toBe(16);
    expect(r.nodes).toBe(16);
    // the sizing panel is ~720px wide at main's 1120px cap
    expect(topologyLayout(r, 8).width).toBeLessThan(700);
  });

  it('spans a replica bar across the node rows it lands on', () => {
    const r = size('kimi-k3', 'h200', { quant: 'MXFP4', selected_ctx: 1048576, target_concurrency: 8, gpus_per_node: 8 });
    expect(r.multi_node).toBe(true);
    const t = topologyLayout(r, 8);
    const spanning = t.pods.filter((p) => p.spans);
    expect(spanning.length).toBeGreaterThan(0);
    for (const pod of spanning) {
      expect(pod.segs.length).toBeGreaterThan(1);
      // the bar covers the gap between the two node boxes it straddles
      expect(pod.y0).toBeLessThan(t.nodes[1].y);
      expect(pod.y1).toBeGreaterThan(t.nodes[1].y);
    }
  });

  it('gives each replica in a shared node its own slot, without overlap', () => {
    const r = size('llama33-70b', 'h200', { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64, gpus_per_node: 8 });
    expect(r.tp).toBeLessThan(8); // several replicas per node
    const t = topologyLayout(r, 8);
    expect(t.pods.length).toBeGreaterThan(1);
    const sorted = [...t.pods].sort((a, b) => a.y0 - b.y0);
    for (let i = 1; i < sorted.length; i++) {
      expect(sorted[i].y0).toBeGreaterThanOrEqual(sorted[i - 1].y1);
    }
    expect(t.pods.every((p) => p.labelInside)).toBe(true);
  });

  it('trades node boxes for height when a node holds many replicas', () => {
    const many = size('gptoss-120b', 'h200', { quant: 'MXFP4', selected_ctx: 131072, target_concurrency: 256, gpus_per_node: 8 });
    expect(many.tp).toBeLessThanOrEqual(2); // four or more replicas share a node
    const t = topologyLayout(many, 8);
    expect(t.bandH).toBeGreaterThan(t.cell); // the band grew to hold the slots
    expect(t.stackH).toBeLessThanOrEqual(420); // ...and the stack did not run away
  });

  it('sizes the replica column to the labels it will draw', () => {
    const r = size('kimi-k3', 'h200', { quant: 'MXFP4', selected_ctx: 1048576, target_concurrency: 8, gpus_per_node: 8 });
    const t = topologyLayout(r, 8);
    for (const pod of t.pods) {
      expect(t.podW).toBeGreaterThan(podLabel(pod.p, r.tp, pod.spans).length * 6.2);
    }
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
    expect(t.truncY).not.toBeNull();
    expect(t.truncY!).toBeGreaterThanOrEqual(t.stackH);
    expect(t.height).toBeGreaterThan(t.truncY!); // the marker is inside the viewBox
  });

  it('claims no hidden replicas when every replica is drawn', () => {
    const r = size('llama33-70b', 'h200', { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64, gpus_per_node: 8 });
    const t = topologyLayout(r, 8);
    expect(t.truncated).toBe(false);
    expect(t.hiddenPods).toBe(0);
    expect(t.hiddenNodes).toBe(0);
    expect(t.truncY).toBeNull();
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

  it('leaves slack past the drawing so right-hand labels are not clipped', () => {
    const r = size('llama33-70b', 'h200', { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64, gpus_per_node: 8 });
    const t = topologyLayout(r, 8);
    expect(t.width).toBeGreaterThan(t.contentW);
    // every drawn element stays inside the viewBox
    for (const g of t.gpus) {
      expect(g.x + t.cell).toBeLessThanOrEqual(t.width);
      expect(g.y + t.cell).toBeLessThanOrEqual(t.height);
    }
    for (const p of t.pods) expect(p.y1).toBeLessThanOrEqual(t.height);
    expect(t.storeY + 26).toBeLessThanOrEqual(t.height);
  });

  it('stacks the fabric only when there is more than one node', () => {
    const one = topologyLayout(size('gptoss-120b', 'h200', { quant: 'MXFP4', selected_ctx: 131072, target_concurrency: 8, gpus_per_node: 8 }), 8);
    expect(one.multi).toBe(false);
    expect(one.spineX).toBeNull();

    const two = topologyLayout(size('kimi-k3', 'h200', { quant: 'MXFP4', selected_ctx: 1048576, target_concurrency: 8, gpus_per_node: 8 }), 8);
    expect(two.multi).toBe(true);
    expect(two.spineX!).toBeGreaterThan(two.nodeX + two.nodeW); // clear of the node boxes
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

  it('rules the cells of each replica so a stacked node still shows the grouping', () => {
    const svg = render('llama33-70b', 'h200', 8, { quant: 'FP8', selected_ctx: 131072, target_concurrency: 64 });
    expect((svg.match(/class="tunder/g) ?? []).length).toBeGreaterThanOrEqual(2); // one per replica
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

  it('leaves a half-drawn replica bar open at the bottom instead of closing it', () => {
    const cut = render('llama33-70b', 'h100', 3, { quant: 'FP8', selected_ctx: 131072, target_concurrency: 512 });
    expect(cut).toMatch(/class="tbox pod(?: spanning)? cut"/);
    expect(cut).toMatch(/<path d="M [\d.]+ [\d.]+ V/); // open-ended, not a closed rect
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

  it('keeps the widest storage caption inside the viewBox', () => {
    // four-digit weights on a two-node plan — the caption is wider than the node column
    const r = size('kimi-k3', 'h200', { quant: 'MXFP4', selected_ctx: 1048576, target_concurrency: 8, gpus_per_node: 8 });
    const t = topologyLayout(r, 8);
    const label = `shared weights · ${r.weights_gb.toFixed(1)} GiB per replica`;
    const svg = topologySvg(t, { tp: r.tp, perNode: 8, multiNode: true, storeLabel: label, desc: 'x' });
    const store = svg.match(/x="([\d.]+)" y="[\d.]+" width="([\d.]+)"[^>]*class="tbox store"/)!;
    expect(Number(store[1])).toBeGreaterThanOrEqual(0);
    expect(Number(store[1]) + Number(store[2])).toBeLessThanOrEqual(t.width);
  });

  it('composes the labels the layout measured', () => {
    expect(podLabel(0, 16, false)).toBe('replica 1 · TP16');
    expect(podLabel(2, 8, true)).toBe('⚠ replica 3 · TP8');
    expect(truncLabels(3, 1)).toEqual(['+1 more replica', 'on +3 more nodes']);
    expect(truncLabels(0, 0)).toEqual(['…']);
  });
});
