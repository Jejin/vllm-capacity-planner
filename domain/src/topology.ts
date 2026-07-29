// Geometry for the deployment topology diagram.
//
// Pure layout: given a sizing and the node width, where does every box go. Kept out of the
// component so it can be unit-tested and rendered headlessly — a diagram whose geometry only
// exists inside a template can only be checked by looking at it, and only in a browser.
//
// The organising decision is that every GPU sits on ONE horizontal axis, grouped into node
// boxes. That is what makes a cross-node tensor-parallel group legible: its bar visibly spans
// the gap between two node boxes, which is exactly where the fabric is drawn.

import type { FeasibleSizing } from './types.js';

export interface TopologyOptions {
  /** Cap on node boxes drawn; beyond this the pattern just repeats. */
  maxNodes?: number;
  cell?: number;
  cellGap?: number;
  nodePad?: number;
  nodeGap?: number;
  labelH?: number;
  padRight?: number;
  /** Bars narrower than this label to the right instead of centring. */
  minPodLabelW?: number;
  /** Node boxes narrower than this use the short caption. */
  minNodeLabelW?: number;
}

export interface TopoGpu {
  /** Global GPU index across the whole deployment. */
  g: number;
  /** Node this GPU belongs to. */
  n: number;
  x: number;
  /** False for GPUs present in the node but not claimed by this plan. */
  used: boolean;
  /** Replica index this GPU serves, or null when idle. */
  pod: number | null;
}

export interface TopoPod {
  p: number;
  x0: number;
  x1: number;
  /** True when this replica's TP group straddles a node boundary. */
  spans: boolean;
  /**
   * Whether the bar is wide enough to hold its own label. A TP1 replica is one 26px cell —
   * centring text on it pushes half the string off-canvas, so narrow bars label to the right.
   */
  labelInside: boolean;
  /**
   * GPUs of this replica that fall inside the drawn nodes. Below `tp` when the node cap cuts
   * a replica in half, and the bar must then say so rather than pass for a whole TP group.
   */
  gpusShown: number;
}

export interface TopologyLayout {
  shown: number;
  perNode: number;
  cell: number;
  nodePad: number;
  nodeW: number;
  nodeGap: number;
  labelH: number;
  nodeH: number;
  width: number;
  /** Drawing width excluding the right-hand text slack — the centre line for stacked boxes. */
  contentW: number;
  height: number;
  nodeY: number;
  switchY: number;
  storeY: number;
  /** More than one node box is drawn, so an inter-node fabric exists in the picture. */
  multi: boolean;
  /** Node boxes are wide enough for the full "node N · G GPU" caption rather than "node N". */
  nodeLabelFull: boolean;
  gpus: TopoGpu[];
  pods: TopoPod[];
  truncated: boolean;
  hiddenNodes: number;
  /** Replicas with a bar in the drawing — `pods.length`, named for the render side. */
  shownPods: number;
  /** Replicas the node cap left out entirely. */
  hiddenPods: number;
  /** x of the "more of the same" marker, or null when nothing is left out. */
  truncX: number | null;
}

export function topologyLayout(
  sizing: FeasibleSizing,
  gpusPerNode: number,
  opts: TopologyOptions = {},
): TopologyLayout {
  const {
    maxNodes = 4, cell = 26, cellGap = 5, nodePad = 12, nodeGap = 44, labelH = 20,
    // Text overhangs its box; without slack the right-most label is clipped by the viewBox.
    padRight = 24, minPodLabelW = 96, minNodeLabelW = 118,
  } = opts;
  const perNode = Math.max(1, Math.floor(gpusPerNode));
  const shown = Math.max(1, Math.min(sizing.nodes, maxNodes));
  const nodeW = perNode * cell + (perNode - 1) * cellGap + nodePad * 2;
  const contentW = shown * nodeW + (shown - 1) * nodeGap;
  const nodeY = 92;
  const nodeH = labelH + cell + nodePad;

  const gpus: TopoGpu[] = [];
  for (let n = 0; n < shown; n++) {
    for (let j = 0; j < perNode; j++) {
      const g = n * perNode + j;
      const used = g < sizing.gpus;
      gpus.push({
        g,
        n,
        used,
        x: n * (nodeW + nodeGap) + nodePad + j * (cell + cellGap),
        pod: used ? Math.floor(g / sizing.tp) : null,
      });
    }
  }

  const pods: TopoPod[] = [];
  const visibleGpus = Math.min(sizing.gpus, shown * perNode);
  for (let p = 0; p < Math.ceil(visibleGpus / sizing.tp); p++) {
    const mine = gpus.filter((x) => x.pod === p);
    if (!mine.length) continue;
    const first = mine[0];
    const last = mine[mine.length - 1];
    const x0 = first.x;
    const x1 = last.x + cell;
    pods.push({
      p, x0, x1,
      spans: first.n !== last.n,
      labelInside: x1 - x0 >= minPodLabelW,
      gpusShown: mine.length,
    });
  }

  const multi = shown > 1;
  const switchY = nodeY + nodeH + 34;
  const storeY = switchY + (multi ? 56 : 30);

  // The heading counts the whole deployment while the drawing stops at `maxNodes`; without a
  // marker in the picture itself, "8 replicas on 16 nodes" sits directly above two replicas on
  // four nodes and reads as a bug. Reserve the width the marker needs before sizing the viewBox.
  const truncated = sizing.nodes > shown;
  const hiddenNodes = Math.max(0, sizing.nodes - shown);
  const hiddenPods = Math.max(0, sizing.pods - pods.length);
  const truncGap = 14;
  const truncW = truncated
    ? truncGap + Math.max(...truncLabels(hiddenNodes, hiddenPods).map(labelWidth))
    : 0;

  return {
    shown, perNode, cell, nodePad, nodeW, nodeGap, labelH, nodeH,
    contentW, width: contentW + Math.max(padRight, truncW),
    height: storeY + 40, nodeY, switchY, storeY, multi, gpus, pods,
    nodeLabelFull: nodeW >= minNodeLabelW,
    truncated, hiddenNodes, hiddenPods,
    shownPods: pods.length,
    truncX: truncated ? contentW + truncGap : null,
  };
}

/**
 * Caption for the elided part of the deployment, one line per thing being elided. Shared by the
 * layout (which sizes the viewBox to it) and the renderer (which draws it), so they cannot
 * disagree about how much room the marker takes.
 */
export function truncLabels(hiddenNodes: number, hiddenPods: number): string[] {
  const lines: string[] = [];
  if (hiddenPods > 0) lines.push(`+${hiddenPods} more replica${hiddenPods > 1 ? 's' : ''}`);
  if (hiddenNodes > 0) lines.push(`on +${hiddenNodes} more node${hiddenNodes > 1 ? 's' : ''}`);
  return lines.length ? lines : ['…'];
}

// ── Rendering ────────────────────────────────────────────────────────────────────────────────
// The SVG is a pure function of the layout plus a handful of label strings, so the app and the
// headless preview harness render from ONE source. Duplicating the markup in a preview script
// makes the preview drift, at which point it stops being evidence about the real component.

export interface TopologyRenderOptions {
  tp: number;
  perNode: number;
  /** At least one replica crosses a node boundary — drives the warning treatment. */
  multiNode: boolean;
  /** Caption on the shared-storage box, e.g. "shared weights · 67.7 GiB per replica". */
  storeLabel: string;
  /** Prose equivalent of the picture, for screen readers. */
  desc: string;
  titleId?: string;
  descId?: string;
}

/** Escape text bound into SVG. Model names are admin-editable, so this is not optional. */
export function escapeSvgText(s: string): string {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/** ~6.2px per character at 10px/700 Manrope — enough to size a box to its own caption. */
export function labelWidth(text: string): number {
  return text.length * 6.2 + 24;
}

export function topologySvg(t: TopologyLayout, o: TopologyRenderOptions): string {
  const esc = escapeSvgText;
  const titleId = o.titleId ?? 'topotitle';
  const descId = o.descId ?? 'topodesc';
  const cx = t.contentW / 2;
  const storeW = Math.max(180, labelWidth(o.storeLabel));
  const warn = o.multiNode;
  const out: string[] = [];

  out.push(
    `<svg class="topo2" viewBox="0 0 ${t.width} ${t.height}" width="${t.width}" height="${t.height}" ` +
      `role="img" aria-labelledby="${titleId} ${descId}">`,
    `<title id="${titleId}">Deployment topology diagram</title>`,
    `<desc id="${descId}">${esc(o.desc)}</desc>`,
  );

  // logical layer
  out.push(
    `<rect x="${cx - 80}" y="0" width="160" height="26" rx="6" class="tbox router"/>`,
    `<text x="${cx}" y="17" class="tlabel mid">router · llm-d</text>`,
  );
  for (const pod of t.pods) {
    const mid = (pod.x0 + pod.x1) / 2;
    const cls = `tbox pod${pod.spans ? ' spanning' : ''}`;
    out.push(`<line x1="${cx}" y1="26" x2="${mid}" y2="48" class="tlink"/>`);
    if (pod.gpusShown < o.tp) {
      // The node cap ended mid-replica. A closed bar over half a TP group asserts the group is
      // that size, so this one runs off the right edge open — the shape says "continues", which
      // is the truth, and it needs no label text that a four-cell bar could not hold.
      const r = 5;
      const xe = pod.x1 + 12;
      out.push(
        `<path d="M ${xe} 48 H ${pod.x0 + r} A ${r} ${r} 0 0 0 ${pod.x0} ${48 + r} V ${70 - r} ` +
          `A ${r} ${r} 0 0 0 ${pod.x0 + r} 70 H ${xe}" class="${cls} cut"/>`,
      );
    } else {
      out.push(
        `<rect x="${pod.x0}" y="48" width="${pod.x1 - pod.x0}" height="22" rx="5" class="${cls}"/>`,
      );
    }
    out.push(
      `<text x="${pod.labelInside ? mid : pod.x1 + 6}" y="63" class="tlabel${pod.labelInside ? ' mid' : ''}">` +
        `${pod.spans ? '⚠ ' : ''}replica ${pod.p + 1} · TP${o.tp}</text>`,
    );
  }

  // physical layer
  for (let n = 0; n < t.shown; n++) {
    const nx = n * (t.nodeW + t.nodeGap);
    out.push(
      `<rect x="${nx}" y="${t.nodeY}" width="${t.nodeW}" height="${t.nodeH}" rx="7" class="tbox node"/>`,
      `<text x="${nx + t.nodePad}" y="${t.nodeY + 14}" class="tlabel">node ${n + 1}${t.nodeLabelFull ? ` · ${o.perNode} GPU` : ''}</text>`,
    );
    for (const g of t.gpus.filter((x) => x.n === n)) {
      out.push(
        `<rect x="${g.x}" y="${t.nodeY + t.labelH}" width="${t.cell}" height="${t.cell}" rx="4" class="tgpu${g.used ? ' used' : ''}"/>`,
        `<text x="${g.x + t.cell / 2}" y="${t.nodeY + t.labelH + 17}" class="tgpulabel mid">${g.used ? g.g : ''}</text>`,
      );
    }
    out.push(
      `<line x1="${nx + t.nodeW / 2}" y1="${t.nodeY + t.nodeH}" x2="${nx + t.nodeW / 2}" ` +
        `y2="${t.multi ? t.switchY : t.storeY}" class="tlink${t.multi && warn ? ' fabric' : ''}"/>`,
    );
  }

  // What the node cap left out, drawn where the drawing stops rather than only stated in prose
  // under it. Ellipses on both rows so the elision reads as "and so on", not "and that's all".
  if (t.truncX !== null) {
    const lines = truncLabels(t.hiddenNodes, t.hiddenPods);
    const midY = t.nodeY + t.labelH + t.cell / 2;
    const y0 = midY - ((lines.length - 1) * 12) / 2 + 3;
    if (t.pods.length) out.push(`<text x="${t.truncX}" y="63" class="tlabel more">⋯</text>`);
    out.push(`<text x="${t.truncX}" y="${midY + 4}" class="tlabel more">⋯</text>`);
    lines.forEach((s, i) =>
      out.push(`<text x="${t.truncX! + 16}" y="${y0 + i * 12}" class="tlabel more">${esc(s)}</text>`),
    );
  }

  if (t.multi) {
    out.push(
      `<rect x="${cx - 92}" y="${t.switchY}" width="184" height="26" rx="6" class="tbox sw${warn ? ' hot' : ''}"/>`,
      `<text x="${cx}" y="${t.switchY + 17}" class="tlabel mid">${warn ? '⚠ ' : ''}InfiniBand / RoCE fabric</text>`,
      `<line x1="${cx}" y1="${t.switchY + 26}" x2="${cx}" y2="${t.storeY}" class="tlink"/>`,
    );
  }

  out.push(
    `<rect x="${cx - storeW / 2}" y="${t.storeY}" width="${storeW}" height="26" rx="6" class="tbox store"/>`,
    `<text x="${cx}" y="${t.storeY + 17}" class="tlabel mid">${esc(o.storeLabel)}</text>`,
    `</svg>`,
  );
  return out.join('');
}
