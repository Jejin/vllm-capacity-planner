// Geometry for the deployment topology diagram.
//
// Pure layout: given a sizing and the node width, where does every box go. Kept out of the
// component so it can be unit-tested and rendered headlessly — a diagram whose geometry only
// exists inside a template can only be checked by looking at it, and only in a browser.
//
// The diagram reads left to right — router, then replicas, then the nodes they land on — and
// stacks the node boxes vertically. Laying the nodes out horizontally instead meant a wide plan
// ran off the panel and needed a scrollbar to see at all; stacking spends the page's cheap axis
// instead of its scarce one.
//
// A replica's bar sits in the middle column at the height of its own GPUs, so a tensor-parallel
// group that straddles a node boundary is drawn as a bar spanning two node rows — the geometry
// says "this crosses the fabric" before the colour does. Where several replicas share one node,
// they divide that node's band into slots, and a rule under the GPU cells marks which cells
// belong to which replica.

import type { FeasibleSizing } from './types.js';

export interface TopologyOptions {
  /** Cap on node boxes drawn; beyond this the pattern just repeats. */
  maxNodes?: number;
  /** Cap on the node stack's height. Nodes holding many replicas are tall, and a plan is more
   *  readable showing one node in full than four squeezed. */
  maxStackH?: number;
  cell?: number;
  cellGap?: number;
  nodePad?: number;
  /** Vertical gap between node boxes — the space a spanning replica visibly crosses. */
  nodeGapY?: number;
  labelH?: number;
  routerW?: number;
  routerH?: number;
  /** Horizontal gap between the router, replica and node columns. */
  colGap?: number;
  /** Shortest a replica bar may be drawn; sets how tall a node band has to be. */
  slotMin?: number;
  slotGap?: number;
  spineGap?: number;
  padRight?: number;
  /** Replica bars narrower than this are widened to it. */
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
  y: number;
  /** False for GPUs present in the node but not claimed by this plan. */
  used: boolean;
  /** Replica index this GPU serves, or null when idle. */
  pod: number | null;
}

/** The slice of one replica's bar that sits alongside a single node. */
export interface TopoSeg {
  n: number;
  y0: number;
  y1: number;
}

export interface TopoPod {
  p: number;
  /** Vertical extent of the whole bar — top of its first node's slot to the bottom of its last. */
  y0: number;
  y1: number;
  /** True when this replica's TP group straddles a node boundary. */
  spans: boolean;
  /**
   * GPUs of this replica that fall inside the drawn nodes. Below `tp` when the node cap cuts
   * a replica in half, and the bar must then say so rather than pass for a whole TP group.
   */
  gpusShown: number;
  /**
   * Whether the bar is tall enough to hold its own label. Slot heights shrink as replicas per
   * node grow, and text centred on a 9px bar overlaps its neighbours.
   */
  labelInside: boolean;
  /** One entry per node this replica touches, in node order. */
  segs: TopoSeg[];
}

export interface TopoNode {
  n: number;
  /** Top of the node box. */
  y: number;
  /** Top of the GPU-cell row inside it. */
  gpuY: number;
}

export interface TopologyLayout {
  shown: number;
  perNode: number;
  cell: number;
  nodePad: number;
  nodeW: number;
  nodeH: number;
  nodeGapY: number;
  labelH: number;
  /** Height of a node's replica band — the GPU row plus whatever the replica slots need. */
  bandH: number;
  routerW: number;
  routerH: number;
  routerY: number;
  podX: number;
  podW: number;
  nodeX: number;
  /** x of the inter-node fabric spine, or null when only one node is drawn. */
  spineX: number | null;
  stackH: number;
  width: number;
  /** Drawing width excluding the right-hand text slack — the centre line for stacked boxes. */
  contentW: number;
  height: number;
  switchY: number;
  storeY: number;
  /** More than one node box is drawn, so an inter-node fabric exists in the picture. */
  multi: boolean;
  /** Node boxes are wide enough for the full "node N · G GPU" caption rather than "node N". */
  nodeLabelFull: boolean;
  gpus: TopoGpu[];
  pods: TopoPod[];
  nodes: TopoNode[];
  truncated: boolean;
  hiddenNodes: number;
  /** Replicas with a bar in the drawing — `pods.length`, named for the render side. */
  shownPods: number;
  /** Replicas the node cap left out entirely. */
  hiddenPods: number;
  /** y of the "more of the same" marker, or null when nothing is left out. */
  truncY: number | null;
}

export function topologyLayout(
  sizing: FeasibleSizing,
  gpusPerNode: number,
  opts: TopologyOptions = {},
): TopologyLayout {
  const {
    maxNodes = 4, maxStackH = 380, cell = 26, cellGap = 5, nodePad = 12, nodeGapY = 14,
    labelH = 20, routerW = 108, routerH = 26, colGap = 30, slotMin = 18, slotGap = 6,
    spineGap = 16,
    // Text overhangs its box; without slack the right-most label is clipped by the viewBox.
    padRight = 24, minPodLabelW = 96, minNodeLabelW = 118,
  } = opts;
  const perNode = Math.max(1, Math.floor(gpusPerNode));
  const tp = Math.max(1, sizing.tp);
  const podOf = (g: number) => Math.floor(g / tp);

  /** Replica indices with at least one drawn GPU inside node n. */
  const podsIn = (n: number) => {
    const set: number[] = [];
    for (let j = 0; j < perNode; j++) {
      const g = n * perNode + j;
      if (g >= sizing.gpus) break;
      const p = podOf(g);
      if (!set.includes(p)) set.push(p);
    }
    return set;
  };

  // A node's band has to hold every replica that lands in it, so how many nodes fit in the
  // stack depends on that count — hence the band before the node count, not the other way round.
  let maxPods = 1;
  for (let n = 0; n < Math.min(sizing.nodes, maxNodes); n++) {
    maxPods = Math.max(maxPods, podsIn(n).length);
  }
  const bandH = Math.max(cell, maxPods * (slotMin + slotGap) - slotGap);
  const nodeH = labelH + bandH + nodePad;
  const rowPitch = nodeH + nodeGapY;
  const fits = Math.max(1, Math.floor((maxStackH + nodeGapY) / rowPitch));
  const shown = Math.max(1, Math.min(sizing.nodes, maxNodes, fits));

  const nodeW = perNode * cell + (perNode - 1) * cellGap + nodePad * 2;
  const drawnGpus = Math.min(sizing.gpus, shown * perNode);

  // Pod labels set the middle column's width, so they are composed here rather than in the
  // renderer — a column sized to a label the renderer then changes is a clipped label.
  const podIds: number[] = [];
  for (let g = 0; g < drawnGpus; g += 1) {
    const p = podOf(g);
    if (!podIds.includes(p)) podIds.push(p);
  }
  const podW = Math.max(
    minPodLabelW,
    ...podIds.map((p) => labelWidth(podLabel(p, tp, true))),
  );

  const podX = routerW + colGap;
  const nodeX = podX + podW + colGap;

  const nodes: TopoNode[] = [];
  const gpus: TopoGpu[] = [];
  for (let n = 0; n < shown; n++) {
    const y = n * rowPitch;
    const gpuY = y + labelH + (bandH - cell) / 2;
    nodes.push({ n, y, gpuY });
    for (let j = 0; j < perNode; j++) {
      const g = n * perNode + j;
      const used = g < sizing.gpus;
      gpus.push({
        g, n, used,
        x: nodeX + nodePad + j * (cell + cellGap),
        y: gpuY,
        pod: used ? podOf(g) : null,
      });
    }
  }

  const pods: TopoPod[] = [];
  for (const p of podIds) {
    const mine = gpus.filter((x) => x.pod === p);
    if (!mine.length) continue;
    const segs: TopoSeg[] = [];
    for (const node of nodes) {
      const here = podsIn(node.n);
      const i = here.indexOf(p);
      if (i < 0) continue;
      const slotH = (bandH - (here.length - 1) * slotGap) / here.length;
      const top = node.y + labelH + i * (slotH + slotGap);
      segs.push({ n: node.n, y0: top, y1: top + slotH });
    }
    if (!segs.length) continue;
    const y0 = segs[0].y0;
    const y1 = segs[segs.length - 1].y1;
    pods.push({
      p, y0, y1,
      spans: segs.length > 1,
      gpusShown: mine.length,
      labelInside: segs[0].y1 - segs[0].y0 >= 14,
      segs,
    });
  }

  const multi = shown > 1;
  const stackH = shown * nodeH + (shown - 1) * nodeGapY;
  const spineX = multi ? nodeX + nodeW + spineGap : null;

  // The heading counts the whole deployment while the drawing stops at the node cap; without a
  // marker in the picture itself, "8 replicas on 16 nodes" sits directly above two replicas on
  // four nodes and reads as a bug.
  const truncated = sizing.nodes > shown;
  const hiddenNodes = Math.max(0, sizing.nodes - shown);
  const hiddenPods = Math.max(0, sizing.pods - pods.length);
  const truncY = truncated ? stackH + 16 : null;
  const afterStack = stackH + (truncated ? 42 : 0);

  const switchY = afterStack + 22;
  const storeY = (multi ? switchY + 26 + 18 : afterStack + 20);

  const drawnW = (spineX ?? nodeX + nodeW) + (multi ? 4 : 0);
  const contentW = Math.max(drawnW, nodeX + nodeW);

  return {
    shown, perNode, cell, nodePad, nodeW, nodeH, nodeGapY, labelH, bandH,
    routerW, routerH, routerY: stackH / 2 - routerH / 2,
    podX, podW, nodeX, spineX, stackH,
    contentW, width: contentW + padRight,
    height: storeY + 26 + 8,
    switchY, storeY, multi,
    nodeLabelFull: nodeW >= minNodeLabelW,
    gpus, pods, nodes,
    truncated, hiddenNodes, hiddenPods,
    shownPods: pods.length,
    truncY,
  };
}

/**
 * Caption on a replica bar. Shared by the layout (which sizes the middle column to it) and the
 * renderer (which draws it), so the column cannot be sized to a string the renderer changes.
 * `spans` only adds the warning glyph — it is passed true when measuring, to size for the worst
 * case rather than reflow the column when a plan starts crossing nodes.
 */
export function podLabel(p: number, tp: number, spans: boolean): string {
  return `${spans ? '⚠ ' : ''}replica ${p + 1} · TP${tp}`;
}

/**
 * Caption for the elided part of the deployment, one line per thing being elided. Shared by the
 * layout (which reserves room for it) and the renderer, so they cannot disagree.
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
  const storeW = Math.max(180, labelWidth(o.storeLabel));
  const cx = Math.max(t.contentW / 2, storeW / 2);
  const warn = o.multiNode;
  const out: string[] = [];

  out.push(
    `<svg class="topo2" viewBox="0 0 ${t.width} ${t.height}" width="${t.width}" height="${t.height}" ` +
      `role="img" aria-labelledby="${titleId} ${descId}">`,
    `<title id="${titleId}">Deployment topology diagram</title>`,
    `<desc id="${descId}">${esc(o.desc)}</desc>`,
  );

  // router — left column, centred on the node stack
  const routerCY = t.routerY + t.routerH / 2;
  out.push(
    `<rect x="0" y="${t.routerY}" width="${t.routerW}" height="${t.routerH}" rx="6" class="tbox router"/>`,
    `<text x="${t.routerW / 2}" y="${t.routerY + 17}" class="tlabel mid">router · llm-d</text>`,
  );

  // replicas — middle column, each at the height of its own GPUs
  for (const pod of t.pods) {
    const cls = `tbox pod${pod.spans ? ' spanning' : ''}`;
    const midY = (pod.y0 + pod.y1) / 2;
    out.push(`<line x1="${t.routerW}" y1="${routerCY}" x2="${t.podX}" y2="${midY}" class="tlink"/>`);
    if (pod.gpusShown < o.tp) {
      // The node cap ended mid-replica. A closed bar over half a TP group asserts the group is
      // that size, so this one runs off the bottom open — the shape says "continues", which is
      // the truth, and it needs no label text a short bar could not hold.
      const r = 5;
      const ye = pod.y1 + 12;
      out.push(
        `<path d="M ${t.podX} ${ye} V ${pod.y0 + r} A ${r} ${r} 0 0 1 ${t.podX + r} ${pod.y0} ` +
          `H ${t.podX + t.podW - r} A ${r} ${r} 0 0 1 ${t.podX + t.podW} ${pod.y0 + r} ` +
          `V ${ye}" class="${cls} cut"/>`,
      );
    } else {
      out.push(
        `<rect x="${t.podX}" y="${pod.y0}" width="${t.podW}" height="${pod.y1 - pod.y0}" rx="5" class="${cls}"/>`,
      );
    }
    out.push(
      `<text x="${t.podX + t.podW / 2}" y="${(pod.labelInside ? midY : pod.y1 + 11) + 3.5}" class="tlabel mid">` +
        `${esc(podLabel(pod.p, o.tp, pod.spans))}</text>`,
    );
    // one link per node this replica lands on — two links means it crosses a boundary
    for (const seg of pod.segs) {
      const node = t.nodes[seg.n];
      out.push(
        `<line x1="${t.podX + t.podW}" y1="${(seg.y0 + seg.y1) / 2}" x2="${t.nodeX}" ` +
          `y2="${node.gpuY + t.cell / 2}" class="tlink${pod.spans && warn ? ' fabric' : ''}"/>`,
      );
    }
  }

  // nodes — right column, stacked
  for (const node of t.nodes) {
    out.push(
      `<rect x="${t.nodeX}" y="${node.y}" width="${t.nodeW}" height="${t.nodeH}" rx="7" class="tbox node"/>`,
      `<text x="${t.nodeX + t.nodePad}" y="${node.y + 14}" class="tlabel">node ${node.n + 1}` +
        `${t.nodeLabelFull ? ` · ${o.perNode} GPU` : ''}</text>`,
    );
    for (const g of t.gpus.filter((x) => x.n === node.n)) {
      out.push(
        `<rect x="${g.x}" y="${g.y}" width="${t.cell}" height="${t.cell}" rx="4" class="tgpu${g.used ? ' used' : ''}"/>`,
        `<text x="${g.x + t.cell / 2}" y="${g.y + 17}" class="tgpulabel mid">${g.used ? g.g : ''}</text>`,
      );
    }
    // which cells belong to which replica — the one thing a vertical stack would otherwise lose
    for (const pod of t.pods) {
      const mine = t.gpus.filter((x) => x.n === node.n && x.pod === pod.p);
      if (!mine.length) continue;
      const x0 = mine[0].x;
      const x1 = mine[mine.length - 1].x + t.cell;
      out.push(
        `<rect x="${x0}" y="${node.gpuY + t.cell + 3}" width="${x1 - x0}" height="2.5" rx="1.25" ` +
          `class="tunder${pod.spans ? ' spanning' : ''}"/>`,
      );
    }
  }

  // the inter-node fabric: a spine every node hangs off, dropping to the labelled box
  if (t.spineX !== null && t.nodes.length) {
    const first = t.nodes[0];
    const last = t.nodes[t.nodes.length - 1];
    const cls = `tlink${warn ? ' fabric' : ''}`;
    for (const node of t.nodes) {
      out.push(
        `<line x1="${t.nodeX + t.nodeW}" y1="${node.gpuY + t.cell / 2}" x2="${t.spineX}" ` +
          `y2="${node.gpuY + t.cell / 2}" class="${cls}"/>`,
      );
    }
    out.push(
      `<line x1="${t.spineX}" y1="${first.gpuY + t.cell / 2}" x2="${t.spineX}" ` +
        `y2="${last.gpuY + t.cell / 2}" class="${cls}"/>`,
      `<line x1="${t.spineX}" y1="${last.gpuY + t.cell / 2}" x2="${t.spineX}" y2="${t.switchY + 13}" class="${cls}"/>`,
      `<line x1="${t.spineX}" y1="${t.switchY + 13}" x2="${cx + 92}" y2="${t.switchY + 13}" class="${cls}"/>`,
      `<rect x="${cx - 92}" y="${t.switchY}" width="184" height="26" rx="6" class="tbox sw${warn ? ' hot' : ''}"/>`,
      `<text x="${cx}" y="${t.switchY + 17}" class="tlabel mid">${warn ? '⚠ ' : ''}InfiniBand / RoCE fabric</text>`,
    );
  }

  // What the node cap left out, marked under the columns it was cut from. Under the replica
  // column rather than the node column: on a narrow node the caption is wider than the node box
  // and would run under the fabric spine descending on its right.
  if (t.truncY !== null) {
    const lines = truncLabels(t.hiddenNodes, t.hiddenPods);
    out.push(`<text x="${t.podX}" y="${t.truncY}" class="tlabel more">⋯</text>`);
    lines.forEach((s, i) =>
      out.push(
        `<text x="${t.podX + 16}" y="${t.truncY! + i * 12 - 3}" class="tlabel more">${esc(s)}</text>`,
      ),
    );
  }

  out.push(
    `<rect x="${cx - storeW / 2}" y="${t.storeY}" width="${storeW}" height="26" rx="6" class="tbox store"/>`,
    `<text x="${cx}" y="${t.storeY + 17}" class="tlabel mid">${esc(o.storeLabel)}</text>`,
    `</svg>`,
  );
  return out.join('');
}
