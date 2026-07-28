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
  const width = contentW + padRight;
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
    pods.push({ p, x0, x1, spans: first.n !== last.n, labelInside: x1 - x0 >= minPodLabelW });
  }

  const multi = shown > 1;
  const switchY = nodeY + nodeH + 34;
  const storeY = switchY + (multi ? 56 : 30);

  return {
    shown, perNode, cell, nodePad, nodeW, nodeGap, labelH, nodeH,
    width, contentW, height: storeY + 40, nodeY, switchY, storeY, multi, gpus, pods,
    nodeLabelFull: nodeW >= minNodeLabelW,
    truncated: sizing.nodes > shown,
    hiddenNodes: Math.max(0, sizing.nodes - shown),
  };
}
