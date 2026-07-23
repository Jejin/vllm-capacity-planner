// Reconciliation (FR-19..21) — committed vs available HBM per SKU, on INTEGER BYTES so
// client-live and server-authoritative verdicts match exactly at the boundary (AD-2c, AD-10).
import type { GpuSku } from './types.js';

const GB = 2n ** 30n;

/** A fleet pool: N nodes × G GPUs of one SKU. */
export interface Pool {
  gpu_sku_id: string;
  gpus_per_node: number;
  node_count: number;
}

/** A planned deployment's demand on a SKU: it commits whole GPUs (§G). */
export interface Commitment {
  gpu_sku_id: string;
  gpus: number;
}

export interface PoolCommitment {
  gpu_sku_id: string;
  fleet_gpus: number;
  committed_gpus: number;
  available_gpus: number;
  fleet_bytes: bigint;
  committed_bytes: bigint;
  available_bytes: bigint;
  commitment_pct: number; // may exceed 100
  over_committed: boolean;
}

export type HeadroomVerdict =
  | { verdict: 'fit'; remaining_gpus: number }
  | { verdict: 'shortage'; shortfall_gpus: number }
  | { verdict: 'sku_absent' };

function memBytes(gpu: GpuSku): bigint {
  return BigInt(gpu.mem_gb) * GB;
}

/** Per-SKU reconciliation of a plan against a fleet. Committed = physical whole-GPU HBM (§G). */
export function reconcile(
  fleet: Pool[],
  commitments: Commitment[],
  gpuById: Map<string, GpuSku>,
): PoolCommitment[] {
  const skus = new Set<string>([...fleet.map((p) => p.gpu_sku_id), ...commitments.map((c) => c.gpu_sku_id)]);
  const rows: PoolCommitment[] = [];
  for (const sku of skus) {
    const gpu = gpuById.get(sku);
    if (!gpu) continue;
    const fleet_gpus = fleet.filter((p) => p.gpu_sku_id === sku).reduce((s, p) => s + p.gpus_per_node * p.node_count, 0);
    const committed_gpus = commitments.filter((c) => c.gpu_sku_id === sku).reduce((s, c) => s + c.gpus, 0);
    const fleet_bytes = BigInt(fleet_gpus) * memBytes(gpu);
    const committed_bytes = BigInt(committed_gpus) * memBytes(gpu);
    const available_bytes = fleet_bytes - committed_bytes; // committed + available = fleet total (invariant)
    rows.push({
      gpu_sku_id: sku,
      fleet_gpus,
      committed_gpus,
      available_gpus: fleet_gpus - committed_gpus,
      fleet_bytes,
      committed_bytes,
      available_bytes,
      commitment_pct: fleet_gpus === 0 ? (committed_gpus > 0 ? Infinity : 0) : (committed_gpus / fleet_gpus) * 100,
      over_committed: committed_gpus > fleet_gpus,
    });
  }
  return rows;
}

/** Headroom check for one candidate sizing on its target SKU (FR-21). */
export function headroomCheck(
  targetSkuId: string,
  candidateGpus: number,
  reconciled: PoolCommitment[],
): HeadroomVerdict {
  const row = reconciled.find((r) => r.gpu_sku_id === targetSkuId);
  if (!row || row.fleet_gpus === 0) return { verdict: 'sku_absent' };
  const remaining = row.available_gpus;
  if (candidateGpus <= remaining) return { verdict: 'fit', remaining_gpus: remaining };
  return { verdict: 'shortage', shortfall_gpus: candidateGpus - remaining };
}
