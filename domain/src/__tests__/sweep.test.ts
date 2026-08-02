// Whole-catalogue invariant sweep.
//
// This exists because the unit tests check the cases someone thought of. Every acceptance
// vector was green while a 5,000,000-token workload sized a 128K model into 512 GPUs and
// reported it feasible — nobody had written that case down, because nobody had imagined it.
//
// So instead of asserting values, this asserts PROPERTIES that must hold for every plan the
// catalogue can express, and runs them across every combination of model, GPU, precision,
// concurrency and workload shape. It is not a substitute for the vectors: they pin the
// arithmetic, this catches the shapes the arithmetic was never asked about.
//
// Cheap enough to keep in CI — the whole sweep is a few thousand pure-function calls.

import { describe, it, expect } from 'vitest';
import { computeSizing } from '../engine.js';
import { seedCatalog } from '../seed.js';
import type { SizingInput } from '../types.js';

const { models, gpus } = seedCatalog();

const WORKLOADS: (SizingInput['workload'] | undefined)[] = [
  undefined,
  { prompt_p50: 2_000, prompt_p95: 60_000, output_p50: 400, output_p95: 4_000 },
];
const CONCURRENCIES = [1, 64];
const PER_NODE = 8;

/** Every (model, GPU, precision, concurrency, workload) the catalogue can express. */
function* everyPlan() {
  for (const model of models) {
    for (const gpu of gpus) {
      for (const quant of model.quants) {
        for (const target_concurrency of CONCURRENCIES) {
          for (const workload of WORKLOADS) {
            const selected_ctx = Math.min(131_072, model.max_ctx);
            const input: SizingInput = {
              quant, kv_dtype_bytes: 1, selected_ctx, avg_context_utilisation: 0.6,
              target_concurrency, mem_util_fraction: 0.9, gpus_per_node: PER_NODE, workload,
            };
            yield { model, gpu, input, tag: `${model.id}/${gpu.id}/${quant}/c${target_concurrency}/${workload ? 'p50p95' : 'flat'}` };
          }
        }
      }
    }
  }
}

describe('catalogue-wide invariant sweep', () => {
  const plans = [...everyPlan()];

  it('covers the whole catalogue, and most of it is feasible', () => {
    expect(plans.length).toBeGreaterThan(2_000);
    const ok = plans.filter(({ model, gpu, input }) => computeSizing(model, gpu, input).ok).length;
    expect(ok / plans.length).toBeGreaterThan(0.8); // a mostly-infeasible catalogue is its own bug
  });

  it('never throws, and an infeasible plan always explains itself', () => {
    const bad: string[] = [];
    for (const { model, gpu, input, tag } of plans) {
      let r;
      try {
        r = computeSizing(model, gpu, input);
      } catch (e) {
        bad.push(`${tag}: threw ${(e as Error).message}`);
        continue;
      }
      if (!r.ok && (typeof r.reason !== 'string' || r.reason.length < 20)) {
        bad.push(`${tag}: infeasible with no usable reason`);
      }
    }
    expect(bad).toEqual([]);
  });

  it('every feasible plan is internally consistent', () => {
    const bad: string[] = [];
    const chk = (cond: boolean, why: string, tag: string) => { if (!cond) bad.push(`${tag}: ${why}`); };

    for (const { model, gpu, input, tag } of plans) {
      const r = computeSizing(model, gpu, input);
      if (!r.ok) continue;

      for (const [k, v] of Object.entries(r)) {
        if (typeof v === 'number') chk(Number.isFinite(v), `${k} is ${v}`, tag);
      }
      // structure
      chk(r.gpus === r.pods * r.tp, 'gpus != pods x tp', tag);
      chk(r.nodes === Math.ceil(r.gpus / PER_NODE), 'nodes != ceil(gpus / per-node)', tag);
      chk(r.multi_node === r.tp > PER_NODE, 'multi_node disagrees with tp vs per-node', tag);
      // memory
      chk(r.concurrency_per_pod >= 1, 'a feasible plan holding zero sessions', tag);
      chk(r.kv_per_request_gb > 0, 'no KV per request', tag);
      chk(r.free_gb > 0, 'no free KV space', tag);
      chk(r.headroom_fraction > 0 && r.headroom_fraction < 1, `headroom ${r.headroom_fraction}`, tag);
      chk(r.tight === r.headroom_fraction < 0.1, 'tight flag disagrees with headroom', tag);
      // the request the plan sizes for must fit the context it serves
      chk(r.kv_tokens <= input.selected_ctx, `sizes ${r.kv_tokens} tokens into a ${input.selected_ctx} context`, tag);
      // decode model
      chk(r.expert_coverage > 0 && r.expert_coverage <= 1, `coverage ${r.expert_coverage}`, tag);
      chk(r.decode_stream_gb > 0, 'streams nothing per step', tag);
      chk(r.decode_stream_gb <= r.weights_gb * 1.001, 'streams more than the whole checkpoint', tag);
      chk(r.collective_share >= 0 && r.collective_share < 1, `collective share ${r.collective_share}`, tag);
      chk(r.tp > 1 || r.collective_sec === 0, 'a single rank paying for an all-reduce', tag);
      // prefill
      chk(r.ttft_ms > 0, 'instant first token', tag);
      chk(r.prefill_collective_sec >= 0, 'negative prefill collective', tag);
      // provenance
      chk(r.measured.status === 'absent', 'measured profile applied without one being given', tag);
      // the page must not contradict itself: both compute figures withhold together
      chk(!!r.throughput_suppressed === !!r.ttft_suppressed, 'throughput and TTFT disagree on withholding', tag);
      // workload split
      if (input.workload) {
        chk(r.kv_tokens === input.workload.prompt_p95 + input.workload.output_p95, 'KV not sized at P95', tag);
        chk(r.prefill_tokens === input.workload.prompt_p50, 'prefill not timed at P50', tag);
      } else {
        chk(Math.abs(r.kv_tokens - r.prefill_tokens) < 1e-9, 'flat input produced two different token counts', tag);
      }
    }
    expect(bad.slice(0, 20)).toEqual([]);
  });
});
