import { describe, it, expect } from 'vitest';
import { serveCommand } from '../serve.js';
import { computeSizing } from '../engine.js';
import { seedCatalog } from '../seed.js';
import type { FeasibleSizing } from '../types.js';

const { models, gpus } = seedCatalog();
const M = (i: string) => models.find((x) => x.id === i)!;
const G = (i: string) => gpus.find((x) => x.id === i)!;
const base = {
  quant: 'FP8' as const, kv_dtype_bytes: 1, selected_ctx: 131072, avg_context_utilisation: 0.6,
  target_concurrency: 64, mem_util_fraction: 0.9, gpus_per_node: 8,
};

describe('vllm serve command generation', () => {
  it('emits the plan\'s own numbers, not defaults', () => {
    const m = M('llama33-70b');
    const s = computeSizing(m, G('h200'), base) as FeasibleSizing;
    const { argv, command } = serveCommand(m, base, s, { hf_id: 'meta-llama/Llama-3.3-70B-Instruct' });
    expect(argv.slice(0, 3)).toEqual(['vllm', 'serve', 'meta-llama/Llama-3.3-70B-Instruct']);
    const flag = (f: string) => argv[argv.indexOf(f) + 1];
    expect(flag('--tensor-parallel-size')).toBe(String(s.tp));
    expect(flag('--max-model-len')).toBe('131072');
    expect(flag('--gpu-memory-utilization')).toBe('0.9');
    expect(flag('--max-num-seqs')).toBe(String(s.concurrency_per_pod));
    expect(command).toContain('vllm serve');
    expect(command).toContain('\\\n'); // wrapped, one flag per line
  });

  it('sets --kv-cache-dtype only when the plan assumes FP8 KV', () => {
    const m = M('llama33-70b');
    const fp8 = computeSizing(m, G('h200'), base) as FeasibleSizing;
    expect(serveCommand(m, base, fp8).argv).toContain('--kv-cache-dtype');
    const in16 = { ...base, kv_dtype_bytes: 2 };
    const fp16 = computeSizing(m, G('h200'), in16) as FeasibleSizing;
    expect(serveCommand(m, in16, fp16).argv).not.toContain('--kv-cache-dtype');
  });

  it('caps --max-num-seqs at the pod KV budget, not the deployment target', () => {
    const m = M('llama33-70b');
    const s = computeSizing(m, G('h200'), base) as FeasibleSizing;
    const { argv } = serveCommand(m, base, s);
    const v = Number(argv[argv.indexOf('--max-num-seqs') + 1]);
    expect(v).toBe(s.concurrency_per_pod);
    expect(v).toBeLessThan(base.target_concurrency); // the whole point: one pod ≠ the fleet
  });

  it('says plainly that the command is one replica of several', () => {
    const m = M('llama33-70b');
    const s = computeSizing(m, G('h200'), base) as FeasibleSizing;
    expect(s.pods).toBeGreaterThan(1);
    const { notes } = serveCommand(m, base, s);
    expect(notes.join(' ')).toContain(`${s.pods} of them`);
  });

  it('warns on a tight plan', () => {
    const m = M('qwen3-32b');
    const input = { ...base, quant: 'Q4_K_M' as const, selected_ctx: 4096, target_concurrency: 1, gpus_per_node: 1 };
    const s = computeSizing(m, G('rtx4090'), input) as FeasibleSizing;
    expect(s.tight).toBe(true);
    expect(serveCommand(m, input, s).notes.join(' ')).toMatch(/tight threshold/i);
  });

  it('warns when a replica spans nodes', () => {
    const m = M('kimi-k3');
    const input = { ...base, quant: 'MXFP4' as const, selected_ctx: 1048576, target_concurrency: 8 };
    const s = computeSizing(m, G('h200'), input) as FeasibleSizing;
    expect(s.multi_node).toBe(true);
    expect(serveCommand(m, input, s).notes.join(' ')).toMatch(/spans nodes/i);
  });

  it('produces a docker form carrying the same argv', () => {
    const m = M('llama33-70b');
    const s = computeSizing(m, G('h200'), base) as FeasibleSizing;
    const d = serveCommand(m, base, s, { docker: true });
    expect(d.command).toContain('docker run --gpus all');
    expect(d.command).toContain('vllm/vllm-openai:latest');
    expect(d.argv).toEqual(serveCommand(m, base, s).argv); // argv is the source of truth
  });

  it('falls back to the catalogue name when no hf id is supplied', () => {
    const m = M('llama33-70b');
    const s = computeSizing(m, G('h200'), base) as FeasibleSizing;
    expect(serveCommand(m, base, s).argv[2]).toBe(m.name);
  });
});
