import { describe, it, expect } from 'vitest';
import { serveCommand } from '../serve.js';
import { HF_ID_RE } from '../schema.js';
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
    const { argv, command } = serveCommand(m, base, s);
    // the artifact comes from the catalogue entry's FP8 deployment, never from the display name
    expect(argv.slice(0, 3)).toEqual(['vllm', 'serve', 'RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic']);
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
    const input = { ...base, quant: 'INT4' as const, selected_ctx: 4096, target_concurrency: 1, gpus_per_node: 1 };
    const s = computeSizing(m, G('rtx4090'), input) as FeasibleSizing;
    expect(s.tight).toBe(true);
    expect(serveCommand(m, input, s).notes.join(' ')).toMatch(/tight threshold/i);
  });

  it('warns when a replica spans nodes', () => {
    const m = M('glm52');
    const input = { ...base, quant: 'FP8' as const, selected_ctx: 1048576, target_concurrency: 8 };
    const s = computeSizing(m, G('h100'), input) as FeasibleSizing;
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

  it('emits --max-num-batched-tokens only when it differs from the default', () => {
    const m = M('llama33-70b');
    const plain = computeSizing(m, G('h200'), base) as FeasibleSizing;
    expect(serveCommand(m, base, plain).argv).not.toContain('--max-num-batched-tokens');

    const input = { ...base, max_num_batched_tokens: 32768 };
    const big = computeSizing(m, G('h200'), input) as FeasibleSizing;
    const cmd = serveCommand(m, input, big);
    expect(cmd.argv[cmd.argv.indexOf('--max-num-batched-tokens') + 1]).toBe('32768');
    // the plan was sized against the larger reserve, so the flag is load-bearing
    expect(cmd.notes.join(' ')).toMatch(/optimistic/i);
    expect(big.activation_gb).toBeGreaterThan(plain.activation_gb);
  });

  // This test previously asserted the OPPOSITE — that the command falls back to the catalogue
  // display name. That fallback is why every seeded model emitted `vllm serve Llama 3.3 70B
  // Instruct`: three positional arguments to the shell, and unresolvable even if it parsed.
  it('never puts a display name where the artifact id goes', () => {
    for (const m of models) {
      for (const q of m.quants) {
        const s = computeSizing(m, G('b300'), { ...base, quant: q, selected_ctx: Math.min(base.selected_ctx, m.max_ctx) });
        if (!s.ok) continue;
        const cmd = serveCommand(m, { ...base, quant: q, selected_ctx: Math.min(base.selected_ctx, m.max_ctx) }, s);
        if (cmd.blocked) {
          expect(cmd.argv).toEqual([]); // blocked means NO command, not a degraded one
          expect(cmd.command).toBe('');
          continue;
        }
        expect(cmd.argv[2]).not.toBe(m.name);
        expect(cmd.argv[2]).toMatch(HF_ID_RE);
        expect(cmd.command).not.toContain(m.name);
      }
    }
  });

  it('blocks rather than guesses when a precision has no launch path', () => {
    // Kimi K3's base repo is bfloat16 and no MXFP4 artifact is catalogued, so the only
    // precision it offers cannot be launched — sizing still works, the command does not.
    const m = M('kimi-k3');
    const input = { ...base, quant: 'MXFP4' as const, selected_ctx: 1048576, target_concurrency: 8 };
    const s = computeSizing(m, G('b300'), input) as FeasibleSizing;
    expect(s.ok).toBe(true); // memory feasibility is unaffected
    const cmd = serveCommand(m, input, s);
    expect(cmd.blocked).toMatch(/No known MXFP4 deployment path/);
    expect(cmd.argv).toEqual([]);
    expect(cmd.artifact).toBeUndefined();
  });

  it('emits --quantization for an online path and never for a checkpoint one', () => {
    // online: an unquantised base repo plus a load-time method
    const mistral = M('mistral-s24');
    const on = { ...base, quant: 'FP8' as const };
    const sm = computeSizing(mistral, G('h200'), on) as FeasibleSizing;
    const online = serveCommand(mistral, on, sm);
    expect(online.argv[2]).toBe('mistralai/Mistral-Small-3.2-24B-Instruct-2506');
    expect(online.argv[online.argv.indexOf('--quantization') + 1]).toBe('fp8_per_tensor');
    expect(online.artifact?.source).toBe('online');

    // checkpoint: the artifact IS FP8, so the flag would conflict with its own metadata
    const ds = M('dsv3');
    const sd = computeSizing(ds, G('b300'), on) as FeasibleSizing;
    const ckpt = serveCommand(ds, on, sd);
    expect(ckpt.argv[2]).toBe('deepseek-ai/DeepSeek-V3'); // base repo is natively FP8
    expect(ckpt.argv).not.toContain('--quantization');
    expect(ckpt.notes.join(' ')).toMatch(/already a FP8 checkpoint/);

    // and the same model at a different precision resolves a DIFFERENT repository
    const int4 = { ...base, quant: 'INT4' as const };
    const l70 = M('llama33-70b');
    const si = computeSizing(l70, G('h200'), int4) as FeasibleSizing;
    expect(serveCommand(l70, int4, si).argv[2]).toBe('RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16');
  });

  it('quotes any argument the shell would split', () => {
    const m = { ...M('llama33-70b'), hf_id: undefined, deployments: { FP8: { source: 'online' as const, method: 'fp8 per tensor' } } };
    const s = computeSizing(M('llama33-70b'), G('h200'), base) as FeasibleSizing;
    // no artifact id at all => blocked, with the reason naming what is missing
    expect(serveCommand(m, base, s).blocked).toMatch(/no Hugging Face artifact id/);
    // and a value carrying a space survives rendering as ONE argument
    const spaced = { ...m, hf_id: 'owner/name' };
    const cmd = serveCommand(spaced, base, s);
    expect(cmd.command).toContain(`'fp8 per tensor'`);
  });
});
