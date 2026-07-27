// Turn a sizing result into a runnable `vllm serve` command.
//
// The planner already decides every value vLLM needs at launch — TP size, KV dtype, context
// length, memory utilisation, how many sessions a replica can hold. Emitting the command closes
// the loop from "how many GPUs" to "how do I start it", and makes the plan checkable: if the
// command OOMs, the sizing was wrong.

import type { FeasibleSizing, Model, SizingInput } from './types.js';

export interface ServeCommandOptions {
  /** Hugging Face id to serve. Falls back to the catalogue model name. */
  hf_id?: string;
  /** Emit a `docker run` wrapper around the same arguments. */
  docker?: boolean;
  /** Image for the docker form. */
  image?: string;
}

export interface ServeCommand {
  /** Argument vector — the source of truth; the strings below are rendered from it. */
  argv: string[];
  /** Shell command, backslash-continued for readability. */
  command: string;
  /** Notes explaining non-obvious flags, shown beside the command. */
  notes: string[];
}

/** vLLM's flag for a KV cache dtype, or null when the default (model dtype) applies. */
function kvCacheDtype(bytes: number): string | null {
  return bytes === 1 ? 'fp8' : null;
}

/**
 * Build the launch command for one replica of a sized plan.
 *
 * Note this describes a POD, not the whole deployment: `--tensor-parallel-size` is the shard
 * width, and the plan's `pods` count is how many copies of this command to run behind a load
 * balancer. Conflating the two is the classic way to under-provision.
 */
export function serveCommand(
  model: Model,
  input: SizingInput,
  sizing: FeasibleSizing,
  opts: ServeCommandOptions = {},
): ServeCommand {
  const id = opts.hf_id?.trim() || model.name;
  const argv: string[] = ['vllm', 'serve', id];
  const notes: string[] = [];

  argv.push('--tensor-parallel-size', String(sizing.tp));
  if (sizing.multi_node) {
    notes.push(
      `TP ${sizing.tp} exceeds ${input.gpus_per_node} GPUs per node, so this replica spans nodes — ` +
        `it needs a Ray cluster and fast interconnect, and MBU degrades against single-node TP.`,
    );
  }

  argv.push('--max-model-len', String(input.selected_ctx));
  argv.push('--gpu-memory-utilization', String(input.mem_util_fraction));

  const kv = kvCacheDtype(input.kv_dtype_bytes);
  if (kv) {
    argv.push('--kv-cache-dtype', kv);
    notes.push('FP8 KV cache halves cache size against FP16 for a small quality cost — the plan assumes it.');
  }

  // Cap the scheduler at what the memory budget actually supports. Left unset, vLLM will admit
  // far more sequences than the KV budget holds and start preempting under load.
  argv.push('--max-num-seqs', String(sizing.concurrency_per_pod));
  notes.push(
    `--max-num-seqs is set to this pod's KV budget (${sizing.concurrency_per_pod} sessions at ` +
      `${Math.round(input.selected_ctx * input.avg_context_utilisation).toLocaleString()} tokens each). ` +
      `Raising it past this causes preemption rather than more throughput.`,
  );

  if (sizing.tight) {
    notes.push(
      `Headroom is ${(sizing.headroom_fraction * 100).toFixed(1)}% — under the 10% tight threshold. ` +
        `Expect this to OOM if prompts run longer than modelled; lower --max-model-len or raise TP.`,
    );
  }
  if (sizing.weights_estimated) {
    notes.push('Weights are a flat-factor estimate (no embedding geometry on this model), so the memory figures carry extra error.');
  }
  if (sizing.pods > 1) {
    notes.push(`This is ONE replica. The plan needs ${sizing.pods} of them (${sizing.gpus} GPUs total) to serve ${input.target_concurrency} concurrent sessions.`);
  }

  const command = opts.docker
    ? dockerWrap(argv, opts.image ?? 'vllm/vllm-openai:latest')
    : renderShell(argv);
  return { argv, command, notes };
}

/** Render argv as a readable shell command, one flag per line. */
function renderShell(argv: string[]): string {
  const head = argv.slice(0, 3).join(' ');
  const rest: string[] = [];
  for (let i = 3; i < argv.length; i += 2) {
    rest.push(argv[i + 1] === undefined ? argv[i] : `${argv[i]} ${argv[i + 1]}`);
  }
  return rest.length ? `${head} \\\n  ${rest.join(' \\\n  ')}` : head;
}

function dockerWrap(argv: string[], image: string): string {
  const inner = renderShell(argv)
    .split('\n')
    .map((l) => `  ${l.trim()}`)
    .join(' \\\n');
  return [
    'docker run --gpus all \\',
    '  --ipc=host -p 8000:8000 \\',
    '  -v ~/.cache/huggingface:/root/.cache/huggingface \\',
    `  ${image} \\`,
    inner.replace(/^ {2}/, '  '),
  ].join('\n');
}
