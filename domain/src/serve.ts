// Turn a sizing result into a runnable `vllm serve` command.
//
// The planner already decides every value vLLM needs at launch — TP size, KV dtype, context
// length, memory utilisation, how many sessions a replica can hold. Emitting the command closes
// the loop from "how many GPUs" to "how do I start it", and makes the plan checkable: if the
// command OOMs, the sizing was wrong.

import { DEFAULT_BATCHED_TOKENS } from './engine.js';
import { HF_ID_RE } from './schema.js';
import { kvScalePolicy } from './kvscale.js';
import type { DeploymentVariant, FeasibleSizing, Model, Quant, SizingInput } from './types.js';

export interface ServeCommandOptions {
  /** Emit a `docker run` wrapper around the same arguments. */
  docker?: boolean;
  /** Image for the docker form. */
  image?: string;
}

export interface ServeCommand {
  /** Argument vector — the source of truth; the strings below are rendered from it. Empty when blocked. */
  argv: string[];
  /** Shell command, backslash-continued for readability. Empty when blocked. */
  command: string;
  /** Notes explaining non-obvious flags, shown beside the command. */
  notes: string[];
  /** Why no command could be generated. Present ⇒ argv/command are empty. */
  blocked?: string;
  /** The artifact this command resolves, once one is known. */
  artifact?: { hf_id: string; source: DeploymentVariant['source']; method?: string; revision?: string };
}

/**
 * Resolve the artifact and launch semantics for one precision, or say why they can't be.
 *
 * The planner used to fall back to `model.name` here, so every seeded model emitted
 * `vllm serve Llama 3.3 70B Instruct` — three positional arguments to the shell, and a
 * resolution failure even if it parsed. Guessing is worse than refusing: a command that starts
 * at a precision the plan was not sized for OOMs later, against arithmetic that was correct for
 * a deployment nobody launched.
 */
export function resolveDeployment(
  model: Model,
  quant: Quant,
): { artifact: NonNullable<ServeCommand['artifact']> } | { blocked: string } {
  const variant = model.deployments?.[quant];
  if (!variant) {
    return {
      blocked:
        `No known ${quant} deployment path for ${model.name}. Sizing is unaffected, but launching at ` +
        `${quant} needs either a pre-quantised checkpoint or a vLLM online-quantisation method — ` +
        `add one to this model's deployments in the catalogue.`,
    };
  }
  const hf_id = variant.hf_id ?? model.hf_id;
  if (!hf_id || !HF_ID_RE.test(hf_id)) {
    return {
      blocked:
        `${model.name} has no Hugging Face artifact id, so there is nothing for \`vllm serve\` to ` +
        `resolve. Add the owner/name repository to the catalogue entry.`,
    };
  }
  return {
    artifact: {
      hf_id,
      source: variant.source,
      method: variant.source === 'online' ? variant.method : undefined,
      revision: variant.revision ?? model.revision,
    },
  };
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
  const resolved = resolveDeployment(model, input.quant);
  if ('blocked' in resolved) return { argv: [], command: '', notes: [], blocked: resolved.blocked };
  const { artifact } = resolved;

  const argv: string[] = ['vllm', 'serve', artifact.hf_id];
  const notes: string[] = [];

  if (artifact.revision) argv.push('--revision', artifact.revision);
  if (artifact.source === 'online') {
    // the plan's bytes/param only materialise if the flag is actually passed
    argv.push('--quantization', artifact.method!);
    notes.push(
      `${input.quant} is applied at load time by vLLM (--quantization ${artifact.method}) to an ` +
        `unquantised checkpoint. Dropping the flag serves the base precision, which needs far more HBM than this plan allows.`,
    );
  } else if (artifact.source === 'checkpoint') {
    notes.push(
      `${artifact.hf_id} is already a ${input.quant} checkpoint, so no --quantization flag is passed — ` +
        `the precision comes from the artifact's own metadata, and a conflicting flag would be an error.`,
    );
  } else {
    notes.push(`${input.quant} is this checkpoint's native precision, so no quantisation flag is needed.`);
  }

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
    // Where the scales come from is not implied by the dtype flag. With none in the checkpoint
    // and no request to compute them, vLLM sets every scale to 1.0 and says nothing.
    const scales = kvScalePolicy(model, input.quant, input.kv_dtype_bytes);
    if (scales.source === 'runtime') {
      // CLI spelling of the `calculate_kv_scales` engine argument
      argv.push('--calculate-kv-scales');
    }
    notes.push(`${scales.headline}. ${scales.detail}${scales.remedy ? ` ${scales.remedy}` : ''}`);
  }

  // Only emit the chunk when it differs from vLLM's default — an explicit flag that repeats the
  // default is noise, but a plan sized against a raised chunk MUST pass it or the reserve lies.
  if (input.max_num_batched_tokens && input.max_num_batched_tokens !== DEFAULT_BATCHED_TOKENS) {
    argv.push('--max-num-batched-tokens', String(input.max_num_batched_tokens));
    notes.push(
      `Prefill chunk raised to ${input.max_num_batched_tokens.toLocaleString()} tokens, which is why ` +
        `${sizing.activation_gb.toFixed(1)} GiB of the ${sizing.runtime_reserve_gb.toFixed(1)} GiB per-GPU ` +
        `reserve is activations. Launching without this flag makes the plan optimistic.`,
    );
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
  return { argv, command, notes, artifact };
}

/**
 * Single-quote any token the shell would not read as one word. argv is the source of truth and
 * every element is one argument; rendering it back to a string has to preserve that.
 */
function shellQuote(tok: string): string {
  return /^[A-Za-z0-9_@%+=:,./-]+$/.test(tok) ? tok : `'${tok.replace(/'/g, `'\\''`)}'`;
}

/** Render argv as a readable shell command, one flag per line. */
function renderShell(argv: string[]): string {
  const q = argv.map(shellQuote);
  const head = q.slice(0, 3).join(' ');
  const rest: string[] = [];
  for (let i = 3; i < q.length; i += 2) {
    rest.push(q[i + 1] === undefined ? q[i] : `${q[i]} ${q[i + 1]}`);
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
