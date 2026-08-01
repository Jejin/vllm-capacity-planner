# LLM Sizing Calculator — Implementation Handoff

**Product:** <https://llm-sizing.vercel.app/>  
**Prepared:** 1 August 2026  
**Audience:** Product, inference engineering, platform engineering, and QA  
**Status:** Implementation-ready; technical validation complete

> **Decision:** Keep the calculator positioned as a first-pass estimator until P0 command-generation and runtime-compatibility defects are resolved. Memory feasibility is directionally strong; generated deployment commands are not yet safe to execute.

## 1. Executive summary

The calculator has a credible architecture-aware memory model. It handles GQA, MLA, local attention, recurrent state, quantization metadata, MoE resident weights, tensor-parallel packing, replicas, and GiB/TB unit conversions. The default Llama 3.3 70B example reconciles mathematically.

The principal delivery risk is the gap between an abstract sizing choice and an executable deployment. A selected precision is not consistently tied to a real checkpoint, supported kernel, calibrated KV-cache configuration, or valid vLLM command. Performance and cost are roofline estimates and require clearer confidence handling.

| Area | Assessment | Release implication |
|---|---|---|
| Memory feasibility | Strong first-pass model | Retain; add measured-profile override |
| GPU count/topology | Useful within one NVLink node | Retain; constrain cross-node claims |
| Launch command | Invalid or incomplete for common selections | P0 release blocker |
| Throughput/TTFT | Directional roofline only | Require ranges and confidence tiers |
| Cost | Comparison metric, not budget forecast | Add utilization and infrastructure scope |
| Procurement use | Not yet suitable | Block “validated” or “production-ready” language |

### 1.1 How to improve the ratings

The scores improve by closing two different gaps. The estimator needs observed runtime data and workload-aware performance calibration. The deployment generator needs artifact-aware, version-aware commands that are continuously proven to start successfully.

| Target | Changes required | Measurable gate |
|---|---|---|
| **Estimator: 7 → 8** | Add measured vLLM memory-profile import; separate memory fit from runtime support; replace point performance values with ranges and confidence tiers. | At least 90% of representative fixtures predict available KV capacity within ±10% of observed startup profiles. |
| **Estimator: 8 → 9** | Calibrate dense and MoE throughput by GPU family, TP topology and workload distribution; model queueing and percentile SLOs. | P50/P95 TTFT, ITL and throughput are within declared confidence bands on the benchmark matrix. |
| **Deployment: 4 → 6** | Use real model IDs; bind every precision to a checkpoint or explicit online quantization method; enforce an FP8 KV-scale policy. | Every copied command resolves the intended artifact and matches selected precision in contract tests. |
| **Deployment: 6 → 8** | Add a versioned GPU/runtime/quantization compatibility matrix; fail closed; test one- and multi-GPU startup paths. | All supported catalog variants pass startup and one-request smoke tests; unsupported paths cannot be exported. |
| **Deployment: 8 → 9** | Generate complete deployment artifacts—container/runtime pin, replica topology, health checks and reproducibility manifest—and continuously revalidate them. | A clean-environment CI job deploys the generated bundle, serves a request, and records the exact environment and measured capacity. |

**Recommended sequence:** Complete the three P0 controls first: valid artifact IDs, executable precision semantics, and compatibility/FP8-KV safety. These changes produce the largest credibility gain and move the deployment generator from 4/10 to roughly 6/10 before deeper benchmark work.

## 2. Scope and target outcomes

This handoff covers changes needed to make recommendations reproducible, executable, and appropriately qualified. It does not require turning the calculator into a full benchmark service.

- Every feasible result maps to a real model artifact and a runtime-supported launch path.
- Memory recommendations can be reconciled with vLLM startup profiling.
- Unsupported GPU × quantization × runtime combinations fail closed.
- Throughput, TTFT, and cost display confidence and ranges instead of false precision.
- QA can verify every catalog entry using deterministic fixtures and smoke tests.

## 3. Prioritized delivery plan

| Priority | Workstream | Outcome | Exit gate |
|---|---|---|---|
| P0 | Artifact-aware command generation | Commands resolve and match selected precision | All seeded models pass command contract tests |
| P0 | Compatibility validation | Unsupported combinations are blocked | Matrix enforced in UI and sizing engine |
| P0 | FP8 KV scale safety | No silent uncalibrated FP8 path | Scale source shown; warning or calibration required |
| P1 | Measured vLLM profile mode | Heuristic memory can be replaced by observed bytes | Imported profile reconciles allocation |
| P1 | Confidence-aware performance | MoE/TP/network uncertainty is explicit | Ranges and confidence tier shown |
| P1 | Workload distribution inputs | Sizing reflects realistic prompt/output demand | P50/P95 or custom distribution supported |
| P2 | Cost model expansion | Cost reflects duty cycle and platform overhead | Assumptions visible and exportable |

## 4. P0 requirements — executable deployments

### 4.1 Separate model identity from display name

Add immutable deployment metadata to every model/quantization variant. The user-facing label must never be passed directly to vLLM.

| Field | Required behavior | Example |
|---|---|---|
| `display_name` | UI only | Llama 3.3 70B Instruct |
| `model_id` | Resolvable repository or approved local artifact | `meta-llama/Llama-3.3-70B-Instruct` |
| `revision` | Pinned tag/commit where reproducibility matters | Optional SHA |
| `quantization_source` | `checkpoint`, `online`, or `none` | `online` |
| `quantization_method` | Runtime-recognized method | `fp8_per_tensor` |
| `runtime` | Supported engine and version range | vLLM validated version range |
| `trust_remote_code` | Explicit, never silently assumed | `false` |

**Acceptance criterion:** Copying the generated command into a clean supported environment must resolve the selected artifact. Display names, spaces, and marketing labels must never appear as the positional model argument.

### 4.2 Make precision selection executable

Treat precision as a deployment variant, not merely a bytes-per-parameter input. Each option must resolve to one path:

1. **Pre-quantized checkpoint:** `model_id` points to an artifact whose metadata declares the quantization. Do not add a conflicting flag.
2. **Online quantization:** base checkpoint plus an explicit supported `--quantization` method.
3. **Native precision:** checkpoint dtype and runtime behavior already match the estimate.
4. **Unsupported:** block sizing output and explain the missing runtime/checkpoint path.

Illustrative command:

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 4 \
  --quantization fp8_per_tensor \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 35
```

The exact method must come from the validated runtime/version matrix; this example is not a universal substitute for checkpoint-specific configuration.

### 4.3 FP8 KV-cache scale policy

When FP8 KV is selected, resolve and display the scale source. vLLM may otherwise use 1.0 scales when checkpoint scales are absent.

| Scale source | Planner behavior | User message |
|---|---|---|
| Checkpoint metadata | Generate FP8 KV command | Using checkpoint-provided K/V scales |
| Calibrated artifact | Generate command and identify calibration | Dataset-calibrated scales |
| Runtime calculation | Generate supported calculation option | Warm-up-calculated scales; lower assurance |
| None/unknown | Warn prominently or block production mode | Uncalibrated scales may reduce quality |

### 4.4 Compatibility engine

Feasibility must combine memory fit with deployability. Maintain a versioned matrix keyed by GPU architecture, runtime/backend, checkpoint format, quantization method, KV dtype, tensor-parallel size, and node topology.

- Fail closed when support is unknown.
- Explain the failed constraint and name the nearest supported alternative.
- Distinguish “fits in HBM” from “validated runtime path.”
- Flag consumer-GPU PCIe/P2P limitations and cross-node tensor parallelism.
- Record the source and last-verified runtime version for each capability.

## 5. P1 requirements — memory fidelity

### 5.1 Provide estimate and measured modes

Rename the current field to **Estimated runtime overhead**.

| Mode | Use | Calculation |
|---|---|---|
| Estimate | Pre-deployment exploration | Structural activation heuristic plus conservative overhead floor |
| Measured | Deployment validation | Use vLLM-reported weights, peak activation, non-framework memory, graph memory, and available KV bytes |

Recommended measured-profile fields:

- `total_memory_bytes`
- `model_memory_bytes`
- `peak_activation_bytes`
- `non_framework_bytes`
- `graph_memory_bytes`
- `available_kv_cache_bytes`
- `runtime_version`
- `gpu_sku` and driver version

**Reconciliation rule:** Measured `available_kv_cache_bytes` becomes authoritative for concurrency. Preserve the estimate and show variance; do not silently blend observed and estimated memory.

### 5.2 Preserve the sound deterministic formulas

- GQA KV/token = 2 × layers × KV heads × head dimension × element bytes.
- MLA KV uses checkpoint latent geometry; do not generalize 576 as universal.
- Local-attention layers cap token-indexed KV at the declared window.
- Recurrent/linear state is fixed per sequence rather than linear in context.
- MoE resident weight memory uses total parameters, not active parameters.
- TP selection minimizes total committed GPUs and breaks ties toward smaller TP.

## 6. P1 requirements — performance and topology

### 6.1 Confidence tiers

| Tier | Criteria | Presentation |
|---|---|---|
| Measured | Matching model, artifact, GPU, TP, runtime and workload benchmark | Point plus sample size and environment |
| Calibrated | Roofline adjusted against an adjacent measured configuration | Range with interpolation explanation |
| Estimated | Analytical model only | Wide range and validation warning |
| Unsupported | Cross-node, format, or backend path not modeled | Suppress numeric commitment |

### 6.2 Correct MoE decode modeling

Per-token active parameters understate memory traffic when a batch routes across a broader union of experts. Add an expert-coverage term driven by active batch size, experts per token, total experts, routing skew, and reuse.

**Minimum safe implementation:** If expert-union coverage is not modeled, show memory feasibility but suppress precise MoE throughput and cost per token.

### 6.3 Model communication before scaling bandwidth linearly

The current `TP × HBM bandwidth` roofline assumes ideal local scaling. Add topology inputs and a collective-communication penalty.

| Topology | Required behavior |
|---|---|
| Single GPU | No collective penalty |
| TP within NVLink/NVSwitch | Apply validated local efficiency curve |
| TP across PCIe-only GPUs | Apply backend-specific penalty and warn |
| TP across nodes | Require fabric bandwidth/latency or suppress numeric performance |
| Replicas across nodes | Scale independently; no TP collective between replicas |

### 6.4 Separate latency metrics

- **TTFT:** prompt-prefill latency, conditioned on prompt length and concurrent prefills.
- **ITL:** decode inter-token latency under active batch size.
- **Aggregate output throughput:** output tokens/second across replicas.
- **End-to-end latency:** TTFT + generated tokens × ITL, with optional queueing.

## 7. P1/P2 requirements — workload and cost

Replace ambiguous “context utilization” with an admitted-request workload profile:

- Prompt-token P50, P95, maximum, or histogram.
- Output-token P50, P95, or histogram.
- Simultaneous concurrency or arrival rate plus queue/SLO target.
- Prefix-cache hit rate and prefix-length distribution.
- TTFT and ITL percentile objectives.
- User-defined operational reserve.

Cost changes:

- Show a range derived from the throughput range.
- Add sustained utilization/duty cycle.
- Separate input, cached-input, and output tokens when relevant.
- Expose excluded host, storage, network, load-balancer, orchestration, observability, and redundancy costs.
- Add N+1 or user-defined failure headroom.

## 8. UX changes

| Surface | Change |
|---|---|
| Result status | Two badges: **Memory fit** and **Runtime support** |
| Command block | Show artifact ID, runtime/version, quantization source, KV-scale source, and replica count |
| Memory chart | Identify estimated versus measured runtime overhead |
| Performance cards | Show range, confidence, and linked assumptions |
| Cross-node topology | Persistent warning; suppress unsupported numerical claims |
| Cost card | Range, duty cycle, exclusions, currency and rate timestamp |
| Export | Include assumptions, sources, versions, and warnings |

## 9. Acceptance test plan

### 9.1 Command contract tests

- Every seeded variant has a non-empty `model_id` and supported runtime path.
- Generated positional model argument equals `model_id`, never `display_name`.
- Checkpoint and online quantization are mutually exclusive.
- Selected quantization changes memory sizing and command semantics.
- FP8 KV exposes a scale source; unknown source triggers the warning/block.
- The command describes one replica and separately states replica count.

### 9.2 Numerical fixtures

- Llama 3.3 70B FP8 KV: 163,840 bytes/token from 80 × 8 × 128 geometry.
- MoE: resident weights use total parameters; decode accounts for expert coverage.
- MLA: KV uses declared latent width.
- Sliding window: windowed KV stops growing beyond the window.
- Hybrid recurrent model: recurrent state does not scale with context.
- TP ladder: minimize total GPUs and favor smaller TP on ties.
- Measured profile: imported KV bytes determine concurrency exactly.
- Cross-node TP: performance is penalized or suppressed, never ideal-linear.

### 9.3 Runtime smoke tests

For each supported deployment class, run the generated command and capture successful model resolution, engine startup, reported KV capacity, one short request, a maximum-admitted-sequence stress case, and absence of unexpected preemption/OOM.

## 10. Rollout and ownership

| Phase | Owner | Deliverable | Gate |
|---|---|---|---|
| 1. Metadata contract | Model catalog + inference | Artifact and compatibility schema | 100% seeded coverage |
| 2. Safe generation | Inference + frontend | Executable commands and FP8 KV policy | Command tests green |
| 3. Memory validation | Runtime/platform | Measured-profile import and reconciliation | Representative profiles match |
| 4. Performance confidence | Performance engineering | MoE/topology calibration and ranges | Benchmarked hardware classes |
| 5. Production positioning | Product + QA | Updated labels, exports and documentation | No unqualified procurement claims |

## 11. Definition of done

- Every recommendation is traceable to model artifact, GPU specification, runtime version, and formula or benchmark.
- Every copied command aligns syntactically and semantically with the selected variant.
- Unsupported configurations fail closed with an actionable alternative.
- Measured vLLM memory overrides heuristic KV capacity without hidden transformations.
- MoE and cross-node performance are not presented with unsupported precision.
- Exports contain enough information to reproduce the plan.

## 12. References

- [vLLM GPU worker memory accounting](https://docs.vllm.ai/en/stable/api/vllm/v1/worker/gpu_worker/)
- [vLLM online quantization](https://docs.vllm.ai/en/stable/features/quantization/online/)
- [vLLM quantized KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
- [vLLM performance tuning](https://docs.vllm.ai/en/v0.22.1/configuration/optimization/)
- [vLLM distributed serving](https://docs.vllm.ai/en/v0.6.6.post1/serving/distributed_serving.html)
- [NVIDIA HGX H200/B200/B300 specifications](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)
- [Meta Llama 3.3 70B model card](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)

## Appendix A — limitations that must remain visible

- Analytical throughput is not a substitute for benchmark data.
- Actual memory depends on runtime version, kernels, graph capture, allocator state, other processes, and checkpoint implementation.
- Maximum context support does not imply acceptable latency at that context.
- A GPU fitting the weights does not imply the selected quantization kernel is supported or performant.
- Per-GPU rental price excludes non-GPU platform costs unless explicitly entered.
