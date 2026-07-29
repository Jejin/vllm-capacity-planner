<script lang="ts">
  // vLLM Capacity Planner SPA. Sizing engine runs client-side (AD-1/AD-2); catalog,
  // reconciliation and saved configs go through the server API. Fleet+plan are session state.
  import { computeSizing, concurrencySweep, seedCatalog, kvPerTokenBytes, weightsGb, serveCommand, topologyLayout, topologySvg, QUANTS, type Model, type GpuSku, type FeasibleSizing } from '@vcp/domain';

  type Ident = { sub: string; role: 'admin' | 'user' };
  let ident = $state<Ident>({ sub: 'u-rana', role: 'user' });
  const authH = $derived({ 'x-dev-sub': ident.sub, 'x-dev-role': ident.role, 'content-type': 'application/json' });

  let catalog = $state(seedCatalog());
  let tab = $state<'sizing' | 'fleet' | 'recon' | 'plan' | 'configs' | 'catalog' | 'methodology'>('sizing');

  const CTXS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];
  let modelId = $state('llama33-70b'), gpuId = $state('h200');
  let quant = $state<any>('FP8'), kvBytes = $state(1), ctx = $state(131072);
  let util = $state(0.6), conc = $state(64), memUtil = $state(0.9), perNode = $state(8);

  type Pool = { gpu_sku_id: string; gpus_per_node: number; node_count: number };
  type Deployment = { id: string; label: string; gpu_sku_id: string; gpus: number; pods: number; tp: number; kv: number; tps: number };
  let fleet = $state<Pool[]>([]);
  let plan = $state<Deployment[]>([]);
  let configs = $state<{ id: string; name: string; updated_at: string }[]>([]);
  let notice = $state('');

  let activeConfig = $state<string | null>(null); // name of the loaded saved config, or null if fleet is manual/empty
  async function loadCatalog() { try { const r = await fetch('/api/v1/catalog', { headers: authH }); if (r.ok) catalog = await r.json(); } catch {} }
  $effect(() => { loadCatalog(); });
  $effect(() => { if (tab === 'configs' || tab === 'sizing') refreshConfigs(); }); // configs available for the sizing-tab picker
  // When the model changes, populate the model-specific deployment defaults from it:
  // preferred quant (FP8 if offered, else the model's first) and its default serving context
  // (min(131072, max_ctx), per addendum §B.3). Deployment-requirement inputs (GPU, concurrency,
  // utilisation, GPUs/node) are left as the user set them.
  let lastModelId = $state(modelId);
  $effect(() => {
    if (model && modelId !== lastModelId) {
      lastModelId = modelId;
      quant = model.quants.includes('FP8') ? 'FP8' : model.quants[0];
      ctx = Math.min(131072, model.max_ctx);
    }
  });

  let batchTokens = $state(2048); // vLLM --max-num-batched-tokens; drives the activation reserve
  const model = $derived<Model>(catalog.models.find((m: Model) => m.id === modelId) ?? catalog.models[0]);
  const gpu = $derived<GpuSku>(catalog.gpus.find((g: GpuSku) => g.id === gpuId) ?? catalog.gpus[0]);
  const ctxChoices = $derived(CTXS.filter((c) => c <= (model?.max_ctx ?? 0)));
  // Effective quant — always one the selected model actually supports, so switching models can
  // never compute with an unsupported quant (e.g. FP16 on GLM-5.2, which only offers FP8/NVFP4).
  const effQuant = $derived(model && model.quants.includes(quant) ? quant : (model?.quants[0] ?? 'FP8'));
  // named so the launch-command builder reuses the exact inputs the sizing ran on
  const sizingInput = $derived({
    quant: effQuant, kv_dtype_bytes: kvBytes, selected_ctx: Math.min(ctx, model?.max_ctx ?? ctx),
    avg_context_utilisation: util, target_concurrency: conc, mem_util_fraction: memUtil, gpus_per_node: perNode,
    max_num_batched_tokens: batchTokens,
  });
  const result = $derived(computeSizing(model, gpu, sizingInput));
  const R = $derived(result.ok ? (result as FeasibleSizing) : null);
  const fmt = (x: number, d = 1) => (x >= 1000 ? Math.round(x).toLocaleString() : x.toFixed(d));
  /** TTFT spans milliseconds to tens of seconds; show whichever unit is readable. */
  const ttftLabel = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`);

  // per-replica HBM split (matches the prototype)
  const activePer = $derived(R ? Math.min(R.concurrency_per_pod, Math.ceil(conc / R.pods)) : 0);
  // Per-GPU HBM, decomposed so the parts sum to the physical card. The old chart drew
  // "everything that isn't weights or KV" and called it Reserve — silently merging the runtime
  // reserve, the gpu-memory-utilization headroom, and unallocated KV space. The middle one is
  // the largest and the only one the operator can reclaim, so hiding it is the worst of three.
  const wPer = $derived(R ? R.weights_gb / R.tp : 0);
  const kvPer = $derived(R ? (activePer * R.kv_per_request_gb) / R.tp : 0);
  /** Usable-but-unallocated KV space — room for more sessions at this exact config. */
  const kvFreePer = $derived(R ? Math.max(0, R.free_gb / R.tp - kvPer) : 0);
  /** vLLM's own overhead: CUDA context, collectives, prefill activations. */
  const reservePer = $derived(R ? R.runtime_reserve_gb : 0);
  /** Withheld by gpu-memory-utilization — recoverable by raising it, at OOM risk. */
  const utilHeadroomPer = $derived(gpu ? gpu.mem_gb * (1 - memUtil) : 0);
  const kvAlloc = $derived(R ? activePer * R.kv_per_request_gb : 0);
  const stacks = $derived(R ? Math.min(R.tp, 8) : 0);

  // ── Deployment topology ──
  // Geometry lives in the domain package so it can be unit-tested and rendered headlessly;
  // a diagram whose layout only exists in a template can only be checked by eye, in a browser.
  const topo = $derived(R ? topologyLayout(R, perNode) : null);
  /** The node cut-off landed inside a TP group, so one bar is drawn open-ended. */
  const topoCut = $derived(!!topo && !!R && topo.pods.some((p) => p.gpusShown < R.tp));
  const topoSvg = $derived(
    topo && R
      ? topologySvg(topo, {
          tp: R.tp,
          perNode,
          multiNode: R.multi_node,
          storeLabel: `shared weights · ${fmt(R.weights_gb)} GiB per replica`,
          desc: `${R.pods} serving replica${R.pods > 1 ? 's' : ''}, each sharded across ${R.tp} GPUs, placed on ${R.nodes} node${R.nodes > 1 ? 's' : ''} of ${perNode} GPUs. ${R.multi_node ? 'At least one replica spans a node boundary, so its tensor-parallel collective crosses the inter-node fabric.' : 'Every replica fits inside a single node.'}${topo.truncated ? ` The diagram draws only the first ${topo.shown} nodes and ${topo.shownPods} replicas; the rest repeat the same pattern.` : ''}`,
        })
      : '',
  );

  // launch command for ONE replica of the current plan
  let cmdDocker = $state(false);
  let cmdCopied = $state(false);
  /** Catalogue entries carry display names; only use one as a serve id if it looks like one. */
  const hfIdFor = (m: Model) => (/^[\w.-]+\/[\w.-]+$/.test(m.name) ? m.name : undefined);
  const serveCmd = $derived(
    R && model ? serveCommand(model, sizingInput, R, { hf_id: hfIdFor(model), docker: cmdDocker }) : null,
  );
  async function copyCmd() {
    if (!serveCmd) return;
    try { await navigator.clipboard.writeText(serveCmd.command); cmdCopied = true; setTimeout(() => (cmdCopied = false), 1600); }
    catch { notice = 'Clipboard unavailable — select the command and copy manually.'; }
  }
  // what this model's KV would cost if every layer were full-context — the comparison the
  // sliding-window banner quotes, computed exactly rather than scaled from the layer ratio
  const kvNominalGb = $derived(
    model ? (kvPerTokenBytes(model, kvBytes) * ctx * util) / 2 ** 30 : 0,
  );

  const pctOf = (n: number) => `${Math.max(0, Math.min(100, (n / (gpu?.mem_gb ?? 1)) * 100)).toFixed(1)}%`;

  // Concurrency rubric — sweep target concurrency at the current config (FR-12 / metrics).
  const SWEEP = [1, 8, 16, 32, 64, 128, 256];
  const sweep = $derived(model && gpu ? concurrencySweep(model, gpu, { quant: effQuant, kv_dtype_bytes: kvBytes, selected_ctx: Math.min(ctx, model.max_ctx), avg_context_utilisation: util, mem_util_fraction: memUtil, gpus_per_node: perNode }, SWEEP) : []);

  // fleet check on the sizing view (uses session fleet + committed plan)
  const fleetCheck = $derived(() => {
    if (!R || fleet.length === 0) return null;
    const avail = fleet.filter((p) => p.gpu_sku_id === gpuId).reduce((s, p) => s + p.gpus_per_node * p.node_count, 0);
    if (avail === 0) return { kind: 'absent' as const };
    const used = plan.filter((d) => d.gpu_sku_id === gpuId).reduce((s, d) => s + d.gpus, 0);
    const head = avail - used;
    if (R.gpus > head) return { kind: 'short' as const, head, need: R.gpus, short: R.gpus - head, nodes: Math.ceil((R.gpus - head) / perNode) };
    return { kind: 'fit' as const, head, maxPods: Math.floor(head / R.tp), concEach: R.concurrency_per_pod };
  });

  function addToPlan() {
    if (!R) return;
    const fc = fleetCheck();
    if (fc?.kind === 'absent') { notice = `Fleet check: ${gpu.name} is not in the defined fleet — add a pool first.`; return; }
    if (fc?.kind === 'short') { notice = `⛔ Over capacity: needs ${R.gpus} ${gpu.name}, only ${fc.head} uncommitted. Short by ${fc.short}. Cannot add.`; return; }
    plan.push({ id: 'd' + Math.round(performance.now()), label: `${model.name} · ${effQuant}`, gpu_sku_id: gpuId, gpus: R.gpus, pods: R.pods, tp: R.tp, kv: kvAlloc, tps: R.throughput_tokens_per_sec });
    notice = `Added ${model.name} (${R.gpus} × ${gpu.name}) to the cluster plan.`;
  }

  const fleetTotals = $derived(() => {
    const byId = new Map(catalog.gpus.map((g: GpuSku) => [g.id, g]));
    return { gpus: fleet.reduce((s, p) => s + p.gpus_per_node * p.node_count, 0), nodes: fleet.reduce((s, p) => s + p.node_count, 0),
      hbm: fleet.reduce((s, p) => s + p.gpus_per_node * p.node_count * ((byId.get(p.gpu_sku_id) as GpuSku)?.mem_gb ?? 0), 0) };
  });
  const planTotals = $derived(() => ({ gpus: plan.reduce((s, d) => s + d.gpus, 0), pods: plan.reduce((s, d) => s + d.pods, 0), kv: plan.reduce((s, d) => s + d.kv, 0) }));
  // Cluster = fleet (supply) reconciled against the plan's committed deployments (demand), per SKU.
  // Committed HBM is physical whole-GPU HBM (§G). Used for the utilization visual + capacity check.
  const clusterBySku = $derived(() => {
    const byId = new Map(catalog.gpus.map((g: GpuSku) => [g.id, g]));
    const skus = [...new Set([...fleet.map((p) => p.gpu_sku_id), ...plan.map((d) => d.gpu_sku_id)])];
    return skus.map((sku) => {
      const g = byId.get(sku) as GpuSku | undefined;
      const mem = g?.mem_gb ?? 0;
      const total = fleet.filter((p) => p.gpu_sku_id === sku).reduce((s, p) => s + p.gpus_per_node * p.node_count, 0);
      const committed = plan.filter((d) => d.gpu_sku_id === sku).reduce((s, d) => s + d.gpus, 0);
      return { sku, name: g?.name ?? sku, mem, total, committed, free: total - committed, totalHbm: total * mem, committedHbm: committed * mem, util: total > 0 ? (committed / total) * 100 : committed > 0 ? Infinity : 0, over: committed > total };
    });
  });
  const clusterTotals = $derived(() => { const r = clusterBySku(); const totalG = r.reduce((s, x) => s + x.total, 0), committedG = r.reduce((s, x) => s + x.committed, 0); return { totalG, committedG, freeG: r.reduce((s, x) => s + Math.max(0, x.free), 0), util: totalG > 0 ? (committedG / totalG) * 100 : 0, over: r.some((x) => x.over) }; });

  // Cost estimate — GPU-hour rental (admin-set price per SKU). Committed GPUs × $/hr.
  const money = (x: number) => (x >= 1000 ? '$' + Math.round(x).toLocaleString() : '$' + x.toFixed(2));
  const cost = $derived(() => {
    const byId = new Map(catalog.gpus.map((g: GpuSku) => [g.id, g]));
    const lines = clusterBySku().filter((r) => r.committed > 0).map((r) => {
      const price = (byId.get(r.sku) as GpuSku)?.price_per_gpu_hour ?? 0;
      return { name: r.name, gpus: r.committed, price, hr: r.committed * price };
    });
    const totalHr = lines.reduce((s, l) => s + l.hr, 0);
    return { lines, totalHr, totalMo: totalHr * 730, totalYr: totalHr * 8760, priced: lines.some((l) => l.price > 0) };
  });
  // Per-model economics — $/million tokens = ($/hr) / (tokens per hour / 1e6).
  const modelEconomics = $derived(() => {
    const byId = new Map(catalog.gpus.map((g: GpuSku) => [g.id, g]));
    return plan.map((d) => {
      const price = (byId.get(d.gpu_sku_id) as GpuSku)?.price_per_gpu_hour ?? 0;
      const hr = d.gpus * price;
      const tokPerHr = (d.tps ?? 0) * 3600;
      return { label: d.label, gpus: d.gpus, hr, tps: d.tps ?? 0, perMtok: tokPerHr > 0 && price > 0 ? (hr * 1e6) / tokPerHr : 0 };
    });
  });

  // Export the cost estimate — CSV (spreadsheet) + JSON (full scenario), client-side download.
  function download(name: string, content: string, type: string) {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const a = document.createElement('a');
    a.href = url; a.download = name; a.click();
    URL.revokeObjectURL(url);
  }
  const csvCell = (s: string) => (/[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s);
  const fileBase = () => `cost-estimate-${(activeConfig ?? 'cluster').replace(/[^\w-]+/g, '_')}`;
  function exportCsv() {
    const ct = clusterTotals(); const c = cost();
    const rows: string[][] = [
      ['vLLM Capacity Planner — Cost Estimate'], ['Scenario', activeConfig ?? '(unsaved)'], ['Generated', new Date().toISOString()], [],
      ['Cluster'], ['Fleet GPUs', String(ct.totalG)], ['Committed GPUs', String(ct.committedG)], ['Free GPUs', String(ct.freeG)], ['Utilisation %', ct.util.toFixed(1)], [],
      ['Cost by GPU SKU (rental)'], ['SKU', 'Committed GPUs', '$/GPU-hr', '$/hr', '$/mo', '$/yr'],
      ...c.lines.map((l) => [l.name, String(l.gpus), l.price.toFixed(2), l.hr.toFixed(2), (l.hr * 730).toFixed(2), (l.hr * 8760).toFixed(2)]),
      ['TOTAL', '', '', c.totalHr.toFixed(2), c.totalMo.toFixed(2), c.totalYr.toFixed(2)], [],
      ['Per-model economics'], ['Model', 'GPUs', '$/hr', 'tok/s', '$/Mtok'],
      ...modelEconomics().map((e) => [e.label, String(e.gpus), e.hr.toFixed(2), String(e.tps), e.perMtok > 0 ? e.perMtok.toFixed(2) : '']),
    ];
    download(`${fileBase()}.csv`, rows.map((r) => r.map(csvCell).join(',')).join('\n'), 'text/csv');
  }
  function exportJson() {
    const c = cost();
    const data = { generated: new Date().toISOString(), scenario: activeConfig, cluster: clusterTotals(), fleet, plan, cost: { basis: 'gpu-hour', per_sku: c.lines, total_per_hour: c.totalHr, total_per_month: c.totalMo, total_per_year: c.totalYr }, model_economics: modelEconomics() };
    download(`${fileBase()}.json`, JSON.stringify(data, null, 2), 'application/json');
  }

  let newPool = $state({ gpu_sku_id: 'h200', gpus_per_node: 8, node_count: 1 });
  function addPool() { fleet.push({ ...newPool }); activeConfig = null; }
  function delPool(i: number) { fleet.splice(i, 1); activeConfig = null; }
  function delDeployment(i: number) { plan.splice(i, 1); }

  async function refreshConfigs() { const r = await fetch('/api/v1/configs', { headers: authH }); if (r.ok) configs = await r.json(); }
  let saveName = $state('');
  async function saveConfig() {
    if (!saveName) return;
    const snapshot = { fleet, plan, geometry: { models: catalog.models, gpus: catalog.gpus } };
    const r = await fetch('/api/v1/configs', { method: 'POST', headers: authH, body: JSON.stringify({ name: saveName, snapshot }) });
    if (r.ok) { saveName = ''; refreshConfigs(); notice = 'Configuration saved.'; }
  }
  async function loadConfig(id: string, goToFleet = true) { const r = await fetch(`/api/v1/configs/${id}`, { headers: authH }); if (r.ok) { const c = await r.json(); fleet = c.snapshot.fleet ?? []; plan = c.snapshot.plan ?? []; activeConfig = c.name; notice = `Loaded "${c.name}".`; if (goToFleet) tab = 'fleet'; } }
  async function delConfig(id: string) { await fetch(`/api/v1/configs/${id}`, { method: 'DELETE', headers: authH }); refreshConfigs(); }
  async function resetCatalog() { if (!confirm('Reset the catalog to seeded defaults? This replaces all models and GPUs.')) return; await fetch('/api/v1/catalog/reset', { method: 'POST', headers: authH }); loadCatalog(); notice = 'Catalog reset to seeded defaults.'; }

  // ── Admin catalog forms (FR-2/3/4/6/7/8, validated via the shared §F schema) ──
  // QUANTS comes from the domain package — a local copy silently went stale and stopped
  // offering the GGUF k-quants after they were added to the engine.
  type Err = { path: string; message: string };
  const blankModel = () => ({ id: '', name: '', total_params_b: 1, active_params_b: 1, layers: 32, kv_heads: 8, head_dim: 128, mla: false, max_ctx: 131072, tp_options: '1,2', quants: ['FP16'] as string[], hidden_size: '' as number | '', vocab_size: '' as number | '', tied_embeddings: false, sliding_window: '' as number | '', full_attention_layers: '' as number | '', linear_attention_layers: '' as number | '', linear_state_bytes_per_layer: '' as number | '' });
  let mf = $state(blankModel());
  let mfEditing = $state(false);
  let mfErrors = $state<Err[]>([]);
  const errFor = (errs: Err[], p: string) => errs.find((e) => e.path === p)?.message;
  function editModel(m: Model) { mf = { ...m, tp_options: m.tp_options.join(','), quants: [...m.quants], hidden_size: m.hidden_size ?? '', vocab_size: m.vocab_size ?? '', tied_embeddings: !!m.tied_embeddings, sliding_window: m.sliding_window ?? '', full_attention_layers: m.full_attention_layers ?? '', linear_attention_layers: m.linear_attention_layers ?? '', linear_state_bytes_per_layer: m.linear_state_bytes_per_layer ?? '' } as any; mfEditing = true; mfErrors = []; document.getElementById('mform')?.scrollIntoView({ behavior: 'smooth' }); }
  function newModelForm() { mf = blankModel(); mfEditing = false; mfErrors = []; }
  function toggleQuant(q: string) { mf.quants = mf.quants.includes(q) ? mf.quants.filter((x) => x !== q) : [...mf.quants, q]; }
  async function saveModel() {
    // hidden_size/vocab_size are optional — send them only when both are filled in, so a blank
    // pair stays undefined (engine falls back) rather than becoming 0 and failing validation.
    const emb = mf.hidden_size !== '' && mf.vocab_size !== ''
      ? { hidden_size: +mf.hidden_size, vocab_size: +mf.vocab_size, tied_embeddings: !!mf.tied_embeddings }
      : {};
    // same all-or-nothing rule for the sliding-window pair
    const win = mf.sliding_window !== '' && mf.full_attention_layers !== ''
      ? { sliding_window: +mf.sliding_window, full_attention_layers: +mf.full_attention_layers }
      : (mf.full_attention_layers !== '' ? { full_attention_layers: +mf.full_attention_layers } : {});
    // linear/recurrent layers — constant state, sent as a pair
    const lin = mf.linear_attention_layers !== '' && mf.linear_state_bytes_per_layer !== ''
      ? { linear_attention_layers: +mf.linear_attention_layers, linear_state_bytes_per_layer: +mf.linear_state_bytes_per_layer }
      : {};
    const body = { id: mf.id, name: mf.name, total_params_b: +mf.total_params_b, active_params_b: +mf.active_params_b, layers: +mf.layers, kv_heads: +mf.kv_heads, head_dim: +mf.head_dim, mla: mf.mla, max_ctx: +mf.max_ctx, tp_options: String(mf.tp_options).split(',').map((s) => parseInt(s.trim(), 10)).filter((n) => n > 0), quants: mf.quants, ...emb, ...win, ...lin };
    const r = await fetch(mfEditing ? `/api/v1/models/${mf.id}` : '/api/v1/models', { method: mfEditing ? 'PUT' : 'POST', headers: authH, body: JSON.stringify(body) });
    if (r.ok) { newModelForm(); loadCatalog(); notice = 'Model saved.'; }
    else { const e = await r.json(); mfErrors = e.error?.fields ?? [{ path: '', message: e.error?.message ?? 'Save failed.' }]; }
  }
  async function deleteModelUi(id: string) { if (!confirm(`Delete model "${id}"?`)) return; const r = await fetch(`/api/v1/models/${id}`, { method: 'DELETE', headers: authH }); if (r.ok) { loadCatalog(); notice = 'Model deleted.'; } else notice = (await r.json()).error?.message ?? 'Delete failed.'; }

  const blankGpu = () => ({ id: '', name: '', mem_gb: 80, bw_tbs: 3, price_per_gpu_hour: 2.5 });
  let gf = $state(blankGpu());
  let gfErrors = $state<Err[]>([]);
  async function saveGpu() {
    const body = { id: gf.id, name: gf.name, mem_gb: +gf.mem_gb, bw_tbs: +gf.bw_tbs, price_per_gpu_hour: +gf.price_per_gpu_hour };
    const r = await fetch('/api/v1/gpus', { method: 'POST', headers: authH, body: JSON.stringify(body) });
    if (r.ok) { gf = blankGpu(); gfErrors = []; loadCatalog(); notice = 'GPU SKU saved.'; }
    else { const e = await r.json(); gfErrors = e.error?.fields ?? [{ path: '', message: e.error?.message ?? 'Save failed.' }]; }
  }
  async function deleteGpuUi(id: string) { if (!confirm(`Delete GPU SKU "${id}"?`)) return; const r = await fetch(`/api/v1/gpus/${id}`, { method: 'DELETE', headers: authH }); if (r.ok) { loadCatalog(); notice = 'GPU SKU deleted.'; } else notice = (await r.json()).error?.message ?? 'Delete failed.'; }
  const tpNodes = $derived(R ? Math.ceil(R.tp / perNode) : 0);

  // ── Hugging Face import (admin) — fetch config.json, map to §F, prefill the model form (FR-30/31) ──
  let hfId = $state('');
  let hfBusy = $state(false);
  let hfCard = $state<any>(null);
  let hfMissing = $state<string[]>([]);
  const HF_SUGGEST = ['Qwen/Qwen2.5-72B-Instruct', 'Qwen/Qwen3-32B', 'mistralai/Mistral-Small-24B-Instruct-2501', 'deepseek-ai/DeepSeek-V3'];
  async function hfFetch() {
    if (!hfId.trim()) return;
    hfBusy = true; hfCard = null; hfMissing = [];
    const r = await fetch('/api/v1/huggingface/fetch', { method: 'POST', headers: authH, body: JSON.stringify({ model_id: hfId.trim() }) });
    hfBusy = false;
    if (r.ok) {
      const d = await r.json();
      hfCard = d; hfMissing = d.missing ?? [];
      const m = d.mapped ?? {};
      mf = { id: m.id ?? '', name: m.name ?? hfId, total_params_b: 1, active_params_b: 1, layers: m.layers ?? 32, kv_heads: m.kv_heads ?? 8, head_dim: m.head_dim ?? 128, mla: !!m.mla, max_ctx: m.max_ctx ?? 131072, tp_options: '', quants: [], hidden_size: m.hidden_size ?? '', vocab_size: m.vocab_size ?? '', tied_embeddings: !!m.tied_embeddings, sliding_window: m.sliding_window ?? '', full_attention_layers: m.full_attention_layers ?? '', linear_attention_layers: m.linear_attention_layers ?? '', linear_state_bytes_per_layer: m.linear_state_bytes_per_layer ?? '' };
      mfEditing = false; mfErrors = [];
      notice = `Fetched ${d.model_id}. Review below, complete params / TP / quants (highlighted), then Create model.`;
      document.getElementById('mform')?.scrollIntoView({ behavior: 'smooth' });
    } else {
      notice = (await r.json()).error?.message ?? 'Fetch failed.';
    }
  }

  /** Our weight estimate (decimal GB) for the model currently in the admin form, at one quant.
   *  Returns null until the form carries enough geometry to compute anything meaningful. */
  function mfWeightsGb(q: string): number | null {
    const total = +mf.total_params_b;
    if (!Number.isFinite(total) || total <= 1) return null;
    if (!(QUANTS as readonly string[]).includes(q)) return null;
    const probe: Model = {
      id: 'probe', name: 'probe', total_params_b: total, active_params_b: total,
      layers: +mf.layers || 1, kv_heads: mf.mla ? 0 : (+mf.kv_heads || 1),
      head_dim: mf.mla ? 0 : (+mf.head_dim || 1), mla: !!mf.mla,
      max_ctx: +mf.max_ctx || 4096, tp_options: [1], quants: [q as any],
      hidden_size: mf.hidden_size !== '' ? +mf.hidden_size : undefined,
      vocab_size: mf.vocab_size !== '' ? +mf.vocab_size : undefined,
      tied_embeddings: !!mf.tied_embeddings,
    };
    try { return (weightsGb(probe, q as any) * 2 ** 30) / 1e9; } catch { return null; }
  }

  // ── vLLM recipe import — fills the four fields config.json never carries ──
  let recipeBusy = $state(false);
  let recipeInfo = $state<any>(null);
  async function recipeFetch(idOverride?: string) {
    const target = (idOverride ?? hfId).trim();
    if (!target) { notice = 'Enter a model ID first.'; return; }
    recipeBusy = true; recipeInfo = null;
    const r = await fetch('/api/v1/recipes/fetch', { method: 'POST', headers: authH, body: JSON.stringify({ model_id: target }) });
    recipeBusy = false;
    if (!r.ok) { notice = (await r.json()).error?.message ?? 'Recipe fetch failed.'; return; }
    const d = await r.json();
    recipeInfo = d;
    const m = d.mapped ?? {};
    // merge, never clobber geometry the HF import already established
    if (m.total_params_b != null) mf.total_params_b = m.total_params_b;
    if (m.active_params_b != null) mf.active_params_b = m.active_params_b;
    if (m.max_ctx != null) mf.max_ctx = m.max_ctx;
    if (m.tp_options?.length) mf.tp_options = m.tp_options.join(',');
    if (m.quants?.length) mf.quants = m.quants;
    if (!mf.id && m.id) mf.id = m.id;
    if (!mf.name && m.name) mf.name = m.name;
    hfMissing = hfMissing.filter((k) => !(d.filled ?? []).includes(k));
    notice = `Recipe applied: ${(d.filled ?? []).join(', ') || 'nothing new'}.`;
    document.getElementById('mform')?.scrollIntoView({ behavior: 'smooth' });
  }

  function toggleTheme() { const r = document.documentElement; r.setAttribute('data-theme', r.getAttribute('data-theme') === 'dark' ? 'light' : 'dark'); }
  const gpuName = (id: string) => catalog.gpus.find((g: GpuSku) => g.id === id)?.name ?? id;
  const tabDefs: [typeof tab, string, boolean][] = [['sizing', 'Sizing', false], ['fleet', 'Fleet', false], ['plan', 'Cluster', false], ['configs', 'My Configurations', false], ['catalog', 'Models', false], ['methodology', 'Methodology', false]];
</script>

<header>
  <div class="brand">
    <svg class="logo" viewBox="0 0 26 26" aria-hidden="true"><rect x="3" y="14" width="4.5" height="9" rx="1.2" fill="var(--slate)"/><rect x="10.75" y="8" width="4.5" height="15" rx="1.2" fill="var(--purple)"/><rect x="18.5" y="3" width="4.5" height="20" rx="1.2" fill="var(--brand)"/></svg>
    <span class="wm">vLLM</span> <span class="app">Capacity Planner</span>
  </div>
  <div class="right">
    <select class="role" bind:value={ident.role}><option value="user">Standard user</option><option value="admin">Admin</option></select>
    <span class="chip {ident.role}">{ident.role}</span>
    <button onclick={toggleTheme} title="Toggle theme">◐</button>
  </div>
</header>
<nav>
  {#each tabDefs as [id, label, admin]}
    {#if !admin || ident.role === 'admin'}
      <button class="tab" class:active={tab === id} onclick={() => (tab = id)}>{label}{#if admin}<i>admin</i>{/if}</button>
    {/if}
  {/each}
</nav>
{#if notice}<div class="banner" onclick={() => (notice = '')}>{notice}<span>✕</span></div>{/if}

<main>
{#if tab === 'sizing'}
  {#if fleet.length === 0}
    <div class="fleetctx suggest">
      <div><b>Pick a fleet to size against.</b> Load one of your saved configurations before adding deployments — or build a fleet first.</div>
      <div class="ctxactions">
        {#if configs.length}
          <select onchange={(e) => { const id = (e.currentTarget as HTMLSelectElement).value; if (id) loadConfig(id, false); (e.currentTarget as HTMLSelectElement).value = ''; }}>
            <option value="">Load saved configuration…</option>
            {#each configs as c}<option value={c.id}>{c.name}</option>{/each}
          </select>
        {/if}
        <button class="btn ghost" onclick={() => (tab = 'fleet')}>Build fleet →</button>
      </div>
    </div>
  {:else}
    <div class="fleetctx active">
      <span>Sizing against <b>{activeConfig ?? 'a manually-defined fleet'}</b> — {fleetTotals().gpus} GPUs · {(fleetTotals().hbm / 1024).toFixed(1)} TiB HBM</span>
      <div class="ctxactions">
        {#if configs.length}<select onchange={(e) => { const id = (e.currentTarget as HTMLSelectElement).value; if (id) loadConfig(id, false); (e.currentTarget as HTMLSelectElement).value = ''; }}><option value="">Switch configuration…</option>{#each configs as c}<option value={c.id}>{c.name}</option>{/each}</select>{/if}
        <button class="btn ghost" onclick={() => (tab = 'fleet')}>Edit fleet</button>
      </div>
    </div>
  {/if}
  <div class="grid">
    <section class="panel">
      <h2>Deployment inputs</h2>
      <label>Model<select bind:value={modelId}>{#each catalog.models as m}<option value={m.id}>{m.name}</option>{/each}</select></label>
      <div class="meta">total {model?.total_params_b} B · active {model?.active_params_b} B · layers {model?.layers} · {model?.mla ? 'MLA (latent 576/layer)' : `GQA ${model?.kv_heads} KV-heads × ${model?.head_dim}`} · max ctx {((model?.max_ctx ?? 0) / 1024)}K · TP {`{${model?.tp_options.join(',')}}`}{#if model?.vocab_size && model?.hidden_size} · emb {model.vocab_size.toLocaleString()}×{model.hidden_size}{model.tied_embeddings ? ' (tied)' : ''}{/if}{#if model?.sliding_window && model?.full_attention_layers != null} · <b>SWA {model.full_attention_layers}/{model.layers} full</b>, {model.sliding_window}-tok window{/if}{#if model?.linear_attention_layers} · <b>{model.linear_attention_layers} linear</b> (constant state){/if}</div>
      <div class="row"><label>Quantisation<select bind:value={quant}>{#each (model?.quants ?? []) as q}<option value={q}>{q}</option>{/each}</select></label>
        <label>KV dtype<select bind:value={kvBytes}><option value={1}>FP8 (1B)</option><option value={2}>FP16 (2B)</option></select></label></div>
      <div class="row"><label>Max context<select bind:value={ctx}>{#each ctxChoices as c}<option value={c}>{c / 1024}K</option>{/each}</select></label>
        <label>Context util<input type="number" min="0.05" max="1" step="0.05" bind:value={util} /></label></div>
      <div class="row"><label>Target concurrency<input type="number" bind:value={conc} /></label>
        <label>GPU SKU<select bind:value={gpuId}>{#each catalog.gpus as g}<option value={g.id}>{g.name}</option>{/each}</select></label></div>
      <div class="row"><label>GPU mem-util<input type="number" min="0.1" max="1" step="0.05" bind:value={memUtil} /></label>
        <label>GPUs / node<input type="number" bind:value={perNode} /></label></div>
      <div class="row"><label>Prefill chunk<small> (--max-num-batched-tokens)</small><input type="number" min="256" step="256" bind:value={batchTokens} /></label>
        <label>Runtime reserve<small> (derived)</small><input type="text" value={R ? `${R.runtime_reserve_gb.toFixed(2)} GiB` : '—'} disabled /></label></div>
      {#if R}<button class="btn primary full" onclick={addToPlan}>+ Add to cluster plan</button>{/if}
    </section>

    <section>
      {#if R}
        <div class="kpis">
          <div class="kpi" class:tight={R.tight}><div class="v">{R.gpus}</div><div class="l">GPUs</div><div class="verdict" class:t={R.tight}>{R.tight ? `Tight · ${(R.headroom_fraction * 100).toFixed(1)}% free` : `Fits · ${(R.headroom_fraction * 100).toFixed(0)}% free`}</div></div>
          <div class="kpi p"><div class="v">{R.pods}<small> × TP{R.tp}</small></div><div class="l">Pods</div></div>
          <div class="kpi a"><div class="v">{fmt(kvAlloc)}<small> GiB</small></div><div class="l">KV cache</div></div>
          <div class="kpi g"><div class="v">~{R.throughput_tokens_per_sec.toLocaleString()}</div><div class="l">Tokens/s</div><div class="cav">±40% · not a commitment</div></div>
          <div class="kpi t"><div class="v">{ttftLabel(R.ttft_ms)}</div><div class="l">Time to first token</div><div class="cav">±50% · {R.ttft_compute_bound ? 'compute-bound' : 'bandwidth floor'}</div></div>
        </div>

        <div class="panel">
          <h2>Per-GPU HBM allocation — one replica (TP{R.tp}, {R.tp} GPU{R.tp > 1 ? 's' : ''})</h2>
          <div class="hbm">
            {#each Array(stacks) as _, i}
              <div class="gpu">
                <div class="stack">
                  <div class="seg w" style="height:{pctOf(wPer)}" title="Weights {fmt(wPer)} GiB"></div>
                  <div class="seg k" style="height:{pctOf(kvPer)}" title="KV in use {fmt(kvPer)} GiB"></div>
                  <div class="seg kf" style="height:{pctOf(kvFreePer)}" title="KV free {fmt(kvFreePer)} GiB — room for more sessions"></div>
                  <div class="seg r" style="height:{pctOf(reservePer)}" title="Runtime reserve {fmt(reservePer, 2)} GiB"></div>
                  <div class="seg u" style="height:{pctOf(utilHeadroomPer)}" title="Withheld by gpu-memory-utilization {fmt(utilHeadroomPer)} GiB"></div>
                </div>
                <div class="cap">GPU {i}<br>{gpu.mem_gb} GiB</div>
              </div>
            {/each}
            {#if R.tp > 8}<div class="more">+{R.tp - 8}<br>more</div>{/if}
          </div>
          <div class="legend">
            <span><i class="w"></i>Weights {fmt(wPer)} GiB</span>
            <span><i class="k"></i>KV in use {fmt(kvPer)} GiB</span>
            <span><i class="kf"></i>KV free {fmt(kvFreePer)} GiB</span>
            <span><i class="r"></i>Runtime reserve {fmt(reservePer, 2)} GiB</span>
            <span><i class="u"></i>Withheld by mem-util {fmt(utilHeadroomPer)} GiB</span>
          </div>
          <p class="tot">The five segments sum to the card's {gpu.mem_gb} GiB. <b>KV free</b> is usable right now — it is what more concurrency would consume. <b>Withheld by mem-util</b> is the {Math.round((1 - memUtil) * 100)}% this plan never hands to vLLM; raising <code>--gpu-memory-utilization</code> reclaims it, at the cost of the margin that keeps the server off an OOM.</p>
          <p class="replnote">This model runs as <b>{R.pods}</b> identical replica{R.pods > 1 ? 's' : ''} → {R.pods} × TP{R.tp} = <b>{R.gpus}</b> GPUs total{#if R.tp > 8} (showing 8 of {R.tp} GPUs in the replica){/if}.</p>
        </div>

        {#if topo}
          <div class="panel">
            <!-- the heading counts the whole deployment; when the drawing is capped it has to say
                 so here, because the marker inside the SVG can sit past the horizontal scroll -->
            <h2>Deployment topology — {R.pods} replica{R.pods > 1 ? 's' : ''} × TP{R.tp} on {R.nodes} node{R.nodes > 1 ? 's' : ''}{#if topo.truncated}<small> · drawing {topo.shownPods} on {topo.shown}</small>{/if}</h2>
            <div class="topowrap">
              <!-- markup comes from the domain renderer so `npm run preview:topology`
                   screenshots the same SVG this component ships -->
              {@html topoSvg}
            </div>

            <div class="topolegend">
              <span><i class="sw-used"></i>GPU in this plan</span>
              <span><i class="sw-idle"></i>idle GPU in the node</span>
              <span><i class="sw-pod"></i>replica (TP group)</span>
              {#if R.multi_node}<span class="warnitem">⚠ crosses the fabric</span>{/if}
            </div>

            {#if R.multi_node}
              <p class="tot"><b>TP{R.tp} does not fit in a {perNode}-GPU node.</b> Each replica's tensor-parallel collective — an all-reduce on every layer, on every token — leaves the node and crosses the fabric shown above. Inside a node that traffic rides NVLink at multiple TB/s; between nodes it rides InfiniBand or RoCE at a fraction of that, and it is on the critical path of every forward pass. Expect materially worse latency and MBU than the throughput figures here assume, and treat a non-NVLink fabric as a hard prerequisite rather than a detail.</p>
            {:else}
              <p class="tot">Every replica sits inside one node, so its tensor-parallel collective stays on NVLink and never touches the inter-node fabric. Replicas are independent — they share only the weights on storage and the router in front, so scaling out adds throughput without adding collective traffic.</p>
            {/if}
            {#if topo.truncated}<p class="tot">The heading counts the whole deployment; the drawing stops at {topo.shown} of {R.nodes} nodes, so it shows {topo.shownPods} of {R.pods} replicas. The remaining {topo.hiddenPods} replica{topo.hiddenPods > 1 ? 's' : ''} repeat{topo.hiddenPods > 1 ? '' : 's'} the same pattern across the other {topo.hiddenNodes} node{topo.hiddenNodes > 1 ? 's' : ''}.{#if topoCut} The last replica bar is left open on the right because the node cut-off falls inside its TP group — it continues past the edge of the drawing.{/if}</p>{/if}
          </div>
        {/if}

        <div class="panel">
          <h2>Breakdown</h2>
          <div class="li"><span>Weights ({effQuant})</span><b>{fmt(R.weights_gb)} GiB</b></div>
          <div class="li"><span>KV per token ({kvBytes === 1 ? 'FP8' : 'FP16'}{#if R.kv_windowed}, effective{/if})</span><b>{(R.kv_per_token_gb * 1024).toFixed(3)} MiB</b></div>
          <div class="li"><span>KV per request ({ctx / 1024}K × {Math.round(util * 100)}%)</span><b>{fmt(R.kv_per_request_gb)} GiB</b></div>
          <div class="li"><span>Concurrency per pod</span><b>{R.concurrency_per_pod} req</b></div>
          <div class="li"><span>Pods → GPUs → nodes ({perNode}/node)</span><b>{R.pods} → {R.gpus} → {R.nodes}</b></div>
          <div class="li"><span>Runtime reserve / GPU <small>(context {R.runtime_reserve_gb > 2.5 ? '+ prefill activations' : ''})</small></span><b>{fmt(R.runtime_reserve_gb, 2)} GiB{#if R.activation_gb > 0.05}<small> · {fmt(R.activation_gb, 2)} act</small>{/if}</b></div>
          <div class="li"><span>Usable HBM per GPU ({memUtil})</span><b>{fmt(R.usable_gb)} GiB</b></div>
          <div class="li"><span>Pod headroom <small>(free after weights + 1 request)</small></span><b class:tightv={R.tight}>{(R.headroom_fraction * 100).toFixed(1)}%</b></div>
          <div class="li"><span>Time to first token <small>({R.ttft_compute_bound ? 'compute-bound prefill' : 'bandwidth floor'})</small></span><b>~{ttftLabel(R.ttft_ms)} <small>±50%</small></b></div>
          <div class="li"><span>Prefill work <small>({Math.round(ctx * util).toLocaleString()} tokens)</small></span><b>{R.prefill_pflops.toFixed(2)} PFLOP</b></div>
          <div class="li"><span>Decode throughput / request</span><b>~{R.decode_tps_per_request} tok/s</b></div>
          <div class="li"><span>Aggregate throughput</span><b>~{R.throughput_tokens_per_sec.toLocaleString()} tok/s <small>±40%</small></b></div>
        </div>

        {#if serveCmd}
          <div class="panel">
            <h2>Launch command
              <span style="float:right;display:flex;gap:6px">
                <button class="btn ghost" onclick={() => (cmdDocker = !cmdDocker)}>{cmdDocker ? 'plain' : 'docker'}</button>
                <button class="btn" onclick={copyCmd}>{cmdCopied ? 'copied' : 'copy'}</button>
              </span>
            </h2>
            <pre class="cmd">{serveCmd.command}</pre>
            <ul class="cmdnotes">{#each serveCmd.notes as n}<li>{n}</li>{/each}</ul>
          </div>
        {/if}

        <div class="panel">
          <h2>Concurrency rubric — pick a target, see the cost &amp; throughput</h2>
          <table class="rubric"><thead><tr><th class="num">Concurrency</th><th class="num">GPUs</th><th class="num">Pods</th><th class="num">TTFT</th><th class="num">tok/s · req</th><th class="num">tok/s · total</th><th></th></tr></thead><tbody>
            {#each sweep as s}
              <tr class:cur={s.concurrency === conc} class:infeasible={!s.feasible}>
                <td class="num">{s.concurrency}</td>
                {#if s.feasible}
                  <td class="num">{s.gpus}{#if s.tight}<span class="badge tightb" title="under 10% pod headroom">tight</span>{/if}</td><td class="num">{s.pods} × TP{s.tp}</td><td class="num">~{ttftLabel(s.ttft_ms)}</td><td class="num">~{s.decode_tps_per_request}</td><td class="num">~{s.throughput_tokens_per_sec.toLocaleString()}</td>
                  <td>{#if s.concurrency !== conc}<button class="btn ghost" onclick={() => (conc = s.concurrency)}>use</button>{:else}<span class="badge">current</span>{/if}</td>
                {:else}
                  <td class="num" colspan="5" style="color:var(--err)">infeasible at this concurrency</td><td></td>
                {/if}
              </tr>
            {/each}
          </tbody></table>
          <p class="tot">Per-request decode rate falls as concurrency rises (more requests share the pod's bandwidth); total throughput and GPU count rise. Throughput ±40%, TTFT ±50% — indicative, validate against benchmarks.</p>
        </div>

        {#if fleetCheck()}
          {@const fc = fleetCheck()}
          {#if fc?.kind === 'fit'}<div class="state ok"><b>Fleet check — fits.</b> {R.gpus} of {fc.head} uncommitted {gpu.name}. Headroom supports up to {fc.maxPods} pods ≈ {fc.maxPods * fc.concEach} concurrent at this config.</div>{/if}
          {#if fc?.kind === 'short'}<div class="state err"><b>Fleet check — shortage.</b> Needs {fc.need} but only {fc.head} {gpu.name} uncommitted. Short by {fc.short} — reduce concurrency/context, quantize harder, or add {fc.nodes} node(s).</div>{/if}
          {#if fc?.kind === 'absent'}<div class="state warn"><b>Fleet check.</b> {gpu.name} is not in the defined fleet. Add a pool or switch SKU.</div>{/if}
        {/if}
        {#if R.tight}<div class="state warn"><b>Tight fit — {(R.headroom_fraction * 100).toFixed(1)}% headroom.</b> Weights + one request of KV leave under 10% of the pod's HBM free, so this plan has no margin for the ±5% weight estimate, fragmentation, or a longer prompt than modelled. It will likely OOM on a real vLLM launch. Drop context, quantise the KV cache, or move to the next TP size.</div>{/if}
        {#if R.kv_windowed && model?.linear_attention_layers}<div class="state ok"><b>Hybrid attention applied.</b> Only {model.full_attention_layers} of {model.layers} layers keep a token-indexed cache; the other {model.linear_attention_layers} are linear/recurrent and hold a constant {fmt((model.linear_attention_layers * (model.linear_state_bytes_per_layer ?? 0)) / 2 ** 30)} GiB regardless of context. KV per request is <b>{fmt(R.kv_per_request_gb)} GiB</b> instead of the {fmt(kvNominalGb)} GiB all-{model.layers}-layer sizing would claim — a {(kvNominalGb / R.kv_per_request_gb).toFixed(1)}× difference.</div>
        {:else if R.kv_windowed}<div class="state ok"><b>Local/global attention applied.</b> {model.full_attention_layers} of {model.layers} layers keep full context; the other {model.layers - (model.full_attention_layers ?? 0)} stop growing at {model.sliding_window} tokens. KV per request is <b>{fmt(R.kv_per_request_gb)} GiB</b> instead of the {fmt(kvNominalGb)} GiB an all-full-attention model of this shape would need — a {(kvNominalGb / R.kv_per_request_gb).toFixed(1)}× saving at this context.</div>{/if}
        {#if R.activation_gb > 2}<div class="state warn"><b>Prefill activations are {fmt(R.activation_gb, 1)} GiB per GPU.</b> A chunk of {batchTokens.toLocaleString()} tokens materialises activations for every token at once, so the runtime reserve has grown to {fmt(R.runtime_reserve_gb, 1)} GiB — memory that is no longer available for KV cache. Lower <code>--max-num-batched-tokens</code> if you would rather spend it on concurrency.</div>{/if}
        {#if R.weights_estimated}<div class="state warn"><b>Weight estimate is approximate.</b> This model carries no embedding geometry, so the weights fall back to a flat overhead factor — optimistic by ~5–15% at INT4/MXFP4, where the 16-bit embedding and lm_head are a large share of the checkpoint. Add <code>hidden_size</code> and <code>vocab_size</code> in the model catalog to sharpen it.</div>{/if}
        {#if R.multi_node}<div class="state warn"><b>TP {R.tp} &gt; {perNode} GPUs/node:</b> this replica spans nodes — needs NVLink/IB fabric; latency &amp; MBU degrade vs single-node TP.</div>{/if}
        {#if util >= 1}<div class="state warn">Sizing at 100% context utilisation buys worst-case memory that mostly sits idle. Size KV at P95 of observed sequence length.</div>{/if}
      {:else}
        <div class="state err"><b>Infeasible.</b> {(result as any).reason} (weights {fmt((result as any).weights_gb)} GiB, KV/request {fmt((result as any).kv_per_request_gb)} GiB)</div>
      {/if}
    </section>
  </div>

{:else if tab === 'fleet'}
  <section class="panel">
    <h2>Fleet — define GPU pools</h2>
    <div class="row3">
      <label>GPU SKU<select bind:value={newPool.gpu_sku_id}>{#each catalog.gpus as g}<option value={g.id}>{g.name}</option>{/each}</select></label>
      <label>GPUs / node<input type="number" bind:value={newPool.gpus_per_node} /></label>
      <label>Nodes<input type="number" bind:value={newPool.node_count} /></label>
    </div>
    <button class="btn primary" onclick={addPool} style="margin-top:10px">Add pool</button>
    {#if fleet.length}
      <table><thead><tr><th>Pool</th><th class="num">GPUs</th><th class="num">HBM</th><th></th></tr></thead><tbody>
        {#each fleet as p, i}<tr><td>{p.node_count} node{p.node_count > 1 ? 's' : ''} × {p.gpus_per_node} × {gpuName(p.gpu_sku_id)}</td><td class="num">{p.node_count * p.gpus_per_node}</td><td class="num">{fmt(p.node_count * p.gpus_per_node * (catalog.gpus.find((g: GpuSku) => g.id === p.gpu_sku_id)?.mem_gb ?? 0) / 1024)} TiB</td><td><button class="btn ghost" onclick={() => delPool(i)}>remove</button></td></tr>{/each}
      </tbody></table>
      <p class="tot">Fleet total: <b>{fleetTotals().gpus}</b> GPUs · {fleetTotals().nodes} nodes · {(fleetTotals().hbm / 1024).toFixed(1)} TiB HBM</p>
    {:else}<div class="empty">No pools yet — add a pool to define your fleet.</div>{/if}
  </section>

{:else if tab === 'plan'}
  {#if fleet.length === 0 && plan.length === 0}
    <div class="empty">A <b>cluster</b> is a fleet + the models you run on it.<br>1 · Define hardware on the <b>Fleet</b> tab &nbsp;→&nbsp; 2 · size models on <b>Sizing</b> and “+ Add to cluster” &nbsp;→&nbsp; they appear here with utilisation.</div>
  {:else}
    <div class="kpis" style="grid-template-columns:repeat(4,1fr)">
      <div class="kpi"><div class="v">{clusterTotals().totalG}</div><div class="l">Fleet GPUs</div></div>
      <div class="kpi p"><div class="v">{clusterTotals().committedG}</div><div class="l">Committed</div></div>
      <div class="kpi g"><div class="v">{clusterTotals().freeG}</div><div class="l">Free</div></div>
      <div class="kpi a"><div class="v">{clusterTotals().util.toFixed(0)}<small>%</small></div><div class="l">Utilisation</div></div>
    </div>
    {#if clusterTotals().over}<div class="state err"><b>⛔ Over capacity.</b> One or more SKUs are committed beyond the fleet — remove deployments or add hardware.</div>{/if}

    <section class="panel"><h2>Fleet utilisation — used vs free</h2>
      {#if fleet.length === 0}<div class="empty">No fleet defined — add pools on the <b>Fleet</b> tab to see utilisation of the stack.</div>
      {:else}
        {#each clusterBySku() as r}
          <div class="skuutil">
            <div class="skuhead"><b>{r.name}</b><span>{r.committed} / {r.total} GPUs · {fmt(r.committedHbm / 1024)} / {fmt(r.totalHbm / 1024)} TiB HBM · <b style="color:{r.over ? 'var(--err)' : r.util > 80 ? 'var(--warn)' : 'var(--brandink)'}">{r.total > 0 ? r.util.toFixed(0) + '%' : '—'}</b></span></div>
            <div class="cells">
              {#each Array(Math.min(r.total, 96)) as _, i}<div class="cell" class:used={i < r.committed}></div>{/each}
              {#if r.total > 96}<span class="more">+{r.total - 96}</span>{/if}
              {#if r.total === 0}<span class="more">⚠ demand for {r.committed} GPUs but no {r.name} in the fleet</span>{/if}
            </div>
          </div>
        {/each}
        <div class="legend" style="margin-top:12px"><span><i class="cellk used"></i>Committed (used by a deployment)</span><span><i class="cellk"></i>Free</span></div>
      {/if}
    </section>

    <section class="panel"><h2>Models on the cluster</h2>
      {#if plan.length}
        <table><thead><tr><th>Deployment</th><th>SKU</th><th class="num">GPUs</th><th class="num">Pods</th><th class="num">KV GiB</th><th></th></tr></thead><tbody>
          {#each plan as d, i}<tr><td>{d.label}</td><td>{gpuName(d.gpu_sku_id)}</td><td class="num">{d.gpus}</td><td class="num">{d.pods} × TP{d.tp}</td><td class="num">{fmt(d.kv)}</td><td><button class="btn ghost danger" onclick={() => delDeployment(i)}>remove</button></td></tr>{/each}
        </tbody></table>
        <p class="tot">Total demand: <b>{planTotals().gpus}</b> GPUs · {planTotals().pods} pods · {fmt(planTotals().kv)} GiB KV</p>
      {:else}<div class="empty">No models added — size one on the <b>Sizing</b> tab and click “+ Add to cluster”.</div>{/if}
    </section>

    <section class="panel">
      <div class="secbar"><h2 style="margin:0">Cost estimate — GPU-hour rental</h2>
        {#if plan.length}<div class="expbtns"><button class="btn ghost" onclick={exportCsv}>⭳ Export CSV</button><button class="btn ghost" onclick={exportJson}>⭳ Export JSON</button></div>{/if}
      </div>
      {#if plan.length === 0}<div class="empty">Add models to the cluster to estimate cost, then export it.</div>
      {:else if !cost().priced}<div class="empty">No GPU prices set. Add a <b>$/GPU-hour</b> to each SKU on the <b>Models</b> tab (admin), or Reset the catalog to load indicative defaults.</div>
      {:else}
        <div class="kpis" style="grid-template-columns:repeat(3,1fr);margin-bottom:14px">
          <div class="kpi g"><div class="v">{money(cost().totalHr)}<small>/hr</small></div><div class="l">Cluster run-rate</div></div>
          <div class="kpi p"><div class="v">{money(cost().totalMo)}<small>/mo</small></div><div class="l">Monthly (730h)</div></div>
          <div class="kpi a"><div class="v">{money(cost().totalYr)}<small>/yr</small></div><div class="l">Annual</div></div>
        </div>
        <table><thead><tr><th>GPU SKU</th><th class="num">Committed</th><th class="num">$/GPU-hr</th><th class="num">$/hr</th><th class="num">$/mo</th></tr></thead><tbody>
          {#each cost().lines as l}<tr><td>{l.name}</td><td class="num">{l.gpus}</td><td class="num">{money(l.price)}</td><td class="num">{money(l.hr)}</td><td class="num">{money(l.hr * 730)}</td></tr>{/each}
        </tbody></table>
        <h2 style="margin-top:18px">Per-model economics — cost per million tokens</h2>
        <table><thead><tr><th>Model</th><th class="num">GPUs</th><th class="num">$/hr</th><th class="num">tok/s</th><th class="num">$ / Mtok</th></tr></thead><tbody>
          {#each modelEconomics() as e}<tr><td>{e.label}</td><td class="num">{e.gpus}</td><td class="num">{money(e.hr)}</td><td class="num">~{e.tps.toLocaleString()}</td><td class="num">{e.perMtok > 0 ? money(e.perMtok) : '—'}</td></tr>{/each}
        </tbody></table>
        <p class="tot">Rental at admin-set $/GPU-hour; $/Mtok = ($/hr) ÷ (tokens per hour ÷ 1e6), using the indicative throughput (±40%). Prices are indicative — set your contracted rates on the Models tab.</p>
      {/if}
    </section>

    <section class="panel"><h2>Save this cluster</h2>
      <div class="row"><label>Save fleet + models as a named scenario<input bind:value={saveName} placeholder="RFP-Acme" /></label><button class="btn primary" style="align-self:end;height:35px" onclick={saveConfig}>Save to My Configurations</button></div>
      <p class="tot">Saves the fleet and all its models as a private, reloadable scenario (recomputes identically later).</p>
    </section>
  {/if}

{:else if tab === 'configs'}
  <section class="panel"><h2>My Configurations</h2>
    <div class="row"><label>Save current fleet + plan as<input bind:value={saveName} placeholder="RFP-Acme" /></label><button class="btn primary" style="align-self:end;height:35px" onclick={saveConfig}>Save</button></div>
    {#if configs.length}
      <table><thead><tr><th>Name</th><th>Updated</th><th></th></tr></thead><tbody>
        {#each configs as c}<tr><td><b>{c.name}</b></td><td><small>{new Date(c.updated_at).toLocaleString()}</small></td><td><button class="btn ghost" onclick={() => loadConfig(c.id)}>load</button> <button class="btn ghost" onclick={() => delConfig(c.id)}>delete</button></td></tr>{/each}
      </tbody></table>
    {:else}<div class="empty">No saved configurations yet — build a fleet + plan and save it. Private to your profile.</div>{/if}
  </section>

{:else if tab === 'catalog'}
  <section class="panel"><h2>Model Cards {#if ident.role === 'admin'}<button class="btn ghost" style="float:right" onclick={resetCatalog}>Reset to defaults</button>{/if}</h2>
    <div class="cards">
      {#each catalog.models as m}
        <div class="mcard">
          <div class="mcard-h"><b>{m.name}</b> <span class="badge {m.mla ? 'mla' : 'gqa'}">{m.mla ? 'MLA' : 'GQA'}</span></div>
          <div class="mcard-b">
            <div><span>Total / Active</span><b>{m.total_params_b} / {m.active_params_b} B</b></div>
            <div><span>Layers</span><b>{m.layers}</b></div>
            <div><span>KV geometry</span><b>{m.mla ? 'latent 576' : `${m.kv_heads}×${m.head_dim}`}</b></div>
            <div><span>Max context</span><b>{(m.max_ctx / 1024)}K</b></div>
            <div><span>TP options</span><b>{m.tp_options.join(', ')}</b></div>
            <div><span>Quants</span><b>{m.quants.join(', ')}</b></div>
            <div><span>Embedding</span><b>{m.vocab_size && m.hidden_size ? `${m.vocab_size.toLocaleString()} × ${m.hidden_size}${m.tied_embeddings ? ' tied' : ''}` : '— (est.)'}</b></div>
            <div><span>Mixed precision</span><b>{m.mixed_precision && Object.keys(m.mixed_precision).length ? Object.entries(m.mixed_precision).map(([q, d]) => `${q}: dense @ ${d}`).join(', ') : '— uniform'}</b></div>
            <div><span>Attention</span><b>{m.linear_attention_layers ? `${m.full_attention_layers}/${m.layers} cached · ${m.linear_attention_layers} linear` : m.sliding_window && m.full_attention_layers != null ? `${m.full_attention_layers}/${m.layers} full · ${m.sliding_window}-tok window` : 'all full-context'}</b></div>
          </div>
          {#if ident.role === 'admin'}<div class="mcard-f"><button class="btn ghost" onclick={() => editModel(m)}>edit</button> <button class="btn ghost danger" onclick={() => deleteModelUi(m.id)}>delete</button></div>{/if}
        </div>
      {/each}
    </div>
    <p class="tot">{catalog.models.length} models{#if ident.role !== 'admin'} · read-only (admin required to edit){/if}</p>
  </section>

  {#if ident.role === 'admin'}
    <section class="panel"><h2>Import a model from Hugging Face</h2>
      <div class="row"><label>Model ID (owner/model)<input bind:value={hfId} placeholder="Qwen/Qwen2.5-72B-Instruct" onkeydown={(e) => { if ((e as KeyboardEvent).key === 'Enter') hfFetch(); }} /></label><button class="btn primary" style="align-self:end;height:35px" onclick={hfFetch} disabled={hfBusy}>{hfBusy ? 'Fetching…' : 'Fetch config'}</button></div>
      <div class="hfsuggest">Try: {#each HF_SUGGEST as s}<button class="chip2" onclick={() => { hfId = s; hfFetch(); }}>{s}</button>{/each}</div>
      <div class="row" style="margin-top:10px">
        <div class="hint" style="margin:0">
          <b>Two sources, one model.</b> Hugging Face <code>config.json</code> gives the geometry — layers, heads, embedding sizes, attention regime.
          It never carries parameter counts, shipped quantisations, or the TP sizes people actually run.
          <a href="https://recipes.vllm.ai" target="_blank" rel="noopener">recipes.vllm.ai</a> carries exactly those four.
        </div>
        <button class="btn" style="align-self:end;height:35px;white-space:nowrap" onclick={() => recipeFetch()} disabled={recipeBusy}>{recipeBusy ? 'Fetching…' : '+ Apply vLLM recipe'}</button>
      </div>
      {#if recipeInfo}
        <div class="state ok" style="margin-top:10px">
          <b>Recipe applied — {recipeInfo.model_id}.</b>
          Supplied: {(recipeInfo.filled ?? []).join(', ') || '—'}{#if recipeInfo.min_vllm_version} · needs vLLM ≥ {recipeInfo.min_vllm_version}{/if}
          {#if recipeInfo.unmapped_precisions?.length}<br><span style="color:var(--warn)">Skipped unmapped precisions: {recipeInfo.unmapped_precisions.join(', ')} (mixed-precision checkpoints have no single bytes/param).</span>{/if}
          {#if recipeInfo.vram_minimums?.length}
            <table class="qtab" style="margin-top:8px"><thead><tr><th>Variant</th><th class="num">Their VRAM floor</th><th class="num">Our weights</th><th class="num">Ratio</th></tr></thead><tbody>
              {#each recipeInfo.vram_minimums as v}
                {@const ours = mfWeightsGb(v.quant)}
                <tr><td>{v.precision}</td><td class="num">{v.vram_gb} GB</td>
                  <td class="num">{ours != null ? ours.toFixed(1) + ' GB' : '—'}</td>
                  <td class="num" style={ours != null && ours / v.vram_gb > 1 ? 'color:var(--err)' : ''}>{ours != null ? (ours / v.vram_gb).toFixed(2) : '—'}</td></tr>
              {/each}
            </tbody></table>
            <div style="font-size:11px;margin-top:4px">Their floor covers weights + KV + overhead, so a ratio around 0.85 is healthy. Above 1.00 means one of the two figures is wrong.</div>
          {/if}
        </div>
      {/if}
      {#if hfCard}
        <div class="hfcard"><b>{hfCard.model_id}</b> — {hfCard.card.model_type ?? hfCard.card.architectures?.[0] ?? 'model'} · {hfCard.card.num_hidden_layers} layers · ctx {(hfCard.card.max_position_embeddings ?? 0).toLocaleString()} {#if hfCard.detectedMla}· <span class="badge mla">MLA</span>{/if}
          <div class="tot" style="margin-top:6px">Mapped into the form below. <b style="color:var(--warn)">Complete before saving: {hfMissing.join(', ')}</b> — HF configs omit parameter counts, TP sizes, and quant variants.</div>
        </div>
      {/if}
    </section>
    <section class="panel" id="mform">
      <h2>{mfEditing ? `Edit model — ${mf.id}` : 'New model'}</h2>
      <div class="row3">
        <label>ID<input bind:value={mf.id} disabled={mfEditing} placeholder="glm-6" /></label>
        <label style="grid-column:span 2">Name<input bind:value={mf.name} placeholder="GLM-6 700B" /></label>
      </div>
      {#if errFor(mfErrors, 'id')}<div class="ferr">{errFor(mfErrors, 'id')}</div>{/if}
      <div class="row3">
        <label>Total params (B)<input type="number" step="0.01" bind:value={mf.total_params_b} /></label>
        <label>Active params (B)<input type="number" step="0.01" bind:value={mf.active_params_b} /></label>
        <label>Layers<input type="number" bind:value={mf.layers} /></label>
      </div>
      {#if errFor(mfErrors, 'active_params_b')}<div class="ferr">{errFor(mfErrors, 'active_params_b')}</div>{/if}
      <div class="row3">
        <label>KV heads{#if mf.mla}<small> (MLA→0)</small>{/if}<input type="number" bind:value={mf.kv_heads} disabled={mf.mla} /></label>
        <label>Head dim{#if mf.mla}<small> (MLA→0)</small>{/if}<input type="number" bind:value={mf.head_dim} disabled={mf.mla} /></label>
        <label>Max context<input type="number" bind:value={mf.max_ctx} /></label>
      </div>
      {#if errFor(mfErrors, 'kv_heads')}<div class="ferr">{errFor(mfErrors, 'kv_heads')}</div>{/if}
      {#if errFor(mfErrors, 'head_dim')}<div class="ferr">{errFor(mfErrors, 'head_dim')}</div>{/if}
      {#if errFor(mfErrors, 'max_ctx')}<div class="ferr">{errFor(mfErrors, 'max_ctx')}</div>{/if}
      <div class="row3">
        <label>Hidden size<small> (config.json)</small><input type="number" bind:value={mf.hidden_size} placeholder="8192" /></label>
        <label>Vocab size<small> (config.json)</small><input type="number" bind:value={mf.vocab_size} placeholder="128256" /></label>
        <label style="display:flex;align-items:center;gap:8px;margin-top:24px"><input type="checkbox" style="width:auto" bind:checked={mf.tied_embeddings} />Tied embeddings</label>
      </div>
      <div class="hint">Optional but recommended: the embedding table and lm_head stay at 16-bit through quantisation. Supplying these sizes the un-quantised tail exactly instead of a flat overhead factor — worth 10%+ of the footprint at INT4/MXFP4. <em>Not used for GGUF quants (Q4_K_M, Q8_0, IQ4_XS), whose published bytes/param already include the embedding layers.</em></div>
      <div class="row3">
        <label>Sliding window<small> (tokens)</small><input type="number" bind:value={mf.sliding_window} placeholder="128" /></label>
        <label>Full-attention layers<small> (rest windowed)</small><input type="number" bind:value={mf.full_attention_layers} placeholder="18" /></label>
      </div>
      <div class="hint">Leave both blank for a normal full-context model. If the model alternates local and global attention (GPT-OSS, Gemma, Mistral v0.1), the windowed layers stop accumulating KV at the window — for GPT-OSS-120B at 128K that halves the cache. From <code>config.json</code>: <code>sliding_window</code> plus <code>layer_types</code> or <code>sliding_window_pattern</code>.</div>
      {#if errFor(mfErrors, 'sliding_window')}<div class="ferr">{errFor(mfErrors, 'sliding_window')}</div>{/if}
      {#if errFor(mfErrors, 'full_attention_layers')}<div class="ferr">{errFor(mfErrors, 'full_attention_layers')}</div>{/if}
      <div class="row3">
        <label>Linear-attn layers<small> (KDA etc)</small><input type="number" bind:value={mf.linear_attention_layers} placeholder="69" /></label>
        <label>Linear state / layer<small> (bytes)</small><input type="number" bind:value={mf.linear_state_bytes_per_layer} placeholder="6291456" /></label>
      </div>
      <div class="hint">For hybrid models (Kimi K3's KDA, Qwen3-Next, MiniMax) whose recurrent layers hold a <em>constant</em> state instead of a growing cache. State per layer is typically <code>num_heads × head_dim² × 4</code> bytes (fp32). Every layer must be accounted for: full + windowed + linear = layers.</div>
      {#if errFor(mfErrors, 'linear_state_bytes_per_layer')}<div class="ferr">{errFor(mfErrors, 'linear_state_bytes_per_layer')}</div>{/if}
      {#if errFor(mfErrors, 'linear_attention_layers')}<div class="ferr">{errFor(mfErrors, 'linear_attention_layers')}</div>{/if}
      {#if errFor(mfErrors, 'vocab_size')}<div class="ferr">{errFor(mfErrors, 'vocab_size')}</div>{/if}
      {#if errFor(mfErrors, 'hidden_size')}<div class="ferr">{errFor(mfErrors, 'hidden_size')}</div>{/if}
      <div class="row3">
        <label>TP options<input bind:value={mf.tp_options} placeholder="8,16" /></label>
        <label style="display:flex;align-items:center;gap:8px;margin-top:24px"><input type="checkbox" style="width:auto" bind:checked={mf.mla} onchange={() => { if (mf.mla) { mf.kv_heads = 0; mf.head_dim = 0; } }} />MLA (latent attention)</label>
      </div>
      <label>Quantisation variants</label>
      <div class="quants">{#each QUANTS as q}<button type="button" class="qbtn" class:on={mf.quants.includes(q)} onclick={() => toggleQuant(q)}>{q}</button>{/each}</div>
      {#if errFor(mfErrors, 'quants')}<div class="ferr">{errFor(mfErrors, 'quants')}</div>{/if}
      {#if mfErrors.find((e) => e.path === '')}<div class="ferr">{mfErrors.find((e) => e.path === '')?.message}</div>{/if}
      <div style="margin-top:14px;display:flex;gap:8px"><button class="btn primary" onclick={saveModel}>{mfEditing ? 'Save changes' : 'Create model'}</button>{#if mfEditing}<button class="btn ghost" onclick={newModelForm}>Cancel</button>{/if}</div>
    </section>
  {/if}

  {#if ident.role === 'admin'}
    <section class="panel"><h2>New GPU SKU</h2>
      <div class="row"><label>ID<input bind:value={gf.id} placeholder="b300" /></label><label>Name<input bind:value={gf.name} placeholder="B300 288 GB" /></label></div>
      <div class="row3"><label>Memory (GiB)<small> as nvidia-smi reports</small><input type="number" bind:value={gf.mem_gb} /></label><label>Bandwidth (TB/s)<input type="number" step="0.1" bind:value={gf.bw_tbs} /></label><label>Price ($/GPU-hr)<input type="number" step="0.1" bind:value={gf.price_per_gpu_hour} /></label></div>
      {#if gfErrors.length}<div class="ferr">{gfErrors[0].message}</div>{/if}
      <button class="btn primary" style="margin-top:12px" onclick={saveGpu}>Add / update GPU SKU</button>
    </section>
  {/if}
  <section class="panel"><h2>GPU SKUs</h2>
    <table><thead><tr><th>SKU</th><th class="num">HBM (GiB)</th><th class="num">BW (TB/s)</th><th class="num">$/GPU-hr</th>{#if ident.role === 'admin'}<th></th>{/if}</tr></thead><tbody>
      {#each catalog.gpus as g}<tr><td>{g.name}</td><td class="num">{g.mem_gb}</td><td class="num">{g.bw_tbs}</td><td class="num">{g.price_per_gpu_hour != null ? money(g.price_per_gpu_hour) : '—'}</td>{#if ident.role === 'admin'}<td><button class="btn ghost danger" onclick={() => deleteGpuUi(g.id)}>del</button></td>{/if}</tr>{/each}
    </tbody></table>
    <p class="tot">{catalog.gpus.length} GPU SKUs</p>
  </section>

{:else if tab === 'methodology'}
  <section class="panel doc">
    <h1 class="doctitle">How the calculator works</h1>
    <p class="lead">LLM capacity planning is deterministic maths, not guesswork. Because autoregressive decoding generates <em>one token at a time</em>, serving is <b>memory-bound</b> — the bottleneck is memory <em>bandwidth</em>, not compute. Every figure on the Sizing tab comes from the formulas below. Fixed constants: runtime reserve <b>2.5 GiB</b>, MBU <b>0.55</b>, MLA latent <b>576</b>, tight-fit threshold <b>10%</b>.</p>

    <h2 class="dh">1 · Hardware memory modelling</h2>
    <p>Start with how much high-bandwidth memory (HBM) the inference engine (e.g. vLLM) may actually use. The <code>gpu_memory_utilization</code> factor caps it; a fixed runtime reserve is subtracted to prevent out-of-memory (OOM) failures.</p>
    <div class="formula">Usable VRAM per GPU = (Physical capacity × Utilisation) − Runtime reserve</div>
    <p>Every GPU's HBM divides into five parts that sum to the card, and the Sizing tab draws them that way: <b>weights</b> (this replica's shard), <b>KV in use</b> (the sessions actually placed), <b>KV free</b> (usable now — what more concurrency would consume), <b>runtime reserve</b> (vLLM's own overhead), and <b>withheld by mem-util</b> (the slice never handed to vLLM). Only the last is recoverable by changing a flag, and only by giving up the margin that keeps the server off an OOM. Collapsing the last three into one "reserve" figure hides that, and makes a 141 GiB card look like it carries 20 GiB of untouchable overhead when 14 of it is a slider position.</p>

    <p>Tensor Parallelism (TP) splits one model replica across several GPUs; their usable memory pools linearly:</p>
    <div class="formula">Usable pod memory = Usable VRAM per GPU × TP size</div>

    <h2 class="dh">2 · Weights vs. dynamic cache</h2>
    <p>A pod's memory is split between <b>static model weights</b> (the parameters) and the <b>dynamic KV cache</b> (per-token attention state during generation). Whatever remains after weights is the budget for concurrency:</p>
    <div class="formula">Free KV space = Usable pod memory − Weights</div>
    <p class="note">For Mixture-of-Experts (MoE) models, <em>all</em> experts must be resident, so weights use the <b>total</b> parameter count — not the active one.</p>

    <h3 class="dh3">Weights are not one flat bytes-per-parameter</h3>
    <p>A quantised checkpoint is not uniformly quantised. The <b>embedding table and output head stay at 16-bit</b> in essentially every real INT4/FP8/MXFP4 release, because quantising them costs disproportionate quality. That tail is invisible at FP16 and dominant at 4-bit:</p>
    <div class="formula">Weights = (Total params − Tail) × Effective bytes/param + Tail × 2</div>
    <div class="formula">Tail (params) = vocab_size × hidden_size × (tied ? 1 : 2)</div>
    <p>For Llama-3.3-70B the tail is 2 × 128,256 × 8,192 = <b>2.10 B parameters = 3.9 GiB</b> — over 10% of an INT4 checkpoint, far more than a flat overhead factor allows. It is the difference between predicting 32.9 GiB (wrong) and 37.1 GiB (what published AWQ/GPTQ checkpoints actually weigh — ~40 × 10⁹ bytes).</p>
    <p><b>Effective</b> bytes/param also exceed the nominal bit-width, because low-bit formats store per-group scale metadata next to the data:</p>
    <table class="qtab"><thead><tr><th>Quant</th><th class="num">Nominal</th><th class="num">Effective</th><th>Metadata</th></tr></thead><tbody>
      <tr><td>FP16</td><td class="num">2</td><td class="num">2</td><td>—</td></tr>
      <tr><td>FP8 / INT8</td><td class="num">1</td><td class="num">1</td><td>per-tensor/channel scale — negligible</td></tr>
      <tr><td>INT4 (grouped, g=128)</td><td class="num">0.5</td><td class="num">0.52</td><td>fp16 scale + int4 zero per 128 weights</td></tr>
      <tr><td>MXFP4 (block=32)</td><td class="num">0.5</td><td class="num">0.53125</td><td>E8M0 scale per 32 weights</td></tr>
      <tr><td>NVFP4 (block=16)</td><td class="num">0.5</td><td class="num">0.5625</td><td>E4M3 scale per 16 weights</td></tr>
      <tr><td>GGUF Q8_0</td><td class="num">1.0</td><td class="num">1.06</td><td>8.5 real bits — whole-file average</td></tr>
      <tr><td>GGUF Q4_K_M</td><td class="num">0.5</td><td class="num">0.61</td><td>4.9 real bits — mixes Q4_K and Q6_K per tensor</td></tr>
      <tr><td>GGUF IQ4_XS</td><td class="num">0.5</td><td class="num">0.53125</td><td>4.25 real bits</td></tr>
    </tbody></table>
    <h3 class="dh3">The GGUF trap</h3>
    <p class="note">Q4_K_M is not 4 bits. It mixes Q4_K and Q6_K tensor by tensor and lands near <b>4.9 effective bits</b>; assuming 4.0 undercounts a 70B model by about 8 GB. The three GGUF figures are whole-<em>file</em> averages measured across a finished checkpoint — GGUF quantises the embedding layers too, so those quants skip the 16-bit tail term below rather than paying it twice. GGUF is llama.cpp / Ollama territory; vLLM's support for it is experimental.</p>
    <p class="note">Models in the catalog without <code>hidden_size</code>/<code>vocab_size</code> fall back to the legacy <code>total × bytes × 1.02</code> estimate, and the sizing view labels the result approximate. Hugging Face import fills all three fields from <code>config.json</code>.</p>

    <h3 class="dh3">Mixed-precision checkpoints</h3>
    <p>Frontier low-bit releases are rarely uniform. NVIDIA ModelOpt's GLM-5.2 NVFP4 card says <em>"only MoE expert linears are quantized"</em>; DeepSeek-V4 ships <em>"MoE experts FP4, remaining params FP8"</em>. Of 425 published recipe variants, <b>32 explicitly name which tensors are quantised</b>. A model can therefore declare a <b>dense remainder</b> — attention, shared experts, router, dense MLP — and the precision it keeps:</p>
    <div class="formula">weights = quantised × bytes(quant) + dense × bytes(dense_quant) + tail × 2</div>
    <p class="note">GLM-5.2's dense block is 16.5 B parameters. Sized uniform, its NVFP4 checkpoint reads 0.75 of the published VRAM floor; keeping the dense block at 16-bit reads <b>0.80</b>. Quants with no declaration stay uniform.</p>

    <h3 class="dh3">Prefill activations</h3>
    <p>The runtime reserve is <b>not a constant</b>. A prefill chunk materialises activations for every token in it at once, so the peak scales with <code>chunk × hidden_size</code>:</p>
    <div class="formula">reserve = max( 2.5 GiB , 1.5 GiB context + chunk × hidden_size × 12 bytes )</div>
    <p>At vLLM's default chunk of 2048 this floors at the historical 2.5 GiB, so default plans size exactly as before. Raise it and it bites: Llama-3.3-70B (hidden 8192) needs 3.0 GiB at 16K, 4.5 GiB at 32K, <b>7.5 GiB at 64K</b> — memory no longer available for KV cache. A narrow model like GPT-OSS-120B (hidden 2880) stays floor-bound at the same chunk.</p>
    <p class="note">The 12-bytes figure folds qkv, MLP up/gate and residual copies into one multiple at 2-byte activations. Order-of-magnitude, not kernel-accurate — but it moves in the right direction, which a flat reserve does not.</p>

    <h2 class="dh">3 · KV cache &amp; concurrency</h2>
    <p>KV cache grows linearly with both sequence length and batch size — the real limiter for long-context, high-concurrency serving. Per-token size depends on the attention geometry:</p>
    <div class="formula">Bytes per token = 2 × layers × KV-heads × head-dim × Bytes per element</div>
    <p>The factor <b>2</b> covers the Key and Value tensors. <b>MLA</b> models (DeepSeek, Kimi) compress KV into a latent instead — <code>layers × 576 × bytes</code> — materially smaller. Per user request:</p>
    <div class="formula">KV per session (GiB) = <span class="frac"><span class="fnum">Bytes per token × Active tokens</span><span class="fden">1024³</span></span></div>
    <p>where <em>active tokens = context length × average utilisation</em>. The most sessions one pod can hold is then bounded by the free space from §2:</p>
    <div class="formula">Max pod concurrency = ⌊ <span class="frac"><span class="fnum">Free KV space</span><span class="fden">KV per session</span></span> ⌋</div>
    <h3 class="dh3">Choosing the tensor-parallel size</h3>
    <p>The obvious rule — smallest TP that holds the weights plus one request — <b>over-recommends hardware</b>. A bigger shard leaves proportionally more room for KV and packs far more sessions per replica, and what you pay for is <code>pods × TP</code>, not <code>TP</code>. Llama-3.3-70B at FP8, 128K/60%, 64 concurrent on H200: TP1 needs 16 GPUs, TP2 needs 10, TP4 and TP8 need <b>8</b>.</p>
    <p class="note">The engine evaluates every TP in the model's ladder and takes the cheapest total, breaking ties toward the <em>smaller</em> shard — same GPU count, less collective traffic. That makes this a cost/throughput objective; a latency-oriented planner would bias toward larger shards.</p>

    <h3 class="dh3">What crossing a node boundary costs</h3>
    <p>TP is not free at any width, but the price jumps at the node boundary. A tensor-parallel group performs an <b>all-reduce on every layer, for every token</b> — inside a node that rides NVLink at multiple TB/s; between nodes it rides InfiniBand or RoCE at a fraction of that, on the critical path of every forward pass. The throughput and TTFT figures here assume the collective is not the bottleneck, which stops being true once a replica spans nodes.</p>
    <p class="note">The Sizing tab draws this rather than asserting it: every GPU on one axis, grouped into node boxes, each replica a bar above them. A replica that fits inside a node is a bar inside one box; one that does not visibly spans the gap where the fabric sits. Replicas are independent — they share only the weights on storage and the router in front — so scaling <em>out</em> adds throughput without adding collective traffic, while scaling <em>up</em> past the node boundary adds both.</p>

    <h3 class="dh3">Local &amp; global attention</h3>
    <p>Not every layer attends over the whole context. Many models alternate <b>full-attention</b> layers with <b>sliding-window</b> layers that only ever look back a fixed number of tokens. The windowed layers' KV stops growing once the sequence passes the window, so long-context KV falls well below the naive all-layers-full figure:</p>
    <div class="formula">KV per request = perLayer × ( full × tokens + windowed × min(tokens, window) )</div>
    <p>GPT-OSS-120B is the clearest case in this catalog: 18 of its 36 layers are locally banded at 128 tokens. At 128K context that is the difference between <b>2.7 GiB</b> and <b>5.4 GiB</b> per request — and the difference between a plan with room for three concurrent requests on one H100 and a plan that looks dangerously tight with room for one. Gemma (one global layer every N) and Mistral v0.1 (every layer windowed) use the same trick with different patterns.</p>
    <p class="note">Because the windowed layers stop growing, the per-token KV figure shown on the Sizing tab is an <em>effective average</em> over the whole request, not a constant marginal rate. It still reconciles exactly: per-token × active tokens = per-request. Models with no window declared are treated as all-full-attention, which is the safe direction to be wrong in.</p>
    <h3 class="dh3">Linear &amp; recurrent layers</h3>
    <p>A third regime is spreading fast. Hybrid models replace most attention layers with a <b>recurrent</b> form — Kimi K3's KDA, Qwen3-Next, MiniMax — whose state is a fixed-size matrix per layer. It does not grow with the sequence <em>at all</em>:</p>
    <div class="formula">KV per request = perLayer × ( full × tokens + windowed × min(tokens, window) ) + linear × constant</div>
    <p>Qwen3.6-27B is 64 layers with only <b>16</b> cached — 4.9 GiB of KV at 256K instead of 19.2. Kimi K3 is 93 layers, of which only <b>24</b> keep a token-indexed cache; the other 69 are KDA, costing a flat ~414 MB per request whether the context is 1K or 1M. At its full 1M window that is <b>8.5 GiB</b> of KV instead of the 31.4 GiB an all-93-layer sizing would claim — the difference between fitting one 8×B300 node and not. The constant term dominates at short context and vanishes at long; both ends matter, so it is modelled rather than dropped.</p>

    <p class="note"><b>Sparse attention is not a memory saving.</b> Schemes like GLM-5.2's DSA (<code>index_topk: 2048</code>) choose which cached tokens each query attends to. That cuts attention <em>compute</em> — the KV cache still holds every token. Treating token-selection sparsity as eviction would under-size the deployment, so this tool does not model it as one.</p>

    <h2 class="dh">4 · Decode roofline throughput</h2>
    <p>For every token generated, the weights and active KV cache must be read from memory to the compute cores — so generation speed is bounded by achievable memory bandwidth (with a Memory-Bandwidth-Utilisation penalty, here 55%).</p>
    <div class="formula">Data read per step = Weight memory + (Active sequences × KV per session)</div>
    <div class="formula">Aggregate throughput (tok/s) = <span class="frac"><span class="fnum">Effective pod bandwidth</span><span class="fden">Data read per step</span></span> × Active sequences</div>
    <p class="note">Effective pod bandwidth = TP size × per-GPU bandwidth × MBU. The calculator also reports <b>time-to-first-token</b> (an indicative prefill estimate) and <b>per-request</b> tokens/s. For MoE models, only the <em>active</em> parameters are read per step.</p>

    <h3 class="dh3">Time to first token is a different problem</h3>
    <p>Decode is memory-bound; <b>prefill is not</b>. It runs the entire prompt through the network before emitting a token, so TTFT is bounded by arithmetic:</p>
    <div class="formula">prefill FLOPs = 2 × active params × tokens + 4 × hidden × ( full × tokens² + windowed × tokens × window + linear × tokens )</div>
    <div class="formula">TTFT = max( prefill FLOPs ÷ (TP × TFLOPS × speedup × MFU) , weight-streaming time )</div>
    <p>The attention term is not a correction — it is usually the larger half. Llama-3.3-70B prefilling 78,643 tokens spends <b>16.2 PFLOPs on attention against 11.1 on the matmuls</b>. Long-context TTFT is an attention problem, which is why the layer regimes matter here as much as for KV: a sliding-window layer costs <code>tokens × window</code> rather than <code>tokens²</code>, making GPT-OSS-120B's prefill <b>38% cheaper</b> than the same shape with full attention.</p>
    <p class="note">Prefill MFU is taken as 0.4. Sub-16-bit formats get a 2× tensor-core speedup, capped there even for 4-bit — Blackwell does better, but the catalog does not track GPU generation and under-promising TTFT is the safe direction. A SKU with no FLOPS figure falls back to the weight-streaming floor and is labelled as such rather than presented as a prefill estimate.</p>

    <h2 class="dh">5 · Worked example — Llama 3.3 70B</h2>
    <p>Host Llama 3.3 70B Instruct at FP8 · 10 concurrent sessions · 128K context at 60% utilisation · on 2× H200 (TP2). <span class="note-i">Reproduce it on the Sizing tab.</span></p>
    <div class="steps2">
      <div class="step"><div class="sh">1 · Usable memory</div>Usable/GPU = (141 × 0.90) − 2.5 = <b>124.4 GiB</b><br>Pod = 124.4 × 2 = <b>248.8 GiB</b></div>
      <div class="step"><div class="sh">2 · Weights &amp; free cache</div>Tail = 2×128,256×8,192 = 2.10 B @ fp16<br>(68.5 + 4.2) × 10⁹ B ÷ 2³⁰ ≈ <b>67.7 GiB</b><br>Free KV = 248.8 − 67.7 = <b>181.1 GiB</b></div>
      <div class="step"><div class="sh">3 · KV per session</div>Per token = 2×80×8×128×1 = 163,840 B (0.156 MiB)<br>Active = 131,072 × 0.60 = 78,643 tok<br>Session KV ≈ <b>12.0 GiB</b></div>
      <div class="step"><div class="sh">4 · Concurrency</div>⌊181.1 ÷ 12.0⌋ = <b>15 sessions/pod</b><br>15 ≥ 10 target → <b>1 pod (2 GPUs)</b></div>
      <div class="step"><div class="sh">5 · Throughput</div>Data/step ≈ (66.7 + 10×12) × 2³⁰ ≈ 201×10⁹ B<br>Bandwidth = 2×4.8×10¹² × 0.55 ≈ 5.28×10¹² B/s<br>≈ 38.1 ms/step → <b>≈ 262 tok/s</b></div>
      <div class="step"><div class="sh">6 · Headroom</div>(248.8 − 67.7 − 12.0) ÷ 248.8 = <b>68%</b><br>68% ≥ 10% → <b>fits, not tight</b></div>
    </div>

    <h2 class="dh">6 · Fits · tight · infeasible</h2>
    <p>A plan is <b>infeasible</b> when weights plus <em>one</em> request's KV can't fit even at the largest permitted TP size. Between that and a comfortable fit sits a band worth naming:</p>
    <div class="formula">Pod headroom = <span class="frac"><span class="fnum">Usable pod memory − Weights − KV per session</span><span class="fden">Usable pod memory</span></span> · Tight = headroom &lt; 10%</div>
    <p>A <b>tight</b> plan is arithmetically feasible but has no margin for the ±5% weight estimate, allocator fragmentation, or a prompt longer than the modelled average — the configuration that passes a spreadsheet and then OOMs on launch. Qwen3-32B at Q4_K_M on a single RTX 4090 is the canonical case: 18.6 GiB of weights against 19.1 GiB usable leaves <b>0.9% headroom</b> at 4K context and room for exactly one request. Push the context to 8K and it needs TP2, where it is comfortable again.</p>

    <h2 class="dh">7 · A note on units</h2>
    <p>Every memory figure in this tool is <b>GiB = 2³⁰ bytes</b> — the unit <code>nvidia-smi</code> reports and the one <code>gpu_memory_utilization</code> is applied against. Parameter counts are in billions (10⁹), so weights are converted explicitly: <code>params × bytes/param × 10⁹ ÷ 2³⁰</code>. Skipping that conversion (a common shortcut) makes weights read <b>7.4% larger</b> than they are relative to GPU capacity — conservative, but wrong, and it compounds against a KV figure that <em>was</em> converted.</p>
    <p class="note">Bandwidth is the exception: <code>bw_tbs</code> is decimal TB/s (10¹² B/s), as vendors quote it. The roofline therefore converts memory to raw bytes before dividing, rather than mixing the two scales.</p>

    <h2 class="dh">8 · Concurrency rubric</h2>
    <p>The rubric re-runs the <b>entire</b> sizing at each target concurrency — it is not a scaling of one result. TP selection depends on the target, so the cheapest shard width at 1 session is rarely the cheapest at 256. A row can change TP, pods and the tight verdict all at once, which a linear extrapolation would hide.</p>
    <p class="note">Per-request decode rate falls as concurrency rises (more sessions share the pod's bandwidth) while aggregate throughput and GPU count rise.</p>

    <h2 class="dh">9 · Cost model</h2>
    <p>Rental-rate arithmetic on an admin-set <code>$/GPU-hour</code> per SKU. Nothing is amortised; power, networking and storage are out of scope.</p>
    <div class="formula">Run rate ($/hr) = Σ<sub>SKU</sub> ( committed GPUs × $/GPU-hour ) · month = ×730 · year = ×8760</div>
    <div class="formula">$ per million tokens = <span class="frac"><span class="fnum">GPUs × $/GPU-hour × 1,000,000</span><span class="fden">tokens/sec × 3600</span></span></div>
    <p class="note">The per-million figure inherits the throughput estimate's ±40% band — a comparison tool between configurations, not a budget line. A config that halves GPU count but also halves throughput costs the same per token.</p>

    <h2 class="dh">10 · Fleet reconciliation and the capacity gate</h2>
    <p>A plan is checked against a declared fleet <b>per SKU</b>, on integer bytes:</p>
    <div class="formula">committed_bytes + available_bytes = fleet_bytes &nbsp;(invariant, per SKU)</div>
    <p>Two properties are deliberate. Commitments count <b>whole GPUs</b> — a replica occupying part of a card still retires the whole card, because vLLM does not share a GPU between deployments. And there is <b>no cross-SKU masking</b>: surplus H200s never offset an H100 shortage, so a plan is over-committed if <em>any</em> SKU is.</p>
    <p class="note">Integer bytes rather than floats keep this browser's live verdict and the server's authoritative one identical at the boundary.</p>

    <h2 class="dh">11 · Launch command</h2>
    <p>Every feasible plan emits the <code>vllm serve</code> command implied by its own numbers. <code>--max-num-seqs</code> is the one worth understanding: it is set to the <b>pod's</b> KV budget, not the deployment target. Left at vLLM's default the scheduler admits more sequences than the cache holds and preempts under load — which presents as a throughput problem and is really a sizing one.</p>
    <p class="note">The command describes <b>one replica</b>. A plan needing <em>n</em> pods needs <em>n</em> copies behind a load balancer; conflating the two is the classic way to under-provision.</p>

    <h2 class="dh">12 · Where catalog numbers come from</h2>
    <p>Model geometry is not guessed. Two sources, separated by what each is authoritative for:</p>
    <table class="qtab"><thead><tr><th>Source</th><th>Supplies</th></tr></thead><tbody>
      <tr><td>Hugging Face <code>config.json</code></td><td>layers, attention geometry, embedding sizes, sliding-window and linear splits</td></tr>
      <tr><td>recipes.vllm.ai</td><td>parameter counts, context length, shipped quantisations, TP sizes</td></tr>
    </tbody></table>
    <p class="note">The second set is precisely what a <code>config.json</code> never carries. A recipe has no authority over geometry and is not allowed to set it. Recipes also publish a <code>vram_minimum_gb</code> per variant, shown beside our own estimate on import — their floor covers weights + KV + overhead, so a ratio near <b>0.85</b> is healthy and above 1.00 means one of the two is wrong.</p>

    <h2 class="dh">Frequently asked</h2>
    <div class="faq">
      <div class="qa"><div class="q">Why memory-bound, not compute-bound?</div><div class="a">Decoding produces one token at a time and must re-read all weights + KV every step. The GPUs finish the arithmetic faster than HBM can feed them, so bandwidth — not FLOPs — sets the ceiling.</div></div>
      <div class="qa"><div class="q">Why does KV cache dominate at long context?</div><div class="a">Weights are fixed, but KV grows linearly with sequence length × concurrency. At 128K it can exceed the weights, becoming the limit on how many users a pod can serve.</div></div>
      <div class="qa"><div class="q">What is MBU?</div><div class="a">Memory-Bandwidth Utilisation — the fraction of peak HBM bandwidth actually achieved in practice (kernel efficiency, overheads). We use 0.55.</div></div>
      <div class="qa"><div class="q">GQA vs. MLA?</div><div class="a">Grouped-Query Attention shares KV across query heads: KV = 2 × layers × KV-heads × head-dim. Multi-head Latent Attention compresses KV to a small latent (layers × 576), so long-context KV is far smaller.</div></div>
      <div class="qa"><div class="q">Why are throughput and TTFT approximate?</div><div class="a">They're roofline estimates (±40% / ±50%). Real numbers depend on kernels, batching, prefix caching and speculative decoding — planning figures, not commitments. Validate against benchmarks before procurement.</div></div>
      <div class="qa"><div class="q">When is a model “infeasible”?</div><div class="a">If the weights plus one request's KV can't fit even at the largest permitted TP size, no valid deployment exists — the tool shows an infeasibility error rather than a plausible-but-wrong number.</div></div>
    </div>
    <p class="tot">These are the exact formulas the Sizing tab runs. Constants: runtime reserve 2.5 GiB · MBU 0.55 · MLA latent 576 · tight-fit threshold 10%. All memory in GiB (2³⁰ bytes).</p>
  </section>
{/if}
</main>

<style>
  :global(:root){--bg:#F2F4F3;--surface:#fff;--surface2:#F7F9F8;--ink:#15181A;--ink2:#535559;--ink3:#8A8F90;--line:#DCE0DF;--line2:#EAEDEC;--brand:#84BD00;--brandink:#5C8300;--wash:#EEF6DC;--purple:#5E366E;--grey:#C8C9C7;--slate:#3E4A52;--warn:#B26E00;--warnbg:#FBEFD6;--warnln:#DD8500;--err:#B3372B;--errbg:#FBE6E2;--errln:#C0362B;--okbg:#EEF6DC;}
  :global(:root[data-theme=dark]){--bg:#0F1213;--surface:#191D1F;--surface2:#1F2426;--ink:#E8EDEB;--ink2:#A8AEB0;--ink3:#767C7E;--line:#2A2F31;--line2:#232829;--brand:#9BD41E;--brandink:#A9DE3B;--wash:#233015;--purple:#B08CC0;--grey:#5A6062;--slate:#66757E;--warn:#E5A93E;--warnbg:#33280F;--warnln:#E0A030;--err:#E5786A;--errbg:#361F1B;--errln:#E5786A;--okbg:#233015;}
  :global(body){margin:0;background:var(--bg);color:var(--ink);font-family:Manrope,system-ui,Arial,sans-serif;font-size:14px;-webkit-font-smoothing:antialiased}
  :global(*){box-sizing:border-box}
  header{display:flex;justify-content:space-between;align-items:center;padding:12px 22px;background:var(--surface);border-bottom:2px solid var(--brand);position:sticky;top:0;z-index:5}
  .brand{font-weight:800;font-size:18px;display:flex;align-items:center;gap:9px}.brand .wm{color:var(--brandink);letter-spacing:-.01em}.brand .app{font-size:12px;color:var(--ink2);font-weight:600}
  .logo{width:22px;height:22px}
  .right{display:flex;gap:10px;align-items:center}
  .role{border:1px solid var(--line);background:var(--surface);color:var(--ink);border-radius:6px;padding:5px;font-size:12px}
  .chip{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;padding:2px 8px;border-radius:10px;background:var(--wash);color:var(--brandink)}.chip.user{background:var(--line);color:var(--ink2)}
  header button{border:1px solid var(--line);background:transparent;color:var(--ink2);border-radius:6px;padding:5px 9px;cursor:pointer;font-family:inherit}
  nav{display:flex;gap:2px;background:var(--surface);padding:0 18px;border-bottom:1px solid var(--line);overflow-x:auto;position:sticky;top:53px;z-index:4}
  .tab{border:none;background:none;padding:11px 13px;font-size:13px;font-weight:600;color:var(--ink2);border-bottom:2px solid transparent;cursor:pointer;white-space:nowrap;font-family:inherit}
  .tab.active{color:var(--ink);border-bottom-color:var(--brand)}.tab i{font-size:9px;background:var(--wash);color:var(--brandink);padding:1px 5px;border-radius:8px;margin-left:5px;font-style:normal}
  .banner{background:var(--wash);color:var(--brandink);padding:9px 22px;font-size:13px;cursor:pointer;display:flex;justify-content:space-between;font-weight:600}
  main{max-width:1120px;margin:18px auto;padding:0 20px 60px}
  .grid{display:grid;grid-template-columns:340px 1fr;gap:18px}@media(max-width:860px){.grid{grid-template-columns:1fr}}
  .panel{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:16px;margin-bottom:16px;box-shadow:0 1px 2px rgba(21,24,26,.05)}
  h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink2);margin:0 0 12px;font-weight:700}
  h2 small{color:var(--ink3);font-weight:600;text-transform:none;letter-spacing:0}
  label{display:block;font-size:11px;font-weight:600;color:var(--ink2);text-transform:uppercase;letter-spacing:.03em;margin:11px 0 4px}
  select,input{width:100%;padding:8px 9px;border:1px solid var(--line);background:var(--surface2);color:var(--ink);border-radius:5px;font-family:'IBM Plex Mono',ui-monospace,monospace;font-size:13px}
  select:focus,input:focus{outline:2px solid var(--brand);outline-offset:-1px}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:10px}.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
  .meta{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--ink2);background:var(--surface2);border:1px dashed var(--line);border-radius:5px;padding:8px 10px;margin-top:8px;line-height:1.6}
  .full{margin-top:14px;width:100%}
  .kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:16px}@media(max-width:980px){.kpis{grid-template-columns:repeat(3,1fr)}}@media(max-width:620px){.kpis{grid-template-columns:1fr 1fr}}
  .kpi{background:var(--surface);border:1px solid var(--line);border-top:3px solid var(--brand);border-radius:6px;padding:12px 13px;box-shadow:0 1px 2px rgba(21,24,26,.05)}
  .kpi.p{border-top-color:var(--purple)}.kpi.a{border-top-color:var(--warnln)}.kpi.g{border-top-color:var(--brandink)}.kpi.t{border-top-color:var(--slate)}
  .kpi .v{font-family:'IBM Plex Mono',monospace;font-size:23px;font-weight:600;line-height:1.05}.kpi small{font-size:12px;color:var(--ink3);font-weight:500}
  .kpi .l{font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:var(--ink3);margin-top:4px;font-weight:600}.kpi .cav{font-size:9.5px;color:var(--warn);margin-top:2px;font-weight:600}
  .kpi.tight{border-top-color:var(--warnln)}
  .kpi .verdict{font-size:9.5px;color:var(--brandink);margin-top:2px;font-weight:700}.kpi .verdict.t{color:var(--warn)}
  .tightv{color:var(--warn)}
  .hint{font-size:11px;color:var(--ink3);line-height:1.5;margin:-2px 0 10px}
  .cmd{font-family:'IBM Plex Mono',monospace;font-size:11.5px;line-height:1.6;background:var(--wash);border:1px solid var(--line);border-radius:6px;padding:12px 14px;overflow-x:auto;white-space:pre;margin:0;color:var(--ink)}
  .cmdnotes{margin:10px 0 0;padding-left:18px;font-size:11.5px;color:var(--ink2);line-height:1.6}
  .cmdnotes li{margin-bottom:4px}
  .hbm{display:flex;gap:10px;align-items:flex-end;padding:6px 2px 0;overflow-x:auto}
  .gpu{display:flex;flex-direction:column;align-items:center;min-width:52px}
  .stack{width:44px;height:168px;border:1.5px solid var(--ink);border-radius:3px;display:flex;flex-direction:column-reverse;overflow:hidden;background:repeating-linear-gradient(0deg,var(--surface2),var(--surface2) 9px,var(--line2) 9px,var(--line2) 10px)}
  /* Unused-but-available space is an outline of its own family rather than a new hue: it
     distinguishes by texture, which survives colour-blindness and greyscale printing. */
  .seg.w{background:var(--slate)}
  .seg.k{background:var(--purple)}
  .seg.kf{background:transparent;box-shadow:inset 0 0 0 1.5px var(--purple)}
  .seg.r{background:var(--grey)}
  .seg.u{background:transparent;box-shadow:inset 0 0 0 1.5px var(--line)}
  .gpu .cap{font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--ink3);margin-top:5px;text-align:center;line-height:1.3}
  .more{font-size:11px;color:var(--ink3);align-self:center}
  .legend{display:flex;gap:16px;margin-top:10px;font-size:11.5px;color:var(--ink2);flex-wrap:wrap}
  .legend span{display:inline-flex;align-items:center;gap:6px}.legend i{width:11px;height:11px;border-radius:2px;display:inline-block}.legend i.w{background:var(--slate)}.legend i.k{background:var(--purple)}.legend i.r{background:var(--grey)}.legend i.kf{background:transparent;box-shadow:inset 0 0 0 1.5px var(--purple)}.legend i.u{background:transparent;box-shadow:inset 0 0 0 1.5px var(--line)}
  .li{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid var(--line2);font-size:13px}.li:last-child{border-bottom:none}.li span{color:var(--ink2)}.li b{font-family:'IBM Plex Mono',monospace}
  .state{border-radius:7px;padding:11px 13px;margin-bottom:10px;font-size:12.5px;border:1px solid;line-height:1.5}
  .state.ok{background:var(--okbg);border-color:var(--brand);color:var(--brandink)}.state.warn{background:var(--warnbg);border-color:var(--warnln);color:var(--warn)}.state.err{background:var(--errbg);border-color:var(--errln);color:var(--err)}
  .state b{color:inherit}
  table{width:100%;border-collapse:collapse;font-size:13px;margin-top:12px}th{text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:.04em;color:var(--ink3);padding:8px;border-bottom:1px solid var(--line);font-weight:700}td{padding:9px 8px;border-bottom:1px solid var(--line2)}
  .num{text-align:right;font-family:'IBM Plex Mono',monospace;font-variant-numeric:tabular-nums}
  .btn{border:none;border-radius:6px;padding:7px 14px;font-size:12px;font-weight:700;cursor:pointer;font-family:inherit}.btn.primary{background:var(--brand);color:#15181A}.btn.ghost{background:var(--surface2);color:var(--ink);border:1px solid var(--line)}
  .empty{padding:28px;text-align:center;color:var(--ink3);border:1px dashed var(--line);border-radius:6px;margin-top:12px}
  .tot{font-size:12.5px;color:var(--ink2);margin-top:12px}
  .badge{font-size:9px;font-weight:700;padding:1px 6px;border-radius:8px;background:var(--wash);color:var(--brandink);vertical-align:middle}.badge.mla{background:var(--warnbg);color:var(--warn)}.badge.tightb{background:var(--warnbg);color:var(--warn);margin-left:6px}
  .meter{position:relative;height:14px;background:var(--line2);border-radius:7px;overflow:hidden}.meter .fill{height:100%;transition:width .2s}.meter .tick{position:absolute;top:0;left:100%;height:100%;border-left:1px dashed var(--ink3)}
  /* fleet-context prompt */
  .fleetctx{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;border-radius:8px;padding:12px 14px;margin-bottom:16px;font-size:13px;border:1px solid}
  .fleetctx.suggest{background:var(--wash);border-color:var(--brand);color:var(--brandink)}
  .fleetctx.active{background:var(--surface);border-color:var(--line);color:var(--ink2)}
  .fleetctx b{color:var(--ink)}.fleetctx.suggest b{color:var(--brandink)}
  .ctxactions{display:flex;gap:8px;align-items:center}.ctxactions select{width:auto;font-size:12px;padding:6px 8px}
  /* TP topology */
  .topo{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:4px 0}
  /* deployment topology — colours come from the app's own tokens, so dark mode is the
     same selection rather than an inverted copy. Status colour is reserved for the fabric
     crossing and always ships with a glyph and a label, never colour alone. */
  .topowrap{overflow-x:auto;padding:4px 0 2px}
  /* the SVG comes from {@html}, so Svelte's scoper cannot see these selectors and would strip
     them as unused — they are globalised, but still anchored to the component's own wrapper */
  .topowrap :global(.topo2){display:block;min-width:100%}
  .topowrap :global(.tbox){fill:var(--surface2);stroke:var(--line);stroke-width:1.5}
  .topowrap :global(.tbox.router){fill:var(--wash);stroke:var(--brand)}
  .topowrap :global(.tbox.pod){fill:var(--surface);stroke:var(--purple);stroke-width:2}
  .topowrap :global(.tbox.pod.spanning){fill:var(--warnbg);stroke:var(--warnln);stroke-dasharray:5 3}
  .topowrap :global(.tbox.node){fill:none;stroke:var(--line);stroke-dasharray:4 3}
  .topowrap :global(.tbox.sw){fill:var(--surface2);stroke:var(--slate)}
  .topowrap :global(.tbox.sw.hot){fill:var(--warnbg);stroke:var(--warnln);stroke-width:2}
  .topowrap :global(.tbox.store){fill:var(--surface2);stroke:var(--grey)}
  .topowrap :global(.tgpu){fill:var(--bg);stroke:var(--line);stroke-width:1.5}
  .topowrap :global(.tgpu.used){fill:var(--wash);stroke:var(--brand)}
  .topowrap :global(.tlink){stroke:var(--line);stroke-width:1.5;fill:none}
  .topowrap :global(.tlink.fabric){stroke:var(--warnln);stroke-dasharray:4 3}
  .topowrap :global(.tlabel){font-family:Manrope,system-ui,sans-serif;font-size:10px;font-weight:700;fill:var(--ink2)}
  .topowrap :global(.tgpulabel){font-family:'IBM Plex Mono',monospace;font-size:9px;font-weight:600;fill:var(--ink3)}
  .topowrap :global(.mid){text-anchor:middle}
  .topowrap :global(.tlabel.more){fill:var(--ink3);font-weight:600}
  .topolegend{display:flex;flex-wrap:wrap;gap:14px;font-size:11px;color:var(--ink2);margin-top:10px}
  .topolegend span{display:flex;align-items:center;gap:5px}
  .topolegend i{width:11px;height:11px;border-radius:3px;display:inline-block;border:1.5px solid var(--line)}
  .topolegend .sw-used{background:var(--wash);border-color:var(--brand)}
  .topolegend .sw-idle{background:var(--bg)}
  .topolegend .sw-pod{background:var(--surface);border-color:var(--purple)}
  .topolegend .warnitem{color:var(--warn);font-weight:700}
  /* admin catalog forms */
  .quants{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
  .qbtn{border:1px solid var(--line);background:var(--surface2);color:var(--ink2);border-radius:5px;padding:5px 10px;font-size:12px;font-weight:600;cursor:pointer;font-family:'IBM Plex Mono',monospace}
  .qbtn.on{background:var(--wash);border-color:var(--brand);color:var(--brandink)}
  .ferr{color:var(--err);font-size:12px;margin:6px 0 0;font-weight:600}
  .btn.danger{color:var(--err);border-color:var(--err)}
  input:disabled{opacity:.5;cursor:not-allowed}
  .replnote{font-size:12px;color:var(--ink2);margin:10px 0 0;padding-top:10px;border-top:1px solid var(--line2)}
  /* fleet utilisation (the stack) */
  .skuutil{margin-bottom:16px}
  .skuhead{display:flex;justify-content:space-between;align-items:baseline;gap:10px;flex-wrap:wrap;font-size:13px;margin-bottom:7px}.skuhead>span{font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:var(--ink2)}
  .cells{display:flex;flex-wrap:wrap;gap:3px;align-items:center}
  .cell{width:16px;height:16px;border-radius:3px;border:1.5px solid var(--line);background:var(--surface2)}
  .cell.used{background:var(--brand);border-color:var(--brand)}
  .cellk{width:12px;height:12px;border-radius:2px;border:1.5px solid var(--line);background:var(--surface2);display:inline-block}.cellk.used{background:var(--brand);border-color:var(--brand)}
  /* concurrency rubric */
  .rubric tr.cur{background:var(--wash)}
  .rubric tr.cur td{color:var(--brandink)}
  .rubric tr.infeasible{opacity:.6}
  /* model cards */
  .cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}
  .mcard{border:1px solid var(--line);border-radius:8px;background:var(--surface2);overflow:hidden}
  .mcard-h{padding:10px 12px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;gap:6px;font-size:13px}
  .mcard-b{padding:10px 12px;font-size:12px}
  .mcard-b>div{display:flex;justify-content:space-between;padding:3px 0;color:var(--ink2)}.mcard-b>div b{color:var(--ink);font-family:'IBM Plex Mono',monospace}
  .mcard-f{padding:8px 12px;border-top:1px solid var(--line);display:flex;gap:6px}
  /* hugging face import */
  .hfsuggest{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;font-size:11px;color:var(--ink3);align-items:center}
  .chip2{border:1px solid var(--line);background:var(--surface2);color:var(--ink2);border-radius:12px;padding:3px 9px;font-size:11px;cursor:pointer;font-family:'IBM Plex Mono',monospace}
  .chip2:hover{border-color:var(--brand);color:var(--brandink)}
  .hfcard{margin-top:12px;padding:11px 13px;background:var(--wash);border:1px solid var(--brand);border-radius:7px;font-size:12.5px;color:var(--brandink)}
  .secbar{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px}
  .expbtns{display:flex;gap:6px}
  /* methodology page */
  .doc{line-height:1.6;max-width:820px}
  .doctitle{font-size:24px;font-weight:800;letter-spacing:-.01em;margin:0 0 6px}
  .doc .lead{font-size:14px;color:var(--ink2);margin:0 0 8px}
  .doc h2.dh{font-size:15px;font-weight:700;text-transform:none;letter-spacing:0;color:var(--ink);margin:26px 0 8px;padding-top:14px;border-top:1px solid var(--line2)}
  .doc h3.dh3{font-size:13px;font-weight:700;color:var(--ink);margin:18px 0 6px}
  .doc .qtab{width:100%;border-collapse:collapse;font-size:12px;margin:8px 0 4px}
  .doc .qtab th{text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:var(--ink3);padding:5px 8px;border-bottom:1px solid var(--line)}
  .doc .qtab td{padding:5px 8px;border-bottom:1px solid var(--line2);color:var(--ink2)}
  .doc .qtab .num{text-align:right;font-family:'IBM Plex Mono',monospace}
  .doc p{font-size:13.5px;margin:8px 0}
  .doc code{font-family:'IBM Plex Mono',monospace;font-size:.86em;background:var(--surface2);padding:1px 5px;border-radius:3px}
  .formula{font-family:'IBM Plex Mono',monospace;font-size:13px;background:var(--surface2);border:1px solid var(--line);border-left:3px solid var(--brand);border-radius:5px;padding:11px 14px;margin:10px 0;overflow-x:auto;display:flex;align-items:center;flex-wrap:wrap;gap:2px}
  .frac{display:inline-flex;flex-direction:column;text-align:center;vertical-align:middle;margin:0 5px}
  .frac .fnum{border-bottom:1.5px solid currentColor;padding:0 7px 2px}
  .frac .fden{padding:2px 7px 0}
  .doc .note{font-size:12px;color:var(--ink3);background:var(--surface2);border-radius:5px;padding:8px 11px}
  .note-i{font-style:italic;color:var(--brandink)}
  .steps2{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin:12px 0}
  .step{background:var(--surface2);border:1px solid var(--line);border-top:3px solid var(--purple);border-radius:6px;padding:11px 13px;font-size:12.5px;font-family:'IBM Plex Mono',monospace;line-height:1.7}
  .step .sh{font-family:Manrope,sans-serif;font-weight:700;font-size:12px;color:var(--ink2);margin-bottom:6px;text-transform:uppercase;letter-spacing:.04em}
  .faq{display:flex;flex-direction:column;gap:2px}
  .qa{border-bottom:1px solid var(--line2);padding:11px 0}
  .qa .q{font-weight:700;font-size:13.5px;margin-bottom:4px}
  .qa .a{font-size:13px;color:var(--ink2)}
</style>
