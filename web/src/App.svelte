<script lang="ts">
  // vLLM Capacity Planner SPA. Sizing engine runs client-side (AD-1/AD-2); catalog,
  // reconciliation and saved configs go through the server API. Fleet+plan are session state.
  import { computeSizing, concurrencySweep, seedCatalog, type Model, type GpuSku, type FeasibleSizing } from '@vcp/domain';

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

  const model = $derived<Model>(catalog.models.find((m: Model) => m.id === modelId) ?? catalog.models[0]);
  const gpu = $derived<GpuSku>(catalog.gpus.find((g: GpuSku) => g.id === gpuId) ?? catalog.gpus[0]);
  const ctxChoices = $derived(CTXS.filter((c) => c <= (model?.max_ctx ?? 0)));
  // Effective quant — always one the selected model actually supports, so switching models can
  // never compute with an unsupported quant (e.g. FP16 on GLM-5.2, which only offers FP8/NVFP4).
  const effQuant = $derived(model && model.quants.includes(quant) ? quant : (model?.quants[0] ?? 'FP8'));
  const result = $derived(computeSizing(model, gpu, {
    quant: effQuant, kv_dtype_bytes: kvBytes, selected_ctx: Math.min(ctx, model?.max_ctx ?? ctx),
    avg_context_utilisation: util, target_concurrency: conc, mem_util_fraction: memUtil, gpus_per_node: perNode,
  }));
  const R = $derived(result.ok ? (result as FeasibleSizing) : null);
  const fmt = (x: number, d = 1) => (x >= 1000 ? Math.round(x).toLocaleString() : x.toFixed(d));

  // per-replica HBM split (matches the prototype)
  const activePer = $derived(R ? Math.min(R.concurrency_per_pod, Math.ceil(conc / R.pods)) : 0);
  const wPer = $derived(R ? R.weights_gb / R.tp : 0);
  const kvPer = $derived(R ? (activePer * R.kv_per_request_gb) / R.tp : 0);
  const kvAlloc = $derived(R ? activePer * R.kv_per_request_gb : 0);
  const stacks = $derived(R ? Math.min(R.tp, 8) : 0);
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
  const QUANTS = ['FP16', 'FP8', 'INT8', 'INT4', 'MXFP4', 'NVFP4'];
  type Err = { path: string; message: string };
  const blankModel = () => ({ id: '', name: '', total_params_b: 1, active_params_b: 1, layers: 32, kv_heads: 8, head_dim: 128, mla: false, max_ctx: 131072, tp_options: '1,2', quants: ['FP16'] as string[] });
  let mf = $state(blankModel());
  let mfEditing = $state(false);
  let mfErrors = $state<Err[]>([]);
  const errFor = (errs: Err[], p: string) => errs.find((e) => e.path === p)?.message;
  function editModel(m: Model) { mf = { ...m, tp_options: m.tp_options.join(','), quants: [...m.quants] } as any; mfEditing = true; mfErrors = []; document.getElementById('mform')?.scrollIntoView({ behavior: 'smooth' }); }
  function newModelForm() { mf = blankModel(); mfEditing = false; mfErrors = []; }
  function toggleQuant(q: string) { mf.quants = mf.quants.includes(q) ? mf.quants.filter((x) => x !== q) : [...mf.quants, q]; }
  async function saveModel() {
    const body = { id: mf.id, name: mf.name, total_params_b: +mf.total_params_b, active_params_b: +mf.active_params_b, layers: +mf.layers, kv_heads: +mf.kv_heads, head_dim: +mf.head_dim, mla: mf.mla, max_ctx: +mf.max_ctx, tp_options: String(mf.tp_options).split(',').map((s) => parseInt(s.trim(), 10)).filter((n) => n > 0), quants: mf.quants };
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
      mf = { id: m.id ?? '', name: m.name ?? hfId, total_params_b: 1, active_params_b: 1, layers: m.layers ?? 32, kv_heads: m.kv_heads ?? 8, head_dim: m.head_dim ?? 128, mla: !!m.mla, max_ctx: m.max_ctx ?? 131072, tp_options: '', quants: [] };
      mfEditing = false; mfErrors = [];
      notice = `Fetched ${d.model_id}. Review below, complete params / TP / quants (highlighted), then Create model.`;
      document.getElementById('mform')?.scrollIntoView({ behavior: 'smooth' });
    } else {
      notice = (await r.json()).error?.message ?? 'Fetch failed.';
    }
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
      <span>Sizing against <b>{activeConfig ?? 'a manually-defined fleet'}</b> — {fleetTotals().gpus} GPUs · {(fleetTotals().hbm / 1024).toFixed(1)} TB HBM</span>
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
      <div class="meta">total {model?.total_params_b} B · active {model?.active_params_b} B · layers {model?.layers} · {model?.mla ? 'MLA (latent 576/layer)' : `GQA ${model?.kv_heads} KV-heads × ${model?.head_dim}`} · max ctx {((model?.max_ctx ?? 0) / 1024)}K · TP {`{${model?.tp_options.join(',')}}`}</div>
      <div class="row"><label>Quantisation<select bind:value={quant}>{#each (model?.quants ?? []) as q}<option value={q}>{q}</option>{/each}</select></label>
        <label>KV dtype<select bind:value={kvBytes}><option value={1}>FP8 (1B)</option><option value={2}>FP16 (2B)</option></select></label></div>
      <div class="row"><label>Max context<select bind:value={ctx}>{#each ctxChoices as c}<option value={c}>{c / 1024}K</option>{/each}</select></label>
        <label>Context util<input type="number" min="0.05" max="1" step="0.05" bind:value={util} /></label></div>
      <div class="row"><label>Target concurrency<input type="number" bind:value={conc} /></label>
        <label>GPU SKU<select bind:value={gpuId}>{#each catalog.gpus as g}<option value={g.id}>{g.name}</option>{/each}</select></label></div>
      <div class="row"><label>GPU mem-util<input type="number" min="0.1" max="1" step="0.05" bind:value={memUtil} /></label>
        <label>GPUs / node<input type="number" bind:value={perNode} /></label></div>
      {#if R}<button class="btn primary full" onclick={addToPlan}>+ Add to cluster plan</button>{/if}
    </section>

    <section>
      {#if R}
        <div class="kpis">
          <div class="kpi"><div class="v">{R.gpus}</div><div class="l">GPUs</div></div>
          <div class="kpi p"><div class="v">{R.pods}<small> × TP{R.tp}</small></div><div class="l">Pods</div></div>
          <div class="kpi a"><div class="v">{fmt(kvAlloc)}<small> GB</small></div><div class="l">KV cache</div></div>
          <div class="kpi g"><div class="v">~{R.throughput_tokens_per_sec.toLocaleString()}</div><div class="l">Tokens/s</div><div class="cav">±40% · not a commitment</div></div>
        </div>

        <div class="panel">
          <h2>Per-GPU HBM allocation — one replica (TP{R.tp}, {R.tp} GPU{R.tp > 1 ? 's' : ''})</h2>
          <div class="hbm">
            {#each Array(stacks) as _, i}
              <div class="gpu">
                <div class="stack">
                  <div class="seg w" style="height:{pctOf(wPer)}" title="Weights {fmt(wPer)} GB"></div>
                  <div class="seg k" style="height:{pctOf(kvPer)}" title="KV {fmt(kvPer)} GB"></div>
                  <div class="seg r" style="height:{pctOf(Math.max(0, gpu.mem_gb - wPer - kvPer))}" title="Reserve"></div>
                </div>
                <div class="cap">GPU {i}<br>{gpu.mem_gb} GB</div>
              </div>
            {/each}
            {#if R.tp > 8}<div class="more">+{R.tp - 8}<br>more</div>{/if}
          </div>
          <div class="legend"><span><i class="w"></i>Weights {fmt(wPer)} GB</span><span><i class="k"></i>KV {fmt(kvPer)} GB</span><span><i class="r"></i>Reserve {fmt(Math.max(0, gpu.mem_gb - wPer - kvPer))} GB</span></div>
          <p class="replnote">This model runs as <b>{R.pods}</b> identical replica{R.pods > 1 ? 's' : ''} → {R.pods} × TP{R.tp} = <b>{R.gpus}</b> GPUs total{#if R.tp > 8} (showing 8 of {R.tp} GPUs in the replica){/if}.</p>
        </div>

        {#if R.tp > 1}
          <div class="panel">
            <h2>Tensor-parallel topology — one replica across {R.tp} GPUs{#if tpNodes > 1} · {tpNodes} nodes{/if}</h2>
            <div class="topo">
              {#each Array(tpNodes) as _, n}
                <div class="node">
                  <span class="nl">Node {n + 1}</span>
                  <div class="gcells">
                    {#each Array(perNode) as _, j}
                      {@const idx = n * perNode + j}
                      <div class="gcell" class:used={idx < R.tp}>{idx < R.tp ? 'G' + idx : ''}</div>
                    {/each}
                  </div>
                </div>
                {#if n < tpNodes - 1}<span class="tpspan">TP{R.tp} →</span>{/if}
              {/each}
            </div>
            <p class="tot">One serving replica splits weights + compute across <b>{R.tp}</b> GPUs{#if R.multi_node}, spanning {tpNodes} nodes — requires NVLink/IB fabric; latency &amp; MBU degrade vs single-node{/if}. The plan runs <b>{R.pods}</b> such replica{R.pods > 1 ? 's' : ''} = {R.gpus} GPUs total.</p>
          </div>
        {/if}

        <div class="panel">
          <h2>Breakdown</h2>
          <div class="li"><span>Weights ({effQuant})</span><b>{fmt(R.weights_gb)} GB</b></div>
          <div class="li"><span>KV per token ({kvBytes === 1 ? 'FP8' : 'FP16'})</span><b>{(R.kv_per_token_gb * 1024).toFixed(3)} MB</b></div>
          <div class="li"><span>KV per request ({ctx / 1024}K × {Math.round(util * 100)}%)</span><b>{fmt(R.kv_per_request_gb)} GB</b></div>
          <div class="li"><span>Concurrency per pod</span><b>{R.concurrency_per_pod} req</b></div>
          <div class="li"><span>Pods → GPUs → nodes ({perNode}/node)</span><b>{R.pods} → {R.gpus} → {R.nodes}</b></div>
          <div class="li"><span>Usable HBM per GPU ({memUtil})</span><b>{fmt(R.usable_gb)} GB</b></div>
          <div class="li"><span>Time to first token <small>(indicative, prefill)</small></span><b>~{R.ttft_ms} ms <small>±50%</small></b></div>
          <div class="li"><span>Decode throughput / request</span><b>~{R.decode_tps_per_request} tok/s</b></div>
          <div class="li"><span>Aggregate throughput</span><b>~{R.throughput_tokens_per_sec.toLocaleString()} tok/s <small>±40%</small></b></div>
        </div>

        <div class="panel">
          <h2>Concurrency rubric — pick a target, see the cost &amp; throughput</h2>
          <table class="rubric"><thead><tr><th class="num">Concurrency</th><th class="num">GPUs</th><th class="num">Pods</th><th class="num">TTFT</th><th class="num">tok/s · req</th><th class="num">tok/s · total</th><th></th></tr></thead><tbody>
            {#each sweep as s}
              <tr class:cur={s.concurrency === conc} class:infeasible={!s.feasible}>
                <td class="num">{s.concurrency}</td>
                {#if s.feasible}
                  <td class="num">{s.gpus}</td><td class="num">{s.pods} × TP{s.tp}</td><td class="num">~{s.ttft_ms} ms</td><td class="num">~{s.decode_tps_per_request}</td><td class="num">~{s.throughput_tokens_per_sec.toLocaleString()}</td>
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
        {#if R.multi_node}<div class="state warn"><b>TP {R.tp} &gt; {perNode} GPUs/node:</b> this replica spans nodes — needs NVLink/IB fabric; latency &amp; MBU degrade vs single-node TP.</div>{/if}
        {#if util >= 1}<div class="state warn">Sizing at 100% context utilisation buys worst-case memory that mostly sits idle. Size KV at P95 of observed sequence length.</div>{/if}
      {:else}
        <div class="state err"><b>Infeasible.</b> {(result as any).reason} (weights {fmt((result as any).weights_gb)} GB, KV/request {fmt((result as any).kv_per_request_gb)} GB)</div>
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
        {#each fleet as p, i}<tr><td>{p.node_count} node{p.node_count > 1 ? 's' : ''} × {p.gpus_per_node} × {gpuName(p.gpu_sku_id)}</td><td class="num">{p.node_count * p.gpus_per_node}</td><td class="num">{fmt(p.node_count * p.gpus_per_node * (catalog.gpus.find((g: GpuSku) => g.id === p.gpu_sku_id)?.mem_gb ?? 0) / 1024)} TB</td><td><button class="btn ghost" onclick={() => delPool(i)}>remove</button></td></tr>{/each}
      </tbody></table>
      <p class="tot">Fleet total: <b>{fleetTotals().gpus}</b> GPUs · {fleetTotals().nodes} nodes · {(fleetTotals().hbm / 1024).toFixed(1)} TB HBM</p>
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
            <div class="skuhead"><b>{r.name}</b><span>{r.committed} / {r.total} GPUs · {fmt(r.committedHbm / 1024)} / {fmt(r.totalHbm / 1024)} TB HBM · <b style="color:{r.over ? 'var(--err)' : r.util > 80 ? 'var(--warn)' : 'var(--brandink)'}">{r.total > 0 ? r.util.toFixed(0) + '%' : '—'}</b></span></div>
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
        <table><thead><tr><th>Deployment</th><th>SKU</th><th class="num">GPUs</th><th class="num">Pods</th><th class="num">KV GB</th><th></th></tr></thead><tbody>
          {#each plan as d, i}<tr><td>{d.label}</td><td>{gpuName(d.gpu_sku_id)}</td><td class="num">{d.gpus}</td><td class="num">{d.pods} × TP{d.tp}</td><td class="num">{fmt(d.kv)}</td><td><button class="btn ghost danger" onclick={() => delDeployment(i)}>remove</button></td></tr>{/each}
        </tbody></table>
        <p class="tot">Total demand: <b>{planTotals().gpus}</b> GPUs · {planTotals().pods} pods · {fmt(planTotals().kv)} GB KV</p>
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
      <div class="row3"><label>Memory (GB)<input type="number" bind:value={gf.mem_gb} /></label><label>Bandwidth (TB/s)<input type="number" step="0.1" bind:value={gf.bw_tbs} /></label><label>Price ($/GPU-hr)<input type="number" step="0.1" bind:value={gf.price_per_gpu_hour} /></label></div>
      {#if gfErrors.length}<div class="ferr">{gfErrors[0].message}</div>{/if}
      <button class="btn primary" style="margin-top:12px" onclick={saveGpu}>Add / update GPU SKU</button>
    </section>
  {/if}
  <section class="panel"><h2>GPU SKUs</h2>
    <table><thead><tr><th>SKU</th><th class="num">HBM (GB)</th><th class="num">BW (TB/s)</th><th class="num">$/GPU-hr</th>{#if ident.role === 'admin'}<th></th>{/if}</tr></thead><tbody>
      {#each catalog.gpus as g}<tr><td>{g.name}</td><td class="num">{g.mem_gb}</td><td class="num">{g.bw_tbs}</td><td class="num">{g.price_per_gpu_hour != null ? money(g.price_per_gpu_hour) : '—'}</td>{#if ident.role === 'admin'}<td><button class="btn ghost danger" onclick={() => deleteGpuUi(g.id)}>del</button></td>{/if}</tr>{/each}
    </tbody></table>
    <p class="tot">{catalog.gpus.length} GPU SKUs</p>
  </section>

{:else if tab === 'methodology'}
  <section class="panel doc">
    <h1 class="doctitle">How the calculator works</h1>
    <p class="lead">LLM capacity planning is deterministic maths, not guesswork. Because autoregressive decoding generates <em>one token at a time</em>, serving is <b>memory-bound</b> — the bottleneck is memory <em>bandwidth</em>, not compute. Every figure on the Sizing tab comes from the formulas below. Fixed constants: runtime reserve <b>2.5 GB</b>, weight overhead <b>×1.02</b>, MBU <b>0.55</b>, MLA latent <b>576</b>.</p>

    <h2 class="dh">1 · Hardware memory modelling</h2>
    <p>Start with how much high-bandwidth memory (HBM) the inference engine (e.g. vLLM) may actually use. The <code>gpu_memory_utilization</code> factor caps it; a fixed runtime reserve is subtracted to prevent out-of-memory (OOM) failures.</p>
    <div class="formula">Usable VRAM per GPU = (Physical capacity × Utilisation) − Runtime reserve</div>
    <p>Tensor Parallelism (TP) splits one model replica across several GPUs; their usable memory pools linearly:</p>
    <div class="formula">Usable pod memory = Usable VRAM per GPU × TP size</div>

    <h2 class="dh">2 · Weights vs. dynamic cache</h2>
    <p>A pod's memory is split between <b>static model weights</b> (the parameters) and the <b>dynamic KV cache</b> (per-token attention state during generation). Whatever remains after weights is the budget for concurrency:</p>
    <div class="formula">Free KV space = Usable pod memory − (Total parameters × Bytes per parameter)</div>
    <p class="note">Bytes per parameter follow the quantisation: FP16 = 2 · FP8/INT8 = 1 · INT4/MXFP4 = 0.5 · NVFP4 ≈ 0.625. For Mixture-of-Experts (MoE) models, <em>all</em> experts must be resident, so weights use the <b>total</b> parameter count.</p>

    <h2 class="dh">3 · KV cache &amp; concurrency</h2>
    <p>KV cache grows linearly with both sequence length and batch size — the real limiter for long-context, high-concurrency serving. Per-token size depends on the attention geometry:</p>
    <div class="formula">Bytes per token = 2 × layers × KV-heads × head-dim × Bytes per element</div>
    <p>The factor <b>2</b> covers the Key and Value tensors. <b>MLA</b> models (DeepSeek, Kimi) compress KV into a latent instead — <code>layers × 576 × bytes</code> — materially smaller. Per user request:</p>
    <div class="formula">KV per session (GB) = <span class="frac"><span class="fnum">Bytes per token × Active tokens</span><span class="fden">1024³</span></span></div>
    <p>where <em>active tokens = context length × average utilisation</em>. The most sessions one pod can hold is then bounded by the free space from §2:</p>
    <div class="formula">Max pod concurrency = ⌊ <span class="frac"><span class="fnum">Free KV space</span><span class="fden">KV per session</span></span> ⌋</div>

    <h2 class="dh">4 · Decode roofline throughput</h2>
    <p>For every token generated, the weights and active KV cache must be read from memory to the compute cores — so generation speed is bounded by achievable memory bandwidth (with a Memory-Bandwidth-Utilisation penalty, here 55%).</p>
    <div class="formula">Data read per step = Weight memory + (Active sequences × KV per session)</div>
    <div class="formula">Aggregate throughput (tok/s) = <span class="frac"><span class="fnum">Effective pod bandwidth</span><span class="fden">Data read per step</span></span> × Active sequences</div>
    <p class="note">Effective pod bandwidth = TP size × per-GPU bandwidth × MBU. The calculator also reports <b>time-to-first-token</b> (an indicative prefill estimate) and <b>per-request</b> tokens/s. For MoE models, only the <em>active</em> parameters are read per step.</p>

    <h2 class="dh">5 · Worked example — Llama 3.3 70B</h2>
    <p>Host Llama 3.3 70B Instruct at FP8 · 10 concurrent sessions · 128K context at 60% utilisation · on 2× H200 (TP2). <span class="note-i">Reproduce it on the Sizing tab.</span></p>
    <div class="steps2">
      <div class="step"><div class="sh">1 · Usable memory</div>Usable/GPU = (141 × 0.90) − 2.5 = <b>124.4 GB</b><br>Pod = 124.4 × 2 = <b>248.8 GB</b></div>
      <div class="step"><div class="sh">2 · Weights &amp; free cache</div>Weights (FP8) ≈ <b>72.0 GB</b><br>Free KV = 248.8 − 72.0 = <b>176.8 GB</b></div>
      <div class="step"><div class="sh">3 · KV per session</div>Per token = 2×80×8×128×1 = 163,840 B (0.156 MB)<br>Active = 131,072 × 0.60 = 78,643 tok<br>Session KV ≈ <b>12.0 GB</b></div>
      <div class="step"><div class="sh">4 · Concurrency</div>⌊176.8 ÷ 12.0⌋ = <b>14 sessions/pod</b><br>14 ≥ 10 target → <b>1 pod (2 GPUs)</b></div>
      <div class="step"><div class="sh">5 · Throughput</div>Data/step = 72 + 10×12 = 192 GB<br>Bandwidth = 2×4.8 TB/s × 0.55 ≈ 5,280 GB/s<br>(5,280 ÷ 192) × 10 ≈ <b>275 tok/s</b></div>
    </div>

    <h2 class="dh">Frequently asked</h2>
    <div class="faq">
      <div class="qa"><div class="q">Why memory-bound, not compute-bound?</div><div class="a">Decoding produces one token at a time and must re-read all weights + KV every step. The GPUs finish the arithmetic faster than HBM can feed them, so bandwidth — not FLOPs — sets the ceiling.</div></div>
      <div class="qa"><div class="q">Why does KV cache dominate at long context?</div><div class="a">Weights are fixed, but KV grows linearly with sequence length × concurrency. At 128K it can exceed the weights, becoming the limit on how many users a pod can serve.</div></div>
      <div class="qa"><div class="q">What is MBU?</div><div class="a">Memory-Bandwidth Utilisation — the fraction of peak HBM bandwidth actually achieved in practice (kernel efficiency, overheads). We use 0.55.</div></div>
      <div class="qa"><div class="q">GQA vs. MLA?</div><div class="a">Grouped-Query Attention shares KV across query heads: KV = 2 × layers × KV-heads × head-dim. Multi-head Latent Attention compresses KV to a small latent (layers × 576), so long-context KV is far smaller.</div></div>
      <div class="qa"><div class="q">Why are throughput and TTFT approximate?</div><div class="a">They're roofline estimates (±40% / ±50%). Real numbers depend on kernels, batching, prefix caching and speculative decoding — planning figures, not commitments. Validate against benchmarks before procurement.</div></div>
      <div class="qa"><div class="q">When is a model “infeasible”?</div><div class="a">If the weights plus one request's KV can't fit even at the largest permitted TP size, no valid deployment exists — the tool shows an infeasibility error rather than a plausible-but-wrong number.</div></div>
    </div>
    <p class="tot">These are the exact formulas the Sizing tab runs. Constants: runtime reserve 2.5 GB · weight overhead ×1.02 · MBU 0.55 · MLA latent 576.</p>
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
  label{display:block;font-size:11px;font-weight:600;color:var(--ink2);text-transform:uppercase;letter-spacing:.03em;margin:11px 0 4px}
  select,input{width:100%;padding:8px 9px;border:1px solid var(--line);background:var(--surface2);color:var(--ink);border-radius:5px;font-family:'IBM Plex Mono',ui-monospace,monospace;font-size:13px}
  select:focus,input:focus{outline:2px solid var(--brand);outline-offset:-1px}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:10px}.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
  .meta{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--ink2);background:var(--surface2);border:1px dashed var(--line);border-radius:5px;padding:8px 10px;margin-top:8px;line-height:1.6}
  .full{margin-top:14px;width:100%}
  .kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}@media(max-width:620px){.kpis{grid-template-columns:1fr 1fr}}
  .kpi{background:var(--surface);border:1px solid var(--line);border-top:3px solid var(--brand);border-radius:6px;padding:12px 13px;box-shadow:0 1px 2px rgba(21,24,26,.05)}
  .kpi.p{border-top-color:var(--purple)}.kpi.a{border-top-color:var(--warnln)}.kpi.g{border-top-color:var(--brandink)}
  .kpi .v{font-family:'IBM Plex Mono',monospace;font-size:23px;font-weight:600;line-height:1.05}.kpi small{font-size:12px;color:var(--ink3);font-weight:500}
  .kpi .l{font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:var(--ink3);margin-top:4px;font-weight:600}.kpi .cav{font-size:9.5px;color:var(--warn);margin-top:2px;font-weight:600}
  .hbm{display:flex;gap:10px;align-items:flex-end;padding:6px 2px 0;overflow-x:auto}
  .gpu{display:flex;flex-direction:column;align-items:center;min-width:52px}
  .stack{width:44px;height:168px;border:1.5px solid var(--ink);border-radius:3px;display:flex;flex-direction:column-reverse;overflow:hidden;background:repeating-linear-gradient(0deg,var(--surface2),var(--surface2) 9px,var(--line2) 9px,var(--line2) 10px)}
  .seg.w{background:var(--slate)}.seg.k{background:var(--purple)}.seg.r{background:var(--grey)}
  .gpu .cap{font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--ink3);margin-top:5px;text-align:center;line-height:1.3}
  .more{font-size:11px;color:var(--ink3);align-self:center}
  .legend{display:flex;gap:16px;margin-top:10px;font-size:11.5px;color:var(--ink2);flex-wrap:wrap}
  .legend span{display:inline-flex;align-items:center;gap:6px}.legend i{width:11px;height:11px;border-radius:2px;display:inline-block}.legend i.w{background:var(--slate)}.legend i.k{background:var(--purple)}.legend i.r{background:var(--grey)}
  .li{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid var(--line2);font-size:13px}.li:last-child{border-bottom:none}.li span{color:var(--ink2)}.li b{font-family:'IBM Plex Mono',monospace}
  .state{border-radius:7px;padding:11px 13px;margin-bottom:10px;font-size:12.5px;border:1px solid;line-height:1.5}
  .state.ok{background:var(--okbg);border-color:var(--brand);color:var(--brandink)}.state.warn{background:var(--warnbg);border-color:var(--warnln);color:var(--warn)}.state.err{background:var(--errbg);border-color:var(--errln);color:var(--err)}
  .state b{color:inherit}
  table{width:100%;border-collapse:collapse;font-size:13px;margin-top:12px}th{text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:.04em;color:var(--ink3);padding:8px;border-bottom:1px solid var(--line);font-weight:700}td{padding:9px 8px;border-bottom:1px solid var(--line2)}
  .num{text-align:right;font-family:'IBM Plex Mono',monospace;font-variant-numeric:tabular-nums}
  .btn{border:none;border-radius:6px;padding:7px 14px;font-size:12px;font-weight:700;cursor:pointer;font-family:inherit}.btn.primary{background:var(--brand);color:#15181A}.btn.ghost{background:var(--surface2);color:var(--ink);border:1px solid var(--line)}
  .empty{padding:28px;text-align:center;color:var(--ink3);border:1px dashed var(--line);border-radius:6px;margin-top:12px}
  .tot{font-size:12.5px;color:var(--ink2);margin-top:12px}
  .badge{font-size:9px;font-weight:700;padding:1px 6px;border-radius:8px;background:var(--wash);color:var(--brandink);vertical-align:middle}.badge.mla{background:var(--warnbg);color:var(--warn)}
  .meter{position:relative;height:14px;background:var(--line2);border-radius:7px;overflow:hidden}.meter .fill{height:100%;transition:width .2s}.meter .tick{position:absolute;top:0;left:100%;height:100%;border-left:1px dashed var(--ink3)}
  /* fleet-context prompt */
  .fleetctx{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;border-radius:8px;padding:12px 14px;margin-bottom:16px;font-size:13px;border:1px solid}
  .fleetctx.suggest{background:var(--wash);border-color:var(--brand);color:var(--brandink)}
  .fleetctx.active{background:var(--surface);border-color:var(--line);color:var(--ink2)}
  .fleetctx b{color:var(--ink)}.fleetctx.suggest b{color:var(--brandink)}
  .ctxactions{display:flex;gap:8px;align-items:center}.ctxactions select{width:auto;font-size:12px;padding:6px 8px}
  /* TP topology */
  .topo{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:4px 0}
  .node{border:1px dashed var(--line);border-radius:7px;padding:9px;background:var(--surface2)}
  .node .nl{font-size:9px;color:var(--ink3);text-transform:uppercase;letter-spacing:.08em;display:block;margin-bottom:6px;font-weight:700}
  .gcells{display:flex;gap:5px}
  .gcell{width:34px;height:34px;border-radius:5px;border:1.5px solid var(--line);background:var(--bg);display:grid;place-items:center;font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:600;color:var(--ink3)}
  .gcell.used{background:var(--wash);border-color:var(--brand);color:var(--brandink)}
  .tpspan{font-size:10px;color:var(--ink3);font-family:'IBM Plex Mono',monospace}
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
