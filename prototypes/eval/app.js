(() => {
  const state = { payload: null, view: 'summary', sortKey: 'scheduler', sortDir: 1 };
  const $ = (id) => document.getElementById(id);
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number = (value, digits = 2) => value == null || Number.isNaN(Number(value)) ? '—' : Number(value).toLocaleString('zh-CN', { maximumFractionDigits: digits });
  const metric = (value, digits = 2) => value == null ? '—' : number(value, digits);
  const selected = (id) => $(id).value;
  const query = () => {
    const params = new URLSearchParams({ mode: selected('mode') });
    for (const id of ['scheduler', 'rate', 'config', 'seed']) if (selected(id)) params.set(id, selected(id));
    return params;
  };
  const setStatus = (message, error = false) => { $('status').textContent = message; $('status').classList.toggle('error', error); };

  function options() {
    const data = state.payload?.options || {};
    const fill = (id, values, labels = (value) => value, blank = '全部') => {
      const select = $(id); const previous = select.value;
      select.innerHTML = `<option value="">${blank}</option>` + (values || []).map((value) => `<option value="${esc(value.id ?? value)}">${esc(labels(value))}</option>`).join('');
      if ([...select.options].some((option) => option.value === previous)) select.value = previous;
    };
    fill('scheduler', data.schedulers || []); fill('rate', data.rates || [], (value) => `${value} req/s`);
    fill('seed', data.seeds || []); fill('config', data.configs || [], (value) => value.label, '全部配置');
    const pivot = $('pivot-rate'); const oldPivot = pivot.value;
    pivot.innerHTML = (data.rates || []).map((value) => `<option value="${esc(value)}">${esc(value)} req/s</option>`).join('');
    if (selected('rate') && [...pivot.options].some((option) => option.value === selected('rate'))) pivot.value = selected('rate');
    else if ([...pivot.options].some((option) => option.value === oldPivot)) pivot.value = oldPivot;
  }

  async function load() {
    setStatus('正在读取结果…');
    try {
      const response = await fetch(`/api/data?${query().toString()}`, { cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      state.payload = await response.json(); options(); render();
      const d = state.payload.discovery || {};
      const warning = (d.incomplete_count || d.error_count) ? `；进行中/异常 ${Number(d.incomplete_count || 0) + Number(d.error_count || 0)} 个未纳入` : '';
      setStatus(`已加载 ${d.selected_runs || 0} 个 run${warning}`);
    } catch (error) { state.payload = null; setStatus(`加载失败：${error.message}`, true); render(); }
  }

  function configTitle(row) { return row.config_detail ? JSON.stringify(row.config_detail) : row.config; }
  function configCell(row, suffix = '') { return `<div class="config-cell" title="${esc(configTitle(row))}"><div class="config-name">${esc(row.config)}</div>${suffix ? `<span class="subline">${suffix}</span>` : ''}</div>`; }
  function formatConfig(row) { return configCell(row, `${row.seeds?.length || 0} 个 seed · ${row.run_count || 0} run`); }
  function sortableRows(rows) {
    const key = state.sortKey; const direction = state.sortDir;
    return [...rows].sort((a, b) => {
      const av = a[key] == null ? '' : a[key]; const bv = b[key] == null ? '' : b[key];
      if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * direction;
      return String(av).localeCompare(String(bv), 'zh-CN', { numeric: true }) * direction;
    });
  }
  function sortButton(key, label) { const cls = state.sortKey === key ? (state.sortDir > 0 ? 'asc' : 'desc') : ''; return `<button class="sort ${cls}" data-sort="${key}">${label}</button>`; }
  const tableEmpty = (message = '当前筛选没有数据。') => `<div class="empty">${esc(message)}</div>`;

  function renderSummary() {
    const rows = sortableRows(state.payload?.rows || []);
    $('summary-meta').textContent = `${rows.length} 个汇总行；每行合并完成请求 ${rows.reduce((total, row) => total + (row.completed_count || 0), 0).toLocaleString()} 条`;
    if (!rows.length) { $('summary-table').innerHTML = tableEmpty(); return; }
    const head = `<thead><tr><th>${sortButton('scheduler', 'Scheduler')}</th><th>${sortButton('rate', '请求速率')}</th><th>配置</th><th>${sortButton('quality_mean', '质量均值')}</th><th>${sortButton('quality_p95', '质量 p95')}</th><th>${sortButton('resource_mean_mib', '资源均值（MiB）')}</th><th>${sortButton('resource_p95_mib', '资源 p95（MiB）')}</th><th>${sortButton('latency_mean_s', '时延均值（秒）')}</th><th>${sortButton('latency_p95_s', '时延 p95（秒）')}</th><th>样本 / 失败</th></tr></thead>`;
    const body = rows.map((row) => `<tr><td class="primary-cell">${esc(row.scheduler)}</td><td>${esc(row.rate)}</td><td>${formatConfig(row)}</td><td>${metric(row.quality_mean, 4)}</td><td>${metric(row.quality_p95, 4)}</td><td>${metric(row.resource_mean_mib)}</td><td>${metric(row.resource_p95_mib)}</td><td>${metric(row.latency_mean_s, 3)}</td><td>${metric(row.latency_p95_s, 3)}</td><td>${number(row.completed_count, 0)} <span class="muted">/ ${number(row.failed_count, 0)}</span></td></tr>`).join('');
    $('summary-table').innerHTML = `<table>${head}<tbody>${body}</tbody></table>`;
  }

  function filteredRowsForRate(rate) { return (state.payload?.rows || []).filter((row) => String(row.rate) === String(rate)); }
  const pivotMetrics = [['quality_mean', '质量均值', 4], ['quality_p95', '质量 p95', 4], ['resource_mean_mib', '资源均值（MiB）', 2], ['resource_p95_mib', '资源 p95（MiB）', 2], ['latency_mean_s', '时延均值（秒）', 3], ['latency_p95_s', '时延 p95（秒）', 3]];
  function renderFixed() {
    const rate = selected('pivot-rate'); const rows = filteredRowsForRate(rate);
    if (!rows.length) { $('fixed-table').innerHTML = tableEmpty('该速率下没有可比较的数据。'); return; }
    const groups = [...new Map(rows.map((row) => [`${row.scheduler}|${row.config_id}`, row])).values()].sort((a, b) => `${a.scheduler}|${a.config}`.localeCompare(`${b.scheduler}|${b.config}`, 'zh-CN'));
    $('fixed-table').innerHTML = `<table><thead><tr><th>Scheduler</th><th>配置</th>${pivotMetrics.map(([, label]) => `<th>${label}</th>`).join('')}<th>样本 / 失败</th></tr></thead><tbody>${groups.map((row) => `<tr><td class="primary-cell">${esc(row.scheduler)}</td><td>${configCell(row)}</td>${pivotMetrics.map(([key, , digits]) => `<td>${metric(row[key], digits)}</td>`).join('')}<td>${number(row.completed_count, 0)} / ${number(row.failed_count, 0)}</td></tr>`).join('')}</tbody></table>`;
  }

  function renderMatrix() {
    const metricKey = $('matrix-metric').value; const label = $('matrix-metric').selectedOptions[0]?.textContent || '';
    const rows = state.payload?.rows || []; const rates = [...new Set(rows.map((row) => row.rate))].sort((a, b) => Number(a) - Number(b)); const groups = [...new Map(rows.map((row) => [`${row.scheduler}|${row.config_id}`, row])).values()].sort((a, b) => `${a.scheduler}|${a.config}`.localeCompare(`${b.scheduler}|${b.config}`, 'zh-CN'));
    const map = new Map(rows.map((row) => [`${row.scheduler}|${row.config_id}|${row.rate}`, row[metricKey]]));
    if (!rates.length) { $('matrix-table').innerHTML = tableEmpty(); return; }
    $('matrix-table').innerHTML = `<table><thead><tr><th>Scheduler</th><th>配置</th>${rates.map((rate) => `<th>${esc(rate)} req/s<br><span class="muted">${esc(label)}</span></th>`).join('')}</tr></thead><tbody>${groups.map((row) => `<tr><td class="primary-cell">${esc(row.scheduler)}</td><td>${configCell(row)}</td>${rates.map((rate) => `<td>${metric(map.get(`${row.scheduler}|${row.config_id}|${rate}`), metricKey.includes('quality') ? 4 : metricKey.includes('latency') ? 3 : 2)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
  }

  function renderRuns() {
    const runs = state.payload?.runs || [];
    if (!runs.length) { $('runs-table').innerHTML = tableEmpty('当前筛选没有 run。'); return; }
    const rows = runs.sort((a, b) => String(b.timestamp).localeCompare(String(a.timestamp)));
    $('runs-table').innerHTML = `<table><thead><tr><th>Scheduler</th><th>速率</th><th>Seed</th><th>配置</th><th>时间</th><th>完成 / 失败</th><th>质量均值 / p95</th><th>资源均值 / p95</th><th>时延均值 / p95（秒）</th></tr></thead><tbody>${rows.map((run) => { const m = run.metrics || {}; return `<tr><td class="primary-cell">${esc(run.scheduler)}</td><td>${esc(run.rate)}</td><td>${esc(run.seed)}</td><td>${configCell({ config: run.config_label, config_detail: run.config_detail })}</td><td class="mono">${esc(run.timestamp)}</td><td>${number(run.completed_count, 0)} / ${number(run.failed_count, 0)}</td><td>${metric(m.quality?.mean, 4)} / ${metric(m.quality?.p95, 4)}</td><td>${metric(m.resource?.mean)} / ${metric(m.resource?.p95)}</td><td>${metric(m.latency?.mean, 3)} / ${metric(m.latency?.p95, 3)}</td></tr>`; }).join('')}</tbody></table>`;
  }

  function renderCards() {
    const rows = state.payload?.rows || []; const d = state.payload?.discovery || {}; const completed = rows.reduce((sum, row) => sum + (row.completed_count || 0), 0); const failed = rows.reduce((sum, row) => sum + (row.failed_count || 0), 0);
    $('summary-cards').innerHTML = [['汇总行', rows.length], ['完成请求', completed.toLocaleString()], ['失败请求', failed.toLocaleString()], ['纳入 run', d.selected_runs || 0]].map(([label, value]) => `<div class="card"><strong>${esc(value)}</strong><span>${label}</span></div>`).join('');
  }
  function render() { renderCards(); renderSummary(); renderFixed(); renderMatrix(); renderRuns(); document.querySelectorAll('.view').forEach((view) => view.classList.toggle('active', view.id === `view-${state.view}`)); document.querySelectorAll('.tab').forEach((tab) => tab.classList.toggle('active', tab.dataset.view === state.view)); }
  function downloadCsv(headers, rows, name) { if (!rows.length) return; const csv = '\ufeff' + [headers.map(([, label]) => label), ...rows.map((row) => headers.map(([key]) => row[key] ?? ''))].map((line) => line.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(',')).join('\n'); const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' }); const url = URL.createObjectURL(blob); const link = document.createElement('a'); link.href = url; link.download = name; link.click(); URL.revokeObjectURL(url); }
  function exportCsv() {
    const summaryHeaders = [['scheduler','scheduler'],['config','配置'],['rate','请求速率'],['quality_mean','质量均值'],['quality_p95','质量 p95'],['resource_mean_mib','资源均值（MiB）'],['resource_p95_mib','资源 p95（MiB）'],['latency_mean_s','时延均值（秒）'],['latency_p95_s','时延 p95（秒）'],['completed_count','完成数'],['failed_count','失败数']];
    if (state.view === 'summary') return downloadCsv(summaryHeaders, state.payload?.rows || [], `scheduler-evaluation-${selected('mode')}.csv`);
    if (state.view === 'fixed') { const rows = filteredRowsForRate(selected('pivot-rate')); return downloadCsv(summaryHeaders, rows, `scheduler-evaluation-rate-${selected('pivot-rate')}.csv`); }
    if (state.view === 'runs') { const headers = [['scheduler','scheduler'],['config_label','配置'],['rate','请求速率'],['seed','seed'],['timestamp','时间'],['completed_count','完成数'],['failed_count','失败数']]; return downloadCsv(headers, state.payload?.runs || [], `scheduler-evaluation-runs-${selected('mode')}.csv`); }
    const metricKey = $('matrix-metric').value; const headers = [['scheduler','scheduler'],['config','配置'],['rate','请求速率'],[metricKey, $('matrix-metric').selectedOptions[0]?.textContent || metricKey]]; const rows = state.payload?.rows || []; return downloadCsv(headers, rows.map((row) => ({ scheduler: row.scheduler, config: row.config, rate: row.rate, [metricKey]: row[metricKey] })), `scheduler-evaluation-matrix-${metricKey}.csv`);
  }

  document.addEventListener('change', (event) => { if (['mode','scheduler','rate','config','seed'].includes(event.target.id)) load(); else if (event.target.id === 'pivot-rate' || event.target.id === 'matrix-metric') render(); });
  document.addEventListener('click', (event) => { const tab = event.target.closest('.tab'); if (tab) { state.view = tab.dataset.view; render(); } const sort = event.target.closest('[data-sort]'); if (sort) { const key = sort.dataset.sort; state.sortDir = state.sortKey === key ? -state.sortDir : 1; state.sortKey = key; renderSummary(); } });
  $('refresh').addEventListener('click', load); $('export').addEventListener('click', exportCsv); load();
})();
