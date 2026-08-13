import { api, showToast, escapeHtml, fmtMs, fmtNum, fmtClock, hasCharts } from './utils.js';

if (hasCharts) Chart.defaults.font.family = "'Inter', sans-serif";

const telemetryModal = document.getElementById('telemetryModal');
const telemetryBody = document.getElementById('telemetryBody');
const telemetryStatus = document.getElementById('telemetryStatus');
const telemetryRefresh = document.getElementById('telemetryRefresh');
const telemetryClear = document.getElementById('telemetryClear');
const telemetryModalClose = document.getElementById('telemetryModalClose');
const analyticsBtn = document.getElementById('analyticsBtn');

let currentTelemetryEvent = null;

export function openTelemetryModal() {
  telemetryModal.hidden = false;
  currentTelemetryEvent = null;
  loadTelemetry();
}
export function closeTelemetryModal() { telemetryModal.hidden = true; currentTelemetryEvent = null; }

let activeCharts = [];

function destroyCharts() {
  activeCharts.forEach(c => { try { c.destroy(); } catch (_) {} });
  activeCharts = [];
}

const CHART_TEXT = '#9a9a9d';
const CHART_GRID = 'rgba(255,255,255,0.06)';
const CHART_LINE = 'rgba(255,255,255,0.85)';
const CHART_DANGER = '#e5645f';
const CHART_TOOLTIP = {
  backgroundColor: '#141416',
  borderColor: 'rgba(255,255,255,0.14)',
  borderWidth: 1,
  titleColor: '#ededed',
  bodyColor: '#dcdcde',
};

// last 24 whole hours ending at the current hour, so the line is continuous
// instead of jumping between sparse buckets. backend hours are naive UTC, so
// labels use UTC too.
function build24hSeries(timeline, key) {
  const byHour = new Map((timeline || []).map(t => [t.hour, t]));
  const labels = [];
  const values = [];
  const now = new Date();
  now.setMinutes(0, 0, 0);
  for (let i = 23; i >= 0; i--) {
    const d = new Date(now.getTime() - i * 3600 * 1000);
    labels.push(d.toLocaleString(undefined, { hour: '2-digit', timeZone: 'UTC' }));
    const iso = d.toISOString().slice(0, 13) + ':00:00';
    const bucket = byHour.get(iso);
    values.push(bucket ? bucket[key] : (key === 'count' || key === 'failures' ? 0 : null));
  }
  return { labels, values };
}

function makeLineChart(canvasId, labels, datasets) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  activeCharts.push(new Chart(canvas, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: CHART_TEXT, font: { size: 11 }, usePointStyle: true, boxWidth: 8 } },
        tooltip: CHART_TOOLTIP,
      },
      scales: {
        x: {
          grid: { color: CHART_GRID },
          ticks: { color: '#66666b', font: { size: 10 }, maxTicksLimit: 7, maxRotation: 0 },
        },
        y: {
          beginAtZero: true,
          grid: { color: CHART_GRID },
          ticks: { color: '#66666b', font: { size: 10 } },
        },
      },
    },
  }));
}

function makeDoughnutChart(canvasId, labels, data) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const palette = [
    'rgba(255,255,255,0.9)', 'rgba(255,255,255,0.6)', 'rgba(255,255,255,0.38)',
    'rgba(255,255,255,0.22)', 'rgba(255,255,255,0.12)', CHART_DANGER,
  ];
  activeCharts.push(new Chart(canvas, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: labels.map((_, i) => palette[i % palette.length]),
        borderColor: 'rgba(10,10,11,0.9)',
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '62%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: CHART_TEXT, font: { size: 11 }, usePointStyle: true, boxWidth: 8 },
        },
        tooltip: CHART_TOOLTIP,
      },
    },
  }));
}

async function loadTelemetry() {
  telemetryStatus.textContent = 'Loading…';
  try {
    const [summaryRes, eventsRes] = await Promise.all([
      api('/telemetry/summary'),
      api('/telemetry/events?limit=100'),
    ]);
    const summary = await summaryRes.json();
    const { events } = await eventsRes.json();
    telemetryStatus.textContent = `${summary.total} event${summary.total === 1 ? '' : 's'} recorded`;
    renderTelemetry(summary, events);
  } catch (err) {
    telemetryStatus.textContent = 'Failed to load';
    telemetryBody.innerHTML = `<div class="dash-empty">Could not load analytics: ${escapeHtml(err.message)}</div>`;
  }
}

function renderTelemetry(summary, events) {
  destroyCharts();
  if (currentTelemetryEvent) { renderTelemetryDetail(currentTelemetryEvent); return; }
  if (summary.total === 0) {
    telemetryBody.innerHTML = '<div class="dash-empty">No telemetry recorded yet. Send a message or upload a document to start logging.</div>';
    return;
  }

  const card = (value, label, cls = '') =>
    `<div class="stat-card"><div class="stat-value ${cls}">${value}</div><div class="stat-label">${label}</div></div>`;
  const stats = `
    <div class="stats-grid">
      ${card(fmtNum(summary.total), 'Events')}
      ${card(summary.success_rate + '%', 'Success rate', 'ok')}
      ${card(fmtNum(summary.failures), 'Failures', summary.failures > 0 ? 'err' : '')}
      ${card(fmtMs(summary.avg_duration_ms), 'Avg response')}
    </div>`;

  const routes = summary.routes || {};
  const maxRouteCount = Math.max(1, ...Object.values(routes).map(r => r.count));
  const routeRows = Object.keys(routes).sort().map(route => {
    const r = routes[route];
    const pct = Math.round((r.count / maxRouteCount) * 100);
    return `
      <div class="route-row">
        <div class="route-name" title="${escapeHtml(route)}">${escapeHtml(route)}</div>
        <div class="route-meta">${r.count} req · ${fmtMs(r.avg_duration_ms)} avg</div>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
      </div>`;
  }).join('');

  const avgs = Object.keys(summary.metric_averages || {}).sort();
  const avgRows = avgs.map(k => `
    <div class="kv-row"><div class="k">${escapeHtml(k)}</div><div class="v">${fmtNum(summary.metric_averages[k])}</div></div>`).join('');

  const hasTimeline = (summary.timeline || []).length > 0;
  let chartsHtml = '';
  let req = null, fail = null, lat = null, routeLabels = [], routeCounts = [];
  if (hasCharts && hasTimeline) {
    req = build24hSeries(summary.timeline, 'count');
    fail = build24hSeries(summary.timeline, 'failures');
    lat = build24hSeries(summary.timeline, 'avg_duration_ms');
    routeLabels = Object.keys(routes).sort();
    routeCounts = routeLabels.map(r => routes[r].count);
    chartsHtml = `
      <div class="charts-grid">
        <div class="chart-card"><div class="chart-head">Requests · last 24h</div><div class="chart-box"><canvas id="chartRequests"></canvas></div></div>
        <div class="chart-card"><div class="chart-head">Avg response · last 24h</div><div class="chart-box"><canvas id="chartLatency"></canvas></div></div>
        <div class="chart-card"><div class="chart-head">Routes</div><div class="chart-box"><canvas id="chartRoutes"></canvas></div></div>
      </div>`;
  }
  const timelineFallback = hasTimeline
    ? `<div class="sec-title">Last 24 hours</div><div class="kv-row"><div class="k">requests</div><div class="v">${(summary.timeline || []).map(t => `${fmtClock(t.hour)}: <b>${t.count}</b>`).join(' · ')}</div></div>`
    : '';

  const table = `
    <table class="dash-table">
      <thead><tr><th>Time</th><th>Route</th><th>Conversation</th><th>Model</th><th>Status</th><th style="text-align:right">Duration</th></tr></thead>
      <tbody>
        ${events.map(e => `
          <tr data-id="${e.id}">
            <td class="tt">${fmtClock(e.started_at)}</td>
            <td class="route">${escapeHtml(e.route)}</td>
            <td>${e.conversation_id ? escapeHtml(e.conversation_id) : '—'}</td>
            <td>${e.model ? escapeHtml(e.model) : '—'}</td>
            <td><span class="dot ${e.success ? 'ok' : 'err'}"></span>${e.success ? 'ok' : (escapeHtml(e.error_type) || 'error')}</td>
            <td class="tt" style="text-align:right">${fmtMs(e.duration_ms)}</td>
          </tr>`).join('')}
      </tbody>
    </table>`;

  telemetryBody.innerHTML = `
    <div>${stats}</div>
    ${chartsHtml}
    ${timelineFallback}
    <div class="sec-title">Routes</div>
    ${routeRows}
    ${avgRows ? `<div class="sec-title">Average metrics</div>${avgRows}` : ''}
    <div class="sec-title">Recent events</div>
    ${table}`;

  if (chartsHtml) {
    makeLineChart('chartRequests', req.labels, [
      { label: 'requests', data: req.values, borderColor: CHART_LINE, backgroundColor: 'rgba(255,255,255,0.08)', fill: true, tension: 0.35, pointRadius: 2, borderWidth: 2 },
      { label: 'failed', data: fail.values, borderColor: CHART_DANGER, backgroundColor: 'rgba(229,100,95,0.08)', fill: true, tension: 0.35, pointRadius: 2, borderWidth: 2 },
    ]);
    makeLineChart('chartLatency', lat.labels, [
      { label: 'avg ms', data: lat.values, borderColor: CHART_LINE, backgroundColor: 'rgba(255,255,255,0.08)', fill: true, tension: 0.35, spanGaps: true, pointRadius: 2, borderWidth: 2 },
    ]);
    makeDoughnutChart('chartRoutes', routeLabels, routeCounts);
  }

  telemetryBody.querySelectorAll('tr[data-id]').forEach(row => {
    row.addEventListener('click', async () => {
      try {
        const res = await api(`/telemetry/events/${row.dataset.id}`);
        currentTelemetryEvent = await res.json();
        renderTelemetry(summary, events);
      } catch (err) {
        showToast(err.message || 'Could not load event');
      }
    });
  });
}

function renderTelemetryDetail(event) {
  const metricRows = Object.keys(event.metrics || {}).sort().map(k =>
    `<div class="kv-row"><div class="k">${escapeHtml(k)}</div><div class="v">${fmtNum(event.metrics[k])}</div></div>`).join('');
  const tagRows = Object.keys(event.tags || {}).sort().filter(k => k !== 'request_id').map(k =>
    `<div class="kv-row"><div class="k">${escapeHtml(k)}</div><div class="v">${escapeHtml(event.tags[k])}</div></div>`).join('');
  const spanRows = (event.spans || []).map(s =>
    `<div class="span-row"><div class="s-name">${escapeHtml(s.name)} <span style="color:var(--text-faint)">· ${escapeHtml(s.span_type)}</span></div><div class="s-ms">${fmtMs(s.duration_ms)}</div></div>`).join('');

  telemetryBody.innerHTML = `
    <button class="icon-btn back-btn" id="telemetryBack" title="Back" aria-label="Back">← Back</button>
    <div class="sec-title">Event #${event.id} · ${escapeHtml(event.route)} · ${event.success ? 'ok' : 'failed'}</div>
    <div class="kv-row"><div class="k">started</div><div class="v">${escapeHtml(event.started_at || '—')}</div></div>
    <div class="kv-row"><div class="k">duration</div><div class="v">${fmtMs(event.duration_ms)}</div></div>
    <div class="kv-row"><div class="k">conversation</div><div class="v">${escapeHtml(event.conversation_id || '—')}</div></div>
    ${event.error_type ? `<div class="kv-row"><div class="k">error</div><div class="v" style="color:var(--danger)">${escapeHtml(event.error_type)}</div></div>` : ''}
    ${spanRows ? `<div class="sec-title">Spans</div>${spanRows}` : ''}
    ${metricRows ? `<div class="sec-title">Metrics</div>${metricRows}` : ''}
    ${tagRows ? `<div class="sec-title">Tags</div>${tagRows}` : ''}`;

  telemetryBody.querySelector('#telemetryBack').addEventListener('click', () => {
    currentTelemetryEvent = null;
    loadTelemetry();
  });
}

async function clearTelemetry() {
  if (!window.confirm('Clear all recorded telemetry logs?')) return;
  try {
    const res = await api('/telemetry/events', { method: 'DELETE' });
    const data = await res.json();
    showToast(`${data.deleted} event${data.deleted === 1 ? '' : 's'} cleared`);
    currentTelemetryEvent = null;
    loadTelemetry();
  } catch (err) {
    showToast(err.message || 'Could not clear logs');
  }
}

/* ---- wiring ---- */

analyticsBtn.addEventListener('click', openTelemetryModal);
telemetryModalClose.addEventListener('click', closeTelemetryModal);
telemetryRefresh.addEventListener('click', loadTelemetry);
telemetryClear.addEventListener('click', clearTelemetry);
telemetryModal.addEventListener('click', (e) => { if (e.target === telemetryModal) closeTelemetryModal(); });