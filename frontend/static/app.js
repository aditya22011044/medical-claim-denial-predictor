/* ============================================================
   DenialIQ frontend — vanilla JS SPA
   ============================================================ */
'use strict';

const API = ''; // same origin
const state = {
  health: null,
  options: null,
  dashboard: null,
  models: null,
  payerTypes: [],
  riskBands: { low_max: 0.33, medium_max: 0.66 },
  authStatus: 'not_obtained',
};

/* ---------- tiny helpers ---------- */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const fmtUSD = (v) =>
  v == null ? '—' : '$' + Number(v).toLocaleString('en-US', { maximumFractionDigits: 0 });
const fmtUSD2 = (v) =>
  v == null ? '—' : '$' + Number(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const fmtPct = (v, d = 1) => (v == null ? '—' : (Number(v) * 100).toFixed(d) + '%');
const fmtNum = (v) => (v == null ? '—' : Number(v).toLocaleString('en-US'));
const titleCase = (s) => String(s || '').replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

async function api(path, opts) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}

let toastTimer;
function toast(msg, bad = false) {
  const t = $('#toast');
  t.textContent = msg;
  t.className = 'toast show' + (bad ? ' bad' : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.className = 'toast'), 4200);
}

function animateBars(root) {
  requestAnimationFrame(() => {
    $$('.bar-fill[data-w]', root).forEach((b) => { b.style.width = b.dataset.w + '%'; });
  });
}

/* ---------- reusable chart pieces ---------- */
function barRow(label, value, max, fmt, sub) {
  const pct = max > 0 ? Math.max(2, (value / max) * 100) : 0;
  const title = sub ? `${label} — ${sub}` : label;
  return `<div class="bar-row">
      <div class="bar-label" title="${title}">${label}</div>
      <div class="bar-track"><div class="bar-fill" data-w="${pct.toFixed(1)}" style="width:0%"></div></div>
      <div class="bar-val">${fmt(value)}</div>
    </div>`;
}

function donut(segments, size = 168) {
  const total = segments.reduce((s, x) => s + x.value, 0) || 1;
  const r = size / 2 - 16;
  const cx = size / 2, cy = size / 2;
  const circ = 2 * Math.PI * r;
  let offset = 0;
  const arcs = segments.map((seg) => {
    const len = (seg.value / total) * circ;
    const arc = `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${seg.color}"
        stroke-width="16" stroke-dasharray="${len} ${circ - len}"
        stroke-dashoffset="${-offset}" transform="rotate(-90 ${cx} ${cy})"></circle>`;
    offset += len;
    return arc;
  }).join('');
  const legend = segments.map((s) =>
    `<div class="lg"><span class="sw" style="background:${s.color}"></span>${s.label}
       <span class="val">${fmtNum(s.value)} <small style="color:var(--text-mute)">(${((s.value / total) * 100).toFixed(1)}%)</small></span>
     </div>`).join('');
  return `<div class="donut-wrap">
      <svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
        <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="rgba(255,255,255,.06)" stroke-width="16"></circle>
        ${arcs}
      </svg>
      <div class="donut-legend">${legend}</div>
    </div>`;
}

function lineChart(seriesA, seriesB, maxY) {
  const W = 640, H = 200, pad = 28;
  const n = seriesA.length;
  const x = (i) => pad + (i * (W - 2 * pad)) / (n - 1);
  const y = (v) => H - pad - (v / maxY) * (H - 2 * pad);
  const path = (arr) => arr.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const area = `${path(seriesA)} L${x(n - 1)},${H - pad} L${x(0)},${H - pad} Z`;
  const grid = [0, 0.25, 0.5, 0.75, 1].map((f) => {
    const gy = pad + f * (H - 2 * pad);
    return `<line x1="${pad}" y1="${gy}" x2="${W - pad}" y2="${gy}" stroke="rgba(255,255,255,.06)"/>`;
  }).join('');
  const xl = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    .map((m, i) => `<text x="${x(i)}" y="${H - 8}" fill="#6d7c9e" font-size="9" text-anchor="middle">${m}</text>`).join('');
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="xMidYMid meet">
      ${grid}
      <defs><linearGradient id="ag" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="rgba(99,102,241,.45)"/><stop offset="100%" stop-color="rgba(99,102,241,0)"/>
      </linearGradient></defs>
      <path d="${area}" fill="url(#ag)"/>
      <path d="${path(seriesA)}" fill="none" stroke="#818cf8" stroke-width="2.4"/>
      <path d="${path(seriesB)}" fill="none" stroke="#fb7185" stroke-width="2.4" stroke-dasharray="5 4"/>
      ${xl}
      <g font-size="10">
        <rect x="${W - 190}" y="8" width="10" height="10" rx="2" fill="#818cf8"/><text x="${W - 176}" y="17" fill="#9aa8c8">All claims</text>
        <rect x="${W - 90}" y="8" width="10" height="10" rx="2" fill="#fb7185"/><text x="${W - 76}" y="17" fill="#9aa8c8">Denials</text>
      </g>
    </svg>`;
}

/* ============================================================
   Navigation
   ============================================================ */
const PAGE_META = {
  analyze: ['Analyze Claim', 'Enter the 5 key claim attributes → get denial risk, recovery and an AI recommendation.'],
  dashboard: ['Dataset Dashboard', 'DenialIQ corpus of 120,000 historical claims at a glance.'],
  payers: ['Payer Intelligence', 'Historical denial behaviour and policy rules per payer.'],
  models: ['Models & Feature Selection', 'How the top-5 features were chosen and how each model performs.'],
};

function setView(view) {
  $$('.nav-item').forEach((b) => b.classList.toggle('active', b.dataset.view === view));
  $$('.view').forEach((v) => v.classList.toggle('active', v.id === 'view-' + view));
  const [t, s] = PAGE_META[view] || ['', ''];
  $('#page-title').textContent = t;
  $('#page-sub').textContent = s;
  if (view === 'dashboard') renderDashboard();
  if (view === 'payers') renderPayers();
  if (view === 'models') renderModels();
}

/* ============================================================
   Init
   ============================================================ */
async function init() {
  $('#nav').addEventListener('click', (e) => {
    const btn = e.target.closest('.nav-item');
    if (btn) setView(btn.dataset.view);
  });

  try {
    const [health, options, dash, models] = await Promise.all([
      api('/health'), api('/options'), api('/dashboard'), api('/models'),
    ]);
    state.health = health;
    state.options = options;
    state.dashboard = dash;
    state.models = models;
    state.payerTypes = options.payer_types || [];
    const bands = models?.metadata?.denial_classification?.risk_bands;
    if (bands) state.riskBands = bands;
    applyHealth(health);
    fillForm(options);
    buildFootChips(options.selected_features);
  } catch (err) {
    toast('Failed to load API data: ' + err.message, true);
  }

  wireForm();
}

function applyHealth(h) {
  const pm = $('#pill-models'), pl = $('#pill-llm');
  const allOk = h.models_loaded && Object.values(h.models_loaded).every(Boolean);
  pm.className = 'pill ' + (allOk ? 'ok' : 'bad');
  pm.innerHTML = `<i class="dot"></i> Models ${allOk ? 'online' : 'degraded'}`;
  pl.className = 'pill ' + (h.llm_available ? 'ok' : 'bad');
  pl.innerHTML = `<i class="dot"></i> AI ${h.llm_available ? 'online' : 'offline'}`;
}

function buildFootChips(feats) {
  const map = {
    documentation_completeness: 'Doc. completeness',
    cpt_code: 'CPT code',
    primary_icd10_dx: 'ICD-10 DX',
    claim_amount_usd: 'Billed amount',
    auth_status: 'Auth status',
  };
  $('#foot-chips').innerHTML = (feats || []).map((f) => `<span>${map[f] || f}</span>`).join('');
}

function fillForm(opts) {
  $('#cpt-list').innerHTML = (opts.cpt_codes || []).map((c) => `<option value="${c}">`).join('');
  $('#icd-list').innerHTML = (opts.icd10_codes || []).map((c) => `<option value="${c}">`).join('');
  $('#payer_type').innerHTML = '<option value="">— none —</option>' +
    (opts.payer_types || []).map((p) => `<option value="${p}">${p}</option>`).join('');
  $('#payer-select').innerHTML = (opts.payer_types || []).map((p) => `<option value="${p}">${p}</option>`).join('');
}

/* ============================================================
   Analyze form
   ============================================================ */
function wireForm() {
  const doc = $('#documentation_completeness');
  const meter = $('#doc-meter');
  const updateDoc = () => {
    const v = parseFloat(doc.value);
    $('#doc-out').textContent = v.toFixed(2);
    meter.style.width = (v * 100) + '%';
  };
  doc.addEventListener('input', updateDoc);
  updateDoc();

  $$('#auth-seg button').forEach((b) => {
    b.addEventListener('click', () => {
      $$('#auth-seg button').forEach((x) => x.classList.remove('active'));
      b.classList.add('active');
      state.authStatus = b.dataset.val;
    });
  });

  $('#example-btn').addEventListener('click', () => {
    $('#cpt_code').value = '93454';
    $('#primary_icd10_dx').value = 'I48.91';
    $('#claim_amount_usd').value = 1500;
    doc.value = 0.45;
    updateDoc();
    $$('#auth-seg button').forEach((x) => x.classList.toggle('active', x.dataset.val === 'not_obtained'));
    state.authStatus = 'not_obtained';
    if (state.payerTypes.includes('Medicare_Advantage')) $('#payer_type').value = 'Medicare_Advantage';
  });

  $('#claim-form').addEventListener('submit', onSubmit);
}

function collectClaim() {
  const payer = $('#payer_type').value;
  const dateVal = $('#claim_submission_date').value;
  const claim = {
    documentation_completeness: parseFloat($('#documentation_completeness').value),
    cpt_code: $('#cpt_code').value.trim(),
    primary_icd10_dx: $('#primary_icd10_dx').value.trim(),
    claim_amount_usd: parseFloat($('#claim_amount_usd').value),
    auth_status: state.authStatus,
  };
  if (payer) claim.payer_type = payer;
  if (dateVal) claim.claim_submission_date = dateVal;
  return claim;
}

async function onSubmit(e) {
  e.preventDefault();
  const btn = $('#analyze-btn');
  if (!btn.disabled) {
    btn._html = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Analyzing…';
  }
  $('#results').innerHTML = skeleton();
  try {
    const claim = collectClaim();
    const data = await api('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(claim),
    });
    renderResult(data);
    $('#results').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    $('#results').innerHTML = `<div class="card empty-state"><svg viewBox="0 0 24 24"><path d="M12 2 1 21h22L12 2Zm1 14h-2v-2h2v2Zm0-4h-2V8h2v4Z"/></svg>
      <h3>Analysis failed</h3><p>${err.message}</p></div>`;
    toast('Analysis failed: ' + err.message, true);
  } finally {
    btn.disabled = false;
    btn.innerHTML = btn._html || 'Analyze Claim';
  }
}

function skeleton() {
  return `<div class="stat-row">
      <div class="card"><div class="skeleton" style="height:14px;width:60%"></div><div class="skeleton" style="height:30px;width:40%;margin-top:12px"></div></div>
      <div class="card"><div class="skeleton" style="height:14px;width:60%"></div><div class="skeleton" style="height:30px;width:40%;margin-top:12px"></div></div>
      <div class="card"><div class="skeleton" style="height:14px;width:60%"></div><div class="skeleton" style="height:30px;width:40%;margin-top:12px"></div></div>
    </div>
    <div class="card"><div class="skeleton" style="height:200px;width:100%"></div></div>`;
}

/* ---------- result rendering ---------- */
function riskClass(level) {
  return level === 'Low' ? 'low' : level === 'Medium' ? 'medium' : 'high';
}
function probColor(p) {
  const b = state.riskBands;
  return p <= b.low_max ? 'var(--ok)' : p <= b.medium_max ? 'var(--warn)' : 'var(--bad)';
}

function gaugeHTML(p) {
  const r = 62, c = 2 * Math.PI * r, off = c * (1 - p);
  const color = probColor(p);
  return `<div class="gauge">
      <svg width="148" height="148" viewBox="0 0 148 148">
        <circle cx="74" cy="74" r="${r}" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="13"/>
        <circle cx="74" cy="74" r="${r}" fill="none" stroke="${color}" stroke-width="13" stroke-linecap="round"
          stroke-dasharray="${c.toFixed(1)}" stroke-dashoffset="${off.toFixed(1)}"></circle>
      </svg>
      <div class="gauge-center">
        <div class="gauge-num" style="color:${color}">${(p * 100).toFixed(1)}%</div>
        <div class="gauge-cap">Denial risk</div>
      </div>
    </div>`;
}

function renderResult(data) {
  const pr = data.prediction || {};
  const payer = data.payer_intelligence || {};
  const rec = data.recommendation || {};
  const p = pr.denial_probability ?? 0;
  const rc = riskClass(pr.risk_level);

  const reasonCard = pr.predicted_denial_reason
    ? `<div class="reason-head">
         <span class="reason-title">${titleCase(pr.predicted_denial_reason)}</span>
         ${pr.predicted_x12_code ? `<span class="code-chip">${pr.predicted_x12_code}</span>` : ''}
         <span class="muted" style="margin-left:auto">confidence ${fmtPct(pr.denial_reason_confidence)}</span>
       </div>
       <div class="bars">
         ${(pr.denial_reason_top3 || []).map((r) =>
           barRow(titleCase(r.category), r.probability, pr.denial_reason_top3[0].probability || 1, (v) => fmtPct(v))).join('')}
       </div>`
    : `<div class="muted">${(pr.notes && pr.notes.find((n) => n.includes('below the threshold')))
        ? 'Denial probability is below the threshold — no denial reason predicted.'
        : 'No denial reason was predicted for this claim.'}</div>`;

  const recovery = pr.predicted_recovery_usd != null
    ? `<div class="recovery-amount">${fmtUSD(pr.predicted_recovery_usd)}</div>
       <p class="muted" style="margin-top:8px">${pr.recovery_note || ''}</p>`
    : `<div class="recovery-amount" style="opacity:.5">—</div>
       <p class="muted" style="margin-top:8px">${pr.recovery_note || 'Not applicable.'}</p>`;

  const factors = (pr.risk_factors || []).map((f) =>
    `<div class="factor ${f.direction}">
        <span class="fx"></span>
        <div><div><span class="fname">${titleCase(f.factor)}</span><span class="fval">${escapeHtml(String(f.value))}</span></div>
        <div class="fdetail">${escapeHtml(f.detail)}</div></div>
     </div>`).join('');

  const notes = (pr.notes || []).length
    ? `<div class="notes"><ul>${pr.notes.map((n) => `<li>${escapeHtml(n)}</li>`).join('')}</ul></div>` : '';

  const payerBlurb = payer.historical_claims != null
    ? `<div class="legend-row"><span class="swatch" style="background:var(--accent)"></span>${payer.payer_type}
         <span style="color:var(--text-mute);font-size:12px;margin-left:6px">(${fmtNum(payer.historical_claims)} claims · denial ${fmtPct(payer.historical_denial_rate)})</span>
       </div><div class="muted" style="margin-top:6px">${escapeHtml(payer.denial_rate_vs_overall || '')}</div>`
    : `<div class="muted">${escapeHtml(payer.note || 'No payer selected.')}</div>`;

  $('#results').innerHTML = `
    <div class="stat-row">
      <div class="stat risk-${rc}">
        <div class="stat-label">Denial Risk</div>
        <div class="stat-value" style="color:${probColor(p)}">${fmtPct(p)}</div>
        <div class="stat-foot">Approval probability ${fmtPct(pr.approval_probability)}</div>
      </div>
      <div class="stat risk-${rc}">
        <div class="stat-label">Predicted Status</div>
        <div class="stat-value">${pr.predicted_status || '—'}</div>
        <div class="stat-foot"><span class="badge ${rc}">${pr.risk_level || '—'} risk</span>
          &nbsp;score ${pr.risk_score ?? '—'}/100</div>
      </div>
      <div class="stat">
        <div class="stat-label">Expected Recovery</div>
        <div class="stat-value" style="color:#a5f3fc">${pr.predicted_recovery_usd != null ? fmtUSD(pr.predicted_recovery_usd) : '—'}</div>
        <div class="stat-foot">if denied &amp; appealed</div>
      </div>
    </div>

    <div class="two-col">
      <div class="card">
        <div class="card-head"><h2>Denial Probability</h2><span class="tag">Model&nbsp;1</span></div>
        <div class="gauge-wrap">
          ${gaugeHTML(p)}
          <div class="gauge-legend">
            <div class="legend-row"><span class="swatch" style="background:var(--ok)"></span>Low risk
              <span class="lv">&le; ${fmtPct(state.riskBands.low_max, 0)}</span></div>
            <div class="legend-row"><span class="swatch" style="background:var(--warn)"></span>Medium risk
              <span class="lv">&le; ${fmtPct(state.riskBands.medium_max, 0)}</span></div>
            <div class="legend-row"><span class="swatch" style="background:var(--bad)"></span>High risk
              <span class="lv">&gt; ${fmtPct(state.riskBands.medium_max, 0)}</span></div>
            <div class="legend-row" style="margin-top:6px;border-top:1px dashed var(--stroke);padding-top:12px">
              ${payerBlurb}
            </div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-head"><h2>Predicted Denial Reason</h2><span class="tag">Model&nbsp;3</span></div>
        ${reasonCard}
      </div>
    </div>

    <div class="two-col">
      <div class="card">
        <div class="card-head"><h2>Risk Factors</h2><span class="tag">Explainability</span></div>
        <div class="factor-list">${factors}</div>
        ${notes}
      </div>
      <div class="card">
        <div class="card-head"><h2>Expected Recovery</h2><span class="tag">Model&nbsp;2</span></div>
        ${recovery}
      </div>
    </div>

    ${aiReport(rec)}`;

  animateBars($('#results'));
}

function aiReport(rec) {
  const head = `<div class="ai-head"><span class="ai-orb">✦</span><h2>AI Recommendation</h2>
      <span class="ai-model">${rec.model ? escapeHtml(rec.model) : ''}</span></div>`;
  if (!rec || Object.keys(rec).length === 0) return '';
  if (!rec.available || !rec.report) {
    return `<div class="card ai-card">${head}
        <div class="ai-status">${escapeHtml(rec.status || 'The AI assistant is preparing your recommendation...')}</div></div>`;
  }
  const r = rec.report || {};
  const verdict = r.final_recommendation || '';
  const vClass = verdict.startsWith('Ready') ? 'ready' : verdict.startsWith('High') ? 'priority' : 'review';
  const list = (arr) => (arr || []).map((x) => `<li>${escapeHtml(x)}</li>`).join('');
  const sec = (title, body) => (body ? `<div class="ai-section"><h4>${title}</h4>${body}</div>` : '');
  return `<div class="card ai-card">${head}
      ${rec.status ? `<div class="ai-status">${escapeHtml(rec.status)}</div>` : ''}
      ${sec('Executive Summary', r.executive_summary ? `<p>${escapeHtml(r.executive_summary)}</p>` : '')}
      ${sec('Claim Risk Assessment', r.claim_risk_assessment ? `<p>${escapeHtml(r.claim_risk_assessment)}</p>` : '')}
      ${sec('Predicted Denial Reason', r.predicted_denial_reason ? `<p>${escapeHtml(r.predicted_denial_reason)}</p>` : '')}
      ${sec('Important Risk Factors', r.important_risk_factors ? `<ul>${list(r.important_risk_factors)}</ul>` : '')}
      ${sec('Recommended Actions Before Submission', r.recommended_actions ? `<ul>${list(r.recommended_actions)}</ul>` : '')}
      ${sec('Billing Code Review', r.billing_code_review ? `<p>${escapeHtml(r.billing_code_review)}</p>` : '')}
      ${sec('Reimbursement Insights', r.reimbursement_insights ? `<p>${escapeHtml(r.reimbursement_insights)}</p>` : '')}
      <div class="ai-section"><h4>Final Recommendation</h4>
        <span class="verdict ${vClass}">${escapeHtml(verdict)}</span></div>
    </div>`;
}

/* ============================================================
   Dashboard
   ============================================================ */
const PALETTE = ['#6366f1', '#22d3ee', '#34d399', '#fbbf24', '#fb7185', '#a78bfa', '#f472b6'];

function renderDashboard() {
  const d = state.dashboard;
  if (!d) return;

  $('#dash-kpis').innerHTML = [
    ['Total Claims', fmtNum(d.total_claims), ''],
    ['Denied', fmtNum(d.denied_claims), `denial rate ${fmtPct(d.overall_denial_rate)}`],
    ['Approved', fmtNum(d.paid_claims), `approval ${fmtPct(d.overall_approval_rate)}`],
    ['Unique Payers', fmtNum(d.unique_payers), 'payer types'],
    ['Avg Claim', fmtUSD(d.avg_claim_amount), 'billed amount'],
    ['Avg Recovery', fmtUSD(d.avg_recovery_usd), 'on denied claims'],
  ].map(([l, v, f]) => `<div class="kpi"><div class="k-label">${l}</div>
      <div class="k-value">${v}</div>${f ? `<div class="stat-foot">${f}</div>` : ''}</div>`).join('');

  const oc = d.outcome_counts || {};
  const segs = [
    { label: 'Paid', value: oc.paid || 0, color: '#34d399' },
    { label: 'Denied', value: oc.denied || 0, color: '#fb7185' },
    { label: 'Partial pay', value: oc.partial_pay || 0, color: '#fbbf24' },
    { label: 'Pending', value: oc.pending || 0, color: '#818cf8' },
  ];
  $('#chart-outcomes').innerHTML = donut(segs);

  const dc = d.denial_category_counts || {};
  const dcMax = Math.max(...Object.values(dc), 1);
  $('#chart-reasons').innerHTML = Object.entries(dc).sort((a, b) => b[1] - a[1])
    .map(([k, v]) => barRow(titleCase(k), v, dcMax, fmtNum)).join('');

  const mc = d.monthly_claims || {}, md = d.monthly_denials || {};
  const claims = Array.from({ length: 12 }, (_, i) => mc[String(i + 1)] || 0);
  const denials = Array.from({ length: 12 }, (_, i) => md[String(i + 1)] || 0);
  $('#chart-trend').innerHTML = lineChart(claims, denials, Math.max(...claims, 1) * 1.1);

  const pp = d.payer_denial_rates || {};
  $('#chart-payers').innerHTML = Object.entries(pp)
    .map(([k, v]) => barRow(k, v, 0.35, (x) => fmtPct(x))).join('');

  const codes = d.denial_code_counts || {};
  const codeMax = Math.max(...Object.values(codes), 1);
  $('#chart-codes').innerHTML = Object.entries(codes).sort((a, b) => b[1] - a[1]).slice(0, 8)
    .map(([k, v]) => barRow(k, v, codeMax, fmtNum)).join('');

  animateBars($('#view-dashboard'));
}

/* ============================================================
   Payers
   ============================================================ */
function renderPayers() {
  const sel = $('#payer-select');
  if (!sel.options.length) return;
  if (!sel.dataset.wired) {
    sel.dataset.wired = '1';
    sel.addEventListener('change', () => loadPayer(sel.value));
  }
  loadPayer(sel.value);
}

async function loadPayer(payer) {
  const body = $('#payer-body');
  if (!payer) { body.innerHTML = ''; return; }
  body.innerHTML = '<div class="card"><div class="skeleton" style="height:120px"></div></div>';
  try {
    const [intel, rules] = await Promise.all([
      api('/payer-intelligence/' + encodeURIComponent(payer)),
      api('/payer-rules/' + encodeURIComponent(payer)),
    ]);
    const rows = (rules.rules || []).map((r) => `<tr>
        <td class="mono">${r.cpt_code}</td>
        <td>${r.requires_prior_auth ? '<span class="badge high">Required</span>' : '<span class="badge low">No</span>'}</td>
        <td>${r.auth_lead_time_days ?? '—'} d</td>
        <td>${fmtPct(r.historical_denial_rate)}</td>
        <td>${r.avg_payment_turnaround_days ?? '—'} d</td>
        <td>${r.timely_filing_limit_days ?? '—'} d</td>
      </tr>`).join('');
    const pgrid = `<div class="pgrid">
        <div class="kpi"><div class="k-label">Historical Claims</div><div class="k-value">${fmtNum(intel.historical_claims)}</div></div>
        <div class="kpi"><div class="k-label">Denial Rate</div><div class="k-value">${fmtPct(intel.historical_denial_rate)}</div>
          <div class="stat-foot">${escapeHtml(intel.denial_rate_vs_overall || '')}</div></div>
        <div class="kpi"><div class="k-label">Top Denial Reason</div><div class="k-value" style="font-size:17px">${titleCase(intel.top_denial_reason || '—')}</div>
          <div class="stat-foot">${intel.top_denial_code || ''}</div></div>
        <div class="kpi"><div class="k-label">Avg Claim / Recovery</div><div class="k-value" style="font-size:19px">${fmtUSD(intel.avg_claim_amount)}</div>
          <div class="stat-foot">recovery ${fmtUSD(intel.avg_recovery_usd)}</div></div>
      </div>`;
    body.innerHTML = pgrid + `<div class="card">
        <div class="card-head"><h2>Payer × CPT policy rules (sample)</h2>
          <span class="tag">rule-table denial ${fmtPct(intel.rule_table_denial_rate)}</span></div>
        <table><thead><tr><th>CPT</th><th>Prior auth</th><th>Auth lead</th><th>Hist. denial</th><th>Turnaround</th><th>Filing limit</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="6">No rule rows.</td></tr>'}</tbody></table></div>`;
  } catch (err) {
    body.innerHTML = `<div class="card empty-state"><h3>No data</h3><p>${escapeHtml(err.message)}</p></div>`;
  }
}

/* ============================================================
   Models & feature selection
   ============================================================ */
function renderModels() {
  const m = state.models;
  if (!m) return;
  const fs = m.feature_selection || {};
  const ranking = fs.ranking || {};
  const selected = new Set(m.selected_features || []);
  $('#fs-method').textContent = fs.method || 'XGBoost gain importance';
  $('#fs-note').textContent =
    `${fs.n_features_full ?? '—'} candidate attributes were ranked; the top ${fs.n_features_selected ?? selected.size} ` +
    `enterable features (highlighted) are the only model inputs.` +
    (selected.size ? ' Selected: ' + [...selected].map(titleCase).join(', ') + '.' : '');

  const max = Math.max(...Object.values(ranking), 1);
  const sorted = Object.entries(ranking).sort((a, b) => b[1] - a[1]);
  $('#feature-ranking').innerHTML = sorted.map(([k, v]) => {
    const isSel = selected.has(k);
    const pct = max > 0 ? Math.max(2, (v / max) * 100) : 0;
    return `<div class="feat-row ${isSel ? 'sel' : ''}">
        <div class="feat-label">${isSel ? '<span class="check">✓</span>' : ''}${titleCase(k)}</div>
        <div class="bar-track"><div class="bar-fill" data-w="${pct.toFixed(1)}" style="width:0%"></div></div>
        <div class="bar-val">${v.toFixed(1)}%</div>
      </div>`;
  }).join('');

  const md = m.metadata || {};
  $('#model-grid').innerHTML = [modelCard1(md.denial_classification), modelCard2(md.recovery_regression), modelCard3(md.denial_reason)]
    .filter(Boolean).join('');

  animateBars($('#view-models'));
}

function metricHTML(label, value) {
  return `<div class="metric"><div class="m-label">${label}</div><div class="m-value">${value}</div></div>`;
}
function modelCard(title, algo, metrics, features) {
  return `<div class="card">
      <div class="card-head"><h2>${title}</h2><span class="model-badge">${algo}</span></div>
      <div class="metric-grid">${metrics}</div>
      <p class="muted" style="margin-top:14px;font-size:11.5px">Features: ${features}</p>
    </div>`;
}
function modelCard1(d) {
  if (!d) return '';
  const t = d.metrics || {};
  return modelCard('Denial Classifier', 'XGBoost binary', [
    metricHTML('ROC-AUC', (t.roc_auc ?? 0).toFixed(4)),
    metricHTML('PR-AUC', (t.pr_auc ?? 0).toFixed(4)),
    metricHTML('Accuracy', fmtPct(t.accuracy)),
    metricHTML('Recall', fmtPct(t.recall)),
    metricHTML('Precision', fmtPct(t.precision)),
    metricHTML('F1 Score', (t.f1 ?? 0).toFixed(4)),
  ].join(''), (d.features_used || []).length + ' selected');
}
function modelCard2(d) {
  if (!d) return '';
  const t = d.metrics || {};
  return modelCard('Recovery Regressor', 'XGBoost reg', [
    metricHTML('R²', (t.r2 ?? 0).toFixed(4)),
    metricHTML('MAE', fmtUSD(t.mae)),
    metricHTML('RMSE', fmtUSD(t.rmse)),
    metricHTML('Target mean', fmtUSD(t.target_mean)),
  ].join(''), (d.features_used || []).length + ' selected');
}
function modelCard3(d) {
  if (!d) return '';
  const t = d.metrics || {};
  return modelCard('Denial Reason', 'XGBoost multi-class', [
    metricHTML('Accuracy', fmtPct(t.accuracy)),
    metricHTML('Macro F1', (t.macro_f1 ?? 0).toFixed(4)),
    metricHTML('Weighted F1', (t.weighted_f1 ?? 0).toFixed(4)),
    metricHTML('Macro precision', (t.precision_macro ?? 0).toFixed(4)),
  ].join(''), (d.features_used || []).length + ' selected');
}

/* ---------- escape ---------- */
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

document.addEventListener('DOMContentLoaded', init);
