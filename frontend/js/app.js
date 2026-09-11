/* BioSentinel 2.0 — frontend application logic
   Chart data flow: SSE → extendCharts → safeExtend → Plotly.extendTraces
*/
"use strict";

// ── State ─────────────────────────────────────────────────────────────────────
const S = {
  regime:       "Healthy",
  streaming:    false,
  paused:       false,
  finished:     false,
  source:       null,      // EventSource
  timer:        null,      // Client simulation clock
  cachedPoints: {},        // regime -> points array
  lastData:     null,      // last point
  chartsReady:  false,
  currentIdx:   0,
};

async function apiFetch(path, options = {}) {
  const clean = path.startsWith("/") ? path : `/${path}`;
  try {
    const res = await fetch(`/api${clean}`, options);
    if (res.ok) return res;
  } catch (e) {}
  return fetch(clean, options);
}

// Plotly dark template shared across all charts
const PLOTLY_TMPL = {
  layout: {
    paper_bgcolor: "#1a2540", plot_bgcolor: "#131e30",
    font: { family: "Inter, sans-serif", color: "#e8edf5", size: 11 },
    xaxis: {
      gridcolor: "#243352", zeroline: false,
      tickfont: { size: 10, color: "#94a3b8" },
      title: { standoff: 8, font: { size: 11, color: "#cbd5e1" } }
    },
    yaxis: {
      gridcolor: "#243352", zeroline: false,
      tickfont: { size: 10, color: "#94a3b8" },
      title: { standoff: 10, font: { size: 11 } }
    },
    margin: { l: 54, r: 64, t: 26, b: 42 },
    legend: { bgcolor: "rgba(0,0,0,0)", orientation: "h", y: 1.15, x: 0, font: { size: 10.5 } },
    hovermode: "x unified",
  }
};


const REGIME_META = {
  Healthy:               { label: "Healthy",              icon: "✅", cls: "healthy",  acls: "healthy-badge" },
  kLa_Limitation:        { label: "kLa Limitation",       icon: "⚠️",  cls: "kla",     acls: "kla-badge" },
  Substrate_Overfeeding: { label: "Substrate Overfeeding", icon: "🔥", cls: "overfeed", acls: "overfeed-badge" },
  Contamination:         { label: "Contamination",         icon: "☣️", cls: "contam",  acls: "contam-badge" },
};
const REGIME_COLORS = {
  Healthy: "#22c55e", kLa_Limitation: "#f59e0b",
  Substrate_Overfeeding: "#f97316", Contamination: "#ef4444",
};
const ACTUATOR_LABELS = {
  Healthy: "Nominal", kLa_Limitation: "O₂ Transfer Limited",
  Substrate_Overfeeding: "Feed Trim Recommended", Contamination: "TRIGGER CIP / HARVEST ABORT",
};
const PROB_ORDER = ["Healthy","kLa_Limitation","Substrate_Overfeeding","Contamination"];


// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll(".tab-section").forEach(s => s.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.getElementById("tab-" + name).classList.add("active");
  document.querySelector(`[data-tab="${name}"]`).classList.add("active");

  if (name === "reference") loadMetrics();
  if (name === "insights" && !S.chartsReady) initCharts();
}

// ── Scenario / stream controls ────────────────────────────────────────────────
function selectRegime(btn, regime) {
  document.querySelectorAll(".scenario-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  S.regime = regime;
  resetStream();
}

function togglePlay() {
  const btn = document.getElementById("btn-play");
  if (S.finished) {
    resetStream();
    return;
  }
  if (!S.streaming && !S.paused) {
    startStream(0);
    return;
  }
  if (S.paused) {
    S.paused = false;
    S.streaming = true;
    if (btn) btn.textContent = "⏸ Pause";
    startStream(S.currentIdx + 1);
  } else {
    S.paused = true;
    S.streaming = false;
    if (btn) btn.textContent = "▶ Resume";
    if (S.source) { S.source.close(); S.source = null; }
    if (S.timer) { clearInterval(S.timer); S.timer = null; }
  }
}

// ── Low-pass filter for rock-solid, jitter-free KPI displays ─────────────────
const EMA = {
  do: null, rq: null, our: null, cer: null, r: null, x: null, s: null, p: null, rpm: null
};
const EMA_ALPHA = 0.20;

function ema(k, val) {
  if (val === undefined || val === null || isNaN(val)) return 0;
  if (EMA[k] === null || isNaN(EMA[k])) {
    EMA[k] = val;
  } else {
    EMA[k] = EMA_ALPHA * val + (1 - EMA_ALPHA) * EMA[k];
  }
  return EMA[k];
}

function resetEMA() {
  for (const k in EMA) EMA[k] = null;
}

function resetStream() {
  stopStream(true);
  clearCharts();
  resetEMA();
  S.paused = false;
  S.finished = false;
  S.currentIdx = 0;
  const btn = document.getElementById("btn-play");
  if (btn) btn.textContent = "⏸ Pause";
  updateBanner(S.regime, null);
  document.getElementById("progress-bar").style.width = "0%";
  ["do","rq","our","r","diw","x"].forEach(k => { const el=document.getElementById("kpi-"+k); if(el)el.textContent="—"; });
  setTimeout(() => startStream(0), 200);
}

function stopStream(full=true) {
  if (S.source) { S.source.close(); S.source = null; }
  if (S.timer) { clearInterval(S.timer); S.timer = null; }
  S.streaming = false;
  if (full) {
    S.lastData = null;
    S.currentIdx = 0;
    S.finished = false;
    clearCounterfactual();
  }
}

function applyPoint(d) {
  S.lastData = d;
  S.currentIdx = d.idx;

  updateKPIs(d);
  updateBanner(S.regime, d);
  extendCharts(d);
  if (d.probs) updateProbs(d.probs, d.pred);
  updateSensorDetail(d);
  updateActuator(d.pred || S.regime);

  // Load counterfactual once
  if (d.idx === 60) loadCounterfactual();

  // Progress
  const pct = Math.round((d.idx / (d.total - 1)) * 100);
  const pbar = document.getElementById("progress-bar");
  if (pbar) pbar.style.width = Math.min(pct, 100) + "%";

  // Stop at 24 hrs: DO NOT restart timer, stop it there!
  if (d.idx >= d.total - 1 || d.t >= 24.0) {
    if (S.source) { S.source.close(); S.source = null; }
    if (S.timer) { clearInterval(S.timer); S.timer = null; }
    S.streaming = false;
    S.paused = false;
    S.finished = true;
    const timeEl = document.getElementById("banner-time");
    if (timeEl) timeEl.textContent = "t = 24.00 h";
    if (pbar) pbar.style.width = "100%";
    const btn = document.getElementById("btn-play");
    if (btn) btn.textContent = "↩ Replay";
  }
}

function playFromPoints(points, startIdx) {
  if (S.timer) { clearInterval(S.timer); S.timer = null; }
  let curr = Math.max(0, Math.min(startIdx, points.length - 1));

  S.timer = setInterval(() => {
    if (S.paused || !S.streaming) {
      clearInterval(S.timer);
      S.timer = null;
      return;
    }
    if (curr >= points.length) {
      clearInterval(S.timer);
      S.timer = null;
      return;
    }
    applyPoint(points[curr]);
    curr++;
  }, 48); // ~21 pts/sec -> smooth 24h simulation batch
}

function fallbackToSSE(startIdx) {
  const url = `/api/stream/${S.regime}?start_idx=${startIdx}`;
  const es = new EventSource(url);
  S.source = es;

  es.onmessage = (e) => {
    try {
      const d = JSON.parse(e.data);
      applyPoint(d);
    } catch(err) {
      console.error(err);
    }
  };

  es.onerror = () => {
    if (S.finished || (S.lastData && S.lastData.idx >= S.lastData.total - 1)) {
      if (S.source) { S.source.close(); S.source = null; }
      S.streaming = false;
      return;
    }
    if (S.paused) {
      if (S.source) { S.source.close(); S.source = null; }
      return;
    }
    S.streaming = false;
  };
}

function startStream(startIdx = 0) {
  if (S.source) { S.source.close(); S.source = null; }
  if (S.timer) { clearInterval(S.timer); S.timer = null; }
  if (!S.chartsReady) initCharts();

  S.finished = false;
  S.paused = false;
  S.streaming = true;
  const btn = document.getElementById("btn-play");
  if (btn) btn.textContent = "⏸ Pause";

  // 1. If points already cached, play immediately (0ms latency!)
  if (S.cachedPoints[S.regime] && S.cachedPoints[S.regime].length > 0) {
    playFromPoints(S.cachedPoints[S.regime], startIdx);
    return;
  }

  // 2. Fetch full simulation points via fast batch JSON endpoint
  apiFetch(`/data/${S.regime}`)
    .then(r => r.json())
    .then(data => {
      if (data && data.points && data.points.length > 0) {
        S.cachedPoints[S.regime] = data.points;
        if (S.streaming && !S.paused) {
          playFromPoints(data.points, startIdx);
        }
      } else {
        fallbackToSSE(startIdx);
      }
    })
    .catch(() => {
      fallbackToSSE(startIdx);
    });
}


// ── Banner ────────────────────────────────────────────────────────────────────
function updateBanner(regime, d) {
  const m = REGIME_META[regime] || REGIME_META.Healthy;
  const banner = document.getElementById("banner");
  banner.className = "status-banner " + m.cls;
  document.getElementById("banner-icon").textContent   = m.icon;
  document.getElementById("banner-regime").textContent = m.label;

  // Dynamic accent color sync:
  // Active fault state: kLa Limitation (amber #f59e0b), Substrate Overfeed (orange/amber #f97316), Contamination (crimson #ef4444).
  // Neon green (#22c55e) is reserved purely for Healthy.
  const isFault = regime !== "Healthy";
  const dot  = document.getElementById("live-dot");
  const pbar = document.getElementById("progress-bar");
  const timeEl = document.getElementById("banner-time");

  if (dot) {
    dot.className = "live-dot " + (isFault ? m.cls : "healthy");
  }
  if (pbar) {
    pbar.className = "progress-bar " + (isFault ? m.cls : "healthy");
  }
  if (timeEl) {
    if (regime === "Contamination") {
      timeEl.style.color = "#ef4444";
    } else if (regime === "kLa_Limitation") {
      timeEl.style.color = "#f59e0b";
    } else if (regime === "Substrate_Overfeeding") {
      timeEl.style.color = "#f97316";
    } else {
      timeEl.style.color = "var(--accent)";
    }
    timeEl.textContent = d ? `t = ${d.t.toFixed(2)} h` : "t = 0.0 h";
  }

  // Contamination biological logic: microbial contamination is irreversible
  if (regime === "Contamination" && (!d || d.t >= 5.0 || d.idx >= 100)) {
    document.getElementById("banner-sub").textContent =
      "ML: Contamination · PNR: IRREVERSIBLE · DIW: 0 min (BREACH DETECTED)";
  } else {
    const ml    = d && d.pred   ? (REGIME_META[d.pred]?.label || d.pred) : m.label;
    const pnr   = d && d.pnr_t ? `PNR: ${d.pnr_t.toFixed(1)} h` : "PNR: Not reached";
    const diw   = d && d.diw   ? `DIW: ${d.diw < 900 ? d.diw+" min" : "Safe"}` : "DIW: —";
    document.getElementById("banner-sub").textContent  = `ML: ${ml} · ${pnr} · ${diw}`;
  }
}

// ── KPI cards (Smooth & Constant displays) ────────────────────────────────────
function updateKPIs(d) {
  const set = (id, val) => { const el = document.getElementById("kpi-"+id); if(el) el.textContent = val; };
  const smoothDO  = ema("do", d.DO);
  const smoothRQ  = ema("rq", d.RQ);
  const smoothOUR = ema("our", d.OUR);
  const smoothR   = ema("r", d.R);
  const smoothX   = ema("x", d.X);

  set("do",  smoothDO.toFixed(2));
  set("rq",  smoothRQ.toFixed(2));
  set("our", smoothOUR.toFixed(3));
  set("x",   smoothX.toFixed(2));

  const rUnit   = document.getElementById("kpi-unit-r");
  const diwUnit = document.getElementById("kpi-unit-diw");

  // Biological Logic Trap Fix:
  // Microbial contamination cannot be cured by physical actuators (agitator/sparger) without killing culture.
  // When Scenario == "Contamination" and active:
  // Recovery R(t) = 0.0% (IRREVERSIBLE), DIW = 0 min (BREACH DETECTED).
  if (S.regime === "Contamination" && (d.t >= 5.0 || d.idx >= 100)) {
    set("r",   "0.0%");
    set("diw", "0 min");
    if (rUnit)   rUnit.textContent   = "IRREVERSIBLE";
    if (diwUnit) diwUnit.textContent = "BREACH DETECTED";
  } else {
    set("r",   (smoothR > 99.8 ? 100.0 : smoothR).toFixed(1) + "%");
    set("diw", d.diw < 900 ? d.diw + " min" : "Safe");
    if (rUnit)   rUnit.textContent   = "%";
    if (diwUnit) diwUnit.textContent = "min to PNR";
  }
}


// ── Sensor detail ─────────────────────────────────────────────────────────────
function updateSensorDetail(d) {
  const set = (id, val) => { const el = document.getElementById("sd-"+id); if(el) el.textContent = val; };
  const smoothP   = ema("p", d.P);
  const smoothOUR = ema("our_sd", d.OUR);
  const smoothCER = ema("cer_sd", d.CER);
  const smoothRPM = ema("rpm", d.RPM);
  const smoothS   = ema("s", d.S);
  const smoothX   = ema("x_sd", d.X);

  set("p",   smoothP.toFixed(2) + " atm");
  set("our", smoothOUR.toFixed(3) + " g/L/h");
  set("cer", smoothCER.toFixed(3) + " g/L/h");
  set("rpm", Math.round(smoothRPM));
  set("s",   smoothS.toFixed(2) + " g/L");
  set("x",   smoothX.toFixed(2) + " g/L");
}

// ── Actuator status ───────────────────────────────────────────────────────────
function updateActuator(regime) {
  const badge = document.getElementById("actuator-badge");
  if (!badge) return;
  const cls  = (REGIME_META[regime] || REGIME_META.Healthy).acls;
  badge.className = "actuator-badge " + cls;
  badge.textContent = ACTUATOR_LABELS[regime] || "Nominal";
}

// ── Probability bars ──────────────────────────────────────────────────────────
function updateProbs(probs, pred) {
  const container = document.getElementById("prob-bars");
  if (!container) return;
  if (!container.children.length) {
    container.innerHTML = PROB_ORDER.map(k => `
      <div class="prob-item" id="prob-${k.replace(/_/g,'-')}">
        <div class="prob-header">
          <span>${(REGIME_META[k]||{}).label||k}</span>
          <span id="pval-${k.replace(/_/g,'-')}">0.0%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill" id="pbar-${k.replace(/_/g,'-')}"
               style="width:0%;background:${REGIME_COLORS[k]};"></div>
        </div>
      </div>`).join("");
  }
  PROB_ORDER.forEach(k => {
    const id  = k.replace(/_/g,"-");
    const pv  = (probs[k] || 0) * 100;
    const bar = document.getElementById("pbar-"+id);
    const val = document.getElementById("pval-"+id);
    const item= document.getElementById("prob-"+id);
    if (bar) bar.style.width = pv.toFixed(1) + "%";
    if (val) val.textContent = pv.toFixed(1) + "%";
    if (item) item.style.opacity = k === pred ? "1" : "0.7";
  });
}

// ── Plotly charts ─────────────────────────────────────────────────────────────
const CHART_DEFS = {
  "chart-do-rq": {
    traces: [
      {x:[],y:[],mode:"lines",name:"DO (mg/L)",line:{color:"#38bdf8",width:2.5},fill:"tozeroy",fillcolor:"rgba(56,189,248,0.07)"},
      {x:[],y:[],mode:"lines",name:"RQ",line:{color:"#fb923c",width:2},yaxis:"y2"},
    ],
    layout: {
      yaxis:  { title: { text: "DO (mg/L)", standoff: 10, font: { color: "#38bdf8", size: 11 } } },
      yaxis2: { title: { text: "RQ", standoff: 14, font: { color: "#fb923c", size: 11 } }, overlaying: "y", side: "right", tickfont: { size: 10, color: "#fb923c" }, showgrid: false, zeroline: false },
      xaxis:  { title: { text: "Batch time (h)", standoff: 8 } }
    }
  },
  "chart-gas": {
    traces: [
      {x:[],y:[],mode:"lines",name:"OUR",line:{color:"#a78bfa",width:2.5},fill:"tozeroy",fillcolor:"rgba(167,139,250,0.07)"},
      {x:[],y:[],mode:"lines",name:"CER",line:{color:"#f472b6",width:2,dash:"dash"}},
    ],
    layout: {
      yaxis: { title: { text: "g/(L·h)", standoff: 10, font: { color: "#cbd5e1", size: 11 } } },
      xaxis: { title: { text: "Batch time (h)", standoff: 8 } }
    }
  },
  "chart-bio": {
    traces: [
      {x:[],y:[],mode:"lines",name:"Biomass X (g/L)",line:{color:"#34d399",width:2.5}},
      {x:[],y:[],mode:"lines",name:"Substrate S (g/L)",line:{color:"#fbbf24",width:2,dash:"dash"},yaxis:"y2"},
    ],
    layout: {
      yaxis:  { title: { text: "X (g/L)", standoff: 10, font: { color: "#34d399", size: 11 } } },
      yaxis2: { title: { text: "S (g/L)", standoff: 14, font: { color: "#fbbf24", size: 11 } }, overlaying: "y", side: "right", tickfont: { size: 10, color: "#fbbf24" }, showgrid: false, zeroline: false },
      xaxis:  { title: { text: "Batch time (h)", standoff: 8 } }
    }
  },
  "chart-recovery": {
    traces: [
      {x:[],y:[],mode:"lines",name:"R(t) %",line:{color:"#22c55e",width:2.5}},
      {x:[],y:[],mode:"lines",name:"λ(t)×100",line:{color:"#ef4444",width:2,dash:"dot"},yaxis:"y2"},
    ],
    layout: {
      yaxis:  { title: { text: "Recoverability (%)", standoff: 10, font: { color: "#22c55e", size: 11 } } },
      yaxis2: { title: { text: "Stress λ(t)×100", standoff: 14, font: { color: "#ef4444", size: 11 } }, overlaying: "y", side: "right", tickfont: { size: 10, color: "#ef4444" }, showgrid: false, zeroline: false },
      xaxis:  { title: { text: "Batch time (h)", standoff: 8 } }
    }
  },
};


function initCharts() {
  if (typeof Plotly === "undefined") {
    console.warn("Plotly is still loading from CDN, will retry in 100ms...");
    setTimeout(initCharts, 100);
    return;
  }
  Object.entries(CHART_DEFS).forEach(([id, def]) => {
    const el = document.getElementById(id);
    if (!el) return;
    const layout = JSON.parse(JSON.stringify(PLOTLY_TMPL.layout));
    Object.assign(layout, def.layout);
    layout.yaxis = Object.assign({}, PLOTLY_TMPL.layout.yaxis, (def.layout && def.layout.yaxis) || {});
    layout.xaxis = Object.assign({}, PLOTLY_TMPL.layout.xaxis, (def.layout && def.layout.xaxis) || {});
    if (def.layout && def.layout.yaxis2) {
      layout.yaxis2 = Object.assign({ gridcolor: "#243352" }, def.layout.yaxis2);
    }
    const freshTraces = def.traces.map(t => ({
      ...t,
      x: [],
      y: []
    }));
    Plotly.newPlot(id, freshTraces, layout, { responsive: true, displayModeBar: false });
  });
  S.chartsReady = true;
}

function clearCharts() {
  _onsetAdded = {};
  Object.keys(CHART_DEFS).forEach(id => {
    const el = document.getElementById(id);
    if (el && el.data && typeof Plotly !== "undefined") {
      const freshTraces = CHART_DEFS[id].traces.map(t => ({
        ...t,
        x: [],
        y: []
      }));
      const layout = JSON.parse(JSON.stringify(el.layout || {}));
      layout.shapes = [];
      layout.annotations = [];
      Plotly.react(id, freshTraces, layout);
    }
  });
}

function safeExtend(chartId, xArrays, yArrays, indices = [0, 1]) {
  const el = document.getElementById(chartId);
  if (!el || !el.data || el.data.length < indices.length) return;

  // Plotly Glitch Fix: Eliminate diagonal loopback lines
  // Prevent any trace from receiving out-of-order or duplicate points (new t <= last t)
  for (let i = 0; i < indices.length; i++) {
    const idx = indices[i];
    const trace = el.data[idx];
    if (trace && trace.x && trace.x.length > 0) {
      const lastX = trace.x[trace.x.length - 1];
      const newX = xArrays[i][0];
      if (newX <= lastX) {
        return; // guard against unsorted / duplicate / loopback tail concatenation
      }
    }
  }

  try {
    Plotly.extendTraces(chartId, { x: xArrays, y: yArrays }, indices, 720);
  } catch (err) {
    console.warn("Plotly.extendTraces error on", chartId, err);
  }
}

function extendCharts(d) {
  if (!S.chartsReady || typeof Plotly === "undefined") return;
  const t = d.t;
  // chart-do-rq (Trace 0: DO, Trace 1: RQ)
  safeExtend("chart-do-rq", [[t], [t]], [[d.DO], [d.RQ]], [0, 1]);
  // chart-gas (Trace 0: OUR, Trace 1: CER)
  safeExtend("chart-gas",   [[t], [t]], [[d.OUR], [d.CER]], [0, 1]);
  // chart-bio (Trace 0: Biomass X, Trace 1: Substrate S)
  safeExtend("chart-bio",   [[t], [t]], [[d.X], [d.S]], [0, 1]);
  // chart-recovery (Trace 0: Recoverability R, Trace 1: Kinetic Stress lam*100)
  // Contamination biological fix: microbial contamination cannot be cured by physical actuators
  const rVal = (S.regime === "Contamination" && t >= 5.0) ? 0.0 : d.R;
  safeExtend("chart-recovery", [[t], [t]], [[rVal], [d.lam * 100]], [0, 1]);
  // fault onset shading
  addOnsetIfNeeded(d);
}

let _onsetAdded = {};
function addOnsetIfNeeded(d) {
  if (_onsetAdded[S.regime]) return;
  const onsets = { kLa_Limitation: 8.0, Substrate_Overfeeding: 6.0, Contamination: 5.0 };
  const t_on   = onsets[S.regime];
  if (!t_on || d.t < t_on) return;
  _onsetAdded[S.regime] = true;
  const clr = REGIME_COLORS[S.regime] || "#f59e0b";
  ["chart-do-rq", "chart-gas", "chart-bio", "chart-recovery"].forEach(id => {
    const el = document.getElementById(id);
    if (!el || !el.layout || typeof Plotly === "undefined") return;
    const shapes = (el.layout.shapes || []).concat([{
      type: "line", x0: t_on, x1: t_on, y0: 0, y1: 1, yref: "paper",
      line: { color: clr, width: 2, dash: "dash" },
    }]);
    const annotations = (el.layout.annotations || []).concat([{
      x: t_on, y: 1, yref: "paper", text: `Fault Onset (t=${t_on}h)`, showarrow: false,
      font: { size: 10, color: clr, family: "Inter, sans-serif" }, xanchor: "left",
    }]);
    Plotly.relayout(id, { shapes, annotations });
  });
}

// ── Counterfactual ────────────────────────────────────────────────────────────
async function loadCounterfactual() {
  const tbody = document.getElementById("cf-tbody");
  const table = document.getElementById("cf-table");
  const best  = document.getElementById("cf-best");
  const loading = document.getElementById("cf-loading");
  if (!tbody) return;

  try {
    const res  = await apiFetch(`/counterfactual/${S.regime}`);
    const data = await res.json();
    loading.classList.add("hidden");
    table.classList.remove("hidden");

    tbody.innerHTML = data.map(r => `
      <tr>
        <td><strong>#${r.rank}</strong></td>
        <td>${r.label}</td>
        <td><strong>${r.ending_R}%</strong></td>
        <td style="color:${r.gain>0?'#22c55e':(r.gain<0?'#ef4444':'#94a3b8')}">${r.gain>0?"+":""}${r.gain}%</td>
        <td style="color:${r.pnr?'#ef4444':'#22c55e'}">${r.pnr?"Yes (Breach)":"No"}</td>
      </tr>`).join("");

    if (S.regime === "Contamination") {
      best.classList.remove("hidden");
      best.innerHTML = `<strong>CRITICAL ACTION:</strong> TRIGGER CIP / HARVEST ABORT
        &nbsp;|&nbsp; Irreversible Microbial Contamination
        &nbsp;|&nbsp; Physical Actuation Ineffective`;
    } else if (data.length > 0) {
      best.classList.remove("hidden");
      const top = data[0];
      best.innerHTML = `<strong>TOP RECOMMENDATION:</strong> ${top.label}
        &nbsp;|&nbsp; Ending R = ${top.ending_R}%
        &nbsp;|&nbsp; Gain vs no-action: ${top.gain > 0 ? "+" : ""}${top.gain}%`;
    }
  } catch(e) {
    loading.textContent = "Failed to load counterfactuals.";
  }
}

function clearCounterfactual() {

  const loading = document.getElementById("cf-loading");
  const table   = document.getElementById("cf-table");
  const best    = document.getElementById("cf-best");
  if (loading) { loading.textContent = "Loading..."; loading.classList.remove("hidden"); }
  if (table) table.classList.add("hidden");
  if (best) best.classList.add("hidden");
  _onsetAdded = {};
}

// ── Helper HTML escape ────────────────────────────────────────────────────────
function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatExplanation(rawText, model, source) {
  if (!rawText) return '<div class="exp-card exp-bio"><div class="exp-card-body">No explanation received.</div></div>';

  // Strip all dollar signs, LaTeX \text{}, math wrappers, and backslashes
  let clean = rawText
    .replace(/\\text\{([^}]*)\}/g, "$1")
    .replace(/\\math\w+\{([^}]*)\}/g, "$1")
    .replace(/\\mu/g, "μ")
    .replace(/\$([^$]+)\$/g, "$1")
    .replace(/\$/g, "")
    .replace(/\\/g, "")
    .replace(/[*#]/g, "");

  const sections = [
    { key: "BIOLOGICAL STATE", icon: "🔬", cls: "exp-bio", title: "Biological State" },
    { key: "ROOT CAUSE", icon: "🔍", cls: "exp-cause", title: "Root Cause" },
    { key: "CONSEQUENCE", icon: "⚠️", cls: "exp-cons", title: "Consequence (2–3h Horizon)" },
    { key: "OPERATOR ACTION", icon: "🛠️", cls: "exp-action", title: "Recommended Operator Action" }
  ];

  let html = `
    <div class="exp-badge-row">
      <span class="exp-model-tag">✨ ${escapeHtml(model || "Gemini AI")}</span>
      <span class="exp-source-tag">${escapeHtml(source || "Physics-Grounded Telemetry")}</span>
    </div>
    <div class="exp-cards-container">
  `;

  let foundCount = 0;
  sections.forEach((sec, idx) => {
    const remaining = sections.slice(idx + 1).map(s => s.key).join("|");
    const lookahead = remaining ? `(?=(?:•|-)?\\s*(?:${remaining}):|$)` : "$";
    const regex = new RegExp(`(?:•|-)?\\s*${sec.key}:?\\s*([\\s\\S]*?)${lookahead}`, "i");
    const match = clean.match(regex);

    if (match && match[1].trim()) {
      foundCount++;
      const bodyText = escapeHtml(match[1].trim()).replace(/\n+/g, "<br/>");
      html += `
        <div class="exp-card ${sec.cls}">
          <div class="exp-card-header">
            <span class="exp-card-icon">${sec.icon}</span>
            <span class="exp-card-title">${sec.title}</span>
          </div>
          <div class="exp-card-body">${bodyText}</div>
        </div>
      `;
    }
  });

  if (foundCount === 0) {
    const cleanBody = escapeHtml(clean).replace(/\n+/g, "<br/><br/>");
    html += `
      <div class="exp-card exp-bio">
        <div class="exp-card-header">
          <span class="exp-card-icon">📋</span>
          <span class="exp-card-title">Expert Bioprocess Assessment</span>
        </div>
        <div class="exp-card-body">${cleanBody}</div>
      </div>
    `;
  }

  html += `</div>`;
  return html;
}

// ── Gemini explanation ────────────────────────────────────────────────────────
async function getExplanation() {
  const d   = S.lastData;
  const box = document.getElementById("explain-box");
  const txt = document.getElementById("explain-text");
  const btn = document.getElementById("btn-explain");
  if (!box || !txt) return;

  box.classList.remove("hidden");
  txt.innerHTML = '<div style="color:var(--muted);font-size:0.85rem;padding:8px 0;">⚡ Consulting Gemini AI for real-time bioprocess diagnostics…</div>';
  btn.disabled    = true;

  const key = document.getElementById("gemini-key").value.trim();
  const conf = d ? (d.probs[d.pred || S.regime] || 0) * 100 : 0;

  const body = {
    regime: S.regime,
    t:    d?.t    ?? 0,
    DO:   d?.DO   ?? 0,
    RQ:   d?.RQ   ?? 1,
    OUR:  d?.OUR  ?? 0,
    CER:  d?.CER  ?? 0,
    X:    d?.X    ?? 0,
    S:    d?.S    ?? 0,
    R:    d?.R    ?? 100,
    diw:  d?.diw  ?? 999,
    pred: d?.pred ?? S.regime,
    conf: conf,
    key:  key,
  };

  try {
    const res  = await apiFetch("/explain", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(body),
    });
    const data = await res.json();
    txt.innerHTML = formatExplanation(data.text, data.model, data.source);
  } catch(e) {
    txt.innerHTML = `<div class="exp-card exp-cons"><div class="exp-card-body">Network error: ${escapeHtml(e.message)}</div></div>`;
  } finally {
    btn.disabled = false;
  }
}

// ── ML metrics (Reference tab) ────────────────────────────────────────────────
let _metricsLoaded = false;

async function loadMetrics() {
  if (_metricsLoaded) return;
  try {
    const res  = await apiFetch("/metrics");
    const meta = await res.json();
    _metricsLoaded = true;
    renderF1Chart(meta);
    renderCMChart(meta);
    renderFIChart(meta);
  } catch(e) {
    console.error("Metrics load failed:", e);
  }
}

function renderF1Chart(meta) {
  const pc = meta.per_class;
  if (!pc) return;
  const labels  = Object.keys(pc).map(k=>(REGIME_META[k]||{}).label||k);
  const f1_v    = Object.values(pc).map(v=>v.f1);
  const prec_v  = Object.values(pc).map(v=>v.precision);
  const rec_v   = Object.values(pc).map(v=>v.recall);
  Plotly.newPlot("chart-f1", [
    {name:"F1",        x:labels, y:f1_v,   type:"bar", marker:{color:"#38bdf8"}},
    {name:"Precision", x:labels, y:prec_v, type:"bar", marker:{color:"#a78bfa"}},
    {name:"Recall",    x:labels, y:rec_v,  type:"bar", marker:{color:"#34d399"}},
  ], Object.assign({},PLOTLY_TMPL.layout,{
    barmode:"group", yaxis:{title:"Score",range:[0.8,1.0],gridcolor:"#243352"},
    xaxis:{gridcolor:"#243352"},
  }), {responsive:true, displayModeBar:false});
}

function renderCMChart(meta) {
  const cm = meta.confusion_matrix;
  if (!cm) return;
  const classes = (meta.classes||[]).map(k=>(REGIME_META[k]||{}).label||k);
  const cm_a    = cm.map(row=>row.slice());
  const cm_n    = cm_a.map(row=>{ const s=row.reduce((a,b)=>a+b,0); return row.map(v=>+(v/s).toFixed(3)); });
  const ann     = cm_n.map((row,i)=>row.map((v,j)=>`${cm_a[i][j]}<br>${(v*100).toFixed(0)}%`));
  Plotly.newPlot("chart-cm", [{
    type:"heatmap", z:cm_n, x:classes, y:classes, text:ann, texttemplate:"%{text}",
    colorscale:[[0,"#131e30"],[0.5,"#1d4ed8"],[1,"#22c55e"]], showscale:true,
    colorbar:{tickfont:{color:"#8a9bbf"}},
  }], Object.assign({},PLOTLY_TMPL.layout,{
    xaxis:{title:"Predicted",tickangle:-25,gridcolor:"#243352"},
    yaxis:{title:"True",autorange:"reversed",gridcolor:"#243352"},
  }), {responsive:true, displayModeBar:false});
}

function renderFIChart(meta) {
  const fi = meta.feature_importance;
  if (!fi || !fi.length) return;
  Plotly.newPlot("chart-fi", [{
    type:"bar", orientation:"h",
    x: fi.map(f=>f.importance),
    y: fi.map(f=>f.feature),
    marker:{color:fi.map(f=>f.importance),
            colorscale:[[0,"#1e40af"],[0.5,"#38bdf8"],[1,"#22c55e"]],
            showscale:false},
  }], Object.assign({},PLOTLY_TMPL.layout,{
    margin:{l:200,r:16,t:14,b:40},
    xaxis:{title:"Importance",gridcolor:"#243352"},
    yaxis:{gridcolor:"#243352",tickfont:{size:10}},
  }), {responsive:true, displayModeBar:false});
}

// ── Boot ──────────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  initCharts();
  startStream();
  updateProbs({Healthy:1,kLa_Limitation:0,Substrate_Overfeeding:0,Contamination:0}, "Healthy");
  clearCounterfactual();
});
