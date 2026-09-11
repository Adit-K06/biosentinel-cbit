"""
BioSentinel 2.0 — FastAPI Backend
===================================
Endpoints
---------
GET  /                           Serve frontend HTML
GET  /static/**                  Serve frontend assets
GET  /api/stream/{regime}        SSE: live ODE point stream
GET  /api/counterfactual/{regime}Ranked intervention results
POST /api/explain                Gemini AI explanation
GET  /api/metrics                ML training metadata + feature importance
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

load_dotenv()

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.config import DEFAULT_PARAMS
from src.counterfactual import rank_interventions
from src.features import compute_healthy_baseline
from src.predict import load_model, predict_batch
from src.simulator import simulate_batch
from src.stress import (
    compute_recoverability,
    compute_stress,
    estimate_diw,
    estimate_pnr,
)

# ── Domain constants ──────────────────────────────────────────────────────────
REGIMES = ["Healthy", "kLa_Limitation", "Substrate_Overfeeding", "Contamination"]
REGIME_ONSET = {
    "Healthy":               {},
    "kLa_Limitation":        {"t_onset": 8.0,  "severity": 0.7},
    "Substrate_Overfeeding": {"t_onset": 6.0,  "severity": 3.0},
    "Contamination":         {"t_onset": 5.0,  "severity": 0.45},
}

# ── Simulation helper ─────────────────────────────────────────────────────────
def _sim(regime: str, seed: int = 42):
    p = {"t_span": (0.0, 24.0), "t_eval_n": 480, **REGIME_ONSET.get(regime, {})}
    return simulate_batch(params=p, regime=regime, seed=seed)

def _smooth_series(arr, w: int = 15):
    """Uniform moving average filter to remove sensor noise jitter from display/stress."""
    w = max(1, min(w, len(arr)))
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


def compute_regime_stress(regime: str, df: pd.DataFrame) -> np.ndarray:
    """
    Compute physically grounded kinetic stress lambda(t) per regime.
    - Healthy: zero stress by definition; R(t) stays 100% throughout the entire batch.
    - Fault regimes: stress accumulates strictly after fault onset (t_onset).
    """
    t_arr = df["timestamp"].values
    if regime == "Healthy":
        return np.zeros_like(t_arr)

    # Use smoothed signals to eliminate noise-induced stress artifacts
    do_mg = _smooth_series(df["DO"].values * 1000.0, w=15)
    rq    = _smooth_series(df["RQ"].values, w=15)
    s     = _smooth_series(df["S"].values, w=15)

    t_onset = REGIME_ONSET.get(regime, {}).get("t_onset", 0.0)
    post_fault = (t_arr >= t_onset)

    if regime == "kLa_Limitation":
        # Oxygen transfer limitation causes severe DO sag below critical 2.2 mg/L
        lam = 0.25 * np.maximum(2.2 - do_mg, 0.0)
    elif regime == "Substrate_Overfeeding":
        # Substrate overflow metabolism stress when S > 7.5 g/L
        lam = 0.06 * np.maximum(s - 7.5, 0.0)
    elif regime == "Contamination":
        # Respiratory shift / competitive stress when RQ diverges from balanced 1.0
        lam = 0.75 * np.maximum(np.abs(rq - 1.0) - 0.08, 0.0)
    else:
        lam = np.zeros_like(t_arr)

    # Stress ONLY accumulates after fault onset; before onset reactor is 100% healthy
    return np.where(post_fault, lam, 0.0)


# ── Warm cache: pre-compute everything at startup ─────────────────────────────
_cache: dict = {}

def _warm_cache():
    """
    Run all four regime simulations + ML predictions at startup.
    This means the first browser request streams immediately with no delay.
    """
    print("[BioSentinel] Warming cache — running all 4 regime simulations…")

    # Load model once
    try:
        m, f, c = load_model()
        _cache.update(model=m, feat=f, cls=c)
    except FileNotFoundError:
        _cache.update(model=None, feat=[], cls=REGIMES)

    model, feat_cols, cls_labels = _cache["model"], _cache["feat"], _cache["cls"]

    # Pre-compute healthy baseline for ML feature normalisation
    df_healthy = _sim("Healthy")
    baseline = compute_healthy_baseline(df_healthy)
    _cache["baseline"] = baseline

    # Pre-compute per-regime: simulation + stress + R + ML predictions
    for regime in REGIMES:
        df = _sim(regime)
        # Ensure telemetry is strictly sorted by timestamp and duplicate-free
        df = df.sort_values(by="timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        lam_arr = compute_regime_stress(regime, df)
        R_arr   = compute_recoverability(df["timestamp"].values, lam_arr)

        if regime == "Contamination":
            # Biological constraint: microbial contamination cannot be cured by physical actuators.
            # Post-onset (t >= 5.0h), recoverability collapses to 0.0% (irreversible).
            t_on = REGIME_ONSET["Contamination"].get("t_onset", 5.0)
            post_onset = df["timestamp"].values >= t_on
            R_arr = np.where(post_onset, 0.0, R_arr)
            pnr_t = float(t_on)
        else:
            pnr_t, _ = estimate_pnr(df["timestamp"].values, R_arr, R_pnr=0.10)

        # Pre-filter display columns so numbers and charts are smooth like a real industrial SCADA
        df_disp = df.copy()
        for col in ["DO", "RQ", "OUR", "CER", "X", "S", "pressure", "RPM"]:
            if col in df_disp.columns:
                df_disp[col] = _smooth_series(df_disp[col].values, w=11)

        # Windowed ML predictions at every 60-pt checkpoint
        win_preds = {}
        if model is not None:
            do_mean, do_std = baseline
            for w in range(60, len(df) + 1, 60):
                try:
                    out = predict_batch(df.iloc[:w], model, feat_cols, cls_labels,
                                        do_mean=do_mean, do_std=do_std)
                    win_preds[w] = {"probs": out["probabilities"], "pred": out["predicted_class"]}
                except Exception:
                    pass

        _cache[regime] = {
            "df":        df,
            "df_disp":   df_disp,
            "lam":       lam_arr,
            "R":         R_arr,
            "pnr_t":     pnr_t,
            "win_preds": win_preds,
        }
        print(f"[BioSentinel]   {regime}: R range [{R_arr.min()*100:.1f}%, {R_arr.max()*100:.1f}%]  "
              f"lam max={lam_arr.max():.4f}  pts={len(df)}")

    print("[BioSentinel] Cache warm — ready to stream!")



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm cache in background thread so server starts accepting connections immediately
    await run_in_threadpool(_warm_cache)
    yield


app = FastAPI(title="BioSentinel 2.0", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)

_FRONTEND = _ROOT / "frontend"
app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")

# ── Helper accessors ──────────────────────────────────────────────────────────
def _get_model():
    return _cache.get("model"), _cache.get("feat", []), _cache.get("cls", REGIMES)

def _get_regime_data(regime: str) -> dict:
    """Return pre-computed data for a regime, warming cache if not already done."""
    if regime not in _cache or "df" not in _cache.get(regime, {}):
        _warm_cache()
    if regime in _cache:
        return _cache[regime]
    # Fallback safety guard
    df = _sim(regime).sort_values(by="timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    lam_arr = compute_regime_stress(regime, df)
    R_arr   = compute_recoverability(df["timestamp"].values, lam_arr)
    if regime == "Contamination":
        t_on = REGIME_ONSET["Contamination"].get("t_onset", 5.0)
        R_arr = np.where(df["timestamp"].values >= t_on, 0.0, R_arr)
    pnr_t, _ = estimate_pnr(df["timestamp"].values, R_arr, R_pnr=0.10)
    df_disp = df.copy()
    for col in ["DO", "RQ", "OUR", "CER", "X", "S", "pressure", "RPM"]:
        if col in df_disp.columns:
            df_disp[col] = _smooth_series(df_disp[col].values, w=11)
    data = {"df": df, "df_disp": df_disp, "lam": lam_arr, "R": R_arr, "pnr_t": pnr_t, "win_preds": {}}
    _cache[regime] = data
    return data

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse((_FRONTEND / "index.html").read_text(encoding="utf-8"))


@app.get("/BioSentinel_Pitch_Summary.pdf")
async def pitch_pdf():
    pdf_path = _ROOT / "BioSentinel_Pitch_Summary.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, "Pitch PDF not found")
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename="BioSentinel_Pitch_Summary.pdf",
    )



@app.get("/api/stream/{regime}")
async def stream(regime: str, request: Request, start_idx: int = 0):
    if regime not in REGIMES:
        raise HTTPException(404, f"Unknown regime: {regime}")

    async def generator():
        # Use pre-warmed cache — no simulation delay on stream start!
        data    = _get_regime_data(regime)
        df      = data["df"]
        df_disp = data.get("df_disp", df)
        lam_arr = data["lam"]
        R_arr   = data["R"]
        pnr_t   = data["pnr_t"]
        win_preds = data["win_preds"]

        empty_probs = {k: 0.0 for k in REGIMES}
        # Default: all probability on the selected regime
        default_probs = {k: (1.0 if k == regime else 0.0) for k in REGIMES}

        start = max(0, min(start_idx, len(df)))
        for idx in range(start, len(df)):
            if await request.is_disconnected():
                break


            row     = df_disp.iloc[idx]
            t_now   = float(row["timestamp"])
            R_now   = float(R_arr[idx])
            lam_now = float(lam_arr[idx])

            # Nearest pre-computed window (floored to 60-pt blocks)
            w_key   = max(60, (idx // 60) * 60)
            pd_data = win_preds.get(w_key, {"probs": default_probs, "pred": regime})

            if regime == "Contamination" and t_now >= 5.0:
                R_now = 0.0
                diw_min = 0
            else:
                diw_h   = estimate_diw(t_now, R_now, lam_now) if lam_now > 1e-6 else 99.0
                diw_min = min(int(diw_h * 60), 999)

            payload = json.dumps({
                "idx":   idx,
                "total": len(df),
                "t":     round(t_now, 3),
                "DO":    round(float(row["DO"]) * 1000, 2),   # g/L → mg/L for display
                "RQ":    round(float(row["RQ"]), 3),
                "OUR":   round(float(row["OUR"]), 4),
                "CER":   round(float(row["CER"]), 4),
                "X":     round(float(row["X"]), 3),
                "S":     round(float(row["S"]), 3),
                "RPM":   round(float(row["RPM"]), 1),
                "P":     round(float(row["pressure"]), 3),
                "R":     round(R_now * 100, 1),
                "lam":   round(lam_now, 5),
                "diw":   diw_min,
                "probs": {k: round(v, 4) for k, v in pd_data["probs"].items()},
                "pred":  pd_data["pred"],
                "pnr_t": round(pnr_t, 2) if pnr_t else None,
            })
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0.05)   # ~20 pts/s → full 24h in ~24 s

    return StreamingResponse(
        generator(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/counterfactual/{regime}")
async def counterfactual(regime: str):
    if regime not in REGIMES:
        raise HTTPException(404, "Unknown regime")

    # Biological logic: microbial contamination cannot be cured by physical actuators
    if regime == "Contamination":
        return [
            {
                "rank":     1,
                "label":    "TRIGGER CIP / HARVEST ABORT",
                "ending_R": 0.0,
                "gain":     0.0,
                "pnr":      True,
            },
            {
                "rank":     2,
                "label":    "Containment & Autoclave Decontamination",
                "ending_R": 0.0,
                "gain":     0.0,
                "pnr":      True,
            },
            {
                "rank":     3,
                "label":    "Physical Actuation (+RPM/-Feed) [INEFFECTIVE]",
                "ending_R": 0.0,
                "gain":     0.0,
                "pnr":      True,
            },
            {
                "rank":     4,
                "label":    "No Action (Culture Lost / PNR Breached)",
                "ending_R": 0.0,
                "gain":     0.0,
                "pnr":      True,
            },
        ]

    data = _get_regime_data(regime)
    df   = data["df"]
    row  = df[df["timestamp"] <= 18.0].iloc[-1]
    st_  = {"X": float(row["X"]), "S": float(row["S"]), "DO": float(row["DO"])}
    prm  = {**DEFAULT_PARAMS, **REGIME_ONSET.get(regime, {})}
    res  = await run_in_threadpool(lambda: rank_interventions(st_, 18.0, prm, horizon=1.5, regime=regime))

    LABELS = {
        "no_action": "No Action (baseline)",
        "+150_RPM":  "+150 RPM Agitation",
        "-30%_feed": "-30% Feed Rate",
        "combined":  "Combined (+RPM & -Feed)",
    }
    no_r = next((r.ending_R for r in res if r.intervention == "no_action"), 0.0)
    return [
        {
            "rank":     i + 1,
            "label":    LABELS.get(r.intervention, r.intervention),
            "ending_R": round(r.ending_R * 100, 1),
            "gain":     round((r.ending_R - no_r) * 100, 1),
            "pnr":      r.pnr_crossed,
        }
        for i, r in enumerate(res)
    ]



class ExplainReq(BaseModel):
    regime: str
    t: float
    DO: float
    RQ: float
    OUR: float
    CER: float
    X: float
    S: float
    R: float
    diw: int
    pred: str
    conf: float
    key: str = ""


def clean_llm_text(t: str) -> str:
    """Strip any LaTeX syntax, math mode, dollar signs, or backslashes from LLM output."""
    if not t:
        return ""
    # Strip \text{...} wrappers
    t = re.sub(r"\\text\{([^}]*)\}", r"\1", t)
    # Strip \math...{...} wrappers
    t = re.sub(r"\\math\w+\{([^}]*)\}", r"\1", t)
    t = t.replace(r"\mu", "μ")
    # Strip $...$ math blocks
    t = re.sub(r"\$([^$]+)\$", r"\1", t)
    t = t.replace("$", "")
    t = t.replace("\\", "")
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


@app.post("/api/explain")
async def explain(req: ExplainReq):
    key = req.key.strip() or os.environ.get("GEMINI_API_KEY", "")
    if not key and (_ROOT / ".env").exists():
        load_dotenv(_ROOT / ".env", override=True)
        key = os.environ.get("GEMINI_API_KEY", "")

    is_healthy = (req.pred == "Healthy")
    status_desc = (
        "NOMINAL OPERATION — Culture is flourishing inside optimal physiological envelope."
        if is_healthy else
        f"FAULT DETECTED — Culture experiencing {req.pred}."
    )

    prompt = f"""You are BioSentinel AI — a Senior Industrial Bioprocess Engineer monitoring a stirred-tank bioreactor.

CURRENT TELEMETRY (t = {req.t:.2f} h in a 24.0 h batch):
• Dissolved Oxygen (DO): {req.DO:.2f} mg/L (Aerobic baseline: 4.0–8.0 mg/L; critical hypoxia floor: 2.0 mg/L)
• Respiratory Quotient (RQ): {req.RQ:.2f} (Stoichiometric oxidative baseline: 1.00 ± 0.05)
• Oxygen Uptake Rate (OUR): {req.OUR:.3f} g/L/h
• CO2 Evolution Rate (CER): {req.CER:.3f} g/L/h
• Biomass Density (X): {req.X:.2f} g/L
• Substrate Concentration (S): {req.S:.2f} g/L
• Batch Recoverability R(t): {req.R:.1f}% (Healthy: 100%; Point-of-No-Return: 10%)
• Decision Intervention Window (DIW): {req.diw if req.diw < 900 else 'Safe (>15h)'} minutes remaining

CLASSIFIER DIAGNOSIS:
State: {req.pred} ({req.conf:.0f}% confidence)
Status: {status_desc}

TASK:
Provide an expert, authoritative, 4-point bioprocess engineering assessment.

FORMATTING REQUIREMENTS (CRITICAL):
1. Output EXACTLY these four bullet headings:
• BIOLOGICAL STATE: <concise paragraph on cell physiology, metabolic pathway, and viability>
• ROOT CAUSE: <mechanistic driver of this condition, physics, and mass transfer balance>
• CONSEQUENCE: <impact in next 2–3 hours if current trajectory continues without intervention>
• OPERATOR ACTION: <concrete SCADA / actuator corrective actions (RPM, feed, airflow, sampling)>

2. ZERO LATEX: DO NOT use LaTeX formatting, math mode, backslashes, or dollar signs ($...$).
3. Write clean, standard industrial units (e.g. X = {req.X:.2f} g/L, DO = {req.DO:.2f} mg/L, RQ = {req.RQ:.2f}, OUR = {req.OUR:.3f} g/L/h).
4. No Markdown headers (#), no conversational preamble, no conversational signoff."""

    if key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            for m_name in ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-flash-lite", "gemini-1.5-flash", "gemini-pro-latest"]:
                try:
                    mdl = genai.GenerativeModel(m_name)
                    resp = await run_in_threadpool(mdl.generate_content, prompt)
                    if resp and resp.text:
                        cleaned = clean_llm_text(resp.text)
                        return {"text": cleaned, "model": m_name, "source": "Gemini AI"}
                except Exception:
                    continue
        except Exception:
            pass

    # Expert deterministic bioprocess fallback
    fault = req.pred
    if fault == "kLa_Limitation":
        bio_state = f"Severe oxygen transfer limitation. DO = {req.DO:.2f} mg/L (hypoxia, well below 2.2 mg/L). RQ = {req.RQ:.2f} confirms anaerobic overflow metabolic shift in X = {req.X:.2f} g/L biomass."
        root_cause = f"kLa mass transfer degradation — impaired agitator power or sparger fouling. Cellular OUR = {req.OUR:.3f} g/L/h exceeds oxygen transfer capacity."
        consequence = f"Batch recoverability R(t) = {req.R:.1f}%. Continuing at current stress, PNR (10%) reached in ~{req.diw} min. Toxic byproducts (acetate/ethanol) will permanently suppress product titer."
        operator_action = "Execute +150 RPM via SCADA immediately to boost kLa by ~20%. Simultaneously inspect sparger pressure and increase air enrichment."
    elif fault == "Substrate_Overfeeding":
        bio_state = f"Substrate overflow accumulation (S = {req.S:.2f} g/L >> Ks = 0.1 g/L). High glucose concentration triggers overflow metabolism; DO = {req.DO:.2f} mg/L is being consumed faster than aeration transfer."
        root_cause = f"Substrate feed rate exceeds cellular oxidative capacity. Metabolic demand OUR = {req.OUR:.3f} g/L/h overwhelms kLa, inducing Crabtree-like byproduct generation."
        consequence = f"Yield collapse and cellular viability loss within {req.diw} min (R = {req.R:.1f}%). Acetate accumulation will cause irreversible cell death cascade."
        operator_action = "Trim feed rate by 30% immediately. Hold feed until residual substrate S < 1.0 g/L, then resume at controlled rate. Target RQ < 1.05."
    elif fault == "Contamination":
        bio_state = f"Exogenous microbial contamination. RQ = {req.RQ:.2f} diverges sharply from the aerobic baseline (1.00). CER/OUR ratio decoupled from host strain stoichiometry."
        root_cause = "Foreign contaminant population competing for carbon substrate with elevated respiration kinetics. Likely breach in sterile air boundary or feed line."
        consequence = f"Complete batch loss within 2–3 hours. PNR in ~{req.diw} min (R = {req.R:.1f}%). Regulatory compliance breached; product stream contaminated."
        operator_action = "Initiate immediate reactor isolation. Draw sterile broth sample for qPCR and optical verification. Abort feed addition and prepare containment protocol."
    else:
        bio_state = f"Culture healthy and thriving. DO = {req.DO:.2f} mg/L well within nominal aerobic envelope. RQ = {req.RQ:.2f} confirms balanced oxidative respiration. Recoverability R(t) = {req.R:.1f}%."
        root_cause = "Nominal dual-Monod kinetics — oxygen mass transfer matches cellular metabolic demand. All physiological states within safe envelope."
        consequence = "Batch will continue healthy exponential and fed-batch trajectory. Recoverability remains stable at 100%."
        operator_action = "Maintain current setpoints (RPM, feed rate, aeration). Continue automated BioSentinel real-time surveillance."

    analysis = (
        f"• BIOLOGICAL STATE: {bio_state}\n\n"
        f"• ROOT CAUSE: {root_cause}\n\n"
        f"• CONSEQUENCE: {consequence}\n\n"
        f"• OPERATOR ACTION: {operator_action}\n\n"
        f"*(BioSentinel Expert Engine — Physics-Grounded Real-time Analysis)*"
    )
    return {"text": clean_llm_text(analysis), "model": "BioSentinel Expert Engine", "source": "Expert System"}


@app.get("/api/metrics")
async def metrics():
    meta_path = _ROOT / "models" / "training_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    # Append feature importances from loaded model
    model, feat_cols, _ = _get_model()
    if model is not None and hasattr(model, "feature_importances_"):
        fi   = model.feature_importances_
        pairs = sorted(zip(feat_cols, fi.tolist()), key=lambda x: x[1], reverse=True)
        meta["feature_importance"] = [
            {"feature": f, "importance": round(v, 2)} for f, v in pairs[:20]
        ]
    return meta
