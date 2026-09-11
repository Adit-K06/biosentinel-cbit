"""
src/simulator.py
================
BioSentinel 2.0 — Physics-Informed Digital Twin (Layer 1).

Implements a dual-Monod fed-batch bioreactor ODE system and integrates it
with ``scipy.integrate.solve_ivp``.  Three failure regimes are supported:

  1. kLa Limitation        — progressive oxygen-transfer fouling
  2. Substrate Overfeeding — operator over-feed causing overflow metabolism
  3. Contamination         — second microbial population causing DO/RQ divergence

Public API
----------
    simulate_batch(params, regime, seed) -> pandas.DataFrame

The returned DataFrame has the stable schema::

    timestamp, DO, RQ, OUR, CER, pressure, RPM, airflow, OD600,
    regime, t_onset, X, S

No Streamlit / UI / database / API / Docker imports.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from src.config import DEFAULT_PARAMS, REGIME_OVERRIDES
from src.regimes import get_regime, ContaminationRegime

__all__ = ["simulate_batch"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Respiratory quotient baseline (moles CO2 / mole O2) for aerobic growth
# on a glucose-like carbon source.  Used to derive CER from OUR.
_RQ_HEALTHY = 1.0   # unity for balanced aerobic oxidation of glucose

# Molecular weights ratio O2/CO2 → used to convert mass-based OUR to CER
_MW_RATIO_CO2_O2 = 44.0 / 32.0  # ≈ 1.375 (g CO2 per g O2 at RQ=1)

# OD600 ≈ 2.5 × [biomass in g/L]  (empirical for E. coli / yeast)
_OD600_FACTOR = 2.5

# ---------------------------------------------------------------------------
# ODE right-hand side
# ---------------------------------------------------------------------------


def _build_rhs(regime, params_base: Dict[str, Any]):
    """Return the ODE RHS closure for solve_ivp.

    State vector: [X, S, CL] for non-contamination regimes,
                  [X, S, CL, X_c] for Contamination.
    """

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        # --- Unpack state ---
        X = max(y[0], 0.0)
        S = max(y[1], 0.0)
        CL = max(y[2], 0.0)

        # --- Get effective params (possibly mutated by regime) ---
        p = regime.mutate(t, tuple(y), params_base)

        mu_max = p["mu_max"]
        K_s    = p["K_s"]
        K_O    = p["K_O"]
        Y_xs   = p["Y_xs"]
        Y_xo   = p["Y_xo"]
        m_s    = p["m_s"]
        m_o    = p["m_o"]
        kLa    = p["kLa"]
        CL_s   = p["CL_star"]
        F_s    = p["F_s"]

        # --- Dual-Monod growth rate ---
        mu = (
            mu_max
            * (S  / (K_s + S  + 1e-12))
            * (CL / (K_O + CL + 1e-12))
        )

        # --- Biomass (primary population) ---
        dX = mu * X

        # --- Substrate ---
        # growth consumption + maintenance - feed
        dS = -((mu / Y_xs) + m_s) * X + F_s

        # --- Oxygen uptake rate (primary population, g/L/h) ---
        OUR_primary = ((mu / Y_xo) + m_o) * X

        # --- Contaminant contribution ---
        OUR_cont = 0.0
        dX_c = 0.0
        if isinstance(regime, ContaminationRegime) and len(y) > 3:
            X_c = max(y[3], 0.0)
            (dX_c,) = regime.extra_rhs(t, tuple(y), p)
            # Contaminant also consumes substrate and O2
            mu_c_effective = dX_c / (X_c + 1e-12)
            dS     -= (mu_c_effective / Y_xs + m_s) * X_c
            OUR_cont = (mu_c_effective / regime.Y_xo_cont + m_o) * X_c

        # --- Dissolved oxygen mass balance ---
        OUR_total = OUR_primary + OUR_cont
        dCL = kLa * (CL_s - CL) - OUR_total

        if isinstance(regime, ContaminationRegime) and len(y) > 3:
            return np.array([dX, dS, dCL, dX_c])
        return np.array([dX, dS, dCL])

    return rhs


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def _add_noise(arr: np.ndarray, pct: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise with std = pct * |arr|, clipped ≥ 0."""
    noise = rng.normal(0.0, pct * np.abs(arr))
    return np.clip(arr + noise, 0.0, None)


def _compute_derived(
    t: np.ndarray,
    X: np.ndarray,
    S: np.ndarray,
    CL: np.ndarray,
    X_c: Optional[np.ndarray],
    params: Dict[str, Any],
    regime_obj,
) -> Dict[str, np.ndarray]:
    """Compute OUR, CER, RQ from integrated trajectories."""
    n = len(t)
    mu_max = params["mu_max"]
    K_s    = params["K_s"]
    K_O    = params["K_O"]
    Y_xo   = params["Y_xo"]
    m_o    = params["m_o"]

    # Effective params may change over time; iterate per timestep
    OUR = np.zeros(n)
    CER = np.zeros(n)
    RQ  = np.zeros(n)

    for i in range(n):
        state_i = (X[i], S[i], CL[i]) if X_c is None else (X[i], S[i], CL[i], X_c[i])
        p = regime_obj.mutate(t[i], state_i, params)

        mu_i = (
            p["mu_max"]
            * (max(S[i], 0.0) / (p["K_s"] + max(S[i], 0.0) + 1e-12))
            * (max(CL[i], 0.0) / (p["K_O"] + max(CL[i], 0.0) + 1e-12))
        )

        # Primary OUR
        our_i = ((mu_i / p["Y_xo"]) + p["m_o"]) * max(X[i], 0.0)

        # Contaminant OUR
        if X_c is not None and isinstance(regime_obj, ContaminationRegime):
            Xc_i = max(X_c[i], 0.0)
            mu_c = (
                regime_obj.mu_max_cont
                * (max(S[i], 0.0) / (regime_obj.K_s_cont + max(S[i], 0.0) + 1e-12))
                * (max(CL[i], 0.0) / (p["K_O"] + max(CL[i], 0.0) + 1e-12))
            )
            our_i += (mu_c / regime_obj.Y_xo_cont + p["m_o"]) * Xc_i

        # CER derived from stoichiometric RQ relationship
        # RQ = CER / OUR; for contamination regime RQ rises as overflow occurs
        if isinstance(regime_obj, ContaminationRegime) and regime_obj.t_onset is not None and t[i] >= regime_obj.t_onset:
            # Contamination shifts apparent RQ upward due to mixed fermentative/respiratory metabolism
            rq_i = _RQ_HEALTHY + regime_obj.severity * 0.5 * min((t[i] - regime_obj.t_onset) / 6.0, 1.0)
        else:
            rq_i = _RQ_HEALTHY

        cer_i = our_i * rq_i * _MW_RATIO_CO2_O2

        OUR[i] = our_i
        CER[i] = cer_i
        RQ[i]  = rq_i

    return {"OUR": OUR, "CER": CER, "RQ": RQ}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def simulate_batch(
    params: Optional[Dict[str, Any]] = None,
    regime: str = "Healthy",
    seed: int = 0,
) -> pd.DataFrame:
    """Run a digital-twin batch simulation and return a tidy DataFrame.

    Parameters
    ----------
    params : dict, optional
        Override any key in ``src.config.DEFAULT_PARAMS``.  Regime-specific
        overrides from ``REGIME_OVERRIDES`` are applied automatically.
    regime : str
        One of ``"Healthy"``, ``"kLa_Limitation"``,
        ``"Substrate_Overfeeding"``, ``"Contamination"``.
        Case-insensitive.
    seed : int
        Random seed for reproducible sensor noise.

    Returns
    -------
    pandas.DataFrame
        Stable schema:
        ``timestamp, DO, RQ, OUR, CER, pressure, RPM, airflow, OD600,
          regime, t_onset, X, S``
    """
    # ------------------------------------------------------------------
    # 1. Build effective parameter dict
    # ------------------------------------------------------------------
    p = copy.deepcopy(DEFAULT_PARAMS)

    # Apply regime-specific defaults
    regime_key = regime.strip()
    regime_overrides = REGIME_OVERRIDES.get(regime_key, {})
    p.update(regime_overrides)

    # Apply caller-supplied overrides (highest priority)
    if params:
        p.update(params)

    # ------------------------------------------------------------------
    # 2. Instantiate regime object
    # ------------------------------------------------------------------
    regime_obj = get_regime(regime_key, p)

    # ------------------------------------------------------------------
    # 3. Build initial state vector
    # ------------------------------------------------------------------
    y0_base = [p["X0"], p["S0"], p["CL0"]]
    extra_init = list(regime_obj.initial_extra_states(p))
    y0 = np.array(y0_base + extra_init, dtype=float)

    # ------------------------------------------------------------------
    # 4. Build time grid
    # ------------------------------------------------------------------
    t0, tf = p["t_span"]
    n_pts  = int(p["t_eval_n"])
    t_eval = np.linspace(t0, tf, n_pts)

    # ------------------------------------------------------------------
    # 5. Integrate ODE
    # ------------------------------------------------------------------
    rhs = _build_rhs(regime_obj, p)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = solve_ivp(
            fun=rhs,
            t_span=(t0, tf),
            y0=y0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-5,
            atol=1e-8,
            max_step=(tf - t0) / 100,
        )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    t_arr = sol.t
    X_arr = np.clip(sol.y[0], 0.0, None)
    S_arr = np.clip(sol.y[1], 0.0, None)
    CL_arr = np.clip(sol.y[2], 0.0, None)
    X_c_arr = np.clip(sol.y[3], 0.0, None) if sol.y.shape[0] > 3 else None

    # ------------------------------------------------------------------
    # 6. Derived signals
    # ------------------------------------------------------------------
    derived = _compute_derived(t_arr, X_arr, S_arr, CL_arr, X_c_arr, p, regime_obj)
    OUR_arr = derived["OUR"]
    CER_arr = derived["CER"]
    RQ_arr  = derived["RQ"]

    # OD600 ≈ 2.5 × X (g/L) [empirical calibration]
    OD600_arr = X_arr * _OD600_FACTOR
    if X_c_arr is not None:
        # Optical density sees all cells
        OD600_arr = (X_arr + X_c_arr) * _OD600_FACTOR

    # ------------------------------------------------------------------
    # 7. Add synthetic sensor noise (5 % Gaussian, nonneg-clipped)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    noise_pct = float(p.get("noise_pct", 0.05))

    CL_noisy    = _add_noise(CL_arr,    noise_pct, rng)
    OUR_noisy   = _add_noise(OUR_arr,   noise_pct, rng)
    CER_noisy   = _add_noise(CER_arr,   noise_pct, rng)
    OD600_noisy = _add_noise(OD600_arr, noise_pct, rng)
    X_noisy     = _add_noise(X_arr,     noise_pct, rng)
    S_noisy     = _add_noise(S_arr,     noise_pct, rng)
    # RQ is a ratio; add small absolute noise proportional to its value
    RQ_noisy    = np.clip(RQ_arr + rng.normal(0, noise_pct * RQ_arr), 0.5, 3.0)

    # Instrument readings (constant ± small noise)
    RPM_arr      = _add_noise(np.full(len(t_arr), p["RPM"]),      0.01, rng)
    airflow_arr  = _add_noise(np.full(len(t_arr), p["airflow"]),  0.01, rng)
    pressure_arr = _add_noise(np.full(len(t_arr), p["pressure"]), 0.005, rng)

    # ------------------------------------------------------------------
    # 8. Assemble DataFrame with stable schema
    # ------------------------------------------------------------------
    t_onset_val = p.get("t_onset", None)

    df = pd.DataFrame(
        {
            "timestamp": t_arr,          # hours since inoculation
            "DO":        CL_noisy,       # g/L dissolved oxygen (noisy)
            "RQ":        RQ_noisy,       # dimensionless respiratory quotient
            "OUR":       OUR_noisy,      # g/(L·h) oxygen uptake rate
            "CER":       CER_noisy,      # g/(L·h) CO2 evolution rate
            "pressure":  pressure_arr,   # atm
            "RPM":       RPM_arr,        # rpm
            "airflow":   airflow_arr,    # vvm
            "OD600":     OD600_noisy,    # AU (absorbance units)
            "regime":    regime_key,     # string label
            "t_onset":   t_onset_val,    # float or NaN
            "X":         X_noisy,        # g/L biomass (noisy)
            "S":         S_noisy,        # g/L substrate (noisy)
        }
    )

    # Replace None t_onset with NaN for consistent numeric handling
    df["t_onset"] = pd.to_numeric(df["t_onset"], errors="coerce")

    # Sanity guard: enforce nonneg on physical quantities
    for col in ["DO", "OUR", "CER", "OD600", "X", "S", "RPM", "airflow", "pressure"]:
        df[col] = df[col].clip(lower=0.0)

    # Ensure strictly sorted timestamp without duplicate rows (prevents Plotly loopback glitches)
    df = df.sort_values(by="timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    return df

