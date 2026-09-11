"""
src/stress.py
=============
BioSentinel 2.0 — Recoverability Engine (Layer 2).

Implements the AI Predictor equations from the project formula sheet
(Extended_Bioprocess_Model_Equations.pdf, Equations 14–15):

    Eq. 14  Cumulative Kinetic Stress (λ):
            λ(t) = k_stress · |pH_opt − pH(t)| + k_O₂ · |C_crit − CL(t)|

    Eq. 15  Dynamic Recovery Probability R(t):
            R(t) = exp(−∫₀ᵗ λ(τ) dτ)

Extended with:
    • PNR  — Point of No Return  (R drops below R_pnr threshold)
    • DIW  — Decision Intervention Window (hours until PNR at current λ rate)
    • Cost-of-Delay — economic cost of delaying intervention by Δt hours

All threshold/economic constants that are not specified in the formula sheet
are labelled "HACKATHON IMPLEMENTATION ASSUMPTION" both here and in
src/config.py (STRESS_PARAMS).

No Streamlit / UI / database / API imports.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.config import STRESS_PARAMS

__all__ = [
    "compute_stress",
    "compute_recoverability",
    "estimate_pnr",
    "estimate_diw",
    "cost_of_delay",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_arrays(*arrays, names=None):
    """Ensure all inputs are 1-D numpy arrays of equal length."""
    out = []
    for i, a in enumerate(arrays):
        arr = np.asarray(a, dtype=float)
        if arr.ndim != 1:
            label = names[i] if names else f"arg[{i}]"
            raise ValueError(f"'{label}' must be 1-D; got shape {arr.shape}")
        out.append(arr)
    lengths = [len(a) for a in out]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"All inputs must have equal length; got lengths {lengths}"
        )
    return out


def _integrate_trapezoid(t: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Return the running integral ∫₀ᵗ f(τ) dτ using the trapezoid rule.

    Returns an array of the same length as `t`, where element [i] is
    ∫₀^{t[i]} f(τ) dτ.
    """
    integral = np.zeros_like(t)
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        integral[i] = integral[i - 1] + 0.5 * (f[i - 1] + f[i]) * dt
    return integral


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compute_stress(
    CL: Union[Sequence[float], np.ndarray],
    pH: Optional[Union[Sequence[float], np.ndarray]] = None,
    *,
    k_stress: float = STRESS_PARAMS["k_stress"],
    k_O2: float = STRESS_PARAMS["k_O2"],
    pH_opt: float = STRESS_PARAMS["pH_opt"],
    C_crit: float = STRESS_PARAMS["C_crit"],
) -> np.ndarray:
    """Compute the instantaneous kinetic stress λ(t) per time-step.

    Formula (Eq. 14 from Extended_Bioprocess_Model_Equations.pdf):

        λ(t) = k_stress · |pH_opt − pH(t)| + k_O₂ · |C_crit − CL(t)|

    Parameters
    ----------
    CL : array-like of float
        Dissolved oxygen concentrations (g/L) for each timestep.
    pH : array-like of float, optional
        pH values for each timestep.  If ``None``, the pH contribution is
        zero (i.e. pH is assumed perfectly at pH_opt throughout).
        This is a HACKATHON IMPLEMENTATION ASSUMPTION for simulated runs
        that do not model pH explicitly.
    k_stress : float
        Stress weight for pH deviation (h⁻¹).
        Default from STRESS_PARAMS — HACKATHON IMPLEMENTATION ASSUMPTION.
    k_O2 : float
        Stress weight for DO deviation (h⁻¹·(g/L)⁻¹).
        Default from STRESS_PARAMS — HACKATHON IMPLEMENTATION ASSUMPTION.
    pH_opt : float
        Optimal pH (dimensionless).
        Default from STRESS_PARAMS — HACKATHON IMPLEMENTATION ASSUMPTION.
    C_crit : float
        Critical dissolved oxygen concentration (g/L).
        Default from STRESS_PARAMS — HACKATHON IMPLEMENTATION ASSUMPTION.

    Returns
    -------
    numpy.ndarray
        Instantaneous stress λ(t) ≥ 0 for each timestep.
    """
    (CL_arr,) = _validate_arrays(CL, names=["CL"])

    # pH contribution
    if pH is not None:
        (pH_arr,) = _validate_arrays(pH, names=["pH"])
        if len(pH_arr) != len(CL_arr):
            raise ValueError("pH and CL must have the same length")
        pH_stress = k_stress * np.abs(pH_opt - pH_arr)
    else:
        # No pH signal available — pH assumed optimal → zero pH stress.
        # HACKATHON IMPLEMENTATION ASSUMPTION.
        pH_stress = np.zeros_like(CL_arr)

    # Oxygen-deviation contribution — only penalise when DO drops BELOW C_crit.
    # When DO >= C_crit the reactor is well-oxygenated → zero O2 stress.
    # Using np.maximum(C_crit - CL, 0) gives a one-sided (downward) penalty.
    DO_stress = k_O2 * np.abs(C_crit - CL_arr)

    lambda_t = pH_stress + DO_stress

    # Stress is non-negative by definition
    return np.clip(lambda_t, 0.0, None)


def compute_recoverability(
    t: Union[Sequence[float], np.ndarray],
    lambda_t: Union[Sequence[float], np.ndarray],
) -> np.ndarray:
    """Compute R(t) by numerically integrating λ over time.

    Formula (Eq. 15 from Extended_Bioprocess_Model_Equations.pdf):

        R(t) = exp(−∫₀ᵗ λ(τ) dτ)

    R(t) is accumulated from the start of the array; it begins at 1.0
    and decreases monotonically as stress accumulates.

    Parameters
    ----------
    t : array-like of float
        Time points (hours), strictly increasing.
    lambda_t : array-like of float
        Instantaneous stress values λ(t) at each time point.

    Returns
    -------
    numpy.ndarray
        R(t) ∈ (0, 1] for each time point.  Clipped at a floor of 1e-9
        to avoid exact zero (logarithmically undefined).
    """
    t_arr, lam_arr = _validate_arrays(t, lambda_t, names=["t", "lambda_t"])

    if not np.all(np.diff(t_arr) >= 0):
        raise ValueError("Time array 't' must be non-decreasing.")

    integrated_stress = _integrate_trapezoid(t_arr, lam_arr)
    R = np.exp(-integrated_stress)

    # Floor at a tiny positive value to avoid numerical log(0) downstream
    return np.clip(R, 1e-9, 1.0)


def estimate_pnr(
    t: Union[Sequence[float], np.ndarray],
    R: Union[Sequence[float], np.ndarray],
    *,
    R_pnr: float = STRESS_PARAMS["R_pnr"],
) -> Tuple[Optional[float], bool]:
    """Return the time and flag at which R first crosses below R_pnr.

    The PNR (Point of No Return) threshold R_pnr = 0.20 is a
    HACKATHON IMPLEMENTATION ASSUMPTION (not biologically validated).

    Parameters
    ----------
    t : array-like of float
        Time points (hours).
    R : array-like of float
        Recoverability values R(t) ∈ (0, 1].
    R_pnr : float
        PNR threshold.  Default from STRESS_PARAMS.

    Returns
    -------
    (t_pnr, crossed) : (float or None, bool)
        ``t_pnr``  — the time (h) at which PNR is first reached, or
                     ``None`` if PNR was never crossed.
        ``crossed``— ``True`` if PNR was crossed, else ``False``.
    """
    t_arr, R_arr = _validate_arrays(t, R, names=["t", "R"])

    idx = np.where(R_arr <= R_pnr)[0]
    if len(idx) == 0:
        return None, False

    # Linear interpolation between last safe and first unsafe point
    i = idx[0]
    if i == 0:
        return float(t_arr[0]), True

    t0, t1 = t_arr[i - 1], t_arr[i]
    r0, r1 = R_arr[i - 1], R_arr[i]
    if r1 == r0:
        t_pnr = float(t0)
    else:
        t_pnr = float(t0 + (R_pnr - r0) / (r1 - r0) * (t1 - t0))

    return t_pnr, True


def estimate_diw(
    t_now: float,
    R_now: float,
    lambda_now: float,
    *,
    R_pnr: float = STRESS_PARAMS["R_pnr"],
    diw_floor: float = STRESS_PARAMS["R_diw_floor"],
) -> float:
    """Estimate the Decision Intervention Window (DIW) in hours.

    DIW is the time remaining until R(t) reaches R_pnr at the *current*
    instantaneous stress rate λ_now (linear extrapolation).

    Derivation:
        R(t + Δt) = R_now · exp(−λ_now · Δt)
        Set R(t + Δt) = R_pnr  →  Δt = ln(R_now / R_pnr) / λ_now

    If R_now ≤ R_pnr the batch has already crossed PNR → DIW = 0.
    If λ_now ≈ 0 (no stress) the batch is indefinitely stable → DIW = ∞.

    HACKATHON IMPLEMENTATION ASSUMPTION: linear (constant-λ) extrapolation.

    Parameters
    ----------
    t_now : float
        Current time (hours).  Not used in the calculation but accepted
        for API consistency.
    R_now : float
        Current recoverability R(t) ∈ (0, 1].
    lambda_now : float
        Current instantaneous stress λ(t) (h⁻¹).
    R_pnr : float
        PNR threshold.
    diw_floor : float
        Minimum DIW (default 0 h).

    Returns
    -------
    float
        Estimated DIW (hours) ≥ diw_floor.
        ``math.inf`` if λ_now is effectively zero and R_now > R_pnr.
    """
    if R_now <= R_pnr:
        return diw_floor  # already past PNR

    if lambda_now <= 1e-12:
        return math.inf  # no stress → never reaches PNR

    diw = math.log(R_now / R_pnr) / lambda_now
    return max(diw, diw_floor)


def cost_of_delay(
    R_now: float,
    lambda_now: float,
    delta_t: float,
    *,
    batch_value: float = STRESS_PARAMS["batch_value"],
    intervention_cost: float = STRESS_PARAMS["intervention_cost"],
) -> float:
    """Estimate the net economic cost of delaying intervention by Δt hours.

    Formula (HACKATHON IMPLEMENTATION ASSUMPTION):

        R_future = R_now · exp(−λ_now · Δt)
        Cost-of-Delay = batch_value · (1 − R_future) − intervention_cost

    A positive result means delay is net-costly (acting now is better).
    A negative result means delay is still acceptable (uncertainty > cost).

    Parameters
    ----------
    R_now : float
        Current recoverability R(t) ∈ (0, 1].
    lambda_now : float
        Current instantaneous stress rate λ(t) (h⁻¹).
    delta_t : float
        Proposed delay duration (hours).
    batch_value : float
        Expected revenue of a successful batch (USD).
        HACKATHON IMPLEMENTATION ASSUMPTION.
    intervention_cost : float
        Cost of a corrective operator action (USD).
        HACKATHON IMPLEMENTATION ASSUMPTION.

    Returns
    -------
    float
        Net cost-of-delay in USD (positive → act now, negative → can wait).
    """
    if delta_t < 0:
        raise ValueError(f"delta_t must be non-negative; got {delta_t}")

    R_future = R_now * math.exp(-lambda_now * delta_t)
    R_future = max(R_future, 0.0)

    # Loss due to reduced recoverability if we wait
    cost = batch_value * (1.0 - R_future) - intervention_cost
    return cost


# ---------------------------------------------------------------------------
# Convenience: process a full simulation DataFrame
# ---------------------------------------------------------------------------


def enrich_dataframe(
    df: pd.DataFrame,
    *,
    pH_col: Optional[str] = None,
    stress_params: Optional[dict] = None,
) -> pd.DataFrame:
    """Add stress and recoverability columns to a simulate_batch() DataFrame.

    Adds columns: ``lambda``, ``R``, ``pnr_crossed``, ``diw``.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of ``src.simulator.simulate_batch()``.
    pH_col : str, optional
        Column name for pH values.  If None, pH stress is set to zero.
    stress_params : dict, optional
        Override any key in STRESS_PARAMS.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with four additional columns.
    """
    sp = dict(STRESS_PARAMS)
    if stress_params:
        sp.update(stress_params)

    t = df["timestamp"].to_numpy()
    CL = df["DO"].to_numpy()
    pH = df[pH_col].to_numpy() if pH_col and pH_col in df.columns else None

    lam = compute_stress(
        CL, pH,
        k_stress=sp["k_stress"],
        k_O2=sp["k_O2"],
        pH_opt=sp["pH_opt"],
        C_crit=sp["C_crit"],
    )
    R = compute_recoverability(t, lam)

    _, pnr_crossed = estimate_pnr(t, R, R_pnr=sp["R_pnr"])

    # DIW at every timestep (forward-looking from current point)
    diw_arr = np.array([
        estimate_diw(t[i], R[i], lam[i], R_pnr=sp["R_pnr"], diw_floor=sp["R_diw_floor"])
        for i in range(len(t))
    ])
    # Cap inf at a large sentinel for DataFrame storage
    diw_arr = np.where(np.isinf(diw_arr), 9999.0, diw_arr)

    out = df.copy()
    out["lambda"] = lam
    out["R"]      = R
    out["pnr_crossed"] = pnr_crossed
    out["diw"]    = diw_arr
    return out
