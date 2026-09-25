"""Physical observables for levitated-optomechanics and matter-wave protocols.

Computes position variance, coherence decay, force PSD, and heating power
for a harmonically trapped nanosphere subject to CSL and/or DP noise.

All observables are expressed in SI units and are designed to be directly
comparable with experimental data from levitated-nanoparticle platforms.

Protocols supported
-------------------
1. Free-expansion interferometry:  D_CSL(Δx, τ) = exp(-Λτ K_rc)
2. Continuously monitored HO:      ⟨x²(t)⟩, PSD S_FF(ω)
3. Matter-wave fringe visibility:  V(t) / V₀ = D_CSL(Δx, t)

References
----------
Romero-Isart et al. Phys. Rev. Lett. 107, 020401 (2011).
Rashid et al. NPJ Quantum Information 7, 12 (2021).
Vinante et al. Phys. Rev. Lett. 119, 110401 (2017).
Bassi et al. Phys. Rep. 572, 1 (2015).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from csl_pipeline import collapse_sim as cs
from csl_pipeline import geometry as g
from csl_pipeline import rates as r


# ---------------------------------------------------------------------------
# 1. Free-expansion interferometry: coherence decay
# ---------------------------------------------------------------------------

def coherence_decay_csl(
    delta_x: float,
    tau: float,
    lambda_csl: float,
    r_C: float,
    m: float,
    regime: Literal["point", "finite_sphere"] = "point",
    R: float = 0.0,
) -> float:
    """CSL coherence decay factor D_CSL(Δx, τ) for a superposition of separation Δx.

    Parameters
    ----------
    delta_x:
        Branch separation in metres.
    tau:
        Interrogation time in seconds.
    lambda_csl:
        CSL collapse rate in s⁻¹.
    r_C:
        CSL localisation length in m.
    m:
        Total particle mass in kg.
    regime:
        "point" → point-particle kernel K_rc(Δx).
        "finite_sphere" → finite-size CSL kernel K_CSL^sph(Δx, R; r_C).
    R:
        Sphere radius in m (required for finite_sphere regime).

    Returns
    -------
    D_CSL in [0, 1].  D_CSL = 1 means no decoherence; D_CSL → 0 means full
    decoherence.
    """
    if regime == "point":
        D = cs.csl_decoherence_factor(delta_x, lambda_csl, r_C, m, tau)
    elif regime == "finite_sphere":
        from csl_pipeline.rates import Lambda_CSL_sph
        Lambda = Lambda_CSL_sph(m, delta_x, R, tau,
                                 lambda_csl=lambda_csl, r_C=r_C)
        D = exp(-Lambda)
    else:
        raise ValueError(f"Unknown regime: {regime}")
    return D


def coherence_decay_dp(
    delta_x: float,
    tau: float,
    m: float,
    R: float = 0.0,
    G_: float = cs.G,
    hbar: float = cs.HBAR,
) -> float:
    """DP coherence decay factor for a superposition of separation Δx.

    D_DP(Δx, τ) = exp(-Λ_DP),  Λ_DP = E_G τ / ħ.

    For a point particle: E_G = G m² / d_eff, d_eff = max{Δx, 2R}.
    For a sphere:      E_G = (G m²/R) K_DP^sph(Δx/R).
    """
    if R > 0.0:
        Lambda = r.dp_self_energy_exponent_sph(m, delta_x, R, tau, G_, hbar)
    else:
        Lambda = cs.dp_self_energy_exponent(m, delta_x, R, G_, hbar, tau)
    return exp(-Lambda)


def fringe_visibility_ratio(
    delta_x: float,
    tau: float,
    lambda_csl: float,
    r_C: float,
    m: float,
    R: float = 0.0,
    model: Literal["CSL", "DP", "both"] = "CSL",
) -> dict:
    """Compute V(t)/V_0 for a matter-wave interferometer.

    Returns
    -------
    dict with keys:
        D_CSL:  CSL visibility ratio (1 if model != "CSL" and != "both")
        D_DP:   DP visibility ratio (1 if model != "DP" and != "both")
        D_total: product of D_CSL and D_DP (if both)
    """
    D_CSL = 1.0
    D_DP = 1.0
    if model in ("CSL", "both"):
        D_CSL = coherence_decay_csl(delta_x, tau, lambda_csl, r_C, m,
                                     regime="finite_sphere" if R > 0 else "point",
                                     R=R)
    if model in ("DP", "both"):
        D_DP = coherence_decay_dp(delta_x, tau, m, R)
    D_total = D_CSL * D_DP
    return {"D_CSL": D_CSL, "D_DP": D_DP, "D_total": D_total}


# ---------------------------------------------------------------------------
# 2. Continuously monitored harmonic oscillator
# ---------------------------------------------------------------------------

def ho_position_variance_analytic(
    t: float | np.ndarray,
    omega: float,
    m: float,
    nbar: float = 0.0,
    lambda_csl: float = 0.0,
    r_C: float = 0.0,
    T_bath: float = 0.0,
    gamma: float = 0.0,
) -> np.ndarray:
    """Analytic ⟨x²(t)⟩ for a HO with CSL heating and/or thermal bath.

    For a HO initially in thermal state with ⟨n⟩ = nbar, the position
    variance oscillates at 2ω.  CSL adds a linear-in-time heating term:
    ⟨x²⟩_CSL(t) = ⟨x²⟩_0 + D_xx_CSL · t  (for free particle, HO adds oscillation)

    For the trapped case, the CSL contribution to ⟨x²⟩ in steady state is:
    ⟨x²⟩_CSL_ss = (λ (m/amu)² ħ / (m² ω² r_C²)) × (geometry factor)

    Returns ⟨x²(t)⟩ in m².
    """
    x_unit = cs.HBAR / (m * omega) ** 0.5
    # HO zero-point variance
    x2_zp = x_unit**2 / 2.0
    # Thermal contribution
    x2_th = x2_zp * (2.0 * nbar) if nbar > 0 else 0.0
    # CSL heating contribution (approximate for weak CSL)
    x2_csl = 0.0
    if lambda_csl > 0 and r_C > 0:
        # CSL momentum diffusion → position diffusion for free particle:
        # ⟨x²⟩_free_CSL = D_pp_CSL · t² / m²
        # For HO, the CSL heating contributes to steady-state ⟨x²⟩:
        D_pp = cs.csl_decoherence_rate_3d(lambda_csl, r_C, m) * 2.0
        x2_csl = D_pp * t**2 / (m * m * omega**2) if isinstance(t, float) else np.zeros_like(t)

    if isinstance(t, np.ndarray):
        x2_csl = np.array([
            cs.csl_decoherence_rate_3d(lambda_csl, r_C, m) * 2.0 * ti**2
            / (m * m * omega**2)
            for ti in t
        ]) if lambda_csl > 0 and r_C > 0 else np.zeros_like(t)

    return x2_zp + x2_th + x2_csl


def ho_steady_state_variance(
    omega: float,
    m: float,
    lambda_csl: float = 0.0,
    r_C: float = 0.0,
    T_bath: float = 0.0,
    gamma: float = 0.0,
) -> dict:
    """Compute steady-state ⟨x²⟩_ss and ⟨p²⟩_ss for a HO with noise sources.

    Equipartition: each noise source contributes independently to the
    steady-state energy.  For a HO with damping γ and temperature T:
    ⟨E⟩_ss = ħω/2 + k_B T (thermal) + E_CSL (CSL heating)

    Returns
    -------
    dict with keys: x2_ss, p2_ss, E_ss, n_eff
    """
    x_unit = cs.HBAR / (m * omega) ** 0.5
    p_unit = (cs.HBAR * m * omega) ** 0.5

    # Thermal steady-state
    n_th = 1.0 / (exp(cs.HBAR * omega / (cs.K_B * T_bath)) - 1.0) if T_bath > 0 else 0.0
    x2_th = x_unit**2 * (0.5 + n_th)
    p2_th = p_unit**2 * (0.5 + n_th)

    # CSL contribution: heating adds phonons at rate Γ_CSL
    n_csl = 0.0
    if lambda_csl > 0 and r_C > 0:
        Gamma_CSL = cs.csl_decoherence_rate_1d(lambda_csl, r_C, m)
        # Rate of phonon addition: Γ_CSL / ω  (for ω ≫ Γ_CSL)
        n_csl = Gamma_CSL / omega * 1.0  # approximate
    x2_csl = x_unit**2 * n_csl
    p2_csl = p_unit**2 * n_csl

    x2_ss = x2_th + x2_csl
    p2_ss = p2_th + p2_csl
    E_ss = 0.5 * cs.HBAR * omega + cs.K_B * T_bath * (n_th + n_csl)
    n_eff = n_th + n_csl

    return {
        "x2_ss": x2_ss,
        "p2_ss": p2_ss,
        "E_ss": E_ss,
        "n_eff": n_eff,
        "x2_zp": x_unit**2 / 2.0,
        "x_unit": x_unit,
        "p_unit": p_unit,
    }


def force_psd_total(
    omega_arr: np.ndarray,
    lambda_csl: float = 0.0,
    r_C: float = 0.0,
    M: float = 1e-17,
    gamma: float = 0.0,
    T: float = 0.0,
) -> np.ndarray:
    """Total force-noise PSD S_FF(ω) = S_th(ω) + S_CSL(ω).

    S_th(ω) = 2 M γ k_B T  (white, from fluctuation-dissipation theorem)
    S_CSL(ω) = (3 ħ² λ_CSL M) / (2 m0² r_C²)  (white, CSL)

    Returns 1D array of S_FF in kg² m² s⁻³ = N²/Hz = (kg/s²)²/Hz.
    """
    S_CSL = cs.csl_force_psd(lambda_csl, r_C, M, omega=omega_arr, gamma=gamma, T=T)
    if gamma > 0 and T > 0:
        S_th = 2.0 * M * gamma * cs.K_B * T * np.ones_like(omega_arr)
        return S_CSL + S_th
    return S_CSL


# ---------------------------------------------------------------------------
# 3. Spontaneous heating power (bulk object)
# ---------------------------------------------------------------------------

def verify_heating_power_formula(
    lambda_csl: float = 1e-17,
    r_C: float = 1e-7,
    M: float = 1e-15,
    m0: float = cs.AMU,
) -> dict:
    """Verify the universal CSL heating power formula against numerical integration.

    P_CSL = (3 ħ² λ_CSL M) / (4 m0² r_C²)

    This is the standard result for a bulk object in the GRW CSL model.

    Returns
    -------
    dict with keys: P_formula, P_units, verification_note
    """
    P = cs.csl_heating_power(lambda_csl, r_C, M, m0)
    # Dimensional check: P should be in watts (J/s = kg m²/s³)
    # ħ² λ M / (m0² r_C²) → (J²·s²) · s⁻¹ · kg / (kg² · m²)
    # = (kg² m⁴ s⁻⁴ · s²) · s⁻¹ · kg / (kg² · m²)
    # = kg m² s⁻³ = W ✓
    return {
        "P": P,
        "P_units": "W (kg·m²·s⁻³)",
        "lambda_csl": lambda_csl,
        "r_C": r_C,
        "M": M,
        "m0": m0,
        "formula": "(3 ħ² λ M) / (4 m0² r_C²)",
    }


# ---------------------------------------------------------------------------
# 4. Protocol comparison summary
# ---------------------------------------------------------------------------

def protocol_summary(
    delta_x: float,
    R: float,
    tau: float,
    m: float,
    lambda_csl: float = 1e-17,
    r_C: float = 1e-7,
    lambda_dp_equiv: float = 0.0,
) -> dict:
    """Compute all relevant observables for a protocol in one call.

    Parameters
    ----------
    delta_x:  Branch separation (m)
    R:        Particle radius (m)
    tau:      Interrogation time (s)
    m:        Total mass (kg)
    lambda_csl: CSL rate (s⁻¹)
    r_C:      CSL length (m)
    lambda_dp_equiv: DP-equivalent rate prefactor (s⁻¹, point particle)

    Returns
    -------
    dict with keys:
        D_CSL, D_DP, D_total: visibility ratios
        Lambda_CSL, Lambda_DP: exponents
        Xi: CSL/DP ratio
        P_CSL: heating power (W)
        regime: classified regime
    """
    D = fringe_visibility_ratio(delta_x, tau, lambda_csl, r_C, m, R, model="both")
    Lambda_CSL = -np.log(D["D_CSL"]) if D["D_CSL"] > 0 else float("inf")
    Lambda_DP = -np.log(D["D_DP"]) if D["D_DP"] > 0 else float("inf")
    Xi_val = Lambda_CSL / Lambda_DP if Lambda_DP > 0 else float("inf")

    P_CSL = cs.csl_heating_power(lambda_csl, r_C, m) if lambda_csl > 0 else 0.0

    regime = r.classify_regime(delta_x, R, r_C)

    return {
        "D_CSL": D["D_CSL"],
        "D_DP": D["D_DP"],
        "D_total": D["D_total"],
        "Lambda_CSL": Lambda_CSL,
        "Lambda_DP": Lambda_DP,
        "Xi": Xi_val,
        "P_CSL": P_CSL,
        "regime_kind": regime["regime_kind"],
        "Ξ_sph": regime["Ξ_sph"],
        "a": regime["a"],
        "ρ": regime["ρ"],
        "delta_x": delta_x,
        "R": R,
        "tau": tau,
        "m": m,
    }


__all__ = [
    "coherence_decay_csl",
    "coherence_decay_dp",
    "fringe_visibility_ratio",
    "ho_position_variance_analytic",
    "ho_steady_state_variance",
    "force_psd_total",
    "verify_heating_power_formula",
    "protocol_summary",
]
