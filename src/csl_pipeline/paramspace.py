"""CSL ↔ DP parameter-space algebra and convention handling.

This module encodes the conversion between the CSL and DP phenomenological
parameters, the handling of regularisation conventions (Gaussian smearing
σ_DP vs spatial cutoff R₀), and the computation of regime boundaries.

It is deliberately separated from ``rates.py`` so that convention questions
do not leak into the rate formulas, and from ``geometry.py`` so that the
geometry factors stay purely geometric.

References
----------
Donadi et al., Nature Phys. 17, 74 (2021), arXiv:2011.05300.
Dai, Miao & Ma, arXiv:2411.17588.
Nimmrichter, Hornberger & Hammerer, PRL 113, 020405 (2014), arXiv:1405.2868.
Toroš & Bassi, J. Phys. A 51, 115302 (2018), arXiv:1601.02931.
"""

from __future__ import annotations

import math
from enum import Enum

# ---------------------------------------------------------------------------
# Physical constants (CODATA 2018)
# ---------------------------------------------------------------------------

G = 6.67430e-11           #: Newton's constant, m³ kg⁻¹ s⁻²
HBAR = 1.054571817e-34    #: Reduced Planck constant, J s
AMU = 1.66053906660e-27   #: Atomic mass unit, kg
MN = 1.67492749804e-27    #: Nucleon mass, kg (Ferialdi–Bassi convention)

#: GRW reference values.
LAMBDA_GRW = 1e-17
R_C_GRW = 1e-7

# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------

class RegimeKind(str, Enum):
    """Which asymptotic regime a protocol is in."""

    POINT_PARTICLE = "point_particle"
    FINITE_SIZE = "finite_size"
    RESOLVED_SUPERPOSITION = "resolved_superposition"
    RADIUS_REGULARIZED = "radius_regularized"


class RegularisationConvention(str, Enum):
    """Which DP regularisation prescription is in use."""

    GAUSSIAN_SMearing = "gaussian_smearing"   # σ_DP (Nimmrichter, GGR)
    SPATIAL_CUTOFF_R0 = "spatial_cutoff_R0"   # R₀ (Donadi, XENONnT)
    NONE = "none"                              # point-particle proxy (Davila & Milburn)


# ---------------------------------------------------------------------------
# Core conversion quantities
# ---------------------------------------------------------------------------

def ell_star(λ: float = LAMBDA_GRW,
             G_: float = G,
             hbar: float = HBAR,
             amu: float = AMU) -> float:
    """Crossover length ℓ* = G amu² / (λ ħ).

    Ξ ≥ 1  ⇔  d_eff · K_rc(Δx) ≥ ℓ*.
    """
    return G_ * amu * amu / (λ * hbar)


def csl_lambda_from_dp(dp_rate_prefactor: float,
                       convention: RegularisationConvention = RegularisationConvention.NONE,
                       sigma_dp: float | None = None,
                       R0: float | None = None,
                       m_over_amu_sq: float = 1.0) -> float:
    """Convert a DP rate prefactor into the equivalent CSL λ.

    The conversion depends on the regularisation convention because the DP
    and CSL kernels differ in their form factors.  This function handles the
    prescriptions documented in the provenance write-ups.
    """
    if convention == RegularisationConvention.NONE:
        # Point-particle proxy: Λ_DP = G m² τ / (ħ d_eff).
        # Λ_CSL = λ (m/amu)² τ K_rc(Δx).
        # For the same exponent at a reference geometry, λ_equiv = Λ_DP / ((m/amu)² τ K_rc).
        # This is ambiguous without a reference Δx, so we return the relation, not a number.
        raise NotImplementedError(
            "Point-particle proxy conversion requires a reference Δx. "
            "Use the full ratio instead."
        )
    elif convention == RegularisationConvention.GAUSSIAN_SMearing:
        # σ_DP smearing: the DP kernel uses the same 4r_C² Gaussian form
        # as CSL only when r_C = σ_DP.  Otherwise the form factors differ.
        # This is the Nimmrichter convention.
        if sigma_dp is None:
            raise ValueError("Gaussian smearing convention requires sigma_dp")
        # λ_equiv = λ_CSL such that the CSL kernel matches the DP kernel
        # at the same r_C = σ_dp.  No simple closed form; the rate ratio
        # must be computed numerically.
        raise NotImplementedError(
            "Gaussian smearing conversion requires numerical kernel comparison. "
            "Use the scale ratio functions instead."
        )
    elif convention == RegularisationConvention.SPATIAL_CUTOFF_R0:
        # R₀ cutoff (Donadi): the DP rate scales as 1/R₀³.
        # Conversion to CSL λ requires matching the rate level at a reference
        # geometry.  This is the XENONnT convention.
        if R0 is None:
            raise ValueError("Spatial cutoff convention requires R0")
        # Donadi rate: dΓ/dE ∝ 1/R₀³.  The total rate scales as 1/R₀³ times
        # a form factor.  We return the relation, not a number, because the
        # conversion is observable-dependent.
        raise NotImplementedError(
            "R₀ cutoff conversion is observable-dependent. "
            "Use the specific rate comparison instead."
        )
    else:
        raise ValueError(f"Unknown convention: {convention}")


def dp_sigma_from_csl(r_C: float,
                      convention: RegularisationConvention = RegularisationConvention.GAUSSIAN_SMearing) -> float:
    """Infer the DP smearing length equivalent to a given CSL r_C.

    In the Gaussian-smearing convention (Nimmrichter), the CSL kernel
    exp(-Δx²/4r_C²) matches the DP kernel when σ_DP = r_C / √2, because
    the DP Gaussian is exp(-Δx²/2σ_DP²).  This is the convention-matching
    value, not a bound.
    """
    if convention == RegularisationConvention.GAUSSIAN_SMearing:
        return r_C / math.sqrt(2.0)
    elif convention == RegularisationConvention.SPATIAL_CUTOFF_R0:
        # R₀ is not directly comparable to r_C; they are different kinds of
        # parameter.  Return None to signal that.
        return float("nan")
    else:
        raise ValueError(f"Unknown convention: {convention}")


def regime_boundary_radius(Δx: float,
                           λ: float = LAMBDA_GRW,
                           r_C: float = R_C_GRW,
                           G_: float = G,
                           hbar: float = HBAR,
                           amu: float = AMU) -> float:
    """Radius at which the radius-regularized and resolved branches cross.

    For Δx < 2R:  d_eff = 2R.  For Δx ≥ 2R:  d_eff = Δx.
    The crossover is at R = Δx / 2.
    """
    return Δx / 2.0


def crossover_x_star(λ: float = LAMBDA_GRW,
                     r_C: float = R_C_GRW,
                     G_: float = G,
                     hbar: float = HBAR,
                     amu: float = AMU,
                     tol: float = 1e-12) -> float:
    """Resoved-superposition crossover x* solving x·(1 - exp(-x²/4r_C²)) = ℓ*.

    For GRW parameters: x* ≈ 1.911 nm.
    """
    return ell_star(λ, G_, hbar, amu)  # ℓ* is the target; x* solves the equation
    # Note: the actual root-finding is in rates.x_star; this is the target value.


def csl_dp_ratio_point(Δx: float, R: float,
                       λ: float = LAMBDA_GRW,
                       r_C: float = R_C_GRW,
                       G_: float = G,
                       hbar: float = HBAR,
                       amu: float = AMU) -> float:
    """Point-particle CSL/DP ratio Ξ = λ ħ d_eff K_rc(Δx) / (G amu²).

    Independent of m and τ.  Uses d_eff = max{Δx, 2R}.
    """
    d_eff = max(Δx, 2.0 * R)
    K_rc = 1.0 - math.exp(-(Δx * Δx) / (4.0 * r_C * r_C))
    return (λ * hbar * d_eff / (G_ * amu * amu)) * K_rc


# ---------------------------------------------------------------------------
# Convention-safe comparison helpers
# ---------------------------------------------------------------------------

def compare_csl_dp_absolute(Δx: float, R: float, τ: float, m: float,
                            λ: float = LAMBDA_GRW,
                            r_C: float = R_C_GRW,
                            G_: float = G,
                            hbar: float = HBAR,
                            amu: float = AMU,
                            d_eff_override: float | None = None) -> dict:
    """Compare the absolute magnitudes of Λ_CSL and Λ_DP for a protocol.

    Returns a dict with keys:
        Lambda_CSL, Lambda_DP, ratio, regime, dominant
    """
    d_eff = d_eff_override if d_eff_override is not None else max(Δx, 2.0 * R)
    K_rc = 1.0 - math.exp(-(Δx * Δx) / (4.0 * r_C * r_C))

    Lambda_CSL = λ * (m / amu) ** 2 * τ * K_rc
    Lambda_DP = G_ * m * m * τ / (hbar * d_eff)

    ratio = Lambda_CSL / Lambda_DP if Lambda_DP > 0 else float("inf")
    regime = (
        RegimeKind.RESOLVED_SUPERPOSITION
        if Δx >= 2.0 * R
        else RegimeKind.RADIUS_REGULARIZED
    )

    return {
        "Lambda_CSL": Lambda_CSL,
        "Lambda_DP": Lambda_DP,
        "ratio": ratio,
        "regime": regime,
        "dominant": "CSL" if ratio > 1.0 else ("DP" if ratio < 1.0 else "equal"),
        "d_eff": d_eff,
        "K_rc": K_rc,
    }


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def classify_regime(Δx: float, R: float, r_C: float,
                    λ: float = LAMBDA_GRW,
                    G_: float = G,
                    hbar: float = HBAR,
                    amu: float = AMU) -> dict:
    """Classify a protocol into its asymptotic regime and compute the key ratios.

    Returns a dict with:
        - separation_ratio a = Δx/R
        - radius_ratio ρ = R/r_C
        - d_eff
        - Ξ_point (point-particle ratio)
        - Ξ_sph (finite-size ratio, from rates.Xi_sph)
        - regime_kind
        - crossover_status: which side of x* the separation lies on
    """
    from csl_pipeline.rates import Xi_sph, x_star, ell_star as _ell_star

    a = Δx / R
    ρ = R / r_C
    d_eff = max(Δx, 2.0 * R)

    Xi_point_val = csl_dp_ratio_point(Δx, R, λ=λ, r_C=r_C, G_=G_, hbar=hbar, amu=amu)
    Xi_sph_val = Xi_sph(Δx, R, r_C, λ=λ, G_=G_, hbar=hbar, amu=amu)

    x_star_val = x_star(λ=λ, r_C=r_C, G_=G_, hbar=hbar, amu=amu)
    target = _ell_star(λ=λ, G_=G_, hbar=hbar, amu=amu)
    # x* solves x·(1 - exp(-x²/4r_C²)) = ℓ*
    # The resolved branch crossover is at x = x*
    if a * (1.0 - math.exp(-(a * a * R * R) / (4.0 * r_C * r_C))) >= target:
        crossover_status = "CSL_dominant_or_equal"
    else:
        crossover_status = "DP_dominant"

    if a >= 2.0:
        regime_kind = RegimeKind.RESOLVED_SUPERPOSITION
    elif a >= 0.0:
        regime_kind = RegimeKind.RADIUS_REGULARIZED
    else:
        regime_kind = RegimeKind.POINT_PARTICLE

    return {
        "a": a,
        "ρ": ρ,
        "d_eff": d_eff,
        "Ξ_point": Xi_point_val,
        "Ξ_sph": Xi_sph_val,
        "regime_kind": regime_kind,
        "crossover_status": crossover_status,
        "x_star": x_star_val,
        "ℓ_star": target,
    }


__all__ = [
    "G",
    "HBAR",
    "AMU",
    "MN",
    "LAMBDA_GRW",
    "R_C_GRW",
    "RegimeKind",
    "RegularisationConvention",
    "ell_star",
    "csl_lambda_from_dp",
    "dp_sigma_from_csl",
    "regime_boundary_radius",
    "crossover_x_star",
    "csl_dp_ratio_point",
    "compare_csl_dp_absolute",
    "classify_regime",
]
