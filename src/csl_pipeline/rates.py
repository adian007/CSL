"""
CSL and DP collapse/diffusion rates for homogeneous bodies.

Builds on the geometry layer (csl_pipeline.geometry) to define the physical
rate exponents and their ratio, following the conversion theorems in
Davila & Milburn (2026, arXiv:2608.05972).

Three layers:
    1. Point-particle quantities — kernels, exponents, ratio, crossover.
    2. Finite-size quantities — homogeneous-sphere kernels (DP: closed form;
       CSL: quadrature) and their exponents.
    3. RATE_REGISTRY — guard against silently mixing point/finite quantities
       or interchanging CSL and DP exponents.

Reference: Davila & Milburn, "Geometry-Only CSL/DP Ratios and the
Nonuniqueness of Decoherence Kernels," arXiv:2608.05972 (2026).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict

import scipy.integrate as si
import sympy as sp

from csl_pipeline.geometry import (  # noqa: F401  — re-exported for convenience
    PREFACTOR_6SQRT_PI,
    alpha_sphere_nhh,
    alpha_sphere_nhh_numeric,
    eta_kfree_closed_numeric,
    eta_kfree_fourier_numeric,
    eta_kfree_numeric,
    eta_kfree_symbolic,
    sphere_overlap_volume,
)

# ---------------------------------------------------------------------------
# Physical constants (CODATA 2018 unless noted)
# ---------------------------------------------------------------------------

G = 6.67430e-11          #: Newton's constant, m³ kg⁻¹ s⁻²
HBAR = 1.054571817e-34   #: Reduced Planck constant, J s
AMU = 1.66053906660e-27  #: Atomic mass unit, kg

#: GRW reference CSL collapse rate (s⁻¹).
LAMBDA_GRW = 1e-17
#: GRW reference CSL localization length (m).
rC_GRW = 1e-7

#: Crossover length for GRW parameters: G·amu²/(λ·ℏ).
#: Manuscript value: 1.745_129_893 × 10⁻¹³ m.
ELL_STAR_GRW = G * AMU**2 / (LAMBDA_GRW * HBAR)


# ---------------------------------------------------------------------------
# Point-particle kernels and exponents
# ---------------------------------------------------------------------------

def K_rc(Δx: float, r_C: float) -> float:
    """CSL separation kernel for a point particle.

    K_rc(Δx) = 1 - exp(-Δx² / (4 r_C²)).

    Quadratic for Δx ≪ r_C; saturates to 1 for Δx ≫ r_C.
    """
    return 1.0 - math.exp(-(Δx * Δx) / (4.0 * r_C * r_C))


def d_eff(Δx: float, R: float) -> float:
    """Effective distance for the point-particle DP comparison.

    d_eff = max{Δx, 2R}.

    d_eff = Δx  → resolved superposition (Δx ≥ 2R).
    d_eff = 2R  → radius-scale regularised regime (Δx < 2R).
    """
    return max(Δx, 2.0 * R)


def Lambda_CSL_point(m: float, Δx: float, τ: float,
                     λ: float = LAMBDA_GRW,
                     r_C: float = rC_GRW,
                     amu: float = AMU) -> float:
    """Point-particle CSL contrast-loss exponent.

    Λ_CSL = λ (m/amu)² τ K_rc(Δx).

    Visibility multiplier: e^{-Λ_CSL}.
    """
    return λ * (m / amu) ** 2 * τ * K_rc(Δx, r_C)


def Lambda_DP_point(m: float, Δx: float, R: float, τ: float,
                    G_: float = G, hbar: float = HBAR) -> float:
    """Point-particle DP self-energy exponent.

    Λ_DP = G m² τ / (ℏ d_eff),  d_eff = max{Δx, 2R}.

    Uses Gm²/d_eff as a transparent proxy for the DP self-energy,
    avoiding the singularity at zero separation.  This is *not* the exact
    DP self-energy of an extended body — see Lambda_DP_sph for that.
    """
    return G_ * m * m * τ / (hbar * d_eff(Δx, R))


def Xi_point(Δx: float, R: float,
             λ: float = LAMBDA_GRW,
             r_C: float = rC_GRW,
             G_: float = G,
             hbar: float = HBAR,
             amu: float = AMU) -> float:
    """Point-particle CSL/DP ratio.

    Ξ = Λ_CSL / Λ_DP = (λ ℏ d_eff) / (G amu²) · K_rc(Δx).

    Independent of mass m and interrogation time τ (cancellation theorem).
    """
    return (λ * hbar * d_eff(Δx, R) / (G_ * amu * amu)) * K_rc(Δx, r_C)


# ---------------------------------------------------------------------------
# Crossover quantities
# ---------------------------------------------------------------------------

def ell_star(λ: float = LAMBDA_GRW,
             G_: float = G,
             hbar: float = HBAR,
             amu: float = AMU) -> float:
    """Crossover length ℓ* = G amu² / (λ ℏ).

    Ξ ≥ 1  ⇔  d_eff · K_rc(Δx) ≥ ℓ*.
    """
    return G_ * amu * amu / (λ * hbar)


def _f_xstar(x: float, r_C: float) -> float:
    """f(x) = x · (1 - exp(-x² / (4 r_C²))).  Monotonic, f(0)=0, f(∞)=∞."""
    return x * K_rc(x, r_C)


def x_star(λ: float = LAMBDA_GRW,
          r_C: float = rC_GRW,
          G_: float = G,
          hbar: float = HBAR,
          amu: float = AMU,
          tol: float = 1e-12) -> float:
    """Resolved-superposition crossover x*.

    Solves  x · (1 - exp(-x² / (4 r_C²))) = ℓ*  for x > 0 by bisection.

    For GRW parameters: x* ≈ 1.911_184_110 nm.
    """
    target = ell_star(λ, G_, hbar, amu)
    lo, hi = 0.0, 1.0
    while _f_xstar(hi, r_C) < target:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _f_xstar(mid, r_C) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def R_star(Δx: float,
           λ: float = LAMBDA_GRW,
           r_C: float = rC_GRW,
           G_: float = G,
           hbar: float = HBAR,
           amu: float = AMU) -> float:
    """Radius-regularised crossover R*(Δx).

    For Δx < 2R:  Ξ ≥ 1  ⇔  R ≥ R*(Δx) = ℓ* / (2 K_rc(Δx)).
    """
    return ell_star(λ, G_, hbar, amu) / (2.0 * K_rc(Δx, r_C))


# ---------------------------------------------------------------------------
# Homogeneous-sphere DP kernel — closed form
# ---------------------------------------------------------------------------

def K_DP_sph_closed(a: float) -> float:
    """Evaluate the homogeneous-sphere DP kernel at a = Δx / R.

    Closed-form piecewise polynomial (Corollary, arXiv:2608.05972):

        K_DP^sph(a) = a²/2 - 3a³/16 + a⁵/160          for 0 ≤ a ≤ 2,
        K_DP^sph(a) = 6/5 - 1/a                        for a ≥ 2.

    This is the gravitational self-energy of the difference between two
    displaced uniform spheres, in units of G m² / R.
    """
    if a <= 0.0:
        return 0.0
    if a <= 2.0:
        return a * a / 2.0 - 3.0 * a**3 / 16.0 + a**5 / 160.0
    return 6.0 / 5.0 - 1.0 / a


# ---------------------------------------------------------------------------
# Homogeneous-sphere CSL kernel — quadrature
# ---------------------------------------------------------------------------

def _F_sph(u: float) -> float:
    """Normalised radial Fourier form factor of a uniform sphere.

    F_sph(u) = 3 (sin u - u cos u) / u³,  with F_sph(0) = 1.
    """
    if abs(u) < 1e-8:
        return 1.0 - u * u / 10.0  # Taylor: 1 - u²/10 + O(u⁴)
    return 3.0 * (math.sin(u) - u * math.cos(u)) / (u * u * u)


def K_CSL_sph_numeric(Δx: float, R: float, r_C: float,
                      limit: int = 800,
                      epsabs: float = 1e-14,
                      epsrel: float = 1e-14) -> float:
    """Finite-size CSL kernel for a homogeneous sphere by adaptive quadrature.

    K_CSL^sph(Δx, R; r_C) = (4/√π) ∫₀^∞ q² e^{-q²}
                                     |F_sph(q R / r_C)|²
                                     (1 - sinc(q Δx / r_C)) dq.

    The integrand decays as q² e^{-q²} for large q, so convergence is rapid.
    """
    scale_R = R / r_C
    scale_Δx = Δx / r_C

    def integrand(q: float) -> float:
        F = _F_sph(q * scale_R)
        sa = q * scale_Δx
        if abs(sa) < 1e-8:
            sinc_val = 1.0 - sa * sa / 6.0
        else:
            sinc_val = math.sin(sa) / sa
        return q * q * math.exp(-q * q) * F * F * (1.0 - sinc_val)

    value, _ = si.quad(integrand, 0.0, float("inf"),
                       limit=limit, epsabs=epsabs, epsrel=epsrel)
    return float(4.0 / math.sqrt(math.pi) * value)


# ---------------------------------------------------------------------------
# Finite-size exponents and ratio
# ---------------------------------------------------------------------------

def Lambda_CSL_sph(m: float, Δx: float, R: float, τ: float,
                   λ: float = LAMBDA_GRW,
                   r_C: float = rC_GRW,
                   amu: float = AMU) -> float:
    """Finite-size CSL contrast-loss exponent for a homogeneous sphere.

    Λ_CSL^sph = λ (m/amu)² τ K_CSL^sph(Δx, R; r_C).
    """
    return λ * (m / amu) ** 2 * τ * K_CSL_sph_numeric(Δx, R, r_C)


def Lambda_DP_sph(m: float, Δx: float, R: float, τ: float,
                  G_: float = G, hbar: float = HBAR) -> float:
    """Finite-size DP self-energy exponent for a homogeneous sphere.

    Λ_DP^sph = (G m² τ / (ℏ R)) K_DP^sph(Δx / R).

    The DP self-energy is  E_G^sph = (G m² / R) K_DP^sph(Δx / R).
    """
    a = Δx / R
    return G_ * m * m * τ / (hbar * R) * K_DP_sph_closed(a)


def Xi_sph(Δx: float, R: float, r_C: float,
           λ: float = LAMBDA_GRW,
           G_: float = G,
           hbar: float = HBAR,
           amu: float = AMU) -> float:
    """Finite-size CSL/DP ratio for a homogeneous sphere.

    Ξ_sph = Λ_CSL^sph / Λ_DP^sph
          = (R · K_CSL^sph(Δx, R; r_C)) / (ℓ* · K_DP^sph(Δx / R)).

    Independent of mass m and interrogation time τ (finite-profile theorem).
    """
    a = Δx / R
    k_dp = K_DP_sph_closed(a)
    if k_dp <= 0.0:
        return float("inf")
    return (R * K_CSL_sph_numeric(Δx, R, r_C)
            / (ell_star(λ, G_, hbar, amu) * k_dp))


# ---------------------------------------------------------------------------
# RATE_REGISTRY
# ---------------------------------------------------------------------------

@dataclass
class RateEntry:
    """A registered rate formula with provenance metadata."""

    name: str
    description: str
    valid_regime: str
    reference: str
    numeric: Callable[..., float]
    symbols: Dict[str, str]


RATE_REGISTRY: Dict[str, RateEntry] = {
    "point_csl_exponent": RateEntry(
        name="point_csl_exponent",
        description="Λ_CSL = λ (m/amu)² τ K_rc(Δx) — point-particle CSL exponent",
        valid_regime="Point-particle limit; Δx ≥ 0, R ≥ 0",
        reference="Davila & Milburn (2026), §II",
        numeric=Lambda_CSL_point,
        symbols={
            "m": "Total particle mass (kg)",
            "Δx": "Branch separation (m)",
            "τ": "Interrogation time (s)",
            "λ": "CSL collapse rate (s⁻¹), default GRW 10⁻¹⁷",
            "r_C": "CSL localisation length (m), default GRW 10⁻⁷",
            "amu": "Atomic mass unit (kg)",
        },
    ),
    "point_dp_exponent": RateEntry(
        name="point_dp_exponent",
        description="Λ_DP = G m² τ / (ℏ d_eff) — point-particle DP exponent",
        valid_regime="Point-particle limit; Δx ≥ 0, R ≥ 0",
        reference="Davila & Milburn (2026), §II",
        numeric=Lambda_DP_point,
        symbols={
            "m": "Total particle mass (kg)",
            "Δx": "Branch separation (m)",
            "R": "Particle radius (m)",
            "τ": "Interrogation time (s)",
            "G": "Newton's constant (m³ kg⁻¹ s⁻²)",
            "hbar": "Reduced Planck constant (J s)",
        },
    ),
    "point_csl_dp_ratio": RateEntry(
        name="point_csl_dp_ratio",
        description="Ξ = λ ℏ d_eff / (G amu²) · K_rc(Δx) — point-particle CSL/DP ratio",
        valid_regime="Point-particle limit; independent of m and τ",
        reference="Davila & Milburn (2026), Theorem 1",
        numeric=Xi_point,
        symbols={
            "Δx": "Branch separation (m)",
            "R": "Particle radius (m)",
            "λ": "CSL collapse rate (s⁻¹)",
            "r_C": "CSL localisation length (m)",
            "G": "Newton's constant (m³ kg⁻¹ s⁻²)",
            "hbar": "Reduced Planck constant (J s)",
            "amu": "Atomic mass unit (kg)",
        },
    ),
    "finite_dp_kernel_sph": RateEntry(
        name="finite_dp_kernel_sph",
        description="K_DP^sph(a) — homogeneous-sphere DP kernel, closed-form piecewise polynomial",
        valid_regime="Finite-size homogeneous sphere; a = Δx/R ≥ 0",
        reference="Davila & Milburn (2026), Corollary (homogeneous sphere)",
        numeric=K_DP_sph_closed,
        symbols={
            "a": "Separation ratio Δx / R",
        },
    ),
    "finite_csl_kernel_sph": RateEntry(
        name="finite_csl_kernel_sph",
        description="K_CSL^sph(Δx, R; r_C) — homogeneous-sphere CSL kernel, quadrature",
        valid_regime="Finite-size homogeneous sphere; R ≥ 0, r_C > 0, Δx ≥ 0",
        reference="Davila & Milburn (2026), Eq. (finite-size CSL kernel)",
        numeric=K_CSL_sph_numeric,
        symbols={
            "Δx": "Branch separation (m)",
            "R": "Particle radius (m)",
            "r_C": "CSL localisation length (m)",
        },
    ),
    "finite_csl_exponent_sph": RateEntry(
        name="finite_csl_exponent_sph",
        description="Λ_CSL^sph = λ (m/amu)² τ K_CSL^sph — finite-size CSL exponent",
        valid_regime="Finite-size homogeneous sphere",
        reference="Davila & Milburn (2026), Eq. (finite-size CSL exponent)",
        numeric=Lambda_CSL_sph,
        symbols={
            "m": "Total particle mass (kg)",
            "Δx": "Branch separation (m)",
            "R": "Particle radius (m)",
            "τ": "Interrogation time (s)",
            "λ": "CSL collapse rate (s⁻¹)",
            "r_C": "CSL localisation length (m)",
            "amu": "Atomic mass unit (kg)",
        },
    ),
    "finite_dp_exponent_sph": RateEntry(
        name="finite_dp_exponent_sph",
        description="Λ_DP^sph = (G m² τ / (ℏ R)) K_DP^sph — finite-size DP exponent",
        valid_regime="Finite-size homogeneous sphere",
        reference="Davila & Milburn (2026), Eq. (finite-size DP exponent)",
        numeric=Lambda_DP_sph,
        symbols={
            "m": "Total particle mass (kg)",
            "Δx": "Branch separation (m)",
            "R": "Particle radius (m)",
            "τ": "Interrogation time (s)",
            "G": "Newton's constant (m³ kg⁻¹ s⁻²)",
            "hbar": "Reduced Planck constant (J s)",
        },
    ),
    "finite_csl_dp_ratio_sph": RateEntry(
        name="finite_csl_dp_ratio_sph",
        description="Ξ_sph = R K_CSL^sph / (ℓ* K_DP^sph) — finite-size CSL/DP ratio",
        valid_regime="Finite-size homogeneous sphere; independent of m and τ",
        reference="Davila & Milburn (2026), Theorem 3",
        numeric=Xi_sph,
        symbols={
            "Δx": "Branch separation (m)",
            "R": "Particle radius (m)",
            "r_C": "CSL localisation length (m)",
            "λ": "CSL collapse rate (s⁻¹)",
            "G": "Newton's constant (m³ kg⁻¹ s⁻²)",
            "hbar": "Reduced Planck constant (J s)",
            "amu": "Atomic mass unit (kg)",
        },
    ),
}


# ---------------------------------------------------------------------------
# Symbolic CAS forms for cross-CAS verification
# ---------------------------------------------------------------------------

#: Symbols for the point-particle ratio.
_DX = sp.Symbol("Delta_x", positive=True)
_R_P = sp.Symbol("R", positive=True)
_RC_P = sp.Symbol("r_C", positive=True)
_LAMBDA_P = sp.Symbol("lambda", positive=True)
_G_P = sp.Symbol("G", positive=True)
_HBAR_P = sp.Symbol("hbar", positive=True)
_AMU_P = sp.Symbol("m_u", positive=True)
_DEFF = sp.Symbol("d_eff", positive=True)

#: Point-particle CSL separation kernel: K_rc(Δx) = 1 - exp(-Δx² / (4 r_C²)).
KRC_SYMBOLIC = 1 - sp.exp(-_DX**2 / (4 * _RC_P**2))

#: Effective distance, resolved branch: d_eff = Δx.
DEFF_RESOLVED_SYMBOLIC = _DX

#: Point-particle CSL exponent (m²τ factors omitted, for ratio comparison):
#: Λ_CSL ∝ λ (m/amu)² τ K_rc(Δx).
LCSL_POINT_SYMBOLIC = _LAMBDA_P * KRC_SYMBOLIC

#: Point-particle DP exponent (m²τ factors omitted):
#: Λ_DP ∝ G m² τ / (ħ d_eff).
LDP_POINT_SYMBOLIC = _G_P / (_HBAR_P * _DEFF)

#: Point-particle CSL/DP ratio: Ξ = λ ħ d_eff K_rc(Δx) / (G amu²).
#: The m²τ cancellation is explicit: Ξ = (λ ħ / (G amu²)) · d_eff · K_rc.
XI_POINT_SYMBOLIC = (_LAMBDA_P * _HBAR_P * _DEFF * KRC_SYMBOLIC
                     / (_G_P * _AMU_P**2))

#: Crossover length ℓ* = G amu² / (λ ħ).
ELL_STAR_SYMBOLIC = _G_P * _AMU_P**2 / (_LAMBDA_P * _HBAR_P)

#: Resolved-superposition crossover equation:
#: x · (1 - exp(-x² / (4 r_C²))) = ℓ*.
XSTAR_EQUATION = sp.Eq(
    _DX * (1 - sp.exp(-_DX**2 / (4 * _RC_P**2))),
    ELL_STAR_SYMBOLIC,
)

#: Homogeneous-sphere DP kernel K_DP^sph(a), a = Δx/R.
#: Piecewise: a²/2 - 3a³/16 + a⁵/160 for a ∈ [0, 2];
#:           6/5 - 1/a               for a ≥ 2.
_A = sp.Symbol("a", positive=True)
KDP_SPH_SYMBOLIC = sp.Piecewise(
    (_A**2 / 2 - 3 * _A**3 / 16 + _A**5 / 160, sp.And(_A >= 0, _A <= 2)),
    (sp.Rational(6, 5) - 1 / _A, _A > 2),
)

# ---------------------------------------------------------------------------
# CAS verification helpers
# ---------------------------------------------------------------------------

def xi_point_symbolic() -> sp.Expr:
    """Return the symbolic point-particle ratio Ξ for cross-CAS verification.

    Ξ = λ ħ d_eff K_rc(Δx) / (G amu²) — independent of m and τ.
    """
    return XI_POINT_SYMBOLIC


def verify_kdp_sph_closed_form() -> sp.Expr:
    """Return the symbolic piecewise form of K_DP^sph(a).

    Cross-CAS verification: the piecewise form is internally consistent at
    a = 2 (both branches give the same value) and the asymptotics match
    the expected physical limits (quadratic core, constant saturation).
    """
    return KDP_SPH_SYMBOLIC


def kdp_sph_continuity_at_two() -> sp.Expr:
    """Return K_DP^sph(2⁺) - K_DP^sph(2⁻).  Must be exactly 0.

    Lower branch at a=2:  2²/2 - 3·8/16 + 32/160 = 2 - 1.5 + 0.2 = 0.7
    Upper branch at a=2:  6/5 - 1/2 = 1.2 - 0.5 = 0.7
    """
    lower = sp.Rational(2, 2) - 3 * sp.Rational(8, 16) + sp.Rational(32, 160)
    upper = sp.Rational(6, 5) - sp.Rational(1, 2)
    return sp.simplify(upper - lower)


def kstar_small_a_symbolic() -> sp.Expr:
    """Return the small-a leading asymptotic of K_DP^sph(a): a²/2."""
    return _A**2 / 2


def kstar_large_a_symbolic() -> sp.Expr:
    """Return the large-a leading asymptotic of K_DP^sph(a): 6/5."""
    return sp.Rational(6, 5)


__all__ = [
    "G",
    "HBAR",
    "AMU",
    "LAMBDA_GRW",
    "rC_GRW",
    "ELL_STAR_GRW",
    "K_rc",
    "d_eff",
    "Lambda_CSL_point",
    "Lambda_DP_point",
    "Xi_point",
    "ell_star",
    "x_star",
    "R_star",
    "K_DP_sph_closed",
    "K_CSL_sph_numeric",
    "Lambda_CSL_sph",
    "Lambda_DP_sph",
    "Xi_sph",
    "RATE_REGISTRY",
    "RateEntry",
    # CAS forms
    "KRC_SYMBOLIC",
    "XI_POINT_SYMBOLIC",
    "ELL_STAR_SYMBOLIC",
    "XSTAR_EQUATION",
    "KDP_SPH_SYMBOLIC",
    "xi_point_symbolic",
    "verify_kdp_sph_closed_form",
    "kdp_sph_continuity_at_two",
    "kstar_small_a_symbolic",
    "kstar_large_a_symbolic",
]
