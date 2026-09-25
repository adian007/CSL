"""Dimensionless CSL/DP scale functions S_CSL(a, rC_R) and S_DP(a).

These are the geometry factors that remain after mass, time, and the
overall rate prefactor cancel from the CSL/DP ratio.  The finite-size
conversion theorem (Davila & Milburn 2026, Theorem 3) is expressed in
terms of them.

Notation
--------
a  = Δx / R   separation ratio (dimensionless)
ρ  = R / r_C   radius-to-localisation ratio (dimensionless)

S_CSL(a, ρ) = K_CSL^sph(Δx, R; r_C) / η_kfree(R, r_C)
             — normalised CSL kernel (point-particle limit → K_rc)

S_DP(a)     = K_DP^sph(a)   — normalised DP kernel (point-particle limit → 1/a)
"""

from __future__ import annotations

import math

from csl_pipeline import geometry as g
from csl_pipeline.rates import (  # noqa: F401  — reuse verified kernels
    K_CSL_sph_numeric,
    K_DP_sph_closed,
    ell_star,
    Lambda_CSL_sph,
    Lambda_DP_sph,
    Xi_sph,
)

# ---------------------------------------------------------------------------
# Point-particle reference scale functions
# ---------------------------------------------------------------------------

def K_rc_symbolic(rC_R: float = 1.0) -> str:
    """Return a human-readable description of the point-particle CSL kernel.

    K_rc(Δx) = 1 - exp(-Δx² / (4 r_C²))

    Parameters
    ----------
    rC_R:
        Ratio r_C / R, used only to build the LaTeX string.
    """
    return "1 - \\exp\\!\\left(-\\frac{\\Delta x^2}{4 r_C^2}\\right)"


def S_DP_point(a: float) -> float:
    """Point-particle DP scale: 1 / a  (from d_eff = max{Δx, 2R}).

    For the resolved branch d_eff = Δx:  S_DP_point = R / Δx = 1 / a.
    """
    if a <= 0.0:
        return float("inf")
    return 1.0 / a


# ---------------------------------------------------------------------------
# Finite-size scale functions
# ---------------------------------------------------------------------------

def S_CSL_sph(Δx: float, R: float, r_C: float) -> float:
    """Finite-size CSL scale function S_CSL^sph(Δx, R; r_C) = K_CSL^sph.

    The CSL kernel for a homogeneous sphere, evaluated by adaptive quadrature
    of the Fourier representation from ``rates.K_CSL_sph_numeric``.
    """
    return K_CSL_sph_numeric(Δx, R, r_C)


def S_DP_sph(a: float) -> float:
    """Finite-size DP scale function S_DP^sph(a) = K_DP^sph(a).

    For a ≪ 1:  S_DP^sph(a) ~ a²/2  (quadratic core).
    For a → ∞:  S_DP^sph(a) ~ 6/5  (saturation to a constant).

    Note the contrast with the point-particle 1/a: the finite-size kernel
    saturates rather than decaying, because the spheres separate beyond
    contact and the self-energy plateaus.
    """
    return K_DP_sph_closed(a)


def S_CSL_sph(Δx: float, R: float, r_C: float) -> float:
    """Finite-size CSL scale function S_CSL^sph(Δx, R; r_C) = K_CSL^sph.

    The CSL kernel for a homogeneous sphere, evaluated by adaptive quadrature
    of the Fourier representation from ``rates.K_CSL_sph_numeric``.
    """
    return K_CSL_sph_numeric(Δx, R, r_C)


def S_CSL_point_symbolic(Δx: float, r_C: float) -> float:
    """Point-particle CSL scale (analytical): K_rc(Δx).

    Equal to the point-particle CSL kernel — no geometry factor beyond the
    separation dependence.  Serves as the reference for the finite-size
    correction.
    """
    return 1.0 - math.exp(-(Δx * Δx) / (4.0 * r_C * r_C))


def scale_ratio_sph(Δx: float, R: float, r_C: float) -> float:
    """Dimensionless CSL/DP scale ratio for a homogeneous sphere.

    Ξ_scale = S_CSL^sph(Δx, R; r_C) / S_DP^sph(Δx / R).

    Mass and time have cancelled; only geometry and the collapse parameters
    remain.  This is the core quantity of the finite-size conversion theorem.
    """
    a = Δx / R
    s_csl = S_CSL_sph(Δx, R, r_C)
    s_dp = S_DP_sph(a)
    if s_dp <= 0.0:
        return float("inf")
    return s_csl / s_dp


def full_xi_sph(Δx: float, R: float, r_C: float,
                λ: float = 1e-17,
                G_: float = 6.67430e-11,
                hbar: float = 1.054571817e-34,
                amu: float = 1.66053906660e-27) -> float:
    """Full dimensionless CSL/DP ratio with collapse parameters restored.

    Ξ = (λ ħ / (G amu²)) · ℓ_scale · S_ratio

    where ℓ_scale = R and the remaining factor is the scale ratio above.
    This recovers the exact value from ``rates.Xi_sph``.
    """
    return Xi_sph(Δx, R, r_C, λ=λ, G_=G_, hbar=hbar, amu=amu)


# ---------------------------------------------------------------------------
# Asymptotic forms
# ---------------------------------------------------------------------------

def S_DP_sph_small_a(a: float) -> float:
    """Leading small-a behaviour of S_DP^sph: a²/2."""
    return 0.5 * a * a


def S_DP_sph_large_a(a: float) -> float:
    """Leading large-a behaviour of S_DP^sph: 6/5."""
    return 1.2


def S_CSL_sph_small_a(Δx: float, R: float, r_C: float) -> float:
    """Leading small-a behaviour of S_CSL^sph.

    For a = Δx/R ≪ 1 and R/r_C arbitrary, the CSL kernel approaches the
    point-particle form K_rc(Δx) to leading order.
    """
    return S_CSL_point_symbolic(Δx, r_C)


def S_CSL_sph_small_rho(R: float, r_C: float) -> float:
    """Leading small-R/r_C behaviour of S_CSL^sph for fixed a.

    When the sphere is much smaller than the localisation length
    (R ≪ r_C), the finite-size CSL kernel → K_rc(Δx) because the sphere
    acts as a point particle.
    """
    return 1.0  # K_rc → 1 in the resolved limit; the geometry factor → 1


def S_CSL_sph_large_rho(Δx: float, R: float, r_C: float) -> float:
    """Leading large-R/r_C behaviour of S_CSL^sph for fixed a.

    When R ≫ r_C, the CSL kernel → K_rc(Δx) × η_kfree(R, r_C),
    and η_kfree ~ 6√π (r_C/R)³.  So the scale function carries the cubic
    suppression of the geometry factor.
    """
    x = R / r_C
    eta = g.eta_kfree_closed_numeric(x, r_C=1.0)
    return eta  # relative to K_rc; the absolute kernel also depends on K_rc


__all__ = [
    "K_rc_symbolic",
    "S_DP_point",
    "S_DP_sph",
    "S_CSL_sph",
    "S_CSL_point_symbolic",
    "scale_ratio_sph",
    "full_xi_sph",
    "S_DP_sph_small_a",
    "S_DP_sph_large_a",
    "S_CSL_sph_small_a",
    "S_CSL_sph_small_rho",
    "S_CSL_sph_large_rho",
]
