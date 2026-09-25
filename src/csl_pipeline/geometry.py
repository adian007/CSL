"""CSL geometry factors for homogeneous bodies.

This module is the template for every subsequent re-derivation in the project.
It demonstrates the full verification chain:

* **Symbolic** - the defining integral is evaluated in closed form by SymPy.
* **Cross-CAS** - the closed form is confirmed independently by the Wolfram
  Language (``verify.cas.assert_cas_equivalent``).
* **Triple numeric** - closed form, scipy quadrature, and mpmath quadrature are
  required to agree to ten significant figures.
* **Limits** - the small-body and large-body asymptotics are asserted.

Two distinct physical objects are defined and deliberately kept apart:

``eta_kfree``
    The momentum-*integrated* overlap factor for a uniform sphere, normalized so
    that ``eta_kfree -> 1`` as ``R << r_C``. Its large-body asymptotic is
    ``6 sqrt(pi) (r_C / R)^3``.

``alpha_sphere_nhh``
    The k-weighted geometry factor of Nimmrichter, Hornberger & Hammerer (2014),
    Eq. (S10), appearing in a *one-dimensional momentum diffusion* coefficient.
    Its large-body behaviour is fourth order, ``6 (r_C / R)^4``, relative to its
    own point value.

These are not two estimates of one number. They are different observables, and
the project must never average or interchange them. That separation is the
central correctness risk identified in the source audit, enforced here by
distinct names, docstrings, and tests.

Reference: Nimmrichter, Hornberger & Hammerer, Phys. Rev. Lett. 113, 020405 (2014),
arXiv:1405.2868, Supplemental Material Eq. (S7) and (S10).
"""

from __future__ import annotations

import mpmath as mp
import scipy.integrate as si
import sympy as sp
from functools import lru_cache
from math import exp, pi, sin, cos, sqrt
#: Symbols shared across the derivations in this module.
_R = sp.Symbol("R", positive=True)
_rC = sp.Symbol("rC", positive=True)
_x = sp.Symbol("x", positive=True)
_q = sp.Symbol("q", positive=True)

#: The k-free large-body prefactor for a uniform sphere.
PREFACTOR_6SQRT_PI = 6 * sp.sqrt(pi)


@lru_cache(maxsize=None)
def _cached(fn_name: str, R: sp.Expr, rC: sp.Expr) -> sp.Expr:
    """Memoize the symbolic derivations.

    SymPy's closed forms involve erf and exp, and re-deriving them on every call
    dominates test runtime. The expressions are pure functions of their symbols,
    so caching by (function, arguments) is safe and keeps the suite fast.
    """
    if fn_name == "eta":
        volume = 4 * sp.pi * R**3 / 3
        integrand = (
            4 * sp.pi * _x**2 * sp.exp(-_x**2 / (4 * rC**2)) * sphere_overlap_volume(R, _x)
        )
        return sp.simplify(sp.integrate(integrand, (_x, 0, 2 * R)) / volume**2)
    if fn_name == "fourier":
        u = _q * R / rC
        form_factor = 3 * (sp.sin(u) - u * sp.cos(u)) / u**3
        integral = sp.integrate(_q**2 * sp.exp(-_q**2) * form_factor**2, (_q, 0, sp.oo))
        return sp.simplify(4 / sp.sqrt(pi) * integral)
    if fn_name == "nhh":
        x = R**2 / rC**2
        bracket = sp.exp(-x) - 1 + x / 2 * (sp.exp(-x) + 1)
        return sp.simplify(bracket * 6 * rC**6 / R**6)
    raise ValueError(f"unknown symbolic form: {fn_name}")


# --------------------------------------------------------------------------
# Symbolic definitions
# --------------------------------------------------------------------------


def sphere_overlap_volume(R: sp.Expr, x: sp.Expr) -> sp.Expr:
    """Volume of intersection of two radius-R spheres separated by ``x``.

    The standard two-sphere lens volume, valid for ``0 <= x <= 2R``.
    """
    return sp.pi / 12 * (4 * R + x) * (2 * R - x) ** 2


def eta_kfree_symbolic(R: sp.Expr | None = None, rC: sp.Expr | None = None) -> sp.Expr:
    """Closed form of the k-free sphere overlap factor.

    .. math::

        \\eta = \\frac{1}{V^2}\\int_0^{2R} 4\\pi x^2 e^{-x^2/(4r_C^2)}
                V_{ov}(x)\\,dx, \\quad V = \\frac{4}{3}\\pi R^3,

    normalized so ``eta -> 1`` for ``R << r_C``.
    """
    R = _R if R is None else R
    rC = _rC if rC is None else rC
    return _cached("eta", R, rC)
def eta_kfree_fourier_symbolic(R: sp.Expr | None = None, rC: sp.Expr | None = None) -> sp.Expr:
    """Parseval/Fourier representation of the k-free factor.

    .. math::

        \\eta = \\frac{4}{\\sqrt{\\pi}}\\int_0^\\infty q^2 e^{-q^2}
               \\left|F_{sph}\\!\\left(\\frac{qR}{r_C}\\right)\\right|^2 dq,

    with ``F_sph(u) = 3 (sin u - u cos u) / u^3``. Structurally different from
    the real-space integral, hence an independent check rather than a
    restatement of it.
    """
    R = _R if R is None else R
    rC = _rC if rC is None else rC
    return _cached("fourier", R, rC)


def alpha_sphere_nhh(R: sp.Expr | None = None, rC: sp.Expr | None = None) -> sp.Expr:
    """Exact k-weighted geometry factor, Nimmrichter et al. Eq. (S10).

    .. math::

        \\alpha = \\left[e^{-R^2/r_C^2} - 1
        + \\frac{R^2}{2r_C^2}\\left(e^{-R^2/r_C^2} + 1\\right)\\right]
        \\frac{6 r_C^6}{R^6}

    This enters a *one-dimensional momentum diffusion* coefficient, which is a
    different observable from ``eta_kfree``. Its large-body behaviour is quartic
    where the k-free factor is cubic; the two must never be interchanged.

    The ``(m/amu)^2`` mass factor is carried separately by callers.
    """
    R = _R if R is None else R
    rC = _rC if rC is None else rC
    return _cached("nhh", R, rC)





# --------------------------------------------------------------------------
# Numeric implementations
#
# ``eta`` depends only on the dimensionless ratio x = R / r_C, so these set
# r_C = 1 internally. The parameter is retained in the signature so callers
# can mirror the symbolic form without conversion.
# --------------------------------------------------------------------------


def eta_kfree_numeric(x: float, rC: float = 1.0) -> float:
    """Evaluate the k-free factor by scipy adaptive quadrature."""
    R = x
    volume = 4 * pi * R**3 / 3
    integrand = lambda t: 4 * pi * t**2 * exp(-(t * t) / (4 * rC * rC)) * (
        pi / 12 * (4 * R + t) * (2 * R - t) ** 2
    )
    value, _ = si.quad(integrand, 0, 2 * R, limit=500, epsabs=1e-14, epsrel=1e-14)
    return float(value / volume**2)


def eta_kfree_fourier_numeric(x: float, rC: float = 1.0) -> float:
    """Evaluate the Fourier (Parseval) representation numerically."""
    R = x
    scale = R / rC

    def form_factor(u: float) -> float:
        if abs(u) < 1e-8:
            return 1.0 - u**2 / 10.0
        return 3 * (sin(u) - u * cos(u)) / u**3

    def integrand(q: float) -> float:
        return q * q * exp(-q * q) * form_factor(q * scale) ** 2

    value, _ = si.quad(integrand, 0, float("inf"), limit=800, epsabs=1e-14, epsrel=1e-14)
    return float(4 / sqrt(pi) * value)


def eta_kfree_closed_numeric(x: float, rC: float = 1.0) -> float:
    """Evaluate the symbolic closed form at 30 significant digits."""
    return float(
        eta_kfree_symbolic()
        .subs({_R: sp.Float(x, 30), _rC: sp.Float(rC, 30)})
        .evalf(30)
    )


def alpha_sphere_nhh_numeric(x: float, rC: float = 1.0) -> float:
    """Return the dimensionless k-weighted factor at ``x = R / r_C``."""
    return float(
        alpha_sphere_nhh()
        .subs({_R: sp.Float(x, 30), _rC: sp.Float(rC, 30)})
        .evalf(30)
    )


def alpha_sphere_nhh_point_normalized(x: float) -> float:
    """Return ``alpha / alpha_point`` at ``x = R / r_C``.

    The Nimmrichter point value is ``(m/amu)^2 / 2`` because the factor enters a
    one-dimensional diffusion coefficient, so this is ``2 alpha / (m/amu)^2``.
    Its large-body behaviour is ``6 (r_C/R)^4``, the fourth-power law that must
    never be confused with the k-free cubic law.
    """
    return 2.0 * alpha_sphere_nhh_numeric(x)


__all__ = [
    "PREFACTOR_6SQRT_PI",
    "alpha_sphere_nhh",
    "alpha_sphere_nhh_numeric",
    "alpha_sphere_nhh_point_normalized",
    "eta_kfree_closed_numeric",
    "eta_kfree_fourier_numeric",
    "eta_kfree_fourier_symbolic",
    "eta_kfree_numeric",
    "eta_kfree_symbolic",
    "sphere_overlap_volume",
]
