"""Physical limit and regime-boundary test helpers.

A closed form that reproduces a published formula can still be wrong for this
project if it fails a physical limit. These helpers turn the limits stated in
the source literature into executable assertions.

Two limits govern essentially every formula in the project:

* ``eta -> 1`` as ``R / r_C -> 0``
  A body much smaller than the localization length behaves as a point particle,
  so the geometry factor must tend to unity.

* ``eta -> 6*sqrt(pi) * (r_C / R)**3`` as ``R / r_C -> infinity``
  A body much larger than the localization length samples only its surface
  layer. The leading term follows from the Gaussian integral
  ``int d^3 s exp(-s^2 / (4 r_C^2)) = (4 pi r_C^2)^{3/2}``.

The second limit is deliberately *not* the Nimmrichter-Hornberger-Hammerer
``6 (r_C / R)**4`` law. That is the k-weighted gradient (diffusion) object, a
different physical quantity. Keeping the two apart is enforced by naming, by
separate tests, and by the registry in ``csl_pipeline.rates``.
"""

from __future__ import annotations

import math

import mpmath as mp
import sympy as sp

#: Digits of precision used by the limit helpers.
DEFAULT_DPS = 50

#: Relative tolerance for agreement in a limit test.
DEFAULT_RTOL = 1e-10


def configure(dps: int = DEFAULT_DPS) -> None:
    """Set the working precision of mpmath."""
    mp.mp.dps = dps


def scaled_limit(
    f_numeric,
    x: float,
    power: float,
    *,
    r_C: float = 1.0,
    expected: float,
    dps: int = DEFAULT_DPS,
    rtol: float = DEFAULT_RTOL,
) -> dict[str, float]:
    """Test that ``f(x) * x**power -> expected`` as ``x -> infinity``.

    Parameters
    ----------
    f_numeric:
        Callable evaluating the quantity under test at ``x = R / r_C``.
    x:
        Large finite value of ``R / r_C`` at which to sample the limit.
    power:
        The power of ``x`` to multiply by so the product approaches a constant.
    r_C:
        The localization length, used only to reconstruct dimensions.
    expected:
        The asymptotic constant the product must approach.
    dps:
        Digits of precision for the evaluation.
    rtol:
        Relative tolerance for the comparison.

    Returns
    -------
    dict
        Diagnostic values including the observed product and relative error.
    """
    configure(dps)
    with mp.workdps(dps):
        value = mp.mpf(f_numeric(x))
        product = value * mp.mpf(x) ** power
        rel_err = abs(product - mp.mpf(expected)) / abs(mp.mpf(expected))
    if rel_err > mp.mpf(rtol):
        raise AssertionError(
            f"Limit test failed: f(x)*x**{power} at x={x:g} gave "
            f"{mp.nstr(product, 12)}, expected {mp.nstr(expected, 12)} "
            f"(rel err {mp.nstr(rel_err, 6)} > {rtol:g})"
        )
    return {
        "x": x,
        "product": float(product),
        "expected": expected,
        "rel_err": float(rel_err),
        "r_C": r_C,
    }


def small_r_limit(
    f_numeric,
    *,
    x: float = 1e-4,
    expected: float = 1.0,
    dps: int = DEFAULT_DPS,
    rtol: float = 1e-8,
) -> dict[str, float]:
    """Test that ``f(R / r_C) -> expected`` (default 1) as ``R / r_C -> 0``."""
    configure(dps)
    with mp.workdps(dps):
        value = mp.mpf(f_numeric(x))
        rel_err = abs(value - mp.mpf(expected)) / abs(mp.mpf(expected))
    if rel_err > mp.mpf(rtol):
        raise AssertionError(
            f"Small-R limit failed: f({x:g}) = {mp.nstr(value, 12)}, "
            f"expected {expected} (rel err {mp.nstr(rel_err, 6)} > {rtol:g})"
        )
    return {"x": x, "value": float(value), "expected": expected, "rel_err": float(rel_err)}


def asymptotic_series(
    expr: sp.Expr,
    var: sp.Symbol,
    x0: float,
    power: float,
    n: int = 3,
) -> sp.Expr:
    """Expand ``expr`` in powers of ``(var / x0)`` about infinity.

    Parameters
    ----------
    expr:
        The symbolic expression to expand.
    var:
        The large variable, assumed positive.
    x0:
        Finite scale introduced to regularise the expansion.
    power:
        The leading power of ``var`` that cancels in the scaled limit.
    n:
        Number of expansion terms to retain.

    Returns
    -------
    sympy.Expr
        The expansion with the leading ``var**power`` term divided out.
    """
    t = sp.Dummy("t", positive=True)
    scaled = sp.simplify(expr * var**power)
    # var = x0 / t makes var -> infinity correspond to t -> 0.
    substituted = sp.simplify(scaled.subs(var, x0 / t))
    series = sp.series(substituted, t, 0, n).removeO()
    return sp.simplify(series)


def require(condition: bool, message: str) -> None:
    """Raise AssertionError with ``message`` when ``condition`` is false."""
    if not condition:
        raise AssertionError(message)


def constant_six_sqrt_pi() -> float:
    """Return 6*sqrt(pi), the k-free large-body sphere prefactor."""
    return 6.0 * math.sqrt(math.pi)
