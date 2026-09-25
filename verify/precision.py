"""Arbitrary-precision numerical agreement harness.

Layer 3 of the verification architecture requires that three structurally
different routes to the same quantity agree to at least ten significant
figures, evaluated at high precision:

1. the closed form,
2. direct quadrature of the defining integral,
3. an alternative representation (Fourier/Parseval), where one exists.

The functions here provide the comparison machinery. They deliberately avoid
numpy for the reference evaluation so that results are independent of the
BLAS/LAPACK build; mpmath is used for the high-precision reference and scipy
for the independent adaptive quadrature, so an error in one library cannot
silently validate the other.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

import mpmath as mp
import scipy.integrate as si

#: Significant figures required of every agreement check.
DEFAULT_SIGFIGS = 10


class Agreement(NamedTuple):
    """Result of comparing two evaluations of the same quantity."""

    label: str
    left: float
    right: float
    rel_err: float
    sigfigs: int

    @property
    def ok(self) -> bool:
        """True when the agreement meets the requested precision."""
        return self.sigfigs >= DEFAULT_SIGFIGS


def configure(dps: int = 50) -> None:
    """Set mpmath working precision."""
    mp.mp.dps = dps


def sigfigs_agree(a: float, b: float) -> int:
    """Return the number of significant figures on which two floats agree.

    Returns a negative number when the values do not agree to even one
    significant figure, and ``inf`` when both are exactly zero.
    """
    if a == b:
        return math.inf
    if a == 0.0 or b == 0.0:
        return -1
    rel = abs(a - b) / max(abs(a), abs(b))
    if rel == 0.0:
        return math.inf
    return int(math.floor(-math.log10(rel)))


def assert_agree(
    left: float,
    right: float,
    *,
    label: str = "value",
    sigfigs: int = DEFAULT_SIGFIGS,
) -> Agreement:
    """Assert two evaluations agree to at least ``sigfigs`` significant figures."""
    n = sigfigs_agree(left, right)
    result = Agreement(label, left, right, abs(left - right) / max(abs(left), abs(right), 1e-300), n)
    if n < sigfigs:
        raise AssertionError(
            f"{label}: agreement is only {n} significant figures "
            f"(need {sigfigs}). left={left!r} right={right!r} rel={result.rel_err:.3e}"
        )
    return result


def assert_three_way(
    routes: dict[str, Callable[[], float]],
    *,
    sigfigs: int = DEFAULT_SIGFIGS,
) -> dict[str, Agreement]:
    """Assert every pair among three independent evaluation routes agrees."""
    if len(routes) < 3:
        raise ValueError("Three-way agreement requires at least three named routes")
    names = list(routes)
    values = {name: float(routes[name]()) for name in names}
    out: dict[str, Agreement] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            out[f"{a} vs {b}"] = assert_agree(values[a], values[b], label=f"{a} vs {b}", sigfigs=sigfigs)
    return out


def quad_reference(
    func,
    lower: float,
    upper: float,
    *,
    limit: int = 500,
    epsabs: float = 1e-14,
    epsrel: float = 1e-14,
) -> float:
    """Adaptive quadrature via scipy, independent of the mpmath reference path."""
    value, _ = si.quad(func, lower, upper, limit=limit, epsabs=epsabs, epsrel=epsrel)
    return float(value)


def mp_reference(
    func_mp,
    lower,
    upper,
    *,
    dps: int = 50,
) -> float:
    """High-precision quadrature via mpmath."""
    configure(dps)
    with mp.workdps(dps):
        return float(mp.quad(func_mp, lower, upper))


def quadrature_route(func_mp, lower, upper, *, dps: int = 50) -> Callable[[], float]:
    """Build a zero-argument route evaluating ``func_mp`` by mpmath quadrature."""

    def _route() -> float:
        return mp_reference(func_mp, lower, upper, dps=dps)

    return _route


def scipy_quadrature_route(
    func, lower: float, upper: float, *, limit: int = 500
) -> Callable[[], float]:
    """Build a zero-argument route evaluating ``func`` by scipy quadrature."""

    def _route() -> float:
        return quad_reference(func, lower, upper, limit=limit)

    return _route


def report(title: str, agreements: dict[str, Agreement]) -> str:
    """Format an agreement report for test output and CI logs."""
    lines = [f"--- {title} ---"]
    for name, agr in agreements.items():
        lines.append(
            f"  {name:<44} sigfigs={str(agr.sigfigs):>6}  rel_err={agr.rel_err:.3e}"
        )
    return "\n".join(lines)
