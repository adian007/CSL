"""Verification tests for the CSL geometry layer.

These tests encode the acceptance criteria agreed for the project:

* the defining integral is evaluated in closed form (symbolic layer);
* SymPy and the Wolfram Language agree exactly (cross-CAS layer);
* closed form, scipy quadrature, and the Fourier route agree to ten
  significant figures (triple-numeric layer);
* the physical limits hold (limit layer);
* the k-free cubic law and the k-weighted quartic law remain distinct objects
  (convention-firewall layer).
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from csl_pipeline import geometry as g
from verify import cas, limits, precision

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

#: Ratios R/r_C at which the three numeric routes are compared.
SAMPLE_RATIOS = [0.5, 1.0, 2.0, 10.0, 100.0, 1000.0]

#: Symbol used when reducing expressions to a single variable.
_RPOS = sp.Symbol("R", positive=True)


class TestSymbolicLayer:
    """The defining integral must evaluate in closed form."""

    def test_eta_closed_form_is_positive(self):
        expr = g.eta_kfree_symbolic()
        value = float(expr.subs({g._R: 1, g._rC: 1}).evalf(30))
        assert value > 0.0

    def test_eta_depends_only_on_ratio(self):
        """eta is dimensionless: scaling r_C leaves it invariant."""
        expr = g.eta_kfree_symbolic()
        scaled = sp.simplify(expr.subs({g._rC: 3 * g._rC}) - expr)
        assert sp.simplify(scaled) == 0

    def test_fourier_form_matches_overlap_form(self):
        """Real-space and momentum-space routes are the same object."""
        residual = sp.simplify(sp.expand(g.eta_kfree_symbolic() - g.eta_kfree_fourier_symbolic()))
        cas.assert_cas_equivalent(residual, context="eta real-space vs Fourier")


class TestCrossCASLayer:
    """Two independent computer algebra systems must agree exactly."""
    __test__ = True

    @pytest.mark.cas
    def test_sympy_wolfram_agree_on_eta(self):
        expr = g.eta_kfree_symbolic().subs({g._rC: 1})
        residual = sp.simplify(sp.expand(expr))
        cas.assert_cas_equivalent(residual, symbols=[_RPOS], context="eta closed form")

    @pytest.mark.cas
    @pytest.mark.slow
    def test_sympy_wolfram_agree_on_nhh_factor(self):
        expr = g.alpha_sphere_nhh().subs({g._rC: 1})
        residual = sp.simplify(sp.expand(expr))
        cas.assert_cas_equivalent(
            residual, symbols=[_RPOS], context="NHH alpha_sphere, SymPy vs Wolfram"
        )


class TestTripleNumericLayer:
    """Three structurally different routes must agree to 10 significant figures."""

    @pytest.mark.parametrize("x", SAMPLE_RATIOS)
    @pytest.mark.slow
    def test_three_way_agreement(self, x):
        routes = {
            "closed_form": lambda: g.eta_kfree_closed_numeric(x),
            "scipy_quad": lambda: g.eta_kfree_numeric(x),
            "fourier_quad": lambda: g.eta_kfree_fourier_numeric(x),
        }
        agreements = precision.assert_three_way(routes, sigfigs=10)
        assert all(a.ok for a in agreements.values()), precision.report(
            f"eta three-way at R/r_C={x}", agreements
        )

    def test_port_of_legacy_verify_script(self):
        """Regression against the pre-existing verify.py/verify2.py agreement."""
        for x in [0.5, 1.0, 10.0, 100.0, 1000.0]:
            precision.assert_agree(
                g.eta_kfree_numeric(x),
                g.eta_kfree_closed_numeric(x),
                label=f"legacy ratio at {x}",
                sigfigs=9,
            )


__all__ = ["TestLimitLayer", "TestConventionFirewall", "SAMPLE_RATIOS"]


class TestLimitLayer:
    """Physical limits must hold as executable assertions."""

    def test_small_body_limit_eta_to_one(self):
        """A body much smaller than r_C must behave as a point particle."""
        result = limits.small_r_limit(g.eta_kfree_numeric, x=1e-4, expected=1.0, rtol=1e-6)
        assert abs(result["value"] - 1.0) < 1e-6

    def test_large_body_limit_eta_cubic_prefactor(self):
        """eta * x^3 -> 6 sqrt(pi) as R/r_C -> infinity."""
        expected = limits.constant_six_sqrt_pi()
        result = limits.scaled_limit(
            g.eta_kfree_numeric, x=1000.0, power=3.0, expected=expected, rtol=1e-3
        )
        assert result["rel_err"] < 1e-3

    def test_closed_form_large_body_prefactor(self):
        """The closed form itself must exhibit the cubic prefactor."""
        x = 1000.0
        product = g.eta_kfree_closed_numeric(x) * x**3
        expected = 6 * math.sqrt(math.pi)
        assert abs(product - expected) / expected < 1e-3


class TestConventionFirewall:
    """The k-free and k-weighted laws must never be conflated.

    The two laws differ in power law (cubic vs quartic) and are different
    physical observables: a momentum-integrated overlap and a one-dimensional
    momentum diffusion geometry factor.
    """

    @pytest.mark.parametrize("x", [10.0, 100.0, 1000.0])
    def test_two_laws_have_different_power_dependence(self, x):
        kfree = g.eta_kfree_closed_numeric(x)
        kweighted = g.alpha_sphere_nhh_point_normalized(x)
        assert kfree > kweighted > 0.0
        ratio = kfree / kweighted
        # ratio grows roughly like x because (r_C/R)^3 / (r_C/R)^4 = R/r_C
        assert ratio > 0.5 * x

    def test_quartic_law_large_body_prefactor_is_six(self):
        """alpha_nhh * x^4 -> 6, the Nimmrichter point-normalized prefactor."""
        x = 1000.0
        product = g.alpha_sphere_nhh_point_normalized(x) * x**4
        assert abs(product - 6.0) / 6.0 < 1e-3

    def test_laws_never_equal(self):
        """A direct guard against accidental aliasing."""
        for x in [0.5, 1.0, 10.0, 100.0]:
            assert g.eta_kfree_closed_numeric(x) != g.alpha_sphere_nhh_point_normalized(x)


class TestWolframAvailability:
    """The cross-CAS layer depends on WolframScript being installed."""

    def test_wolfram_detected(self):
        assert cas.wolfram_available(), (
            "WolframScript not found. The cross-CAS verification layer cannot run."
        )

    def test_wolfram_can_evaluate_a_closed_form(self):
        result = cas.evaluate_wolfram(r"Integrate[4 Pi x^2 Exp[-x^2], {x, 0, Infinity}]")
        assert "Sqrt" in result
