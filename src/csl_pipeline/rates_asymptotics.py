"""
Add missing symbolic asymptotic functions to rates.py.
These are needed for the cross-CAS verification of K_DP^sph asymptotics.
"""

import sympy as sp
from csl_pipeline.rates import _A

def kstar_small_a_symbolic() -> sp.Expr:
    """Return the small-a leading asymptotic of K_DP^sph(a): a²/2."""
    return _A**2 / 2

def kstar_large_a_symbolic() -> sp.Expr:
    """Return the large-a leading asymptotic of K_DP^sph(a): 6/5."""
    return sp.Rational(6, 5)
