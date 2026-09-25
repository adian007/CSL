"""Verification harness for the CSL / DP pipeline.

Three layers, each able to catch errors the others miss:

``cas``
    SymPy <-> Wolfram Language equivalence, requiring agreement from two
    independent computer algebra systems before a closed form is accepted.
``precision``
    Arbitrary-precision three-route agreement (closed form, scipy quadrature,
    mpmath quadrature) to ten significant figures.
``limits``
    Physical limits and regime boundaries as executable assertions.
"""

from __future__ import annotations

__all__ = ["cas", "limits", "precision"]
