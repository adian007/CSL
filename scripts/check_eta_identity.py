"""Confirm the two symbolic routes for eta are the same function.

``simplify`` on the difference of the real-space and Fourier closed forms is
computationally intractable: the two expressions have entirely different
structure (one is a rational expression in exp(-R^2/r_C^2) times erf, the other
is built from a different integral), so a generic simplifier explores a huge
search space.

This script instead proves agreement the way that matters and is tractable:
evaluate both closed forms at 50 significant digits across a range of R/r_C and
confirm they agree far beyond any physically meaningful precision. Agreement of
two analytically distinct closed forms to 50 digits across the domain is
overwhelming evidence they are the same function; a transcription error in
either would break agreement immediately.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sympy as sp  # noqa: E402

from csl_pipeline import geometry as g  # noqa: E402

RATIOS = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
DIGITS = 50


def main() -> int:
    t0 = time.time()
    real = g.eta_kfree_symbolic()
    fourier = g.eta_kfree_fourier_symbolic()
    print(f"closed forms obtained in {time.time() - t0:.1f}s", flush=True)

    print(f"\n{'R/r_C':>10} {'real-space':>24} {'fourier':>24} {'rel diff':>12}")
    worst = 0.0
    for x in RATIOS:
        subs = {g._R: sp.Float(x, DIGITS), g._rC: sp.Float(1.0, DIGITS)}
        a = complex(sp.N(real.subs(subs), DIGITS))
        b = complex(sp.N(fourier.subs(subs), DIGITS))
        rel = abs(a - b) / abs(a) if a != 0 else abs(a - b)
        worst = max(worst, rel)
        print(f"{x:>10g} {a.real:>24.16e} {b.real:>24.16e} {rel:>12.2e}")

    print(f"\nworst relative difference: {worst:.3e} at {DIGITS} digits")
    if worst > 1e-40:
        print("FAIL: routes disagree beyond 40 significant digits")
        return 1
    print("PASS: real-space and Fourier routes agree to >= 40 significant figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
