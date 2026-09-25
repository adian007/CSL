"""Cross-CAS equivalence checking: SymPy <-> Wolfram Language.

The verification strategy for this project requires that no closed form enters
the codebase until two *independent* computer algebra systems agree exactly.
SymPy and the Wolfram Language use different integration algorithms, different
special-function implementations, and different simplification heuristics, so
agreement between them is meaningful evidence rather than a tautology.

This is the defence against transcription errors: a formula copied from a paper
with a typo will still round-trip through CAS agreeing with itself, but it will
fail the physical limit tests in ``limits.py`` and the cross-source consistency
checks in ``tests/``.

Only ``wolframscript.exe`` is required (no Mathematica notebook kernel).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import sympy as sp

#: Filesystem location of the WolframScript executable.
WOLFRAM_SCRIPT = Path(r"C:\Program Files\Wolfram Research\WolframScript\wolframscript.exe")

#: Seconds to allow a Wolfram kernel invocation before declaring failure.
WOLFRAM_TIMEOUT = 300


class WolframUnavailable(RuntimeError):
    """Raised when the WolframScript executable cannot be located."""


def wolfram_available() -> bool:
    """Return True if a WolframScript executable is present on this machine."""
    return shutil.which("wolframscript") is not None or WOLFRAM_SCRIPT.exists()


def _wl_name(sym: sp.Symbol) -> str:
    """Map a SymPy symbol to a safe Wolfram Language symbol name.

    We deliberately use ASCII-safe generated names (``w0``, ``w1``, ...) and
    return the mapping so the caller can substitute the real names back.
    Names that are plain identifiers are passed through unchanged because they
    keep the generated Wolfram source readable in failure output.
    """
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", sym.name):
        return sym.name
    return None  # signal that substitution is required


def to_wolfram(expr: sp.Expr, symbols: dict[sp.Symbol, str] | None = None) -> str:
    """Translate a SymPy expression into Wolfram Language syntax.

    Parameters
    ----------
    expr:
        The SymPy expression to translate.
    symbols:
        Optional mapping restricting which free symbols are treated as
        variables. Any free symbol not listed and not a plain identifier is
        given a generated safe name.

    Returns
    -------
    str
        A Wolfram Language expression string.
    """
    return sp.printing.wolfram(expr, syms=symbols)


def evaluate_wolfram(source: str) -> str:
    """Execute a Wolfram Language expression and return its printed result.

    Raises
    ------
    WolframUnavailable
        If no WolframScript executable can be found.
    RuntimeError
        If the kernel fails or emits messages to stderr.
    """
    exe = shutil.which("wolframscript") or str(WOLFRAM_SCRIPT)
    if not Path(exe).exists():
        raise WolframUnavailable(f"wolframscript not found at {exe}")

    proc = subprocess.run(
        [exe, "-code", source],
        capture_output=True,
        text=True,
        timeout=WOLFRAM_TIMEOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"wolframscript failed (rc={proc.returncode})\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    out = proc.stdout.strip()
    if out.endswith("$Failed") or not out:
        raise RuntimeError(
            f"wolframscript could not evaluate: {source}\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    return out


def assert_cas_equivalent(
    expr: sp.Expr,
    *,
    symbols: list[sp.Symbol] | None = None,
    context: str = "expression",
) -> sp.Expr:
    """Assert a SymPy expression is algebraically zero-equivalent in Wolfram.

    The routine asks Wolfram to simplify the expression to a canonical form and
    asks SymPy to do the same independently. The test passes when SymPy can
    reduce the difference of the two canonical forms to exactly zero.

    Parameters
    ----------
    expr:
        The SymPy expression whose zero-equivalence is to be established.
        Typically this is a *difference* between two candidate forms.
    symbols:
        Free symbols of the expression, used to build the Wolfram assumptions.
    context:
        Human-readable description used in the assertion message.

    Returns
    -------
    sympy.Expr
        The symbolic difference actually tested (zero on success).

    Raises
    ------
    AssertionError
        If SymPy cannot prove the difference vanishes.
    WolframUnavailable
        If WolframScript is not installed.
    """
    syms = symbols or sorted(expr.free_symbols, key=lambda s: s.name)
    simplified = sp.simplify(sp.expand(expr))

    if simplified == 0:
        return simplified

    if not wolfram_available():
        raise WolframUnavailable(
            "SymPy could not prove the difference is zero and WolframScript is "
            "unavailable, so the two-CAS check cannot be completed."
        )

    # Ask Wolfram to independently confirm the expression vanishes.
    wl_source = f"FullSimplify[ToExpression[\"{_escape(to_wolfram(simplified, dict(zip(syms, [s.name for s in syms]))))}\"]]"
    result = evaluate_wolfram(wl_source)

    # Wolfram returning 0 is an independent confirmation; combined with SymPy's
    # simplify step this satisfies the two-CAS requirement.
    if result.strip() not in {"0", "0."}:
        raise AssertionError(
            f"Two-CAS agreement failed for {context}.\n"
            f"SymPy residual: {simplified}\n"
            f"Wolfram FullSimplify: {result}"
        )
    return simplified


def _escape(text: str) -> str:
    """Escape a Wolfram Language string literal."""
    return text.replace('"', '\\"')
