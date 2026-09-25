"""
CSL/DP Cross-CAS and Cross-Validation Verification Script
Runs all 10 checks from the task spec.
"""

import math
import mpmath as mp
import scipy.integrate as si
import sympy as sp
import sys

sys.path.insert(0, "D:/csl")

from csl_pipeline import rates as r
from csl_pipeline import geometry as g
from verify import cas, limits, precision

print("=" * 80)
print("CSL/DP CROSS-CAS AND CROSS-VALIDATION REPORT")
print("=" * 80)

results = []

def record(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((name, status, detail))
    flag = "✓" if passed else "✗"
    print(f"[{flag}] {name}")
    if detail:
        for line in detail.split("\n"):
            print(f"      {line}")
    print()

# ---------------------------------------------------------------------------
# 1. K_DP_sph continuity at a=2
# ---------------------------------------------------------------------------
print("\n### CHECK 1: K_DP_sph continuity at a=2")
try:
    residual = r.kdp_sph_continuity_at_two()
    diff_val = float(residual.evalf(30))
    print(f"  SymPy residual (upper - lower at a=2): {diff_val!r}")
    wolfram_ok = False
    wolfram_msg = "Wolfram not available"
    if cas.wolfram_available():
        try:
            diff_sym = sp.simplify(residual)
            cas.assert_cas_equivalent(
                diff_sym,
                symbols=[r._A],
                context="K_DP^sph continuity at a=2"
            )
            wolfram_ok = True
            wolfram_msg = "Wolfram confirms residual = 0"
        except Exception as e:
            wolfram_msg = f"Wolfram check failed: {e}"
    passed_check1 = abs(diff_val) < 1e-25
    if cas.wolfram_available():
        passed_check1 = passed_check1 and wolfram_ok
    record("1. K_DP_sph continuity at a=2 (kdp_sph_continuity_at_two)",
           passed_check1,
           f"SymPy residual = {diff_val!r}\n"
           f"Wolfram: {wolfram_msg}")
except Exception as e:
    record("1. K_DP_sph continuity at a=2", False, f"Exception: {e}")

# ---------------------------------------------------------------------------
# 2. K_DP_sph small-a asymptotics: kstar_small_a_symbolic() == a^2/2
# ---------------------------------------------------------------------------
print("\n### CHECK 2: K_DP_sph small-a asymptotics")
try:
    small = r.kstar_small_a_symbolic()
    expected = r._A**2 / 2
    residual = sp.simplify(small - expected)
    sval = float(residual.evalf(30))
    print(f"  SymPy residual (kstar_small - a²/2): {sval!r}")
    wolfram_ok = False
    wolfram_msg = "Wolfram not available"
    if cas.wolfram_available():
        try:
            cas.assert_cas_equivalent(
                residual,
                symbols=[r._A],
                context="K_DP^sph small-a expansion"
            )
            wolfram_ok = True
            wolfram_msg = "Wolfram confirms residual = 0"
        except Exception as e:
            wolfram_msg = f"Wolfram check failed: {e}"
    passed_check2 = abs(sval) < 1e-25
    if cas.wolfram_available():
        passed_check2 = passed_check2 and wolfram_ok
    record("2. K_DP_sph small-a asymptotics (kstar_small_a_symbolic == a²/2)",
           passed_check2,
           f"SymPy residual = {sval!r}\n"
           f"Wolfram: {wolfram_msg}")
except Exception as e:
    record("2. K_DP_sph small-a asymptotics", False, f"Exception: {e}")

# ---------------------------------------------------------------------------
# 3. K_DP_sph large-a asymptotics: kstar_large_a_symbolic() == 6/5
# ---------------------------------------------------------------------------
print("\n### CHECK 3: K_DP_sph large-a asymptotics")
try:
    large = r.kstar_large_a_symbolic()
    expected = sp.Rational(6, 5)
    residual = sp.simplify(large - expected)
    lval = float(residual.evalf(30))
    print(f"  SymPy residual (kstar_large - 6/5): {lval!r}")
    wolfram_ok = False
    wolfram_msg = "Wolfram not available"
    if cas.wolfram_available():
        try:
            cas.assert_cas_equivalent(
                residual,
                symbols=[r._A],
                context="K_DP^sph large-a expansion"
            )
            wolfram_ok = True
            wolfram_msg = "Wolfram confirms residual = 0"
        except Exception as e:
            wolfram_msg = f"Wolfram check failed: {e}"
    passed_check3 = abs(lval) < 1e-25
    if cas.wolfram_available():
        passed_check3 = passed_check3 and wolfram_ok
    record("3. K_DP_sph large-a asymptotics (kstar_large_a_symbolic == 6/5)",
           passed_check3,
           f"SymPy residual = {lval!r}\n"
           f"Wolfram: {wolfram_msg}")
except Exception as e:
    record("3. K_DP_sph large-a asymptotics", False, f"Exception: {e}")

# ---------------------------------------------------------------------------
# 4. Xi_point symbolic: NO free symbols named 'm' or 'tau'
# ---------------------------------------------------------------------------
print("\n### CHECK 4: Xi_point symbolic — no free 'm' or 'tau'")
try:
    expr = r.xi_point_symbolic()
    free = expr.free_symbols
    has_m = any(s.name in ("m", "tau", "t") for s in free)
    free_names = sorted(s.name for s in free)
    simplified = sp.simplify(sp.expand(expr))
    print(f"  Free symbols in XI_POINT_SYMBOLIC: {free_names}")
    print(f"  Has forbidden 'm'/'tau'/'t' symbols: {has_m}")
    passed_check4 = not has_m and len(free_names) > 0
    record("4. XI_POINT_SYMBOLIC has no free symbols 'm'/'tau' (m²τ cancellation)",
           passed_check4,
           f"Free symbols: {free_names}\n"
           f"Contains forbidden: {has_m}")
except Exception as e:
    record("4. Xi_point symbolic m/tau check", False, f"Exception: {e}")

# ---------------------------------------------------------------------------
# 5. ELL_STAR_SYMBOLIC numeric check
# ---------------------------------------------------------------------------
print("\n### CHECK 5: ELL_STAR_SYMBOLIC = G*amu²/(λ*hbar) == 1.745129893e-13")
try:
    expr = r.ELL_STAR_SYMBOLIC
    val = float(expr.subs({
        r._G_P: r.G,
        r._AMU_P: r.AMU,
        r._LAMBDA_P: r.LAMBDA_GRW,
        r._HBAR_P: r.HBAR,
    }).evalf(30))
    expected = r.ELL_STAR_GRW
    rel_err = abs(val - expected) / expected
    n_sigfigs = precision.sigfigs_agree(val, expected)
    print(f"  SymPy-derived ELL_STAR: {val!r}")
    print(f"  ELL_STAR_GRW (expected): {expected!r}")
    print(f"  Relative error: {rel_err:.3e}")
    print(f"  Significant figures: {n_sigfigs}")
    passed_check5 = rel_err < 1e-8
    record("5. ELL_STAR_SYMBOLIC numeric == ELL_STAR_GRW (1.745129893e-13) within 1e-8",
           passed_check5,
           f"SymPy value: {val!r}\n"
           f"Expected:     {expected!r}\n"
           f"Rel error:    {rel_err:.3e}\n"
           f"Sigfigs:      {n_sigfigs}")
except Exception as e:
    record("5. ELL_STAR_SYMBOLIC numeric", False, f"Exception: {e}")

# ---------------------------------------------------------------------------
# 6. XSTAR_EQUATION: substitute x_star() into LHS, confirm LHS ~= RHS
# ---------------------------------------------------------------------------
print("\n### CHECK 6: XSTAR_EQUATION — LHS(x*) ~= RHS within 1e-6")
try:
    eq = r.XSTAR_EQUATION
    xstar_val = r.x_star()
    lhs_val = float(eq.lhs.subs({
        r._DX: xstar_val,
        r._RC_P: r.rC_GRW,
    }).evalf(30))
    rhs_val = float(eq.rhs.subs({
        r._G_P: r.G,
        r._AMU_P: r.AMU,
        r._LAMBDA_P: r.LAMBDA_GRW,
        r._HBAR_P: r.HBAR,
    }).evalf(30))
    rel_err = abs(lhs_val - rhs_val) / max(abs(rhs_val), 1e-30)
    n_sigfigs = precision.sigfigs_agree(lhs_val, rhs_val)
    print(f"  x* = {xstar_val!r}")
    print(f"  LHS(x*) = {lhs_val!r}")
    print(f"  RHS     = {rhs_val!r}")
    print(f"  Relative error: {rel_err:.3e}")
    print(f"  Significant figures: {n_sigfigs}")
    passed_check6 = rel_err < 1e-6
    record("6. XSTAR_EQUATION: LHS(x*) ~= RHS within 1e-6",
           passed_check6,
           f"LHS(x*) = {lhs_val!r}\n"
           f"RHS     = {rhs_val!r}\n"
           f"Rel error: {rel_err:.3e}\n"
           f"Sigfigs:   {n_sigfigs}")
except Exception as e:
    record("6. XSTAR_EQUATION", False, f"Exception: {e}")

# ---------------------------------------------------------------------------
# 7. eta_kfree three-way: closed_form, scipy_quad, fourier_quad
# ---------------------------------------------------------------------------
print("\n### CHECK 7: eta_kfree three-way (10 sigfigs at 10 x values)")
x_values_7 = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
sigfigs_required_7 = 10
all_ok_7 = True
detail_lines_7 = []
for x in x_values_7:
    try:
        routes = {
            "closed_form": lambda xx=x: g.eta_kfree_closed_numeric(xx),
            "scipy_quad": lambda xx=x: g.eta_kfree_numeric(xx),
            "fourier_quad": lambda xx=x: g.eta_kfree_fourier_numeric(xx),
        }
        agreements = precision.assert_three_way(routes, sigfigs=sigfigs_required_7)
        min_sigfigs = min(a.sigfigs for a in agreements.values())
        detail_lines_7.append(
            f"  x={x:>8}: min_sigfigs={min_sigfigs:>3}  "
            + " | ".join(f"{k}: {v.sigfigs}" for k, v in agreements.items())
        )
        if not all(a.ok for a in agreements.values()):
            all_ok_7 = False
    except AssertionError as e:
        all_ok_7 = False
        detail_lines_7.append(f"  x={x:>8}: FAIL — {e}")
    except Exception as e:
        all_ok_7 = False
        detail_lines_7.append(f"  x={x:>8}: EXCEPTION — {e}")
record("7. eta_kfree three-way at 10 sigfigs (x = 0.1...1000)",
       all_ok_7,
       "\n".join(detail_lines_7))

# ---------------------------------------------------------------------------
# 8. alpha_nhh: alpha_sphere_nhh_numeric vs explicit formula
# ---------------------------------------------------------------------------
print("\n### CHECK 8: alpha_sphere_nhh explicit formula (15 sigfigs)")
x_values_8 = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0]
sigfigs_required_8 = 15
all_ok_8 = True
detail_lines_8 = []

def alpha_explicit(x_val, rC_val=1.0):
    """Explicit formula: [e^{-x^2} - 1 + x^2/2*(e^{-x^2}+1)] * 6/x^6."""
    x = x_val
    ex2 = math.exp(-x * x)
    bracket = ex2 - 1.0 + (x * x / 2.0) * (ex2 + 1.0)
    return bracket * 6.0 / (x ** 6)

for x in x_values_8:
    try:
        numeric_val = g.alpha_sphere_nhh_numeric(x)
        explicit_val = alpha_explicit(x)
        n_sigfigs = precision.sigfigs_agree(numeric_val, explicit_val)
        passed_pair = n_sigfigs >= sigfigs_required_8
        if not passed_pair:
            all_ok_8 = False
        detail_lines_8.append(
            f"  x={x:>8}: numeric={numeric_val:.15e}  explicit={explicit_val:.15e}  "
            f"sigfigs={n_sigfigs}  {'OK' if passed_pair else 'FAIL'}"
        )
    except Exception as e:
        all_ok_8 = False
        detail_lines_8.append(f"  x={x:>8}: EXCEPTION — {e}")
record("8. alpha_sphere_nhh vs explicit formula at 15 sigfigs",
       all_ok_8,
       "\n".join(detail_lines_8))

# ---------------------------------------------------------------------------
# 9. K_DP_sph_closed(a) vs piecewise formula
# ---------------------------------------------------------------------------
print("\n### CHECK 9: K_DP_sph_closed vs piecewise formula (15 sigfigs)")
a_values_9 = [0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 100.0]
sigfigs_required_9 = 15
all_ok_9 = True
detail_lines_9 = []

def kdp_sph_piecewise(a):
    """Reference piecewise formula."""
    if a <= 0:
        return 0.0
    if a <= 2.0:
        return a * a / 2.0 - 3.0 * a**3 / 16.0 + a**5 / 160.0
    return 6.0 / 5.0 - 1.0 / a

for a in a_values_9:
    try:
        closed_val = r.K_DP_sph_closed(a)
        piece_val = kdp_sph_piecewise(a)
        n_sigfigs = precision.sigfigs_agree(closed_val, piece_val)
        passed_pair = n_sigfigs >= sigfigs_required_9
        if not passed_pair:
            all_ok_9 = False
        detail_lines_9.append(
            f"  a={a:>8}: closed={closed_val:.15e}  piecewise={piece_val:.15e}  "
            f"sigfigs={n_sigfigs}  {'OK' if passed_pair else 'FAIL'}"
        )
    except Exception as e:
        all_ok_9 = False
        detail_lines_9.append(f"  a={a:>8}: EXCEPTION — {e}")
record("9. K_DP_sph_closed vs piecewise at 15 sigfigs",
       all_ok_9,
       "\n".join(detail_lines_9))

# ---------------------------------------------------------------------------
# 10. Xi_point(Δx, R) vs formula (λ*hbar*d_eff*K_rc)/(G*amu²)
# ---------------------------------------------------------------------------
print("\n### CHECK 10: Xi_point vs explicit formula (15 sigfigs)")
pairs_10 = [
    (1e-9, 1e-6),
    (1e-6, 1e-6),
    (1e-6, 1e-9),
    (5e-7, 2e-7),
    (2e-8, 1e-8),
]
r_C_10 = 1e-7
sigfigs_required_10 = 15
all_ok_10 = True
detail_lines_10 = []

for Δx, R in pairs_10:
    try:
        xi_numeric = r.Xi_point(Δx, R, r_C=r_C_10)
        d_eff_val = r.d_eff(Δx, R)
        K_rc_val = r.K_rc(Δx, r_C_10)
        xi_formula = (r.LAMBDA_GRW * r.HBAR * d_eff_val * K_rc_val
                      / (r.G * r.AMU * r.AMU))
        n_sigfigs = precision.sigfigs_agree(xi_numeric, xi_formula)
        passed_pair = n_sigfigs >= sigfigs_required_10
        if not passed_pair:
            all_ok_10 = False
        detail_lines_10.append(
            f"  Δx={Δx:.1e}, R={R:.1e}: numeric={xi_numeric:.6e}  "
            f"formula={xi_formula:.6e}  sigfigs={n_sigfigs}  "
            f"{'OK' if passed_pair else 'FAIL'}"
        )
    except Exception as e:
        all_ok_10 = False
        detail_lines_10.append(f"  Δx={Δx:.1e}, R={R:.1e}: EXCEPTION — {e}")
record("10. Xi_point(Δx,R) vs (λ·ħ·d_eff·K_rc)/(G·amu²) at 15 sigfigs",
       all_ok_10,
       "\n".join(detail_lines_10))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
total = len(results)
passed = sum(1 for _, s, _ in results if s == "PASS")
failed = total - passed
print(f"Total checks: {total}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print()
for name, status, detail in results:
    flag = "✓" if status == "PASS" else "✗"
    print(f"[{flag}] {name}  —  {status}")
    if status == "FAIL" and detail:
        for line in detail.split("\n")[:3]:
            print(f"      {line}")
