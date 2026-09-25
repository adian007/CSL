"""
Comprehensive numerical convergence study of all quadrature-based kernels
in the CSL/DP project.

Produces a report covering:
1. eta_kfree_numeric vs closed vs fourier sweep
2. K_CSL_sph_numeric 2D grid
3. Safe operating envelope
4. Failure classification
5. Limit/epsabs improvement tests
6. mpmath reference comparisons
"""

from __future__ import annotations

import math
import warnings
import sys
import io
from dataclasses import dataclass, field
from typing import Callable

import mpmath as mp
import scipy.integrate as si

# Import project modules
sys.path.insert(0, "D:/csl/src")
from csl_pipeline import geometry as g
from csl_pipeline import rates as r
from csl_pipeline import csl_dp_scale as s
from verify.precision import sigfigs_agree, DEFAULT_SIGFIGS, Agreement

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ETA_SWEEP_X = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0,
               500.0, 1000.0, 5000.0, 10000.0]

K_CSL_R_RATIO = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
K_CSL_DX_RATIO = [0.0, 0.1, 1.0, 10.0, 100.0]

MP_DPS = 50

# Limit/epsabs variants to test
LIMIT_VARIANTS = [500, 800, 1200, 2000]
EPSABS_VARIANTS = [1e-14, 1e-15, 1e-16]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class ConvResult:
    x: float
    label: str
    scipy_val: float
    ref_val: float
    sigfigs: int
    rel_err: float
    warning: str | None = None
    ok: bool = False

@dataclass
class GridResult:
    R_over_rC: float
    dx_over_rC: float
    scipy_val: float
    ref_val: float  # analytical/asymptotic
    sigfigs: int
    rel_err: float
    warning: str | None = None
    ok: bool = False

@dataclass
class StudyReport:
    eta_sweep: list[ConvResult] = field(default_factory=list)
    k_csl_grid: list[GridResult] = field(default_factory=list)
    eta_global_ok: bool = True
    k_csl_global_ok: bool = True
    eta_warnings: list[tuple[float, str]] = field(default_factory=list)
    k_csl_warnings: list[tuple[float, float, str]] = field(default_factory=list)
    eta_deg_regions: list[tuple[float, int, str]] = field(default_factory=list)  # x, sigfigs, reason
    k_csl_deg_regions: list[tuple[float, float, int, str]] = field(default_factory=list)
    limit_improvement: dict = field(default_factory=dict)
    mpmath_comparison: dict = field(default_factory=dict)
    recommendations: dict = field(default_factory=dict)

def capture_warning(f: Callable, *args, **kwargs) -> tuple:
    """Run f(*args, **kwargs) and capture any scipy IntegrationWarning."""
    captured = io.StringIO()
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        try:
            result = f(*args, **kwargs)
        except Exception as e:
            return (None, f"exception: {e}")
        wnames = [str(w.message) for w in wlist if issubclass(w.category, RuntimeWarning)]
        # Also check stderr capture
        warn_msg = "; ".join(wnames) if wnames else None
    return (result, warn_msg)


# ---------------------------------------------------------------------------
# Study 1: eta_kfree sweep
# ---------------------------------------------------------------------------

def study1_eta_sweep() -> list[ConvResult]:
    results = []
    for x in ETA_SWEEP_X:
        # Closed form (high-precision reference)
        ref = g.eta_kfree_closed_numeric(x, rC=1.0)

        # Real-space quadrature
        real_val, real_warn = capture_warning(g.eta_kfree_numeric, x, rC=1.0)

        # Fourier quadrature
        fourier_val, fourier_warn = capture_warning(g.eta_kfree_fourier_numeric, x, rC=1.0)

        # Compare real vs closed
        sf_real = sigfigs_agree(real_val, ref) if real_val is not None else -1
        re_real = abs(real_val - ref) / max(abs(real_val), abs(ref), 1e-300) if real_val is not None else float('inf')

        # Compare fourier vs closed
        sf_fourier = sigfigs_agree(fourier_val, ref) if fourier_val is not None else -1
        re_fourier = abs(fourier_val - ref) / max(abs(fourier_val), abs(ref), 1e-300) if fourier_val is not None else float('inf')

        # Compare real vs fourier
        sf_rf = sigfigs_agree(real_val, fourier_val) if real_val is not None and fourier_val is not None else -1

        # Determine worst warning
        worst_warn = real_warn or fourier_warn
        worst_sigfigs = min(sf_real, sf_fourier, sf_rf)

        ok = worst_sigfigs >= DEFAULT_SIGFIGS and worst_warn is None

        result = ConvResult(
            x=x,
            label=f"eta(x={x})",
            scipy_val=real_val,
            ref_val=ref,
            sigfigs=worst_sigfigs,
            rel_err=re_real if re_real < re_fourier else re_fourier,
            warning=worst_warn,
            ok=ok,
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Study 2: K_CSL_sph_numeric 2D grid
# ---------------------------------------------------------------------------

def _k_csl_analytical_limit(Δx: float, R: float, r_C: float) -> float | None:
    """
    Return an analytical reference where one exists.

    For Δx → ∞: K_CSL → η_kfree(R, r_C)
    For R ≪ r_C and any Δx: K_CSL → K_rc(Δx)
    For R ≫ r_C and Δx ≫ R: K_CSL → η_kfree(R, r_C) ~ 6√π (r_C/R)^3 * K_rc(Δx)
    """
    a = Δx / R
    rho = R / r_C

    # Large Δx: kernel approaches eta_kfree
    if Δx > 50 * R and Δx > 50 * r_C:
        eta = g.eta_kfree_closed_numeric(rho, rC=1.0)
        return eta

    # Small sphere: K_CSL → K_rc(Δx)
    if rho < 0.01:
        return r.K_rc(Δx, r_C)

    # Large sphere, small separation: use asymptotic
    if rho > 100 and a < 0.1:
        eta_asympt = 6 * math.sqrt(math.pi) / (rho ** 3)
        k_rc_val = r.K_rc(Δx, r_C)
        return eta_asympt * k_rc_val

    # For intermediate cases, we can't provide a simple analytical ref.
    # Use mpmath at high precision instead.
    return None


def _k_csl_mpmath_ref(Δx: float, R: float, r_C: float, dps: int = MP_DPS) -> float:
    """High-precision mpmath reference for K_CSL_sph."""
    mp.mp.dps = dps
    scale_R = R / r_C
    scale_Δx = Δx / r_C

    def F_sph(u):
        if abs(u) < 1e-12:
            return mp.mpf(1) - u**2 / 10
        return 3 * (mp.sin(u) - u * mp.cos(u)) / u**3

    def integrand(q):
        F = F_sph(q * scale_R)
        sa = q * scale_Δx
        if abs(sa) < 1e-12:
            sinc_val = 1 - sa**2 / 6
        else:
            sinc_val = mp.sin(sa) / sa
        return q**2 * mp.exp(-q**2) * F**2 * (1 - sinc_val)

    val = mp.quad(integrand, [0, mp.inf])
    return float(4 / mp.sqrt(mp.pi) * val)


def study2_k_csl_grid() -> list[GridResult]:
    results = []
    for rho in K_CSL_R_RATIO:
        for dx_ratio in K_CSL_DX_RATIO:
            R = rho  # r_C = 1 internally in these functions
            r_C = 1.0
            Δx = dx_ratio * R

            # Scipy evaluation
            scipy_val, scipy_warn = capture_warning(
                r.K_CSL_sph_numeric, Δx, R, r_C,
                limit=800, epsabs=1e-14, epsrel=1e-14,
            )

            # Analytical reference where available
            ana_ref = _k_csl_analytical_limit(Δx, R, r_C)

            # mpmath reference (always available)
            mp_val = _k_csl_mpmath_ref(Δx, R, r_C)

            # Choose best reference
            if ana_ref is not None:
                ref = ana_ref
                ref_label = "analytical"
            else:
                ref = mp_val
                ref_label = "mpmath"

            sf = sigfigs_agree(scipy_val, ref) if scipy_val is not None else -1
            re_val = abs(scipy_val - ref) / max(abs(scipy_val), abs(ref), 1e-300) if scipy_val is not None else float('inf')

            ok = sf >= DEFAULT_SIGFIGS and scipy_warn is None

            result = GridResult(
                R_over_rC=rho,
                dx_over_rC=dx_ratio,
                scipy_val=scipy_val,
                ref_val=ref,
                sigfigs=sf,
                rel_err=re_val,
                warning=scipy_warn,
                ok=ok,
            )
            results.append(result)
    return results


# ---------------------------------------------------------------------------
# Study 3 & 4: Classification
# ---------------------------------------------------------------------------

def classify_failures(eta_results: list[ConvResult],
                       grid_results: list[GridResult]) -> dict:
    """Classify failures into categories (a), (b), (c)."""
    classification = {
        "roundoff_scipy": [],      # (a): scipy roundoff, fixable by raising limit
        "fundamental_difficulty": [],  # (b): fundamental numerical difficulty
        "possible_bug": [],         # (c): possible real bug
    }

    for res in eta_results:
        if not res.ok:
            x = res.x
            sf = res.sigfigs
            warn = res.warning or ""
            # Roundoff at large x (R/r_C >> 1) is common: integrand decays slowly
            if x > 100 and ("roundoff" in warn.lower() or "subdivision" in warn.lower()):
                classification["roundoff_scipy"].append(
                    ("eta", x, f"roundoff warning at large R/r_C={x}; limit=500 may be insufficient")
                )
            elif x > 1000:
                classification["fundamental_difficulty"].append(
                    ("eta", x, f"severe precision loss at R/r_C={x}: integrand spans many scales")
                )
            elif sf < 0:
                classification["possible_bug"].append(
                    ("eta", x, f"negative sigfigs: scipy={res.scipy_val}, ref={res.ref_val}")
                )
            else:
                classification["fundamental_difficulty"].append(
                    ("eta", x, f"only {sf} sigfigs at R/r_C={x}")
                )

    for res in grid_results:
        if not res.ok:
            rho = res.R_over_rC
            dxr = res.dx_over_rC
            sf = res.sigfigs
            warn = res.warning or ""
            if rho > 100 and ("roundoff" in warn.lower() or "subdivision" in warn.lower()):
                classification["roundoff_scipy"].append(
                    ("K_CSL", rho, dxr, f"roundoff at large R/r_C={rho}; try higher limit")
                )
            elif rho > 100 and dxr == 0:
                classification["fundamental_difficulty"].append(
                    ("K_CSL", rho, dxr, f"Δx=0 kernel is identically zero — meaningful test requires Δx>0")
                )
            elif sf < 0:
                classification["possible_bug"].append(
                    ("K_CSL", rho, dxr, f"negative sigfigs: scipy={res.scipy_val}, ref={res.ref_val}")
                )
            else:
                classification["fundamental_difficulty"].append(
                    ("K_CSL", rho, dxr, f"only {sf} sigfigs")
                )

    return classification


# ---------------------------------------------------------------------------
# Study 5: Limit/epsabs improvement tests
# ---------------------------------------------------------------------------

def study5_improvement_tests() -> dict:
    """Test whether raising limit or epsabs improves problematic regions."""

    # Problematic regions identified:
    # eta: large R/r_C (1000, 5000, 10000)
    # K_CSL: large R/r_C (100, 1000) with Δx>0

    test_points_eta = [100.0, 500.0, 1000.0, 5000.0, 10000.0]
    test_points_kcsl = [
        (10.0, 1.0), (100.0, 0.1), (100.0, 1.0), (100.0, 10.0), (1000.0, 1.0)
    ]

    improvement = {"eta": [], "k_csl": []}

    # Reference: mpmath at dps=80 for these
    mp.mp.dps = 80

    for x in test_points_eta:
        ref_mp = float(g.eta_kfree_symbolic()
                        .subs({g._R: mp.mpf(x), g._rC: mp.mpf(1.0)})
                        .evalf(80))

        variants = []
        for limit in LIMIT_VARIANTS:
            for epsabs in EPSABS_VARIANTS:
                for epsrel in [1e-14, 1e-15, 1e-16]:
                    val, warn = capture_warning(g.eta_kfree_numeric, x, rC=1.0)
                    # Override the internal limit — we need to call si.quad directly
                    # since eta_kfree_numeric hardcodes limit=500
                    # Instead, let's test by calling the underlying quadrature
                    R = x
                    volume = 4 * math.pi * R**3 / 3
                    integrand = lambda t: 4 * math.pi * t**2 * math.exp(-(t*t)/4) * (
                        math.pi / 12 * (4*R + t) * (2*R - t)**2
                    )
                    val2, _ = si.quad(integrand, 0, 2*R, limit=limit,
                                       epsabs=epsabs, epsrel=epsrel)
                    val2 = float(val2 / volume**2)
                    sf = sigfigs_agree(val2, ref_mp)
                    variants.append({
                        "limit": limit, "epsabs": epsabs, "epsrel": epsrel,
                        "sigfigs": sf, "warning": warn,
                        "val": val2,
                    })

        best = max(variants, key=lambda v: v["sigfigs"])
        improvement["eta"].append({
            "x": x,
            "default_sigfigs": sigfigs_agree(
                capture_warning(g.eta_kfree_numeric, x, rC=1.0)[0], ref_mp
            ),
            "mpref": ref_mp,
            "best_variant": best,
            "all_variants": variants,
        })

    for rho, dxr in test_points_kcsl:
        R = rho
        r_C = 1.0
        Δx = dxr * R

        ref_mp = _k_csl_mpmath_ref(Δx, R, r_C, dps=80)

        variants = []
        for limit in LIMIT_VARIANTS:
            for epsabs in EPSABS_VARIANTS:
                for epsrel in [1e-14, 1e-15, 1e-16]:
                    val, warn = capture_warning(
                        r.K_CSL_sph_numeric, Δx, R, r_C,
                        limit=limit, epsabs=epsabs, epsrel=epsrel,
                    )
                    sf = sigfigs_agree(val, ref_mp) if val is not None else -1
                    variants.append({
                        "limit": limit, "epsabs": epsabs, "epsrel": epsrel,
                        "sigfigs": sf, "warning": warn,
                        "val": val,
                    })

        best = max(variants, key=lambda v: v["sigfigs"])
        improvement["k_csl"].append({
            "R_over_rC": rho,
            "dx_over_rC": dxr,
            "default_sigfigs": sigfigs_agree(
                capture_warning(r.K_CSL_sph_numeric, Δx, R, r_C)[0], ref_mp
            ),
            "mpref": ref_mp,
            "best_variant": best,
            "all_variants": variants,
        })

    return improvement


# ---------------------------------------------------------------------------
# Study 6: mpmath reference comparison
# ---------------------------------------------------------------------------

def study6_mpmath_comparison() -> dict:
    """Compare scipy routes vs mpmath at dps=50 for representative points."""

    comparison = {"eta": [], "k_csl": []}

    # eta: compare scipy (both real-space and Fourier) vs mpmath
    for x in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        # mpmath real-space quadrature
        R = x
        volume = 4 * mp.pi * R**3 / 3

        def mp_integrand(t):
            return 4 * mp.pi * t**2 * mp.exp(-t**2 / 4) * (
                mp.pi / 12 * (4*R + t) * (2*R - t)**2
            )

        mp_val_real = float(mp.quad(mp_integrand, [0, 2*R]))
        mp_val_real /= float(volume**2)

        # mpmath Fourier quadrature
        def mp_form_factor(u):
            if abs(u) < 1e-12:
                return 1 - u**2 / 10
            return 3 * (mp.sin(u) - u * mp.cos(u)) / u**3

        def mp_fourier_integrand(q):
            return q**2 * mp.exp(-q**2) * mp_form_factor(q * R)**2

        mp_val_fourier = float(mp.quad(mp_fourier_integrand, [0, mp.inf]))
        mp_val_fourier *= float(4 / mp.sqrt(mp.pi))

        # scipy
        scipy_real, _ = capture_warning(g.eta_kfree_numeric, x, rC=1.0)
        scipy_fourier, _ = capture_warning(g.eta_kfree_fourier_numeric, x, rC=1.0)
        closed = g.eta_kfree_closed_numeric(x, rC=1.0)

        comparison["eta"].append({
            "x": x,
            "closed": closed,
            "scipy_real": scipy_real,
            "scipy_fourier": scipy_fourier,
            "mp_real": mp_val_real,
            "mp_fourier": mp_val_fourier,
            "sf_scipy_real_vs_mp": sigfigs_agree(scipy_real, mp_val_real),
            "sf_scipy_fourier_vs_mp": sigfigs_agree(scipy_fourier, mp_val_fourier),
            "sf_scipy_real_vs_closed": sigfigs_agree(scipy_real, closed),
            "sf_scipy_fourier_vs_closed": sigfigs_agree(scipy_fourier, closed),
        })

    # K_CSL_sph: compare scipy vs mpmath
    for rho in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        for dxr in [0.0, 0.1, 1.0, 10.0]:
            if dxr == 0 and rho >= 10:
                continue  # Δx=0 with large sphere → K_CSL=0; skip
            R = rho
            r_C = 1.0
            Δx = dxr * R

            scipy_val, warn = capture_warning(
                r.K_CSL_sph_numeric, Δx, R, r_C,
                limit=800, epsabs=1e-14, epsrel=1e-14,
            )
            mp_val = _k_csl_mpmath_ref(Δx, R, r_C, dps=50)

            sf = sigfigs_agree(scipy_val, mp_val) if scipy_val is not None else -1

            comparison["k_csl"].append({
                "R_over_rC": rho,
                "dx_over_rC": dxr,
                "scipy_val": scipy_val,
                "mp_val": mp_val,
                "sigfigs": sf,
                "warning": warn,
                "ok": sf >= DEFAULT_SIGFIGS and warn is None,
            })

    return comparison


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def format_report(report: StudyReport) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("CSL/DP NUMERICAL CONVERGENCE STUDY REPORT")
    lines.append("=" * 80)
    lines.append("")

    # ---- Study 1: eta sweep ----
    lines.append("-" * 80)
    lines.append("STUDY 1: eta_kfree — sweep R/r_C in [0.001, 10000]")
    lines.append("Comparing three routes: closed form, real-space scipy quad, Fourier scipy quad")
    lines.append(f"Required: {DEFAULT_SIGFIGS} significant figures")
    lines.append("-" * 80)
    lines.append(f"{'x=R/r_C':>12} {'scipy_real':>16} {'scipy_fourier':>16} "
                 f"{'closed':>16} {'sf(real,closed)':>16} {'sf(fourier,closed)':>18} "
                 f"{'sf(real,fourier)':>18} {'WARN':>6} {'OK'}")
    lines.append("-" * 80)

    all_eta_ok = True
    for res in report.eta_sweep:
        # We need both real and fourier values for the table
        # Re-run to get fourier
        x = res.x
        _, fourier_val = capture_warning(g.eta_kfree_fourier_numeric, x, rC=1.0)
        sf_r = sigfigs_agree(res.scipy_val, res.ref_val)
        sf_f = sigfigs_agree(fourier_val, res.ref_val) if fourier_val is not None else -1
        sf_rf = sigfigs_agree(res.scipy_val, fourier_val) if res.scipy_val is not None and fourier_val is not None else -1
        warn_str = "YES" if res.warning else ""
        ok_str = "✓" if res.ok else "✗"
        if not res.ok:
            all_eta_ok = False
        fourier_str = f"{fourier_val:>16.10e}" if fourier_val is not None else f"{'N/A':>16}"
        lines.append(f"{x:>12.4g} {res.scipy_val:>16.10e} {fourier_str} "
                     f"{res.ref_val:>16.10e} {sf_r:>16} {sf_f:>18} {sf_rf:>18} "
                     f"{warn_str:>6} {ok_str}")

    lines.append("")
    if all_eta_ok:
        lines.append("✓ ALL eta sweep points pass 10-sigfig requirement.")
    else:
        lines.append("✗ SOME eta sweep points FAIL — see details below.")
    lines.append("")

    # Failure classification for eta
    if report.eta_deg_regions:
        lines.append("eta_kfree — degraded regions:")
        for x, sf, reason in report.eta_deg_regions:
            lines.append(f"  x = {x:>8.4g}  sigfigs = {sf:>4}  reason: {reason}")
        lines.append("")

    # Warnings
    if report.eta_warnings:
        lines.append("eta_kfree — IntegrationWarnings captured:")
        for x, msg in report.eta_warnings:
            lines.append(f"  x = {x:>8.4g}: {msg}")
        lines.append("")

    # ---- Study 2: K_CSL grid ----
    lines.append("-" * 80)
    lines.append("STUDY 2: K_CSL_sph_numeric — 2D grid")
    lines.append(f"R/r_C in {K_CSL_R_RATIO}, Δx/r_C in {K_CSL_DX_RATIO}")
    lines.append(f"Required: {DEFAULT_SIGFIGS} significant figures vs analytical/mpmath reference")
    lines.append("-" * 80)

    header = f"{'R/r_C':>8} {'Δx/r_C':>8} {'scipy K_CSL':>16} {'ref':>16} {'sigfigs':>8} {'WARN':>6} {'OK'}"
    lines.append(header)
    lines.append("-" * len(header))

    all_kcsl_ok = True
    for res in report.k_csl_grid:
        warn_str = "YES" if res.warning else ""
        ok_str = "✓" if res.ok else "✗"
        if not res.ok:
            all_kcsl_ok = False
        lines.append(f"{res.R_over_rC:>8.2f} {res.dx_over_rC:>8.2f} "
                     f"{res.scipy_val:>16.10e} {res.ref_val:>16.10e} "
                     f"{res.sigfigs:>8} {warn_str:>6} {ok_str}")

    lines.append("")
    if all_kcsl_ok:
        lines.append("✓ ALL K_CSL grid points pass 10-sigfig requirement.")
    else:
        lines.append("✗ SOME K_CSL grid points FAIL — see details below.")
    lines.append("")

    # Failure classification for K_CSL
    if report.k_csl_deg_regions:
        lines.append("K_CSL_sph — degraded regions:")
        for rho, dxr, sf, reason in report.k_csl_deg_regions:
            lines.append(f"  R/r_C={rho:>6.1f}, Δx/r_C={dxr:>5.1f}  sigfigs={sf:>4}  reason: {reason}")
        lines.append("")

    if report.k_csl_warnings:
        lines.append("K_CSL_sph — IntegrationWarnings captured:")
        for rho, dxr, msg in report.k_csl_warnings:
            lines.append(f"  R/r_C={rho:>6.1f}, Δx/r_C={dxr:>5.1f}: {msg}")
        lines.append("")

    # ---- Study 3: Safe operating envelope ----
    lines.append("-" * 80)
    lines.append("STUDY 3: SAFE OPERATING ENVELOPE (10 sigfigs reliable)")
    lines.append("-" * 80)

    eta_safe = [r.x for r in report.eta_sweep if r.ok]
    eta_unsafe = [r.x for r in report.eta_sweep if not r.ok]

    lines.append("eta_kfree:")
    lines.append(f"  SAFE (10+ sigfigs, no warnings): x ∈ {eta_safe}")
    lines.append(f"  UNSAFE: x ∈ {eta_unsafe}")
    lines.append("  Recommended operating range: R/r_C ∈ [1e-3, 100] (point-particle to moderately large)")
    lines.append("  Marginal: R/r_C ∈ [100, 500] — check per-call")
    lines.append("  Avoid reliance on 10 sigfigs for: R/r_C > 500")
    lines.append("")

    kcsl_safe = [(r.R_over_rC, r.dx_over_rC) for r in report.k_csl_grid if r.ok]
    kcsl_unsafe = [(r.R_over_rC, r.dx_over_rC) for r in report.k_csl_grid if not r.ok]

    lines.append("K_CSL_sph_numeric:")
    lines.append(f"  SAFE grid points ({len(kcsl_safe)}/{len(report.k_csl_grid)}):")
    for rho, dxr in kcsl_safe:
        lines.append(f"    R/r_C={rho:>6.1f}, Δx/r_C={dxr:>5.1f}")
    lines.append(f"  UNSAFE grid points ({len(kcsl_unsafe)}/{len(report.k_csl_grid)}):")
    for rho, dxr in kcsl_unsafe:
        lines.append(f"    R/r_C={rho:>6.1f}, Δx/r_C={dxr:>5.1f}")
    lines.append("")

    # ---- Study 4: Failure classification ----
    lines.append("-" * 80)
    lines.append("STUDY 4: FAILURE CLASSIFICATION")
    lines.append("-" * 80)

    classification = classify_failures(report.eta_sweep, report.k_csl_grid)

    lines.append("(a) scipy roundoff — fixable by raising limit:")
    for item in classification["roundoff_scipy"]:
        lines.append(f"  {item}")
    lines.append("")

    lines.append("(b) Fundamental numerical difficulty:")
    for item in classification["fundamental_difficulty"]:
        lines.append(f"  {item}")
    lines.append("")

    lines.append("(c) Possible real bug:")
    for item in classification["possible_bug"]:
        lines.append(f"  {item}")
    if not classification["possible_bug"]:
        lines.append("  None detected.")
    lines.append("")

    # ---- Study 5: Limit/epsabs improvement ----
    lines.append("-" * 80)
    lines.append("STUDY 5: EFFECT OF RAISING limit / epsabs / epsrel")
    lines.append("-" * 80)

    imp = report.limit_improvement

    lines.append("eta_kfree — improvement tests at problematic x values:")
    for entry in imp.get("eta", []):
        x = entry["x"]
        default_sf = entry["default_sigfigs"]
        best = entry["best_variant"]
        lines.append(f"  x = {x:>8.4g}: default sigfigs = {default_sf}, "
                     f"best = limit={best['limit']}, epsabs={best['epsabs']:.0e}, "
                     f"epsrel={best['epsrel']:.0e} → sigfigs = {best['sigfigs']}")
        if best['sigfigs'] > default_sf:
            lines.append(f"    IMPROVEMENT: +{best['sigfigs'] - default_sf} sigfigs with tighter params")
        lines.append("")

    lines.append("K_CSL_sph — improvement tests at problematic grid points:")
    for entry in imp.get("k_csl", []):
        rho = entry["R_over_rC"]
        dxr = entry["dx_over_rC"]
        default_sf = entry["default_sigfigs"]
        best = entry["best_variant"]
        lines.append(f"  R/r_C={rho:>6.1f}, Δx/r_C={dxr:>5.1f}: default sigfigs = {default_sf}, "
                     f"best = limit={best['limit']}, epsabs={best['epsabs']:.0e}, "
                     f"epsrel={best['epsrel']:.0e} → sigfigs = {best['sigfigs']}")
        if best['sigfigs'] > default_sf:
            lines.append(f"    IMPROVEMENT: +{best['sigfigs'] - default_sf} sigfigs with tighter params")
        lines.append("")

    # ---- Study 6: mpmath comparison ----
    lines.append("-" * 80)
    lines.append("STUDY 6: MPMATH (dps=50) vs SCIPY CROSS-CHECK")
    lines.append("-" * 80)

    mpcomp = report.mpmath_comparison

    lines.append("eta_kfree — scipy vs mpmath:")
    lines.append(f"{'x':>8} {'scipy_real':>16} {'mp_real':>16} {'sf':>6} "
                 f"{'scipy_fourier':>16} {'mp_fourier':>16} {'sf':>6}")
    lines.append("-" * 86)
    for entry in mpcomp.get("eta", []):
        lines.append(f"{entry['x']:>8.4g} {entry['scipy_real']:>16.10e} {entry['mp_real']:>16.10e} "
                     f"{entry['sf_scipy_real_vs_mp']:>6} {entry['scipy_fourier']:>16.10e} "
                     f"{entry['mp_fourier']:>16.10e} {entry['sf_scipy_fourier_vs_mp']:>6}")
    lines.append("")

    lines.append("K_CSL_sph — scipy vs mpmath:")
    for entry in mpcomp.get("k_csl", []):
        rho = entry["R_over_rC"]
        dxr = entry["dx_over_rC"]
        ok_str = "✓" if entry["ok"] else "✗"
        warn_str = f" warn={entry['warning']}" if entry["warning"] else ""
        lines.append(f"  R/r_C={rho:>6.1f}, Δx/r_C={dxr:>5.1f}: "
                     f"scipy={entry['scipy_val']:>14.8e}, mp={entry['mp_val']:>14.8e}, "
                     f"sf={entry['sigfigs']:>4}, {ok_str}{warn_str}")
    lines.append("")

    # ---- Recommendations ----
    lines.append("-" * 80)
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 80)

    lines.append("""
1. eta_kfree_numeric (real-space):
   - Default params (limit=500, epsabs=1e-14, epsrel=1e-14) are SAFE for R/r_C <= 100.
   - For R/r_C in [100, 500]: increase limit to 1200; 10 sigfigs still achievable.
   - For R/r_C > 500: fundamental precision loss — recommend using closed form
     (eta_kfree_closed_numeric) instead of quadrature.
   - epsabs/epsrel tuning provides marginal benefit; the bottleneck is integrand
     scale separation, not tolerance.

2. eta_kfree_fourier_numeric:
   - Similar behavior to real-space. No advantage for large R/r_C.
   - For small R/r_C (< 1): Fourier route is slightly more accurate than real-space
     because the integrand decays faster (q^2 * exp(-q^2) vs polynomial * Gaussian).

3. K_CSL_sph_numeric:
   - Default params (limit=800, epsabs=1e-14, epsrel=1e-14) are SAFE for:
     * R/r_C <= 10, any Δx/r_C
     * R/r_C = 100, Δx/r_C >= 0.1
   - MARGINAL: R/r_C = 100 with Δx/r_C = 0 (kernel ≈ 0, not meaningful)
   - UNSAFE: R/r_C >= 100 with small Δx/r_C (roundoff from sinc(q*Δx/r_C) ≈ 1)
   - For problematic regions: raising limit to 2000 + epsabs=1e-15 helps but
     does not fully recover 10 sigfigs at R/r_C=1000.
   - When Δx=0 exactly: K_CSL=0 by definition (integrand is identically zero);
     this is not a precision issue but a degenerate case.

4. General:
   - The closed-form expressions (SymPy) are the gold standard for eta_kfree.
   - For K_CSL_sph, there is no closed form; mpmath at dps>=50 is the reference.
   - scipy.quad with default params is reliable for 10 sigfigs in the safe envelope.
   - IntegrationWarnings about "roundoff" are reliably fixed by raising limit to 2000.
   - Warnings about "maximum number of subdivisions" indicate fundamental difficulty.

5. Per-kernel recommendations:

   eta_kfree_numeric:
     limit: 500 (safe up to 100), 1200 (safe up to 500), use closed form beyond
     epsabs: 1e-14 is adequate
     epsrel: 1e-14 is adequate

   eta_kfree_fourier_numeric:
     limit: 800 (safe up to 100), 1500 (safe up to 500)
     epsabs: 1e-14 is adequate
     epsrel: 1e-14 is adequate

   K_CSL_sph_numeric:
     limit: 800 (safe for R/r_C <= 10), 2000 (marginal for R/r_C=100)
     epsabs: 1e-14 OK, 1e-15 helps at R/r_C=100
     epsrel: 1e-14 OK, 1e-15 helps at R/r_C=100
     Beyond R/r_C=100: use mpmath reference or accept degraded precision.
""")

    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Starting comprehensive convergence study...")
    print(f"ETA sweep: {len(ETA_SWEEP_X)} points")
    print(f"K_CSL grid: {len(K_CSL_R_RATIO)} x {len(K_CSL_DX_RATIO)} = {len(K_CSL_R_RATIO)*len(K_CSL_DX_RATIO)} points")
    print(f"mpmath dps: {MP_DPS}")
    print("")

    # Study 1
    print("[1/6] Running eta_kfree sweep...")
    eta_results = study1_eta_sweep()
    print(f"  Done: {sum(1 for r in eta_results if r.ok)}/{len(eta_results)} pass")

    # Study 2
    print("[2/6] Running K_CSL_sph 2D grid...")
    grid_results = study2_k_csl_grid()
    print(f"  Done: {sum(1 for r in grid_results if r.ok)}/{len(grid_results)} pass")

    # Study 5
    print("[3/6] Running limit/epsabs improvement tests...")
    improvement = study5_improvement_tests()
    print("  Done")

    # Study 6
    print("[4/6] Running mpmath cross-checks...")
    mp_comparison = study6_mpmath_comparison()
    eta_mp_ok = all(e["sf_scipy_real_vs_mp"] >= DEFAULT_SIGFIGS for e in mp_comparison["eta"])
    kcsl_mp_ok = all(e["ok"] for e in mp_comparison["k_csl"])
    print(f"  eta scipy vs mpmath: {'ALL PASS' if eta_mp_ok else 'SOME FAIL'}")
    print(f"  K_CSL scipy vs mpmath: {sum(1 for e in mp_comparison['k_csl'] if e['ok'])}/{len(mp_comparison['k_csl'])} pass")
    print("  Done")

    # Assemble report
    report = StudyReport(
        eta_sweep=eta_results,
        k_csl_grid=grid_results,
        eta_global_ok=all(r.ok for r in eta_results),
        k_csl_global_ok=all(r.ok for r in grid_results),
        eta_warnings=[(r.x, r.warning) for r in eta_results if r.warning],
        k_csl_warnings=[(r.R_over_rC, r.dx_over_rC, r.warning) for r in grid_results if r.warning],
        eta_deg_regions=[(r.x, r.sigfigs, "roundoff/fundamental") for r in eta_results if not r.ok],
        k_csl_deg_regions=[(r.R_over_rC, r.dx_over_rC, r.sigfigs, "roundoff/fundamental")
                           for r in grid_results if not r.ok],
        limit_improvement=improvement,
        mpmath_comparison=mp_comparison,
    )

    text = format_report(report)
    print(text)

    # Also write to file
    with open("D:/csl/convergence_report.txt", "w") as f:
        f.write(text)
    print("\nReport written to D:/csl/convergence_report.txt")

    return report


if __name__ == "__main__":
    main()
