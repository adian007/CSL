"""Bayesian inference of CSL parameters from force-noise PSD data.

Implements MCMC sampling with PyMC to estimate log10(λ_CSL) and log10(r_C)
from synthetic or experimental force-noise power spectral density measurements.

The likelihood compares the CSL + thermal PSD against data:

    S_FF(ω) = 2 M γ k_B T + S_CSL(ω; λ, r_C) + noise_floor

where S_CSL is the white CSL contribution:
    S_CSL = (3 ħ² λ M) / (2 m0² r_C²)

The prior on λ is log-uniform over [10⁻²⁰, 10⁻¹⁴] s⁻¹.
The prior on r_C is log-uniform over [10⁻⁹, 10⁻⁵] m.

Convergence is assessed via the Gelman-Rubin R-hat statistic (R_GR < 1.05).

References
----------
Carlesso et al. Phys. Rep. 572, 1 (2015) [review of bounds].
Vinante et al. Phys. Rev. Lett. 119, 110401 (2017) [cantilever bounds].
Wittweg et al. Phys. Rev. Lett. 126, 053601 (2021) [levitated bounds].
"""

from __future__ import annotations

from typing import Literal

import numpy as np

#: Fallback reference to the collapse-simulation layer (SI constants and PSD helpers).
#: Imported lazily inside the functions that need it so that this module can be
#: imported without qutip when only the non-simulation helpers are used.
import csl_pipeline.collapse_sim as cs

#: Whether PyMC is available in this environment.
try:
    import pymc as pm
    HAS_PYMCI = True
except ImportError:
    HAS_PYMCI = False


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_psd(
    omega: np.ndarray,
    M: float,
    gamma: float,
    T: float,
    lambda_csl_true: float,
    r_C_true: float,
    noise_sigma: float = 0.0,
    seed: int = 42,
) -> dict:
    """Generate synthetic force-noise PSD data with known CSL parameters.

    Parameters
    ----------
    omega:
        Frequency array in rad/s.
    M:
        Test mass in kg.
    gamma:
        Mechanical damping in s⁻¹.
    T:
        Bath temperature in K.
    lambda_csl_true:
        True CSL rate in s⁻¹.
    r_C_true:
        True CSL localisation length in m.
    noise_sigma:
        Standard deviation of measurement noise (added in quadrature).
    seed:
        RNG seed.

    Returns
    -------
    dict with keys:
        omega:     frequency array
        S_FF_true: true PSD (no noise)
        S_FF_data: observed PSD (with noise)
        S_FF_noise: noise realisation
        lambda_true: lambda_csl_true
        r_C_true:   r_C_true
    """
    rng = np.random.default_rng(seed)
    s_csl = cs.csl_force_psd(lambda_csl_true, r_C_true, M,
                              omega=omega, gamma=gamma, T=T)
    s_th = 2.0 * M * gamma * cs.K_B * T * np.ones_like(omega)
    S_true = s_csl + s_th
    if noise_sigma > 0:
        noise = rng.normal(0, noise_sigma, size=omega.shape)
        S_data = S_true + noise
    else:
        noise = np.zeros_like(omega)
        S_data = S_true.copy()
    return {
        "omega": omega,
        "S_FF_true": S_true,
        "S_FF_data": S_data,
        "S_FF_noise": noise,
        "lambda_true": lambda_csl_true,
        "r_C_true": r_C_true,
        "M": M,
        "gamma": gamma,
        "T": T,
    }


# ---------------------------------------------------------------------------
# PyMC model
# ---------------------------------------------------------------------------

def build_csl_psd_model(
    omega: np.ndarray,
    S_data: np.ndarray,
    M: float,
    gamma: float,
    T: float,
    m0: float = cs.AMU,
    log10_lambda_prior: tuple[float, float] = (-20.0, -14.0),
    log10_rC_prior: tuple[float, float] = (-9.0, -5.0),
    noise_sigma: float | None = None,
    noise_sigma_prior: tuple[float, float] = (1e-20, 1e-10),
) -> pm.Model:
    """Build a PyMC model for CSL parameter estimation from PSD data.

    Parameters
    ----------
    omega:
        Frequency array (rad/s).
    S_data:
        Observed PSD data (kg² m² s⁻³).
    M, gamma, T:
        Known instrumental parameters.
    m0:
        Reference mass (default: atomic mass unit, NHH convention).
    log10_lambda_prior:
        (low, high) for log10(λ_CSL) Uniform prior.
    log10_rC_prior:
        (low, high) for log10(r_C) Uniform prior.
    noise_sigma:
        Fixed measurement noise std.  If None, it is inferred.
    noise_sigma_prior:
        (low, high) for HalfNormal noise sigma prior when inferred.

    Returns
    -------
    pymc.Model ready for sampling.
    """
    if not HAS_PYMCI:
        raise ImportError(
            "PyMC is required for build_csl_psd_model but is not installed. "
            "Install it with: pip install pymc"
        )
    import pymc as pm

    with pm.Model() as model:
        # Priors on log10 of CSL parameters
        log10_lambda = pm.Uniform(
            "log10_lambda",
            lower=log10_lambda_prior[0],
            upper=log10_lambda_prior[1],
        )
        log10_rC = pm.Uniform(
            "log10_rC",
            lower=log10_rC_prior[0],
            upper=log10_rC_prior[1],
        )

        # Transform to linear scale
        lambda_csl = pm.Deterministic("lambda_csl", 10.0**log10_lambda)
        r_C = pm.Deterministic("r_C", 10.0**log10_rC)

        # CSL PSD contribution (deterministic, given parameters)
        S_CSL = pm.Deterministic(
            "S_CSL",
            cs.csl_force_psd(lambda_csl, r_C, M, omega=omega,
                              gamma=gamma, T=T),
        )

        # Thermal background (known)
        S_th = pm.Deterministic(
            "S_th",
            2.0 * M * gamma * cs.K_B * T * np.ones_like(omega),
        )

        # Noise model
        if noise_sigma is not None:
            sigma = noise_sigma
        else:
            sigma = pm.HalfNormal("noise_sigma", sigma=noise_sigma_prior[1])

        # Likelihood
        S_total = pm.Deterministic("S_total", S_CSL + S_th)
        pm.Normal(
            "S_obs",
            mu=S_total,
            sigma=sigma,
            observed=S_data,
        )

    return model


# ---------------------------------------------------------------------------
# Gelman-Rubin R-hat convergence diagnostic
# ---------------------------------------------------------------------------

def gelman_rubin(
    trace: dict[str, np.ndarray],
    var_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute the Gelman-Rubin R-hat statistic for each variable in the trace.

    R_hat = sqrt((N-1)/N · W + B/N) / sqrt(W)

    where W is the within-chain variance and B is the between-chain variance.

    Parameters
    ----------
    trace:
        Dict of variable name → array of shape (n_chains, n_samples).
    var_names:
        Variables to include.  If None, all are used.

    Returns
    -------
    dict: var_name → R_hat value.  R_hat < 1.05 indicates convergence.
    """
    if var_names is None:
        var_names = list(trace.keys())

    results = {}
    for name in var_names:
        samples = trace[name]  # (n_chains, n_samples)
        n_chains, n_samples = samples.shape

        # Chain means
        chain_means = np.mean(samples, axis=1)  # (n_chains,)
        # Overall mean
        overall_mean = np.mean(chain_means)
        # Between-chain variance
        B = n_samples / (n_chains - 1) * np.sum((chain_means - overall_mean) ** 2)
        # Within-chain variance
        W = np.mean([np.var(samples[c]) for c in range(n_chains)])

        if W == 0:
            R_hat = float("inf")
        else:
            R_hat = np.sqrt((n_samples - 1) / n_samples + B / (n_samples * W))

        results[name] = float(R_hat)

    return results


# ---------------------------------------------------------------------------
# Run MCMC
# ---------------------------------------------------------------------------

def run_csl_inference(
    omega: np.ndarray,
    S_data: np.ndarray,
    M: float,
    gamma: float,
    T: float,
    n_samples: int = 2000,
    n_tune: int = 1000,
    n_chains: int = 4,
    seed: int = 42,
    **model_kwargs,
) -> dict:
    """Run full CSL Bayesian inference on PSD data.

    Parameters
    ----------
    omega, S_data, M, gamma, T:
        Data and known parameters (see generate_synthetic_psd).
    n_samples, n_tune, n_chains:
        PyMC sampling parameters.
    seed:
        RNG seed.

    Returns
    -------
    dict with keys:
        model:           pymc.Model
        trace:           pymc trace (ArviZ format)
        summary:         summary DataFrame
        r_hat:           dict of var_name → R_hat
        lambda_posterior: 1D array of lambda samples (flattened chains)
        rC_posterior:      1D array of r_C samples
        omega:            input frequency array
        S_data:           input data
    """
    if not HAS_PYMCI:
        raise ImportError(
            "PyMC is required for run_csl_inference but is not installed. "
            "Install it with: pip install pymc"
        )
    import pymc as pm

    model = build_csl_psd_model(omega, S_data, M, gamma, T, **model_kwargs)

    with model:
        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            random_seed=seed,
            progressbar=False,
            return_inferencedata=True,
        )

    # Extract samples as numpy arrays
    lambda_samples = trace.posterior["lambda_csl"].values.flatten()
    rC_samples = trace.posterior["r_C"].values.flatten()

    # Build trace dict for R-hat
    trace_dict = {
        "log10_lambda": trace.posterior["log10_lambda"].values,
        "log10_rC": trace.posterior["log10_rC"].values,
        "lambda_csl": trace.posterior["lambda_csl"].values,
        "r_C": trace.posterior["r_C"].values,
    }

    r_hat = gelman_rubin(trace_dict)

    # Summary via ArviZ
    try:
        import arviz as az
        summary = az.summary(trace, var_names=["log10_lambda", "log10_rC",
                                                "lambda_csl", "r_C"])
    except ImportError:
        summary = None

    return {
        "model": model,
        "trace": trace,
        "summary": summary,
        "r_hat": r_hat,
        "lambda_posterior": lambda_samples,
        "rC_posterior": rC_samples,
        "log10_lambda_posterior": trace_dict["log10_lambda"].flatten(),
        "log10_rC_posterior": trace_dict["log10_rC"].flatten(),
        "omega": omega,
        "S_data": S_data,
        "n_samples": n_samples,
        "n_chains": n_chains,
    }


# ---------------------------------------------------------------------------
# Exclusion contour generation
# ---------------------------------------------------------------------------

def exclusion_contour_2d(
    omega: np.ndarray,
    S_data: np.ndarray,
    S_error: np.ndarray,
    M: float,
    gamma: float,
    T: float,
    lambda_grid: np.ndarray,
    rC_grid: np.ndarray,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a 2D exclusion contour in the (r_C, λ_CSL) plane.

    For each (r_C, λ) pair, compute the χ² relative to the data and
    determine the confidence-level exclusion boundary.

    Parameters
    ----------
    omega:
        Frequency array.
    S_data:
        Observed PSD.
    S_error:
        Measurement uncertainty at each frequency.
    M, gamma, T:
        Instrument parameters.
    lambda_grid:
        1D array of λ values to scan.
    rC_grid:
        1D array of r_C values to scan.
    confidence:
        Confidence level for the contour (0.95 → 2σ, 0.997 → 3σ).

    Returns
    -------
    tuple (lambda_grid, rC_grid, chi2_grid) where chi2_grid[i,j] is the
    χ² for (lambda_grid[i], rC_grid[j]).
    """
    # χ² = Σ (S_data - S_model)² / S_error²
    chi2 = np.zeros((len(rC_grid), len(lambda_grid)))

    for i, r_C in enumerate(rC_grid):
        for j, lam in enumerate(lambda_grid):
            S_model = cs.csl_force_psd(lam, r_C, M, omega=omega, gamma=gamma, T=T)
            S_th = 2.0 * M * gamma * cs.K_B * T * np.ones_like(omega)
            residual = S_data - S_model - S_th
            chi2[i, j] = np.sum(residual**2 / S_error**2)

    return lambda_grid, rC_grid, chi2


def contour_level(chi2_grid: np.ndarray, confidence: float = 0.95) -> float:
    """Return the χ² threshold for the given confidence level.

    For 2 degrees of freedom (λ, r_C), the χ² threshold at CL=0.95 is 5.99.
    """
    from scipy.stats import chi2 as chi2_dist
    dof = 2  # two parameters
    threshold = chi2_dist.ppf(confidence, dof)
    return threshold


# ---------------------------------------------------------------------------
# Convenience: full inference pipeline
# ---------------------------------------------------------------------------

def csl_inference_pipeline(
    M: float = 1e-17,
    gamma: float = 1e-3,
    T: float = 1.0,
    lambda_true: float = 1e-17,
    r_C_true: float = 1e-7,
    n_freq: int = 30,
    omega_min: float = 2e5,
    omega_max: float = 2e7,
    noise_level: float = 1e-24,
    seed: int = 42,
    n_samples: int = 2000,
    n_chains: int = 4,
) -> dict:
    """Run a complete CSL parameter estimation pipeline.

    1. Generate synthetic PSD data with known (λ, r_C).
    2. Run MCMC to recover posteriors.
    3. Check convergence (R-hat).
    4. Return everything in one dict.

    Returns
    -------
    dict with all pipeline results.

    Raises
    ------
    ImportError
        If PyMC is not installed.
    """
    if not HAS_PYMCI:
        raise ImportError(
            "PyMC is required for csl_inference_pipeline but is not installed. "
            "Install it with: pip install pymc"
        )
    import pymc as pm  # noqa: F811 — local import for clarity

    omega = np.linspace(omega_min, omega_max, n_freq)
    data = generate_synthetic_psd(
        omega=omega,
        M=M,
        gamma=gamma,
        T=T,
        lambda_csl_true=lambda_true,
        r_C_true=r_C_true,
        noise_sigma=noise_level,
        seed=seed,
    )

    result = run_csl_inference(
        omega=data["omega"],
        S_data=data["S_FF_data"],
        M=M,
        gamma=gamma,
        T=T,
        n_samples=n_samples,
        n_chains=n_chains,
        seed=seed + 1,
    )

    # Add truth and data to result
    result["data"] = data
    result["lambda_true"] = lambda_true
    result["r_C_true"] = r_C_true

    # Recovery diagnostics
    lam_median = np.median(result["lambda_posterior"])
    lam_true = lambda_true
    lam_recovery_factor = lam_median / lam_true if lam_true > 0 else float("inf")
    rC_median = np.median(result["rC_posterior"])
    rC_recovery_factor = rC_median / r_C_true if r_C_true > 0 else float("inf")

    result["recovery"] = {
        "lambda_median": lam_median,
        "lambda_true": lam_true,
        "lambda_recovery_factor": lam_recovery_factor,
        "rC_median": rC_median,
        "rC_true": r_C_true,
        "rC_recovery_factor": rC_recovery_factor,
        "R_hat_ok": all(v < 1.05 for v in result["r_hat"].values()),
    }

    return result


__all__ = [
    "generate_synthetic_psd",
    "build_csl_psd_model",
    "gelman_rubin",
    "run_csl_inference",
    "exclusion_contour_2d",
    "contour_level",
    "csl_inference_pipeline",
]
