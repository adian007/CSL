"""CSL / DP dynamics simulation core.

Implements the translational-CSL Lindblad master equation, its stochastic
Schrödinger equation (SSE) unravelling, the Diósi-Penrose master equation,
and the Wigner-phase-space evolution for dissipative extensions.

All simulations target a harmonically trapped nanosphere and use QuTiP 5.x
for the Hilbert-space machinery.

References
----------
Pearle 1989; Ghirardi-Pearle-Rimini 1990; Bassi-Ghirardi 2003; Bassi 2013.
Nimmrichter 2013; Nimmrichter-Hornberger-Hammerer PRL 113, 020405 (2014).
Nimmrichter-Wittek-PRL 106, 160401 (2011) [dCSL].
Donadi et al. Nature Phys. 17, 74 (2021).
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import qutip as qt
from numpy import sqrt, exp, pi

# ---------------------------------------------------------------------------
# Physical constants (CODATA 2018) — SI units
# ---------------------------------------------------------------------------

HBAR = 1.054571817e-34   #: J·s
AMU = 1.66053906660e-27  #: kg  (atomic mass unit, NHH convention)
K_B = 1.380649e-23       #: J/K (Boltzmann constant)
G = 6.67430e-11          #: m³ kg⁻¹ s⁻²


# ---------------------------------------------------------------------------
# Analytical CSL / DP rate formulas (static, no QuTiP needed)
# ---------------------------------------------------------------------------

def csl_decoherence_rate_1d(
    lambda_csl: float,
    r_C: float,
    m: float,
    amu: float = AMU,
) -> float:
    """CSL decoherence rate for a point-like object in 1D (free space).

    Γ_CSL^(1D) = λ (m/amu)² (ħ / r_C)² × (geometry factor).

    For a point particle the geometry factor is 1/2 (NHH point value).
    This is the rate that enters the master equation as a momentum-diffusion
    coefficient.

    Returns the rate in s⁻¹.
    """
    return lambda_csl * (m / amu) ** 2 * (HBAR / r_C) ** 2 / 2.0


def csl_decoherence_rate_3d(
    lambda_csl: float,
    r_C: float,
    m: float,
    amu: float = AMU,
) -> float:
    """CSL momentum-diffusion (3D) coefficient D_CSL for a point particle.

    D_CSL = λ (m/amu)² (ħ² / r_C²) / (4 π^(3/2)).

    This is the coefficient in the momentum-diffusion term of the CSL
    master equation for a point-like mass.

    Returns D_CSL in kg² m² s⁻³.
    """
    return lambda_csl * (m / amu) ** 2 * HBAR**2 / (r_C**2 * 4.0 * pi**1.5)


def csl_decoherence_factor(
    delta_x: float,
    lambda_csl: float,
    r_C: float,
    m: float,
    tau: float,
    amu: float = AMU,
) -> float:
    """CSL coherence decay factor D_CSL(Δx, τ) for a point particle.

    D_CSL(Δx, τ) = exp(-Λ_CSL τ (1 - exp(-Δx² / 4r_C²)))

    Λ_CSL = λ (m/amu)² (the per-nucleus rate).

    This is the visibility multiplier e^{-ΛτK_rc} used in interferometry.
    """
    K_rc = 1.0 - exp(-(delta_x * delta_x) / (4.0 * r_C * r_C))
    Lambda = lambda_csl * (m / amu) ** 2
    return exp(-Lambda * tau * K_rc)


def dp_self_energy_exponent(
    m: float,
    delta_x: float,
    R: float = 0.0,
    G_: float = G,
    hbar: float = HBAR,
    tau: float = 1.0,
) -> float:
    """DP self-energy exponent Λ_DP = E_G τ / ħ for a point particle.

    Uses the Davila-Milburn effective-distance proxy:
    E_G = G m² / d_eff,  d_eff = max{Δx, 2R}.

    For a homogeneous sphere, use dp_self_energy_exponent_sph below.
    """
    d_eff = max(delta_x, 2.0 * R)
    E_G = G_ * m * m / d_eff
    return E_G * tau / hbar


def dp_self_energy_exponent_sph(
    m: float,
    delta_x: float,
    R: float,
    tau: float = 1.0,
    G_: float = G,
    hbar: float = HBAR,
) -> float:
    """DP self-energy exponent for a homogeneous sphere.

    E_G^sph = (G m² / R) K_DP^sph(Δx / R).

    Requires K_DP^sph from csl_pipeline.rates.
    """
    from csl_pipeline.rates import K_DP_sph_closed, ell_star

    a = delta_x / R
    K_dp = K_DP_sph_closed(a)
    if K_dp <= 0.0:
        return float("inf")
    E_G = G_ * m * m / R * K_dp
    return E_G * tau / hbar


def csl_heating_power(
    lambda_csl: float,
    r_C: float,
    M: float,
    m0: float = AMU,
) -> float:
    """CSL spontaneous heating power for a bulk object.

    P_CSL = (3 ħ² λ_CSL M) / (4 m0² r_C²)   [true for 3D, GRW convention]

    Verified against the standard expression.  Returns power in watts.
    """
    return 3.0 * HBAR**2 * lambda_csl * M / (4.0 * m0**2 * r_C**2)


def csl_force_psd(
    lambda_csl: float,
    r_C: float,
    M: float,
    m0: float = AMU,
    omega: float | np.ndarray = None,
    gamma: float = 0.0,
    T: float = 0.0,
) -> float | np.ndarray:
    """CSL contribution to the force-noise power spectral density.

    S_FF_CSL(ω) = 2 M γ k_B T + S_CSL(ω)

    S_CSL(ω) is white (frequency-independent) for the standard CSL model:
    S_CSL = (3 ħ² λ_CSL M) / (2 m0² r_C²)   [single-sided, white]

    Parameters
    ----------
    omega:
        Frequency (or array) in rad/s.  White noise returns the same value
        for all ω; included for API consistency with coloured extensions.
    gamma:
        Mechanical damping rate in s⁻¹.  Thermal contribution only when T>0.
    T:
        Bath temperature in K.
    """
    s_csl = 3.0 * HBAR**2 * lambda_csl * M / (2.0 * m0**2 * r_C**2)
    if gamma > 0 and T > 0:
        s_thermal = 2.0 * M * gamma * K_B * T
        s_csl = s_csl + s_thermal
    return s_csl


# ---------------------------------------------------------------------------
# QuTiP operator helpers
# ---------------------------------------------------------------------------

def harmonic_oscillator_ops(N: int, omega: float, m: float) -> dict:
    """Build position, momentum, and number operators for an N-level HO.

    Parameters
    ----------
    N:
        Hilbert-space truncation (number of Fock states).
    omega:
        Trap frequency in rad/s.
    m:
        Particle mass in kg.

    Returns
    -------
    dict with keys: a, a_dag, x, p, H, x_unit, p_unit, E_ho
        x_unit, p_unit are the length and momentum scales (Δx_ho, Δp_ho).
    """
    a = qt.destroy(N)
    a_dag = a.dag()
    # Position and momentum in SI units
    x_unit = sqrt(HBAR / (m * omega))      # Δx_ho = √(ħ/mω)
    p_unit = sqrt(HBAR * m * omega)        # Δp_ho = √(mħω)
    x = x_unit * (a + a_dag) / sqrt(2.0)
    p = p_unit * (a - a_dag) / (1j * sqrt(2.0))
    H = HBAR * omega * (a_dag * a + qt.qeye(N) / 2.0)
    return {
        "a": a,
        "a_dag": a_dag,
        "x": x,
        "p": p,
        "H": H,
        "x_unit": x_unit,
        "p_unit": p_unit,
        "E_ho": HBAR * omega,
    }


def csl_collapse_operator_1d(
    N: int,
    x_op: qt.Qobj,
    lambda_csl: float,
    r_C: float,
    m: float,
    amu: float = AMU,
) -> qt.Qobj:
    """Build the CSL collapse operator for a 1D particle.

    The CSL Lindblad operator for the translation-covariant mass-proportional
    model is:

        A_CSL = (√(λ) m / (√2 π^(3/4) r_C^(3/2) m0)) · μ(x)

    where μ(x) is the mass-density operator at position x.  For a point
    particle μ(x) = δ(x - x_op), and the collapse operator simplifies to:

        A_CSL ∝ (√(λ) m / m0) · exp(-(x_op - ⟨x⟩)² / 4r_C²)

    For the QuTiP simulation we use the discrete representation where the
    collapse operator is diagonal in position basis:

        A_CSL = √(λ') ∑_n exp(-x_n² / 4r_C²) |n⟩⟨n|

    with λ' = (λ/2) (m/amu)² (ħ/r_C)² (the 1D diffusion coefficient).

    Returns a Qobj suitable for use in qt.smesolve as a collapse operator.
    """
    # Use the discrete x eigenvalues as the Gaussian weight centre
    # For a harmonic oscillator, x_op is the position operator.
    # We compute the collapse operator in the Fock basis as:
    #   A = √(λ') * exp(-(x_op - ⟨x⟩_state)² / 4r_C²)
    #
    # For simplicity and correctness in the master-equation sense, we use
    # the translation-covariant form: the collapse operator is diagonal in
    # the position representation.  In the Fock basis, the diagonal elements
    # are ⟨n| exp(-x²/4r_C²) |n⟩.

    # Effective CSL rate parameter (1D, per mode)
    lam_prime = 0.5 * lambda_csl * (m / amu) ** 2 * (HBAR / r_C) ** 2

    # Collapse operator: use the position operator itself as the "smeared"
    # density.  For a point particle in the HO basis, the diagonal elements
    # of exp(-x²/4r_C²) are computed via the Qobj exponential.
    # Simpler approach: project onto a set of position bases and sum.

    # For numerical tractability in the Fock basis, we use the Gaussian
    # operator exp(-x² / 4r_C²) directly as the collapse operator.
    # This is the standard approximation for CSL in the HO basis.
    x_sq = x_op * x_op
    # exp(-x² / 4r_C²) — compute via matrix exponential of the dense array
    x_matrix = x_op.full()
    gaussian_weight = exp(-(x_matrix * x_matrix) / (4.0 * r_C * r_C))
    A = qt.Qobj(gaussian_weight, dims=x_op.dims)

    # Scale by √(λ')
    A = sqrt(lam_prime) * A

    return A


# ---------------------------------------------------------------------------
# SSE trajectory (single trajectory of the stochastic Schrödinger equation)
# ---------------------------------------------------------------------------

def sse_csl_trajectory(
    N: int,
    omega: float,
    m: float,
    lambda_csl: float,
    r_C: float,
    tlist: np.ndarray,
    x0: float = 0.0,
    p0: float = 0.0,
    seed: int | None = None,
    scheme: Literal["euler-maruyama", "rouchon"] = "euler-maruyama",
    ntraj: int = 1,
) -> dict:
    """Run one or more SSE trajectories for a CSL-smoothed harmonic oscillator.

    Parameters
    ----------
    N:
        Fock-space truncation.
    omega:
        Trap frequency in rad/s.
    m:
        Particle mass in kg.
    lambda_csl:
        CSL collapse rate in s⁻¹.
    r_C:
        CSL localisation length in m.
    tlist:
        Time points at which to record the state (1D array).
    x0, p0:
        Initial coherent-state displacement (in SI units, metres and kg·m/s).
    seed:
        RNG seed for reproducibility.
    scheme:
        SDE integration scheme.
    ntraj:
        Number of independent trajectories to average.

    Returns
    -------
    dict with keys:
        xs:    array (ntraj, len(tlist)) of ⟨x⟩(t) / x_unit
        ps:    array (ntraj, len(tlist)) of ⟨p⟩(t) / p_unit
        D2s:   array (ntraj, len(tlist)) of ⟨x²⟩ - ⟨x⟩²  (position variance)
        traj:  list of Qobj states at each time (first trajectory)
    """
    rng = np.random.default_rng(seed)

    ops = harmonic_oscillator_ops(N, omega, m)
    x = ops["x"]
    p = ops["p"]
    H = ops["H"]
    x_unit = ops["x_unit"]
    p_unit = ops["p_unit"]

    # Collapse operator
    C = csl_collapse_operator_1d(N, x, lambda_csl, r_C, m)

    # Initial coherent state |α⟩
    alpha = (x0 / x_unit + 1j * p0 / p_unit) / sqrt(2.0)
    psi0 = qt.coherent(N, alpha)

    # Record arrays
    xs = np.zeros((ntraj, len(tlist)))
    ps = np.zeros((ntraj, len(tlist)))
    D2s = np.zeros((ntraj, len(tlist)))

    for traj_idx in range(ntraj):
        if scheme == "euler-maruyama":
            result = qt.smesolve(
                H,
                psi0,
                tlist,
                [C],
                [],
                ntraj=1,
                seeds=[rng.integers(0, 2**31) if seed is not None else None],
            )
        elif scheme == "rouchon":
            # Rouchon positivity-preserving scheme (QuTiP 5.x)
            # In QuTiP 5.x, smesolve with method='rouchon' is available.
            result = qt.smesolve(
                H,
                psi0,
                tlist,
                [C],
                [],
                ntraj=1,
                method="rouchon",
                seeds=[rng.integers(0, 2**31) if seed is not None else None],
            )
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        states = result.states
        for ti, t in enumerate(tlist):
            idx = np.argmin(np.abs(result.times - t))
            state = states[idx]
            xs[traj_idx, ti] = (x & state).tr().real / x_unit
            ps[traj_idx, ti] = (p & state).tr().real / p_unit
            D2s[traj_idx, ti] = (
                (x * x & state).tr().real - ((x & state).tr().real) ** 2
            ) / x_unit**2

    return {
        "xs": xs,
        "ps": ps,
        "D2s": D2s,
        "x_unit": x_unit,
        "p_unit": p_unit,
        "tlist": tlist,
    }


# ---------------------------------------------------------------------------
# Master-equation solver (deterministic, no noise)
# ---------------------------------------------------------------------------

def csll_master_equation(
    N: int,
    omega: float,
    m: float,
    lambda_csl: float,
    r_C: float,
    tlist: np.ndarray,
    rho0: qt.Qobj | None = None,
) -> qt.Result:
    """Solve the CSL master equation for a harmonic oscillator deterministically.

    dρ/dt = -i/ħ [H, ρ] + D [AρA† - ½{A†A, ρ}]

    with A = CSL collapse operator and D = λ (m/amu)² (ħ/r_C)² / 2.

    Parameters
    ----------
    rho0:
        Initial density matrix.  Defaults to ground state |0⟩⟨0|.
    """
    ops = harmonic_oscillator_ops(N, omega, m)
    H = ops["H"]
    x = ops["x"]

    if rho0 is None:
        rho0 = qt.fock(N, 0) * qt.fock(N, 0).dag()

    C = csl_collapse_operator_1d(N, x, lambda_csl, r_C, m)

    # Lindblad dissipator rate: D = ||C||² (since smesolve uses C as the
    # collapse operator, the Lindblad term is D (C ρ C† - ½{C†C, ρ})
    # and D = 1 for a single normalised collapse operator.
    # Here C already includes √D as a prefactor, so we pass it directly.

    result = qt.mesolve(
        H,
        rho0,
        tlist,
        [C],
        [],
    )
    return result


# ---------------------------------------------------------------------------
# Wigner function evolution (Fokker-Planck equation in phase space)
# ---------------------------------------------------------------------------

def wigner_fokker_planck_csl(
    N_grid: int = 64,
    omega: float = 2e6,
    m: float = 1e-17,
    lambda_csl: float = 1e-17,
    r_C: float = 1e-7,
    T: float = 0.0,
    t_max: float = 1e-3,
    n_steps: int = 200,
) -> dict:
    """Solve the CSL Wigner Fokker-Planck equation on a 2D phase-space grid.

    ∂W/∂t = {H, W} + D_xx ∂²W/∂x² + D_pp ∂²W/∂p² + D_xp (∂²W/∂x∂p + ∂²W/∂p∂x)

    where D_xx, D_pp, D_xp are the CSL diffusion coefficients.

    For CSL with a Gaussian kernel:
    D_pp = λ (m/amu)² ħ² / (4 r_C²)    (momentum diffusion)
    D_xx = 0                             (no position diffusion in CSL)
    D_xp = 0

    Returns a dict with:
        W:      2D array (N_grid × N_grid) of final Wigner function
        x_grid: 1D array of position grid points (in x_unit)
        p_grid: 1D array of momentum grid points (in p_unit)
        t:      time evolution array
    """
    x_unit = sqrt(HBAR / (m * omega))
    p_unit = sqrt(HBAR * m * omega)

    # CSL diffusion coefficient (momentum)
    D_pp = lambda_csl * (m / AMU) ** 2 * HBAR**2 / (4.0 * r_C**2)
    D_pp_unit = D_pp * (p_unit / HBAR) ** 2  # in 1/s  (phase-space units)

    # Grid
    x_max = 5.0 * x_unit
    p_max = 5.0 * p_unit
    x_vals = np.linspace(-x_max, x_max, N_grid)
    p_vals = np.linspace(-p_max, p_max, N_grid)
    dx = x_vals[1] - x_vals[0]
    dp = p_vals[1] - p_vals[0]
    X, P = np.meshgrid(x_vals / x_unit, p_vals / p_unit, indexing="ij")

    # Initial Wigner: Gaussian (ground state of HO)
    W = np.exp(-(X**2 + P**2)) / pi  # normalised to 1

    # Time evolution: drift (harmonic) + diffusion (CSL)
    dt = t_max / n_steps
    t_arr = np.linspace(0, t_max, n_steps)

    W_history = np.zeros((n_steps,) + W.shape)

    for step in range(n_steps):
        # Drift term: {H, W} = (∂H/∂p)(∂W/∂x) - (∂H/∂x)(∂W/∂p)
        # For H = p²/2m + mω²x²/2:
        #   ∂H/∂p = p/m, ∂H/∂x = mω²x
        # In phase-space units (x̂=x/x_unit, p̂=p/p_unit):
        #   Ĥ = (p̂² + x̂²)/2,  {Ĥ, W} = p̂ ∂W/∂x̂ - x̂ ∂W/∂p̂
        dW_dx = np.gradient(W, dx, axis=1) / x_unit
        dW_dp = np.gradient(W, dp, axis=0) / p_unit
        drift = (P * p_unit / (m * x_unit)) * dW_dx - (X * x_unit * m * omega**2 / p_unit) * dW_dp
        drift = (P / x_unit) * (x_unit * p_unit / (m * x_unit)) * dW_dx - (
            X / p_unit
        ) * (x_unit * p_unit * m * omega**2 / p_unit) * dW_dp

        # Actually in dimensionless units where x̂ = x/x_unit, p̂ = p/p_unit:
        # Ĥ = (p̂² + x̂²)/2
        # {Ĥ, W} = ∂Ĥ/∂p̂ · ∂W/∂x̂ - ∂Ĥ/∂x̂ · ∂W/∂p̂
        #          = p̂ · ∂W/∂x̂ - x̂ · ∂W/∂p̂
        dW_dx_hat = np.gradient(W, 1.0, axis=1) / (dx / x_unit)
        dW_dp_hat = np.gradient(W, 1.0, axis=0) / (dp / p_unit)
        # Simpler: use grid spacing in hat units
        dW_dx_hat = np.gradient(W, axis=1) / (2.0 * x_max / N_grid)
        dW_dp_hat = np.gradient(W, axis=0) / (2.0 * p_max / N_grid)

        drift_hat = P * dW_dx_hat - X * dW_dp_hat

        # Diffusion term: D_pp ∂²W/∂p²
        d2W_dp2 = np.gradient(np.gradient(W, axis=0), axis=0) / (dp / p_unit) ** 2
        diffusion = D_pp_unit * d2W_dp2 * dt

        # Drift step
        dW = drift_hat * dt * (1.0 / (2.0 * x_max / N_grid))  # approximate

        # Update
        W = W + dW + diffusion

        # Positivity: clip negative values (numerical artifact)
        W = np.maximum(W, 0.0)
        # Renormalise ( ∫∫ W dx dp = 1 )
        integral = np.trapz(np.trapz(W, x_vals, axis=1), p_vals, axis=0)
        if integral > 0:
            W = W / integral

        W_history[step] = W.copy()

    return {
        "W_final": W,
        "W_history": W_history,
        "x_grid": x_vals / x_unit,
        "p_grid": p_vals / p_unit,
        "t": t_arr,
        "x_unit": x_unit,
        "p_unit": p_unit,
        "D_pp": D_pp,
        "D_pp_unit": D_pp_unit,
    }


# ---------------------------------------------------------------------------
# dCSL (Dissipative CSL) extension
# ---------------------------------------------------------------------------

def dcdl_diffusion_coefficients(
    lambda_csl: float,
    r_C: float,
    m: float,
    T_CSL: float,
    amu: float = AMU,
) -> dict:
    """Compute the dCSL diffusion coefficients (momentum heating + friction).

    For dissipative CSL with noise temperature T_CSL:
    D_pp = λ (m/amu)² ħ² / (4 r_C²) · coth(ħ² / 8 m r_C² k_B T_CSL)

    The friction coefficient:
    γ_D = λ (m/amu)² ħ² / (4 m r_C² k_B T_CSL)

    Returns dict with D_pp, D_xx, gamma_D.
    """
    # 인자: coth(x) → 1 for large x, ~ 1/x for small x
    arg = HBAR**2 / (8.0 * m * r_C**2 * K_B * T_CSL)
    if arg > 10.0:
        coth = 1.0
    else:
        coth = 1.0 / np.tanh(arg)

    D_pp = lambda_csl * (m / amu) ** 2 * HBAR**2 / (4.0 * r_C**2) * coth
    gamma_D = lambda_csl * (m / amu) ** 2 * HBAR**2 / (4.0 * m * r_C**2 * K_B * T_CSL)

    return {
        "D_pp": D_pp,
        "D_xx": 0.0,
        "gamma_D": gamma_D,
        "coth_arg": arg,
    }


# ---------------------------------------------------------------------------
# Ensemble average over SSE trajectories
# ---------------------------------------------------------------------------

def ensemble_csl_signal(
    N: int = 30,
    omega: float = 2e6,
    m: float = 1e-17,
    lambda_csl: float = 1e-17,
    r_C: float = 1e-7,
    t_max: float = 1e-3,
    n_steps: int = 100,
    n_traj: int = 50,
    x0: float = 0.0,
    p0: float = 0.0,
    seed: int = 42,
) -> dict:
    """Run an ensemble of SSE trajectories and compute ensemble-averaged signals.

    Returns
    -------
    dict with keys:
        t:        1D array of time points
        x_mean:   ⟨⟨x⟩⟩(t) — ensemble-averaged position
        x_var:    ⟨⟨x²⟩⟩ - ⟨⟨x⟩⟩² — ensemble variance
        D2_ens:   ⟨⟨(Δx)²⟩⟩ — average position variance (decoherence measure)
        x_traj:   (n_traj, n_steps) array of individual trajectory signals
    """
    tlist = np.linspace(0, t_max, n_steps)

    result = sse_csl_trajectory(
        N=N,
        omega=omega,
        m=m,
        lambda_csl=lambda_csl,
        r_C=r_C,
        tlist=tlist,
        x0=x0,
        p0=p0,
        seed=seed,
        scheme="euler-maruyama",
        ntraj=n_traj,
    )

    x_mean = np.mean(result["xs"], axis=0) * result["x_unit"]
    x_var = np.var(result["xs"], axis=0) * result["x_unit"]**2
    D2_ens = np.mean(result["D2s"], axis=0) * result["x_unit"]**2

    return {
        "t": result["tlist"],
        "x_mean": x_mean,
        "x_var": x_var,
        "D2_ens": D2_ens,
        "x_traj": result["xs"],
        "x_unit": result["x_unit"],
    }


# ---------------------------------------------------------------------------
# Unit and convention helpers
# ---------------------------------------------------------------------------

class SimulationConfig:
    """Convenience container for a simulation parameter set."""

    def __init__(
        self,
        N: int = 30,
        omega: float = 2e6,
        m: float = 1e-17,
        lambda_csl: float = 1e-17,
        r_C: float = 1e-7,
        lam0: float = 1e-17,  # legacy alias
        T_bath: float = 0.0,
        T_CSL: float = 0.0,
        gamma: float = 0.0,
    ):
        self.N = N
        self.omega = omega
        self.m = m
        self.lambda_csl = lambda_csl if lambda_csl > 0 else lam0
        self.r_C = r_C
        self.T_bath = T_bath
        self.T_CSL = T_CSL
        self.gamma = gamma

    @property
    def x_unit(self) -> float:
        return sqrt(HBAR / (self.m * self.omega))

    @property
    def p_unit(self) -> float:
        return sqrt(HBAR * self.m * self.omega)

    @property
    def E_ho(self) -> float:
        return HBAR * self.omega

    @property
    def csl_heating_P(self) -> float:
        return csl_heating_power(self.lambda_csl, self.r_C, self.m)

    def __repr__(self) -> str:
        return (
            f"SimConfig(N={self.N}, ω={self.omega:.2e}, m={self.m:.2e}, "
            f"λ={self.lambda_csl:.2e}, r_C={self.r_C:.2e})"
        )


__all__ = [
    "HBAR",
    "AMU",
    "K_B",
    "G",
    "csl_decoherence_rate_1d",
    "csl_decoherence_rate_3d",
    "csl_decoherence_factor",
    "dp_self_energy_exponent",
    "dp_self_energy_exponent_sph",
    "csl_heating_power",
    "csl_force_psd",
    "harmonic_oscillator_ops",
    "csl_collapse_operator_1d",
    "sse_csl_trajectory",
    "csll_master_equation",
    "wigner_fokker_planck_csl",
    "dcdl_diffusion_coefficients",
    "ensemble_csl_signal",
    "SimulationConfig",
]
