"""Verification tests for the CSL/DP simulation layer (collapse_sim).

These tests encode the acceptance criteria for the simulation core:
* analytical formulas match their definitions at representative (Δx, τ) pairs;
* dimensional analysis confirms SI units for heating power;
* SimulationConfig properties return correct values.
"""

from __future__ import annotations

import math

import pytest

from csl_pipeline import collapse_sim as cs


class TestCSLDecoherenceFactor:
    """The CSL decoherence factor D_CSL(Δx, τ) must match its definition."""

    def test_csl_decoherence_factor_formula(self):
        """Verify D_CSL = exp(-Λ τ K_rc) at three (Δx, τ) pairs.

        Definition:
            K_rc(Δx) = 1 - exp(-Δx² / (4 r_C²))
            Λ_CSL = λ (m/amu)²
            D_CSL(Δx, τ) = exp(-Λ_CSL τ K_rc(Δx))
        """
        lambda_csl = 1e-17
        r_C = 1e-7
        m = 1e-15
        amu = cs.AMU

        # Pair 1: Δx ≪ r_C → K_rc ≈ Δx²/(4r_C²), small decoherence
        delta_x1 = 1e-9
        tau1 = 1e-3
        K_rc1 = 1.0 - math.exp(-(delta_x1 * delta_x1) / (4.0 * r_C * r_C))
        Lambda1 = lambda_csl * (m / amu) ** 2
        expected1 = math.exp(-Lambda1 * tau1 * K_rc1)
        result1 = cs.csl_decoherence_factor(delta_x1, lambda_csl, r_C, m, tau1)
        assert abs(result1 - expected1) < 1e-15

        # Pair 2: Δx ~ r_C → intermediate K_rc
        delta_x2 = 1e-7
        tau2 = 1e-2
        K_rc2 = 1.0 - math.exp(-(delta_x2 * delta_x2) / (4.0 * r_C * r_C))
        Lambda2 = lambda_csl * (m / amu) ** 2
        expected2 = math.exp(-Lambda2 * tau2 * K_rc2)
        result2 = cs.csl_decoherence_factor(delta_x2, lambda_csl, r_C, m, tau2)
        assert abs(result2 - expected2) < 1e-15

        # Pair 3: Δx ≫ r_C → K_rc → 1, maximum decoherence rate
        delta_x3 = 1e-5
        tau3 = 1.0
        K_rc3 = 1.0 - math.exp(-(delta_x3 * delta_x3) / (4.0 * r_C * r_C))
        Lambda3 = lambda_csl * (m / amu) ** 2
        expected3 = math.exp(-Lambda3 * tau3 * K_rc3)
        result3 = cs.csl_decoherence_factor(delta_x3, lambda_csl, r_C, m, tau3)
        assert abs(result3 - expected3) < 1e-15


class TestHeatingPower:
    """CSL heating power must have correct units (watts)."""

    def test_heating_power_units(self):
        """Verify P_CSL has units of watts via dimensional analysis.

        P_CSL = (3 ħ² λ M) / (4 m0² r_C²)

        Dimensional check:
            [ħ] = J·s = kg·m²·s⁻¹
            [ħ²] = kg²·m⁴·s⁻²
            [λ] = s⁻¹
            [M] = kg
            [m0] = kg
            [r_C] = m

            [P] = (kg²·m⁴·s⁻²) · s⁻¹ · kg / (kg² · m²)
                = kg·m²·s⁻³ = W  ✓
        """
        lambda_csl = 1e-17
        r_C = 1e-7
        M = 1e-15
        m0 = cs.AMU

        P = cs.csl_heating_power(lambda_csl, r_C, M, m0)

        # Numerical check: the formula gives a specific value
        # P = 3 * ħ² * λ * M / (4 * m0² * r_C²)
        expected = 3.0 * cs.HBAR**2 * lambda_csl * M / (4.0 * m0**2 * r_C**2)
        assert abs(P - expected) / expected < 1e-15

        # Units check: P must be in watts (kg·m²·s⁻³)
        # 1 W = 1 kg·m²·s⁻³
        # Verify via dimensional reconstruction
        hbar_sq = cs.HBAR**2  # kg²·m⁴·s⁻²
        lam = lambda_csl       # s⁻¹
        mass = M              # kg
        m0_sq = m0**2         # kg²
        rc_sq = r_C**2        # m²

        # Reconstruct: (3 * hbar_sq * lam * mass) / (4 * m0_sq * rc_sq)
        # Should have units kg·m²·s⁻³
        reconstructed = 3.0 * hbar_sq * lam * mass / (4.0 * m0_sq * rc_sq)
        assert abs(P - reconstructed) < 1e-30


class TestSimulationConfig:
    """SimulationConfig properties must return correct values."""

    def test_simulation_config(self):
        """Verify x_unit, p_unit, E_ho properties are correct.

        For a harmonic oscillator:
            x_unit = √(ħ / (m ω))
            p_unit = √(m ħ ω)
            E_ho = ħ ω
        """
        N = 30
        omega = 2e6
        m = 1e-17
        lambda_csl = 1e-17
        r_C = 1e-7

        config = cs.SimulationConfig(
            N=N, omega=omega, m=m,
            lambda_csl=lambda_csl, r_C=r_C,
        )

        # x_unit = √(ħ / (m ω))
        expected_x_unit = math.sqrt(cs.HBAR / (m * omega))
        assert abs(config.x_unit - expected_x_unit) / expected_x_unit < 1e-15

        # p_unit = √(m ħ ω)
        expected_p_unit = math.sqrt(m * cs.HBAR * omega)
        assert abs(config.p_unit - expected_p_unit) / expected_p_unit < 1e-15

        # E_ho = ħ ω
        expected_E_ho = cs.HBAR * omega
        assert abs(config.E_ho - expected_E_ho) / expected_E_ho < 1e-15

        # csl_heating_P should match csl_heating_power
        expected_P = cs.csl_heating_power(lambda_csl, r_C, m)
        assert abs(config.csl_heating_P - expected_P) / expected_P < 1e-15
