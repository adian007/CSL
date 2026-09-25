# CSL/DP Cross-CAS and Cross-Validation Pipeline

This repository implements a high-precision verification pipeline for the CSL (Continuous Spontaneous Localization) and DP (Diosi-Penrose) collapse models, specifically focusing on the geometry-only ratios for homogeneous bodies.

The project implements the theoretical framework established in **Davila & Milburn (2026, arXiv:2608.05972)**, focusing on the "Cancellation Theorem" where the CSL/DP ratio $\\Xi$ becomes independent of the mass $m$ and interrogation time $\\tau$.

## 🚀 Project Structure

- **`src/csl_pipeline/`**: The core implementation.
  - `geometry.py`: Implementations of the k-free overlap factor ($\\eta$) and the k-weighted geometry factor ($\\alpha$).
  - `rates.py`: Definition of the point-particle and finite-size collapse rates and their ratios.
  - `verify/`: Utility modules for Cross-CAS (Computer Algebra System) verification and numerical precision checks.
- **`scripts/verification/`**: Standalone scripts to run the verification suite.
  - `run_verification.py`: The primary report generator running the 10 core checks.
- **`tests/`**: Unit tests for the pipeline.
- **`resources/`**: External references, including source files from cited arXiv papers.
- **`docs/`**: Technical documentation and convergence reports.

## 🧪 Verification Workflow

The pipeline employs a rigorous four-stage verification chain:

1. **Symbolic Derivation**: Defining the integrals in SymPy to obtain closed-form expressions.
2. **Cross-CAS Verification**: Confirming SymPy results against the Wolfram Language to ensure no algebraic errors.
3. **Triple-Numeric Agreement**: Comparing closed-form values against `scipy` adaptive quadrature and `mpmath` high-precision integration.
4. **Asymptotic Limits**: Asserting that the implementations recover known physical limits (e.g., quadratic behavior for small bodies and constant saturation for large bodies).

## 🛠️ Getting Started

### Prerequisites
- Python 3.13+
- Dependencies: `numpy`, `scipy`, `sympy`, `mpmath`, `pytest`

### Running the Verification Suite
To run the full cross-validation report:

```bash
export PYTHONPATH=src
python scripts/verification/run_verification.py
```

### Running Tests
```bash
pytest tests/
```

## 📚 References
- **Davila & Milburn (2026)**: "Geometry-Only CSL/DP Ratios and the Nonuniqueness of Decoherence Kernels," [arXiv:2608.05972](https://arxiv.org/abs/2608.05972).
- **Nimmrichter, Hornberger & Hammerer (2014)**: [Phys. Rev. Lett. 113, 020405](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.113.020405).
