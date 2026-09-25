"""Package root for the CSL / Diocsi-Penrose multi-channel pipeline.

The package is organised so that every published quantity carries its own
provenance and so that physically distinct objects cannot be silently merged.

Layout
------
``geometry``
    Homogeneous-body geometry factors (sphere, disc, cuboid).
``rates``
    Collapse and diffusion rates. Registered in ``RATE_REGISTRY`` with their
    valid regimes so that callers cannot pass an invalid combination silently.
``paramspace``
    Parameter-space algebra, including the CSL <-> DP correspondence.
"""

from __future__ import annotations

from csl_pipeline import (
    bayesian_inference,
    csl_dp_scale,
    collapse_sim,
    geometry,
    paramspace,
    rates,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "bayesian_inference",
    "csl_dp_scale",
    "collapse_sim",
    "geometry",
    "paramspace",
    "rates",
]
