# CSL damping function η(ρ, r_C) for a uniform solid sphere — prefactor in the R ≫ r_C regime

Date: 2026-09-25. All sources read from original arXiv LaTeX sources (exact equation text).

## 1. The answer for YOUR functional

Your integral (kernel `exp(-|r-r'|²/4r_C²)`, η = (1/m²)∫∫, i.e. normalized so η→1 for R≪r_C):

    η(R,r_C) = 6√π (r_C/R)³ [ 1 − (3/√π)(r_C/R) + O((r_C/R)²) ]
    ⇒ prefactor = 6√π ≈ 10.6347231

Derivation (exact): with ϱ₀ = m/V, V = 4πR³/3, sphere-overlap volume
V_ov(s) = (π/12)(4R+s)(2R−s)²,
    η = (ϱ₀²/V²)·4π ∫₀^{2R} s² e^{−s²/4r_C²} V_ov(s) ds = (3/16R⁶)[16R³A₂ − 12R²A₃ + A₅],
with β = 1/(4r_C²), c = 2R, U = βc² = R²/r_C², A_n = ∫₀^c sⁿ e^{−βs²}ds:
    A₂ = (1/4β)[√(π/β)·erf(c√β) − 2c·e^{−βc²}],
    A₃ = (1/2β²)[1 − e^{−βc²}(1+βc²)],
    A₅ = (1/2β³)[2 − e^{−U}(U²+2U+2)].
Closed form verified against direct quadrature to machine precision (ratio 1.0000000000 at
R/r_C = 0.5, 1, 10, 100, 1000). Leading term: ∫d³s e^{−s²/4r_C²} = (4πr_C²)^{3/2} = 8π^{3/2}r_C³,
so η = 8π^{3/2}r_C³/V = 6√π (r_C/R)³.

Equivalent Fourier form (Parseval shortcut): η = (4/√π)∫q²e^{−q²} F_sph(qR/r_C)² dq with
F_sph(u) = 3(sin u − u cos u)/u³ and ∫₀^∞ u²F_sph²du = 3π/2 ⇒ C = (4/√π)(3π/2) = 6√π.

Numerics: R/r_C=10 → η(R/r_C)³ = 8.847; 100 → 10.455; 1000 → 10.617; limit 10.6347.

Kernel-convention sensitivity: only the 4r_C² kernel gives 6√π. A 2r_C² kernel would give
(2π)^{3/2}·3/(4π) = (3/2)√(2π) ≈ 3.7600. All papers listed below use the 4r_C² convention.

## 2. Source-by-source (no mixing, each convention separately)

### A. Nimmrichter, Hornberger, Hammerer, PRL 113, 020405 (2014) [arXiv:1405.2868v2]
- Main text Eq. (1): α = r_CSL⁵/(π^{3/2} amu²) ∫d³k k_x² e^{−r_CSL²k²}|ϱ̃(k)|², ϱ̃(0)=m;
  D_CSL = λ_CSL(ħ/r_CSL)²α. NOTE the k_x² weight (1-D momentum diffusion). Real-space kernel
  is your Gaussian (e^{−r²k²} ⇔ e^{−|r−r'|²/4r_C²}), but this is NOT your k-free η.
- Sphere form factor: Supplemental Material, section "CSL diffusion for cuboids, spheres and
  discs", Eq. (S7): ϱ̃_sphere(k) = 3m(sin kR − kR cos kR)/(kR)³.
- Exact α_sphere: SM Eq. (S10): α = (m/amu)²[e^{−R²/r²} − 1 + (R²/2r²)(e^{−R²/r²}+1)]·6r⁶/R⁶.
- Asymptotic expansion: main-text Eq. (2): α_sph ≈ 16π²ϱ²r_CSL⁴ R²/(3 amu²), R ≫ r_CSL
  (SM explicitly: "Equations (2) and (3) in the main text are obtained by expanding the exact
  geometry factors in R/r_CSL …").
- SM numbering (S7)/(S10) is from ar5iv rendering of the identical arXiv v2 source; APS
  supplemental page returned HTTP 401 (paywall) — verify against published SM PDF.
- Small-R: α → (m/amu)²/2 (point value); no "η=1" statement, no η symbol in the paper.
- LARGE-R PHYSICS: relative to the point value, η_NHH = α/(α_point) = 6(r_CSL/R)⁴ − 12(r_CSL/R)⁶…
  (FOURTH power, prefactor 6) — because the k_x² weight cancels the bulk and only the surface
  layer contributes. NHH Eq. (2) is NOT a (r_C/R)³ law; do not cite it for your prefactor.
  Absolute statement (their convention): α_sph = 3(m/amu)²(r_CSL/R)⁴ = 16π²ϱ²r_CSL⁴R²/(3 amu²).

### B. Ferialdi & Bassi, PRA 102, 042213 (2020) [arXiv:2006.09013]
- Uses η^{αβ} for diffusion coefficients with EXACTLY your kernel:
  Eq. label `etadisc`: η^{αβ} = λ/(8r_C⁴m_N²) Σ_{ij} m_i m_j e^{−(r_i−r_j)²/4r_C²}{…};
  Eq. label `etazzvar`: η^{zz} = λ/(2m_N²)∫∫ν(u)ν(v)e^{−(u−v)²/4r_C²};
  Eq. label `gammasmall`: Γ_C = Δ²η^{zz} (Δ ≪ r_C).
- Sphere, subsection "Other geometries" (cites Nimmrichter-Hornberger-Hammerer):
  Γ_sph = λ N_TOT² (3r_C⁴/R⁶)[e^{−R²/r_C²} − 1 + (R²/2r_C²)(e^{−R²/r_C²}+1)] Δ².
  Same bracket as SM (S10); normalization λN_TOT² (nucleon-count SQUARED).
  ⇒ same 6(r_C/R)⁴ relative damping. This is the gradient/Δ² object, not your k-free η.

### C. Toroš & Bassi, J. Phys. A 51, 115302 (2018) [arXiv:1601.02931] "…Calculational details"
- Kernel e^{−Q²r_C²/ħ²} ⇔ e^{−|r−r'|²/4r_C²} (same convention); Eq. `finally_x_term`:
  Λ exp(−(Δx²+Δy²+Δz²)/4r_C²).
- EXACT homogeneous-disk evaluation of your integral family (subsection "homogeneous disk"):
  Λ = 4λ(m/m₀)² (r_C²/r_s²)(1 − e^{−r_s²/4r_C²}).
  Large disk: → 4(r_C/r_s)² — the exact 2-D analogue of your 6√π(r_C/R)³ (2-D Gaussian
  ∫e^{−s²/4r_C²}d²s = 4πr_C² ⇒ factor 4). Closest published evaluation of this family, but a
  DISK — NO paper found printing the sphere's 6√π.
- N vs N², explicitly: Eq. `eq:dhggg` R(Q) ≈ Σ1 = N² (all pairs within r_C);
  Eq. `rq_eq` R(Q) ≈ Σ_{j=l}1 = N (inter-particle distances ≫ r_C); intermediate Eq. `mass2`:
  Λ = (n_A/n(r_C))(m_A n(r_C)/m₀)²λ. For a CONTINUUM sphere, N² bookkeeping always applies;
  linear-N only if nucleon spacing > r_C.
- No sphere form factor (planar molecule/disk/2-D lattice only; they cite PRL 113, 020405 for
  the "2D lattice structure approximation").

### D. Toroš, Gasbarri, Bassi, arXiv:1601.03672 (cdCSL/KDTL bounds)
- No sphere geometry factor. Λ = (n_A/n(r_C))(m_A n(r_C)/m₀)²λ; states "the localization rate
  of a point-like system with N particles is amplified by a factor N²". Kernel factors
  e^{−(q−kτ/m)²/4r_C²} confirm the 4r_C² convention.

### E. Davila & Milburn, arXiv:2608.05972 (2026), "Geometry-Only CSL/DP Ratios…"
- Point kernel K(Δx) = 1 − exp(−Δx²/4r_C²); finite-size CSL kernel (unnumbered display):
  K_CSL = (4/√π)∫q²e^{−q²}|F_ϱ(qR/r_C)|²(1 − sinc(qΔx/r_C))dq, F_ϱ(0)=1,
  F_sph(u) = 3(sin u − u cos u)/u³; Λ_CSL = λ(m/amu)²τ K_CSL.
- K_CSL at Δx→∞ is EXACTLY your η (same →1 normalization). Evaluated by "direct quadrature"
  for the homogeneous sphere — no closed form, no 6√π asymptotic printed. Does NOT cite
  PRL 113, 020405 (bib has only Nimmrichter2013).

## 3. Normalization (N vs N²) verdict
- Every source normalizes the point-particle damping by the SQUARE of the size measure:
  m², (m/amu)², N_TOT², m²/m₀² — N² convention with η(R→0)=1 (NHH's α carries an extra ½
  because it is a 1-D diffusion geometry factor: α_point = (m/amu)²/2).
- None divides by N; a per-N normalization would scale your prefactor by N and break η→1.
- Minor spread: NHH divide by amu²; Ferialdi–Bassi by m_N² (nucleon mass ≈0.998623 u) —
  ~0.27% shift per nucleon, not an order-one difference.
- Two DISTINCT published "sphere damping" laws: k-free (coherence self-overlap): 6√π(r_C/R)³;
  k-weighted gradient (diffusion/reduction rates, PRL 113, 020405 & PRA 102, 042213):
  6(r_C/R)⁴ relative to the point value. Different objects — report separately, never average.

Artifacts kept in D:\csl\arxiv_src\ (arXiv LaTeX sources + verify.py, verify2.py).

