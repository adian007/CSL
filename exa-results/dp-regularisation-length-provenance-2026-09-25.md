# DP-model regularisation length: provenance of σ_DP ~ 285.5 fm, R₀ > 4.9×10⁻¹⁰ m, R_DP > 2.8×10⁻⁹ m

Date: 2026-09-25. All source text read from arXiv e-print LaTeX sources (exact equation text), not from abstracts or secondary summaries, unless explicitly labelled.

## 0. Bottom line

There is ONE physical parameter here — the DP mass-density regularisation ("smearing") length — but it is quoted under three different symbols and, critically, the three numbers come from **three different kinds of statement**:

| Value | Symbol | What it actually is | Direction | Origin |
|---|---|---|---|---|
| ~285.5 fm | σ_DP | Experimental **lower** bound from LISA Pathfinder acceleration noise | σ_DP **>** 285.5 fm | Dai, Miao, Ma (2024), arXiv:2411.17588 |
| > 4.9×10⁻¹⁰ m | R₀ | Experimental **lower** bound from XENONnT X-ray data | R₀ **>** 4.9×10⁻¹⁰ m | Aprile et al. (XENONnT), arXiv:2506.05507, PRL 136, 120201 (2026) |
| > 2.8×10⁻⁹ m | R_DP | **NOT FOUND as a published lower bound.** Numerically matches an optomechanical **upper** bound σ_DP < Σ_DP ≈ 2.8–2.9 nm | (published form is **<**, opposite) | likely Nimmrichter, Hornberger, Hammerer (2014), arXiv:1405.2868 — inferred, not confirmed |

Your suspicion is right and then some: two of the three are experimental lower bounds, and the third appears to be a **detectability ceiling with the opposite inequality direction**. These are not three estimates of one number.

## 1. σ_DP ~ 285.5 fm — Dai, Miao, Ma, arXiv:2411.17588

**It is an experimental LOWER bound, not a model input.**

Source line (document_YM.tex:156–160), immediately after their Eq. for λ_CSL < 8.3×10⁻¹¹ s⁻¹:

> For the DP model, we can update $\sigma_{\text{DP}}$ with the new bound
> $$\sigma_{\text{DP}}>285.5\,\text{fm},$$
> however the strongest bound for DP model is set by X-ray emission test [arnquist2022search], at $\sigma_{\text{DP}}>4.94\times10^{5}$ fm, which is much larger than the bound derived from macroscopic test masses.

Note the abstract says "$\sigma_{\rm DP}\sim 285.5$ fm" but the displayed equation is a strict **lower** bound.

**Controlling formula** (their Eq., document_YM.tex:131–135):
σ_DP^min = a ( (2ℏG)/(3√π) · (ρ/m) · (1/S_a) )^{1/3} = 40.1 fm, with a = 4.0 Å (LPF test-mass lattice constant), ρ = 19881 kg/m³, M = 1.928 kg.

**I reproduced both numbers exactly** from this formula (constants G, ℏ standard; M = test mass, NOT nucleon mass):
- √S_a = 5.2 fm s⁻² Hz^(−1/2) → σ_DP = **40.13 fm** (paper: 40.1 fm) — this is the Helou et al. 2017 value.
- ΔS_Brown = 0.075 fm² s⁻⁴ Hz⁻¹ → σ_DP = **285.60 fm** (paper: 285.5 fm).

Ratio 285.5/40.1 = 7.12, exactly (S_a,old/S_a,new)^(1/3) = 7.117. The improvement is purely the smaller LISA Brownian noise floor, through the same 1/S_a^(1/3) form.

**Mechanism / direction.** The DP acceleration-noise contribution scales as 1/σ_DP³, so it is matched against the measured noise floor S_a. Smaller noise ⇒ larger excluded σ_DP. Hence a **lower** bound: σ_DP below 285.5 fm would produce diffusion above the LISA noise.

## 2. R₀ > 4.9×10⁻¹⁰ m — XENONnT, arXiv:2506.05507 (PRL 136, 120201 (2026))

**It is an experimental LOWER bound on the DP correlation/cut-off length.**

Verbatim (main.tex:132–134):
> Due to the more complex rate dependencies on the free parameter $R_0$ in the DP model, which can result in changes of the spectral shape beyond a global scaling, the fitting method employing a single linear rate multiplier requires an iterative approach... Any starting values not excluded by former experiments lead to the convergence to a best-fit value of $R_0 \approx 9.1\times10^{-10}$ m. With a local DP discovery significance of $0.2\,\sigma$, a lower limit of $R_0 > 4.9\times10^{-10}$ m [$4.5\times10^{-10}$ m] at 90% C.L. [95% C.L.] is set. This constitutes an improvement by approximately a factor of five compared to the previous world-leading limit by the Majorana Demonstrator.

Definition (main.tex:83):
> The collapse in the DP model is linked to gravity, and the spatial correlation is proportional to the Newtonian potential. The correlation length $R_0$ acts as a spatial cutoff parameter to avoid a divergence in the collapse rate for point-like particles, while the strength is given by the gravitational constant $G$.

Version history: v1 gave R₀ > 4.5×10⁻¹⁰ m [4.4×10⁻¹⁰ m]; v2 (24 Mar 2026) gives 4.9×10⁻¹⁰ m [4.5×10⁻¹⁰ m].

**Caution — a second, independent route to "4.9×10⁻¹⁰ m".** The identical number also appears as (a) the Majorana Demonstrator's own published DP bound, (4.94 ± 0.15)×10⁻¹⁰ m at 95% CL, and (b) in the review arXiv:2508.18822: "the DP model to $R_0\ge4.9\times10^{-10}$ m [arnquist2022search]". If your source cites Arnquist rather than XENONnT for 4.9×10⁻¹⁰ m, it is quoting the **superseded** Majorana figure (see §5).

## 3. R_DP > 2.8×10⁻⁹ m — NOT FOUND as published; best candidate has the OPPOSITE direction

I grepped the LaTeX sources of 25+ papers spanning the entire DP-bound literature (Dai 2411.17588; XENONnT 2506.05507 v1+v2; Donadi 2111.13490; Figurato 2406.18494; Majorana 2202.01343 v1–v5; Piscicchia 2301.09920; reviews 2508.18822 and 2502.19278; 2511.00644; 2608.07205; 2501.08971; 2504.06109; 2501.17637; 2601.00651; 2502.03173; 2401.04665; 2402.13057; 2503.11882; 2604.21705). **The string "2.8" does not occur in any of them.** No paper states R_DP > 2.8×10⁻⁹ m as a lower bound.

**The exact numerical match is an UPPER bound in Nimmrichter, Hornberger & Hammerer, PRL 113, 020405 (2014), arXiv:1405.2868** — a file already in your workspace at D:\csl\arxiv_src\p1405\DiffusionMeasurement_ReSub_arxiv.tex.

Their Eq. (label `eq:DPbound`, source line 263):
> The greatest detectable blurring parameter $\Sigma_\DP$ in the presence of thermal and measurement-induced noise is then mass-*independent*,
> $$\sigma_\DP < \Sigma_\DP \equiv \left[ \frac{G\hbar \varrho}{6\sqrt{\pi}\left( \hbar\omega^2 + 2\gamma k_B T \right)} \right]^{1/3} a.$$

Substituting their own "hypothetical setup" parameters from Table I (ρ = 2.0 g/cm³, T = 0.2 K, ω/2π = 100 Hz, Q = 10⁶, disc radius a = 0.4 mm) gives **Σ_DP = 2.889×10⁻⁹ m**. Since Σ_DP is *linear* in the lattice constant a, a = 0.388 mm gives **2.803 nm**. So 2.8×10⁻⁹ m sits exactly on this curve.

**Why this is not the same claim as the other two:**
- Inequality is **σ_DP < Σ_DP**, a *detectability ceiling*: for DP diffusion to remain above thermal/back-action noise, the smearing must be SMALLER than Σ_DP. Increasing σ_DP weakens the effect and makes it undetectable. This is a statement about apparatus reach, not a constraint on nature.
- It is not an exclusion: a DP model with σ_DP = 10 fm simply produces no signal in that apparatus. Nothing is ruled out.
- Their Table I also contains a bare "2×10⁻⁹" entry, but that column is Λ_SQL in **Hz** (a CSL *rate* threshold) — a different parameter again, and a likely source of conflation.

**So: treat 2.8×10⁻⁹ m as suspect provenance.** The most likely explanation is a transcription of the Nimmrichter et al. Σ_DP sensitivity ceiling (≈2.8–2.9 nm), carrying over Majorana's "R_DP" symbol and/or having its inequality direction flipped from < to >. I could not confirm this; verify against the exact source you encountered it in.

## 4. Can 8π convert any published bound to 2.8×10⁻⁹ m? No.

XENONnT supplemental (supplemental.tex:17):
> It follows the Diósi as opposed to the Penrose convention, which differ by a scaling factor of $8\pi$.

Two independent reasons this cannot produce a length bound:

(a) **Wrong type of quantity.** 8π = 25.13 multiplies a *rate/emission coefficient*, not a length. Nimmrichter's DP diffusion Eq. (`eq:DDP`) is D_DP = (Gℏ/2π²)∫d³k (k_x²/k²)|ρ̃(k)|², versus the master-equation form L_DP ρ = −(G/2ℏ)∫∫/|s₁−s₂|[M(y),[M(x),ρ]]. The π-structure differs between the generator and the derived rate; that is where the 8π lives. A factor on a rate maps to a factor 8π on a *time*, or (8π)^(1/3) = 2.929 on a length only if you first cube-root a rate — which is not a legitimate re-labelling of a measured length bound.

(b) **The arithmetic does not close.** Full ratio table (value ÷ 2.8×10⁻⁹ m):
- Donadi 2021, 5.4×10⁻¹¹ m → 0.0193
- Majorana published (superseded), 4.94×10⁻¹⁰ → 0.1764
- Majorana corrected, 2.54×10⁻¹⁰ → 0.0907
- XENONnT, 4.9×10⁻¹⁰ → 0.1750
- XENONnT best-fit, 9.1×10⁻¹⁰ → 0.3250
- LISA Helou, 4.01×10⁻¹⁴ → 0.000004
- neutron-star, 1×10⁻¹³ → 0.000036
- Penrose Ge prescription, 5×10⁻¹² → 0.0018


## 5. Diósi's original 1989 vs GGR 1990 vs Donadi 2021 — the convention history

**Diósi 1989, Phys. Rev. A 40, 1165, "Models for universal reduction of macroscopic quantum fluctuations."** (title/abstract confirmed from the APS abstract page). Its abstract claims "a new parameter-free unification of micro- and macrodynamics" using "gravitational measures for reducing macroscopic quantum fluctuations of the mass density." **In the original 1989 form the smearing length was not a free parameter** — that is the whole point of Ghirardi-Grassi-Rimini's critique below. (The PRA is paywalled; I have not read its body, so treat the internals as inferred from the GGR abstract and from later verbatim reproductions, not as verified.)

**Ghirardi, Grassi, Rimini 1990, Phys. Rev. A 42, 1057, "Continuous-spontaneous-reduction model involving gravity"** — this is the nucleon-smearing treatment you asked about. Verbatim from its abstract:
> A continuous-reduction model implying the dynamical suppression of linear superpositions of macroscopically distinguishable states, presented recently by Diòsi [Phys. Rev. A 40, 1165 (1989)], is investigated. The model exhibits appealing features; in particular, it relates reduction to gravity and contains no constants besides Newton's gravitational constant G. It turns out, however, that the model is not fully consistent. A slight modification of this model is proposed, which overcomes the difficulties... Reduction is related to gravity in the same way as in Diòsi's model, but **a fundamental length must be introduced to avoid inconsistencies.**

This is the origin of the free parameter. Nimmrichter et al. cite precisely this paper for it: "one must introduce a blurring parameter σ_DP > 0 to account for the fact that the DP collapse effect diverges for point masses [Ghirardi1990a, Bassi2003]. We model each nucleus in the crystal lattice as a **Gaussian mass distribution of width σ_DP**."

**Donadi et al. 2021, Nature Phys. 17, 74** re-derives the same thing and renames it R₀, the *spatial cut-off*. Their Eq. (2) is the Lindblad/commutator form
dρ/dt = −(i/ℏ)[H,ρ] − (4πG/ℏ)∫d³x∫d³y (1/|x−y|)[M(y),[M(x),ρ]], M(x) = Σ_n μ_n(x, x̂_n),
which they state is "equivalent to the master equation derived in [diosi1987universal, diosi1989models]". They then adopt **Penrose's** physical prescription for R₀: the nuclear wave-function spread in the material, R₀ = √⟨u²⟩ = √(B/8π²) with the Ge Debye–Waller factor B = 0.20 Å² at liquid-nitrogen temperature, giving R₀ = 0.05×10⁻¹⁰ m. Their result R₀ > 0.54×10⁻¹⁰ m (Eq. `eq:lowbound`, 95% probability) therefore **excludes Penrose's own prescribed value** by about one order of magnitude.

So the evolution is: **Diósi 1989 (no free length, claimed parameter-free) → GGR 1990 (a fundamental length is unavoidable; Gaussian nuclear smearing) → Donadi 2021 and after (length is an explicit free parameter, bounded from below by radiation searches).**

No published value equals 2.8×10⁻⁹ m, and none is 8π or (8π)^(1/3) times another. Even granting the illegitimate cube-root rescaling, 2.8×10⁻⁹ ÷ 2.929 = 9.559×10⁻¹⁰ m is the required input — a number no paper reports (the nearest is XENONnT's 9.1×10⁻¹⁰ m best-fit, which is a best-fit, not a bound, and gives 2.67×10⁻⁹ m).



## 6. Units/definition conversion, or genuinely different physical claims?

**Both, and they are separable.** Concretely:

- **σ_DP and R₀ are the same physical object** — the width of the smeared mass-density function μ(x). The symbol differs (σ_DP in the LISA/optomechanics community; R₀ in the radiation community; R_DP only in Majorana). 285.5 fm and 4.9×10⁻¹⁰ m are the same claim *type* (experimental lower bounds) on the same quantity, from two different experiments, and they are mutually consistent: 4.9×10⁻¹⁰ m is ~1700× larger than 285.5 fm = 2.855×10⁻¹³ m. Nothing needs converting. They differ by 3+ orders because they constrain different regimes: LISA constrains the centre-of-mass diffusion of a ~2 kg solid; XENONnT constrains spontaneous X-ray emission from xenon atoms. Both are "smaller R₀ = stronger effect = excluded", so both point the same way.
- **The 2.8×10⁻⁹ m value is NOT reconcilable by units or definitions.** It has the opposite inequality direction, depends on an apparatus-dependent lattice constant a, and is a sensitivity ceiling rather than a physical exclusion. It is also ~5.7× *above* the XENONnT bound, so if it were a lower bound it would be a *weaker* constraint, not a conflicting one — another sign of transcription trouble.
- **A fourth category you should not confuse with these: a model input.** Penrose's R₀ = 0.05×10⁻¹⁰ m for cooled germanium (and equivalently the nuclear/nucleon scale ~1 fm) is a *prescribed physical value*, not a measurement. It is the value the experiments exclude. Do not put it in the same column as the bounds.

## 7. For your own derivation

- If you are deriving a DP rate/diffusion, state which regularisation you assume: **Gaussian width σ** (Nimmrichter, GGR) vs **spatial cut-off R₀** in the Donadi rate, dΓ/dE = (Ge²/(12π^{5/2}ε₀c³R₀³E))(N_p² + N_e) (Piscicchia et al. 2024, arXiv:2301.09920). The 1/R₀³ scaling is what makes every one of these a *lower* bound.
- Dai et al.'s LISA formula is **1/σ_DP³ against a noise floor**; do not reuse it for a radiation-emission calculation — different observable, different prefactor (2ℏG/3√π vs Ge²/12π^{5/2}ε₀c³).
- Your D:\csl\arxiv_src\p2608\Davila_Milburn_2026a_PRD.tex manuscript uses a point-particle proxy Λ_DP = Gm²τ/(ℏ d_eff) and the homogeneous-sphere DP kernel K_DP^sph(a) = a²/2 − 3a³/16 + a⁵/160. Note that proxy **omits the regularisation length entirely** — it is a d_eff-rescaled Newtonian self-energy, not a σ_DP-regularised one. That is a defensible modelling choice, but it means your crossover scale is not directly comparable to any of the three numbers above unless you reintroduce a regularisation and fix its convention.
- Sanity anchors you can check against: Penrose/Donadi Ge prescription 5×10⁻¹² m; Donadi 2021 5.4×10⁻¹¹ m; Majorana corrected 2.54×10⁻¹⁰ m; XENONnT 4.9×10⁻¹⁰ m; Figurato et al. upper bound R₀ ≲ 10⁻⁴ m (and R₀ ≲ 10⁻⁷ m for L ~ 1 μm objects). The experimentally viable window for the free-parameter model is roughly 5×10⁻¹⁰ m ≲ R₀ ≲ 10⁻⁴ m.

## 8. Open items / limitations

- Could not retrieve the Majorana erratum full text (APS paywall); the corrected value 2.54×10⁻¹⁰ m is confirmed from arXiv v5 source text, and the erratum citation 10.1103/PhysRevLett.130.239902 is confirmed from the APS erratum landing page.
- XENONnT's "improvement by approximately a factor of five" over Majorana does not match the *current* corrected Majorana bound (4.9×10⁻¹⁰ ÷ 2.54×10⁻¹⁰ ≈ 1.9). XENONnT appears to compare against a pre-erratum/90% figure. Flagging as an inconsistency in the literature, not a resolved point.
- The 2.8×10⁻⁹ m provenance is an inference from numerical coincidence, not a confirmed source. Treat as unverified until you locate the document you saw it in.
- The 1989 PRA body was not read (paywalled); claims about its internal normalisation (Gaussian vs uniform, variance convention) are unverified.
