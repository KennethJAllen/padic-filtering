# References

Everything this project builds on, with the pointer to where each result is
actually used. Two lines of work meet here: the Caruso–Roe–Vaccon (CRV)
lattice-precision recursion, which supplies the prediction step, and the
Allen–DeMark–Petsche (ADP) analysis of non-archimedean Hénon maps, which
supplies the dynamical regime and every parameter set in
`padic_filtering/params.py`.

## Precision tracking (the prediction step)

- Caruso, Roe, Vaccon, *Tracking p-adic precision*, LMS JCM **17** (2014)
  274–294, [arXiv:1402.7142](https://arxiv.org/abs/1402.7142). The recursion
  `v ← f(v)`, `H ← f'(v)·H` on a coset `v + H`. §3 (Lattices and
  differentials) is the section to read first.
- Caruso, Roe, Vaccon, *p-adic Stability in Linear Algebra*,
  [arXiv:1506.05644](https://arxiv.org/abs/1506.05644). Lattice-vs-naive
  comparisons; a useful template for the plots.
- Caruso, Roe, Vaccon, *ZpL: a p-adic precision package*,
  [arXiv:1802.08532](https://arxiv.org/abs/1802.08532). The Sage
  implementation (`sage.rings.padics.lattice_precision`) that
  `experiments/exp_5_6_baseline.py` compares against when Sage is available.
- Roe's research statement, <https://math.mit.edu/~roed>. The coset `v + H`
  framing in one paragraph, plus an explicit note that scaling with dimension
  is the method's main weakness — which is why this project stays in
  dimension 2.
- <https://github.com/roed314/padicprec> — their code.

The gap this project fills: the CRV recursion has no *update* step, because in
a deterministic computation there is nothing to condition on. Inverting an
automorphism supplies one.

## The Hénon dynamics

Allen, DeMark, Petsche, *Non-archimedean Hénon maps, attractors, and
horseshoes*, [arXiv:1610.04271](https://arxiv.org/abs/1610.04271) — referred to
throughout as **ADP**. A copy of the PDF may be kept in the repository root as
`1610.04271.pdf`; it is gitignored, not committed.

### Dictionary between the two normal forms

ADP use `φ_{a,b}(x, y) = (a + b y − x², x)` with `det Dφ = −b`. This repository
uses `f(x, y) = (y, y² + c − δx)` with `det J = δ`. The two are
affine-conjugate via `(x, y) ↦ (−y, −x)`, so

```
a = −c,     b = −δ,     |a| = |c|,     v_p(b) = v_p(δ) = m
```

and ADP's parameter-space partition reads, in our parameters:

```
H_I    : v_p(c) ≥ 0, m = 0          good reduction; unimodular; a no-op
H⁺_II  : v_p(c) ≥ 0, m ≥ 1          attractor; anisotropy demo only
H_III  : |c| > max(1, |δ|²)         horseshoe; the filtering demo
```

`padic_filtering/params.py` carries the same dictionary and cites the ADP
result behind each individual parameter set; this file is the bibliography, not
a second copy of that table.

### Results used

- **Thm 1(a)** — the filled Julia set is empty unless `a` is a square in `ℚ_p`.
  Every horseshoe parameter set is built to satisfy this.
- **Thm 1(e)** — on `J(f)`, `f` is conjugate to the two-sided 2-shift; there
  are `2^ℓ` points of period `ℓ`, indexed by symbol sequences, all in `ℚ_p²`,
  with `|x| = |y| = p^s`. This is what supplies certified ground-truth orbits,
  and the shell condition `v(x_t) = v(y_t) = −s` is the *only* thing the proofs
  in `NOTE.md` §3 take from ADP.
- **Thm 2** — over `ℚ₃` with `a ≡ 2 (mod 9)` and `b = 3` the attractor is
  infinite and orbits equidistribute with respect to an SRB-type measure: long,
  not-eventually-periodic test orbits in `H⁺_II` (`ATTRACTOR_3ADIC`).
- **§4.4, Table 1** — attractor candidates over `ℚ₅` and `ℚ₇`. The
  infinite-attractor entries `(a, b) = (1, 5)` and `(2, 7)` are *conjectural*
  and flagged as such in `params.py` (`proven=False`); `(4, 5)` over `ℚ₅`
  (3-cycle) and `(1, 7)` over `ℚ₇` (2-cycle) are finite cycles, kept in
  `params.py` only to be tested *against* — on them the orbit collapses and
  tangency statistics go trivial.
- **Lemmas 23–24** — Lipschitz-`|b/γ|` stable/unstable tubes with product of
  constants `< 1`. These were the original de-risking argument for the
  smoother's transversality. The proof that landed does **not** use them: the
  invariant-cone induction of `NOTE.md` §3 gives defect `0` directly. Kept here
  for context only.

### Extras not currently used

- **Prop. 3** — explicit fixed points and 2-cycles, an alternative source of
  short test orbits to the Newton solver in `henon.periodic_orbit`.
- The involution `ι(a, b) = (a/b², 1/b)`, under which `H⁻_II` mirrors `H⁺_II`
  by time reversal. This is a free consistency check: it should swap the roles
  of the forward and backward passes. Not implemented.

## Periodic orbits from itineraries

The recipe `henon.periodic_orbit` implements: a period-`ℓ` orbit solves the
circulant system

```
x_{t+2} = x_{t+1}² + c − δ·x_t      (indices mod ℓ)
```

Multivariate Newton from the initial guess `x_t = ε_t·γ`, with `ε ∈ {±1}^ℓ` the
chosen itinerary and `a = γ²`, converges quadratically because hyperbolicity
makes the Newton Jacobian invertible — the shadowing lemma in computational
form, with ADP §5's contraction argument guaranteeing one solution per
itinerary. Periodicity is then verified by exact forward iteration. The
simplest case is a fixed point `(α, α)` with `c = α + δα − α²` and
`v_p(α) = −s`; `p = 3, δ = 1, α = 1/3, c = 5/9, s = 1` is the worked example
that `review_checks.py` uses.

## Prior art for the contribution itself

No prior work adds an update step or a smoother to the CRV recursion. This was
checked against the CRV line above, ZpL, and general searches for
ultrametric/p-adic filtering; the p-adic shadowing literature is qualitative
and forward-time. The prediction-only framing is all that exists. See
`docs/THEOREM.md` §7 for the full positioning argument.
