# Roadmap — open work

What is left, in priority order, with enough detail to start a session on any
line of it. The mathematics is in `NOTE.md`; the numerical record and the
generalisation targets are in `THEOREM.md` §7–§8. This file is the plan only.

Everything in the original specification is built and green: `uv run pytest`
(215 tests, ~5 s) and `uv run python experiments/run_all.py` (10 experiments,
~230 s). Theorems A, B′ and C are proved.

## Priorities

| tier | item | effort | value |
|---|---|---|---|
| 1 | **D** — measure, then close, the `C = 0` threshold (Open Problem 1) | 1 session | highest remaining *mathematical* value |
| 2 | **F** — `f²` composition sanity check | 1 h | cheap probe of `NOTE.md` §7 item 2 |
| 2 | **H** — fixed-lag smoother | 1 session | the Kalman analogy the write-up promises |
| 2 | **G2** — does the budget see the degree? | 1–2 sessions | needs package changes |
| 2 | **I** — SageMath baseline | 1 h + install | first thing a CRV referee asks |
| 3 | **J** — paper assembly | 2 sessions | unblocked |
| 4 | deferred: horseshoe regime for compositions; `𝔽_p((T))`; `n > 2`; residue characteristic 2 | — | do not start mid-session |

Suggested session boundaries: **D** alone; then **F + H** together (both are
about compositions and windows); then **I** if Sage is available, then **J**.
D is the one that changes what the paper is *about*.

---

## D. Open Problem 1: the exact threshold for `C = 0`

This is the honest remaining gap, and it is *not* what Theorem B′ closes.
Theorem B′ gives the threshold at which the two passes stop being exact. But
`C = 0` survives well past it: at `s=1, m=1, T=16, k=70` the backward pass
inflates and `C` is still `0`. Theorem B's clause "without that hypothesis the
conclusion degrades gracefully, at a rate governed by the exactness horizon" is
**not proved anywhere** — `NOTE.md` says nothing about the post-horizon regime,
and states this as Open Problem 1.

So there are two thresholds, and only the first is understood:

```
k_exact(T) = (3s+2m)(T−1)          proved, an iff      (NOTE.md Prop 4.4)
k_C(T)     = ?                     ≤ k_exact,  unproved
```

### The data that exists

From `THEOREM.md` §0.2 / `exp_theorem.py` block C, at `T = 16, 20, 24, 32, 40,
56`:

```
k = 40:   C = 0, 0, 3, 14, 24, 46
k = 60:   C = 0, 0, 0,  0, 11, 32
```

Both look like `C ≈ 1.35·(T − T₀(k))` with `T₀` linear in `k` — which is *not*
a clean integer law, and a non-integer slope in an exact integer computation is
a sign that the right variable has not been found yet.

### Plan

1. **Measure `k_C` precisely first.** Bisect on `k` for each `(s, m, T)` in a
   grid, recording the largest `k` with `C > 0`. Each evaluation is one
   forward + backward + smoother run, ~0.1 s at these sizes. New
   `experiments/exp_c_threshold.py`. Do not theorise before this table exists —
   see the "measure before theorising" trap in `DEVELOPMENT.md`.
2. Separately record *where* in the window `C` is attained and which pass
   inflated first. The hypothesis to test: `C > 0` only once inflation reaches
   a `t` on the *binding* arm of the tent, i.e. `t ≤ t* = sT/(2s+m)` for
   backward inflation.
3. Only then attempt the proof. Past the first inflation the lattices are no
   longer Jacobian products, so `NOTE.md` §3's inductions do not apply
   directly; the object to track is the inflated lattice's divisor pair, which
   satisfies a max-recursion rather than a product recursion.

**Done when:** either `k_C` has a proved closed form, or `NOTE.md` gains a
clearly labelled **Conjecture** next to Open Problem 1, with the measured table
and the exact boundary it must reproduce. A labelled conjecture with data is an
acceptable outcome; a hand-wave is not.

## F. `f²` composition sanity check

Compose Jacobians of `f` with itself, check the rates read `2s` and `2(s+m)`
and that the tent holds with those rates on the existing machinery. No new code
beyond composing Jacobians. Do it as a block in `exp_theorem.py`, not a new
file. It is the smallest possible probe of the composition story in
`NOTE.md` §7 item 2 — and note the caveat there: Setup 6.1 fixes *one* pair of
cones, so a Friedland–Milnor composition is covered by Theorem C only when
every factor shares the same `s` and the same `m`. The per-cycle rates
`Σ s_i`, `Σ(s_i + m_i)` for mixed factors need a moving-cone version of
Theorem C and remain conjectural. **Do not quote the composition result without
that caveat.**

## H. Fixed-lag smoother

The current smoother is offline: it needs the whole orbit. A lag-`L` window
gives an *online* filter with bounded memory. With Theorem A in hand the result
is predictable in advance — state the prediction before running it, as usual.
For a lag-`L` window ending at the current time, the smoothed divisor at the
trailing edge should be

```
d₁ = k        at the trailing edge,
```

with the interior of each window on the same skewed tent. So the online filter
holds `k` digits at bounded memory and a working precision of `(3s+2m)·L`
rather than `(3s+2m)·T`. That is the statement the Kalman framing has been
promising and has not delivered. One experiment, `exp_fixed_lag.py`, and one
paragraph in `NOTE.md`.

## G2. Does the budget see the degree?

G1 — the four-divisor law for degree-`d` companion Jacobians
`[[0,1],[−δ, P′(y)]]` — is **done**, as `NOTE.md` Corollary 6.6; it fell out of
Theorem C in a paragraph.

G2 is the expensive half. The quadratic remainder generalises to
`Σ_{j=2}^{d} P^{(j)}(y)/j! · h_y^j`, and the claim to test is that `j = 2`
stays binding. Done honestly this needs actual degree-3 orbits, hence a
horseshoe regime for `deg P = 3`, which ADP does not supply. **This is the
first item that requires touching package code**, so scope it deliberately:
`padic_filtering/henon.py` hardcodes the quadratic in `f`, `f_inv`, `jacobian`,
`quadratic_remainder_exponent` and `periodic_orbit`'s Newton system.

An honest intermediate that avoids the regime problem: run the membership test
on *synthetic* admissible sequences with prescribed `v(P^{(j)}(y))`, and see
which `j` binds as a function of those valuations. That answers the actual
question — "does the budget see the degree" — without proving a new horseshoe
theorem.

## I. SageMath baseline

The one part of the de-risking plan never executed, and the first thing a CRV
referee will ask about. `experiments/exp_5_6_baseline.py` already detects Sage
and skips with instructions; `results/exp_5_6_baseline.json` records exactly
what to run (`sage.rings.padics.lattice_precision`, matched against the
prediction-only `(d₁, d₂)` from `exp_5_1_anisotropy.json`). Needs an install,
not thought. Until then the independent baseline is `review_checks.py`, which
shares no code with the package.

## J. Paper assembly

`NOTE.md` → LaTeX (pandoc gets most of the way; the code fences become
`align*`). Add `THEOREM.md` §7's literature positioning as an introduction
section, and use `OVERVIEW.md` as the first draft of the motivation. If the
generalised statement leads, Theorem C is the main theorem and Hénon is the
worked corollary.

Four slide-ready figures already exist:

- `results/headline.png` — the two headline plots;
- `results/exp_5_3_smoother.png` — three panels: the tent, all periods, `H⁺_II`
  vacuity;
- `results/exp_theorem.png` — the constant `C`;
- `results/exp_aperiodic_window.png` — the tent on windows of effectively
  aperiodic orbits.

Venue profile: Research in Number Theory, LMS JCM, or ISSAC.

---

## Non-goals

These are not "not yet"; they are out of scope by decision.

- **No factoring.** The filtering framing does not apply to Pollard rho, and
  claiming it does would discredit the rest.
- **No invented measurement oracles as a headline claim.** The backward pass is
  justified; "suppose someone tells us three digits" is not. The oracle update
  exists in `precision.py` only as a machinery test and is labelled as such.
- **No noisy-HNP / ECDSA angle.** Separate project, different reviewers.

## Deferred generalisations

Listed so the write-up can claim them as future work, and so nobody "improves"
the demo into them mid-build. The mathematical targets themselves are stated in
`NOTE.md` §7 (moving cones for mixed compositions, degree-`d` budgets,
`𝔽_p((T))`, `n > 2`, residue characteristic 2) and are not repeated here. What
follows is the framing to keep alongside them:

- **Hénon is the generic case, not a special one.** The filter uses only
  invertibility, a constant Jacobian determinant, and a terminating Taylor
  expansion — degree `d` gives `d − 1` extra terms in the exactness test — and
  by Friedland–Milnor every dynamically nontrivial polynomial automorphism of
  the plane is a composition of Hénon maps. Say so in the write-up.
- **Dimension scaling.** The smoother argument is dimension-agnostic and HNF is
  not harder in higher dimension, just bigger. The obstacle is dynamical:
  identifying hyperbolic regimes with known invariant sets, where nothing
  analogous to the ADP horseshoe is proven. CRV name dimension scaling as their
  method's main weakness, so bounded-precision smoothing in higher dimension is
  where the payoff would be largest.
- **Genuinely noisy measurements in 2D.** The 1D tree posterior in
  `probabilistic.py` is the seed of ultrametric Bayesian filtering (p-adic
  HMMs). Different machinery — measures, not lattices — and a different
  project.
