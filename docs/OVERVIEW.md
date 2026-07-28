# Precision tracking as a filtering problem, and the smoother that was missing

*A short overview for a reader deciding whether this is worth their time.
The mathematics, with full proofs, is in the companion note `NOTE.md`; this
document duplicates none of it. Kenneth Allen, 2026-07-27.*

---

## 1. Motivation: the update step is missing

When you compute over `Q_p` you carry finitely many digits, and every operation
degrades them. The standard way to say this precisely is Caruso–Roe–Vaccon's
lattice-precision framework ("Tracking p-adic precision", LMS JCM 2014): don't
carry a scalar digit counter, carry a **lattice** `H ⊂ Q_p^n`, so that the true
value is guaranteed to lie in the coset `v + H`, and push it forward through a
map `f` by its differential, `H ↦ Df(v)·H`. The lattice is sharper than a
counter because it knows that error in different directions has different sizes,
and it is *exact* rather than heuristic: as long as the linearisation absorbs
the map's nonlinear remainder, the coset is not an estimate of where the true
value is, it is a proof.

That framework is structurally **forward-only**. In filtering language it is
exactly the *prediction* step of a Kalman filter: propagate the uncertainty
through the dynamics, and watch it grow. What a filter does next is the
*update* step — fold in a measurement — and what a Rauch–Tung–Striebel smoother
does is run the whole thing backwards as well and intersect the two, so that the
estimate at an interior time is conditioned on information from both sides.
In the p-adic setting the update step has no obvious meaning: there are no
noisy observations to fold in.

The observation this project is built on is that for an **automorphism** there
is a second source of information, and it is free. If `f` is invertible and you
know the endpoint `v_T` to `k` digits, you can run the differential *backwards*
from time `T` and obtain a second certified coset at every interior time. Both
cosets contain the truth, so their intersection does too. That intersection is
the update step, and the resulting two-sided pass is a genuine smoother — exact
rather than Gaussian.

Whether this buys anything is not obvious in advance. The forward pass loses
digits in the unstable direction; the backward pass loses them in the stable
direction; whether the two cosets are transverse enough for the intersection to
be small is a real question, and over `R` the analogous construction always pays
a *wrapping* tax — interval methods lose a constant per step even in the best
case. The answer here turns out to be as good as it could possibly be, and for a
reason specific to the ultrametric.

The test case is the p-adic Hénon map `f(x,y) = (y, y² + c − δx)`, restricted to
its horseshoe, where Allen–DeMark–Petsche (arXiv:1610.04271) prove `f` is
conjugate to the two-sided 2-shift and every point of the invariant set has
`|x| = |y| = p^s`. The eigenvalue valuations are `{−s, s+m}` with
`m = v(δ) ≥ 0`: genuine expansion and genuine contraction.

## 2. The main idea in one page

Fix a window `0 ≤ t ≤ T`. Start from balls of radius `p^{-k}` at both ends, push
one forward and one backward, and intersect:

```
    H_t^F = J_{t−1}⋯J_0 · p^k Z_p²        (prediction from time 0)
    H_t^B = (J_{T−1}⋯J_t)^{−1} · p^k Z_p²  (prediction from time T, backwards)
    H_t   = H_t^F ∩ H_t^B                  (the smoother)
```

The forward pass loses `s` digits per step in its worst direction; the backward
pass loses `s+m` per step in *its* worst direction. The point is that these are
different directions. The forward pass's worst direction is the unstable one,
the backward pass's is the stable one, and each pass is *sharp* in the direction
the other one is losing. Intersecting recovers what each threw away.

What makes this exact rather than merely helpful is that over `Q_p` the two
directions are not just uniformly transverse — they are **orthogonal**, in the
strong sense that the two terms of the relevant `2×2` determinant have
valuations differing by `2s + m > 0`, so ultrametric addition can never cancel
them. There is no wrapping effect to pay. The transversality defect is `0`,
identically, not asymptotically. That single margin is what the whole proof runs
on, and it is why the archimedean analogue of this statement is an inequality
while this one is an equality.

![headline](../results/headline.png)

The figure is the headline case: `p = 3`, `s = 1`, `m = 0`, starting precision
`k = 80`, a window of `T = 24` steps. Forward-only tracking falls from 80
digits to 56, linearly, and — worth saying out loud before a referee does —
the lattice buys *nothing* over a naive scalar digit counter in its worst
direction here; the two curves coincide exactly. The lattice's whole advantage
in the prediction-only setting is anisotropy (right panel), which grows at
`2s+m`. The smoothed track never drops below its starting 80 digits and peaks
at 92 in the middle of the window. Bounded precision comes from the update
step, not from the lattice representation.

## 3. What is proved

Three statements, all in `NOTE.md` with complete proofs.

**Theorem A (the four-divisor law).** For every orbit in the horseshoe, every
`T`, `t`, `k`, all four elementary divisors of the three lattices are given by
an exact closed formula. In particular the worst-direction precision of the
smoother is

```
    d₁(H_t) = k + min((s+m)·t, s·(T−t)) ≥ k .
```

A tent, and a **skewed** one: it peaks at `t* = sT/(2s+m)`, not at `T/2`, and
the symmetric form `k + s·min(t, T−t)` that one writes down first is a valid
lower bound but is not tight when `δ` is not a unit. The information-theoretic
loss over the window is exactly zero, and the interior of the window is
strictly *better* determined than its endpoints.

The proof is two short valuation inductions plus a change of frame — no
estimate, no constant, no periodicity, and no use of the shift conjugacy. It
consumes exactly three facts: `v(x_t) = v(y_t) = −s` at each point, `det J = δ`,
and the margin `m + 2s > 0`. The first of these is a *hypothesis* of the note
(the `s`-admissible sequences of `NOTE.md` Definition 1.1), not an imported
theorem; ADP's only role anywhere is to guarantee that the hypothesis is
non-vacuous by producing the horseshoe on which it holds. Because the proof
never uses a relation between consecutive points, it covers aperiodic orbits
for free — indeed sequences that are not orbits at all.

**Theorem B′ (the certified budget, an iff).** Theorem A is about *idealised*
lattices — pure Jacobian products. An algorithm cannot compute those; it
propagates a coset, and each propagation is exact only while the map's
quadratic remainder is absorbed by the propagated lattice. Every step of both
passes is exact **if and only if**

```
    k ≥ (3s + 2m)(T − 1).
```

Under that budget the computed objects literally *are* the idealised ones, and
the truth is certified to lie in the smoothed coset at every interior time. (The
`iff` is about exactness, which is what the budget buys. Below it the truth is
still enclosed — the tracker inflates rather than lying — but in a larger
lattice that Theorem A no longer describes; that is Open Problem 1.)

This is sharp in both directions, and it is where the cost went. The `O(T)` did
not vanish; it moved out of the answer's accuracy and into the computation's
word size. That trade is precisely what a filtering framing predicts and what a
forward-only framing cannot express. The exchange rate — digits of starting
precision per step of certified window — is `3s+m` forward-only and `3s+2m`
two-sided, reducing to `3s` when `δ` is a unit. Note the rate is the *backward*
one: an earlier guess based on the forward rate is insufficient as soon as
`m > 0`, and `NOTE.md` §4.5 gives an explicit counterexample.

**Theorem C (the linear-algebra core).** The proof of Theorem A mentions the
Hénon map only through one lemma, and that lemma only checks two cone
conditions. Removing the map leaves a theorem about sequences in `GL₂(Q_p)`:
if each `J_t` maps a fixed unstable cone into itself with valuation gain
exactly `−s`, each `J_t^{−1}` maps a stable cone into itself with gain
`−(s+m)`, and `v(det J_t) = m`, then the forward and backward products obey the
same four-divisor law, with Theorem A as a corollary. Nothing relates
consecutive `J_t`; no companion or other structure is imposed on any of them.

One honest qualification, which I would rather state than have a referee find.
The axioms are less permissive than "arbitrary cone-preserving matrices"
sounds: they *force* every admissible matrix to have the valuation shape of a
Hénon companion matrix (`NOTE.md` Lemma 6.2 — the entries stay free, their
valuations do not). So the correct headline is not "the law needs nothing about
Hénon" but "the law needs nothing about Hénon beyond a valuation shape the cone
axioms themselves force". What is genuinely gained is the removal of the orbit
relation and of the companion *form* — which is what the intended application
needs: by Friedland–Milnor every dynamically nontrivial polynomial automorphism
of the plane is a composition of generalised Hénon maps, and Theorem C covers
their Jacobian sequences whenever the factors share one `(s, m)`. The missing
ingredient there is dynamical, not linear-algebraic: an ADP-style horseshoe
theorem for compositions.

**Open Problem 1, stated as open.** Below Theorem B′'s budget some step
inflates and the certified lattices become strictly larger than the idealised
ones. Numerically the *conclusion* `d₁(H_t) ≥ k` survives well past that point,
because early inflation lands where the other arm of the tent is binding. I
have no proof of this and the threshold at which it fails is unknown: past the
first inflation the lattices are no longer Jacobian products and the inductions
break. Any claim that the result "degrades gracefully" below the budget is a
conjecture, and I would like it read as one.

## 4. Certified versus measured

Everything computational here is **exact**: integers and rationals, never
floating point, and every claim is checked against independently constructed
ground truth rather than assessed statistically. Concretely, `v_true ∈ v + H` is
*asserted* at every step of every track — 247,325 certified steps across 10⁴
random orbits with zero violations, against a mutation control (inflation
disabled) that is caught on 2,976 of them, so the assertion has teeth. The
lattices are also measured to be *minimal*: slack `0` at every step, so the
enclosure is not merely correct but tight.

What that buys is that the numbers below are not evidence for the theorems in
the statistical sense; each is a finite exact verification of an equality, and
each identity used in the proofs was pinned as an exact assertion *before* it
was relied on. The four-divisor law was confirmed as an equality (`C = 0`, not
a bound) with defect `0` throughout, over periods 1–20, `T = 4…100`,
`k = 0…194`, `p ∈ {3,5,7,11}`, `s ∈ {1,2,3}`, `m ∈ {0,1,2}`; periodicity was
ruled out as an artefact on 15 windows cut from random length-200 itineraries.
The repository runs 215 unit tests and ten experiments, each writing a figure
and a JSON record of every parameter and per-step number.

![theorem](../results/exp_theorem.png)

This figure is also where the project corrected itself, which is the reason to
show it. The original conjecture was `d₁(H_t) ≥ k + s·min(t, T−t) − C` for some
constant `C` uniform in `T`. Measurement said `C = 0` — better than conjectured
— but *also* that "uniform in `T`" is false without the budget hypothesis: at
fixed `k = 40` the constant is `0, 0, 3, 14, 24, 46` for `T = 16, 20, 24, 32,
40, 56`. The cause is the quadratic-remainder budget, not the transversality
that the constant was expected to come from; the defect is `0` in every
configuration measured, including all of those with `C > 0`.

## 5. Positioning

Against **Caruso–Roe–Vaccon**: their framework is forward-only, and I found
nothing in that line, or citing it, that exploits invertibility to run the
differential backwards and intersect. Searches for a p-adic or ultrametric
Kalman/smoother analogue come up empty. Against the **ADP line** in
non-archimedean dynamics: the recent p-adic shadowing literature is qualitative
and forward-time, so a digit-exact *two-sided* shadowing statement — endpoints
known to `k` digits determine the interior exactly, with a certified cost — is a
different kind of result. Against **validated numerics over `R`**: multiple
shooting is classical, but interval methods always pay the wrapping tax, and
the exportable point is that the ultrametric analogue is an equality because
transversality is orthogonality.

This resolves no named open problem. The claim is: first two-sided precision
tracking in the ultrametric setting, with an exact law and an exact
certification cost. The referee risk is not falsity but "Theorem A is easy once
stated" — to which the defence is that the statement is the contribution.

## 6. What I would like from you

Concretely, in decreasing order of how much a second opinion would change what
I do next:

1. **Is Theorem C stated at the right level?** Given that the cone axioms force
   the companion valuation shape (Remark 6.7), is the generalisation worth
   presenting as a theorem about matrix sequences, or is it more honest as a
   remark inside the Hénon proof? This decides what the paper is about.
2. **Open Problem 1** — is the sub-budget regime worth attacking, and does the
   max-recursion on inflated divisor pairs look tractable to you, or is the
   right move to state the budget as a hypothesis and stop?
3. **Venue and framing.** The filtering framing is what makes the result
   legible to a computational-algebra audience, and irrelevant to a dynamics
   audience. Res. Number Theory, an LMS JCM-style venue, and ISSAC all look
   plausible and want different framings.
4. **Is the CRV literature really silent on this?** A negative literature
   result is the kind of thing a second pair of eyes catches. If a p-adic
   two-sided or smoother-like construction exists somewhere, I would rather
   know now.
5. **The one gap in the numerics** is a SageMath cross-check of the
   prediction-only pass against `sage.rings.padics.lattice_precision`, which is
   not installed here; `results/exp_5_6_baseline.json` records exactly what to
   run. It is the first thing a CRV referee will ask about.

**Technical companion:** `NOTE.md` — ~1000 lines, self-contained in the strong
sense that no proof in it invokes an external result; ADP and CRV are cited for
non-vacuity and for provenance of the formalism respectively, and neither is
assumed. §3 proves Theorem A, §4 Theorem B′, §6 Theorem C, and §8 gives the
reproduction instructions.
