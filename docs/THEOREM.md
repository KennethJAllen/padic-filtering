# The question and the theorem

Companion to `NOTE.md` (which carries the proofs). This file
states what the project is actually *for*. Status (updated 2026-07-27): the
theorems are **proved** — Theorem A (unconditional, aperiodic orbits
included) and the sharp certified budget `k ≥ (3s+2m)(T−1)` (Theorem B′, an
iff) are proved in **`NOTE.md`** §3–§4. This file remains the motivation, the
numerical record, the literature positioning (§7) and the generalisation
roadmap (§8); where a constant here disagrees with `NOTE.md`, `NOTE.md` wins
— in particular the certified budget is `(3s+2m)(T−1)`, not `(3s+m)T + s+m`
(`3s+m` is the forward rate only; superseded statements below keep their
`[NUM]`/`[PROVED]` markers rather than being rewritten).

---

## 0. Numerical status (2026-07-26) — READ FIRST

The specification is now implemented, and §3's constant `C` has been measured
directly rather than assumed bounded. Everything below is reproduced by
**`experiments/exp_theorem.py`** (exact arithmetic, assertions, JSON in
`results/exp_theorem.json`). Corrections are marked **[NUM]** in place.

1. **The mechanism holds, exactly.** For the lattices §2 actually defines —
   pure Jacobian products, no inflation — `C = 0`, with **no hypothesis at
   all**: verified for `T` up to 100, `k` down to 0, nine itineraries of
   period 1–20, `p ∈ {3,5,7,11}`, `s ∈ {1,2,3}`, `m ∈ {0,1,2}`. Not "bounded by a
   constant" — the tent is *attained*, as an equality in all four divisors.
   **The tent is also skewed**: the backward pass loses `s+m` per step where
   the forward loses `s`, so the sharp law is
   `d₁(H_t) = k + min((s+m)·t, s·(T−t))`, peaking at `t* = sT/(2s+m)` rather
   than `T/2`. The symmetric form `k + s·min(t, T−t)` in §3 is the `m = 0`
   case and is a lower bound otherwise (§3 [NUM]).

2. **The theorem as literally stated is false**, because it omits a
   hypothesis. `C = C(p,c,δ)` independent of `T` fails at *fixed* `k` once
   the window outruns the exactness horizon (`NOTE.md` §4):

   ```
   k = 40 (horizon 15):  T = 16  20  24  32  40  56
                         C =  0   0   3  14  24  46
   k = 60 (horizon 22):  C =  0   0   0   0  11  32
   ```

3. **The repair is one hypothesis, and it strengthens the conclusion.** With
   `k ≥ (3s+m)·T + O(1)` — the precision budget that keeps the linearisation
   exact — `C = 0` for every `T` tested (up to 64), and the conclusion
   becomes an *equality*, not an inequality. See §3 [NUM].
   **[Superseded 2026-07-27: the sharp budget is `k ≥ (3s+2m)(T−1)`, an iff
   (`NOTE.md` Thm B′); `(3s+m)T` is sufficient only when `m = 0`.]**

4. **The failure mode is not the one §4 item 2 is about.** Where `C > 0`, the
   cause is the quadratic-remainder budget, not loss of transversality: the
   transversality defect (valuation of `det` of the two passes' minimal
   vectors, minus `d₁(H^F) + d₁(H^B)`) is **0 in every configuration
   measured, including all those with `C > 0`**. The stable/unstable
   directions never came close to aligning.

5. **Certification never failed:** 0 violations in 247,325 certified steps
   over 10⁴ random orbits, with a mutation control (inflation disabled) that
   *is* caught on 2,976 of them — so the assertion has teeth
   (`experiments/exp_5_4_certification.py`).

6. **The periodicity worry is retired at the referee level** (added later on
   2026-07-26). All of the above used short itineraries (period ≤ 20). Now
   `experiments/exp_aperiodic_window.py` cuts `T`-step windows (`T = 24, 48`,
   three positions) out of orbits built from *random length-200 itineraries*
   (minimal cyclic period asserted to be 200), so the Jacobian sequence seen
   by the window carries no period. On all 15 windows, over
   `(s,m) ∈ {(1,0),(1,1),(2,0)}` and `p = 5`: sharp law as an equality,
   `C = 0` idealised and certified, transversality defect 0. What remains of
   §5 item 3 is purely a proof obligation, not an empirical one.

**Verdict: worth proving.** The target should be the sharpened statement of
§3 [NUM], and the interesting quantity is no longer `C` (it is 0) but the
**exchange rate**: digits of starting precision per step of certified window
— `3s+m` forward, `3s+2m` two-sided; the two-sided rate is what a certified
window actually costs, and it reduces to `3s` exactly when `δ` is a unit.
**[Both targets since proved — `NOTE.md` §3–§4.]**

---

## 1. The high-level question

> When you iterate a map p-adically at finite precision, precision decays.
> Is that decay **inevitable**, or merely an artifact of computing forward
> in time?

The standard answer is that it is inevitable: in a hyperbolic system, digits
are lost at a rate set by the expanding eigenvalue, and iterating `T` steps
costs you `O(T)` digits. Every precision-tracking framework (Caruso–Roe–
Vaccon and successors) propagates uncertainty *forward* and therefore only
ever degrades.

The claim of this project is that the decay is an artifact of one-sided
information. For an **invertible** map, the same orbit can be constrained
from both ends of time, and the expanding direction of `f` is the
contracting direction of `f⁻¹`. Combining the two constraints should give
precision at intermediate times that is **bounded independently of `T`**.

Stated in filtering language: CRV precision tracking is the *prediction*
step of a Kalman filter with no *update* step. Invertibility supplies the
missing update for free, and the resulting object is an exact, ultrametric
Rauch–Tung–Striebel smoother. The question is whether it provably works.

**[NUM] The answer, numerically, is: the decay is an artifact, and not even
approximately inevitable.** Smoothed precision does not decay by a bounded
amount — it does not decay *at all* (`C = 0`; the interior of the window is
*better* determined than its endpoints, by up to `s(s+m)T/(2s+m)` digits,
which is `sT/2` when `δ` is a unit). What is genuinely
inevitable is a different cost, in a different currency: to *certify* the
claim you must carry working precision linear in `T`, at `3s + 2m` digits per
step (`3s + m` is the forward-only rate; the two coincide iff `δ` is a unit). The `O(T)` has not vanished; it has moved from the answer's accuracy to
the computation's word size, which is the trade a filtering framing predicts
and the forward-only framing cannot express.

---

## 2. Setup

Let `p` be an odd prime, `K = ℚ_p`, and

```
f(x, y) = (y, y² + c − δx),    δ ≠ 0,    J(x,y) = [[0, 1], [−δ, 2y]]
```

Assume the **horseshoe regime** (region `H_III` of Allen–DeMark–Petsche,
arXiv:1610.04271, via the dictionary of `REFERENCES.md`):

```
|c| > max(1, |δ|²)     and     −c = γ² is a square in ℚ_p
```

Write `s = −v_p(γ) > 0` and `m = v_p(δ) ≥ 0`. By ADP Thm 1(e), `f` restricted
to its filled Julia set `J(f)` is conjugate to the two-sided 2-shift; every
point of `J(f)` has `|x| = |y| = |γ| = p^s`, and the eigenvalue valuations of
`J` along `J(f)` are `{−s, s+m}` — genuine expansion and genuine contraction.

Fix an orbit `v_0, …, v_T ∈ J(f)` and an initial precision `k`. Define the
**forward**, **backward**, and **smoothed** precision lattices

```
H_t^F  =  (J_{t−1} ⋯ J_0) · p^k ℤ_p²                     (predict from t = 0)
H_t^B  =  (J_{T−1} ⋯ J_t)^{−1} · p^k ℤ_p²                (predict from t = T)
H_t    =  H_t^F ∩ H_t^B                                   (smooth)
```

where `J_t = J(v_t)`. Let `d₁(L) ≤ d₂(L)` be the elementary-divisor
exponents of a lattice `L` (so `d₁` = digits of precision guaranteed in the
**worst** direction; larger is better, negative means digits lost).

---

## 3. The main theorem (to prove) **[NUM — restated]**

The original statement was:

> ~~**Theorem (bounded smoothed precision).**~~
> ~~There is a constant `C = C(p, c, δ)`, independent of `T`, of `t`, and of~~
> ~~the choice of orbit in `J(f)`, such that for all `0 ≤ t ≤ T`~~
> ~~`d₁(H_t) ≥ k + s·min(t, T − t) − C ≥ k − C`.~~

**This is false as written**, and it is false for a reason worth keeping: it
silently conflates the lattices of §2 with the *certified* lattices of the
Proposition below. Measured (§0.2), `C` grows linearly in `T` at fixed `k`.
Split into the two statements it was trying to make:

> **Theorem A (exact smoothing, unconditional).**
> **[PROVED 2026-07-27, exactly as stated — `NOTE.md` §3. The proof uses only
> `v(x_t) = v(y_t) = −s`, `det J = δ` and the margin `m+2s > 0`, so it covers
> aperiodic orbits too, and it needs no ADP transversality.]**
> For the lattices of §2 — pure Jacobian products — and for every orbit in
> `J(f)`, every `T`, every `t`, and every `k ∈ ℤ`, all four divisors are given
> exactly:
>
> ```
>   forward    d₁(H_t^F) = k − s·t                d₂ = k + (s+m)·t
>   backward   d₁(H_t^B) = k − (s+m)·(T−t)        d₂ = k + s·(T−t)
>   smoothed   d₁(H_t)   = k + min((s+m)·t, s·(T−t))
>              d₂(H_t)   = k + max((s+m)·t, s·(T−t))
> ```
>
> In particular `d₁(H_t) ≥ k` for all `t`, i.e. `C = 0`.

**[NUM] The tent is skewed unless `δ` is a unit.** The `k + s·min(t, T−t)`
written above (and in §0) is the `m = 0` case. In general the two arms have
*different* slopes — the forward pass loses `s` per step, but the backward
pass loses `s + m`, the extra `m` being the division by `δ` in `f⁻¹` — so the
peak sits at

```
    t* = sT / (2s + m)          (= T/2 only when m = 0)
```

Measured peaks: `t* = 4` for `(s,m) = (1,1)`, `3` for `(1,2)`, `5` for
`(2,1)` at `T = 12`, against predictions `4.0, 3.0, 4.8`. For `m > 0` the
symmetric tent remains a valid lower bound — so `C = 0` still holds as stated
— but it is not tight, and asserting it as an equality would be wrong. Both
forms are in `experiments/exp_theorem.py` (`tent` vs `symmetric_tent`), and
the sharp law is asserted as an equality in every configuration of block A.

> **Theorem B (certified smoothing, with a budget).**
> **[PROVED 2026-07-27 — but this budget is *wrong* for `m > 0`. The sharp
> condition is `k ≥ (3s+2m)(T−1)`, and it is an iff: `3s+m` is the forward
> rate only, and the backward remainder `(h_x²/δ, 0)` costs an extra `m`.
> Counterexample `p=3, s=1, m=2, T=10` pinned in
> `tests/test_proof_lemmas.py`. Proof and corrected statement: `NOTE.md` §4.]**
> If in addition `k ≥ (3s + m)·T + s + m + O(1)` — the condition that keeps
> the `NOTE.md` Definition 4.1 membership test satisfied for the whole window — then
> the *certified* lattices satisfy the same equality, and `v_t^true ∈ v_t + H_t`
> holds at every step. Without that hypothesis the conclusion degrades
> gracefully, at a rate governed by the exactness horizon rather than by the
> dynamics.

**In words:** forward-only precision decays linearly and without bound;
smoothed precision does not decay at all. The cost of iterating is `0`
digits — but *certifying* that costs a working-precision budget linear in
`T`, at `3s + 2m` digits per step (`3s + m` if only the forward pass is run).
Compare `s` digits per step just to keep forward-only from going negative:
certification is a constant-factor `3 + 2m/s` more expensive, and buys `k`
guaranteed digits instead of `0`.

Two companion statements make the theorem meaningful rather than
book-keeping:

> **Proposition (exactness).** Under the membership condition of
> `NOTE.md` Definition 4.1, the pushforward is an *equality* of sets,
> `f(v + H) = f(v) + J(v)H`, not a first-order approximation. Hence every
> lattice above is a **certified** enclosure of the true orbit:
> `v_t^true ∈ v_t + H_t` holds exactly, with no truncation error.
>
> **[NUM]** Verified on 10⁴ random orbits / 247,325 steps with zero
> violations, against a mutation control that is caught. Note this
> Proposition is exactly what Theorem B's hypothesis pays for: it is not
> automatic, and it is where the original statement's `C` was hiding.

> **Corollary (certified shadowing).** Given `v_0` and `v_T` each known to
> `k` digits, every intermediate point of the orbit connecting them is
> determined to within `O(1)` digits of `k`, uniformly in `T`. A p-adic
> horseshoe orbit is reconstructible from its endpoints at essentially no
> loss.
>
> **[NUM]** True with `O(1) = 0` — but "uniformly in `T`" requires the
> working precision to grow linearly in `T`. The information-theoretic
> content survives (the endpoints really do determine the interior); the
> arithmetic to certify it is not free.

---

## 4. Why it should be true

The ingredients are already in the literature; the theorem is their
combination.

1. **Hyperbolic splitting.** The Newton polygon of `λ² − 2yλ + δ` on
   `J(f)` has vertices `(0, m), (1, −s), (2, 0)`, giving eigenvalue
   valuations `{−s, s+m}` at *every* point of `J(f)` — uniformly, with no
   tangency events (`|2y| = p^s` always). So `H_t^F` degrades at exactly
   rate `s` in the unstable direction and ~~`H_t^B` at exactly rate `s` in the
   stable one~~ **[NUM]** `H_t^B` at rate `s + m` in the stable one: `f⁻¹`
   has `det = 1/δ`, so its eigenvalue valuations are `{s, −(s+m)}` — the
   mirror image only when `δ` is a unit. This asymmetry is what skews the
   tent (§3 [NUM]), and it means the backward pass is the *more* expensive
   direction whenever `m > 0`.
2. **Uniform transversality.** ADP Lemmas 23–24: stable and unstable tubes
   are graphs of `|δ/γ|`-Lipschitz functions with Lipschitz product `< 1`,
   and meet in exactly one point (Banach fixed point). This is precisely the
   statement that the two directions stay uniformly transverse in the
   ultrametric sense — which is what makes `H_t^F ∩ H_t^B` small rather
   than merely nonempty. ~~**The constant `C` should come from this Lipschitz
   product.**~~
   **[NUM] This expectation was wrong, in the project's favour.** The
   transversality is not approximate-and-quantified but *exact*: the measured
   defect `v_p(det[u | w]) − (d₁(H^F) + d₁(H^B))`, for `u, w` minimal vectors
   of the two lattices, is **0** in every configuration measured — every
   itinerary, every `T`, every `(p, s, m)`, including all the cases where the
   certified `C > 0`. Ultrametrically the two directions are not merely
   uniformly transverse, they are *orthogonal*, and they contribute nothing
   to `C`. The Lipschitz product bounds a quantity that turns out to vanish.
   What `C` actually measures is item 3's budget.
3. **Exact linearisation.** `f` is quadratic, so the Taylor expansion
   terminates and the remainder is a single term `(0, h_y²)`; ultrametrically
   it is absorbed by the propagated lattice under a checkable condition. No
   error accumulates from the linearisation itself.

The proof strategy is therefore: show the SNF bases of `H_t^F` and `H_t^B`
converge to the unstable/stable directions at a geometric rate (item 1),
bound their ultrametric angle below by item 2, and conclude that the
intersection's worst divisor is controlled by the sum of the two good
directions. Item 3 upgrades the conclusion from "approximately" to
"certified".

**[NUM] Revised strategy.** Given that the defect is identically 0, the proof
of **Theorem A** should be much shorter than anticipated and need not be
quantitative at all: it suffices to show that `H_t^F` and `H_t^B` are, in the
`Z_p`-Hermite form, *complementary* — the forward one carrying `p^{k−st}` in
the unstable direction and `p^{k+(s+m)t}` in the stable one, the backward one
carrying `p^{k+s(T−t)}` and `p^{k−(s+m)(T−t)}` respectively (note this is a
reflection only when `m = 0`) — after which the intersection's divisors are
forced by taking the max exponent in each direction, with no estimate needed. Item 2 is then not the source of a
constant but the qualitative input guaranteeing the two Hermite forms are in
general position.

**[NUM] The first step of that route is now verified numerically and looks
like a three-line induction.** Write `M_t = J_{t−1} ⋯ J_0` and
`N_j = J_{T−j}⁻¹ ⋯ J_{T−1}⁻¹` (so `H_t^F = M_t·p^k ℤ_p²` and
`H_t^B = N_{T−t}·p^k ℤ_p²`). Since `J = [[0, 1], [−δ, 2y]]` with
`v(2y) = −s` at every point of `J(f)`, the update `M_{t+1} = J_t M_t` sends
(top row, bottom row) ↦ (bottom row, `−δ·top + 2y·bottom`), and the two
summands' valuations differ by exactly `m + 2s > 0` — so ultrametrically there
is **never** a cancellation, and by induction the row valuations are exactly

```
    M_t:  top −s(t−1),  bottom −st          (min entry −st)
    N_j:  top −(s+m)j,  bottom −(s+m)(j−1)  (min entry −(s+m)j)
```

With `v(det M_t) = mt` and `v(det N_j) = −mj` this pins both SNFs,
i.e. the forward and backward lines of Theorem A, with no appeal to
eigenvectors.

**[NUM] The intersection step is also an invariant-cone induction, and it is
measured (2026-07-27).** The minimal vectors of the two lattices sit in the
two *eigenvector cones* with exact valuation gaps: for `u` minimal in
`H_t^F` and `w` minimal in `H_t^B`,

```
    v(u_x) − v(u_y) = s        (unstable cone: eigenvector (1, λ), v(λ) = −s)
    v(w_y) − v(w_x) = s + m    (stable cone:   eigenvector (1, λ′), v(λ′) = s+m)
```

as an equality at every `1 ≤ t ≤ T−1`, every regime tested. Each cone is
invariant under one step (`J` maps gap-`s` vectors to gap-`s` vectors losing
exactly `s`, the same `m + 2s` margin as the row induction; `J⁻¹` likewise
for the stable cone), and the ball's image enters the cone at the very first
step. Then `det[u|w] = u_x w_y − u_y w_x` has its two terms separated by
exactly `2s + m`, so `v(det[u|w]) = v(u) + v(w)` — defect 0 with margin,
**no appeal to ADP Lemmas 23–24 anywhere** (ADP is needed only for the
regime itself, `v(2y) = −s` on `J(f)`). Both lemmas are pinned as exact
assertions on random length-40 itineraries in
`tests/test_proof_lemmas.py`. (An earlier note here guessed the defect
statement would come from directions *converging* at a geometric rate; that
was measured and is wrong — the minimal direction tracks the moving orbit
point at `O(1)` per step. The invariant-cone form above is what is true.)

**Theorem B** is where the analysis actually lives: it is a
statement about the §2.4 membership condition surviving `T` steps, i.e. about
`2·w_t` versus the largest divisor of `J(v_t)H_t`, and the budget rate is
`2s` (the remainder is quadratic in a projection falling at rate `s`) plus
the rise rate of the divisor it must clear: `s + m` forward (total `3s+m`),
`s + 2m` backward (total `3s+2m`, the binding rate — `NOTE.md` §4.3).

---

## 5. Evidence so far **[NUM — the next experiment has been run]**

All exact (integer/rational arithmetic, asserted, no floating point).

**Original evidence**, `review_checks.py`:

- At the fixed point `p = 3, δ = 1, α = 1/3, c = 5/9` (`s = 1`, `m = 0`),
  `T = 12`: forward divisors are `(−t, t)`, backward `(−(T−t), T−t)`, and
  the smoothed divisors are exactly `(min(t, T−t), T − min(t, T−t))`.
  **The theorem holds there with `C = 0`.**
- In the attractor regime `H⁺_II` the theorem is *false and vacuous*: the
  map is nonexpanding, the forward pass loses nothing, and `H^F ∩ H^B = H^F`
  identically. Hyperbolicity is not decoration — it is the hypothesis.
  (Now asserted in `experiments/exp_5_3_smoother.py`, panel 3.)

**The passage to general orbits** — named above as "the next experiment" —
is done. Period-`ℓ` orbits are built by multivariate Newton from symbolic
itineraries (`padic_filtering/henon.py:periodic_orbit`, the shadowing lemma
in computational form), so the Jacobian genuinely varies along the orbit:

| what varies | range tested | result |
|---|---|---|
| itinerary / period | periods 1, 2, 3, 5, 8, 12, 13, 19, 20 | `C = 0`, defect `0` |
| window `T` | 4 … 100 | `C = 0`, defect `0` |
| starting precision `k` | 0 … 197 | `C = 0`, defect `0` |
| prime `p` | 3, 5, 7, 11 | `C = 0`, defect `0` |
| expansion `s` | 1, 2, 3 | `C = 0`, defect `0`; sharp law skewed for `m > 0` |
| `m = v_p(δ)` | 0, 1, 2 | `C = 0`, defect `0` |
| aperiodic windows | 15 windows, `T ∈ {24,48}`, cut from random length-200 itineraries | `C = 0` (idealised *and* certified), defect `0` |

`experiments/exp_theorem.py` (Theorem A and B, and the fixed-`k` failure),
`experiments/exp_5_3_smoother.py` (the tent across periods),
`experiments/exp_5_2_horizon.py` (the horizon and hence the budget),
`experiments/exp_5_5_slack.py` (the lattice is the *minimal* correct one at
every step, slack 0 — so `C = 0` is not an artefact of a loose enclosure).

**So the open work has moved.** It is no longer "is `C` bounded" — `C` is 0,
and the constant-hunting framing of §4 item 2 was misdirected. What remains:

1. **Prove Theorem A.** Should be short and non-quantitative (§4 [NUM]).
   **[DONE 2026-07-27 — `NOTE.md` §3.]**
2. **Prove Theorem B**, i.e. the `(3s+m)T` budget, which is now the only
   place an estimate is needed. Steps for both proofs, and for the
   generalisation beyond the Hénon family, are laid out in §8.
   **[DONE 2026-07-27 — `NOTE.md` §4, with the budget corrected to
   `(3s+2m)(T−1)`, an iff.]**
3. **Non-periodic orbits.** ~~Every orbit tested is periodic … this is the one
   genuine gap.~~ **[NUM] Closed at the empirical level** by
   `experiments/exp_aperiodic_window.py` (§0.6): windows cut from random
   length-200 itineraries, whose Jacobian sequence is indistinguishable from
   aperiodic, satisfy the sharp law exactly. Truly infinite orbits are now a
   proof obligation only — and the row-valuation induction of §4 [NUM] never
   uses periodicity, so the proof of Theorem A should cover them for free.
4. Residue characteristic 2, open in ADP too, remains out of reach.

---

## 6. What would falsify it **[NUM — all three tested]**

- `d₁(H_t)` for mid-trajectory `t` drifting downward as `T` grows, on
  genuine (non-fixed-point) horseshoe orbits — i.e. no uniform `C`.
  → **Fired, but not at the theorem.** It happens at fixed `k`, and the
  cause is the §2.4 exactness budget, not the dynamics: with `k` scaled to
  the budget, or with the idealised lattices of §2, the drift is exactly
  zero out to `T = 100`. This is what forced the A/B split in §3.
- SNF bases of `H_t^F` and `H_t^B` becoming ultrametrically parallel along
  some itinerary, making the intersection buy nothing.
  → **Never.** Defect 0 in every configuration measured, including those
  with `C > 0`. This was the most likely way for the idea to fail and it
  did not fail at all.
- Certification failure (`v_t^true ∉ v_t + H_t`) at any step, which would
  indicate the exactness condition is being misapplied rather than that the
  bound is wrong.
  → **Never**, in 247,325 certified steps over 10⁴ random orbits. The
  mutation control (inflation disabled) *is* caught on 2,976 of them, so the
  assertion is not vacuous.

**Remaining falsifiers, for the work that is actually open:**

- A non-periodic orbit of the 2-shift on which the smoothed divisors miss the
  tent, i.e. Theorem A failing off the periodic points.
  → **Tested at the cheap-attack level and did not fire:** 15 effectively
  aperiodic windows (random length-200 itineraries, `T ∈ {24, 48}`), all
  exactly on the tent (`exp_aperiodic_window.py`). Only an actual proof can
  close it completely.
- A budget below `(3s+m)T + O(1)` still giving `C = 0` for all `T`, or one
  above it still failing — either would mean the exchange rate in Theorem B
  is misidentified.
  → **Fired (2026-07-27), in the second direction: the `(3s+m)T` budget
  itself fails for `m > 0`** (counterexample `p=3, s=1, m=2, T=10`, pinned in
  `tests/test_proof_lemmas.py`). The two-sided rate is `3s+2m`, and `NOTE.md`
  Prop 4.4 now gives both horizons as iffs, retiring this falsifier.

---

## 7. Where this sits in the literature (checked 2026-07-26)

Positioning, based on an actual literature pass rather than the project's own
framing. Three communities, in decreasing order of how sharp the
contribution is:

1. **p-adic precision tracking (Caruso–Roe–Vaccon).** The CRV framework
   ("Tracking p-adic precision", arXiv:1402.7142, LMS JCM 2014; "p-adic
   stability in linear algebra", arXiv:1506.05644, ISSAC 2015) propagates
   precision lattices by differentials and is *structurally forward-only*.
   Nothing found in that line, or citing it, exploits invertibility to run
   the differential backward and intersect. Theorems A+B say something the
   CRV formalism cannot express: prediction-only `d₁` decays at rate `s`
   inevitably (it exactly equals the naive scalar counter in `H_III`), yet
   the two-sided lattice loses zero digits, with the true cost relocated to
   an `O(T)` word-size budget at exchange rate `3s+2m` (`3s+m` forward-only). Searches for any
   p-adic / ultrametric Kalman-filter or smoother analogue come up empty —
   the "CRV = prediction step of a Kalman filter with no update step"
   framing of §1 appears to be new.

2. **Non-archimedean dynamics (the ADP line).** ADP (arXiv:1610.04271,
   Res. Number Theory 2018) and the follow-up attractor paper
   (arXiv:1810.06708) give the conjugacy to the 2-shift; the recent p-adic
   shadowing literature (arXiv:2001.02737, arXiv:2408.04779) is qualitative
   and forward-time. A digit-exact *two-sided* shadowing statement —
   endpoints known to `k` digits determine the interior exactly, with a
   certified budget — is a different kind of result and a natural next paper
   in that thread.

3. **Validated numerics over ℝ.** Boundary-value / multiple-shooting
   reformulations of orbit verification are classical there, and interval
   methods always pay a wrapping-effect tax (`C = O(1)` at best). The
   exportable story: over `ℚ_p` the analogous construction is *exact* —
   equality, not enclosure, `C = 0` — because ultrametric transversality is
   orthogonality (§4 item 2 [NUM]).

**Honest caveats.** This resolves no named open problem; the claim is "first
two-sided precision tracking in the ultrametric setting, with an exact law
and an exact certification cost". The referee risk is not falsity but
"Theorem A is easy once stated" — the defence is that the *statement* (the
A/B split, the skewed tent, the exchange rate) is the contribution, and that
Theorem B still needs real analysis. Venue profile: Res. Number Theory
(where ADP appeared), LMS JCM-style computational journals, or ISSAC (where
the Vaccon line publishes), rather than a general dynamics journal. The
biggest available upgrade in impact is §8: prove it for a *class* of maps,
not one family.

---

## 8. The generalised problem, and the steps to solve it

The proof route in §4 [NUM] barely uses the Hénon map. What it actually
uses, at each step of the orbit, is:

- **(H1) uniform splitting** — the Jacobian's characteristic polynomial has
  Newton-polygon slopes `−s` and `s+m` with `s > 0`, at *every* point of the
  invariant set, no tangencies;
- **(H2) a no-cancellation margin** — in the row-valuation induction the two
  summands' valuations differ by `m + 2s > 0`, so ultrametric addition never
  cancels; the generalisation is a positive gap between the two Newton
  slopes;
- **(H3) finite Taylor expansion** — the map is polynomial, so the
  linearisation remainder is a finite sum, absorbable under a membership
  condition (this is what Theorem B's budget pays for);
- **(H4) polynomial inverse** — the map is a polynomial *automorphism*, so
  the backward pass is the same kind of object as the forward one.

Nothing in (H1)–(H4) is specific to `f(x,y) = (y, y² + c − δx)`. The
generalised theorem should be: *for any hyperbolic polynomial automorphism
of the plane over `ℚ_p` (equivalently, by Friedland–Milnor / Jung, any
composition of generalised Hénon maps) restricted to a horseshoe-type
invariant set, smoothed precision satisfies a sharp tent law with `C = 0`,
and certification costs a per-step budget determined by the eigenvalue
valuations and the degree.* Steps, in dependency order:

1. **Theorem A for the Hénon family** — **DONE, `NOTE.md` §3**; (a), (b), (c)
   below are Lemmas 3.1, 3.3–3.5 and 3.6 there.
   a. The row-valuation induction — three lines, gives both SNFs.
   b. The invariant-cone induction: `J` preserves the unstable cone
      `v(x) − v(y) = s` (losing exactly `s` per step), `J⁻¹` preserves the
      stable cone `v(y) − v(x) = s + m`; the minimal vectors of `H_t^F` and
      `H_t^B` live in these cones from step 1 on, and the cone gaps make
      `det[u|w]` split with margin `2s + m`. This *is* the defect-0
      statement, with no ADP input.
   c. Intersection bookkeeping: with `u, w` unimodularly transverse the two
      Hermite forms are diagonal in the frame `(u, w)` and the intersection
      takes the max exponent per direction. Purely formal given (b).

2. **Theorem B for the Hénon family.** **[DONE — `NOTE.md` §4. The plan below
   is what was carried out, with two corrections: the forward horizon is
   `(k−m)/(3s+m)` exactly (not `(k−s−m)/…`; `exp_5_2` only asserts `≥`), and
   the *binding* rate is the backward one, `3s+2m`.]** Induct on the §2.4 membership
   condition. The Taylor expansion terminates:
   `f(v+h) = f(v) + J(v)h + (0, h_y²)`, so the linearisation stays exact at
   step `t` iff the remainder module `(0, p^{2w_t} ℤ_p)` lies inside
   `J(v_t)H_t`, where `p^{w_t}ℤ_p` is the projection of `H_t` onto the
   squared coordinate (`precision.py:is_linearisation_exact`; the backward
   remainder sits on the other axis at `2w_t − m`). Theorem A gives all the
   ingredients exactly: `w_t` falls like `k − st` while the worst divisor of
   `J(v_t)H_t` also moves, and the condition `2w_t ≥ (that divisor)` survives
   all `T` steps iff `k ≥ (3s+m)T + O(1)` — the `3s+m` decomposing as `2s`
   (the remainder is quadratic in a direction falling at rate `s`) plus
   `s+m` (the divisor it must clear rises at that rate). `exp_5_2` already
   matches the horizon `(k−s−m)/(3s+m)` exactly, so the constant is pinned
   before the proof starts; the proof is the two monotone sequences from
   Theorem A plus one inequality.

3. **Axiomatise the linear-algebra core.** Extract from step 1 a standalone
   lemma about *matrix sequences*, with no dynamics at all: for any sequence
   `J_0, J_1, …` of `GL₂(ℚ_p)` matrices satisfying (H1)+(H2) in a common
   unimodular frame (each `J_t` maps the "unstable cone" into itself with
   valuation gain exactly `−s`, and its inverse does the same for the stable
   cone at `−(s+m)`), the forward/backward products satisfy the four-divisor
   law of §3. This makes Theorem A a corollary and is the statement that
   travels: it applies to any system that supplies such a Jacobian
   sequence. (The right cone formalism to steal from is the standard
   invariant-cone-field proof of hyperbolicity, transplanted to valuations.)

4. **Verify the axioms for compositions of generalised Hénon maps.**
   Friedland–Milnor: every dynamically nontrivial polynomial automorphism of
   the plane is conjugate to `g_n ∘ ⋯ ∘ g_1` with
   `g_i(x,y) = (y, P_i(y) − δ_i x)`, `deg P_i ≥ 2`. The Jacobian of each
   factor is the same companion shape `[[0,1],[−δ_i, P_i′(y)]]`, so step 3
   applies verbatim whenever `v(P_i′(y))` is constant `= −s_i < 0` on the
   invariant set — the composite has per-cycle rates `Σ s_i` and
   `Σ (s_i + m_i)`. What needs proving fresh is the *horseshoe regime*: an
   ADP-style theorem (filled Julia set conjugate to the shift, uniform
   `|P_i′|` on it) for compositions. This is where genuinely new dynamics
   lives; the 2018 follow-up (arXiv:1810.06708) already moves in this
   direction for attractors. A cheaper intermediate: iterates `f^n` of one
   Hénon map, where the regime work is already done.

5. **Generalise Theorem B's budget to degree `d`.** The remainder is no
   longer one quadratic term; for `deg P = d` the membership condition
   involves `j·w` for `j = 2…d` and the binding constraint is still `j = 2`
   (higher terms are ultrametrically smaller once the quadratic one is
   absorbed) — *conjecturally*, so the exchange rate stays `2s_i` + (rise
   rate), the rise rate being `s+m` forward / `s+2m` backward as in the
   quadratic case. Test numerically before proving: one experiment with
   `P(y) = y³ + …` would settle whether the budget sees the degree.

6. **Portable extensions, in order of increasing effort:**
   - **Equal characteristic** `𝔽_p((T))`: the argument is valuation-theoretic
     throughout, so it should transfer verbatim; only `padic.py` touches the
     field. Do this last as a one-section remark.
   - **Dimension `n > 2`**: (H1) becomes an `(n_u, n_s)` Newton-polygon
     splitting; the row induction becomes a block induction; the tent
     becomes per-divisor. No conceptual obstacle, heavy bookkeeping.
   - **Residue characteristic 2**: open in ADP already (`|2y|` degenerates);
     genuinely out of reach of this route, say so and stop.

**What to run before proving anything** (cheap, sharpens the targets):
step 5's degree-3 budget probe, and a step-4 sanity check on `f²` (the
composite of a map with itself — the regime is inherited, and the per-cycle
rates should read `2s` and `2(s+m)` on the existing machinery with no new
code beyond composing Jacobians).
