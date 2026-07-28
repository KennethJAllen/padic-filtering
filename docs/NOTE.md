# Two-sided precision tracking for p-adic Hénon horseshoes

*Research note, 2026-07-27. Self-contained: every proof below is carried out
for the `s`-admissible sequences of Definition 1.1, a hypothesis stated here,
and no step of any proof invokes an external result.*

*Two works are cited, neither of them as an assumption. Allen–DeMark–Petsche
(ADP), "Non-Archimedean Hénon maps, attractors, and horseshoes",
arXiv:1610.04271, is what makes the hypothesis non-vacuous: it supplies the
horseshoe on which `v(x) = v(y) = −s` holds, so that the theorems below are
about a non-empty set. A reader willing to take Definition 1.1 as given needs
nothing from it. Caruso–Roe–Vaccon (CRV), "Tracking p-adic precision", LMS J.
Comput. Math. 17 (2014), is the origin of the lattice-precision formalism —
carry a lattice `H`, certify `v_true ∈ v + H`, propagate by the differential.
CRV's recursion is forward-only; the object studied here is the intersection of
a forward and a backward CRV pass.*

**Summary.** For a p-adic Hénon map restricted to its horseshoe, forward
precision tracking loses `s` digits per step without bound. Intersecting the
forward pass with the backward pass loses **nothing**: all four elementary
divisors of the smoothed precision lattice are given by an exact closed
formula, a *skewed* tent peaking at `t* = sT/(2s+m)` and lying everywhere at or
above the starting precision `k`. Theorem A below states this and §3 proves it
unconditionally, by two short valuation inductions plus a change of frame; no
estimate, no constant, and no periodicity or shift-conjugacy is used. The real
cost is relocated to arithmetic: to *certify* the statement one needs starting
precision linear in the window length. §4 determines that budget exactly —
`k ≥ (3s+2m)(T−1)`, sharp in both directions — which corrects the previously
conjectured `(3s+m)T + s + m` for `m > 0` (§4.5 gives an explicit
counterexample). Both theorems are proved in full, and every intermediate
identity was checked numerically before being used (§5, §8). What happens
*below* that budget is left open and stated as such (Open Problem 1). §6
extracts the linear-algebra core — a four-divisor theorem for sequences in
`GL₂(Q_p)` obeying two cone axioms, with Theorem A as a corollary — and §7
lists the remaining generalisation targets.

---

## 1. Setup

Let `p` be an odd prime, `K = Q_p`, `v = v_p` the valuation normalised by
`v(p) = 1`, extended to vectors and matrices by

```
    v(z_1, z_2) = min(v(z_1), v(z_2)),      v(A) = min over entries.
```

Fix `c, δ ∈ Q_p` with `δ ≠ 0` and consider the Hénon map and its inverse

```
    f(x, y)     = (y,  y² + c − δx)
    f⁻¹(x, y)   = ((x² + c − y)/δ,  x)
```

with Jacobians

```
    J(x,y)  = Df(x,y)    = [[ 0, 1], [−δ, 2y]],        det J  = δ
    K(x,y)  = Df⁻¹(x,y)  = [[2x/δ, −1/δ], [1, 0]],     det K  = 1/δ.
```

**Standing hypotheses (the horseshoe regime).** Assume `|c| > max(1, |δ|²)`,
that `−c = γ²` is a square in `Q_p`, and that `v(δ) ≥ 0`. Write

```
    s = −v(γ) > 0,        m = v(δ) ≥ 0.
```

By ADP Thm 1(a),(e) the filled Julia set `J(f)` is non-empty and `f|J(f)` is
topologically conjugate to the two-sided full 2-shift; moreover every point of
`J(f)` satisfies `|x| = |y| = |γ| = p^s`, i.e. `v(x) = v(y) = −s`. Only that
last fact is ever used, and it is used only *here*, to know that
Definition 1.1's hypothesis has instances: no proof in §3, §4 or §6 refers back
to ADP, and the shift conjugacy is quoted for context and never used at all.
The standing hypotheses above play the same role — they are the conditions
under which ADP produces the horseshoe, not conditions any argument below
consumes.

In particular the characteristic
polynomial `λ² − 2yλ + δ` has Newton polygon with vertices `(0, m), (1, −s),
(2, 0)`, so the eigenvalue valuations are `{−s, s+m}` at every point of `J(f)`,
with no tangency events (`p` is odd, so `v(2y) = −s`).

**Definition 1.1 (admissible sequence).** A finite sequence
`v_0, …, v_T ∈ Q_p²`, `v_t = (x_t, y_t)`, is *`s`-admissible* if
`v(x_t) = v(y_t) = −s` for `0 ≤ t ≤ T`.

Every orbit segment in `J(f)` is `s`-admissible. Everything below is proved for
arbitrary `s`-admissible sequences; **no orbit relation between consecutive
`v_t` is ever used**, so in particular periodicity plays no role and the
results apply verbatim to orbit segments of infinite aperiodic orbits, and to
the perturbed representatives that a tracking algorithm actually holds (§4.4).

**Definition 1.2 (precision lattices).** Fix an `s`-admissible sequence, an
integer `k`, and write `J_t = J(v_t)`, `K_t = K(v_t)`. Put

```
    M_t = J_{t−1} ⋯ J_0     (M_0 = I),        N_j = K_{T−j+1} ⋯ K_T   (N_0 = I)
```

and define the **forward**, **backward** and **smoothed** lattices

```
    H_t^F = M_t · p^k Z_p²,      H_t^B = N_{T−t} · p^k Z_p²,
    H_t   = H_t^F ∩ H_t^B .
```

Because `K(v_i) = J(v_{i−1})⁻¹` along an orbit, `H_t^B` is the pullback
`(J_{T−1} ⋯ J_t)⁻¹ p^k Z_p²`, i.e. the prediction from time `T` run backwards.

**Notation.** For a rank-2 `Z_p`-lattice `L ⊂ Q_p²` let `d₁(L) ≤ d₂(L)` be its
elementary-divisor exponents: `L = A Z_p²` with `A = U · diag(p^{d₁}, p^{d₂}) ·
V`, `U, V ∈ GL₂(Z_p)`. Then `d₁(L) = min{ v(z) : z ∈ L }` (equivalently
`p^{d₁}Z_p²` is the smallest ball containing `L`), `d₁(L) = v(A)`, and
`d₁ + d₂ = v(det L)`. `d₁` is the number of guaranteed digits in the *worst*
direction; it may be negative. Write `w_x(L), w_y(L)` for the exponents of the
coordinate projections, `pr_x(L) = p^{w_x}Z_p` and `pr_y(L) = p^{w_y}Z_p`.

**Definition 1.3 (transversality defect).** Call `u ∈ L` *minimal* if
`v(u) = d₁(L)`. For two lattices `L, L′` with `d₁ < d₂` in each (so that each
has a distinguished worst direction) put

```
    def(L, L′) = min { v(det[u | u′]) : u ∈ L, u′ ∈ L′ minimal } − d₁(L) − d₁(L′).
```

Each term `v(det[u|u′]) ≥ v(u) + v(u′) = d₁(L) + d₁(L′)` because every entry of
the determinant is a product of one coordinate from each, so `def ≥ 0` always.
`def(L, L′) = 0` says the two worst directions are ultrametrically *orthogonal*:
the two terms of the determinant cannot cancel, and the normalised minimal
vectors form a `Z_p`-basis of `Z_p²` (Lemma 3.5). Positive defect is
cancellation — the two passes are partly informative about the same direction
and their intersection gains less than the tent of Theorem A predicts. The
quantity is undefined for a ball `d₁ = d₂`, which has no worst direction.

---

## 2. Statements

> **Theorem A (exact smoothing, unconditional).**
> For the lattices of Definition 1.2 — pure Jacobian products — over any
> `s`-admissible sequence, and for every `T`, every `t`, and every `k ∈ Z`, all
> four divisors are given exactly:
>
> ```
>   forward    d₁(H_t^F) = k − s·t                d₂ = k + (s+m)·t
>   backward   d₁(H_t^B) = k − (s+m)·(T−t)        d₂ = k + s·(T−t)
>   smoothed   d₁(H_t)   = k + min((s+m)·t, s·(T−t))
>              d₂(H_t)   = k + max((s+m)·t, s·(T−t))
> ```
>
> In particular `d₁(H_t) ≥ k` for all `t`. Every orbit segment in `J(f)` is
> such a sequence, so the law holds in particular along the horseshoe; but the
> horseshoe is where the sequences come from, not something the proof uses.

Throughout, `C` denotes the constant of the conjecture this note started from,
`d₁(H_t) ≥ k + s·min(t, T−t) − C` with `C` uniform in `T`; Theorem A gives
`C = 0`, and the numerics of §5 confirm it as an equality rather than a bound.
(Its `T`-uniformity, however, is false for the *certified* lattices without the
budget of Theorem B′ — see §4.5 and Open Problem 1.)

The smoothed law is a tent, but a **skewed** one: the forward pass loses `s`
digits per step and the backward pass `s + m` (the extra `m` is the division by
`δ` in `f⁻¹`), so the peak sits at

```
    t* = sT/(2s+m),        d₁(H_{t*}) = k + s(s+m)T/(2s+m),
```

which is `T/2` and `k + sT/2` only when `δ` is a unit. (When `t*` is not an
integer the peak is attained at a neighbouring integer `t`, within `s+m` of
the stated value.) The symmetric form
`k + s·min(t, T−t)` is a valid lower bound for `m > 0` but is not tight.

Theorem A is proved in §3. It is a statement about *idealised* lattices — pure
Jacobian products. An algorithm cannot compute those directly; it propagates a
coset, and each propagation is exact only while the map's quadratic remainder
is absorbed by the propagated lattice. §4 determines exactly when that holds.

> **Theorem B′ (certified smoothing, sharp budget).**
> Let `T ≥ 2`. Run the two passes with the step of Definition 4.1,
> starting from `v_0^true + p^kZ_p²` and `v_T^true + p^kZ_p²`. Then
>
> ```
>     every step of both passes is exact   ⟺   k  ≥  (3s + 2m)·(T − 1).
> ```
>
> When it holds, the computed lattices coincide with those of Definition 1.2,
> so Theorem A applies to them verbatim, and `v_t^true ∈ v_t + H_t` at every
> `0 ≤ t ≤ T`.
>
> The two passes have different horizons: the forward one is exact at the step
> `t → t+1` iff `(3s+m)·t ≤ k − m`, the backward one at `j → j+1` iff
> `(3s+2m)·j ≤ k`.

The `iff` is an `iff` about *exactness*, which is what the budget is a budget
for. It is **not** a claim that the enclosure `v_t^true ∈ v_t + H_t` fails below
the budget: inflation (Definition 4.1(b)) preserves the enclosure
unconditionally, at the price of a lattice strictly larger than
Definition 1.2's, to which Theorem A no longer applies. What is unknown below
the budget is not whether the truth is enclosed but how large the enclosure
becomes — Open Problem 1.

**Remark 2.1 (an earlier budget, and why it fails).** The natural first guess
is the *forward* rate: `k ≥ (3s+m)·T + s + m`. That is what the forward
exactness threshold alone gives, and it is correct when `δ` is a unit. It is
**insufficient** as soon as `m > 0`, because the backward remainder carries an
extra factor `1/δ` and so pays `3s+2m` per step. §4.5 gives an explicit
counterexample (`p = 3`, `s = 1`, `m = 2`, `T = 10`) and the exact arithmetic of
the discrepancy.

**Open Problem 1 (graceful degradation).** Below the threshold of Theorem B′
some step inflates, the certified lattices are strictly larger than the
idealised ones, and the equality of Theorem A fails for them. Numerically the
*conclusion* `d₁(H_t) ≥ k` survives well past that point — inflation at the far
end of a pass lands where the other arm of the tent is binding, so it does not
move `d₁` — and the observed loss grows only once inflation reaches the binding
arm. Nothing in this note proves any of that: past the first inflation the
lattices are no longer Jacobian products, so the inductions of §3 do not apply,
and the object to track is an inflated divisor pair satisfying a max-recursion
rather than a product recursion. **The threshold at which `d₁(H_t) ≥ k` fails
is unknown**, and any claim that the conclusion "degrades gracefully" below
Theorem B′'s budget should be cited as a conjecture, not as a result of this
note.

---

## 3. Proof of Theorem A

Throughout, `(v_t)` is `s`-admissible and `s > 0`, `m ≥ 0`. The single
arithmetic fact driving everything is the **margin**: whenever a step of `J_t`
or `K_t` adds a `−δ·(·)` term to a `2y_t·(·)` term applied to quantities `a, b`
of equal valuation,

```
    v(−δ a) − v(2y_t b) = (m + v(a)) − (v(b) − s) = m + 2s > 0,
```

i.e. multiplication by `−δ` costs `m` while multiplication by `2y_t` gains `s`,
so the second term strictly dominates and the two can never tie. Every lemma
below is an instance of this comparison, with `a` and `b` taken from a row, a
cone vector, or a determinant.

### Lemma 3.1 (row valuations)

*For `1 ≤ t ≤ T`, the rows of `M_t` have valuations*

```
    v(top row of M_t) = −s(t−1),        v(bottom row of M_t) = −s·t,
```

*and for `1 ≤ j ≤ T` the rows of `N_j` have valuations*

```
    v(top row of N_j) = −(s+m)j,        v(bottom row of N_j) = −(s+m)(j−1).
```

*Proof.* Forward: `M_{t+1} = J_t M_t` with `J_t = [[0,1],[−δ, 2y_t]]`, so
writing `R, S` for the top and bottom rows of `M_t`,

```
    top(M_{t+1}) = S,        bottom(M_{t+1}) = −δ·R + 2y_t·S.
```

For `t = 1`, `M_1 = J_0` has rows `(0,1)` and `(−δ, 2y_0)` of valuations `0 =
−s·0` and `min(m, −s) = −s` (as `m ≥ 0 > −s`), which is the claim. Assume it
for `t ≥ 1`. The top row of `M_{t+1}` is `S`, of valuation `−st = −s((t+1)−1)`.
For the bottom row, work coordinatewise: `v(−δ R_i) = m + v(R_i) ≥ m − s(t−1)`
and `v(2y_t S_i) = −s + v(S_i) ≥ −s(t+1)`, and

```
    (m − s(t−1)) − (−s(t+1)) = m + 2s > 0,
```

so in the coordinate `i` where `v(S_i) = −st` is attained the second summand
strictly dominates and `v(−δR_i + 2y_tS_i) = −s(t+1)` exactly, while in the
other coordinate the valuation is `≥ −s(t+1)`. Hence the row valuation is
exactly `−s(t+1)`.

Backward: `N_{j+1} = K_{T−j} N_j` with `K = [[2x/δ, −1/δ],[1,0]]`, so with `R,
S` the rows of `N_j`,

```
    top(N_{j+1}) = (2x_{T−j}/δ)·R − (1/δ)·S,      bottom(N_{j+1}) = R.
```

For `j = 1`, `N_1 = K_T` has rows of valuations `min(−s−m, −m) = −(s+m)` and
`0`, as claimed. Inductively the new bottom row is `R`, of valuation `−(s+m)j`;
for the new top row the two summands have valuations `≥ −s−m−(s+m)j =
−(s+m)(j+1)` and `≥ −m − (s+m)(j−1)`, whose difference is again

```
    (−m − (s+m)(j−1)) − (−(s+m)(j+1)) = 2s + m > 0,
```

so the first dominates coordinatewise and the top row valuation is exactly
`−(s+m)(j+1)`. ∎

### Corollary 3.2 (forward and backward divisors, and all four projections)

```
    d₁(H_t^F) = k − s·t,           d₂(H_t^F) = k + (s+m)·t
    d₁(H_t^B) = k − (s+m)(T−t),    d₂(H_t^B) = k + s·(T−t)
```

*and, writing `j = T − t`,*

```
    w_x(H_t^F) = k − s(t−1),       w_y(H_t^F) = k − s·t          (t ≥ 1)
    w_x(H_t^B) = k − (s+m)j,       w_y(H_t^B) = k − (s+m)(j−1)   (j ≥ 1).
```

*Proof.* `d₁ = v(M_t) + k`, and `v(M_t) = min(−s(t−1), −st) = −st` by Lemma 3.1
(`t ≥ 1`; for `t = 0` all statements are trivial). Since `det J = δ`,
`v(det M_t) = mt`, so `d₁ + d₂ = 2k + mt` gives `d₂ = k + (s+m)t`. Likewise
`v(N_j) = −(s+m)j` and `v(det N_j) = −mj`, giving `d₁ = k − (s+m)j` and
`d₂ = k − mj + (s+m)j = k + sj`. The projection statements are immediate:
`pr_x(M_t p^kZ_p²)` is generated by `p^k` times the entries of the top row of
`M_t`, and `pr_y` by the bottom row. ∎

This already proves the first two lines of Theorem A, using nothing but the
shell condition `v(y_t) = −s` and the margin `m + 2s > 0`.

### Lemma 3.3 (invariant cones)

*Define the* unstable *and* stable *cones*

```
    C^u = { (a,b) ∈ Q_p² : v(a) − v(b) = s },
    C^s = { (a,b) ∈ Q_p² : v(b) − v(a) = s + m }.
```

*Then for every `t`:*

1. `J_t(C^u) ⊆ C^u`, and `v(J_t z) = v(z) − s` for `z ∈ C^u`;
2. `K_t(C^s) ⊆ C^s`, and `v(K_t z) = v(z) − (s+m)` for `z ∈ C^s`.

*Proof.* (1) Let `z = (a,b) ∈ C^u`, so `v(a) = v(b) + s` and hence
`v(z) = v(b)`. Then `J_t z = (b, −δa + 2y_t b)` with

```
    v(−δa) = m + v(b) + s,        v(2y_t b) = v(b) − s,
```

differing by `m + 2s > 0`; so the second coordinate has valuation exactly
`v(b) − s`. The image is `(b, b')` with `v(b) − v(b') = s`, i.e. in `C^u`, and
`v(J_t z) = min(v(b), v(b) − s) = v(z) − s`.

(2) Let `z = (a,b) ∈ C^s`, so `v(b) = v(a) + s + m` and `v(z) = v(a)`. Then
`K_t z = ((2x_t/δ)a − (1/δ)b, a)` with

```
    v((2x_t/δ)a) = v(a) − s − m,        v((1/δ)b) = v(a) + s,
```

differing by `2s + m > 0`; the first coordinate has valuation exactly
`v(a) − s − m`, the image is `(a', a)` with `v(a) − v(a') = s + m`, and
`v(K_t z) = v(a) − s − m = v(z) − (s+m)`. ∎

### Lemma 3.4 (the minimal vectors lie in the cones)

*Set `u_t = M_t · p^k e₂` with `e₂ = (0,1)`, and `w_j = N_j · p^k e₁` with
`e₁ = (1,0)`. Then for `t ≥ 1` and `j ≥ 1`:*

```
    u_t ∈ H_t^F ∩ C^u,      v(u_t) = k − s·t   = d₁(H_t^F),
    w_j ∈ H_{T−j}^B ∩ C^s,  v(w_j) = k − (s+m)j = d₁(H_{T−j}^B).
```

*Moreover every vector of `H_t^F` of valuation `d₁(H_t^F)` lies in `C^u`, and
every vector of `H_{T−j}^B` of valuation `d₁(H_{T−j}^B)` lies in `C^s`.*

*Proof.* `u_1 = p^k J_0 e₂ = p^k(1, 2y_0)`, of coordinate valuations `k` and
`k − s`, hence in `C^u` with `v(u_1) = k − s`. Since `u_{t+1} = J_t u_t`,
Lemma 3.3(1) gives `u_t ∈ C^u` and `v(u_t) = k − st` for all `t ≥ 1`, which
equals `d₁(H_t^F)` by Corollary 3.2; membership `u_t ∈ H_t^F` is by
construction. Similarly `w_1 = p^k K_T e₁ = p^k(2x_T/δ, 1)` has coordinate
valuations `k − s − m` and `k`, so lies in `C^s` with `v(w_1) = k − s − m`, and
Lemma 3.3(2) propagates.

For the last claim, let `g = d₂(H_t^F) − d₁(H_t^F) = (2s+m)t`. In an SNF frame
`(e, e′)` of `H_t^F` the vectors of valuation `d₁` are exactly
`p^{d₁}(αe + p^{g}βe′)` with `α ∈ Z_p^×`, `β ∈ Z_p`; two such differ, after
dividing by `p^{d₁}α`, by a vector of valuation `≥ g`. The cone condition
`v(a) − v(b) = s` on a vector of valuation `v(b)` depends only on that vector
modulo `p^{v(b)+s+1}`, and `g ≥ 2s+m > s`, so all minimal vectors lie in the
same cone as `u_t`. The backward case is identical with `g = (2s+m)j > s+m`. ∎

### Lemma 3.5 (transversality defect 0, with margin)

*Let `1 ≤ t ≤ T−1`, `u = u_t`, `w = w_{T−t}`, `A = v(u) = d₁(H_t^F)`,
`B = v(w) = d₁(H_t^B)`. Then*

```
    v(det[u | w]) = A + B,
```

*and the two terms of the determinant are separated by exactly `2s + m`.
Consequently `ũ = p^{−A}u` and `w̃ = p^{−B}w` form a `Z_p`-basis of `Z_p²`.*

*Proof.* By Lemma 3.4, `v(u_y) = A`, `v(u_x) = A + s`, `v(w_x) = B`,
`v(w_y) = B + s + m`. Hence

```
    v(u_x w_y) = A + B + 2s + m,        v(u_y w_x) = A + B,
```

and `det[u|w] = u_xw_y − u_yw_x` has valuation exactly `A + B`. Then `ũ, w̃`
have valuation `0` — they are primitive in `Z_p²` — and
`det[ũ|w̃] = p^{−A−B}det[u|w]` is a unit, so `[ũ|w̃] ∈ GL₂(Z_p)`. ∎

This is the step that was expected to need a quantitative transversality
estimate (ADP Lemmas 23–24 bound a Lipschitz product `|δ/γ|² < 1` for the
stable and unstable tubes). It needs none: ultrametrically the two directions
are not merely uniformly transverse but *orthogonal*, and the margin `2s+m` is
uniform. Nothing is imported from ADP here: the shell condition
`v(x)=v(y)=−s` is Definition 1.1's hypothesis, and ADP's role is only to
exhibit points that satisfy it.

### Lemma 3.6 (both lattices are diagonal in the common frame)

*With `1 ≤ t ≤ T−1` and `(ũ, w̃)` as in Lemma 3.5, and writing
`A = d₁(H_t^F)`, `A′ = d₂(H_t^F)`, `B = d₁(H_t^B)`, `B′ = d₂(H_t^B)`:*

```
    H_t^F = p^{A}Z_p·ũ  ⊕  p^{A′}Z_p·w̃,
    H_t^B = p^{B′}Z_p·ũ ⊕  p^{B}Z_p·w̃.
```

*Proof.* Work in the coordinates given by the unimodular frame `(ũ, w̃)`; this
changes no valuation and no divisor. In these coordinates `p^A ũ = (p^A, 0) ∈
H_t^F`, and `H_t^F ⊆ p^AZ_p²` because `A = d₁(H_t^F)`. So `H_t^F = p^A L` with
`L ⊆ Z_p²` a sublattice of index `p^{A′−A}` containing `(1,0)`. For such an
`L`: `L ∩ (Z_p × 0)` is a `Z_p`-submodule of `Z_p×0` containing `(1,0)`, hence
equals `Z_p(1,0)`; the second-coordinate projection of `L` is `p^hZ_p` for some
`h ≥ 0`, realised by some `(β, p^h) ∈ L`; every `l ∈ L` satisfies
`l − (l_2/p^h)(β,p^h) ∈ L ∩ (Z_p×0)`, so `L = Z_p(1,0) + Z_p(β,p^h)` and,
subtracting `β(1,0)`, `L = Z_p(1,0) ⊕ Z_p(0,p^h)`. Comparing indices,
`h = A′ − A`. This is the first display. The second is the same argument with
the roles of `ũ` and `w̃` exchanged, using `p^B w̃ ∈ H_t^B ⊆ p^BZ_p²`. ∎

### Proof of Theorem A

The forward and backward lines are Corollary 3.2. For the smoothed line, fix
`t`.

*Case `1 ≤ t ≤ T−1`.* By Lemma 3.6 both lattices are diagonal in the single
unimodular frame `(ũ, w̃)`, so their intersection is diagonal too, with the
larger exponent in each direction:

```
    H_t = p^{max(A, B′)}Z_p·ũ ⊕ p^{max(A′, B)}Z_p·w̃ .
```

Substituting `A = k − st`, `A′ = k + (s+m)t`, `B = k − (s+m)(T−t)`,
`B′ = k + s(T−t)`, and using `−st ≤ 0 ≤ s(T−t)` and
`−(s+m)(T−t) ≤ 0 ≤ (s+m)t`:

```
    max(A, B′) = k + s(T−t),        max(A′, B) = k + (s+m)t .
```

Since the frame is unimodular these two exponents are the elementary divisors
of `H_t`, so `{d₁(H_t), d₂(H_t)} = {k + s(T−t), k + (s+m)t}`, which is the
claim.

*Case `t = 0`.* `H_0^F = p^kZ_p²` is a ball, hence diagonal with equal
exponents in *every* unimodular frame — in particular in an SNF frame of
`H_0^B`, whose divisors are `k − (s+m)T` and `k + sT`. Taking maxima gives
`{k, k + sT}`, matching `k + min(0, sT)` and `k + max(0, sT)`. The case `t = T`
is symmetric. ∎

### Remarks

1. **No periodicity, no eigenvectors, no ADP transversality.** The proof uses
   only: `v(y_t) = v(x_t) = −s` at each point, which is Definition 1.1 and not
   an imported theorem; `det J = δ`; and the margin `m + 2s > 0`. It therefore
   covers orbit segments of arbitrary infinite aperiodic orbits of the 2-shift,
   closing the gap that the numerics could only probe — and more, since the
   sequence need not be an orbit at all.
2. **Lemma 3.1 is logically redundant** given Lemmas 3.3–3.4: `d₁(H_t^F) =
   v(u_t) = k − st` and `d₁ + d₂ = v(det)` already determine both divisors. It
   is kept because it is the cheapest independent confirmation of the divisor
   law, and because Corollary 3.2's *projection* exponents — which the row
   valuations give for free and the cone argument does not — are exactly what
   §4 needs. (Both survive the generalisation of §6; see Lemma 6.4.)
3. **Where the gain comes from.** `d₁(H_t) − k = min((s+m)t, s(T−t)) ≥ 0`: the
   interior of the window is strictly *better* determined than its endpoints,
   by up to `s(s+m)T/(2s+m)` digits. Prediction-only is `d₁(H_t^F) = k − st`,
   which in this regime coincides exactly with a naive scalar precision
   counter: in the worst direction the lattice formalism buys nothing on its
   own, and the entire gain comes from the update step.

---

## 4. Proof of Theorem B′

An algorithm holds a coset `v_t + H_t` and pushes it forward, so the statement
that has to be certified is that the pushforward of a coset under `f` *is* a
coset, with lattice `J(v_t)H_t`. Because `f` is quadratic its Taylor expansion
terminates:

```
    f(v + h)    = f(v)    + J(v)h  + (0, h_y²)
    f⁻¹(v + h)  = f⁻¹(v)  + K(v)h  + (h_x²/δ, 0)
```

so the only obstruction is whether the remainder is absorbed by the propagated
lattice.

**Definition 4.1(a) (exactness / membership test).** Let `H` be a lattice with
`w_y = w_y(H)`, `w_x = w_x(H)`. The forward step at `(v, H)` is **exact** if

```
    R^F(H) := {0} × p^{2w_y}Z_p  ⊆  J(v)·H ,
```

and the backward step is exact if `R^B(H) := p^{2w_x−m}Z_p × {0} ⊆ K(v)·H`.

`R^F(H)` is exactly the `Z_p`-module generated by `{(0, h_y²) : h ∈ H}`, since
`{h_y : h ∈ H} = p^{w_y}Z_p` and squaring an ideal doubles its exponent; so
Definition 4.1(a) is not merely sufficient but *necessary* for the pushforward
to be contained in a coset of `J(v)H`.

**Definition 4.1(b) (inflation).** A tracker cannot stop when the test fails, so
the step is defined for every `(v, H)`: the propagated lattice is

```
    H′ = J(v)·H              if the step is exact,
    H′ = J(v)·H + R^F(H)     otherwise                      (backward: R^B),
```

the smallest lattice containing both the propagated one and the remainder
module. The lattice is thus only ever enlarged, never shrunk, so the enclosure
`v^true ∈ v + H` is preserved unconditionally — Lemma 4.2's containment argument
applies to `H′ ⊇ J(v)H + R^F(H)` verbatim — and a step is called **inflated**
when the second branch is taken. Under Theorem B′'s budget the second branch is
never taken, and the tracked lattices are literally those of Definition 1.2;
past the first inflation they are not, and §3 no longer speaks about them.

**Lemma 4.2 (exact linearisation ⟹ certified enclosure).** *If the forward step
at `(v,H)` is exact then `f(v + H) ⊆ f(v) + J(v)H`, and this containment fails
if the step is not exact. If moreover `w_y(H) > m`, equality holds. Similarly
backward.*

*Proof.* Containment: `f(v+h) − f(v) = J(v)h + (0,h_y²)` and
`(0,h_y²) ∈ R^F(H) ⊆ J(v)H`. Conversely if the step is not exact then, since
`R^F(H)` is generated by the `(0,h_y²)`, some `h ∈ H` has
`(0,h_y²) ∉ J(v)H`, and `f(v+h) ∉ f(v) + J(v)H`.

Equality: given `h ∈ H`, we must solve `f(v+h′) = f(v) + J(v)h` for `h′ ∈ H`,
i.e. find a fixed point of `Φ(h′) = h − J(v)⁻¹(0, h′_y²)`. Exactness gives
`J(v)⁻¹R^F(H) ⊆ H`, hence `Φ(H) ⊆ H`. For `h₁, h₂ ∈ H`,

```
   Φ(h₁) − Φ(h₂) = −J(v)⁻¹(0, z),      z = (h₁_y + h₂_y)(h₁_y − h₂_y),
```

and `J(v)⁻¹(0,z) = (−z/δ, 0)`, so — using `v(h_y) ≥ w_y` for `h ∈ H` —

```
   v(Φ(h₁) − Φ(h₂)) = v(z) − m ≥ w_y + v(h₁ − h₂) − m .
```

Thus `Φ` contracts by the factor `p^{−(w_y − m)} < 1` on the complete
ultrametric space `H`, and has a unique fixed point. The extra hypothesis
`w_y > m` holds throughout under the budget of Theorem B′: there
`w_y(H_t^F) = k − st ≥ 2st + mt + m` with `k ≥ 3s + 2m > m`. ∎

Only the containment is used below; the equality is recorded because it is what
makes the smoother an exact statement about sets rather than an enclosure.

### 4.3 The exactness thresholds

**Lemma 4.3 (axis exponents).** *For a rank-2 lattice `L ⊂ Q_p²`,*

```
    L ∩ ({0} × Q_p) = {0} × p^{κ(L)}Z_p    with  κ(L) = v(det L) − w_x(L),
    L ∩ (Q_p × {0}) = p^{λ(L)}Z_p × {0}    with  λ(L) = v(det L) − w_y(L).
```

*Proof.* Put `L` in column Hermite form `p^{−e}[[p^a, 0],[b, p^d]]Z_p²`. An
element `z₁c₁ + z₂c₂` has first coordinate `0` iff `z₁ = 0`, so the vertical
intersection is generated by `c₂ = p^{−e}(0,p^d)`, giving `κ = d − e`; and
`w_x = a − e`, `v(det L) = a + d − 2e`. The horizontal statement follows from
the row Hermite form. ∎

**Proposition 4.4 (thresholds).** *Let `(v_t)` be `s`-admissible and let the
forward pass carry the idealised lattices `H_t^F` of Definition 1.2. Then:*

```
    the forward step  t → t+1  is exact   ⟺   (3s + m)·t  ≤  k − m ;
    the backward step j → j+1  is exact   ⟺   (3s + 2m)·j ≤  k .
```

*Proof.* Forward. By Corollary 3.2, `w_y(H_t^F) = k − st`, so
`R^F(H_t^F) = {0} × p^{2(k−st)}Z_p`, and the test is
`2(k−st) ≥ κ(H_{t+1}^F)`. By Lemma 4.3 and Corollary 3.2,

```
    κ(H_{t+1}^F) = v(det H_{t+1}^F) − w_x(H_{t+1}^F)
                 = (2k + m(t+1)) − (k − s·t) = k + m(t+1) + s·t .
```

Hence exactness `⟺ 2k − 2st ≥ k + mt + m + st ⟺ k − m ≥ (3s+m)t`.

Backward. `w_x(H_{T−j}^B) = k − (s+m)j`, so `R^B = p^{2(k−(s+m)j)−m}Z_p × {0}`
and the test is `2(k−(s+m)j) − m ≥ λ(H^B_{T−j−1})`. Again by Lemma 4.3 and
Corollary 3.2 (with index `j+1`),

```
    λ = v(det) − w_y = (2k − m(j+1)) − (k − (s+m)j) = k + s·j − m .
```

Hence exactness `⟺ 2k − 2(s+m)j − m ≥ k + sj − m ⟺ k ≥ (3s+2m)j`. ∎

The two rates decompose as advertised — `2s` because the remainder is quadratic
in a projection falling at rate `s`, plus the rate at which the divisor it must
clear rises: `s+m` forward, `s+2m` backward, the extra `m` being the `1/δ` in
the backward remainder `(h_x²/δ, 0)`.

### 4.4 Proof of Theorem B′

Let `T ≥ 2` and `k ≥ (3s+2m)(T−1)`.

*Step 1: the tracked representatives are `s`-admissible.* The forward pass
holds `(v_t, H_t^F)` with `v_t^true ∈ v_t + H_t^F` and `H_t^F ⊆ p^{d₁}Z_p²`,
`d₁ = k − st`. From `k ≥ 3s(T−1)` and `t ≤ T` one gets
`d₁ ≥ 3s(T−1) − sT = s(2T−3) > −s` for `T ≥ 2`. Since `v_t^true ∈ J(f)` has
`v(x) = v(y) = −s` and `v_t − v_t^true` has valuation `> −s`, also
`v(x_t) = v(y_t) = −s`. The same computation backward uses
`d₁ = k − (s+m)(T−t) ≥ k − (s+m)T` and

```
    (3s+2m)(T−1) − ((s+m)T − s) = 2s(T−1) + m(T−2) ≥ 0   for T ≥ 2,
```

so `d₁ > −s` there too. Hence Lemmas 3.1–3.6 apply to the sequence the tracker
actually uses. (Re-reducing the representative modulo its lattice after each
step moves it within the coset, so this is preserved.)

*Step 2: no step inflates.* By Proposition 4.4, every forward step
`t → t+1` with `0 ≤ t ≤ T−1` is exact iff `(3s+m)(T−1) ≤ k − m`, and every
backward step `j → j+1` with `0 ≤ j ≤ T−1` is exact iff `(3s+2m)(T−1) ≤ k`.
For `T ≥ 2`,

```
    (3s+2m)(T−1) − [(3s+m)(T−1) + m] = m(T−2) ≥ 0,
```

so the backward condition is the binding one and `k ≥ (3s+2m)(T−1)` is
equivalent to both holding. In that case the computed lattices are literally
`J_{t−1}⋯J_0 p^kZ_p²` and `K_{t+1}⋯K_T p^kZ_p²`, i.e. those of
Definition 1.2, and Theorem A gives all four divisor laws.

*Step 3: certification.* By induction using Lemma 4.2:
`v_0^true ∈ v_0 + p^kZ_p²` by hypothesis, and if `v_t^true ∈ v_t + H_t^F` then
`v_{t+1}^true = f(v_t^true) ∈ f(v_t + H_t^F) = f(v_t) + J(v_t)H_t^F =
v_{t+1} + H_{t+1}^F`. Likewise backward. Hence `v_t^true` lies in both cosets,
so in their intersection, which is `v_t + H_t` with `H_t = H_t^F ∩ H_t^B`.
Conversely, if `k < (3s+2m)(T−1)` then some step fails Definition 4.1(a), and
because `R^F` (resp. `R^B`) is *generated* by actual remainders there is an
`h ∈ H` with `(0,h_y²) ∉ J(v)H`, so the pushforward genuinely leaves the coset
and the lattice must be enlarged. ∎

**Corollary 4.5 (certified shadowing).** Given `v_0^true` and `v_T^true` each
known to `k ≥ (3s+2m)(T−1)` digits, every intermediate point of the connecting
orbit is determined to at least `k` digits, and to `k + s(s+m)T/(2s+m)` digits
at `t = t*` (up to integer rounding of `t*`). The information-theoretic loss is `0`; the arithmetic cost is a
working precision linear in `T` at `3s+2m` digits per step.

### 4.5 The forward-rate budget `(3s+m)T + s + m` is insufficient for `m > 0`

This makes Remark 2.1 explicit. `3s+m` is the *forward* rate. Since

```
    (3s+m)T + s + m − (3s+2m)(T−1) = 4s + m(3 − T),
```

the stated budget falls below the true one as soon as `m > 0` and
`T > 3 + 4s/m`. Explicit counterexample, verified in code: `p = 3`, `s = 1`,
`m = 2`, `T = 10`. The stated budget gives `k = 53`; the sharp budget is
`(3s+2m)(T−1) = 63`. At `k = 53` the forward pass is exact throughout, but the
**backward pass inflates** — first at step `j = 8`, since `(3s+2m)·8 = 56 >
53`, i.e. at times `t = 1, 0` — so the certified lattices are strictly larger
than the idealised ones and the equality of Theorem A fails for the certified
objects.
At `k = 63` both passes are clean. For `m = 0` the two rates agree and the
correction is invisible — which is why it was not caught: every horizon
measurement made so far had `m = 0` or `s = 0`.

---

## 5. What was verified numerically

Everything below is exact integer/rational arithmetic with assertions — no
floating point — on orbits built from symbolic itineraries by multivariate
Newton, so the Jacobian genuinely varies along the orbit.

*Theorem A.* The four-divisor law was confirmed **as an equality** (`C = 0`,
not a bound), with transversality defect `0` throughout, over: itineraries of
period 1–20; `T = 4 … 100`; `k = 0 … 194`; `p = 3, 5, 7, 11`; `s = 1, 2, 3`;
`m = 0, 1, 2`. Periodicity was ruled out as an artefact by cutting 15 windows
of length `T ∈ {24, 48}` from orbits of random length-200 itineraries whose
minimal cyclic period is asserted to be 200. The lattices are also the
*minimal* correct ones — measured slack `0` at every step — so `C = 0` is not
an artefact of a loose enclosure. Here **slack** is
`v(det H_sampled) − v(det H_t)`, where `H_sampled` is the lattice generated by
the exact images of many sampled members of the initial coset: it satisfies
`H_sampled ⊆ H_true ⊆ H_t` with `H_true` the smallest correct lattice, so the
slack is an upper bound, in digits, on how much the tracker overstates its own
uncertainty, and slack `0` forces `H_t = H_true`. Finally,
`v_t^true ∈ v_t + H_t` held at 247,325 certified steps over 10⁴ random orbits
with zero violations, against a mutation control (inflation disabled) caught on
2,976 of them.

*Theorem B′.* Proposition 4.4's two thresholds were confirmed as exact
predictions of the first inflating step in each pass, and the budget as an
*iff*: at `k = (3s+2m)(T−1)` neither pass inflates and at `k − 1` one does.
Under the budget the certified lattices are literally equal to the idealised
ones.

*The lemmas.* Every identity used in §3–§4 was checked before being relied on,
over six `(p, s, m)` regimes including `m = 1, 2`: the row valuations of
Lemma 3.1; the projection and axis exponents of Corollary 3.2 and Lemma 4.3;
the cone gaps of Lemma 3.4; the unimodularity of `(ũ, w̃)` and the diagonality
of *both* lattices in that frame (Lemma 3.6); the shell condition on the
tracker's own representatives (§4.4 Step 1); and the counterexample of §4.5.

One earlier guess — that the forward and backward minimal directions *converge*
at a geometric rate, which would have given the defect-0 statement — was
measured and is false: the minimal direction tracks the moving orbit point at
`O(1)` per step. The invariant-cone formulation of Lemma 3.3 replaced it.

---

## 6. The linear-algebra core

The proof of §3 mentions the Hénon map only through Lemma 3.3, and Lemma 3.3
mentions it only to verify two cone conditions. This section states what is
left when the map is removed: a theorem about sequences in `GL₂(Q_p)`, with
Theorem A as a corollary. The point of doing this is item 2 of §7 — the factors
of a Friedland–Milnor composition have different maps but the same cones — and
the honest scope of the result is recorded in Remark 6.7.

**Setup 6.1 (the axioms).** Fix an odd prime `p`, integers `s > 0` and `m ≥ 0`,
and the cones of Lemma 3.3,

```
    C^u = { (a,b) : v(a) − v(b) = s },      C^s = { (a,b) : v(b) − v(a) = s+m }.
```

Let `J_0, …, J_{T−1} ∈ GL₂(Q_p)`. Call the sequence **admissible** if for every
`t`:

```
  (H1u)   J_t(C^u) ⊆ C^u    and   v(J_t z)    = v(z) − s        for z ∈ C^u ;
  (H1s)   J_t⁻¹(C^s) ⊆ C^s  and   v(J_t⁻¹ z)  = v(z) − (s+m)    for z ∈ C^s ;
  (H2)    v(det J_t) = m .
```

No relation whatever is imposed between consecutive `J_t`, and no companion or
other structure is imposed on any single one. Put, as in Definition 1.2,

```
    M_t = J_{t−1}⋯J_0  (M_0 = I),     N_j = J_{T−j}⁻¹⋯J_{T−1}⁻¹  (N_0 = I),
    H_t^F = M_t p^kZ_p²,   H_t^B = N_{T−t} p^kZ_p²,   H_t = H_t^F ∩ H_t^B .
```

Along a Hénon orbit `J_t⁻¹ = K(v_{t+1})` by the chain rule, so this `N_j` is
the `N_j` of Definition 1.2.

> **Theorem C (four-divisor law for admissible sequences).**
> Let `J_0, …, J_{T−1}` be admissible and `k ∈ Z`. Then for every `0 ≤ t ≤ T`
>
> ```
>   forward    d₁(H_t^F) = k − s·t                d₂ = k + (s+m)·t
>   backward   d₁(H_t^B) = k − (s+m)·(T−t)        d₂ = k + s·(T−t)
>   smoothed   d₁(H_t)   = k + min((s+m)·t, s·(T−t))
>              d₂(H_t)   = k + max((s+m)·t, s·(T−t))
> ```
>
> and for `1 ≤ t ≤ T−1` the transversality defect of `(H_t^F, H_t^B)` is `0`.

The proof needs one preliminary, which is also the finding of Remark 6.7.

### Lemma 6.2 (the axioms force a valuation shape)

*Let `J = [[a, b], [c, d]] ∈ GL₂(Q_p)` satisfy (H1u), (H1s), (H2). Then*

```
    v(b) = 0,      v(d) = −s,      v(a) > −s,      v(c) = m ,
```

*which is exactly the valuation pattern of the Hénon companion matrix
`[[0, 1], [−δ, 2y]]`. Consequently*

```
    v(J) = −s,     v(J⁻¹) = −(s+m),     J e₂ ∈ C^u,     J⁻¹ e₁ ∈ C^s .
```

*Proof.* Let `z = (z₁, z₂) ∈ C^u` with `v(z₂) = 0`, so `v(z₁) = s` and
`v(z) = 0`. By (H1u), `Jz ∈ C^u` with `v(Jz) = −s`; since the gap in `C^u` is
`s`, this says the two coordinates of `Jz` have valuations exactly `0` and `−s`.

*First coordinate.* `az₁ + bz₂` must have valuation `0` for **every** such `z`,
and `v(az₁) ≥ v(a) + s`, `v(bz₂) = v(b)`. If `v(a) + s = v(b)` with `a, b ≠ 0`,
take `z₁ = −(b/a)z₂`, which is admissible because `v(b/a) = s`; the first
coordinate then vanishes and `Jz ∉ C^u`. So the two terms never tie, the
valuation is `min(v(a)+s, v(b))` identically, and that minimum is `0`. Hence

```
    either   (I)  v(b) = 0 and v(a) > −s,     or   (II)  v(a) = −s and v(b) > 0.
```

*Second coordinate.* Identically, `cz₁ + dz₂` has valuation `−s` for every such
`z`, `v(c) + s ≠ v(d)` (else choose `z₁ = −(d/c)z₂`, admissible since
`v(d/c) = s`, killing the coordinate), and `min(v(c)+s, v(d)) = −s`:

```
    either   (i)  v(d) = −s and v(c) > −2s,   or   (ii) v(c) = −2s and v(d) > −s.
```

*Three of the four combinations are impossible.* In (I)+(ii),
`v(bc) = −2s` while `v(ad) > −2s`, so `v(det J) = −2s < 0 ≤ m`, contradicting
(H2); in (II)+(i), `v(ad) = −2s` while `v(bc) > −2s`, the same contradiction.
For (II)+(ii) use (H1s): write `J⁻¹ = (det J)⁻¹[[d, −b], [−c, a]]` and take
`z ∈ C^s` with `v(z₁) = 0`, `v(z₂) = s+m`. Then `J⁻¹z ∈ C^s` with
`v(J⁻¹z) = −(s+m)`, so its first coordinate has valuation exactly `−(s+m)`,
i.e. `v(dz₁ − bz₂) = m − (s+m) = −s`. But in (II)+(ii) `v(dz₁) = v(d) > −s` and
`v(bz₂) = v(b) + s + m > −s`, so that valuation is `> −s`. Hence (I) and (i)
hold: `v(b) = 0`, `v(a) > −s`, `v(d) = −s`, `v(c) > −2s`.

*The value of `v(c)`.* With `z ∈ C^s` as above, `J⁻¹z ∈ C^s` has gap `s+m` and
minimum `−(s+m)`, so its **second** coordinate has valuation exactly `0`, i.e.
`v(−cz₁ + az₂) = m`. Here `v(cz₁) = v(c)` and
`v(az₂) = v(a) + s + m > m` by (I). If `v(c) > m` both terms exceed `m` and so
does the sum; therefore `v(c) ≤ m`, no tie is possible, and the valuation is
`v(c)`. Hence `v(c) = m`.

The consequences are immediate: `v(J) = min(v(a), 0, m, −s) = −s` because
`m ≥ 0 > −s` and `v(a) > −s`; `v(J⁻¹) = −m + min(−s, 0, m, v(a)) = −(s+m)`;
`Je₂ = (b, d)` has gap `0 − (−s) = s`; and `J⁻¹e₁ = (d, −c)/det J` has
coordinate valuations `−(s+m)` and `0`, gap `s+m`. ∎

### Proof of Theorem C

*Lower bounds on `d₁`.* By Lemma 6.2, `v(J_t) = −s` and `v(J_t⁻¹) = −(s+m)`
for every `t`, so `v(M_t) ≥ −st` and `v(N_j) ≥ −(s+m)j` because the valuation of
a matrix product is at least the sum of the valuations. Since
`d₁(AZ_p²) = v(A)`,

```
    d₁(H_t^F) ≥ k − st,        d₁(H_{T−j}^B) ≥ k − (s+m)j .
```

*Matching upper bounds.* Put `u_t = M_t p^k e₂` and `w_j = N_j p^k e₁`. By
Lemma 6.2, `u_1 = p^k J_0 e₂ ∈ C^u` and `w_1 = p^k J_{T−1}⁻¹ e₁ ∈ C^s`, with
`v(u_1) = k − s` and `v(w_1) = k − (s+m)`. Applying (H1u) and (H1s) repeatedly,

```
    u_t ∈ H_t^F ∩ C^u  with  v(u_t) = k − st,
    w_j ∈ H_{T−j}^B ∩ C^s  with  v(w_j) = k − (s+m)j
```

for all `t, j ≥ 1`. As `d₁` is the minimum valuation attained on the lattice,
this bounds `d₁` above by the same quantities, so both are equalities. The
second divisors follow from `d₁ + d₂ = v(det)`, which is `2k + mt` forward and
`2k − mj` backward by (H2).

*The rest is §3 verbatim.* The remaining steps consume only the divisor values
just established, the cone memberships of `u_t` and `w_j`, the two gaps `s` and
`s+m`, and the margin `2s + m > 0` — no property of the matrices beyond those.
Concretely: the second half of Lemma 3.4 (every minimal vector of `H_t^F` lies
in `C^u`, because the divisor gap `(2s+m)t` exceeds `s`, and dually for
`H_t^B`); Lemma 3.5 (the determinant of the two minimal vectors has valuation
`d₁(H_t^F) + d₁(H_t^B)` exactly, its two terms being separated by `2s+m`, so
the defect is `0` and the normalised pair is a unimodular frame); Lemma 3.6
(both lattices are diagonal in that frame); and the case analysis proving
Theorem A, which takes the larger exponent in each direction, and treats
`t = 0` and `t = T` by the ball argument. Substituting the same four values
gives the smoothed line. ∎

### Lemma 6.4 (the row valuations, hence the projections, also survive)

*For an admissible sequence and `1 ≤ t, j ≤ T`, the rows of `M_t` have
valuations `−s(t−1)` (top) and `−st` (bottom), and the rows of `N_j` have
valuations `−(s+m)j` (top) and `−(s+m)(j−1)` (bottom). Consequently
Corollary 3.2's projection exponents hold verbatim:*

```
    w_x(H_t^F) = k − s(t−1),      w_y(H_t^F) = k − s·t
    w_x(H_t^B) = k − (s+m)j,      w_y(H_t^B) = k − (s+m)(j−1),    j = T − t.
```

*Proof.* This is Lemma 3.1's induction with the companion entries replaced by
the shape of Lemma 6.2. Writing `R, S` for the top and bottom rows of `M_t`,
the rows of `M_{t+1} = J_t M_t` are `aR + bS` and `cR + dS`. For the first,
`v(aR) > −s − s(t−1) = −st` and `v(bS) = −st`, so in the coordinate attaining
`v(S) = −st` the second term strictly dominates and the row valuation is
exactly `−st`. For the second, `v(cR) ≥ m − s(t−1)` and `v(dS) = −s(t+1)`,
separated by `m + 2s > 0`, so it is exactly `−s(t+1)`. The base case `t = 1` is
Lemma 6.2's shape read off `M_1 = J_0`. Backward, the rows of
`N_{j+1} = J_{T−j−1}⁻¹N_j` are `(dR − bS)/det` and `(−cR + aS)/det`; the first
has terms of valuations `−(s+m)(j+1)` and `−m − (s+m)(j−1)`, separated by
`2s + m > 0`; the second has `v(cR/det) = −(s+m)j` and
`v(aS/det) > −(s+m)j`. The projection statements follow as in Corollary 3.2,
`pr_x` being generated by the top row and `pr_y` by the bottom row. ∎

This matters because §4's budget is computed from the projection exponents, not
from the elementary divisors. Lemma 6.4 says that input is available for any
admissible sequence; what does **not** transfer is the rest of §4, which is
about the quadratic remainder of a specific map and not about its Jacobians.

### Corollary 6.5 (Theorem A)

*Theorem A, for orbit segments, is the case of Theorem C in which
`J_t = J(v_t)` along an orbit segment in `J(f)`. (§3 proves Theorem A for every
`s`-admissible sequence, orbit or not; the route through Theorem C needs the
orbit relation, because Setup 6.1 builds the backward products from `J_t⁻¹`,
which agrees with `K(v_{t+1})` only along an orbit.)*

*Proof.* (H1u) and (H2) are Lemma 3.3(1) and `det J = δ`. For (H1s), the chain
rule gives `J(v_t)⁻¹ = K(v_{t+1})` and `v_{t+1} ∈ J(f)` is again `s`-admissible,
so Lemma 3.3(2) applies to it. ∎

### Corollary 6.6 (companion Jacobians of any degree)

*Let `g(x,y) = (y, P(y) − δx)` with `deg P ≥ 2`, `v(δ) = m ≥ 0`, and let
`(v_t)` be a sequence of points at which `v(P′(y_t)) = −s < 0`. Then the
Jacobians `J_t = [[0, 1], [−δ, P′(y_t)]]` are admissible, so Theorem C applies
to them.*

*Proof.* The computation of Lemma 3.3 uses `2y_t` only through its valuation
`−s`; replacing it by `P′(y_t)` changes nothing. ∎

This is the linear half of item 2 of §7: for a Friedland–Milnor composition
`g_n ∘ ⋯ ∘ g_1` the Jacobian sequence over one cycle is a concatenation of such
factors. Two caveats. It is only the linear half — the missing ingredient is a
horseshoe theorem guaranteeing the hypothesis `v(P′) = −s` on an invariant set,
which is exactly what ADP supplies for a single quadratic map and what is not
available for compositions. And Setup 6.1 fixes **one** pair of cones, so a
composition is covered only when every factor has the same `s` and the same
`m`; distinct `(s_i, m_i)` would need a moving-cone version of Theorem C, which
is not proved here.

### Remark 6.7 (what the hypotheses really cost)

Theorem C is stated for arbitrary matrices, but Lemma 6.2 shows the axioms are
not as permissive as that sounds: any admissible `J` has the valuation shape of
a Hénon companion matrix, with all four entries free but their valuations
pinned. The generalisation is therefore not "the law needs nothing about
Hénon"; it is "the law needs nothing about Hénon beyond a valuation shape that
the cone axioms themselves force". What is genuinely gained is the removal of
*orbit* structure — nothing relates consecutive `J_t`, so §3's `s`-admissibility
hypothesis is not used — and the removal of the companion *form*, which is what
Corollary 6.6 and the composition programme need. It is worth stating in this
direction because the converse reading, that cone-preservation alone is a weak
hypothesis, is false.

The four-divisor law and the defect-0 statement were checked numerically at all
three levels of generality — generalised-companion sequences, conjugates of the
diagonal model with all entries generic and eigendirections moving at each
step, and matrices found by rejection sampling against the hypotheses with no
construction recipe at all — as were Lemma 6.2, its consequences, and
Lemma 6.4, over the same six `(p, s, m)` regimes as §5 (§8).

---

## 7. Generalisation targets (conjectural)

Nothing in §3 uses the specific map beyond four properties, at every point of
the invariant set: **(H1)** the Jacobian's characteristic polynomial has Newton
slopes `−s < 0` and `s+m`, with no tangencies; **(H2)** a positive
no-cancellation margin between the two slopes, here `m + 2s`; **(H3)** a
terminating Taylor expansion, so the linearisation remainder is a finite sum
absorbable under a membership condition; **(H4)** a polynomial inverse, so the
backward pass is the same kind of object.

The pure linear-algebra core that was listed here as the first target is now
§6 (Theorem C), and it is proved rather than conjectural; Corollary 6.6 covers
the companion Jacobians of any degree. The remaining targets:

2. **Compositions of generalised Hénon maps.** By Friedland–Milnor every
   dynamically nontrivial polynomial automorphism of the plane is conjugate to
   `g_n ∘ ⋯ ∘ g_1` with `g_i(x,y) = (y, P_i(y) − δ_i x)`, `deg P_i ≥ 2`. Each
   factor has companion Jacobian `[[0,1],[−δ_i, P_i′(y)]]`, so Corollary 6.6
   applies verbatim whenever `v(P_i′)` is constant `= −s < 0` and `v(δ_i) = m`
   for **all** factors — the cones of Setup 6.1 are fixed, so a composition
   whose factors have different `(s_i, m_i)` is not covered, and the expected
   per-cycle rates `Σ s_i` and `Σ (s_i + m_i)` are conjectural: they would need
   a version of §6 in which the cone moves from step to step. What is genuinely
   missing is an ADP-style horseshoe theorem for such compositions; a cheaper
   intermediate is `f^n` for a single `f`, where the regime is inherited.
3. **Degree `d` budget.** For `deg P = d` the membership condition involves
   `j·w` for `j = 2, …, d`; the binding constraint is conjecturally still
   `j = 2`, so the exchange rate would stay `2s + (rise rate)` and not see the
   degree. This is cheap to test before proving.
4. **Portable extensions.** The whole argument is valuation-theoretic, so
   equal characteristic `F_p((T))` should transfer verbatim. Dimension `n > 2`
   replaces (H1) by an `(n_u, n_s)` Newton-polygon splitting and the row
   induction by a block induction — heavy bookkeeping, no visible conceptual
   obstacle. Residue characteristic 2 is open in ADP already (`|2y|`
   degenerates) and is out of reach of this route.

Open Problem 1 of §2 — what happens below Theorem B′'s budget — is the one gap
in the *proved* material rather than a generalisation target.

---

## 8. Reproducibility

Everything reported in §5 and §6 is executable. The accompanying Python package
implements rank-2 lattices in `Q_p²` in Hermite normal form over `Z_p` (a
canonical representation, so lattice equality is tuple equality), the Hénon map
and its inverse, periodic orbits by multivariate Newton on a symbolic
itinerary, and the forward, backward and smoothed precision passes with the
membership test of Definition 4.1. All arithmetic is `int`/`Fraction`.

- `uv run pytest` runs the unit tests. `tests/test_proof_lemmas.py` pins every
  identity listed in §5 as an exact assertion, one test per lemma, over the six
  `(p, s, m)` regimes; the `m = 2` counterexample of §4.5 is
  `test_the_3s_plus_m_budget_is_insufficient_for_m_positive`, and the *iff* of
  Theorem B′ is `test_budget_threshold_is_an_iff`. `tests/test_cone_axioms.py`
  covers §6: it generates matrix sequences that are not Hénon Jacobians at the
  three levels of generality described in Remark 6.7, asserts (H1u), (H1s),
  (H2) on the generated matrices before asserting any conclusion, and then
  checks Theorem C's four divisors and defect, Lemma 6.2's valuation shape and
  its consequences, and Lemma 6.4's projection exponents.
- `uv run python experiments/run_all.py` reruns all ten measurements. Each
  writes a figure and a JSON record carrying every parameter and every per-step
  number, both under `results/`.

  | script | what it measures |
  |---|---|
  | `exp_theorem.py` | the Theorem A sweep (§5) |
  | `exp_aperiodic_window.py` | windows cut from aperiodic orbits (§5) |
  | `exp_5_2_horizon.py` | Proposition 4.4's two horizons and the `iff` boundary sweep |
  | `exp_5_4_certification.py` | the certified-step count and its mutation control |
  | `exp_5_5_slack.py` | minimality of the lattices (slack, §5) |
  | `exp_headline.py` | the headline forward-vs-smoothed comparison |
  | `exp_5_1_anisotropy.py` | growth of the anisotropy `d₂ − d₁`, by regime |
  | `exp_5_3_smoother.py` | whether the backward pass recovers the lost direction, by regime |
  | `exp_probabilistic.py` | a separate 1D noisy-digit experiment: measures on the depth-`k` `p`-ary tree, not lattices |
  | `exp_5_6_baseline.py` | cross-check against `review_checks.py`, an independent implementation; records the optional SageMath comparison that is not run here |

Two implementation facts are worth stating because they are not optional.
First, the coset representative must be reduced modulo its own lattice at every
step: `f` squares its argument, so an unreduced 15-digit start becomes ~2.5×10⁸
digits in 24 steps. Second, tracking is done in the *unscaled* coordinates. The
scaled coordinates `X = p^s x` put the orbit in `Z_p²`, but the scaled map
divides by `p^s` and that division is exact only on the invariant set; since
the scaling is by the scalar matrix `p^s I` it shifts every lattice exponent
uniformly and changes no rate, slope or tent, so nothing is lost by avoiding
it.
