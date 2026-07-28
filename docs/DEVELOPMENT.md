# Development notes

Read this before touching code. Every item below cost real time at least once.
The code-level invariants are guarded in code now; don't remove the guards.

## Running things

```bash
uv run pytest                          # 215 tests, ~5 s
uv run python experiments/run_all.py   # 10 experiments, ~230 s
uv run python main.py                  # smallest end-to-end demo
uv run python review_checks.py         # the original review numbers
```

Timings: `exp_5_4_certification.py` is ~154 s (10⁴ orbits, by spec) and
`exp_aperiodic_window.py` ~63 s (length-200 Newton solves); every other script
is under 7 seconds. When iterating, lower `N_ORBITS` in
`exp_5_4_certification.py`.

## Environment

**Run long jobs under `ulimit -v 6000000`.** The map squares its argument, so
anything that escapes the reduction guards allocates fast.

**`uv run pytest` can fail with `Failed to spawn: pytest` after a directory
rename.** The venv's console-script shebang still points at the old absolute
path; `uv run python` is unaffected, which makes it look like a pytest problem
rather than a path problem. Fix with `uv sync --reinstall-package pytest`. The
workaround `uv run python -m pytest` always works and is the safer form in
scripts.

**One unexplained pytest hang.** A `uv run pytest -q` invocation once hung for
hours after the same command had completed in ~5 s minutes earlier; it did not
recur across many subsequent runs and nothing was changed to address it. The
suite has no I/O, no network and no background processes. If it comes back,
bisect with `-p no:cacheprovider` and `--timeout`, and suspect the environment
before the tests.

**Don't poll with `pgrep -f <script>`.** It matches the polling shell's own
command line and loops forever. Wait on the background task's output file or
its completion notification instead.

## Code traps

**Unbounded exact arithmetic will take the machine down.** The map squares its
argument, so an unreduced iterate doubles in digit count every step: a 15-digit
start is ~2.5×10⁸ digits after 24 steps. Every representative must be reduced
modulo its own lattice each step (`Lattice.reduce_vector`).
`henon.MAX_REPR_BITS` and `lattice.MAX_EXPONENT` are tripwires that raise
instead of allocating. `Henon.orbit` takes a `prec` argument that reduces each
iterate — pass it for any `T` beyond a handful, and note it is lossless only
when `s = 0`; on the horseshoe build ground truth with `periodic_orbit` in
scaled coordinates instead.

**Track in unscaled coordinates.** The scaled form `X = p^s x` puts the orbit
in `ℤ_p²`, but the scaled update divides by `p^s` and that division is exact
*only on the invariant set*. Once `d₁ < 0` the canonical coset representative
can leave the horseshoe and the arithmetic dies with a spurious "not divisible"
error. Scaling is by the **scalar** matrix `p^s I`, so it shifts every lattice
exponent uniformly and changes no rate, slope or tent — the trackers therefore
iterate the unscaled polynomial map, which has no division at all.
`Henon.f_scaled` keeps the scaled form and documents this.

**Working precision must exceed `d₂`, not `d₁`.** `_assert_certified` raises
`working precision exhausted` once `H.d2 >= truth_precision`, and `d₂` climbs
at `s+m` per step. Building a `T`-step window at precision `k` therefore needs
ground truth to about `k + (s+m)·T`. This bites as soon as `m > 0`: a
hardcoded `prec=600` is fine at `s=2, m=0` and blows through the ceiling at
`m = 2`. `exp_5_2_horizon.truth_for` computes `prec` from `k` and `T`; any new
experiment in a skewed regime must do the same.

**`transversality_defect` returns `None` for isotropic lattices.** A ball
(`d₁ = d₂`, e.g. at `t = 0`) has no distinguished worst direction. Filter those
out before asserting; don't count them as defects.

**`Lattice.x_projection_exponent` and `y_projection_exponent` are methods, not
properties.** Comparing them without calling them compares bound methods
against ints, which is silently `False` rather than an error. Anything that
reports uniformly `False` across every regime is a bug in the probe, not a
discovery.

**`general_sequence` asserts all four entries of `J` are nonzero**, so some RNG
seeds raise `AssertionError` from inside the *generator*. That is a degenerate
frame, not a failed law. Seeds 4093, 7717 and 90210 all pass for the six
regimes; pick one that passes and say why in a comment, rather than relaxing
the assertion.

**Rejection sampling over valuation grids needs a tuned proposal.** The
admissible set of the cone axioms is cut out by exact valuation equalities, so
a uniform draw over a wide grid accepts *nothing* — the first attempt found
0/12 in 4000 tries and looked like a failed law rather than a failed sampler.
The proposal ranges in `tests/test_cone_axioms.py` are documented as a
proposal, not a hypothesis; the filter is the definition. If you widen them,
raise `tries` accordingly, and keep the `seen > 3*n` assertion that stops the
filter from silently accepting everything.

**`_common.EXTRA` holds four colours.** `exp_5_2` plots five distinct `(s, m)`
curves and silently reused blue for two of them until the figure was looked at;
it extends the palette locally. Look at the PNG, not just the assertions — the
assertions cannot see a colour collision.

**Tangencies break the clean slope law in `H⁺_II`.** Over `ℚ₅` with ADP
Table 1's `(a, b) = (1, 5)`, the orbit from `(1,1)` has `v_p(y_t) = 1` at every
odd step; the middle Newton-polygon vertex leaves the lower hull, eigenvalue
valuations ramify to `m/2`, and anisotropy oscillates. Assert the slope-`m` law
only on tangency-free orbits, as `exp_5_1` does. `H_III` has no tangencies —
`|2y| = p^s` always.

## Invariants asserted in code — don't loosen them

- `v_true ∈ v + H` at every step of every track;
- `v_p(det H)` rises by exactly `m` per step (`det J = δ`);
- inflation only ever *enlarges* a lattice;
- the sharp four-divisor law of `NOTE.md` Theorem A;
- the `H⁺_II` smoother is vacuous (`F ∩ B = F`).

**The precision floor is not optional.** Past `d₁ < −s` the coset is wider than
the invariant set and the quadratic remainder *squares* the uncertainty each
step — exponents double per iteration (−2, −4, −8, … −2048 by `t = 24`).
Tracks stop at the floor and report `exhausted_at`
(`precision.PrecisionExhausted`). Correct, worthless, and a memory hazard; do
not "fix" this by raising the floor.

## Methodology

**`C = 0` passing is not evidence the budget is right.** Inflation at the ends
of a pass does not move `d₁` in the middle of the tent, so an
under-provisioned run passes `assert C == 0` while silently violating the
hypothesis it claims to honour. Always assert `first_inflation is None`
alongside `C == 0`. That single change is what caught the wrong-budget bug, and
it is now enforced in `exp_theorem.certified_C` (via `expect_exact`) and in
`exp_aperiodic_window.certified`.

**The two passes are not mirror images.** Anything derived for the forward pass
must be re-derived for the backward one whenever `m > 0`:

| | forward | backward |
|---|---|---|
| rate | `s` | `s+m` |
| remainder axis | `y` | `x` |
| remainder exponent | `2w` | `2w − m` |
| budget | `3s+m` | `3s+2m` |
| horizon | `(k−m)/(3s+m)` | `k/(3s+2m)` |

Only `m = 0` is symmetric, and almost all of the historical test coverage was
`m = 0` — which is exactly how the wrong budget survived. **Check every new
claim at `m = 1` and `m = 2`**; `tests/test_proof_lemmas.py:REGIMES` includes
both, and `params.py` has `HORSESHOE_3ADIC_M1` and `_M2` for the purpose.

**Measure before theorising.** Two plausible-sounding claims were written into
a roadmap and later measured false: "the SNF directions converge at a geometric
rate" and "the budget is `(3s+m)T + s + m`". Each cost a session.
`tests/test_proof_lemmas.py` exists so that any derived identity can be
falsified in seconds; use it *before* the identity becomes load-bearing. This
is not caution, it is the repo's actual comparative advantage — and it has paid
off in the other direction too: measuring the exactness horizons into a scratch
table before writing them as `==` assertions took two minutes and meant the
tightening landed first try.

**`NOTE.md` Prop 4.4 at `s = 0` is measured, not proved.** `NOTE.md`'s standing
hypotheses put `s > 0` (the horseshoe), but §4's computation only consumes the
divisor law of Cor 3.2, which in `H⁺_II` reads `d₁ = k`, `d₂ = k + mt` — Cor
3.2 with `s = 0`. Both horizons come out exactly right there, and `exp_5_2`
relies on it, so it is pinned in
`test_prop_4_4_also_holds_in_the_attractor_region` and flagged in that test's
docstring as an empirical extension. Do not quote it as a consequence of
`NOTE.md` without doing the (probably short) work.

## Determinism

Seeds are fixed at call sites; every experiment logs its parameters into
`results/<name>.json` alongside the figure. `results/` is committed on purpose:
the figures are referenced from the documentation and the JSON files are the
record of what each run actually produced.
