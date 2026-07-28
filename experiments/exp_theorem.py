"""Evidence for THEOREM.md: is the constant C uniform in T, t, and the orbit?

THEOREM.md §3 claims a constant ``C = C(p, c, delta)``, independent of ``T``,
``t`` and the choice of orbit in ``J(f)``, with

    d1(H_t)  >=  k + s*min(t, T-t) - C.

This measures ``C`` directly, as ``max_t [ k + s*min(t, T-t) - d1(H_t) ]``, in
two settings that THEOREM.md treats as one:

  A. **Idealised** lattices -- the ones §2 actually defines, pure Jacobian
     products with no inflation.  Here C = 0 unconditionally.

  B. **Certified** lattices -- §2 plus the Proposition, i.e. enclosures for
     which ``v_true in v + H`` is asserted at every step.  Here C = 0 only
     while the quadratic remainder stays inside the propagated lattice, which
     costs a starting-precision budget linear in T.

The difference between A and B is the whole content of the update: the
smoothing mechanism never fails, but certifying it is not free.
"""

from __future__ import annotations

import random
from fractions import Fraction as F

from _common import EXTRA, INK_MUTED, report, save, style, write_json  # noqa: I001
from padic_filtering.henon import (Henon, fixed_point_c, periodic_orbit,
                                   truth_orbit, truth_precision)  # noqa: F401
from padic_filtering.lattice import Lattice, transversality_defect
from padic_filtering.precision import (perturb, run_backward, run_forward,
                                       run_smoother)

SEED = 20260726

# (p, alpha, delta) triples: alpha of valuation -s makes (alpha, alpha) a fixed
# point of the horseshoe map with c = alpha + delta*alpha - alpha^2.
REGIMES = [
    (3, F(1, 3), 1), (5, F(1, 5), 1), (7, F(1, 7), 1), (11, F(1, 11), 1),
    (3, F(1, 9), 1), (3, F(1, 27), 1),
    (3, F(1, 3), 3), (3, F(1, 3), 9), (5, F(1, 25), 5),
]
ITINERARIES = ["+", "+-", "++-", "++-+--+-", "+--++-+-+--+",
               "-++--+-+++--+-+--+-"]


def build(p, alpha, delta, itinerary, T, k, slack=6):
    hen = Henon.from_c(p, fixed_point_c(p, delta, alpha), delta)
    hen.check_regime()
    prec = slack * (k + T * (hen.s + hen.m)) + 300
    X = periodic_orbit(hen, itinerary, prec)
    return hen, truth_orbit(hen, X, T, prec), truth_precision(hen, prec)


def budget(hen, T) -> int:
    """Starting precision keeping BOTH passes exact for T steps (NOTE.md Thm B').

    The sharp threshold is ``k >= (3s+2m)(T-1)`` -- an *iff* (NOTE.md Prop 4.4).
    ``3s+m`` is the *forward* rate only; the backward remainder ``(h_x^2/delta,
    0)`` carries an extra ``1/delta``, so the backward pass pays ``3s+2m`` and
    the two coincide only when ``m = 0``.  The ``s + m + 4`` is slack.
    """
    return (3 * hen.s + 2 * hen.m) * (T - 1) + hen.s + hen.m + 4


def tent(hen, k, T):
    """The sharp smoothed law -- *skewed* unless delta is a unit.

    THEOREM.md §3 writes the tent as ``k + s*min(t, T-t)``, which is the
    ``m = 0`` case.  In general the forward pass loses ``s`` per step while the
    backward pass loses ``s + m`` (the extra ``m`` is the division by delta in
    ``f^-1``), so the two arms have different slopes and the peak sits at
    ``t* = sT/(2s+m)``, not at ``T/2``:

        d1(H_t) = k + min((s+m)*t, s*(T-t))
        d2(H_t) = k + max((s+m)*t, s*(T-t))

    For ``m > 0`` the symmetric tent is a valid but *not tight* lower bound.
    """
    s, m = hen.s, hen.m
    lo = [k + min((s + m) * t, s * (T - t)) for t in range(T + 1)]
    hi = [k + max((s + m) * t, s * (T - t)) for t in range(T + 1)]
    return lo, hi


def symmetric_tent(hen, k, T):
    """The tent exactly as THEOREM.md §3 writes it (tight only when m = 0)."""
    return [k + hen.s * min(t, T - t) for t in range(T + 1)]


def idealised_C(p, alpha, delta, itinerary, T, k):
    """Setting A: pure Jacobian products, exactly as THEOREM.md §2 defines them."""
    hen, truth, _ = build(p, alpha, delta, itinerary, T, k)
    H0 = Lattice.ball(p, k)
    fwd, bwd = [H0], [H0]
    for t in range(T):
        fwd.append(fwd[-1].image(hen.jacobian(truth[t])))
    for t in range(T - 1, -1, -1):
        bwd.append(bwd[-1].image(hen.jacobian_inv(truth[t + 1])))
    bwd = bwd[::-1]
    sm = [fwd[t].intersect(bwd[t]) for t in range(T + 1)]
    lo, hi = tent(hen, k, T)
    # the sharp law is an *equality*, in both divisors and in both passes
    assert [x.d1 for x in sm] == lo and [x.d2 for x in sm] == hi, "sharp law broken"
    assert [x.d1 for x in fwd] == [k - hen.s * t for t in range(T + 1)]
    assert [x.d1 for x in bwd] == [k - (hen.s + hen.m) * (T - t) for t in range(T + 1)]
    C = max(lo[t] - sm[t].d1 for t in range(T + 1))
    defects = [d for d in (transversality_defect(fwd[t], bwd[t])
                           for t in range(T + 1)) if d is not None]
    return hen, C, max(defects, default=None), [s_.d1 for s_ in sm], lo


def certified_C(p, alpha, delta, itinerary, T, k, expect_exact=False):
    """Setting B: the full tracker -- inflation on, certified at every step.

    ``expect_exact`` asserts that *neither* pass inflated.  It must be set
    wherever the budget is claimed to be honoured: ``C == 0`` alone does not
    detect an under-provisioned run, because inflation at the far end of a pass
    lands where the *other* arm of the tent is binding and so does not move
    ``d1`` (docs/DEVELOPMENT.md -- this is exactly how the stale ``(3s+m)T`` budget
    survived).
    """
    hen, truth, tp = build(p, alpha, delta, itinerary, T, k)
    H0 = Lattice.ball(p, k)
    rng = random.Random(SEED)
    fwd = run_forward(hen, perturb(truth[0], H0, rng), H0, truth, T, tp)
    bwd = run_backward(hen, perturb(truth[T], H0, rng), H0, truth, T, tp)
    if expect_exact:
        assert fwd.first_inflation is None and bwd.first_inflation is None, (
            f"budget k={k} not honoured at p={p}, s={hen.s}, m={hen.m}, T={T}: "
            f"fwd inflates at {fwd.first_inflation}, bwd at {bwd.first_inflation}")
    sm = run_smoother(fwd, bwd, truth, tp)
    lo, _ = tent(hen, k, T)
    C = max(lo[t] - sm.d1()[i] for i, t in enumerate(sm.times))
    defects = [d for d in (transversality_defect(fwd.at(t)[1], bwd.at(t)[1])
                           for t in sm.times) if d is not None]
    return (hen, C, max(defects, default=None), sm.d1(), lo,
            fwd.first_inflation, bwd.first_inflation)


def main() -> None:
    style()
    import matplotlib.pyplot as plt

    out = {"seed": SEED, "A_idealised": [], "B_certified_scaled_k": [],
           "C_certified_fixed_k": []}

    # ---- A: idealised lattices, no hypothesis at all ---------------------
    for T, k in [(8, 0), (24, 0), (56, 2), (100, 0), (24, 40), (100, 40)]:
        hen, C, defect, _, _ = idealised_C(3, F(1, 3), 1, "++-+--+-", T, k)
        assert C == 0 and defect == 0, f"idealised C={C}, defect={defect}"
        out["A_idealised"].append({"p": 3, "s": hen.s, "m": hen.m, "T": T, "k": k,
                                   "C": C, "max_defect": defect,
                                   "itinerary": "++-+--+-"})
    for it in ITINERARIES:
        hen, C, defect, _, _ = idealised_C(3, F(1, 3), 1, it, 24, 0)
        assert C == 0 and defect == 0
        out["A_idealised"].append({"p": 3, "s": hen.s, "m": hen.m, "T": 24, "k": 0,
                                   "C": C, "max_defect": defect, "itinerary": it})
    for p, alpha, delta in REGIMES:
        hen, C, defect, _, _ = idealised_C(p, alpha, delta, "++-+--+-", 16, 0)
        assert C == 0 and defect == 0
        out["A_idealised"].append({"p": p, "s": hen.s, "m": hen.m, "T": 16, "k": 0,
                                   "C": C, "max_defect": defect,
                                   "itinerary": "++-+--+-"})

    # ---- B: certified, with the precision budget honoured ----------------
    for T in [4, 8, 16, 24, 32, 48, 64]:
        hen, C, defect, _, _, hz_f, hz_b = certified_C(
            3, F(1, 3), 1, "++-+--+-", T, budget(Henon.from_c(3, F(5, 9), 1), T),
            expect_exact=True)
        assert C == 0, f"certified C={C} at T={T} with the budget honoured"
        out["B_certified_scaled_k"].append(
            {"T": T, "k": budget(hen, T), "C": C, "max_defect": defect,
             "exactness_horizon": hz_f, "backward_exactness_horizon": hz_b})
    for p, alpha, delta in REGIMES:
        h0 = Henon.from_c(p, fixed_point_c(p, delta, alpha), delta)
        hen, C, defect, _, _, hz_f, hz_b = certified_C(
            p, alpha, delta, "++-+--+-", 16, budget(h0, 16), expect_exact=True)
        assert C == 0
        out["B_certified_scaled_k"].append(
            {"p": p, "s": hen.s, "m": hen.m, "T": 16, "k": budget(h0, 16),
             "C": C, "max_defect": defect, "exactness_horizon": hz_f,
             "backward_exactness_horizon": hz_b})

    # ---- C: certified at FIXED k -- the literal claim ---------------------
    fixed = {}
    for k in [40, 60]:
        row = []
        for T in [8, 12, 16, 20, 24, 32, 40, 56]:
            hen, C, defect, d1, tent, hz_f, hz_b = certified_C(
                3, F(1, 3), 1, "++-+--+-", T, k)
            # NOTE.md Thm B' is an *iff*: both passes are exact for the whole
            # window exactly when k >= (3s+2m)(T-1).  Assert both directions.
            sharp = (3 * hen.s + 2 * hen.m) * (T - 1)
            assert (hz_f is None and hz_b is None) == (k >= sharp), (
                f"Thm B' iff broken at k={k}, T={T} (sharp {sharp}): "
                f"fwd {hz_f}, bwd {hz_b}")
            row.append({"T": T, "C": C, "max_defect": defect,
                        "exactness_horizon": hz_f,
                        "backward_exactness_horizon": hz_b,
                        "sharp_budget": sharp})
            out["C_certified_fixed_k"].append({"k": k, **row[-1]})
        fixed[k] = row
    # C is NOT uniform in T at fixed k -- that is the finding, so assert it
    assert max(r["C"] for r in fixed[40]) > 0, \
        "expected C to grow once T outruns the exactness horizon"
    # ...and the cause is not loss of transversality
    assert all(r["max_defect"] == 0 for r in out["C_certified_fixed_k"]), \
        "transversality must hold even where C > 0"

    # ---- D: the tent is SKEWED unless delta is a unit ---------------------
    # THEOREM.md §3 writes k + s*min(t, T-t); that is the m = 0 case.  Record
    # where the symmetric form under-states the truth.
    out["D_skew"] = []
    for p_, alpha, delta in REGIMES:
        T, k = 12, 7
        hen, C, defect, d1, lo = idealised_C(p_, alpha, delta, "++-+--+-", T, k)
        sym = symmetric_tent(hen, k, T)
        gap = [a - b for a, b in zip(d1, sym)]
        s_, m_ = hen.s, hen.m
        peak = [t for t in range(T + 1) if d1[t] == max(d1)]
        out["D_skew"].append({
            "p": p_, "s": s_, "m": m_, "T": T, "k": k,
            "sharp_equals_symmetric": gap == [0] * (T + 1),
            "max_understatement": max(gap),
            "peak_t": peak, "predicted_peak": s_ * T / (2 * s_ + m_),
        })
        assert min(gap) >= 0, "the symmetric tent must remain a lower bound"
        assert (m_ == 0) == (gap == [0] * (T + 1)), \
            "the symmetric tent is tight exactly when m = 0"

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")

    ax = axes[0]
    for i, (k, row) in enumerate(fixed.items()):
        colour = EXTRA[i]
        ax.plot([r["T"] for r in row], [r["C"] for r in row], color=colour,
                marker="o", label=f"certified, fixed $k$={k}")
        h = row[0]["exactness_horizon"] or next(
            (r["exactness_horizon"] for r in row if r["exactness_horizon"]), None)
        if h:
            ax.axvline(2 * h, color=colour, ls=":", lw=1, alpha=0.6)
    ts = [r["T"] for r in out["B_certified_scaled_k"][:7]]
    ax.plot(ts, [0] * len(ts), color=EXTRA[2], marker="s",
            label="certified, $k \\geq (3s{+}2m)(T{-}1)$   ($C$=0)")
    ax.plot(ts, [0] * len(ts), color=EXTRA[3], marker="^", ls="--",
            label="idealised, any $k$   ($C$=0, coincides above)")
    ax.set_xlabel("window length $T$")
    ax.set_ylabel("observed constant $C$")
    ax.set_title("$C$ is uniform in $T$ only if the precision\n"
                 "budget grows with $T$", loc="left", fontsize=11)
    ax.legend(loc="upper left", fontsize=8, labelcolor=INK_MUTED)

    ax = axes[1]
    for i, T in enumerate([24, 56]):
        hen, C, defect, d1, tent, hz_f, hz_b = certified_C(
            3, F(1, 3), 1, "++-+--+-", T, 40)
        colour = EXTRA[i]
        ax.plot(range(len(d1)), d1, color=colour, marker="o", markevery=8,
                label=f"certified $d_1(H_t)$, $T$={T}  ($C$={C})")
        ax.plot(range(len(tent)), tent, color=colour, ls=":", lw=1.2, alpha=0.7)
    ax.axhline(40, color=INK_MUTED, lw=0.8, ls="--")
    ax.annotate("$k$=40", xy=(1, 40), xytext=(0, 4), textcoords="offset points",
                fontsize=8, color=INK_MUTED)
    ax.set_xlabel("iteration $t$")
    ax.set_ylabel("worst-direction digits  $d_1(H_t)$")
    ax.set_title("Where the tent sags: the budget runs out,\n"
                 "not the transversality (dotted: ideal tent)", loc="left",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=8, labelcolor=INK_MUTED)

    fig.suptitle("THEOREM.md §3: measuring the constant $C$",
                 x=0.005, ha="left", fontsize=12.5)
    path = save(fig, "exp_theorem")

    out["sharp_law"] = (
        "forward  d1 = k - s*t,             d2 = k + (s+m)*t; "
        "backward d1 = k - (s+m)*(T-t),     d2 = k + s*(T-t); "
        "smoothed d1 = k + min((s+m)t, s(T-t)), d2 = k + max((s+m)t, s(T-t)). "
        "Asserted as an equality in every configuration of block A.  THEOREM.md "
        "§3's symmetric tent k + s*min(t, T-t) is the m = 0 case; for m > 0 it "
        "is a valid but not tight lower bound, and the peak sits at "
        "t* = sT/(2s+m), not T/2.")
    out["verdict"] = (
        "The smoothing mechanism is confirmed and is exact: for the idealised "
        "lattices of THEOREM.md §2 (pure Jacobian products) C = 0 with no "
        "hypothesis -- verified for T up to 100, k down to 0, periods 1-19, "
        "p in {3,5,7,11}, s in {1,2,3}, m in {0,1,2}, with transversality "
        "defect 0 throughout.  For *certified* lattices C = 0 once the starting "
        "precision honours the sharp budget k >= (3s+2m)(T-1) of NOTE.md Thm B' "
        "(an iff for exactness of both passes, asserted in both directions in "
        "block C); at fixed k, C grows linearly once T "
        "outruns the exactness horizon (k=40: C = 0,0,3,14,24,46 for "
        "T = 16,20,24,32,40,56).  So the theorem as literally stated -- C "
        "independent of T at fixed k -- is false, and the repair is to add that "
        "hypothesis, after which the conclusion strengthens from an inequality "
        "to equality with C = 0.  The failure mode is the quadratic-remainder "
        "budget, NOT the stable/unstable transversality: the "
        "defect is 0 in every configuration measured, including all those with "
        "C > 0.")
    write_json("exp_theorem", out)
    report("THEOREM.md §3: the constant C", [
        f"A. idealised lattices: C=0 in all {len(out['A_idealised'])} configurations "
        f"(T <= 100, k >= 0, periods 1-19, p <= 11, s <= 3, m <= 2)",
        f"B. certified with k >= (3s+2m)(T-1): C=0 and neither pass inflates, "
        f"in all {len(out['B_certified_scaled_k'])} configurations, T up to 64",
        "C. certified at fixed k: C grows once T passes the horizon --",
        *[f"     k={k}: " + ", ".join(f"T={r['T']}:C={r['C']}" for r in row)
          for k, row in fixed.items()],
        "transversality defect 0 everywhere, including where C > 0",
        "D. the tent is skewed when m > 0 -- " + ", ".join(
            f"p={d['p']},s={d['s']},m={d['m']}: peak t={d['peak_t'][0]}"
            f" (pred {d['predicted_peak']:.1f})" for d in out["D_skew"] if d["m"]),
        f"figure: {path}"])


if __name__ == "__main__":
    main()
