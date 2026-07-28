"""The two headline figures, run in H_III (the horseshoe).

Plot 1 -- valid digits vs iteration, four tracks:

    truth     exact iteration at full working precision (the reference line)
    naive     one scalar absolute-precision counter (capped precision)
    lattice   CRV prediction only
    filtered  CRV prediction + update (the backward pass, §2.5b)

Expected, and asserted: naive and forward-only lattice both fall linearly at
``s`` digits per step, while the smoothed track stays bounded on the tent
``k + min(t, T-t) s``.

Plot 2 -- the elementary divisor exponents d1, d2 separating linearly, with
``d1 + d2`` rising by exactly ``m`` per step.  That last one is ``det J =
delta``: an assertion, not an observation.  This is the slide-worthy plot --
p-adic hyperbolicity rendered as covariance elongation.
"""

from __future__ import annotations

import random

from _common import INK_MUTED, SERIES, label_end, report, save, style, write_json  # noqa: I001
from padic_filtering.henon import periodic_orbit, truth_orbit, truth_precision
from padic_filtering.lattice import Lattice
from padic_filtering.params import DEFAULT_ITINERARY, HORSESHOE_3ADIC
from padic_filtering.precision import (naive_track, perturb, run_backward,
                                       run_forward, run_smoother)

SEED = 20260726
# K is chosen so the exactness horizon ~ (K - s - m)/(3s + m) exceeds T: per
# experiment 5.2, the plots need a parameter region with runway.  With K = 40 the
# horizon is 15 < T and both passes have inflated by the time they reach the far
# end of the window, so the tent sags at t = 1, 2 and 22, 23 -- the claim would
# then be about the precision budget, not about the smoother.
T, K, PREC = 24, 80, 900


def build():
    hen = HORSESHOE_3ADIC.henon()
    hen.check_regime()
    X = periodic_orbit(hen, DEFAULT_ITINERARY, PREC)
    truth = truth_orbit(hen, X, T, PREC)
    tp = truth_precision(hen, PREC)
    H0 = Lattice.ball(hen.p, K)
    rng = random.Random(SEED)
    fwd = run_forward(hen, perturb(truth[0], H0, rng), H0, truth, T, tp)
    bwd = run_backward(hen, perturb(truth[T], H0, rng), H0, truth, T, tp)
    sm = run_smoother(fwd, bwd, truth, tp)
    naive = naive_track(hen, K, T)
    return hen, truth, fwd, bwd, sm, naive


def main() -> None:
    style()
    import matplotlib.pyplot as plt

    hen, truth, fwd, bwd, sm, naive = build()
    s, m = hen.s, hen.m
    tent = [K + min(t, T - t) * s for t in range(T + 1)]

    # ---- the claims, as assertions ---------------------------------------
    assert naive == [K - t * s for t in range(T + 1)], "naive must fall at s/step"
    assert fwd.d1() == [K - t * s for t in range(T + 1)], "forward must fall at s/step"
    assert sm.d1() == tent, "smoothed must be the bounded tent"
    assert min(sm.d1()) == K and max(fwd.d1()) - min(fwd.d1()) == T * s
    # det J = delta: v_p(det H) rises by exactly m per step
    assert [r.lattice_digits for r in fwd.records] == [2 * K + t * m for t in fwd.times]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")

    # ---- Plot 1 ----------------------------------------------------------
    ax = axes[0]
    # The truth track is exact at the working precision -- three orders of
    # magnitude above everything else, so it is stated rather than plotted.
    ax.text(0.02, 0.965, f"truth: exact to {PREC - hen.s} digits (off scale)",
            transform=ax.transAxes, fontsize=8, color=SERIES["truth"], va="top")
    # naive and the forward lattice's worst direction coincide *exactly*: in the
    # horseshoe the lattice buys nothing in its weakest direction, and its whole
    # advantage over the scalar counter is the anisotropy (Plot 2).  Drawn as a
    # thick underlay plus a dashed overlay so both are legible.
    ax.plot(range(len(naive)), naive, color=SERIES["naive"], lw=3.4, alpha=0.55,
            solid_capstyle="round", label="naive (scalar counter)")
    ax.plot(fwd.times, fwd.d1(), color=SERIES["lattice"], lw=1.6, ls="--",
            marker="o", markevery=6, label="lattice $d_1$ (coincides with naive)")
    ax.plot(sm.times, sm.d1(), color=SERIES["filtered"], lw=1.9, marker="o",
            markevery=6, label="filtered $d_1$ (smoother)")
    label_end(ax, fwd.times, fwd.d1(), "naive = lattice $d_1$", SERIES["lattice"])
    label_end(ax, sm.times, sm.d1(), "filtered", SERIES["filtered"])
    ax.set_ylim(min(naive) - 4, K + T * s // 2 + 8)
    ax.set_xlim(right=T * 1.4)
    ax.set_xlabel("iteration $t$")
    ax.set_ylabel("guaranteed digits in every direction")
    ax.set_title("Plot 1  Forward-only decays linearly; smoothing stays bounded",
                 loc="left", fontsize=11)
    ax.legend(loc="lower left", fontsize=8, labelcolor=INK_MUTED)

    # ---- Plot 2 ----------------------------------------------------------
    ax = axes[1]
    ax.plot(fwd.times, fwd.d1(), color=SERIES["lattice"], marker="o", markevery=6,
            label="$d_1$  (unstable direction, slope $-s$)")
    ax.plot(fwd.times, fwd.d2(), color=SERIES["filtered"], marker="s", markevery=6,
            label="$d_2$  (stable direction, slope $s+m$)")
    ax.fill_between(fwd.times, fwd.d1(), fwd.d2(), color=SERIES["lattice"], alpha=0.07)
    label_end(ax, fwd.times, fwd.d1(), "$d_1$", SERIES["lattice"])
    label_end(ax, fwd.times, fwd.d2(), "$d_2$", SERIES["filtered"])
    mid = len(fwd.times) // 2
    ax.annotate("anisotropy $d_2-d_1$\ngrows at $2s+m$",
                xy=(fwd.times[mid], (fwd.d1()[mid] + fwd.d2()[mid]) / 2),
                xytext=(-6, 0), textcoords="offset points", fontsize=8,
                color=INK_MUTED, ha="right", va="center")
    ax.set_xlabel("iteration $t$")
    ax.set_ylabel("elementary divisor exponent")
    ax.set_title("Plot 2  p-adic hyperbolicity as covariance elongation",
                 loc="left", fontsize=11)
    ax.set_xlim(right=T * 1.3)
    ax.legend(loc="upper left", fontsize=8, labelcolor=INK_MUTED)

    fig.suptitle(f"H$_{{III}}$ horseshoe: $p$={hen.p}, $s$={s}, $m$={m}, "
                 f"period-{len(DEFAULT_ITINERARY)} orbit, $k$={K}, $T$={T}",
                 x=0.005, ha="left", fontsize=12.5)
    path = save(fig, "headline")

    out = {
        "naive_equals_lattice_d1": naive == fwd.d1(),
        "seed": SEED, "T": T, "k": K, "prec": PREC, "p": hen.p, "s": s, "m": m,
        "itinerary": DEFAULT_ITINERARY, "region": hen.region,
        "naive": naive, "lattice_d1": fwd.d1(), "lattice_d2": fwd.d2(),
        "filtered_d1": sm.d1(), "filtered_d2": sm.d2(), "tent": tent,
        "k_choice": ("chosen so the exactness horizon exceeds T (see exp_5_2); "
                     "with k=40 the horizon is 15 < T=24 and the tent sags at "
                     "the window edges"),
        "forward_exactness_horizon": fwd.first_inflation,
        "filtered_exactness_horizon": sm.first_inflation,
        "digits_lost_forward": fwd.d1()[0] - fwd.d1()[-1],
        "digits_lost_smoothed": sm.d1()[0] - min(sm.d1()),
        "verdict": (
            f"Forward-only and naive both lose {s} digits per step "
            f"({fwd.d1()[0]} -> {fwd.d1()[-1]} over {T} steps).  The smoother is "
            f"bounded below by k={K} and peaks at {max(sm.d1())} in the middle of "
            f"the window: conditioning on the endpoint recovers the direction the "
            f"forward pass lost."),
    }
    write_json("headline", out)
    report("headline (§3 Plots 1 and 2)", [
        f"naive:     {naive[0]} -> {naive[-1]}",
        f"lattice:   {fwd.d1()[0]} -> {fwd.d1()[-1]} (forward only)",
        f"filtered:  min {min(sm.d1())}, max {max(sm.d1())} (tent)",
        f"exactness horizon: forward t={fwd.first_inflation}, "
        f"filtered t={sm.first_inflation}",
        f"figure: {path}"])


if __name__ == "__main__":
    main()
