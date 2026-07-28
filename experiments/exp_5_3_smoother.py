"""5.3 -- Does the backward pass recover the lost direction?

The core claim, regime-dependent and settled during the review:

  H+_II   NO.  The forward pass degrades in no direction (nonexpanding), the
          backward pass only loses, and F ∩ B = F identically.  Asserted here.
  H_III   YES.  Forward-only loses s digits/step; the smoothed worst-direction
          precision is the bounded tent k + min(t, T-t)*s.

The remaining work named in §5.3 is done here: the tent is extended from the
fixed point to genuine periodic orbits (periods 5-20 via Newton from symbolic
itineraries), confirming it survives a *variable* Jacobian along the orbit, and
the two passes are compared by their SNF *bases* -- not just their divisors --
via the transversality defect, which is the computational form of the ADP
Lemma 23-24 stable/unstable tube transversality.
"""

from __future__ import annotations

import random
from fractions import Fraction as F

from _common import EXTRA, INK_MUTED, report, save, style, write_json  # noqa: I001
from padic_filtering.henon import periodic_orbit, truth_orbit, truth_precision
from padic_filtering.lattice import Lattice, transversality_defect
from padic_filtering.params import ATTRACTOR_3ADIC, HORSESHOE_3ADIC
from padic_filtering.precision import (perturb, run_backward, run_forward,
                                       run_smoother)

SEED = 20260726
ATTRACTOR_PREC = 400
ITINERARIES = ["+", "+-", "++-+-", "++-+--+-", "++-+--+-+-++-",
               "++-+--+-+-++--+-+--+"]


def run_case(hen, truth, T, k, tp=None):
    H0 = Lattice.ball(hen.p, k)
    rng = random.Random(SEED)
    fwd = run_forward(hen, perturb(truth[0], H0, rng), H0, truth, T, tp or float("inf"))
    bwd = run_backward(hen, perturb(truth[T], H0, rng), H0, truth, T, tp or float("inf"))
    sm = run_smoother(fwd, bwd, truth, tp or float("inf"))
    return fwd, bwd, sm


def periodic_case(itinerary, T=24, k=80, prec=800, ps=HORSESHOE_3ADIC):
    hen = ps.henon()
    hen.check_regime()
    X = periodic_orbit(hen, itinerary, prec)
    truth = truth_orbit(hen, X, T, prec)
    fwd, bwd, sm = run_case(hen, truth, T, k, truth_precision(hen, prec))
    tent = [k + min(t, T - t) * hen.s for t in range(T + 1)]
    assert fwd.d1() == [k - t * hen.s for t in range(T + 1)], "forward must lose s/step"
    assert sm.d1() == tent, f"tent broken for {itinerary}: {sm.d1()} != {tent}"
    # None at t = 0 and t = T, where one pass is still the isotropic initial
    # ball and "worst direction" is not defined.
    defects = [transversality_defect(fwd.at(t)[1], bwd.at(t)[1]) for t in range(T + 1)]
    return hen, fwd, bwd, sm, tent, defects


def main() -> None:
    style()
    import matplotlib.pyplot as plt

    out = {"seed": SEED, "periodic": [], "itineraries": ITINERARIES}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), layout="constrained")

    # ---- panel 1: the tent, on a genuine period-8 orbit -------------------
    T, k = 24, 80
    hen, fwd, bwd, sm, tent, defects = periodic_case("++-+--+-", T=T, k=k)
    ax = axes[0]
    for name, ys, colour in [("forward only", fwd.d1(), EXTRA[0]),
                             ("backward only", bwd.d1(), EXTRA[1]),
                             ("smoothed", sm.d1(), EXTRA[2])]:
        ax.plot(range(T + 1), ys, color=colour, marker="o", markevery=6, label=name)
    ax.axhline(k, color=INK_MUTED, lw=0.8, ls=":")
    ax.annotate(f"$k$={k}", xy=(0.5, k), xytext=(0, 5), textcoords="offset points",
                fontsize=8, color=INK_MUTED)
    ax.set_title("Period-8 orbit: smoothing is a bounded tent",
                 loc="left", fontsize=11)
    ax.set_xlabel("iteration $t$")
    ax.set_ylabel("worst-direction digits  $d_1$")
    ax.set_xlim(-0.6, T + 0.6)
    ax.legend(loc="lower left", fontsize=8, labelcolor=INK_MUTED)

    # ---- panel 2: the tent survives every period tested ------------------
    ax = axes[1]
    for i, it in enumerate(ITINERARIES):
        h, f_, b_, s_, tent_, defects_ = periodic_case(it)
        colour = EXTRA[i % len(EXTRA)]
        ax.plot(range(len(s_.d1())), s_.d1(), color=colour, lw=1.4, alpha=0.85)
        out["periodic"].append({
            "itinerary": it, "period": len(it), "p": h.p, "s": h.s,
            "smoothed_d1": s_.d1(), "forward_d1": f_.d1(), "tent": tent_,
            "tent_holds": s_.d1() == tent_,
            "transversality_defect": defects_,
            "max_defect": max([d for d in defects_ if d is not None], default=None),
            "defect_undefined_at": [t for t, d in enumerate(defects_) if d is None],
        })
    ax.plot(range(len(tent)), tent, color=INK_MUTED, ls="--", lw=1.2,
            label="predicted tent $k + \\min(t, T-t)s$")
    ax.annotate(f"{len(ITINERARIES)} orbits, periods "
                f"{min(len(i) for i in ITINERARIES)}-{max(len(i) for i in ITINERARIES)},\n"
                "exactly superimposed", xy=(0.5, 0.72), xycoords="axes fraction",
                fontsize=8, color=INK_MUTED, ha="center")
    ax.set_title("Every period lands on the same tent",
                 loc="left", fontsize=11)
    ax.set_xlabel("iteration $t$")
    ax.set_ylabel("smoothed  $d_1$")
    ax.legend(loc="lower center", fontsize=8, labelcolor=INK_MUTED)

    # ---- panel 3: H+_II, where the smoother is vacuous -------------------
    ax = axes[2]
    hen2 = ATTRACTOR_3ADIC.henon()
    T2, k2 = 24, 40
    truth2 = hen2.orbit((F(1), F(1)), T2, prec=ATTRACTOR_PREC)
    fwd2, bwd2, sm2 = run_case(hen2, truth2, T2, k2)
    identical = [sm2.at(t)[1] == fwd2.at(t)[1] for t in range(T2 + 1)]
    assert all(identical), "H+_II: F ∩ B must equal F -- the smoother is vacuous"
    for name, ys, colour in [("forward only", fwd2.d1(), EXTRA[0]),
                             ("backward only", bwd2.d1(), EXTRA[1]),
                             ("smoothed (= forward)", sm2.d1(), EXTRA[2])]:
        ax.plot(range(T2 + 1), ys, color=colour, marker="o", markevery=6,
                label=name, lw=2.6 if "backward" in name else 1.6)

    ax.set_title("H$^+_{II}$ attractor: the smoother adds nothing",
                 loc="left", fontsize=11)
    ax.set_xlabel("iteration $t$")
    ax.set_ylabel("worst-direction digits  $d_1$")
    ax.set_xlim(-0.6, T2 + 0.6)
    ax.legend(loc="lower right", fontsize=8, labelcolor=INK_MUTED)

    out["attractor"] = {
        "smoother_is_vacuous": all(identical),
        "forward_d1": fwd2.d1(), "backward_d1": bwd2.d1(), "smoothed_d1": sm2.d1(),
    }
    out["transversality"] = {
        "max_defect_over_all_orbits": max(
            c["max_defect"] for c in out["periodic"] if c["max_defect"] is not None),
        "note": ("defect 0 means the forward and backward worst directions are "
                 "ultrametrically orthogonal, so the intersection gains exactly "
                 "what the tent predicts (ADP Lemmas 23-24).  The defect is "
                 "undefined where one pass is still the isotropic initial ball."),
    }
    assert out["transversality"]["max_defect_over_all_orbits"] == 0, \
        "the two passes must stay transverse"

    fig.suptitle("5.3  The backward pass recovers the lost direction "
                 "-- in H$_{III}$ only", x=0.005, ha="left", fontsize=12.5)
    path = save(fig, "exp_5_3_smoother")
    out["verdict"] = (
        "PASS in H_III: the smoothed worst-direction precision is the bounded tent "
        "k + min(t, T-t)*s on every periodic orbit tested (periods 1-20), the "
        "forward/backward directions stay exactly transverse (defect 0), and "
        "forward-only falls linearly.  VACUOUS in H+_II, as predicted: F ∩ B = F.")
    write_json("exp_5_3_smoother", out)
    report("5.3 smoother", [
        f"period {c['period']:>2} ({c['itinerary']}): tent holds={c['tent_holds']}, "
        f"max transversality defect={c['max_defect']}" for c in out["periodic"]] + [
        f"H+_II smoother vacuous: {out['attractor']['smoother_is_vacuous']}",
        f"forward-only d1 falls {fwd.d1()[0]} -> {fwd.d1()[-1]}; "
        f"smoothed stays >= {min(sm.d1())}",
        f"figure: {path}"])


if __name__ == "__main__":
    main()
