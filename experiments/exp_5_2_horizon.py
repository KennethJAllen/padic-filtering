"""5.2 -- Where is the exactness horizon?

The quadratic Taylor remainder eventually escapes the propagated lattice and
the prediction step has to inflate.  The first such iterate is the *horizon*,
and it is the quantity that decides whether the demo has any runway.

`NOTE.md` Prop 4.4 makes this exact, and as an **iff**, in both passes:

    forward step  t -> t+1  is exact  <=>  (3s + m)*t  <= k - m
    backward step j -> j+1  is exact  <=>  (3s + 2m)*j <= k

so the first inflating forward step is at time ``(k-m)//(3s+m) + 2`` and the
first inflating backward step is at backward index ``k//(3s+2m) + 2``, i.e. at
time ``T - (k//(3s+2m) + 2)``.  (``first_inflation`` records the *destination*
index, hence the ``+2``.)  Both are asserted here as **equalities**, not as
lower bounds, in all three of:

  1. forward horizons vs k, over five regimes covering s in {0,1,2} and
     m in {0,1,2};
  2. backward horizons vs k, over the same regimes -- the panel that makes the
     ``3s+m`` / ``3s+2m`` asymmetry visible, and the one whose absence let a
     stale budget survive in this repo (docs/DEVELOPMENT.md: the two passes are not
     mirror images);
  3. a threshold boundary sweep: over a fixed window, ``k = (3s+2m)(W-1)``
     leaves both passes exact and ``k - 1`` does not.  Prop 4.4 is an iff, so
     both directions are checked.

Both horizons are linear in the starting precision k, so the runway is tunable.
The updated track's horizon is reported separately: an update restores d1,
which also postpones the exactness failure.
"""

from __future__ import annotations

import random
from fractions import Fraction as F

from _common import EXTRA, INK_MUTED, report, save, style, write_json  # noqa: I001
from padic_filtering.henon import periodic_orbit, truth_orbit, truth_precision
from padic_filtering.lattice import Lattice
from padic_filtering.padic import INFINITY
from padic_filtering.params import (ATTRACTOR_3ADIC, DEFAULT_ITINERARY,
                                    HORSESHOE_3ADIC, HORSESHOE_3ADIC_M1,
                                    HORSESHOE_3ADIC_M2, HORSESHOE_3ADIC_S2,
                                    HORSESHOE_5ADIC)
from padic_filtering.precision import (perturb, run_backward, run_forward,
                                       run_oracle)

SEED = 20260726
# H+_II ground truth is exact modulo p^ATTRACTOR_PREC (the map is nonexpanding
# there, so the reduction loses nothing); ample room above any d2 reached.
ATTRACTOR_PREC = 400
KS = [6, 9, 12, 18, 24, 30, 36, 48, 60]

# Regimes for the horizon curves.  s in {0,1,2} and m in {0,1,2}: the m > 0
# sets are what separate the forward rate 3s+m from the two-sided rate 3s+2m.
HORIZON_REGIMES = [HORSESHOE_3ADIC, HORSESHOE_5ADIC, HORSESHOE_3ADIC_S2,
                   HORSESHOE_3ADIC_M1, HORSESHOE_3ADIC_M2, ATTRACTOR_3ADIC]
# Five distinct (s, m) curves, one more than `_common.EXTRA` carries.
CURVE_COLOURS = [*EXTRA, "#b8408f"]


def horizon_of(track) -> int | None:
    return track.first_inflation


# ---------------------------------------------------------- predictions

def forward_predicted(hen, k: int) -> int:
    """Time of the first inflating forward step (NOTE.md Prop 4.4), exactly."""
    return (k - hen.m) // (3 * hen.s + hen.m) + 2


def backward_predicted(hen, k: int) -> int:
    """Backward *index* j of the first inflating backward step, exactly."""
    return k // (3 * hen.s + 2 * hen.m) + 2


# ------------------------------------------------------------- truth

def truth_for(ps, T: int, k: int):
    """``(hen, truth, truth_precision)`` for a T-step window at precision k.

    The working precision must stay above the largest divisor any pass reaches,
    ``k + (s+m)*T``; ``_assert_certified`` raises rather than silently
    certifying against an exhausted ground truth.
    """
    hen = ps.henon()
    if hen.region == "H_III":
        hen.check_regime()
        prec = 2 * (k + T * (hen.s + hen.m)) + 300
        X = periodic_orbit(hen, DEFAULT_ITINERARY, prec)
        return hen, truth_orbit(hen, X, T, prec), truth_precision(hen, prec)
    # H+_II: nonexpanding, so reducing the orbit mod p^ATTRACTOR_PREC is lossless
    return hen, hen.orbit((F(1), F(1)), T, prec=ATTRACTOR_PREC), INFINITY


def horizons(ps, ks=KS):
    """Forward and backward horizons over ``ks``, asserted equal to Prop 4.4."""
    fwd_got, bwd_got, fwd_pred, bwd_pred = [], [], [], []
    hen = ps.henon()
    for k in ks:
        T = 4 * k + 20
        hen, truth, tp = truth_for(ps, T, k)
        H0 = Lattice.ball(hen.p, k)
        fwd = run_forward(hen, perturb(truth[0], H0, random.Random(SEED)), H0,
                          truth, T, tp)
        bwd = run_backward(hen, perturb(truth[T], H0, random.Random(SEED)), H0,
                           truth, T, tp)
        assert fwd.first_inflation is not None and bwd.first_inflation is not None, \
            f"{ps.name}: a pass never reached its horizon at k={k}"
        fwd_got.append(horizon_of(fwd))
        bwd_got.append(T - horizon_of(bwd))       # as a backward index j
        fwd_pred.append(forward_predicted(hen, k))
        bwd_pred.append(backward_predicted(hen, k))
    assert fwd_got == fwd_pred, \
        f"{ps.name}: forward horizon {fwd_got} != predicted {fwd_pred}"
    assert bwd_got == bwd_pred, \
        f"{ps.name}: backward horizon {bwd_got} != predicted {bwd_pred}"
    return hen, fwd_got, fwd_pred, bwd_got, bwd_pred


def threshold_sweep(ps, W: int = 6):
    """Prop 4.4 is an *iff*: k = (3s+2m)(W-1) passes and k-1 fails.

    Returns the record; asserts both directions.  ``k`` is the sharp two-sided
    budget of Theorem B', so at ``k`` neither pass inflates anywhere in the
    window, and at ``k-1`` at least one does.
    """
    hen = ps.henon()
    k = (3 * hen.s + 2 * hen.m) * (W - 1)
    hen, truth, tp = truth_for(ps, W, k)

    def inflations(kk):
        H0 = Lattice.ball(hen.p, kk)
        fwd = run_forward(hen, perturb(truth[0], H0, random.Random(SEED)), H0,
                          truth, W, tp)
        bwd = run_backward(hen, perturb(truth[W], H0, random.Random(SEED)), H0,
                           truth, W, tp)
        return fwd.first_inflation, bwd.first_inflation

    at_k = inflations(k)
    below = inflations(k - 1)
    assert at_k == (None, None), \
        f"{ps.name}: k={k} is the sharp budget but a pass inflated: {at_k}"
    assert below != (None, None), \
        f"{ps.name}: k={k - 1} is below the sharp budget but nothing inflated"
    # and the pass that fails first is the *backward* one whenever m > 0, since
    # (3s+2m)(W-1) - [(3s+m)(W-1) + m] = m(W-2) >= 0  (NOTE.md §4.4 Step 2)
    if hen.m > 0:
        assert below[1] is not None, \
            f"{ps.name}: for m>0 the backward pass must be the binding one"
    return {"name": ps.name, "region": hen.region, "p": hen.p, "s": hen.s,
            "m": hen.m, "W": W, "k_threshold": k,
            "at_threshold": {"fwd": at_k[0], "bwd": at_k[1]},
            "below_threshold": {"fwd": below[0], "bwd": below[1]}}


def updated_horizon(ps, k=12, lag=3, T=60, prec=600):
    """Horizon of a track that receives an update every ``lag`` steps.

    Restoring d1 also restores the exactness margin, so the horizon moves out.
    Uses the oracle update (§2.5a) because it is an *online* measurement; the
    offline smoother has the same effect on d1 but needs the whole orbit.
    """
    hen = ps.henon()
    X = periodic_orbit(hen, DEFAULT_ITINERARY, prec)
    truth = truth_orbit(hen, X, T, prec)
    tp = truth_precision(hen, prec)
    H0 = Lattice.ball(hen.p, k)
    v0 = perturb(truth[0], H0, random.Random(SEED))
    plain = run_forward(hen, v0, H0, truth, T, tp)
    updated = run_oracle(hen, v0, H0, truth, T, reveal_k=k,
                         reveal_times=range(lag, T + 1, lag), truth_precision=tp)
    return hen, plain, updated


def main() -> None:
    style()
    import matplotlib.pyplot as plt

    out = {"seed": SEED, "ks": KS, "cases": [], "threshold_sweep": []}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")

    # Horizons depend on (s, m), not on p: identical curves would overlap and
    # hide each other, so group them and name every prime that produces them.
    curves = {}
    for ps in HORIZON_REGIMES:
        hen, fwd_got, fwd_pred, bwd_got, bwd_pred = horizons(ps)
        out["cases"].append({
            "name": ps.name, "region": hen.region, "p": hen.p, "s": hen.s,
            "m": hen.m, "horizon": fwd_got, "predicted": fwd_pred,
            "backward_horizon_j": bwd_got, "backward_predicted": bwd_pred,
            "slope_observed": (fwd_got[-1] - fwd_got[0]) / (KS[-1] - KS[0]),
            "slope_predicted": 1 / (3 * hen.s + hen.m),
            "backward_slope_observed":
                (bwd_got[-1] - bwd_got[0]) / (KS[-1] - KS[0]),
            "backward_slope_predicted": 1 / (3 * hen.s + 2 * hen.m),
        })
        curve = curves.setdefault(
            (tuple(fwd_got), tuple(bwd_got)),
            {"fwd": fwd_got, "bwd": bwd_got, "primes": [],
             "region": hen.region, "s": hen.s, "m": hen.m})
        curve["primes"].append(hen.p)

    for panel, key, rate, title in [
            (0, "fwd", "3s{+}m", "Forward horizon: slope $1/(3s{+}m)$"),
            (1, "bwd", "3s{+}2m", "Backward horizon: slope $1/(3s{+}2m)$")]:
        ax = axes[panel]
        for i, curve in enumerate(curves.values()):
            colour = CURVE_COLOURS[i % len(CURVE_COLOURS)]
            primes = ", ".join(str(q) for q in curve["primes"])
            name = (f"{curve['region']}  $p$={primes}, "
                    f"$s$={curve['s']}, $m$={curve['m']}")
            ax.plot(KS, curve[key], color=colour, marker="o", label=name)
        ax.set_xlabel("starting precision $k$")
        ax.set_ylabel("first inflating step" +
                      ("  (time $t$)" if key == "fwd" else "  (index $j=T{-}t$)"))
        ax.set_title(f"{title}\n(asserted as an equality, NOTE.md Prop 4.4)",
                     loc="left", fontsize=11)
        ax.legend(loc="upper left", fontsize=8, labelcolor=INK_MUTED)

    ax = axes[2]
    hen, plain, updated = updated_horizon(HORSESHOE_3ADIC)
    ax.plot(plain.times, plain.d1(), color=EXTRA[0], marker="o", markevery=8,
            label="predict only")
    ax.plot(updated.times, updated.d1(), color=EXTRA[2], marker="o", markevery=8,
            label="predict + update every 3 steps")
    for track, colour in [(plain, EXTRA[0]), (updated, EXTRA[2])]:
        h = track.first_inflation
        if h is not None:
            ax.axvline(h, color=colour, ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("iteration $t$")
    ax.set_ylabel("worst-direction digits  $d_1$")
    ax.set_title("Updates postpone the horizon\n(dashed: first inflation)",
                 loc="left", fontsize=11)
    ax.set_xlim(-1, max(plain.times[-1], updated.times[-1]) + 1)
    ax.legend(loc="lower right", fontsize=8, labelcolor=INK_MUTED)
    out["updated"] = {
        "k": 12, "lag": 3,
        "predict_only_horizon": plain.first_inflation,
        "updated_horizon": updated.first_inflation,
        "predict_only_exhausted_at": plain.exhausted_at,
        "updated_exhausted_at": updated.exhausted_at,
    }
    assert (updated.first_inflation or 10**9) > (plain.first_inflation or 0), \
        "an update must postpone the exactness failure"

    # ---- the threshold is an iff: k passes, k-1 fails --------------------
    for ps in HORIZON_REGIMES:
        out["threshold_sweep"].append(threshold_sweep(ps))

    fig.suptitle("5.2  The exactness horizon, both passes, and how updates "
                 "move it out", x=0.005, ha="left", fontsize=12.5)
    path = save(fig, "exp_5_2_horizon")
    out["verdict"] = (
        "PASS, as exact equalities.  Both horizons are exactly the NOTE.md "
        "Prop 4.4 values -- forward (k-m)//(3s+m) + 2, backward "
        "k//(3s+2m) + 2 -- in every regime measured (s in {0,1,2}, m in "
        "{0,1,2}, p in {3,5}), so both grow linearly in the starting precision "
        "k and the runway is tunable.  The two rates differ whenever m > 0: the "
        "backward pass is the binding one, which is why a budget computed from "
        "the forward rate 3s+m under-provisions it.  Prop 4.4's iff is checked "
        "in both directions: at k = (3s+2m)(W-1) neither pass inflates over the "
        "window and at k-1 one does.  Periodic updates push the forward horizon "
        "out further.")
    write_json("exp_5_2_horizon", out)
    report("5.2 exactness horizon", [
        f"{c['name']}: fwd {c['horizon']} (slope {c['slope_observed']:.2f} = "
        f"1/(3s+m) = {c['slope_predicted']:.2f}), bwd {c['backward_horizon_j']} "
        f"(slope {c['backward_slope_observed']:.2f} = 1/(3s+2m) = "
        f"{c['backward_slope_predicted']:.2f}) -- both exact"
        for c in out["cases"]] + [
        "threshold iff: " + ", ".join(
            f"{r['name']}(k={r['k_threshold']})" for r in out["threshold_sweep"]),
        f"predict-only horizon {out['updated']['predict_only_horizon']} -> "
        f"updated {out['updated']['updated_horizon']}",
        f"figure: {path}"])


if __name__ == "__main__":
    main()
