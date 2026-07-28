"""docs/THEOREM.md §5 item 3: get off (short) periodic orbits.

Every orbit tested so far was built from a short symbolic itinerary (period
1-20), so a referee can object that the exact tent law might be an artefact of
periodicity.  The cheap attack: take a *random* itinerary of length L >> T
(L = 200 here), build the orbit by the same Newton solve, and run the smoother
on a window of T steps taken from the middle.  The orbit is still technically
periodic, but the Jacobian sequence seen by the window carries no period
(the full itinerary's minimal cyclic period is asserted to be L itself, and
the window is shorter than L/4), so this tests exactly what matters -- the
lattice identity along a Jacobian sequence indistinguishable from aperiodic --
without any new machinery.

Asserted, per window:
  A. idealised lattices (pure Jacobian products): the sharp law of THEOREM.md
     §3 holds as an *equality* in all four divisors, so C = 0; transversality
     defect 0 wherever defined.
  B. certified lattices with the sharp (3s+2m)(T-1) budget honoured (NOTE.md
     Thm B'): neither pass inflates, C = 0, and ``v_true in v + H`` at every
     step.
"""

from __future__ import annotations

import random
from fractions import Fraction as F

from _common import EXTRA, INK_MUTED, report, save, style, write_json  # noqa: I001
from padic_filtering.henon import (Henon, fixed_point_c, periodic_orbit,
                                   truth_orbit, truth_precision)
from padic_filtering.lattice import Lattice, transversality_defect
from padic_filtering.precision import (perturb, run_backward, run_forward,
                                       run_smoother)

SEED = 20260726
L = 200

# (p, alpha, delta): same fixed-point parametrisation as exp_theorem.py.
REGIMES = [
    (3, F(1, 3), 1),    # s=1, m=0 -- the workhorse
    (3, F(1, 3), 3),    # s=1, m=1 -- skewed tent
    (3, F(1, 9), 1),    # s=2, m=0
    (5, F(1, 5), 1),    # different residue characteristic
]


def budget(hen, T) -> int:
    """Starting precision keeping BOTH passes exact for T steps (NOTE.md Thm B').

    Sharp threshold ``(3s+2m)(T-1)`` plus slack.  ``3s+m`` is the forward rate
    only and under-provisions the backward pass whenever ``m > 0``.
    """
    return (3 * hen.s + 2 * hen.m) * (T - 1) + hen.s + hen.m + 4


def tent(hen, k, T):
    s, m = hen.s, hen.m
    lo = [k + min((s + m) * t, s * (T - t)) for t in range(T + 1)]
    hi = [k + max((s + m) * t, s * (T - t)) for t in range(T + 1)]
    return lo, hi


def random_itinerary(rng: random.Random) -> str:
    it = "".join(rng.choice("+-") for _ in range(L))
    assert minimal_cyclic_period(it) == L, "random itinerary is degenerate"
    return it


def minimal_cyclic_period(word: str) -> int:
    n = len(word)
    return next(d for d in range(1, n + 1)
                if n % d == 0 and word == word[:d] * (n // d))


def build_window(p, alpha, delta, itinerary, t0, T, k):
    """Truth points for the T-step window starting at step t0 of the long orbit."""
    hen = Henon.from_c(p, fixed_point_c(p, delta, alpha), delta)
    hen.check_regime()
    prec = 3 * (k + T * (hen.s + hen.m)) + 300
    X = periodic_orbit(hen, itinerary, prec)
    truth = truth_orbit(hen, X, t0 + T, prec)[t0:t0 + T + 1]
    return hen, truth, truth_precision(hen, prec)


def idealised(hen, truth, T, k):
    """Setting A on the window: pure Jacobian products, sharp law as equality."""
    H0 = Lattice.ball(hen.p, k)
    fwd, bwd = [H0], [H0]
    for t in range(T):
        fwd.append(fwd[-1].image(hen.jacobian(truth[t])))
    for t in range(T - 1, -1, -1):
        bwd.append(bwd[-1].image(hen.jacobian_inv(truth[t + 1])))
    bwd = bwd[::-1]
    sm = [fwd[t].intersect(bwd[t]) for t in range(T + 1)]
    lo, hi = tent(hen, k, T)
    assert [x.d1 for x in sm] == lo and [x.d2 for x in sm] == hi, \
        "sharp law broken on aperiodic window"
    assert [x.d1 for x in fwd] == [k - hen.s * t for t in range(T + 1)]
    assert [x.d1 for x in bwd] == [k - (hen.s + hen.m) * (T - t) for t in range(T + 1)]
    C = max(lo[t] - sm[t].d1 for t in range(T + 1))
    defects = [d for d in (transversality_defect(fwd[t], bwd[t])
                           for t in range(T + 1)) if d is not None]
    return C, max(defects, default=None)


def certified(hen, truth, tp, T, k, rng):
    """Setting B on the window: full tracker, inflation on, certified."""
    H0 = Lattice.ball(hen.p, k)
    fwd = run_forward(hen, perturb(truth[0], H0, rng), H0, truth, T, tp)
    bwd = run_backward(hen, perturb(truth[T], H0, rng), H0, truth, T, tp)
    # C == 0 alone does not witness the budget: inflation at the far end of a
    # pass lands where the other arm of the tent binds (docs/DEVELOPMENT.md).
    assert fwd.first_inflation is None and bwd.first_inflation is None, (
        f"budget k={k} not honoured (s={hen.s}, m={hen.m}, T={T}): "
        f"fwd inflates at {fwd.first_inflation}, bwd at {bwd.first_inflation}")
    sm = run_smoother(fwd, bwd, truth, tp)
    lo, _ = tent(hen, k, T)
    C = max(lo[t] - sm.d1()[i] for i, t in enumerate(sm.times))
    return C, sm.d1(), lo


def main() -> None:
    style()
    import matplotlib.pyplot as plt

    rng = random.Random(SEED)
    out = {"seed": SEED, "L": L, "windows": []}
    curves = {}  # one representative tent per (s, m), keyed for the figure

    for p, alpha, delta in REGIMES:
        main_regime = (p, alpha, delta) == REGIMES[0]
        # three independent random itineraries on the workhorse, one elsewhere
        for _ in range(3 if main_regime else 1):
            it = random_itinerary(rng)
            # three window positions and two lengths on the workhorse
            windows = [(40, 24), (88, 24), (140, 24), (76, 48)] if main_regime \
                else [(88, 24)]
            for t0, T in windows:
                k = budget(Henon.from_c(p, fixed_point_c(p, delta, alpha), delta), T)
                hen, truth, tp = build_window(p, alpha, delta, it, t0, T, k)
                C_a, defect = idealised(hen, truth, T, 0)
                C_b, d1, lo = certified(hen, truth, tp, T, k, rng)
                assert C_a == 0, f"idealised C={C_a} on window t0={t0}"
                assert defect == 0, f"defect={defect} on window t0={t0}"
                assert C_b == 0, f"certified C={C_b} on window t0={t0}"
                out["windows"].append({
                    "p": p, "s": hen.s, "m": hen.m, "t0": t0, "T": T, "k": k,
                    "itinerary_window": it[t0 % L:t0 % L + T] if t0 + T <= L
                    else (it + it)[t0 % L:t0 % L + T],
                    "C_idealised": C_a, "C_certified": C_b, "max_defect": defect,
                })
                if t0 == 88 and T == 24:
                    curves.setdefault((hen.s, hen.m), (hen, T, k, d1, lo))

    # ---- figure: the tent on a window of an effectively aperiodic orbit ----
    fig, ax = plt.subplots(figsize=(6.4, 4.4), layout="constrained")
    shown = [curves[key] for key in [(1, 0), (1, 1)] if key in curves]
    for i, (hen, T, k, d1, lo) in enumerate(shown):
        colour = EXTRA[i]
        ax.plot(range(T + 1), d1, color=colour, marker="o", markevery=4,
                label=f"certified $d_1(H_t)$, $s$={hen.s}, $m$={hen.m}, $k$={k}")
        ax.plot(range(T + 1), lo, color=colour, ls=":", lw=1.2, alpha=0.7)
    ax.set_xlabel("window step $t$  (orbit step $t_0{+}t$, $t_0$=88, $L$=200)")
    ax.set_ylabel("worst-direction digits  $d_1(H_t)$")
    ax.set_title("The tent on a random length-200 itinerary,\n"
                 "middle-$T$ window (dotted: predicted sharp law)",
                 loc="left", fontsize=11)
    ax.legend(loc="lower center", fontsize=8, labelcolor=INK_MUTED)
    path = save(fig, "exp_aperiodic_window")

    n = len(out["windows"])
    out["verdict"] = (
        f"THEOREM.md §5 item 3 (the one genuine empirical gap) is closed at the "
        f"cheap-attack level: on {n} windows of length T in {{24, 48}} cut from "
        f"random length-{L} itineraries (minimal cyclic period {L}, window "
        f"positions 40/88/140, regimes (s,m) in {{(1,0),(1,1),(2,0)}} and p=5), "
        "the sharp smoothed law holds as an equality with C = 0 in both the "
        "idealised and the budget-honouring certified settings, transversality "
        "defect 0 throughout.  The Jacobian sequence over each window carries "
        "no period, so the tent is not an artefact of short-period orbits.")
    write_json("exp_aperiodic_window", out)
    report("aperiodic window (THEOREM §5.3)", [
        f"{n} windows, all C_idealised = C_certified = 0, defect 0",
        f"L = {L}, minimal cyclic period {L} (asserted), T = 24 and 48",
        "regimes: (s,m) = (1,0) x3 itineraries x4 windows, (1,1), (2,0), p=5",
        f"figure: {path}"])


if __name__ == "__main__":
    main()
