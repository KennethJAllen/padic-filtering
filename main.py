"""Smallest end-to-end demonstration: the smoother beats forward-only propagation.

    uv run python main.py

Prints the three tracks side by side on a period-8 horseshoe orbit.  For the
figures and the full de-risking suite, run `python experiments/run_all.py`.
"""

from __future__ import annotations

import random

from padic_filtering.henon import periodic_orbit, truth_orbit, truth_precision
from padic_filtering.lattice import Lattice
from padic_filtering.params import DEFAULT_ITINERARY, HORSESHOE_3ADIC
from padic_filtering.precision import (naive_track, perturb, run_backward,
                                       run_forward, run_smoother)

T, K, PREC = 24, 80, 900


def main() -> None:
    hen = HORSESHOE_3ADIC.henon()
    hen.check_regime()
    print(f"Henon over Q_{hen.p}: c = {hen.c}, delta = {hen.delta}   "
          f"region {hen.region} (s = {hen.s}, m = {hen.m})")
    print(f"eigenvalue valuations {hen.eigenvalue_valuations} -- "
          f"expansion and contraction\n")

    X = periodic_orbit(hen, DEFAULT_ITINERARY, PREC)
    truth = truth_orbit(hen, X, T, PREC)
    tp = truth_precision(hen, PREC)
    H0 = Lattice.ball(hen.p, K)
    rng = random.Random(20260726)

    fwd = run_forward(hen, perturb(truth[0], H0, rng), H0, truth, T, tp)
    bwd = run_backward(hen, perturb(truth[T], H0, rng), H0, truth, T, tp)
    sm = run_smoother(fwd, bwd, truth, tp)
    naive = naive_track(hen, K, T)

    print(f"period-{len(DEFAULT_ITINERARY)} orbit, T = {T} steps, "
          f"starting precision k = {K}\n")
    print(f"{'t':>3} {'naive':>7} {'lattice':>9} {'filtered':>10} "
          f"{'anisotropy':>11}")
    for i, t in enumerate(fwd.times):
        print(f"{t:>3} {naive[t]:>7} {fwd.d1()[i]:>9} {sm.d1()[i]:>10} "
              f"{fwd.records[i].anisotropy:>11}")

    print(f"\nforward-only: {fwd.d1()[0]} -> {fwd.d1()[-1]} digits "
          f"({hen.s} lost per step)")
    print(f"smoothed:     never below {min(sm.d1())}, peaks at {max(sm.d1())}")
    print("every step certified: v_true in v + H")


if __name__ == "__main__":
    main()
