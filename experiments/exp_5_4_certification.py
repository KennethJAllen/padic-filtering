"""5.4 -- Is the certification ever violated?

Runs 10^4 random orbits with assertions on.  ``v_true in v + H`` is checked at
every step of every track by :func:`padic_filtering.precision.certify`, so a
violation surfaces as a ``CertificationError`` rather than a silent wrong
answer.  Any violation is a bug in the lattice code or a misuse of the
exactness condition -- to be tracked down, not loosened away.

A passing run of an assertion proves nothing unless the assertion can fail, so
this experiment also includes a **mutation control**: the same orbits are run
with inflation disabled (the propagated lattice is taken to be ``J(v)H`` even
when the quadratic remainder escapes it).  That is exactly the misuse §5.4
warns about, and it must produce violations.  If the control passes silently,
the certification is vacuous and the headline result is worthless.
"""

from __future__ import annotations

import random
from fractions import Fraction as F

from _common import EXTRA, INK_MUTED, report, save, style, write_json  # noqa: I001
from padic_filtering.henon import (Henon, fixed_point_c, periodic_orbit,
                                   truth_orbit, truth_precision)
from padic_filtering.lattice import Lattice
from padic_filtering.precision import (CertificationError, PrecisionExhausted,
                                       certify, perturb, propagate,
                                       run_backward, run_forward, run_smoother)

SEED = 20260726
N_ORBITS = 10_000
PRIMES = [3, 5, 7]
ITINERARY_LENGTHS = [1, 2, 3, 5, 8]


def random_case(rng: random.Random):
    """A random horseshoe orbit, with random precision budget and horizon."""
    p = rng.choice(PRIMES)
    s = rng.choice([1, 1, 2])
    alpha = F(rng.randrange(1, p**s), p**s) if s > 1 else F(1, p)
    # keep alpha a genuine valuation -s point
    alpha = F(rng.choice([n for n in range(1, p**s) if n % p != 0]), p**s)
    c = fixed_point_c(p, 1, alpha)
    hen = Henon.from_c(p, c, 1)
    if hen.region != "H_III" or not hen.is_square_a():
        return None
    itinerary = "".join(rng.choice("+-") for _ in range(rng.choice(ITINERARY_LENGTHS)))
    k = rng.randrange(8, 40)
    T = rng.randrange(2, 14)
    prec = 4 * (k + T * (hen.s + 2)) + 40
    return hen, itinerary, k, T, prec


def run_one(case, mutate: bool = False):
    """Returns (n_steps_checked, violation_or_None)."""
    hen, itinerary, k, T, prec = case
    try:
        X = periodic_orbit(hen, itinerary, prec)
    except (ValueError, ZeroDivisionError, AssertionError):
        return 0, None  # itinerary not realisable at this precision; not a violation
    truth = truth_orbit(hen, X, T, prec)
    tp = truth_precision(hen, prec)
    H0 = Lattice.ball(hen.p, k)
    rng = random.Random(hash((itinerary, k, T)) & 0xFFFF)
    v0 = perturb(truth[0], H0, rng)
    if mutate:
        return _run_mutated(hen, v0, H0, truth, T, tp)
    try:
        fwd = run_forward(hen, v0, H0, truth, T, tp)
        bwd = run_backward(hen, perturb(truth[T], H0, rng), H0, truth, T, tp)
        sm = run_smoother(fwd, bwd, truth, tp)
    except CertificationError as exc:
        return 0, str(exc)
    return len(fwd.times) + len(bwd.times) + len(sm.times), None


def _run_mutated(hen, v0, H0, truth, T, tp):
    """Propagate with inflation suppressed -- the misuse the assertion guards."""
    v, H, checked = v0, H0, 0
    for t in range(1, T + 1):
        J = hen.jacobian(v)
        strict = H.image(J)           # <- no inflation, even when inexact
        try:
            if strict.d1 < -hen.s:
                return checked, None  # floor reached before the bug could show
            v, H = strict.reduce_vector(hen.f(v)), strict
        except (PrecisionExhausted, ValueError):
            return checked, None
        checked += 1
        if not certify(v, H, truth[t], tp):
            return checked, f"violation at t={t} with inflation disabled"
    return checked, None


def main() -> None:
    style()
    import matplotlib.pyplot as plt

    rng = random.Random(SEED)
    cases = []
    while len(cases) < N_ORBITS:
        case = random_case(rng)
        if case is not None:
            cases.append(case)

    honest_violations, checked_steps = [], 0
    for case in cases:
        n, violation = run_one(case)
        checked_steps += n
        if violation:
            honest_violations.append({"case": str(case[:4]), "violation": violation})

    mutated_violations, mutated_checked = 0, 0
    for case in cases:
        n, violation = run_one(case, mutate=True)
        mutated_checked += n
        if violation:
            mutated_violations += 1

    out = {
        "seed": SEED, "n_orbits": N_ORBITS, "primes": PRIMES,
        "certified_steps": checked_steps,
        "violations": honest_violations,
        "n_violations": len(honest_violations),
        "mutation_control": {
            "description": "inflation disabled; the exactness condition is misused",
            "orbits_run": N_ORBITS,
            "steps_checked": mutated_checked,
            "violations_detected": mutated_violations,
            "detection_rate": mutated_violations / N_ORBITS,
        },
    }
    assert not honest_violations, f"certification violated: {honest_violations[:3]}"
    assert mutated_violations > 0, (
        "the mutation control produced no violations -- the certification is "
        "vacuous and proves nothing")

    fig, ax = plt.subplots(figsize=(6.5, 4.2), layout="constrained")
    bars = ["as implemented", "inflation disabled\n(mutation control)"]
    vals = [len(honest_violations), mutated_violations]
    ax.bar(bars, vals, color=[EXTRA[2], EXTRA[1]], width=0.55, zorder=3)
    ax.axhline(0, color="#d6d5d0", lw=1.2, zorder=2)
    for x, v in enumerate(vals):
        ax.annotate(f"{v:,} / {N_ORBITS:,}", xy=(x, v), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=10,
                    color=INK_MUTED)
    ax.set_ylabel("orbits with a certification violation")
    ax.set_title(f"5.4  {N_ORBITS:,} random horseshoe orbits, "
                 f"{checked_steps:,} certified steps", loc="left", fontsize=11.5)
    ax.set_ylim(0, max(vals) * 1.25 or 1)
    path = save(fig, "exp_5_4_certification")

    out["verdict"] = (
        f"PASS: zero violations in {N_ORBITS:,} random orbits "
        f"({checked_steps:,} certified steps).  The mutation control (inflation "
        f"disabled) is caught on {mutated_violations:,} orbits, so the assertion "
        f"has teeth.")
    write_json("exp_5_4_certification", out)
    report("5.4 certification", [
        f"orbits: {N_ORBITS:,}   certified steps: {checked_steps:,}",
        f"violations as implemented: {len(honest_violations)}",
        f"violations with inflation disabled (control): {mutated_violations:,} "
        f"({100 * mutated_violations / N_ORBITS:.1f}% of orbits)",
        f"figure: {path}"])


if __name__ == "__main__":
    main()
