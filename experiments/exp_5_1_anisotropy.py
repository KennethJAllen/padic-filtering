"""5.1 -- Does anisotropy actually grow?  (KILL CRITERION, RESOLVED)

Settled during the review, analytically (Newton polygon) and numerically.  The
expected slopes are therefore *assertions*, not observations:

    H+_II   anisotropy grows at slope m       (divisors (k, k + t m))
    H_III   anisotropy grows at slope 2s + m  (divisors (k - t s, k + t(s+m)))

Kept as a cheap regression test of the lattice code, swept over p in {3,5,7}
with the ADP-backed parameters of §2.3b, and it still records the tangency
events (y_t = 0 mod p) that break the w = 0 genericity in H+_II.
"""

from __future__ import annotations

import random
from fractions import Fraction as F

from _common import EXTRA, INK_MUTED, label_end, report, save, style, write_json  # noqa: E402,I001
from padic_filtering.henon import periodic_orbit, truth_orbit, truth_precision
from padic_filtering.lattice import Lattice
from padic_filtering.padic import vp
from padic_filtering.params import (ATTRACTOR_3ADIC, ATTRACTOR_5ADIC,
                                    ATTRACTOR_7ADIC, DEFAULT_ITINERARY,
                                    HORSESHOE_3ADIC, HORSESHOE_3ADIC_S2,
                                    HORSESHOE_5ADIC, HORSESHOE_7ADIC)
from padic_filtering.precision import perturb, run_forward

SEED = 20260726
# H+_II ground truth is exact modulo p^ATTRACTOR_PREC (the map is nonexpanding
# there, so the reduction loses nothing); ample room above any d2 reached.
ATTRACTOR_PREC = 400


def exact_prefix(track) -> int:
    """Number of leading steps for which the linearisation stayed exact."""
    return track.first_inflation if track.first_inflation is not None else len(track.times)


def attractor_case(ps, T=15, k=40):
    hen = ps.henon()
    truth = hen.orbit((F(1), F(1)), T, prec=ATTRACTOR_PREC)
    H0 = Lattice.ball(hen.p, k)
    fwd = run_forward(hen, perturb(truth[0], H0, random.Random(SEED)), H0, truth, T)
    tangencies = [t for t, (_, y) in enumerate(truth) if vp(y, hen.p) >= 1]

    # ASSERTIONS, not observations.  (a) det J = delta is a free invariant, so
    # v_p(det H) rises by exactly m per step -- this holds unconditionally.
    assert [r.lattice_digits for r in fwd.records] == [2 * k + t * hen.m for t in fwd.times]
    # (b) H+_II is nonexpanding (ADP Thm 1(d)): the forward pass never loses.
    assert all(b >= a for a, b in zip(fwd.d1(), fwd.d1()[1:])), "H+_II must not lose"
    assert min(fwd.d1()) >= k
    # (c) the clean slope-m law needs w = 0 genericity.  A tangency y_t = 0 mod p
    # puts the middle Newton-polygon vertex (1, w) above the lower hull, the
    # eigenvalue valuations become m/2 each (ramified), and the anisotropy kinks
    # instead of climbing -- so assert the law only where it is claimed to hold.
    if not tangencies:
        assert fwd.d1() == [k] * len(fwd.times), "no tangency: d1 must be flat"
        assert fwd.d2() == [k + t * hen.m for t in fwd.times], "slope must be m"
    return hen, fwd, truth, tangencies


def horseshoe_case(ps, T=15, k=60, prec=400):
    hen = ps.henon()
    hen.check_regime()
    X = periodic_orbit(hen, DEFAULT_ITINERARY, prec)
    truth = truth_orbit(hen, X, T, prec)
    tp = truth_precision(hen, prec)
    H0 = Lattice.ball(hen.p, k)
    fwd = run_forward(hen, perturb(truth[0], H0, random.Random(SEED)), H0, truth, T, tp)
    s, m = hen.s, hen.m
    # The clean rates hold while the linearisation is exact; past the horizon the
    # inflation term takes over (that kink is 5.2's subject, and is plotted here).
    n = exact_prefix(fwd)
    assert fwd.d1()[:n] == [k - t * s for t in fwd.times[:n]]
    assert fwd.d2()[:n] == [k + t * (s + m) for t in fwd.times[:n]]
    # det J = delta holds unconditionally, inflation or not, only while exact
    assert [r.lattice_digits for r in fwd.records[:n]] == \
        [2 * k + t * m for t in fwd.times[:n]]
    return hen, fwd, truth, []


def main() -> None:
    style()
    import matplotlib.pyplot as plt

    attractors = [ATTRACTOR_3ADIC, ATTRACTOR_5ADIC, ATTRACTOR_7ADIC]
    horseshoes = [HORSESHOE_3ADIC, HORSESHOE_5ADIC, HORSESHOE_7ADIC, HORSESHOE_3ADIC_S2]
    out = {"seed": SEED, "cases": []}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, cases, runner, title, legend_loc in [
        (axes[0], attractors, attractor_case, "H$^+_{II}$ attractor: slope $m$",
         "upper left"),
        (axes[1], horseshoes, horseshoe_case, "H$_{III}$ horseshoe: slope $2s+m$",
         "lower right"),
    ]:
        # Curves for different primes coincide *exactly* when the rate is the
        # same -- that p-independence is the result, but drawn naively it is
        # just overlapping lines with colliding labels.  Group by the curve and
        # name every prime that produces it.
        curves: dict[tuple, dict] = {}
        for ps in cases:
            hen, fwd, truth, tang = runner(ps)
            aniso = [r.anisotropy for r in fwd.records]
            slope = hen.m if hen.s == 0 else 2 * hen.s + hen.m
            n = exact_prefix(fwd)
            observed = (aniso[n - 1] - aniso[0]) / max(1, fwd.times[n - 1])
            if not tang:
                assert observed == slope, f"{ps.name}: slope {observed} != {slope}"
            key = tuple(aniso)
            curve = curves.setdefault(key, {
                "times": fwd.times, "aniso": aniso, "primes": [], "slope": slope,
                "tang": tang, "horizon": fwd.first_inflation, "s": hen.s})
            curve["primes"].append(hen.p)
            out["cases"].append({
                "name": ps.name, "p": hen.p, "region": hen.region,
                "m": hen.m, "s": hen.s, "delta": hen.delta, "c": hen.c,
                "source": ps.source, "proven": ps.proven,
                "predicted_slope": slope, "observed_slope": observed,
                "exactness_horizon": fwd.first_inflation,
                "anisotropy": aniso, "d1": fwd.d1(), "d2": fwd.d2(),
                "tangencies": tang,
            })
        for i, curve in enumerate(curves.values()):
            colour = EXTRA[i % len(EXTRA)]
            primes = ", ".join(f"{p}" for p in curve["primes"])
            if curve["tang"]:
                name = f"$p$={primes}: tangencies"
            else:
                name = f"$p$={primes}, $s$={curve['s']}: slope {curve['slope']}"
            ax.plot(curve["times"], curve["aniso"], color=colour, marker="o",
                    markevery=5, label=name)
            label_end(ax, curve["times"], curve["aniso"], name, colour)
            for t in curve["tang"]:
                ax.axvline(t, color=colour, lw=0.8, alpha=0.2, zorder=0)
            h = curve["horizon"]
            if h is not None:
                ax.plot([curve["times"][h - 1]], [curve["aniso"][h - 1]],
                        marker="o", ms=9, mfc="none", mec=colour, mew=1.6)
                ax.annotate("exactness horizon\n(inflation begins)",
                            xy=(curve["times"][h - 1], curve["aniso"][h - 1]),
                            xytext=(8, -34), textcoords="offset points",
                            fontsize=8, color=INK_MUTED, ha="left")
        ax.set_title(title)
        ax.set_xlabel("iteration $t$")
        ax.set_xlim(right=curve["times"][-1] * 1.55)
        ax.legend(loc=legend_loc, fontsize=8, labelcolor=INK_MUTED)
    axes[0].set_ylabel("anisotropy  $d_2 - d_1$")
    fig.suptitle("5.1  Anisotropy grows linearly in both regimes (kill criterion passed)",
                 x=0.02, ha="left", fontsize=12)
    path = save(fig, "exp_5_1_anisotropy")

    tangency_cases = [c["name"] for c in out["cases"] if c["tangencies"]]
    out["verdict"] = (
        "PASS: anisotropy grows at the predicted slope in every tangency-free case. "
        f"Cases with tangencies ({', '.join(tangency_cases) or 'none'}) kink instead: "
        "the middle Newton-polygon vertex leaves the lower hull, the eigenvalue "
        "valuations ramify to m/2, and anisotropy stalls -- the nonexpanding and "
        "det-J invariants still hold and are asserted.")
    write_json("exp_5_1_anisotropy", out)
    report("5.1 anisotropy", [
        f"{c['name']}: slope {c['observed_slope']} (predicted {c['predicted_slope']})"
        f"{'  tangencies at ' + str(c['tangencies']) if c['tangencies'] else ''}"
        for c in out["cases"]] + [f"figure: {path}"])


if __name__ == "__main__":
    main()
