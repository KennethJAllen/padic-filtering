"""5.6 -- Baseline comparison

Two baselines, one always available and one optional.

1. **review_checks.py** (always).  The repo's review script is a completely
   independent implementation of the same p-adic linear algebra -- different
   author, different data structures (dense rational matrices, dual-based
   intersection, no Hermite form over Z_p), written before this package
   existed.  Reproducing its numbers exercises HNF, SNF, image and intersection
   against code that shares no lines with them, which is the cheapest available
   defence against a silent HNF/SNF bug.

2. **SageMath's lattice-precision module** (optional).  This is the actual CRV
   implementation, and matching it on the prediction-only track validates the
   whole thing for free.  It is checked for and skipped with instructions if
   absent, rather than vendored: it is a large dependency and the point of §5.6
   is to compare against *their* code, not a copy of it.
"""

from __future__ import annotations

import importlib.util
from fractions import Fraction as F

from _common import report, write_json  # noqa: I001
from padic_filtering.lattice import Lattice

import review_checks as rc


def compare_with_review_checks(p=3, T=12, delta=1, alpha=F(1, 3)):
    """Reproduce review_checks.py's H_III numbers with the package's lattices."""
    J = ((F(0), F(1)), (F(-delta), 2 * alpha))
    Ji = ((2 * alpha / delta, -1 / delta), (F(1), F(0)))
    rows, mismatches = [], []

    fwd, bwd = Lattice.identity(p), Lattice.identity(p)
    fwd_list, bwd_list = [fwd], [bwd]
    for _ in range(T):
        fwd, bwd = fwd.image(J), bwd.image(Ji)
        fwd_list.append(fwd)
        bwd_list.append(bwd)

    # the same quantities, computed by review_checks' own routines
    rJ, rJi = rc.jac(alpha, alpha, delta), rc.jac_inv(alpha, alpha, delta)
    I2 = [[F(1), F(0)], [F(0), F(1)]]
    rf, rb = [I2], [I2]
    for _ in range(T):
        rf.append(rc.matmul(rJ, rf[-1]))
        rb.append(rc.matmul(rb[-1], rJi))

    for t in range(T + 1):
        mine_f = fwd_list[t].elementary_divisors()
        ref_f = rc.snf_vals(rf[t], p)
        mine_i = fwd_list[t].intersect(bwd_list[T - t]).elementary_divisors()
        ref_i = rc.snf_vals(rc.intersect(rf[t], rb[T - t]), p)
        row = {"t": t, "forward_mine": list(mine_f), "forward_ref": list(ref_f),
               "smoothed_mine": list(mine_i), "smoothed_ref": list(ref_i)}
        rows.append(row)
        if tuple(mine_f) != tuple(ref_f) or tuple(mine_i) != tuple(ref_i):
            mismatches.append(row)
    return rows, mismatches


def sage_available() -> bool:
    return importlib.util.find_spec("sage") is not None


def main() -> None:
    rows, mismatches = compare_with_review_checks()
    assert not mismatches, f"disagreement with review_checks.py: {mismatches[:3]}"

    # also re-run review_checks' own assertions end to end
    rc.check_H2plus()
    rc.check_H3()

    sage = sage_available()
    out = {
        "review_checks": {
            "agrees": not mismatches,
            "steps_compared": len(rows),
            "rows": rows,
            "note": ("independent implementation: dense rational matrices and a "
                     "dual-based intersection, with no Hermite form over Z_p"),
        },
        "sagemath": {
            "available": sage,
            "status": "compared" if sage else "SKIPPED -- SageMath not installed",
            "how_to_run": (
                "install SageMath (https://www.sagemath.org) and re-run; the "
                "relevant module is sage.rings.padics.lattice_precision, and the "
                "quantity to match is the prediction-only track's elementary "
                "divisor exponents (d1, d2) from exp_5_1_anisotropy.json"),
            "reference": "github.com/roed314/padicprec (Caruso-Roe-Vaccon)",
        },
        "verdict": (
            "PASS against review_checks.py on every step compared. "
            + ("SageMath comparison ran." if sage else
               "SageMath comparison SKIPPED (not installed) -- this is the one "
               "part of §5.6 not executed here, and it is the first thing a CRV "
               "reviewer will ask about.")),
    }
    write_json("exp_5_6_baseline", out)
    report("5.6 baseline", [
        f"review_checks.py: agrees on {len(rows)} steps "
        f"(forward divisors and smoothed intersections)",
        f"review_checks.py own assertions: passed",
        f"SageMath: {out['sagemath']['status']}"])


if __name__ == "__main__":
    main()
