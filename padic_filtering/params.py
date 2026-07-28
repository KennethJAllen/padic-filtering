"""Theorem-backed parameter sets from ADP (see ``docs/REFERENCES.md``).

Allen-DeMark-Petsche, *Non-Archimedean Henon maps, attractors, and horseshoes*
(arXiv:1610.04271) use ``phi_{a,b}(x,y) = (a + b y - x^2, x)`` with
``det Dphi = -b``.  Our form ``f(x,y) = (y, y^2 + c - delta x)`` is
affine-conjugate to ``phi_{-c,-delta}`` via ``(x,y) -> (-y,-x)``, so

    a = -c,     b = -delta,     |a| = |c|,     v_p(b) = v_p(delta) = m

and the parameter-space partition reads

    H_I    : v_p(c) >= 0, m = 0          good reduction; unimodular; a no-op
    H+_II  : v_p(c) >= 0, m >= 1         attractor; anisotropy demo
    H_III  : |c| > max(1, |delta|^2)     horseshoe; filtering demo

Nothing here is guessed: each entry cites the ADP result that supports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .henon import Henon, fixed_point_c
from .padic import Rat


@dataclass(frozen=True)
class ParamSet:
    name: str
    p: int
    c: Fraction
    delta: Fraction
    region: str
    source: str
    proven: bool = True

    def henon(self) -> Henon:
        return Henon.from_c(self.p, self.c, self.delta)

    @property
    def a(self) -> Fraction:
        """The ADP parameter ``a = -c``."""
        return -self.c

    @property
    def b(self) -> Fraction:
        """The ADP parameter ``b = -delta``."""
        return -self.delta


def _attractor(name, p, a, b, source, proven=True) -> ParamSet:
    """Build from ADP's ``(a, b)`` rather than from our ``(c, delta)``."""
    return ParamSet(name=name, p=p, c=Fraction(-a), delta=Fraction(-b),
                    region="H+_II", source=source, proven=proven)


# --------------------------------------------------------- H+_II (attractor)

# ADP Thm 2: over Q_3 with a = 2 (mod 9) and b = 3, the attractor is infinite
# and every orbit equidistributes with respect to an SRB-type measure -- so the
# test orbits are long and not eventually periodic.
ATTRACTOR_3ADIC = _attractor("attractor-3adic", 3, 2, 3, "ADP Thm 2")

# ADP §4.4 Table 1: numerically-supported infinite attractors, flagged as
# conjectural.  Used only for the p-sweep of experiment 5.1.
ATTRACTOR_5ADIC = _attractor("attractor-5adic", 5, 1, 5,
                             "ADP §4.4 Table 1 (conjectural)", proven=False)
ATTRACTOR_7ADIC = _attractor("attractor-7adic", 7, 2, 7,
                             "ADP §4.4 Table 1 (conjectural)", proven=False)

# ADP Table 1 counterexamples: these attractors are finite cycles, so the orbit
# collapses and tangency statistics go trivial.  Kept to be tested *against*.
FINITE_CYCLE_5ADIC = _attractor("finite-cycle-5adic", 5, 4, 5,
                                "ADP Table 1: 3-cycle -- avoid")
FINITE_CYCLE_7ADIC = _attractor("finite-cycle-7adic", 7, 1, 7,
                                "ADP Table 1: 2-cycle -- avoid")


# --------------------------------------------------------- H_III (horseshoe)

def horseshoe_fixed_point(p: int, delta: Rat, alpha: Rat, name: str) -> ParamSet:
    """``c`` such that ``(alpha, alpha)`` is a fixed point on the horseshoe."""
    return ParamSet(name=name, p=p, c=fixed_point_c(p, delta, alpha),
                    delta=Fraction(delta), region="H_III",
                    source="ADP Thm 1(a),(e)")


# The working example of the review: p=3, delta=1 (unit -- "area preserving",
# which is fine in H_III), alpha = 1/3, c = 5/9, s = 1.
HORSESHOE_3ADIC = horseshoe_fixed_point(3, 1, Fraction(1, 3), "horseshoe-3adic")
HORSESHOE_5ADIC = horseshoe_fixed_point(5, 1, Fraction(1, 5), "horseshoe-5adic")
HORSESHOE_7ADIC = horseshoe_fixed_point(7, 1, Fraction(1, 7), "horseshoe-7adic")

# A deeper horseshoe: s = 2 doubles the per-step expansion rate.
HORSESHOE_3ADIC_S2 = horseshoe_fixed_point(3, 1, Fraction(1, 9), "horseshoe-3adic-s2")

# Skewed horseshoes: ``delta`` is no longer a unit, so ``m = v(delta) > 0`` and
# the two passes stop being mirror images -- the forward pass loses ``s`` digits
# per step, the backward one ``s + m``, and the budget rates split into ``3s+m``
# forward and ``3s+2m`` two-sided (NOTE.md Thm A, Prop 4.4).  Every historical
# measurement in this repo had ``m = 0`` or ``s = 0``, where the two coincide;
# these two sets exist so that claims can be checked where they differ.
HORSESHOE_3ADIC_M1 = horseshoe_fixed_point(3, 3, Fraction(1, 3),
                                           "horseshoe-3adic-m1")
HORSESHOE_3ADIC_M2 = horseshoe_fixed_point(3, 9, Fraction(1, 3),
                                           "horseshoe-3adic-m2")

# An itinerary that is aperiodic under rotation, so it names a genuine
# period-8 orbit rather than a repeat of a shorter one (ADP Thm 1(e)).
DEFAULT_ITINERARY = "++-+--+-"

ALL = [ATTRACTOR_3ADIC, ATTRACTOR_5ADIC, ATTRACTOR_7ADIC,
       HORSESHOE_3ADIC, HORSESHOE_5ADIC, HORSESHOE_7ADIC, HORSESHOE_3ADIC_S2,
       HORSESHOE_3ADIC_M1, HORSESHOE_3ADIC_M2]
