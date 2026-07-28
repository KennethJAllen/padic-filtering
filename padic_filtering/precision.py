"""Precision trackers: naive, lattice (predict only), filtered.

Four *tracks* over one orbit; ``truth`` lives in :mod:`henon`.

  naive     a single scalar absolute-precision counter -- the standard
            capped-precision model.  Loses ``s`` digits per forward step
            (``s + m`` backward).
  lattice   the CRV recursion: ``v <- f(v)``, ``H <- f'(v) H``, with an explicit
            exactness test and honest inflation when it fails.
  filtered  lattice prediction plus an update step: either revealed oracle
            digits (§2.5a) or, the real contribution, the backward pass
            intersected with the forward pass (§2.5b) -- a Rauch-Tung-Striebel
            smoother that is exact rather than Gaussian.

Every step is certified against ground truth: ``v_true in v + H``.  A filter
that is fast and wrong is worthless, and here it can be proven not to be.

All representatives are reduced modulo their own lattice after every step.
That is required for the coset to stay canonical, and it is also what keeps
the exact rational arithmetic from doubling in size every iteration.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from fractions import Fraction

from .henon import Henon
from .lattice import Lattice, Vec, as_vec, intersect_cosets
from .padic import INFINITY, ppow


@dataclass
class StepRecord:
    t: int
    d1: int
    d2: int
    anisotropy: int
    lattice_digits: int
    naive_digits: int
    exact: bool
    inflated: bool


@dataclass
class Track:
    name: str
    records: list[StepRecord] = field(default_factory=list)
    states: list[tuple[Vec, Lattice]] = field(default_factory=list)
    times: list[int] = field(default_factory=list)
    first_inflation: int | None = None
    exhausted_at: int | None = None

    def d1(self) -> list[int]:
        return [r.d1 for r in self.records]

    def d2(self) -> list[int]:
        return [r.d2 for r in self.records]

    def at(self, t: int) -> tuple[Vec, Lattice] | None:
        try:
            return self.states[self.times.index(t)]
        except ValueError:
            return None

    def record(self, t: int, v: Vec, H: Lattice, exact: bool, inflated: bool) -> None:
        d1, d2 = H.elementary_divisors()
        self.records.append(StepRecord(t, d1, d2, d2 - d1, d1 + d2, 2 * d1,
                                       exact, inflated))
        self.states.append((v, H))
        self.times.append(t)
        if inflated and self.first_inflation is None:
            self.first_inflation = t

    def sort_by_time(self) -> None:
        order = sorted(range(len(self.times)), key=lambda i: self.times[i])
        self.records = [self.records[i] for i in order]
        self.states = [self.states[i] for i in order]
        self.times = [self.times[i] for i in order]


class PrecisionExhausted(RuntimeError):
    """The lattice no longer localises the point at all.

    Once ``d1`` falls below the precision floor the coset is wider than the
    invariant set, so the estimate carries no information -- and because the map
    is quadratic, the remainder module then *squares* the uncertainty every
    step, doubling the lattice exponents.  Continuing is both meaningless and a
    memory hazard, so the track stops here and reports the time.
    """

    def __init__(self, t: int, H: Lattice, floor: int):
        super().__init__(f"precision exhausted at t={t}: d1={H.d1} < floor={floor}")
        self.t, self.H, self.floor = t, H, floor


# ------------------------------------------------------------------ naive


def naive_step(k: int, hen: Henon, backward: bool = False) -> int:
    """Absolute precision after one step of the capped-precision model.

    Forward, ``y' = y^2 + c - delta x``.  A value known to absolute precision
    ``k`` squares to one known to ``k + v_p(y) = k - s`` (on the horseshoe
    ``v_p(y) = -s``), while ``delta x`` is known to ``k + m``, so the sum is
    known to ``min(k - s, k + m) = k - s``.  Backward there is an additional
    division by ``delta``, costing ``m``.

    Equivalently, in scaled coordinates ``X = p^s x`` the same ``s`` is
    the cost of the exact division by ``p^s`` in ``(Y^2 + C)/p^s`` -- the two
    descriptions are the same loss seen in two coordinate systems.
    """
    return k - hen.s - (hen.m if backward else 0)


# ---------------------------------------------------------------- lattice


def _remainder(hen: Henon, H: Lattice, backward: bool, tight: bool):
    """Generator of the quadratic remainder module, and the axis it lives on."""
    if backward:
        w = H.x_projection_exponent() if tight else H.d1
        axis = 0
    else:
        w = H.y_projection_exponent() if tight else H.d1
        axis = 1
    rem_exp = hen.quadratic_remainder_exponent(w, backward)
    gen = ((ppow(hen.p, rem_exp), Fraction(0)) if axis == 0
           else (Fraction(0), ppow(hen.p, rem_exp)))
    return gen, rem_exp


def is_linearisation_exact(hen: Henon, v, H: Lattice, backward: bool = False,
                           tight: bool = True) -> bool:
    """Explicit membership test for exactness (NOTE.md Definition 4.1).

    The Taylor expansion of the quadratic map terminates::

        f(v + h) = f(v) + J(v) h + (0, h_y^2)

    so the linearisation is exact exactly when the remainder module lies inside
    the propagated lattice ``J(v) H``.  The remainder module is
    ``(0, p^(2w) Z_p)`` where ``p^w Z_p`` is the projection of ``H`` onto the
    squared coordinate (``p^(2w - m)`` on the other axis, backward).

    ``tight=True`` uses that projection exponent ``w``; ``tight=False`` uses the
    cruder bound ``w = d1`` from ``H subset p^d1 Z_p^2``.  Both are sufficient
    conditions and both are tested by explicit
    membership -- no inequality is hardcoded.
    """
    J = hen.jacobian_inv(v) if backward else hen.jacobian(v)
    gen, _ = _remainder(hen, H, backward, tight)
    return H.image(J).contains(gen)


def default_floor(hen: Henon) -> int:
    """Precision floor: ``-s``, the scale of the points themselves.

    Orbit points on the horseshoe have valuation ``-s``, so a coset of
    ``p^-s Z_p^2`` pins down nothing at all.  Below this the estimate is vacuous.
    """
    return -hen.s


def propagate(hen: Henon, v, H: Lattice, backward: bool = False,
              tight: bool = True, floor: int | None = None,
              t: int = -1) -> tuple[Vec, Lattice, bool, bool]:
    """One CRV prediction step, with exactness test and honest inflation.

    Returns ``(v', H', exact, inflated)``.  When the linearisation is not exact
    the propagated lattice is inflated to the smallest lattice containing both
    ``J(v) H`` and the quadratic remainder module -- never shrunk, so the
    certificate ``v_true in v + H`` is preserved unconditionally.

    Raises :class:`PrecisionExhausted` if the result falls below ``floor``.
    """
    J = hen.jacobian_inv(v) if backward else hen.jacobian(v)
    JH = H.image(J)
    gen, _ = _remainder(hen, H, backward, tight)
    exact = JH.contains(gen)
    Hn = JH if exact else JH.add_vector(gen)
    if floor is not None and Hn.d1 < floor:
        raise PrecisionExhausted(t, Hn, floor)
    vn = hen.f_inv(v) if backward else hen.f(v)
    return Hn.reduce_vector(vn), Hn, exact, not exact


# ---------------------------------------------------------- certification


def certify(v, H: Lattice, v_true, truth_precision: int | float = INFINITY) -> bool:
    """Is the claim ``v_true in v + H`` true?

    ``truth_precision`` is the absolute precision to which the ground-truth
    orbit itself is known (``INFINITY`` for an exactly rational orbit such as a
    fixed point).  When it is finite the test is made against
    ``H + p^truth_precision Z_p^2``, and the caller is expected to keep the
    working precision well above ``d2(H)`` so the test stays meaningful.
    """
    v, v_true = as_vec(v), as_vec(v_true)
    diff = (v_true[0] - v[0], v_true[1] - v[1])
    if truth_precision == INFINITY:
        return H.contains(diff)
    return (H + Lattice.ball(H.p, int(truth_precision))).contains(diff)


class CertificationError(AssertionError):
    pass


# ------------------------------------------------------------------ runs


def run_forward(hen: Henon, v0, H0: Lattice, truth: list[Vec], T: int,
                truth_precision: int | float = INFINITY,
                tight: bool = True, floor: int | None = None) -> Track:
    """CRV prediction-only forward pass, certified at every step.

    Stops early (recording ``exhausted_at``) if the lattice falls through the
    precision floor; pass ``floor=-inf`` to disable, at your own risk.
    """
    floor = default_floor(hen) if floor is None else floor
    track = Track("lattice")
    v, H = v0, H0
    _assert_certified(v, H, truth[0], truth_precision, "lattice", 0)
    track.record(0, v, H, True, False)
    for t in range(1, T + 1):
        try:
            v, H, exact, inflated = propagate(hen, v, H, backward=False,
                                              tight=tight, floor=floor, t=t)
        except PrecisionExhausted:
            track.exhausted_at = t
            break
        _assert_certified(v, H, truth[t], truth_precision, "lattice", t)
        track.record(t, v, H, exact, inflated)
    return track


def run_backward(hen: Henon, vT, HT: Lattice, truth: list[Vec], T: int,
                 truth_precision: int | float = INFINITY,
                 tight: bool = True, floor: int | None = None) -> Track:
    """Backward pass from ``t = T`` down to ``t = 0``; records are in forward order.

    This is the measurement source of the smoother: ``f^-1`` swaps the
    expanding and contracting directions, so the backward lattice is precise
    exactly where the forward one is not.
    """
    floor = default_floor(hen) if floor is None else floor
    track = Track("backward")
    v, H = vT, HT
    _assert_certified(v, H, truth[T], truth_precision, "backward", T)
    track.record(T, v, H, True, False)
    for t in range(T - 1, -1, -1):
        try:
            v, H, exact, inflated = propagate(hen, v, H, backward=True,
                                              tight=tight, floor=floor, t=t)
        except PrecisionExhausted:
            track.exhausted_at = t
            break
        _assert_certified(v, H, truth[t], truth_precision, "backward", t)
        track.record(t, v, H, exact, inflated)
    track.sort_by_time()
    return track


def run_smoother(fwd: Track, bwd: Track, truth: list[Vec],
                 truth_precision: int | float = INFINITY) -> Track:
    """Intersect the forward and backward cosets at every time.

    The exact analogue of a Rauch-Tung-Striebel smoother: it exists precisely
    because the Henon map is an automorphism.
    """
    track = Track("filtered")
    for t in sorted(set(fwd.times) & set(bwd.times)):
        (vf, Hf), (vb, Hb) = fwd.at(t), bwd.at(t)
        got = intersect_cosets(vf, Hf, vb, Hb)
        if got is None:
            raise CertificationError(
                f"forward and backward cosets are disjoint at t={t}: "
                "at least one of the two passes is unsound")
        v, H = got
        _assert_certified(v, H, truth[t], truth_precision, "filtered", t)
        track.record(t, v, H, True, False)
    return track


def run_oracle(hen: Henon, v0, H0: Lattice, truth: list[Vec], T: int, reveal_k: int,
               reveal_times, truth_precision: int | float = INFINITY,
               floor: int | None = None) -> Track:
    """Forward pass with oracle-digit updates (machinery test only).

    At each time in ``reveal_times`` the true point is revealed modulo
    ``p^reveal_k``, i.e. the coset is intersected with a ball.  This is *not* a
    justified measurement model (see docs/ROADMAP.md, non-goals) -- it exists
    to validate the
    update machinery independently of the backward pass.
    """
    floor = default_floor(hen) if floor is None else floor
    track = Track("oracle")
    v, H = v0, H0
    reveal = set(reveal_times)
    for t in range(T + 1):
        if t:
            try:
                v, H, exact, inflated = propagate(hen, v, H, floor=floor, t=t)
            except PrecisionExhausted:
                track.exhausted_at = t
                break
        else:
            exact, inflated = True, False
        if t in reveal:
            ball = Lattice.ball(hen.p, reveal_k)
            got = intersect_cosets(v, H, truth[t], ball)
            if got is None:
                raise CertificationError(f"oracle contradicts the filter at t={t}")
            v, H = got
        _assert_certified(v, H, truth[t], truth_precision, "oracle", t)
        track.record(t, v, H, exact, inflated)
    return track


def naive_track(hen: Henon, k0: int, T: int) -> list[int]:
    ks = [k0]
    for _ in range(T):
        ks.append(naive_step(ks[-1], hen))
    return ks


def _assert_certified(v, H, v_true, truth_precision, name, t) -> None:
    if not certify(v, H, v_true, truth_precision):
        raise CertificationError(
            f"certification failed on track '{name}' at t={t}: "
            f"v_true not in v + H (H={H})")
    if truth_precision != INFINITY and H.d2 >= truth_precision:
        raise CertificationError(
            f"working precision exhausted on track '{name}' at t={t}: "
            f"d2={H.d2} >= truth precision {truth_precision}; "
            "the certificate is no longer meaningful -- raise the precision")


# ------------------------------------------------------------- utilities


def perturb(v, H: Lattice, rng: random.Random) -> Vec:
    """A different representative of the same coset ``v + H``.

    The tracker must not be handed ground truth: it knows the initial point
    only modulo ``H``, so it starts from an arbitrary member of the coset, and
    the offset is chosen with valuation *exactly* that of the lattice so the
    starting representative really does differ from the truth.

    Deliberately not reduced: reduction would map it straight back to the
    canonical representative of the coset and undo the perturbation.
    """
    v = as_vec(v)
    c1, c2 = H.columns()
    z1 = rng.randrange(1, H.p**3) | 1
    z2 = rng.randrange(1, H.p**3) | 1
    return (v[0] + z1 * c1[0] + z2 * c2[0], v[1] + z1 * c1[1] + z2 * c2[1])
