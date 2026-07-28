"""Numerical ground truth for the two lemmas behind Theorem A (THEOREM.md §8).

These pin down, as exact assertions on random long itineraries, the two
inductions the proof of Theorem A rests on.  If either fails after a code
change, the proof strategy documented in THEOREM.md §4/§8 is describing a
different object than the code computes.

Lemma 1 (row valuations, forward and backward products).  For
``M_t = J_{t-1} ... J_0`` the update sends (top, bottom) to
(bottom, -delta*top + 2y*bottom); the summands' valuations differ by exactly
``m + 2s > 0``, so no ultrametric cancellation ever occurs and

    rows of M_t:  top -s(t-1),  bottom -st
    rows of N_j:  top -(s+m)j,  bottom -(s+m)(j-1)     (N_j the backward product)

Lemma 2 (invariant cones).  The minimal vector u of ``H_t^F`` lies in the
unstable eigenvector cone and w of ``H_t^B`` in the stable one, with *exact*
valuation gaps

    v(u_x) - v(u_y) = s          v(w_y) - v(w_x) = s + m

whence v(det[u|w]) = v(u) + v(w) exactly (the cross term is smaller by
``2s + m``), i.e. transversality defect 0 with margin ``2s + m``.

The remaining tests pin the identities the write-up (``NOTE.md``) derives *from*
those two lemmas and then relies on, so that none of them is asserted on the
strength of an argument alone:

Lemma 3 (projections and axis exponents).  Read off the rows of Lemma 1,

    H_t^F:  x-projection k - s(t-1),        y-projection k - st
            vertical exponent  kappa_t = k + mt + s(t-1)
    H_t^B:  x-projection k - (s+m)j,        y-projection k - (s+m)(j-1)
            horizontal exponent lambda_j = k + sj - s - m       (j = T - t)

where ``{0} x p^kappa Z_p = H cap (vertical axis)`` and likewise horizontally.

Lemma 4 (the common frame).  ``(p^-A u, p^-B w)`` is a ``Z_p``-basis of
``Z_p^2`` (A, B the two ``d1``) and *both* lattices are diagonal in it, so the
intersection takes the max exponent per direction and the tent follows.

Lemma 5 (exactness thresholds -- Theorem B).  The forward linearisation is
exact at the step ``t -> t+1`` iff ``t <= (k-m)/(3s+m)``, and the backward one
at ``j -> j+1`` iff ``j <= k/(3s+2m)``.  Both are equalities, not bounds: the
tracked ``first_inflation`` is exactly the failing index.  Note the backward
rate is ``3s+2m``, *not* ``3s+m``.
"""

from __future__ import annotations

import random
from fractions import Fraction as F

import pytest

from padic_filtering.henon import (Henon, fixed_point_c, periodic_orbit,
                                   truth_orbit)
from padic_filtering.lattice import Lattice
from padic_filtering.padic import vp

REGIMES = [
    (3, F(1, 3), 1),   # s=1, m=0
    (3, F(1, 3), 3),   # s=1, m=1
    (3, F(1, 3), 9),   # s=1, m=2  -- separates 3s+m from 3s+2m
    (3, F(1, 9), 1),   # s=2, m=0
    (3, F(1, 9), 3),   # s=2, m=1
    (5, F(1, 5), 1),   # p=5
]
L, T, PREC = 40, 16, 200


def _orbit(p, alpha, delta):
    hen = Henon.from_c(p, fixed_point_c(p, delta, alpha), delta)
    rng = random.Random(7)
    itinerary = "".join(rng.choice("+-") for _ in range(L))
    X = periodic_orbit(hen, itinerary, PREC)
    return hen, truth_orbit(hen, X, T, PREC)


def _mul(A, B):
    return [[A[0][0] * B[0][0] + A[0][1] * B[1][0],
             A[0][0] * B[0][1] + A[0][1] * B[1][1]],
            [A[1][0] * B[0][0] + A[1][1] * B[1][0],
             A[1][0] * B[0][1] + A[1][1] * B[1][1]]]


def _vrow(row, p):
    return min(vp(x, p) for x in row)


@pytest.mark.parametrize("p,alpha,delta", REGIMES)
def test_row_valuation_induction(p, alpha, delta):
    hen, truth = _orbit(p, alpha, delta)
    s, m = hen.s, hen.m
    M = [[F(1), F(0)], [F(0), F(1)]]
    for t in range(1, T + 1):
        M = _mul(hen.jacobian(truth[t - 1]), M)
        assert _vrow(M[0], p) == -s * (t - 1)
        assert _vrow(M[1], p) == -s * t
    N = [[F(1), F(0)], [F(0), F(1)]]
    for j in range(1, T + 1):
        N = _mul(hen.jacobian_inv(truth[T - j + 1]), N)
        assert _vrow(N[0], p) == -(s + m) * j
        assert _vrow(N[1], p) == -(s + m) * (j - 1)


@pytest.mark.parametrize("p,alpha,delta", REGIMES)
def test_minimal_vectors_sit_in_eigen_cones(p, alpha, delta):
    hen, truth = _orbit(p, alpha, delta)
    s, m = hen.s, hen.m
    fwd = [Lattice.ball(p, 0)]
    for t in range(T):
        fwd.append(fwd[-1].image(hen.jacobian(truth[t])))
    bwd = [Lattice.ball(p, 0)]
    for t in range(T - 1, -1, -1):
        bwd.append(bwd[-1].image(hen.jacobian_inv(truth[t + 1])))
    bwd = bwd[::-1]
    # t=0 (fwd) and t=T (bwd) are balls with no distinguished direction
    for t in range(1, T + 1):
        u = fwd[t].min_vector()
        assert vp(u[0], p) - vp(u[1], p) == s, f"fwd cone broken at t={t}"
    for t in range(T):
        w = bwd[t].min_vector()
        assert vp(w[1], p) - vp(w[0], p) == s + m, f"bwd cone broken at t={t}"
    # the two gaps force defect 0: det[u|w] = u_x w_y - u_y w_x, and the
    # first product is smaller than the second by exactly 2s + m
    for t in range(1, T):
        u, w = fwd[t].min_vector(), bwd[t].min_vector()
        det = u[0] * w[1] - u[1] * w[0]
        assert vp(det, p) == fwd[t].d1 + bwd[t].d1, f"defect != 0 at t={t}"


# --------------------------------------------------------------- Lemma 3


def _passes(hen, truth, k, T):
    """Idealised (pure Jacobian product) forward and backward lattices."""
    fwd = [Lattice.ball(hen.p, k)]
    for t in range(T):
        fwd.append(fwd[-1].image(hen.jacobian(truth[t])))
    bwd = [Lattice.ball(hen.p, k)]
    for t in range(T - 1, -1, -1):
        bwd.append(bwd[-1].image(hen.jacobian_inv(truth[t + 1])))
    return fwd, bwd[::-1]


@pytest.mark.parametrize("p,alpha,delta", REGIMES)
def test_projection_and_axis_exponents(p, alpha, delta):
    """The four projections and the two axis exponents, exactly."""
    hen, truth = _orbit(p, alpha, delta)
    s, m, k = hen.s, hen.m, 5
    fwd, bwd = _passes(hen, truth, k, T)
    for t in range(T + 1):
        H = fwd[t]
        xp, yp = H.x_projection_exponent(), H.y_projection_exponent()
        assert xp == k - s * max(t - 1, 0), f"fwd x-proj at t={t}"
        assert yp == k - s * t, f"fwd y-proj at t={t}"
        # H cap ({0} x Q_p) = {0} x p^kappa Z_p, and kappa = v(det H) - x-proj
        assert H.det_valuation() - xp == k + m * t + s * max(t - 1, 0)
    for t in range(T + 1):
        j, H = T - t, bwd[t]
        xp, yp = H.x_projection_exponent(), H.y_projection_exponent()
        assert xp == k - (s + m) * j, f"bwd x-proj at j={j}"
        assert yp == k - (s + m) * max(j - 1, 0), f"bwd y-proj at j={j}"
        assert H.det_valuation() - yp == k + s * j - (s + m if j else 0)


# --------------------------------------------------------------- Lemma 4


@pytest.mark.parametrize("p,alpha,delta", REGIMES)
def test_common_frame_diagonalises_both_lattices(p, alpha, delta):
    """``(p^-A u, p^-B w)`` is unimodular and *both* lattices are diagonal in it.

    This is the whole content of the intersection step: once both Hermite forms
    are diagonal in one frame, ``H^F cap H^B`` takes the max exponent per
    direction, which is the tent.
    """
    hen, truth = _orbit(p, alpha, delta)
    s, m, k = hen.s, hen.m, 5
    fwd, bwd = _passes(hen, truth, k, T)
    for t in range(1, T):
        HF, HB = fwd[t], bwd[t]
        A, B = HF.d1, HB.d1
        u, w = HF.min_vector(), HB.min_vector()
        ut = tuple(x * F(p) ** (-A) for x in u)
        wt = tuple(x * F(p) ** (-B) for x in w)
        # unimodular frame: both primitive, determinant a unit
        assert min(vp(x, p) for x in ut) == 0 and min(vp(x, p) for x in wt) == 0
        assert vp(ut[0] * wt[1] - ut[1] * wt[0], p) == 0

        def diag(e1, e2):
            return Lattice.from_columns(
                [tuple(F(p) ** e1 * x for x in ut),
                 tuple(F(p) ** e2 * x for x in wt)], p)

        assert diag(A, HF.d2) == HF, f"H^F not diagonal in the frame at t={t}"
        assert diag(HB.d2, B) == HB, f"H^B not diagonal in the frame at t={t}"
        H = HF.intersect(HB)
        assert (H.d1, H.d2) == (k + min((s + m) * t, s * (T - t)),
                                k + max((s + m) * t, s * (T - t))), f"tent at t={t}"


# --------------------------------------------------------------- Lemma 5


@pytest.mark.parametrize("p,alpha,delta", REGIMES)
@pytest.mark.parametrize("k", [6, 12])
def test_exactness_thresholds(p, alpha, delta, k):
    """Forward horizon ``(k-m)/(3s+m)``, backward horizon ``k/(3s+2m)``, exactly.

    ``first_inflation`` is the *time* of the first non-exact step, i.e. the
    destination index; the last exact source index is therefore two less.
    """
    from padic_filtering.henon import truth_precision
    from padic_filtering.precision import perturb, run_backward, run_forward

    hen = Henon.from_c(p, fixed_point_c(p, delta, alpha), delta)
    s, m = hen.s, hen.m
    rng = random.Random(7)
    itinerary = "".join(rng.choice("+-") for _ in range(L))
    prec = 400
    X = periodic_orbit(hen, itinerary, prec)
    tp = truth_precision(hen, prec)
    n = 4 * k + 20
    truth = truth_orbit(hen, X, n, prec)
    H0 = Lattice.ball(p, k)

    fwd = run_forward(hen, perturb(truth[0], H0, random.Random(1)), H0, truth, n, tp)
    assert fwd.first_inflation == (k - m) // (3 * s + m) + 2

    bwd = run_backward(hen, perturb(truth[n], H0, random.Random(1)), H0, truth, n, tp)
    assert n - bwd.first_inflation == k // (3 * s + 2 * m) + 2


@pytest.mark.parametrize("p,alpha,delta", REGIMES)
def test_certified_equals_idealised_under_budget(p, alpha, delta):
    """Under ``k >= (3s+2m)(T-1)`` no step inflates, so certified == idealised.

    This is Theorem B: the budget buys the *hypothesis* of Theorem A for the
    lattices the tracker actually computes, and nothing else is needed.
    """
    from padic_filtering.henon import truth_precision
    from padic_filtering.precision import (perturb, run_backward, run_forward,
                                           run_smoother)

    hen = Henon.from_c(p, fixed_point_c(p, delta, alpha), delta)
    s, m = hen.s, hen.m
    win = 6
    k = (3 * s + 2 * m) * (win - 1)
    prec = 400
    rng = random.Random(7)
    itinerary = "".join(rng.choice("+-") for _ in range(L))
    X = periodic_orbit(hen, itinerary, prec)
    tp = truth_precision(hen, prec)
    truth = truth_orbit(hen, X, win, prec)
    H0 = Lattice.ball(p, k)

    fwd = run_forward(hen, perturb(truth[0], H0, random.Random(1)), H0, truth, win, tp)
    bwd = run_backward(hen, perturb(truth[win], H0, random.Random(2)), H0,
                       truth, win, tp)
    assert fwd.first_inflation is None and bwd.first_inflation is None
    ideal_f, ideal_b = _passes(hen, truth, k, win)
    for t in range(win + 1):
        assert fwd.at(t)[1] == ideal_f[t], f"certified fwd != idealised at t={t}"
        assert bwd.at(t)[1] == ideal_b[t], f"certified bwd != idealised at t={t}"
    sm = run_smoother(fwd, bwd, truth, tp)
    for t, rec in zip(sm.times, sm.records):
        assert rec.d1 == k + min((s + m) * t, s * (win - t)), f"tent at t={t}"
        assert rec.d2 == k + max((s + m) * t, s * (win - t))


def test_the_3s_plus_m_budget_is_insufficient_for_m_positive():
    """Counterexample to the budget ``k >= (3s+m)T + s + m`` of THEOREM.md §3.

    ``3s+m`` is the *forward* rate; the backward pass pays ``3s+2m`` because its
    quadratic remainder is divided by ``delta``.  At ``s=1, m=2, T=10`` the
    stated budget gives ``k=53`` and the backward pass inflates immediately,
    while ``(3s+2m)(T-1) = 63`` is clean.  See NOTE.md Theorem B'.
    """
    from padic_filtering.henon import truth_precision
    from padic_filtering.precision import perturb, run_backward, run_forward

    p, s, m, win = 3, 1, 2, 10
    hen = Henon.from_c(p, fixed_point_c(p, 9, F(1, 3)), 9)
    assert (hen.s, hen.m) == (s, m)
    prec = 600
    rng = random.Random(7)
    X = periodic_orbit(hen, "".join(rng.choice("+-") for _ in range(L)), prec)
    tp = truth_precision(hen, prec)
    truth = truth_orbit(hen, X, win, prec)

    def run(k):
        H0 = Lattice.ball(p, k)
        return (run_forward(hen, perturb(truth[0], H0, random.Random(1)), H0,
                            truth, win, tp).first_inflation,
                run_backward(hen, perturb(truth[win], H0, random.Random(2)), H0,
                             truth, win, tp).first_inflation)

    assert run((3 * s + m) * win + s + m) == (None, 1)   # k = 53: backward fails
    assert run((3 * s + 2 * m) * (win - 1)) == (None, None)   # k = 63: clean


@pytest.mark.parametrize("p,alpha,delta", REGIMES)
def test_tracked_representatives_stay_on_the_shell(p, alpha, delta):
    """``v(y) = -s`` and ``v(x) = -s`` hold for the *tracker's* representatives.

    The Jacobians in both passes are evaluated at the tracked point, not at the
    truth, so the lemmas need the shell condition there.  It survives because
    the representative differs from the truth by an element of a lattice with
    ``d1 > -s`` -- which is exactly the precision floor.
    """
    from padic_filtering.henon import truth_precision
    from padic_filtering.precision import perturb, run_backward, run_forward

    hen = Henon.from_c(p, fixed_point_c(p, delta, alpha), delta)
    s, m = hen.s, hen.m
    win = 6
    k = (3 * s + 2 * m) * (win - 1)
    prec = 400
    rng = random.Random(7)
    X = periodic_orbit(hen, "".join(rng.choice("+-") for _ in range(L)), prec)
    tp = truth_precision(hen, prec)
    truth = truth_orbit(hen, X, win, prec)
    H0 = Lattice.ball(p, k)
    for track in (run_forward(hen, perturb(truth[0], H0, random.Random(1)), H0,
                              truth, win, tp),
                  run_backward(hen, perturb(truth[win], H0, random.Random(2)), H0,
                               truth, win, tp)):
        for (v, H), t in zip(track.states, track.times):
            assert H.d1 > -s, f"below the floor at t={t}"
            assert vp(v[0], p) == -s and vp(v[1], p) == -s, f"off the shell at t={t}"


@pytest.mark.parametrize("p,alpha,delta", REGIMES)
@pytest.mark.parametrize("win", [4, 6, 9])
def test_budget_threshold_is_an_iff(p, alpha, delta, win):
    """``k = (3s+2m)(W-1)`` keeps both passes exact; ``k-1`` does not.

    Theorem B' is an *iff*, and the ``k-1`` direction is the half nothing in the
    repo used to check: an under-provisioned run still satisfies ``C == 0``,
    because inflation at the far end of a pass lands where the *other* arm of
    the tent is binding (docs/DEVELOPMENT.md).  So the threshold has to be pinned on
    ``first_inflation``, which is what this asserts.

    For ``m > 0`` the backward pass is the binding one: NOTE.md §4.4 Step 2,
    ``(3s+2m)(W-1) - [(3s+m)(W-1) + m] = m(W-2) >= 0``.
    """
    from padic_filtering.henon import truth_precision
    from padic_filtering.precision import perturb, run_backward, run_forward

    hen = Henon.from_c(p, fixed_point_c(p, delta, alpha), delta)
    s, m = hen.s, hen.m
    k = (3 * s + 2 * m) * (win - 1)
    prec = 2 * (k + win * (s + m)) + 300
    rng = random.Random(7)
    X = periodic_orbit(hen, "".join(rng.choice("+-") for _ in range(L)), prec)
    tp = truth_precision(hen, prec)
    truth = truth_orbit(hen, X, win, prec)

    def inflations(kk):
        H0 = Lattice.ball(p, kk)
        return (run_forward(hen, perturb(truth[0], H0, random.Random(1)), H0,
                            truth, win, tp).first_inflation,
                run_backward(hen, perturb(truth[win], H0, random.Random(2)), H0,
                             truth, win, tp).first_inflation)

    assert inflations(k) == (None, None), f"sharp budget k={k} should be clean"
    below = inflations(k - 1)
    assert below != (None, None), f"k={k - 1} is below the budget yet clean"
    if m > 0:
        assert below[1] is not None, "for m > 0 the backward pass must bind first"


@pytest.mark.parametrize("k", [6, 9, 12, 24])
def test_prop_4_4_also_holds_in_the_attractor_region(k):
    """Empirical extension: Prop 4.4's horizons hold verbatim at ``s = 0``.

    NOTE.md's standing hypotheses put ``s > 0`` (the horseshoe), but its §4
    computation only consumes the divisor law of Cor 3.2, which in ``H+_II``
    reads ``d1(H_t^F) = k``, ``d2 = k + mt`` -- i.e. Cor 3.2 with ``s = 0``.
    So the same thresholds should apply, and they do.  This is measured, not
    proved; `exp_5_2_horizon.py` relies on it, so it is pinned here.
    """
    from padic_filtering.params import ATTRACTOR_3ADIC
    from padic_filtering.precision import perturb, run_backward, run_forward

    hen = ATTRACTOR_3ADIC.henon()
    s, m = hen.s, hen.m
    assert (hen.region, s) == ("H+_II", 0)
    n = 4 * k + 20
    truth = hen.orbit((F(1), F(1)), n, prec=400)
    H0 = Lattice.ball(hen.p, k)

    fwd = run_forward(hen, perturb(truth[0], H0, random.Random(1)), H0, truth, n)
    assert fwd.first_inflation == (k - m) // (3 * s + m) + 2

    bwd = run_backward(hen, perturb(truth[n], H0, random.Random(2)), H0, truth, n)
    assert n - bwd.first_inflation == k // (3 * s + 2 * m) + 2
