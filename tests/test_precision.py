import random
from fractions import Fraction as F

import pytest

from padic_filtering.henon import (Henon, periodic_orbit, truth_orbit,
                                   truth_precision)
from padic_filtering.lattice import Lattice
from padic_filtering.params import (ATTRACTOR_3ADIC, DEFAULT_ITINERARY,
                                    HORSESHOE_3ADIC)
from padic_filtering.precision import (CertificationError, PrecisionExhausted,
                                       Track, certify, default_floor,
                                       is_linearisation_exact, naive_track,
                                       perturb, propagate, run_backward,
                                       run_forward, run_oracle, run_smoother)


def horseshoe_fixed_point_setup(T, k):
    hen = HORSESHOE_3ADIC.henon()
    hen.check_regime()
    truth = [(F(1, 3), F(1, 3))] * (T + 1)   # the fixed point alpha = 1/3
    return hen, truth, Lattice.ball(hen.p, k)


def test_forward_loses_s_digits_per_step():
    T, k = 12, 60
    hen, truth, H0 = horseshoe_fixed_point_setup(T, k)
    fwd = run_forward(hen, perturb(truth[0], H0, random.Random(0)), H0, truth, T)
    assert fwd.d1() == [k - t * hen.s for t in range(T + 1)]
    assert fwd.d2() == [k + t * (hen.s + hen.m) for t in range(T + 1)]
    # det H changes by exactly m per step: this is det J = delta, an assertion
    assert [r.lattice_digits for r in fwd.records] == [2 * k + t * hen.m for t in range(T + 1)]


def test_smoother_gives_the_bounded_tent():
    T, k = 12, 60
    hen, truth, H0 = horseshoe_fixed_point_setup(T, k)
    rng = random.Random(1)
    fwd = run_forward(hen, perturb(truth[0], H0, rng), H0, truth, T)
    bwd = run_backward(hen, perturb(truth[T], H0, rng), H0, truth, T)
    sm = run_smoother(fwd, bwd, truth)
    assert sm.d1() == [k + min(t, T - t) * hen.s for t in range(T + 1)]
    # forward-only falls linearly; smoothed never drops below the endpoints
    assert min(sm.d1()) == k
    assert fwd.d1()[-1] == k - T * hen.s


def test_smoother_is_vacuous_in_the_attractor_regime():
    """In H+_II the forward pass never loses, so
    F ∩ B = F identically and the smoother has no content."""
    T, k = 15, 40
    hen = ATTRACTOR_3ADIC.henon()
    assert hen.region == "H+_II"
    truth = hen.orbit((F(1), F(1)), T)
    H0 = Lattice.ball(hen.p, k)
    rng = random.Random(2)
    fwd = run_forward(hen, perturb(truth[0], H0, rng), H0, truth, T)
    bwd = run_backward(hen, perturb(truth[T], H0, rng), H0, truth, T)
    # forward gains m digits per step in the contracting direction, loses none
    assert fwd.d1() == [k] * (T + 1)
    assert fwd.d2() == [k + t * hen.m for t in range(T + 1)]
    # backward only loses
    assert bwd.d1() == [k - (T - t) * hen.m for t in range(T + 1)]
    sm = run_smoother(fwd, bwd, truth)
    for t in range(T + 1):
        assert sm.states[t][1] == fwd.states[t][1], "F ∩ B must equal F in H+_II"


def test_naive_is_flat_in_the_attractor_and_falls_in_the_horseshoe():
    att = ATTRACTOR_3ADIC.henon()
    assert naive_track(att, 20, 10) == [20] * 11          # constant: nothing lost
    hs = HORSESHOE_3ADIC.henon()
    assert naive_track(hs, 20, 10) == [20 - t for t in range(11)]


def test_certification_detects_a_wrong_claim():
    T, k = 8, 40
    hen, truth, H0 = horseshoe_fixed_point_setup(T, k)
    v0 = perturb(truth[0], H0, random.Random(3))
    assert certify(v0, H0, truth[0])
    # a lattice too small to contain the true offset is rejected
    tiny = Lattice.ball(hen.p, k + 5)
    assert not certify(v0, tiny, truth[0])
    with pytest.raises(CertificationError):
        run_forward(hen, v0, tiny, truth, T)


def test_inflation_only_ever_enlarges_the_lattice():
    T, k = 30, 12
    hen, truth, H0 = horseshoe_fixed_point_setup(T, k)
    fwd = run_forward(hen, perturb(truth[0], H0, random.Random(4)), H0, truth, T)
    assert fwd.first_inflation is not None, "the exactness horizon must be reached"
    assert fwd.exhausted_at is not None, "the precision floor must stop the track"
    for t in range(1, len(fwd.times)):
        v, H = fwd.states[t]
        vprev, Hprev = fwd.states[t - 1]
        J = hen.jacobian(vprev)
        strict = Hprev.image(J)
        for c in strict.columns():
            assert H.contains(c), "inflation must contain the strict image"
    # and the claim stays true throughout -- run_forward already certified it
    assert all(certify(*fwd.states[i], truth[t]) for i, t in enumerate(fwd.times))


def test_exactness_test_agrees_with_propagate():
    T, k = 30, 12
    hen, truth, H0 = horseshoe_fixed_point_setup(T, k)
    v, H = truth[0], H0
    floor = default_floor(hen)
    for t in range(T):
        expect = is_linearisation_exact(hen, v, H)
        try:
            v, H, exact, inflated = propagate(hen, v, H, floor=floor, t=t)
        except PrecisionExhausted:
            break
        assert exact == expect and inflated == (not expect)
    else:
        raise AssertionError("expected the floor to stop this run")


def test_tight_bound_is_never_worse_than_the_d1_bound():
    T, k = 30, 12
    hen, truth, H0 = horseshoe_fixed_point_setup(T, k)
    v, H = truth[0], H0
    floor = default_floor(hen)
    for t in range(T):
        assert (is_linearisation_exact(hen, v, H, tight=True)
                or not is_linearisation_exact(hen, v, H, tight=False))
        try:
            v, H, _, _ = propagate(hen, v, H, floor=floor, t=t)
        except PrecisionExhausted:
            break


def test_oracle_update_restores_precision():
    T, k = 12, 40
    hen, truth, H0 = horseshoe_fixed_point_setup(T, k)
    v0 = perturb(truth[0], H0, random.Random(5))
    plain = run_forward(hen, v0, H0, truth, T)
    with_oracle = run_oracle(hen, v0, H0, truth, T, reveal_k=k, reveal_times=[6])
    assert with_oracle.d1()[6] > plain.d1()[6]
    assert with_oracle.d1()[T] > plain.d1()[T]


def test_smoother_on_a_genuine_periodic_orbit():
    """The tent must survive a variable Jacobian."""
    hen = HORSESHOE_3ADIC.henon()
    T, k, prec = 16, 60, 200
    X = periodic_orbit(hen, DEFAULT_ITINERARY, prec)
    truth = truth_orbit(hen, X, T, prec)
    H0 = Lattice.ball(hen.p, k)
    rng = random.Random(6)
    tp = truth_precision(hen, prec)
    fwd = run_forward(hen, perturb(truth[0], H0, rng), H0, truth, T, tp)
    bwd = run_backward(hen, perturb(truth[T], H0, rng), H0, truth, T, tp)
    sm = run_smoother(fwd, bwd, truth, tp)
    assert fwd.d1() == [k - t * hen.s for t in range(T + 1)]
    assert sm.d1() == [k + min(t, T - t) * hen.s for t in range(T + 1)]
