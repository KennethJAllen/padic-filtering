from fractions import Fraction as F

import pytest

from padic_filtering.henon import (MAX_REPR_BITS, Henon, fixed_point_c,
                                   periodic_orbit, truth_orbit)
from padic_filtering.lattice import mat_mul
from padic_filtering.padic import vp
from padic_filtering.params import ATTRACTOR_3ADIC, HORSESHOE_3ADIC


def test_regions_match_the_ADP_dictionary():
    # H+_II: v_p(c) >= 0, m >= 1  (attractor)
    att = ATTRACTOR_3ADIC.henon()
    assert att.region == "H+_II" and att.m == 1 and att.s == 0
    assert att.eigenvalue_valuations == (0, 1)
    # H_III: |c| > max(1, |delta|^2)  (horseshoe)
    hs = HORSESHOE_3ADIC.henon()
    assert hs.region == "H_III" and hs.s == 1 and hs.m == 0
    assert hs.eigenvalue_valuations == (-1, 1)
    # H_I: unimodular, a no-op
    assert Henon.from_c(3, 1, 1).region == "H_I"


def test_check_regime_rejects_empty_julia_set():
    # ADP Thm 1(a): J is empty unless a = -c is a square in Q_p
    bad = Henon.from_c(3, F(2, 9), 1)  # -c = -2/9, unit part -2 = 1 mod 3 -> square
    assert bad.is_square_a()
    worse = Henon.from_c(3, F(1, 9), 1)  # -c = -1/9, unit part -1 = 2 mod 3 -> not
    assert not worse.is_square_a()
    with pytest.raises(AssertionError):
        worse.check_regime()


def test_f_inv_inverts_f():
    for hen, v in [(HORSESHOE_3ADIC.henon(), (F(1), F(1))),
                   (ATTRACTOR_3ADIC.henon(), (F(1), F(2)))]:
        assert hen.f_inv(hen.f(v)) == v
        assert hen.f(hen.f_inv(v)) == v


def test_det_jacobian_is_constant():
    for hen in [HORSESHOE_3ADIC.henon(), ATTRACTOR_3ADIC.henon()]:
        v = (F(1), F(1))
        for _ in range(6):
            J = hen.jacobian(v)
            assert J[0][0] * J[1][1] - J[0][1] * J[1][0] == hen.delta
            Ji = hen.jacobian_inv(v)
            assert Ji[0][0] * Ji[1][1] - Ji[0][1] * Ji[1][0] == 1 / hen.delta
            # jacobian_inv at v really is the inverse of the jacobian at f^-1(v)
            prod = mat_mul(hen.jacobian(hen.f_inv(v)), Ji)
            assert prod == ((F(1), F(0)), (F(0), F(1)))
            v = hen.f(v)


def test_fixed_point():
    c = fixed_point_c(3, 1, F(1, 3))
    assert c == F(5, 9)
    hen = Henon.from_c(3, c, 1)
    assert hen.f((F(1, 3), F(1, 3))) == (F(1, 3), F(1, 3))
    assert hen.f_scaled((F(1), F(1))) == (F(1), F(1))  # scaled: X = p^s x
    assert hen.s == 1 and vp(c, 3) == -2


def test_horseshoe_points_have_unit_scaled_coordinates():
    hen = HORSESHOE_3ADIC.henon()
    X = periodic_orbit(hen, "++-+--+-", 30)
    assert all(x % hen.p != 0 for x in X), "on the horseshoe |x| = p^s exactly"


@pytest.mark.parametrize("itinerary", ["+", "+-", "++-+--+-", "+-+--"])
def test_newton_periodic_orbits_are_genuinely_periodic(itinerary):
    hen = HORSESHOE_3ADIC.henon()
    prec = 40
    X = periodic_orbit(hen, itinerary, prec)
    n = len(X)
    mod = hen.p**prec
    # verify by exact forward iteration of the map itself, not by the solver
    orbit = truth_orbit(hen, X, n, prec)
    v = orbit[0]
    for t in range(n):
        v = hen.f(v)
        assert vp(v[0] - orbit[t + 1][0], hen.p) >= prec - (t + 2) * hen.s
    assert vp(v[0] - orbit[0][0], hen.p) >= prec - (n + 1) * hen.s


def test_distinct_itineraries_give_distinct_orbits():
    hen = HORSESHOE_3ADIC.henon()
    seen = {tuple(periodic_orbit(hen, it, 20)) for it in ["++", "+-", "-+", "--"]}
    assert len(seen) == 4


def test_guard_rejects_unreduced_iterates():
    hen = HORSESHOE_3ADIC.henon()
    huge = F(1 << (MAX_REPR_BITS + 1))
    with pytest.raises(MemoryError):
        hen.f((huge, huge))
    with pytest.raises(MemoryError):
        hen.f_inv((huge, huge))


def test_attractor_orbit_stays_integral():
    hen = ATTRACTOR_3ADIC.henon()
    v = (F(1), F(1))
    for _ in range(20):
        v = hen.f(v)
        assert vp(v[0], 3) >= 0 and vp(v[1], 3) >= 0
