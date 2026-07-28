from fractions import Fraction as F

import pytest

from padic_filtering.padic import (INFINITY, exact_div_ppow, inv_mod, mod_pk,
                                   sqrt_zp, unit_part, vp, zp_int)


def test_vp_of_zero_is_infinity():
    assert vp(0, 3) == INFINITY
    assert vp(F(0), 3) == INFINITY
    # and it compares correctly against every exponent
    assert vp(0, 3) > 10**9


def test_vp_integers_and_fractions():
    assert vp(9, 3) == 2
    assert vp(-9, 3) == 2
    assert vp(F(1, 3), 3) == -1
    assert vp(F(5, 9), 3) == -2
    assert vp(F(4, 5), 3) == 0


def test_unit_part():
    assert unit_part(F(5, 9), 3) == 5
    assert vp(unit_part(18, 3), 3) == 0
    with pytest.raises(ZeroDivisionError):
        unit_part(0, 3)


def test_division_by_non_unit_raises():
    with pytest.raises(ZeroDivisionError):
        inv_mod(3, 3, 5)
    with pytest.raises(ZeroDivisionError):
        zp_int(F(1, 3), 3, 5)


def test_zp_int_roundtrip():
    x = F(2, 5)
    r = zp_int(x, 3, 6)
    assert vp(x - r, 3) >= 6
    assert 0 <= r < 3**6


def test_mod_pk_is_a_representative():
    for x in [F(5, 9), F(2, 5), F(0), 17, F(-4, 7)]:
        r = mod_pk(x, 3, 6)
        assert vp(x - r, 3) >= 6
    assert mod_pk(F(1, 3), 3, -1) == 0  # already inside p^k Z_p


def test_sqrt_zp():
    r = sqrt_zp(-5, 3, 20)
    assert (r * r + 5) % 3**20 == 0
    with pytest.raises(ValueError):
        sqrt_zp(2, 3, 10)  # 2 is not a square mod 3


def test_exact_div_ppow_raises_when_inexact():
    assert exact_div_ppow(9, 3, 2) == 1
    with pytest.raises(ValueError):
        exact_div_ppow(1, 3, 1)
