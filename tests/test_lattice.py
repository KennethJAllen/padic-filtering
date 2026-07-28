import random
from fractions import Fraction as F

import pytest

from padic_filtering.lattice import (Lattice, intersect_cosets, mat_mul,
                                     solve_in_span)
from padic_filtering.padic import ppow, vp

P = 3


def rand_lattice(rng, p=P, span=4):
    while True:
        cols = [(F(rng.randint(-27, 27), rng.choice([1, 2, 4, 5])) * ppow(p, rng.randint(-span, span)),
                 F(rng.randint(-27, 27), rng.choice([1, 2, 4, 5])) * ppow(p, rng.randint(-span, span)))
                for _ in range(2)]
        try:
            return Lattice.from_columns(cols, p)
        except ValueError:
            continue


def test_hnf_roundtrip_is_idempotent():
    rng = random.Random(1)
    for _ in range(200):
        L = rand_lattice(rng)
        assert Lattice.from_columns(L.columns(), P) == L


def test_basis_generates_itself():
    rng = random.Random(2)
    for _ in range(200):
        L = rand_lattice(rng)
        for c in L.columns():
            assert L.contains(c)
        # a generic non-member
        assert not L.contains((L.columns()[0][0] * F(1, P), F(0)))


def test_intersect_idempotent_and_contained():
    rng = random.Random(3)
    for _ in range(200):
        A, B = rand_lattice(rng), rand_lattice(rng)
        assert A.intersect(A) == A
        C = A.intersect(B)
        for c in C.columns():
            assert A.contains(c) and B.contains(c)
        # index multiplicativity: [A:C][B:C] accounted by det valuations
        S = A + B
        assert C.det_valuation() + S.det_valuation() == A.det_valuation() + B.det_valuation()


def test_sum_contains_both():
    rng = random.Random(4)
    for _ in range(100):
        A, B = rand_lattice(rng), rand_lattice(rng)
        S = A + B
        for c in A.columns() + B.columns():
            assert S.contains(c)


def test_dual_involution():
    rng = random.Random(5)
    for _ in range(200):
        L = rand_lattice(rng)
        assert L.dual().dual() == L


def test_known_small_examples():
    # p Z x Z: divisors (0,1)
    L = Lattice.from_columns([(3, 0), (0, 1)], 3)
    assert L.elementary_divisors() == (0, 1)
    assert L.det_valuation() == 1
    assert L.anisotropy() == 1
    # a ball is isotropic
    B = Lattice.ball(3, 2)
    assert B.elementary_divisors() == (2, 2) and B.anisotropy() == 0
    # negative d1: the H_III case, outside Z_p^2
    N = Lattice.from_columns([(F(1, 9), 0), (0, 9)], 3)
    assert N.elementary_divisors() == (-2, 2)
    assert N.det_valuation() == 0
    assert not N.contains((F(1, 27), 0)) and N.contains((F(1, 9), 0))


def test_elementary_divisors_match_snf_of_product():
    # J = [[0,1],[-delta,2y]] repeatedly applied, as in review_checks
    rng = random.Random(6)
    for _ in range(50):
        L = rand_lattice(rng)
        A = ((F(0), F(1)), (F(-3), F(rng.randint(-9, 9))))
        if A[1][0] * A[0][1] == 0:
            continue
        M = L.image(A)
        assert M.det_valuation() == L.det_valuation() + vp(3, 3) * 1


def test_image_composes():
    rng = random.Random(7)
    A = ((F(0), F(1)), (F(-3), F(2)))
    B = ((F(2), F(1)), (F(1), F(1)))
    for _ in range(100):
        L = rand_lattice(rng)
        assert L.image(A).image(B) == L.image(mat_mul(B, A))


def test_reduce_vector_stays_in_coset_and_is_canonical():
    rng = random.Random(8)
    for _ in range(200):
        L = rand_lattice(rng)
        v = (F(rng.randint(-100, 100), 7), F(rng.randint(-100, 100), 5))
        r = L.reduce_vector(v)
        assert L.contains((v[0] - r[0], v[1] - r[1]))
        assert L.reduce_vector(r) == r
        # any two members of a coset reduce to the same representative
        c = L.columns()[0]
        assert L.reduce_vector((v[0] + c[0], v[1] + c[1])) == r


def test_solve_in_span():
    rng = random.Random(9)
    for _ in range(100):
        L = rand_lattice(rng)
        cols = L.columns()
        z = (rng.randint(-9, 9), rng.randint(-9, 9))
        w = (z[0] * cols[0][0] + z[1] * cols[1][0], z[0] * cols[0][1] + z[1] * cols[1][1])
        got = solve_in_span(cols, w, P)
        assert got is not None
        assert (got[0] * cols[0][0] + got[1] * cols[1][0],
                got[0] * cols[0][1] + got[1] * cols[1][1]) == w
    # a vector outside the span is rejected
    L = Lattice.ball(3, 1)
    assert solve_in_span(L.columns(), (F(1), F(0)), 3) is None


def test_intersect_cosets():
    rng = random.Random(10)
    for _ in range(200):
        A, B = rand_lattice(rng), rand_lattice(rng)
        base = (F(rng.randint(-50, 50), 7), F(rng.randint(-50, 50), 11))
        ca, cb = A.columns()[0], B.columns()[1]
        v1 = (base[0] + ca[0], base[1] + ca[1])
        v2 = (base[0] + cb[0], base[1] + cb[1])
        got = intersect_cosets(v1, A, v2, B)
        assert got is not None, "cosets share `base`, so they cannot be disjoint"
        v, H = got
        assert H == A.intersect(B)
        assert A.contains((v[0] - v1[0], v[1] - v1[1]))
        assert B.contains((v[0] - v2[0], v[1] - v2[1]))


def test_disjoint_cosets_detected():
    A = Lattice.ball(3, 1)
    got = intersect_cosets((F(0), F(0)), A, (F(1), F(0)), A)
    assert got is None


def test_review_checks_H_III_numbers():
    """Reproduce the ``review_checks.py`` horseshoe numbers."""
    p, T, delta, alpha = 3, 12, F(1), F(1, 3)
    J = ((F(0), F(1)), (-delta, 2 * alpha))
    Ji = ((2 * alpha / delta, -1 / delta), (F(1), F(0)))
    Fwd, Bwd = Lattice.identity(p), Lattice.identity(p)
    fwd_list, bwd_list = [Fwd], [Bwd]
    for _ in range(T):
        Fwd, Bwd = Fwd.image(J), Bwd.image(Ji)
        fwd_list.append(Fwd)
        bwd_list.append(Bwd)
    for t in range(T + 1):
        assert fwd_list[t].elementary_divisors() == (-t, t)
        lo = min(t, T - t)
        assert fwd_list[t].intersect(bwd_list[T - t]).elementary_divisors() == (lo, T - lo)


def test_min_vector_attains_d1():
    rng = random.Random(11)
    for _ in range(200):
        L = rand_lattice(rng)
        v = L.min_vector()
        assert L.contains(v)
        assert min(vp(v[0], P), vp(v[1], P)) == L.d1


def test_transversality_defect():
    from padic_filtering.lattice import transversality_defect
    p = 3
    # a ball has no distinguished worst direction: undefined
    assert transversality_defect(Lattice.ball(p, 2), Lattice.ball(p, 0)) is None
    # orthogonal thin lattices: defect 0
    A = Lattice.from_columns([(1, 0), (0, F(1, 9))], p)   # precise in x, loose in y
    B = Lattice.from_columns([(F(1, 9), 0), (0, 1)], p)   # loose in x, precise in y
    assert A.d1 != A.d2 and B.d1 != B.d2
    assert transversality_defect(A, B) == 0
    # parallel worst directions: positive defect (they are not transverse)
    assert transversality_defect(A, A) > 0
