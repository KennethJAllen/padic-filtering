"""Does the four-divisor law depend on Henon at all?  (NOTE.md §6.1)

NOTE.md §3 proves Theorem A from three ingredients only: the shell condition
``v(x_t) = v(y_t) = -s``, ``v(det J) = m``, and the margin ``m + 2s > 0``.  It
uses them exclusively through **Lemma 3.3**, which says the Jacobians preserve a
pair of cones with exact valuation gains.  If that is really all the proof
consumes, then the law should hold for *any* sequence of matrices with that
property -- no dynamics, no orbit relation between consecutive terms, no
companion structure.

This file tests exactly that, on matrix sequences that are not Henon Jacobians.
The hypotheses tested are, for a fixed ``s > 0``, ``m >= 0`` and the cones

    C^u = { (a,b) : v(a) - v(b) = s },     C^s = { (a,b) : v(b) - v(a) = s+m }

  (H1u)  J_t(C^u) subset C^u  with  v(J_t z) = v(z) - s        for z in C^u
  (H1s)  J_t^-1(C^s) subset C^s  with  v(J_t^-1 z) = v(z) - (s+m)
  (H2)   v(det J_t) = m

and the conclusion tested is the whole of Theorem A, for

    H_t^F = J_{t-1} ... J_0 . p^k Z_p^2      (forward products)
    H_t^B = (J_{T-1} ... J_t)^-1 . p^k Z_p^2 (backward products)
    H_t   = H_t^F cap H_t^B

namely all four divisors as *equalities*, plus transversality defect 0.

Two levels of generality, as separate tests:

  **Level 1, generalised companion.**  ``J_t = [[0, 1], [-delta, beta_t]]`` with
  ``v(beta_t) = -s`` chosen independently at random at each ``t``.  Already
  strictly more general than Henon: there ``beta_t = 2 y_t`` and consecutive
  ``y_t`` lie on one orbit, which is the ``s``-admissibility hypothesis.  Here
  there is no relation between consecutive terms at all.

  **Level 2, fully general.**  ``J_t = Q_t . diag(p^-s, p^(s+m)) . Q_t^-1`` with
  ``Q_t = [u_t | w_t]`` a random unimodular frame whose columns are random
  vectors of ``C^u`` and ``C^s``.  No companion structure: all four entries are
  generic, the eigendirections move at every step, and nothing but the cone
  conditions is imposed.  If this level failed, NOTE.md §6.1 would be wrong and
  the abstraction would have a hidden Henon-specific ingredient.

Both levels assert the hypotheses (H1u), (H1s), (H2) numerically on the
generated matrices before asserting the conclusion, so a failure is never
ambiguous between "bad generator" and "false law".
"""

from __future__ import annotations

import random
from fractions import Fraction as F

import pytest

from padic_filtering.lattice import (Lattice, mat_inv, mat_vec,
                                     transversality_defect)
from padic_filtering.padic import vp

# (p, s, m).  m = 1, 2 are the cases where the two passes are not mirror
# images, and s = 1, 2, 3 exercises the margin m + 2s at its smallest.
REGIMES = [
    (3, 1, 0),
    (3, 1, 1),
    (3, 1, 2),
    (3, 2, 0),
    (5, 2, 1),
    (7, 3, 2),
]
T = 10


# ------------------------------------------------------------- generators


def _unit(rng: random.Random, p: int) -> int:
    """A random integer that is a unit in ``Z_p`` (and small, to keep exact
    arithmetic cheap: the forward product multiplies T of these together)."""
    while True:
        u = rng.randrange(1, p ** 3)
        if u % p:
            return u


def _in_cone_u(rng, p, s):
    """Random ``z`` with ``v(z_x) - v(z_y) = s`` exactly, at a random scale."""
    e = rng.randrange(-2, 3)
    return (F(p) ** (e + s) * _unit(rng, p), F(p) ** e * _unit(rng, p))


def _in_cone_s(rng, p, s, m):
    """Random ``z`` with ``v(z_y) - v(z_x) = s + m`` exactly."""
    e = rng.randrange(-2, 3)
    return (F(p) ** e * _unit(rng, p), F(p) ** (e + s + m) * _unit(rng, p))


def companion_sequence(rng, p, s, m, n):
    """Level 1: ``[[0, 1], [-delta, beta_t]]``, ``v(beta_t) = -s`` at random."""
    delta = F(p) ** m * _unit(rng, p)
    return [((F(0), F(1)),
             (-delta, F(p) ** -s * _unit(rng, p)))
            for _ in range(n)]


def general_sequence(rng, p, s, m, n):
    """Level 2: random cone-preserving matrices with no companion structure.

    ``Q_t = [u_t | w_t]`` with ``u_t in C^u``, ``w_t in C^s``, both normalised
    to valuation 0.  Then ``v(det Q_t) = 0`` -- the two cone terms of the
    determinant are separated by ``2s + m > 0``, which is Lemma 3.5 -- so
    ``Q_t`` is unimodular and ``J_t = Q_t diag(p^-s, p^(s+m)) Q_t^-1`` really
    is a conjugate of the diagonal model rather than a rescaling of it.
    """
    out = []
    for _ in range(n):
        u = (F(p) ** s * _unit(rng, p), F(1) * _unit(rng, p))
        w = (F(1) * _unit(rng, p), F(p) ** (s + m) * _unit(rng, p))
        Q = ((u[0], w[0]), (u[1], w[1]))
        assert vp(Q[0][0] * Q[1][1] - Q[0][1] * Q[1][0], p) == 0, \
            "frame is not unimodular -- the cones failed to be transverse"
        D = ((F(p) ** -s, F(0)), (F(0), F(p) ** (s + m)))
        Qi = mat_inv(Q)
        J = tuple(tuple(sum(Q[i][a] * D[a][b] * Qi[b][j] for a in range(2)
                            for b in range(2)) for j in range(2))
                  for i in range(2))
        # a genuine full matrix, not a rescaled diagonal or companion form
        assert J[0][0] != 0 and J[0][1] != 0 and J[1][0] != 0 and J[1][1] != 0
        out.append(J)
    return out


# ------------------------------------------------------------ hypotheses


def hypotheses_hold(J, p, s, m, rng, trials=6) -> bool:
    """(H1u), (H1s), (H2), tested on random cone vectors.  No proof, a probe."""
    det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
    if det == 0 or vp(det, p) != m:
        return False
    Ji = mat_inv(J)
    for _ in range(trials):
        z = _in_cone_u(rng, p, s)
        Jz = mat_vec(J, z)
        if vp(Jz[0], p) - vp(Jz[1], p) != s:
            return False
        if min(vp(Jz[0], p), vp(Jz[1], p)) != min(vp(z[0], p), vp(z[1], p)) - s:
            return False
        z = _in_cone_s(rng, p, s, m)
        Jiz = mat_vec(Ji, z)
        if vp(Jiz[1], p) - vp(Jiz[0], p) != s + m:
            return False
        if min(vp(Jiz[0], p), vp(Jiz[1], p)) != \
                min(vp(z[0], p), vp(z[1], p)) - (s + m):
            return False
    return True


def assert_hypotheses(J, p, s, m, rng, trials=6):
    """(H1u), (H1s), (H2) -- asserted on the generated matrix, not assumed."""
    assert hypotheses_hold(J, p, s, m, rng, trials), \
        f"generated matrix does not satisfy the cone hypotheses: {J}"


def rejection_sequence(rng, p, s, m, n, tries=6000):
    """Level 3: matrices found by *rejection sampling*, not by construction.

    Levels 1 and 2 both build ``J_t`` from a recipe, so a law that held only
    for those recipes would still pass.  Here each entry is an independent
    random unit times an independent random power of ``p``, and a matrix is
    kept only if it passes ``hypotheses_hold``.  Nothing about its shape is
    imposed: the *filter* is the definition, and it rejects well over 90% of
    proposals.

    The per-entry valuation ranges below are a proposal distribution, not a
    hypothesis.  They are centred on the region where admissible matrices can
    live only so that acceptance is not astronomically rare; widening them
    changes nothing except the running time (checked), because the accepted set
    is cut out by ``hypotheses_hold`` either way.
    """
    ranges = [[(-s - 1, 2 * s + 2 * m + 3), (-1, 1)],
              [(m - 2, m + 2), (-s - 1, -s + 1)]]
    out, seen = [], 0
    for _ in range(tries):
        if len(out) == n:
            break
        seen += 1
        J = tuple(tuple(F(0) if rng.random() < 0.15
                        else F(p) ** rng.randrange(lo, hi + 1) * _unit(rng, p)
                        for lo, hi in row) for row in ranges)
        if hypotheses_hold(J, p, s, m, rng):
            out.append(J)
    assert len(out) == n, \
        f"rejection sampling found only {len(out)}/{n} admissible matrices"
    assert seen > 3 * n, \
        f"acceptance rate {n}/{seen} too high -- the filter is not filtering"
    return out


# ------------------------------------------------------------- the law


def passes(p, Js, k, n):
    """Forward products and backward (inverse-of-tail) products, as lattices."""
    ball = Lattice.ball(p, k)
    fwd = [ball]
    for J in Js:
        fwd.append(fwd[-1].image(J))
    bwd = [None] * (n + 1)
    bwd[n] = ball
    for t in range(n - 1, -1, -1):
        bwd[t] = bwd[t + 1].image(mat_inv(Js[t]))
    return fwd, bwd


def assert_four_divisor_law(p, Js, k, n, s, m):
    fwd, bwd = passes(p, Js, k, n)
    assert [H.d1 for H in fwd] == [k - s * t for t in range(n + 1)]
    assert [H.d2 for H in fwd] == [k + (s + m) * t for t in range(n + 1)]
    assert [H.d1 for H in bwd] == [k - (s + m) * (n - t) for t in range(n + 1)]
    assert [H.d2 for H in bwd] == [k + s * (n - t) for t in range(n + 1)]

    sm = [fwd[t].intersect(bwd[t]) for t in range(n + 1)]
    assert [H.d1 for H in sm] == \
        [k + min((s + m) * t, s * (n - t)) for t in range(n + 1)]
    assert [H.d2 for H in sm] == \
        [k + max((s + m) * t, s * (n - t)) for t in range(n + 1)]

    defects = [d for d in (transversality_defect(fwd[t], bwd[t])
                           for t in range(n + 1)) if d is not None]
    assert defects and set(defects) == {0}, f"transversality defect {defects}"


# ------------------------------------------------------------- the tests


@pytest.mark.parametrize("p,s,m", REGIMES)
@pytest.mark.parametrize("k", [0, 7])
def test_level_1_generalised_companion(p, s, m, k):
    """``[[0,1],[-delta, beta_t]]`` with independent random ``beta_t``.

    Henon is the special case ``beta_t = 2 y_t`` along an orbit; nothing here
    relates consecutive terms, so ``s``-admissibility is genuinely not used.
    """
    rng = random.Random(4093 + p * 100 + s * 10 + m)
    Js = companion_sequence(rng, p, s, m, T)
    for J in Js:
        assert_hypotheses(J, p, s, m, rng)
    assert_four_divisor_law(p, Js, k, T, s, m)


@pytest.mark.parametrize("p,s,m", REGIMES)
@pytest.mark.parametrize("k", [0, 7])
def test_level_2_fully_general_cone_preserving(p, s, m, k):
    """Random cone-preserving matrices, no companion structure at all.

    This is the level that decides whether NOTE.md §6.1's proposed abstraction
    is right: if the law needed anything Henon-specific beyond the cone
    conditions, it would fail here.
    """
    rng = random.Random(7717 + p * 100 + s * 10 + m)
    Js = general_sequence(rng, p, s, m, T)
    for J in Js:
        assert_hypotheses(J, p, s, m, rng)
    assert_four_divisor_law(p, Js, k, T, s, m)


@pytest.mark.parametrize("p,s,m", REGIMES)
@pytest.mark.parametrize("k", [0, 7])
def test_level_3_rejection_sampled(p, s, m, k):
    """Matrices found by rejection sampling, so no construction bias.

    The strongest form available here: nothing about the shape of ``J_t`` is
    imposed, only the cone conditions, and the law still holds exactly.
    """
    rng = random.Random(90210 + p * 100 + s * 10 + m)
    Js = rejection_sequence(rng, p, s, m, T)
    assert_four_divisor_law(p, Js, k, T, s, m)


@pytest.mark.parametrize("p,s,m", REGIMES)
def test_admissible_matrices_all_have_companion_valuations(p, s, m):
    """What (H1u)+(H2) actually force, read off the rejection-sampled set.

    Not assumed anywhere -- measured over the accepted set, and worth
    recording: every matrix the filter accepts has ``v(J_01) = 0``,
    ``v(J_11) = -s``, ``v(J_00) > -s`` and
    ``v(J_10) = m``, i.e. exactly the valuation pattern of the Henon
    companion matrix ``[[0, 1], [-delta, 2y]]``.  So the cone axioms pin the
    valuation *shape* even though they leave every entry free, and the second
    branch of the (H1u) analysis (``v(J_10) = -2s``) is killed by (H2), since
    it would force ``v(det) = -2s < 0 <= m``.  This is NOTE.md Lemma 6.2.
    """
    rng = random.Random(555 + p * 100 + s * 10 + m)
    Js = rejection_sequence(rng, p, s, m, 12)
    for J in Js:
        assert vp(J[0][1], p) == 0
        assert vp(J[1][1], p) == -s
        assert vp(J[0][0], p) > -s
        assert vp(J[1][0], p) == m


@pytest.mark.parametrize("p,s,m", REGIMES)
@pytest.mark.parametrize(
    "generator", [companion_sequence, general_sequence, rejection_sequence])
def test_shape_lemma_gives_the_operator_valuations(p, s, m, generator):
    """What NOTE.md §6's proof consumes in place of the companion Lemma 3.1.

    Lemma 3.1's row induction is companion-specific, so the general theorem
    cannot use it.  Its role -- pinning ``d_1`` of each pass from *below*, the
    cone vectors only giving upper bounds -- is taken over by the valuation
    shape of Lemma 6.2, which forces

        v(J) = -s   and   v(J^-1) = -(s+m)   entrywise,

    hence ``v(M_t) >= -s t`` and ``v(N_j) >= -(s+m) j`` for the products.  The
    same shape supplies the *seed* vectors of Lemma 3.4: ``J e_2`` is in
    ``C^u`` and ``J^-1 e_1`` is in ``C^s``, whatever ``J`` is.  Measured here
    over all three generators, at ``m = 0, 1, 2``.
    """
    # seed chosen so that ``general_sequence``'s own genericity assertion (all
    # four entries nonzero) holds; a rejected seed is a degenerate *frame*, not
    # a failure of anything asserted below.
    rng = random.Random(90210 + p * 100 + s * 10 + m)
    for J in generator(rng, p, s, m, 8):
        Ji = mat_inv(J)
        assert min(vp(J[i][j], p) for i in range(2) for j in range(2)) == -s
        assert min(vp(Ji[i][j], p)
                   for i in range(2) for j in range(2)) == -(s + m)
        u = mat_vec(J, (F(0), F(1)))            # seed of the forward cone
        assert vp(u[0], p) - vp(u[1], p) == s
        w = mat_vec(Ji, (F(1), F(0)))           # seed of the backward cone
        assert vp(w[1], p) - vp(w[0], p) == s + m


@pytest.mark.parametrize("p,s,m", REGIMES)
@pytest.mark.parametrize(
    "generator", [companion_sequence, general_sequence, rejection_sequence])
def test_projection_exponents_survive_the_generalisation(p, s, m, generator):
    """Corollary 3.2's *projection* exponents, for non-Henon sequences.

    Theorem C gives the four elementary divisors, which is all Theorem A
    needs.  NOTE.md §4's budget also needs ``w_x`` and ``w_y``, and those come
    from Lemma 3.1's row induction, which is stated for companion matrices.
    Measured here: the row induction goes through for any admissible sequence,
    because Lemma 6.2's valuation shape is exactly what it consumes, so

        w_x(H_t^F) = k - s(t-1),      w_y(H_t^F) = k - s t
        w_x(H_t^B) = k - (s+m) j,     w_y(H_t^B) = k - (s+m)(j-1),  j = T - t

    hold verbatim off Henon.  Pinned because NOTE.md §6 claims it.
    """
    k = 7
    Js = generator(random.Random(90210 + p * 100 + s * 10 + m), p, s, m, T)
    fwd, bwd = passes(p, Js, k, T)
    assert [fwd[t].x_projection_exponent() for t in range(1, T + 1)] == \
        [k - s * (t - 1) for t in range(1, T + 1)]
    assert [fwd[t].y_projection_exponent() for t in range(1, T + 1)] == \
        [k - s * t for t in range(1, T + 1)]
    assert [bwd[T - j].x_projection_exponent() for j in range(1, T + 1)] == \
        [k - (s + m) * j for j in range(1, T + 1)]
    assert [bwd[T - j].y_projection_exponent() for j in range(1, T + 1)] == \
        [k - (s + m) * (j - 1) for j in range(1, T + 1)]


@pytest.mark.parametrize("p,s,m", REGIMES)
def test_level_2_mixed_with_level_1(p, s, m):
    """Interleaving the two families is still admissible, so the law holds.

    The hypotheses are per-matrix, so a sequence may mix generators freely.
    This is the cheapest evidence that the law is a property of the *sequence
    of cone conditions* rather than of either construction.
    """
    rng = random.Random(31337 + p * 100 + s * 10 + m)
    a = companion_sequence(rng, p, s, m, T // 2)
    b = general_sequence(rng, p, s, m, T - T // 2)
    Js = [x for pair in zip(a, b) for x in pair]
    for J in Js:
        assert_hypotheses(J, p, s, m, rng)
    assert_four_divisor_law(p, Js, 3, len(Js), s, m)


@pytest.mark.parametrize("p,s,m", REGIMES)
def test_the_margin_is_what_makes_the_cones_work(p, s, m):
    """Sanity check on the mechanism: the two determinant terms never tie.

    Lemma 3.5's margin is ``2s + m``, and the whole proof is the observation
    that it is strictly positive so no ultrametric cancellation can occur.
    Measure it directly on the generated frames.
    """
    rng = random.Random(11 + p * 100 + s * 10 + m)
    for _ in range(20):
        u, w = _in_cone_u(rng, p, s), _in_cone_s(rng, p, s, m)
        cross, straight = u[0] * w[1], u[1] * w[0]
        assert vp(cross, p) - vp(straight, p) == 2 * s + m
        det = cross - straight
        assert vp(det, p) == vp(u[1], p) + vp(w[0], p)
        # ...which is v(u) + v(w), i.e. transversality defect 0
        assert vp(det, p) == (min(vp(u[0], p), vp(u[1], p))
                              + min(vp(w[0], p), vp(w[1], p)))
