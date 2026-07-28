"""The p-adic Henon map, in plain and scaled coordinates.

Plain form::

    f(x, y)     = (y, y^2 + c - delta*x)
    f^-1(x, y)  = ((x^2 + c - y)/delta, x)
    J(x, y)     = [[0, 1], [-delta, 2y]]        det J = delta  (constant)

In the horseshoe regime ``H_III`` (``v_p(c) = -2s < 0``) every point of the
filled Julia set has ``|x| = |y| = p^s``.  Scaled coordinates ``X = p^s x``
put the orbit in ``Z_p^2`` and are available as :meth:`Henon.f_scaled`, but
the trackers iterate the plain map above: scaling is by the *scalar* matrix
``p^s I``, which shifts every lattice exponent by ``s`` and changes no rate or
slope, while the plain map -- being a polynomial with no division -- stays
defined on coset representatives that have drifted off the invariant set.
See :meth:`Henon.f_scaled` for the full argument.

The eigenvalue valuations of ``J`` come from the Newton polygon of
``lambda^2 - 2y lambda + delta``, whose vertices are ``(0, m)``, ``(1, v_p(y))``,
``(2, 0)``.  On the horseshoe ``v_p(y) = -s`` identically, so they are
``{-s, s+m}`` where ``m = v_p(delta)``: genuine expansion *and* contraction,
with no tangency events (``|2y| = p^s`` always).  In ``H+_II`` (``s = 0``,
``y`` generically a unit) they are ``{0, m}`` -- nonexpanding, which is why
the smoother is vacuous there.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .lattice import Mat, Vec, as_vec
from .padic import (Rat, exact_div_ppow, inv_mod, mod_pk, ppow, sqrt_zp, vp,
                    zp_int)


# The map squares its argument, so an unreduced iterate doubles in digit count
# every step: 24 steps of a 15-digit start is ~2.5e8 digits, i.e. gigabytes, and
# the process will take the machine down before it takes itself down.  Every
# iteration in this project must reduce its representative modulo the tracked
# lattice (Lattice.reduce_vector); this guard makes a missed reduction fail loudly
# and immediately instead of silently allocating.
MAX_REPR_BITS = 1 << 20  # ~315k digits: far above any legitimate reduced value


def _guard(v: Vec) -> Vec:
    for x in v:
        if max(x.numerator.bit_length(), x.denominator.bit_length()) > MAX_REPR_BITS:
            raise MemoryError(
                "Henon iterate exceeded MAX_REPR_BITS: the representative was not "
                "reduced modulo its lattice between steps (see padic_filtering.henon)"
            )
    return v


@dataclass(frozen=True)
class Henon:
    """Henon map over ``Q_p`` in scaled coordinates with scaling exponent ``s``.

    ``C`` is the scaled parameter ``p^(2s) c``; the unscaled parameter is
    ``c = C / p^(2s)``.  ``s = 0`` gives the plain map with ``C = c``.
    """

    p: int
    C: Fraction
    delta: Fraction
    s: int = 0

    @classmethod
    def from_c(cls, p: int, c: Rat, delta: Rat) -> Henon:
        """Build from the unscaled parameter ``c``, inferring ``s``."""
        vc = vp(c, p)
        s = 0 if vc >= 0 else (-int(vc) + 1) // 2
        if vc < 0 and int(vc) % 2 != 0:
            raise ValueError(f"v_p(c) = {vc} is odd; H_III needs v_p(c) = -2s")
        return cls(p=p, C=Fraction(c) * ppow(p, 2 * s), delta=Fraction(delta), s=s)

    # ------------------------------------------------------------ regime

    @property
    def m(self) -> int:
        return int(vp(self.delta, self.p))

    @property
    def c(self) -> Fraction:
        return self.C * ppow(self.p, -2 * self.s)

    @property
    def region(self) -> str:
        """ADP region, via the dictionary of ``docs/REFERENCES.md`` (``a = -c``, ``b = -delta``)."""
        if self.s > 0:
            return "H_III"
        return "H_I" if self.m == 0 else "H+_II"

    @property
    def eigenvalue_valuations(self) -> tuple[int, int]:
        """``{-s, s+m}`` -- the Newton polygon of the characteristic polynomial."""
        return (-self.s, self.s + self.m)

    def check_regime(self) -> None:
        """Assert the ADP hypotheses for the declared region."""
        if self.region == "H_III":
            # |c| > max(1, |delta|^2)  <=>  2s > max(0, -2m); m >= 0 always here
            assert 2 * self.s > 0 and 2 * self.s > -2 * self.m, "not in H_III"
            # J is empty unless a = -c is a square (ADP Thm 1(a))
            assert self.is_square_a(), "-c is not a square: the filled Julia set is empty"
        else:
            assert vp(self.c, self.p) >= 0, "H_I / H+_II need v_p(c) >= 0"

    def is_square_a(self) -> bool:
        """Is ``a = -c`` a square in ``Q_p``?  (ADP Thm 1(a).)"""
        a = -self.c
        if a == 0:
            return True
        va = vp(a, self.p)
        if int(va) % 2 != 0:
            return False
        u = a * ppow(self.p, -int(va))
        r = zp_int(u, self.p, 1)
        return pow(r, (self.p - 1) // 2, self.p) == 1

    def gamma(self, prec: int) -> Fraction:
        """``Gamma = p^s * sqrt(-c)``, a unit of ``Z_p``, modulo ``p^prec``."""
        return Fraction(sqrt_zp(-self.C, self.p, prec))

    # -------------------------------------------------------------- maps

    def f(self, v) -> Vec:
        x, y = _guard(as_vec(v))
        return (y, y * y + self.c - self.delta * x)

    def f_inv(self, v) -> Vec:
        x, y = _guard(as_vec(v))
        return ((x * x + self.c - y) / self.delta, x)

    def f_scaled(self, V) -> Vec:
        """The same step in scaled coordinates ``X = p^s x``.

        ``F(X, Y) = ((Y^2 + C)/p^s - delta X)`` puts the orbit in ``Z_p^2``, and
        the division by ``p^s`` is exact *only on the invariant set* -- it is
        the horseshoe cancellation, and :func:`exact_div_ppow` raises when it
        fails.  That division is also precisely where the ``s``-digits-per-step
        loss shows up in the raw arithmetic (the loss the smoother repairs), so
        this method is kept as the honest statement of §2.3 and is what the
        naive tracker models.

        The trackers themselves iterate the *unscaled* map instead.  Scaling is
        by the scalar matrix ``p^s I``, which commutes with everything and just
        shifts every lattice exponent by ``s``, so no slope, rate or tent
        changes -- but unscaled ``f`` is a polynomial with no division at all,
        and therefore stays defined on representatives that have drifted off
        the invariant set.  Once ``d1 < 0`` the coset is wider than the
        invariant set and such representatives are unavoidable.
        """
        X, Y = _guard(as_vec(V))
        return (Y, exact_div_ppow(Y * Y + self.C, self.p, self.s) - self.delta * X)

    def to_scaled(self, v) -> Vec:
        sc = ppow(self.p, self.s)
        return (sc * Fraction(v[0]), sc * Fraction(v[1]))

    def from_scaled(self, V) -> Vec:
        sc = ppow(self.p, -self.s)
        return (sc * Fraction(V[0]), sc * Fraction(V[1]))

    def jacobian(self, v) -> Mat:
        _, y = as_vec(v)
        return ((Fraction(0), Fraction(1)), (-self.delta, 2 * y))

    def jacobian_inv(self, v) -> Mat:
        """Jacobian *of* ``f^-1`` at ``v`` (= ``J(f^-1(v))^-1``)."""
        x, _ = as_vec(v)
        return ((2 * x / self.delta, -1 / self.delta), (Fraction(1), Fraction(0)))

    def quadratic_remainder_exponent(self, w: int, backward: bool = False) -> int:
        """Exponent of the quadratic Taylor remainder module (NOTE.md Def 4.1).

        The Taylor expansion of a quadratic map terminates::

            f(v + h)     = f(v)    + J h     + (0, h_y^2)
            f^-1(v + h)  = f^-1(v) + J^-1 h  + (h_x^2 / delta, 0)

        so if the projection of ``H`` onto the squared coordinate is
        ``p^w Z_p``, the remainder lies in the module ``p^(2w) Z_p`` forward and
        ``p^(2w - m) Z_p`` backward.
        """
        return 2 * w - (self.m if backward else 0)

    def orbit(self, v0, T: int, prec: int | None = None) -> list[Vec]:
        """``T+1`` exact iterates, each reduced modulo ``p^prec``.

        ``prec`` is not optional in practice: the map squares its argument, so
        unreduced iterates double in digit-count every step and a few dozen
        steps exhaust memory (:data:`MAX_REPR_BITS` catches it).  Reduction is
        lossless only where the map is nonexpanding, i.e. ``s == 0``: then
        ``f(v + p^prec z) = f(v) mod p^prec`` because ``c`` and ``delta`` are
        integral and ``v_p(2y) >= 0``.  On the horseshoe (``s >= 1``) each step
        costs ``s`` digits, so build ground truth with :func:`periodic_orbit`
        in scaled coordinates instead.
        """
        if prec is not None and self.s:
            raise ValueError(
                f"orbit(prec=...) is lossless only for s = 0, but s = {self.s}; "
                "use periodic_orbit() for H_III ground truth")
        v = as_vec(v0)
        out = [v]
        for _ in range(T):
            v = self.f(v)
            if prec is not None:
                v = (mod_pk(v[0], self.p, prec), mod_pk(v[1], self.p, prec))
            out.append(v)
        return out


# ---------------------------------------------------------------- orbits


def fixed_point_c(p: int, delta: Rat, alpha: Rat) -> Fraction:
    """``c`` making ``(alpha, alpha)`` a fixed point: ``c = alpha + delta*alpha - alpha^2``."""
    alpha = Fraction(alpha)
    return alpha + Fraction(delta) * alpha - alpha * alpha


def periodic_orbit(hen: Henon, itinerary: str, prec: int) -> list[int]:
    """Period-``l`` horseshoe orbit from a symbolic itinerary.

    Solves the circulant system ``X_{t+2} = (X_{t+1}^2 + C)/p^s - delta X_t``
    (indices mod ``l``) by multivariate Newton over ``Z/p^prec``, started from
    ``X_t = eps_t * Gamma`` with ``Gamma = sqrt(-C)``.  ADP Thm 1(e) gives the
    conjugacy to the 2-shift, so each itinerary in ``{+,-}^l`` names exactly one
    orbit, and hyperbolicity makes the Newton Jacobian invertible mod ``p``
    (quadratic convergence -- the shadowing lemma in computational form).

    Returns the scaled coordinates ``X_0, ..., X_{l-1}`` as integers mod
    ``p^prec``.
    """
    p, s, delta = hen.p, hen.s, hen.delta
    eps = [1 if ch == "+" else -1 for ch in itinerary]
    n = len(eps)
    if n < 1:
        raise ValueError("empty itinerary")

    # Cleared system: G_t = p^s X_{t+2} - X_{t+1}^2 - C + p^s delta X_t = 0
    ps = p**s
    mod = p**prec
    Ci = zp_int(hen.C, p, prec)
    dl = zp_int(delta, p, prec)

    def G(X):
        return [(ps * X[(t + 2) % n] - X[(t + 1) % n] ** 2 - Ci
                 + ps * dl * X[t]) % mod for t in range(n)]

    def jac_rows(X):
        rows = []
        for t in range(n):
            row = [0] * n
            row[(t + 2) % n] = (row[(t + 2) % n] + ps) % mod
            row[(t + 1) % n] = (row[(t + 1) % n] - 2 * X[(t + 1) % n]) % mod
            row[t] = (row[t] + ps * dl) % mod
            rows.append(row)
        return rows

    g = hen.gamma(prec)
    X = [int(g) * e % mod for e in eps]
    assert all(v % p == 0 for v in G(X)), "initial guess is not a mod-p solution"

    for _ in range(2 * prec.bit_length() + 8):
        g_ = G(X)
        if all(v == 0 for v in g_):
            break
        step = _solve_mod(jac_rows(X), g_, p, prec)
        X = [(X[i] - step[i]) % mod for i in range(n)]
    assert all(v == 0 for v in G(X)), "Newton did not converge"

    # verify genuine periodicity by exact forward iteration in Z/p^prec
    for t in range(n):
        lhs = (ps * X[(t + 2) % n]) % mod
        rhs = (X[(t + 1) % n] ** 2 + Ci - ps * dl * X[t]) % mod
        assert lhs == rhs
    return X


def _solve_mod(rows, rhs, p: int, prec: int) -> list[int]:
    """Solve ``A z = rhs`` over ``Z/p^prec`` when ``A`` is invertible mod ``p``."""
    mod = p**prec
    n = len(rows)
    A = [list(r) + [rhs[i]] for i, r in enumerate(rows)]
    where = []
    for col in range(n):
        piv = next((r for r in range(col, n) if A[r][col] % p != 0), None)
        if piv is None:
            raise ZeroDivisionError("Newton Jacobian is singular mod p")
        A[col], A[piv] = A[piv], A[col]
        inv = inv_mod(A[col][col], p, prec)
        A[col] = [x * inv % mod for x in A[col]]
        for r in range(n):
            if r != col and A[r][col] % mod:
                f = A[r][col]
                A[r] = [(A[r][j] - f * A[col][j]) % mod for j in range(n + 1)]
        where.append(col)
    return [A[i][n] for i in range(n)]


def truth_orbit(hen: Henon, X: list[int], T: int, prec: int) -> list[Vec]:
    """Unroll a periodic orbit into ``T+1`` states in *unscaled* coordinates.

    ``X`` holds the scaled coordinates ``p^s x_t`` as integers mod ``p^prec``,
    so the returned points are known to absolute precision ``prec - s``.
    """
    n = len(X)
    mod = hen.p**prec
    sc = ppow(hen.p, -hen.s)
    return [(sc * (X[t % n] % mod), sc * (X[(t + 1) % n] % mod))
            for t in range(T + 1)]


def truth_precision(hen: Henon, prec: int) -> int:
    """Absolute precision of an orbit produced by :func:`truth_orbit`."""
    return prec - hen.s
