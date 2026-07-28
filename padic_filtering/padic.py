"""p-adic arithmetic helpers.

Every p-adic number in this project is represented exactly, as a
:class:`fractions.Fraction` whose denominator may contain powers of ``p``
(elements of ``Q_p``) or not (elements of ``Z_(p)``).  There is no floating
point anywhere; a "p-adic number known to precision N" is a Fraction together
with the separately-tracked statement that it is correct modulo ``p^N``.

Convention for the valuation of zero: ``vp(0) == INFINITY`` (``math.inf``).
It is a float, so it compares correctly against every integer exponent and
poisons any arithmetic that would silently use it as an exponent.
"""

from __future__ import annotations

from fractions import Fraction
from math import inf

INFINITY = inf

Rat = Fraction | int


def vp(x: Rat, p: int) -> int | float:
    """p-adic valuation.  ``vp(0)`` is ``INFINITY``."""
    if x == 0:
        return INFINITY
    if isinstance(x, Fraction):
        return _vp_int(x.numerator, p) - _vp_int(x.denominator, p)
    return _vp_int(int(x), p)


def _vp_int(n: int, p: int) -> int:
    v = 0
    n = abs(n)
    while n % p == 0:
        n //= p
        v += 1
    return v


def unit_part(x: Rat, p: int) -> Fraction:
    """``x / p^vp(x)``, a p-adic unit.  Raises on ``x == 0``."""
    if x == 0:
        raise ZeroDivisionError("0 has no unit part")
    return Fraction(x) * ppow(p, -vp(x, p))


def ppow(p: int, k: int | float) -> Fraction:
    """``p^k`` for possibly negative integer ``k``."""
    if k == INFINITY:
        raise ValueError("infinite exponent")
    k = int(k)
    return Fraction(p**k) if k >= 0 else Fraction(1, p**-k)


def is_integral(x: Rat, p: int) -> bool:
    """Is ``x`` in ``Z_p``?"""
    return vp(x, p) >= 0


def inv_mod(a: int, p: int, k: int) -> int:
    """Inverse of ``a`` in ``Z/p^k``.  Raises if ``a`` is not a unit."""
    if a % p == 0:
        raise ZeroDivisionError(f"{a} is not a unit mod {p}: division by a non-unit")
    return pow(a, -1, p**k)


def zp_int(x: Rat, p: int, k: int) -> int:
    """Representative in ``[0, p^k)`` of ``x in Z_p`` modulo ``p^k``.

    Raises if ``x`` is not p-adically integral (its denominator is then a
    non-unit, and the reduction does not exist).
    """
    if k <= 0:
        return 0
    if x == 0:
        return 0
    x = Fraction(x)
    if x.denominator % p == 0:
        raise ZeroDivisionError(f"{x} is not in Z_{p}: division by a non-unit")
    mod = p**k
    return (x.numerator % mod) * inv_mod(x.denominator, p, k) % mod


def mod_pk(x: Rat, p: int, k: int | float) -> Fraction:
    """Canonical small representative ``r`` of ``x`` modulo ``p^k Z_p``.

    Returns the unique ``r = p^j * n`` with ``j = min(0, vp(x))`` and
    ``0 <= n < p^(k-j)`` such that ``vp(x - r) >= k``.  Used to keep coset
    representatives from growing without bound under iteration.
    """
    if x == 0 or k == -INFINITY:
        return Fraction(0)
    v = vp(x, p)
    if v >= k:
        return Fraction(0)
    j = min(0, int(v))
    y = Fraction(x) * ppow(p, -j)  # in Z_p
    return Fraction(zp_int(y, p, int(k) - j)) * ppow(p, j)


def sqrt_zp(u: Rat, p: int, k: int) -> int:
    """Square root of a unit ``u in Z_p^*`` modulo ``p^k``, by Hensel lifting.

    ``p`` must be odd and ``u`` a square (``u mod p`` a quadratic residue);
    raises otherwise.  Returns the root that is a square root of ``u mod p^k``.
    """
    if p == 2:
        raise NotImplementedError("odd residue characteristic only")
    u0 = zp_int(u, p, k)
    if u0 % p == 0:
        raise ValueError("sqrt_zp expects a unit")
    r = next((r for r in range(1, p) if r * r % p == u0 % p), None)
    if r is None:
        raise ValueError(f"{u} is not a square mod {p}")
    prec = 1
    while prec < k:
        prec = min(2 * prec, k)
        mod = p**prec
        # Newton on z^2 - u
        r = (r + u0 * inv_mod(r, p, prec)) * inv_mod(2, p, prec) % mod
    assert (r * r - u0) % p**k == 0
    return r % p**k


def exact_div_ppow(x: Rat, p: int, s: int) -> Fraction:
    """``x / p^s``, raising unless the division is exact in ``Z_p``.

    This is the horseshoe cancellation: on the filled Julia
    set ``Y^2 + C`` is divisible by ``p^s``, and a failure here means the
    point is not on the invariant set (or precision has run out).
    """
    q = Fraction(x) * ppow(p, -s)
    if q != 0 and vp(q, p) < 0:
        raise ValueError(f"{x} is not divisible by {p}^{s} in Z_p")
    return q
