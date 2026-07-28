"""Noisy-digit filtering by exact forward algorithm.

This is the *probabilistic* variant, and it is deliberately built on different
machinery from the rest of the package: measures over a finite state space
rather than lattices.  Two constraints from the design shape it:

  * **1D, not 2D.**  The state space is the depth-``k`` p-ary tree, i.e.
    ``Z/p^k``.  In 2D Henon that tree has ``p^(2N)`` leaves and is hopeless;
    in 1D on a p-adic quadratic map it is ``p^k`` and entirely tractable.
    2D Henon stays on lattices.  One demo does not do both.

  * **Exact, not floating point.**  Every probability is a
    :class:`~fractions.Fraction`, so the posterior is exact and the reported
    numbers are certified in the same sense as the lattice track.

The model.  The state evolves deterministically, ``x_{t+1} = g(x_t)`` with
``g(x) = x^2 + c`` on ``Z/p^k``.  At each time every p-ary digit of ``x_t`` is
observed through an independent symmetric channel: with probability
``1 - eps`` the digit is reported correctly, and with probability ``eps`` it is
replaced by one of the other ``p - 1`` digits, uniformly.

Because ``g`` is not injective mod ``p^k``, the prediction step *sums* mass
(several preimages can land on one state) -- unlike the lattice filter, where
the map is an automorphism and the prediction step is a bijection.  That is the
honest difference between the two settings, not a defect.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

Dist = list[Fraction]


def digits(x: int, p: int, k: int) -> list[int]:
    out = []
    for _ in range(k):
        out.append(x % p)
        x //= p
    return out


def from_digits(ds: list[int], p: int) -> int:
    return sum(d * p**i for i, d in enumerate(ds))


@dataclass(frozen=True)
class DigitChannel:
    """Symmetric p-ary channel: each digit is flipped with probability ``eps``."""

    p: int
    k: int
    eps: Fraction

    def __post_init__(self):
        assert 0 <= self.eps < 1, "eps must be a probability below 1"
        assert self.p > 1

    def likelihood(self, observed: int, true: int) -> Fraction:
        """``P(observed | true)``, exact."""
        od, td = digits(observed, self.p, self.k), digits(true, self.p, self.k)
        agree = sum(1 for a, b in zip(od, td) if a == b)
        disagree = self.k - agree
        if disagree and self.eps == 0:
            return Fraction(0)
        per_wrong = self.eps / (self.p - 1)
        return (1 - self.eps) ** agree * per_wrong**disagree

    def corrupt(self, x: int, rng) -> int:
        ds = digits(x, self.p, self.k)
        out = []
        for d in ds:
            if rng.random() < float(self.eps):
                out.append(rng.choice([e for e in range(self.p) if e != d]))
            else:
                out.append(d)
        return from_digits(out, self.p)


@dataclass(frozen=True)
class QuadraticMap:
    """``g(x) = x^2 + c`` on ``Z/p^k`` -- the 1D stand-in for Henon."""

    p: int
    k: int
    c: int

    @property
    def modulus(self) -> int:
        return self.p**self.k

    def __call__(self, x: int) -> int:
        return (x * x + self.c) % self.modulus

    def orbit(self, x0: int, T: int) -> list[int]:
        out, x = [x0 % self.modulus], x0 % self.modulus
        for _ in range(T):
            x = self(x)
            out.append(x)
        return out

    def preimages(self) -> list[list[int]]:
        """``pre[y]`` lists the states mapping to ``y`` (``g`` is not injective)."""
        pre: list[list[int]] = [[] for _ in range(self.modulus)]
        for x in range(self.modulus):
            pre[self(x)].append(x)
        return pre


def uniform(n: int) -> Dist:
    return [Fraction(1, n)] * n


def normalise(w: Dist) -> Dist:
    total = sum(w)
    if total == 0:
        raise ValueError("posterior collapsed to zero mass: the observation "
                         "sequence is impossible under the model")
    return [x / total for x in w]


def update(prior: Dist, observation: int, channel: DigitChannel) -> Dist:
    """Bayes update on a single noisy observation of the whole state."""
    return normalise([w * channel.likelihood(observation, x) if w else Fraction(0)
                      for x, w in enumerate(prior)])


def predict(prior: Dist, g: QuadraticMap) -> Dist:
    """Push the distribution through the (non-injective) deterministic map."""
    out = [Fraction(0)] * g.modulus
    for x, w in enumerate(prior):
        if w:
            out[g(x)] += w
    return out


def forward_algorithm(g: QuadraticMap, channel: DigitChannel,
                      observations: list[int], prior: Dist | None = None
                      ) -> list[Dist]:
    """Exact forward algorithm: one posterior per time step."""
    post = prior if prior is not None else uniform(g.modulus)
    out = []
    for t, obs in enumerate(observations):
        if t:
            post = predict(post, g)
        post = update(post, obs, channel)
        out.append(post)
    return out


# ------------------------------------------------------------------ metrics


def map_estimate(post: Dist) -> int:
    return max(range(len(post)), key=lambda x: post[x])


def certain_digits(post: Dist, p: int, k: int, threshold: Fraction) -> int:
    """How many p-ary digits the posterior pins down beyond ``threshold``.

    The lattice-filter analogue of "digits we are sure of", so the two tracks
    can be compared on the same axis.
    """
    count = 0
    for i in range(k):
        marg = [Fraction(0)] * p
        for x, w in enumerate(post):
            if w:
                marg[(x // p**i) % p] += w
        if max(marg) >= threshold:
            count += 1
    return count


def support_size(post: Dist) -> int:
    return sum(1 for w in post if w)
