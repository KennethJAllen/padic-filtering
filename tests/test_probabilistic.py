import random
from fractions import Fraction as F

import pytest

from padic_filtering.probabilistic import (DigitChannel, QuadraticMap,
                                           certain_digits, digits,
                                           forward_algorithm, from_digits,
                                           map_estimate, predict, support_size,
                                           uniform, update)

P, K = 3, 4


def test_digit_roundtrip():
    for x in range(P**K):
        assert from_digits(digits(x, P, K), P) == x


def test_likelihood_is_a_probability_distribution():
    ch = DigitChannel(p=P, k=K, eps=F(1, 5))
    for true in [0, 7, P**K - 1]:
        total = sum(ch.likelihood(obs, true) for obs in range(P**K))
        assert total == 1, "likelihoods must sum to exactly 1 (exact arithmetic)"


def test_noiseless_channel_is_a_point_mass():
    ch = DigitChannel(p=P, k=K, eps=F(0))
    assert ch.likelihood(5, 5) == 1
    assert ch.likelihood(5, 6) == 0


def test_map_is_not_injective_so_prediction_sums_mass():
    g = QuadraticMap(p=P, k=K, c=5)
    pre = g.preimages()
    assert any(len(xs) > 1 for xs in pre), "g should not be injective mod p^k"
    post = predict(uniform(g.modulus), g)
    assert sum(post) == 1
    assert max(post) > F(1, g.modulus), "mass must pile up on shared images"


def test_forward_algorithm_recovers_the_state_without_noise():
    g = QuadraticMap(p=P, k=K, c=5)
    ch = DigitChannel(p=P, k=K, eps=F(0))
    rng = random.Random(0)
    x0 = rng.randrange(g.modulus)
    truth = g.orbit(x0, 8)
    posts = forward_algorithm(g, ch, truth)
    for post, x in zip(posts, truth):
        assert support_size(post) == 1 and post[x] == 1


def test_posterior_concentrates_under_noise():
    g = QuadraticMap(p=P, k=K, c=5)
    ch = DigitChannel(p=P, k=K, eps=F(1, 5))
    rng = random.Random(1)
    x0 = rng.randrange(g.modulus)
    truth = g.orbit(x0, 12)
    obs = [ch.corrupt(x, rng) for x in truth]
    posts = forward_algorithm(g, ch, obs)
    assert all(sum(post) == 1 for post in posts), "posteriors stay normalised"
    assert map_estimate(posts[-1]) == truth[-1]
    assert posts[-1][truth[-1]] > posts[0][truth[0]], "information must accumulate"


def test_certain_digits_counts_correctly():
    g = QuadraticMap(p=P, k=K, c=5)
    point = [F(0)] * g.modulus
    point[11] = F(1)
    assert certain_digits(point, P, K, F(9, 10)) == K
    assert certain_digits(uniform(g.modulus), P, K, F(9, 10)) == 0


def test_impossible_observation_is_reported_not_swallowed():
    ch = DigitChannel(p=P, k=K, eps=F(0))
    point = [F(0)] * (P**K)
    point[3] = F(1)
    with pytest.raises(ValueError):
        update(point, 4, ch)  # incompatible with a noiseless channel
