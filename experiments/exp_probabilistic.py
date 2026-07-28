"""Noisy-digit filtering on a 1D p-adic quadratic map.

Run only after §5 passes, and kept strictly separate from the 2D lattice demo:
this is measures over the depth-k p-ary tree, not lattices, and the two are not
mixed into one story.

Two things are shown, both with exact rational probabilities:

  1. With ``eps = 0`` the posterior collapses onto the true state immediately
     -- the noiseless case degenerates to the deterministic filter, as it must.
  2. With ``eps > 0`` a single observation is not enough, but the deterministic
     dynamics mixes the digits, so successive noisy observations concentrate
     the posterior: the MAP estimate locks onto the truth and stays there.
"""

from __future__ import annotations

import random
from fractions import Fraction as F

from _common import EXTRA, INK_MUTED, report, save, style, write_json  # noqa: I001
from padic_filtering.probabilistic import (DigitChannel, QuadraticMap,
                                           certain_digits, forward_algorithm,
                                           map_estimate, support_size)

SEED = 20260726
P, K, T = 3, 6, 14
C = 5
EPSILONS = [F(0), F(1, 20), F(1, 10), F(1, 5)]


def run(eps: F, seed: int = SEED):
    g = QuadraticMap(p=P, k=K, c=C)
    channel = DigitChannel(p=P, k=K, eps=eps)
    rng = random.Random(seed)
    x0 = rng.randrange(g.modulus)
    truth = g.orbit(x0, T)
    obs = [channel.corrupt(x, rng) for x in truth]
    posts = forward_algorithm(g, channel, obs)
    return {
        "eps": eps, "x0": x0, "truth": truth, "observations": obs,
        "map_correct": [map_estimate(p_) == x for p_, x in zip(posts, truth)],
        "prob_truth": [p_[x] for p_, x in zip(posts, truth)],
        "support": [support_size(p_) for p_ in posts],
        "certain_digits": [certain_digits(p_, P, K, F(9, 10)) for p_ in posts],
    }


def main() -> None:
    style()
    import matplotlib.pyplot as plt

    results = {}
    for eps in EPSILONS:
        results[str(eps)] = run(eps)

    noiseless = results[str(F(0))]
    assert all(noiseless["map_correct"]), "eps=0 must identify the state exactly"
    assert noiseless["support"][0] == 1, \
        "eps=0: a single clean observation must determine the state"

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")

    ax = axes[0]
    for i, eps in enumerate(EPSILONS):
        r = results[str(eps)]
        ys = [float(x) for x in r["prob_truth"]]
        colour = EXTRA[i % len(EXTRA)]
        name = f"$\\varepsilon$={eps}"
        ax.plot(range(len(ys)), ys, color=colour, marker="o", markevery=4, label=name)
    ax.set_xlabel("iteration $t$")
    ax.set_ylabel("posterior probability of the true state")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(-0.4, T + 0.4)
    ax.set_title("Noisy digits: the posterior concentrates\non the truth",
                 loc="left", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, labelcolor=INK_MUTED)

    ax = axes[1]
    for i, eps in enumerate(EPSILONS):
        r = results[str(eps)]
        colour = EXTRA[i % len(EXTRA)]
        ax.plot(range(len(r["certain_digits"])), r["certain_digits"], color=colour,
                marker="o", markevery=4, lw=2.6 - 0.3 * i, alpha=0.9,
                label=f"$\\varepsilon$={eps}")
    ax.axhline(K, color=INK_MUTED, ls=":", lw=1)
    ax.annotate(f"all {K} digits", xy=(0.5, K), xytext=(0, 4),
                textcoords="offset points", fontsize=8, color=INK_MUTED)
    ax.set_xlabel("iteration $t$")
    ax.set_ylabel(f"digits pinned down at $\\geq$ 90% posterior")
    ax.set_xlim(-0.4, T + 0.4)
    ax.set_title("Digits recovered, exactly counted", loc="left", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, labelcolor=INK_MUTED)

    fig.suptitle(f"§4.6  Noisy-digit posterior on $g(x)=x^2+{C}$ over "
                 f"$\\mathbb{{Z}}/{P}^{{{K}}}$ (exact rational probabilities)",
                 x=0.005, ha="left", fontsize=12)
    path = save(fig, "exp_probabilistic")

    # with moderate noise the filter should still lock on and stay locked
    moderate = results[str(F(1, 10))]
    locked = moderate["map_correct"]
    assert any(locked), "the filter never identified the state at eps=1/10"
    first_lock = locked.index(True)
    assert all(locked[first_lock:]), "the MAP estimate must not come unstuck"

    out = {
        "seed": SEED, "p": P, "k": K, "c": C, "T": T,
        "state_space_size": P**K,
        "epsilons": [str(e) for e in EPSILONS],
        "results": {k: {kk: (vv if kk != "prob_truth" else [str(x) for x in vv])
                        for kk, vv in v.items()} for k, v in results.items()},
        "first_lock_at_eps_0.1": first_lock,
        "verdict": (
            "PASS: with eps=0 the posterior is a point mass on the truth from the "
            "first observation.  With eps=1/10 the MAP estimate locks onto the "
            f"true state at t={first_lock} and never comes unstuck, because the "
            "deterministic dynamics mixes the digits and successive noisy "
            "observations are informative about different ones.  All "
            "probabilities are exact rationals."),
    }
    write_json("exp_probabilistic", out)
    report("§4.6 probabilistic (noisy digits, 1D)", [
        f"state space: {P}^{K} = {P**K} states, exact Fractions",
        f"eps=0: identified immediately, support {noiseless['support'][0]}",
        f"eps=1/10: MAP locks at t={first_lock}, "
        f"P(truth) {float(moderate['prob_truth'][-1]):.4f} at t={T}",
        f"figure: {path}"])


if __name__ == "__main__":
    main()
