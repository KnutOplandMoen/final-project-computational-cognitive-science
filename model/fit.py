"""
S&S fit in memo: pi_m(h), the probability of a six-draw record h under story m.

A story says which urn each draw came from (urn 0 or urn 1). Every urn starts
with one red and one green ball and keeps its own balls (Polya rule), so draw i
is red with chance (1 + s) / (2 + t), where t and s count the earlier draws
from the same urn and how many of them were red. The fit is the product of
these chances over the six draws; this equals prod_u s_u!(t_u - s_u)!/(t_u+1)!.

Records are integers 0..63: bit i is 1 if draw i+1 was red.
"""
from enum import IntEnum

import jax

jax.config.update("jax_enable_x64", True)                   # exact to ~1e-16, not float32

import jax.numpy as np
from memo import memo

from model.config import N_DRAWS
from model.stories import NAMES, URN, Record, Story, record  # noqa: F401  (re-exported)


class Ball(IntEnum):
    GREEN = 0
    RED = 1


@jax.jit
def p_ball(m, i, past, ball):
    """Chance that draw i (0-based) is `ball` under story m, given earlier draws `past`."""
    j = np.arange(N_DRAWS)
    reds = (past >> j) & 1
    same = (j < i) & (URN[m] == URN[m, i])                  # earlier draws, same urn
    t, s = same.sum(), (same * reds).sum()
    p_red = (1 + s) / (2 + t)
    return np.where(ball == Ball.RED, p_red, 1 - p_red)


# ---------------------------------------------------------------- memo model
@memo
def fit[m: Story, h: Record]():
    """Computes the probability of a record `h` given a story `m`.

    Args:
        m::Story
            The story of type `Story`.
        h::Record
            The observed record of type `Record`.

    Returns:
        a::float
            p(h|m) - Probability of the record `h` given story `m`.
    """

    urn: knows(m)
    urn: chooses(d0 in Ball, wpp=p_ball(m, 0, 0, d0))
    urn: chooses(d1 in Ball, wpp=p_ball(m, 1, d0, d1))
    urn: chooses(d2 in Ball, wpp=p_ball(m, 2, d0 + 2 * d1, d2))
    urn: chooses(d3 in Ball, wpp=p_ball(m, 3, d0 + 2 * d1 + 4 * d2, d3))
    urn: chooses(d4 in Ball, wpp=p_ball(m, 4, d0 + 2 * d1 + 4 * d2 + 8 * d3, d4))
    urn: chooses(d5 in Ball, wpp=p_ball(m, 5, d0 + 2 * d1 + 4 * d2 + 8 * d3 + 16 * d4, d5))
    return Pr[urn.d0 + 2 * urn.d1 + 4 * urn.d2 + 8 * urn.d3 + 16 * urn.d4 + 32 * urn.d5 == h]


# ---------------------------------------------------------------- helpers
FIT = fit()                                                 # (24 stories, 64 records) table of pi_m(h)


@jax.jit
def fit_table(m, h):
    """pi_m(h) looked up by index, so other memo models can use it."""
    return FIT[m, h]


def pi(story, rec):
    """pi_m(h) for a story name such as 'reset(4)' and a record such as 'GGGGRR'."""
    return float(FIT[NAMES.index(story), record(rec)])


if __name__ == "__main__":
    F = FIT
    print("each story sums to 1 over records:", bool(np.allclose(F.sum(axis=1), 1)))
    for rec, stories in [("GGGGRR", ["ORD", "reset(4)", "other(5,6)"]),
                         ("GGGRRR", ["ORD", "reset(3)", "pattern(3)", "other(4,6)"])]:
        print(rec)
        for m in stories:
            p, r = pi(m, rec), pi(m, rec) / pi("ORD", rec)
            print(f"  {m:11s} pi = {p:.6f} = 1/{1 / p:.2f}   r = {r:.2f}")


