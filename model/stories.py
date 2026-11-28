"""
Stories module for the model. Defines the possible stories that agents can adopt in the model.

A story says which urn (0 or 1) each of draws 1..7 came from. Draws 1..6 are the
record the listener sees; draw 7 is the risky one, so the story's urn for draw 7
decides its forecast q_m(h).

Draw 7 under each kind:
    ORD           urn 0 (nothing changed)
    reset(c)      urn 1 (the new urn keeps going)
    other(j, c)   urn 0 (the odd draws are over, back to the original urn)
    pattern(l)    the alternation continues: urn ((7 - 1) // l) % 2
"""
import jax.numpy as np

from model.config import N_DRAWS

KINDS = ["ORD", "reset", "other", "pattern"]


def _stories():
    """(name, kind, urn of each draw 1..7) for ORD and the 23 false stories."""
    draws = range(1, N_DRAWS + 2)                               # 1..7
    out = [("ORD", "ORD", [0 for _ in draws])]
    for c in range(1, N_DRAWS):                                 # reset(c): new urn after draw c
        out.append((f"reset({c})", "reset", [0 if d <= c else 1 for d in draws]))
    for j in range(2, N_DRAWS + 1):                             # other(j, c): draws j..c from urn 1
        for c in range(j, N_DRAWS + 1):
            out.append((f"other({j},{c})", "other", [1 if j <= d <= c else 0 for d in draws]))
    for l in range(1, 4):                                       # pattern(l): turns of l draws
        out.append((f"pattern({l})", "pattern", [((d - 1) // l) % 2 for d in draws]))
    return out


STORIES = _stories()
NAMES = [name for name, _, _ in STORIES]
KIND = np.array([KINDS.index(kind) for _, kind, _ in STORIES])  # (24,) kind index of each story
URN = np.array([urns[:N_DRAWS] for _, _, urns in STORIES])      # (24, 6) urn of draws 1..6
URN7 = np.array([urns[N_DRAWS] for _, _, urns in STORIES])      # (24,)  urn of draw 7

# n_k: how many versions each kind has (ORD has one)
N_VERSIONS_OF_KIND = np.array([sum(k == kind for _, k, _ in STORIES) for kind in KINDS])
N_VERSIONS = N_VERSIONS_OF_KIND[KIND]                           # (24,) n_k of each story's kind

Story = np.arange(len(STORIES))
Record = np.arange(2 ** N_DRAWS)


def story(name):
    """'reset(4)' -> story index."""
    return NAMES.index(name)


def record(s):
    """'GGGGRR' -> record index (bit i is 1 if draw i+1 was red)."""
    return sum(1 << i for i, c in enumerate(s) if c == "R")
