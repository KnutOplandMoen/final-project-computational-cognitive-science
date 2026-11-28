"""
Prior module for the model. Defines the prior probabilities used by the agents in the model.

The listener's prior is generated in two steps, first the kind and then a version
of that kind:

    p(ORD) = rho
    p(m)   = (1 - rho) / kappa ** a_kappa * 1 / nu_k,    nu_k = n_k ** a_nu

Only the story held (ORD) and the story offered in the note are considered
(assumption 2), so every other story gets 0. A note of ORD means silence: the
listener then puts all its weight on ORD.

The values are not normalised over the pair; memo normalises them. What matters is
the prior odds of the note against ORD:

    omega = p(m) / p(ORD) = (1 - rho) / (rho * kappa ** a_kappa * nu_k)
"""
import jax
import jax.numpy as np

from model.stories import N_VERSIONS


@jax.jit
def prior(story : int, note : int, rho: float, kappa: int, a_kappa: float, a_nu: float):
    """Prior weight p(story) for a listener who has been sent `note`.

    Args:
        story::int
            index of the story being weighed.
        note::int
            index of the story in the note (0 = ORD = silence).
        rho::float
            prior chance of the ordinary urn.
        kappa::int
            kinds the listener has in mind.
        a_kappa::float
            exponent on kappa.
        a_nu::float
            exponent on n_k, so nu_k = n_k ** a_nu.

    Returns:
        float
            rho for ORD, (1 - rho) / (kappa ** a_kappa * nu_k) for the note, else 0.
    """
    nu = N_VERSIONS[note] ** a_nu
    p_note = (1 - rho) / (kappa ** a_kappa * nu)
    is_ord = story == 0
    is_note = (story == note) & (note != 0)
    return np.where(is_ord, rho, np.where(is_note, p_note, 0.0))


def omega(note, rho, kappa, a_kappa, a_nu):
    """Prior odds of the note's story against ORD (closed form, for checks)."""
    return (1 - rho) / (rho * kappa ** a_kappa * N_VERSIONS[note] ** a_nu)


if __name__ == "__main__":
    from model.config import A_KAPPA, A_NU, KAPPA, RHO
    from model.stories import NAMES, Story

    rho = float(RHO)
    args = (rho, KAPPA, A_KAPPA, A_NU)
    P = jax.vmap(lambda n: jax.vmap(lambda s: prior(s, n, *args))(Story))(Story)  # (note, story)

    ratio = np.diagonal(P)[1:] / P[1:, 0]
    print("prior odds match omega:", bool(np.allclose(ratio, omega(Story[1:], *args))))
    print("only ORD and the note get weight:", bool(((P > 0).sum(axis=1)[1:] == 2).all()))
    print("silence puts all weight on ORD:", bool((P[0] > 0).sum() == 1))
    for name in ["reset(4)", "other(5,6)", "pattern(3)"]:
        n = NAMES.index(name)
        print(f"  {name:11s} p = {float(P[n, n]):.4f}   omega = {float(omega(n, *args)):.4f}")
