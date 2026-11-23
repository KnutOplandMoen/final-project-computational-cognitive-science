"""
Polya-urn stopping game.

Rules (as specified, with the interpretive choices flagged):
  * Endowment 50$. Each round: QUIT or DRAW one ball.
  * white -> +0,  green -> +10,  red -> -50; the drawn ball is returned with one
    extra ball of the same colour (draw 1, put back 2) => classic Polya urn.
  * QUIT: repay the 50$, payoff = wealth - 50.
  * BUST: wealth <= 0 -> game over.  Choice: limited liability, payoff = -50
    (BUST_PAYOFF below; set to None to use wealth-50 instead).
  * A horizon T is needed: once green dominates, every draw has positive EV forever.

State is the vector of counts drawn so far (dw, dr, dg). Wealth = 50 + 10dg - 50dr.
Because the Polya urn is exchangeable, the ORDER of draws never matters, only counts.
"""
import numpy as np
from functools import lru_cache
from scipy.special import gammaln

ENDOW, G_REW, R_REW, N0 = 50, 10, -50, 100
BUST_PAYOFF = -50          # limited liability


def wealth(dr, dg):
    return ENDOW + G_REW * dg + R_REW * dr


# ---------------------------------------------------------------- known urn
def solve_known(W0, R0, G0, T):
    @lru_cache(maxsize=None)
    def V(dw, dr, dg):
        w = wealth(dr, dg)
        if w <= 0:
            return BUST_PAYOFF if BUST_PAYOFF is not None else w - ENDOW
        quit_v = w - ENDOW
        t = dw + dr + dg
        if t >= T:
            return quit_v
        N = N0 + t
        pw, pr, pg = (W0 + dw) / N, (R0 + dr) / N, (G0 + dg) / N
        cont = pw * V(dw + 1, dr, dg) + pr * V(dw, dr + 1, dg) + pg * V(dw, dr, dg + 1)
        return max(quit_v, cont)

    def act(dw, dr, dg):
        w = wealth(dr, dg)
        t = dw + dr + dg
        if w <= 0 or t >= T:
            return "stop"
        return "draw" if V(dw, dr, dg) > w - ENDOW + 1e-12 else "quit"

    return V, act


# ---------------------------------------------------------------- unknown urn
def solve_bayes(T, prior="uniform"):
    """Composition unknown; uniform prior over all (W0,R0,G0) summing to 100."""
    comps = np.array([(w, r, N0 - w - r) for w in range(N0 + 1) for r in range(N0 + 1 - w)], float)
    logprior = np.zeros(len(comps))

    def lrf(a, k):  # log rising factorial a^(k) ; a=0,k>0 -> -inf
        with np.errstate(divide="ignore"):
            return np.where(a > 0, gammaln(a + k) - gammaln(np.maximum(a, 1e-300)),
                            np.where(k == 0, 0.0, -np.inf))

    @lru_cache(maxsize=None)
    def predictive(dw, dr, dg):
        ll = logprior + lrf(comps[:, 0], dw) + lrf(comps[:, 1], dr) + lrf(comps[:, 2], dg)
        p = np.exp(ll - ll.max()); p /= p.sum()
        N = N0 + dw + dr + dg
        return tuple(p @ ((comps + [dw, dr, dg]) / N))

    @lru_cache(maxsize=None)
    def V(dw, dr, dg):
        w = wealth(dr, dg)
        if w <= 0:
            return BUST_PAYOFF if BUST_PAYOFF is not None else w - ENDOW
        quit_v = w - ENDOW
        if dw + dr + dg >= T:
            return quit_v
        pw, pr, pg = predictive(dw, dr, dg)
        cont = pw * V(dw + 1, dr, dg) + pr * V(dw, dr + 1, dg) + pg * V(dw, dr, dg + 1)
        return max(quit_v, cont)

    return V, predictive


# ---------------------------------------------------------------- Q-values
def q_values(V, p, dw=0, dr=0, dg=0):
    """Q(quit), Q(draw) at state (dw,dr,dg). p = predictive (pw,pr,pg) at that state."""
    pw, pr, pg = p
    q_quit = wealth(dr, dg) - ENDOW
    q_draw = pw * V(dw + 1, dr, dg) + pr * V(dw, dr + 1, dg) + pg * V(dw, dr, dg + 1)
    return q_quit, q_draw

# ---------------------------------------------------------------- Monte Carlo
def simulate(W0, R0, G0, policy, T, n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        c = [W0, R0, G0]; d = [0, 0, 0]
        while True:
            w = wealth(d[1], d[2])
            if w <= 0:
                out[i] = BUST_PAYOFF; break
            if sum(d) >= T or policy(*d) != "draw":
                out[i] = w - ENDOW; break
            k = rng.choice(3, p=np.array(c) / sum(c))
            c[k] += 1; d[k] += 1
    return out


# ---------------------------------------------------------------- demo
if __name__ == "__main__":
    import sys; sys.setrecursionlimit(100_000)
    T = 100
    print("Known composition (W,R,G): first-action Q-values, V0 = max(Q)")
    for comp in [(60, 10, 30), (45, 10, 45), (40, 10, 50), (70, 5, 25), (50, 5, 45)]:
        V, act = solve_known(*comp, T)
        qq, qd = q_values(V, [c / N0 for c in comp])
        print(f"  {comp}: Q_quit={qq:6.2f}  Q_draw={qd:7.2f}  V0={V(0,0,0):7.2f}  {act(0,0,0)}")
    V, act = solve_known(40, 10, 50, T)
    mc = simulate(40, 10, 50, act, T, n=20_000)
    print(f"MC check (40,10,50): {mc.mean():.2f} vs DP {V(0,0,0):.2f}; P(bust)={np.mean(mc==BUST_PAYOFF):.2f}")
    # path dependence: same wealth & round, different optimal action
    V, act = solve_known(60, 10, 30, T)
    print("Urn (60,10,30), wealth 180 after 19 draws:",
          "18G+1R ->", act(0, 1, 18), "| 13G+6W ->", act(6, 0, 13))
    Vb, pred = solve_bayes(60)
    a = lambda dw, dr, dg: "draw" if Vb(dw, dr, dg) > wealth(dr, dg) - ENDOW + 1e-9 else "quit"
    qq, qd = q_values(Vb, pred(0, 0, 0))
    print(f"Unknown urn (uniform prior), T=60: Q_quit={qq:.2f} Q_draw={qd:.2f} V0={Vb(0,0,0):.2f}; "
          f"wealth 50 after 6 draws: 5G+1R -> {a(0,1,5)} | 6W -> {a(6,0,0)}")
