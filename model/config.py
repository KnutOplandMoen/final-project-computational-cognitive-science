"""
Every setting of the model, in one place to work from to build model
"""

from fractions import Fraction as F

# ---------------------------------------------------------------- the urn parameters
N_DRAWS = 6                 # draws the listener watches; draw N_DRAWS + 1 is the risky one
RED0, GREEN0 = 1, 1         # starting dist of every urn starts with one of each
ADD_BACK = 1                # a drawn ball goes back with this many more of its colour

# ---------------------------------------------------------------- the risk parameters
GAIN = 15                   # green ball on draw 7 pays G
LOSS = 10                   # red ball on draw 7 costs L
B_NEUTRAL = F(GAIN, GAIN + LOSS)    # risk-neutral threshold b = G/(G+L) = 3/5
B_LOW, B_HIGH = F(3, 10), F(7, 10)  # b uniform on [B_LOW, B_HIGH] when averaging over listeners

# ---------------------------------------------------------------- the prior parameters
KAPPA = 3                   # kinds the listener has in mind (3 = list shown; 1 = only the note)
RHO = F(1, 4)          # prior chance of the ordinary urn
A_KAPPA = 1                 # exponent on kappa: kinds weigh in as kappa ** A_KAPPA
A_NU = 1                    # exponent on n_k: versions in mind nu = n_k ** A_NU (0 = only the named one, 1 = all)

# ---------------------------------------------------------------- the listener's rule parameters
ALPHA_SAMPLER = 1               # graded rule with alpha = 1: adopt with the posterior probability
ALPHA_SHARP = float("inf")      # alpha -> infinity: adopt iff omega * r >= 1 (ties go to the story)