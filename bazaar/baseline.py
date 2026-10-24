import numpy as np

from bazaar.env import (
    ACCEPT,
    CONCEDE_ACTIONS,
    CONCEDE_FRACTIONS,
    OBS_ASK,
    OBS_LAST_OFFER,
    OPEN_ACTIONS,
    OPEN_FRACTIONS,
)

OPENING_TARGET = 0.6  # first offer, as a fraction of V
CONCESSION_STEP = 0.25  # share of the remaining distance to V given up per offer


def next_target_offer(last_offer):
    """The offer the strategy wants to make next, in units of V (so V = 1)."""
    if last_offer == 0.0:
        return OPENING_TARGET
    return last_offer + CONCESSION_STEP * (1.0 - last_offer)


class RuleBasedBuyer:
    """Fixed-strategy buyer used only as a baseline for the DQN.

    It opens at 0.6 V, then moves 25% of the remaining distance towards V with
    each new offer. The environment only offers fixed fractions of the ask, so
    each turn it picks the legal offer closest to its target (ties go to the
    cheaper one). If the seller's ask is already at or below the target it
    accepts instead, since that is no more than it was about to offer. It never
    walks away.

    It reads only the observation: prices there are already divided by V, which
    the buyer knows. The seller's hidden minimum price is never used.
    """

    def select_action(self, obs):
        ask = float(obs[OBS_ASK])
        last_offer = float(obs[OBS_LAST_OFFER])
        target = next_target_offer(last_offer)

        if ask <= target:
            return ACCEPT

        if last_offer == 0.0:
            actions, offers = OPEN_ACTIONS, [f * ask for f in OPEN_FRACTIONS]
        else:
            actions, offers = CONCEDE_ACTIONS, [last_offer + f * (ask - last_offer) for f in CONCEDE_FRACTIONS]

        closest = int(np.argmin([abs(offer - target) for offer in offers]))
        return actions[closest]
