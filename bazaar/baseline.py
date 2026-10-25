"""Fixed-strategy buyers, used only as baselines to compare the DQN against."""

import numpy as np

from bazaar.env import (
    ACCEPT,
    CONCEDE_ACTIONS,
    CONCEDE_FRACTIONS,
    OBS_ASK,
    OBS_LAST_OFFER,
    OPEN_ACTIONS,
    OPEN_FRACTIONS,
    action_mask,
)

OPENING_TARGET = 0.6  # first offer, as a fraction of V
CONCESSION_STEP = 0.25  # share of the remaining distance to V given up per offer

# The best settings found by tune_baseline.py on the validation sellers
# (validation reward +0.181, against +0.172 for the hand-set 0.6 / 0.25).
TUNED_OPENING_TARGET = 0.7
TUNED_CONCESSION_STEP = 0.25


def open_at(fraction):
    return OPEN_ACTIONS[OPEN_FRACTIONS.index(fraction)]


def concede(fraction):
    return CONCEDE_ACTIONS[CONCEDE_FRACTIONS.index(fraction)]


def next_target_offer(last_offer, opening_target=OPENING_TARGET, concession_step=CONCESSION_STEP):
    """The offer the strategy wants to make next, in units of V (so V = 1)."""
    if last_offer == 0.0:
        return opening_target
    return last_offer + concession_step * (1.0 - last_offer)


class RuleBasedBuyer:
    """Opens at a target price, then moves part of the way towards V each turn.

    By default it opens at 0.6 V, then moves 25% of the remaining distance
    towards V with each new offer. The environment only offers fixed fractions
    of the ask, so each turn it picks the legal offer closest to its target
    (ties go to the cheaper one). If the seller's ask is already at or below
    the target it accepts instead, since that is no more than it was about to
    offer. It never walks away.

    It reads only the observation: prices there are already divided by V, which
    the buyer knows. The seller's hidden minimum price is never used.
    """

    def __init__(self, opening_target=OPENING_TARGET, concession_step=CONCESSION_STEP):
        self.opening_target = opening_target
        self.concession_step = concession_step

    def select_action(self, obs):
        ask = float(obs[OBS_ASK])
        last_offer = float(obs[OBS_LAST_OFFER])
        target = next_target_offer(last_offer, self.opening_target, self.concession_step)

        if ask <= target:
            return ACCEPT

        if last_offer == 0.0:
            actions, offers = OPEN_ACTIONS, [f * ask for f in OPEN_FRACTIONS]
        else:
            actions, offers = CONCEDE_ACTIONS, [last_offer + f * (ask - last_offer) for f in CONCEDE_FRACTIONS]

        closest = int(np.argmin([abs(offer - target) for offer in offers]))
        return actions[closest]


class TouristBuyer:
    """Pays whatever the seller first asks, like an unprepared tourist."""

    def select_action(self, obs):
        return ACCEPT


class SplitTheDifferenceBuyer:
    """Opens at half the ask, then always meets the seller halfway."""

    def select_action(self, obs):
        if obs[OBS_LAST_OFFER] == 0:
            return open_at(0.50)
        return concede(0.50)


class SteadyConcessionBuyer:
    """Opens at 40% of the ask, then gives up a quarter of the gap every round.

    The original plan was 20% a round, but 25% is the nearest concession the
    environment allows.
    """

    def select_action(self, obs):
        if obs[OBS_LAST_OFFER] == 0:
            return open_at(0.40)
        return concede(0.25)


class RandomBuyer:
    """Picks uniformly among the actions that make sense, as a floor to beat."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs):
        return int(self.rng.choice(np.flatnonzero(action_mask(obs))))
