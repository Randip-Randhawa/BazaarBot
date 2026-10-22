import numpy as np
import pytest

from bazaar.baseline import RuleBasedBuyer, next_target_offer
from bazaar.env import ACCEPT, OBS_ASK, OBS_LAST_OFFER, OBS_SIZE, BazaarEnv


def observation(ask, last_offer=0.0):
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    obs[OBS_ASK] = ask
    obs[OBS_LAST_OFFER] = last_offer
    return obs


def test_target_offers_follow_the_concession_formula():
    assert next_target_offer(0.0) == pytest.approx(0.60)
    assert next_target_offer(0.60) == pytest.approx(0.70)  # 0.60 + 0.25 * 0.40
    assert next_target_offer(0.70) == pytest.approx(0.775)  # 0.70 + 0.25 * 0.30
    assert next_target_offer(0.775) == pytest.approx(0.83125)


def test_opening_offer_is_the_legal_one_closest_to_60_percent_of_v():
    buyer = RuleBasedBuyer()
    # Ask 1.5 V: legal openings are 0.45, 0.60, 0.75, 0.90 V -> 40% of the ask.
    assert buyer.select_action(observation(ask=1.5)) == 1
    # Ask 1.0 V: legal openings are 0.30 ... 0.60 V -> 60% of the ask.
    assert buyer.select_action(observation(ask=1.0)) == 3


def test_concession_is_the_legal_one_closest_to_the_target():
    buyer = RuleBasedBuyer()
    # Last offer 0.6 V, ask 1.0 V: legal offers are 0.60, 0.64, 0.70, 0.80 V and
    # the target is 0.70 V, which is the 25% concession.
    assert buyer.select_action(observation(ask=1.0, last_offer=0.6)) == 6


def test_accepts_when_the_ask_is_no_more_than_the_target():
    buyer = RuleBasedBuyer()
    # Target after 0.8 V is 0.85 V, and the seller already asks 0.84 V.
    assert buyer.select_action(observation(ask=0.84, last_offer=0.8)) == ACCEPT


def test_same_observation_gives_same_action():
    buyer = RuleBasedBuyer()
    obs = observation(ask=1.3, last_offer=0.7)
    assert len({buyer.select_action(obs) for _ in range(5)}) == 1


def test_actions_are_legal_in_real_negotiations():
    env = BazaarEnv()
    buyer = RuleBasedBuyer()
    for seed in range(200):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            action = buyer.select_action(obs)
            assert env.action_space.contains(action)
            # step() raises if called after the negotiation is over, so finishing
            # this loop also shows the buyer never acts on a finished episode.
            obs, _, terminated, truncated, info = env.step(action)
            assert not info["invalid_action"]
            done = terminated or truncated
