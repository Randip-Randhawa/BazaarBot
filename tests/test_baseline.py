import numpy as np
import pytest

from bazaar.baseline import (
    RandomBuyer,
    RuleBasedBuyer,
    SplitTheDifferenceBuyer,
    SteadyConcessionBuyer,
    TouristBuyer,
    next_target_offer,
)
from bazaar.env import ACCEPT, CONCEDE_ACTIONS, OBS_ASK, OBS_LAST_OFFER, OBS_SIZE, OPEN_ACTIONS, BazaarEnv, action_mask


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


def test_rule_based_settings_can_be_changed():
    assert next_target_offer(0.0, opening_target=0.4) == pytest.approx(0.40)
    assert next_target_offer(0.6, concession_step=0.5) == pytest.approx(0.80)
    # Ask 1.0 V: a 0.3 V opening target picks 30% of the ask instead of 60%.
    assert RuleBasedBuyer(opening_target=0.3).select_action(observation(ask=1.0)) == OPEN_ACTIONS[0]


def test_tourist_always_accepts():
    assert TouristBuyer().select_action(observation(ask=2.0)) == ACCEPT


def test_split_the_difference_opens_at_half_then_meets_halfway():
    buyer = SplitTheDifferenceBuyer()
    assert buyer.select_action(observation(ask=1.5)) == OPEN_ACTIONS[2]  # 50% of the ask
    assert buyer.select_action(observation(ask=1.2, last_offer=0.75)) == CONCEDE_ACTIONS[3]  # 50% of the gap


def test_steady_concession_opens_at_40_percent_then_gives_up_a_quarter():
    buyer = SteadyConcessionBuyer()
    assert buyer.select_action(observation(ask=1.5)) == OPEN_ACTIONS[1]
    assert buyer.select_action(observation(ask=1.2, last_offer=0.6)) == CONCEDE_ACTIONS[2]


def test_random_buyer_is_reproducible_and_only_picks_valid_actions():
    first, second = RandomBuyer(seed=3), RandomBuyer(seed=3)
    for obs in (observation(ask=1.5), observation(ask=1.2, last_offer=0.6)):
        picks = [first.select_action(obs) for _ in range(200)]
        assert picks == [second.select_action(obs) for _ in range(200)]
        assert all(action_mask(obs)[action] for action in picks)


@pytest.mark.parametrize("buyer", [TouristBuyer(), SplitTheDifferenceBuyer(), SteadyConcessionBuyer(), RandomBuyer(seed=0)])
def test_other_baselines_only_make_valid_moves_in_real_negotiations(buyer):
    env = BazaarEnv()
    for seed in range(200):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(buyer.select_action(obs))
            assert not info["invalid_action"]
            done = terminated or truncated
