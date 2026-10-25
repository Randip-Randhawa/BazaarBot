import itertools
from collections import Counter

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from bazaar import env as bazaar_env
from bazaar.baseline import RandomBuyer, RuleBasedBuyer
from bazaar.env import (
    CONCEDE_ACTIONS,
    DESPERATE_RATE_GROWTH,
    DESPERATE_START_RATE_RANGE,
    MAX_CONCESSION_RATE,
    OBS_ASK,
    OBS_CALLED_BACK_FLAG,
    OBS_FINAL_PRICE_FLAG,
    OBS_LAST_CONCESSION,
    OBS_LAST_OFFER,
    OBS_SIZE,
    OPEN_ACTIONS,
    PERSONALITIES,
    STUBBORN_RATE_RANGE,
    TOURIST_MARKUP,
    WALK_AWAY,
    BazaarEnv,
)
from bazaar.evaluation import EpisodeResult, evaluate, summarize_by_personality

OPEN_30, OPEN_40, OPEN_50 = OPEN_ACTIONS[0], OPEN_ACTIONS[1], OPEN_ACTIONS[2]
HOLD = CONCEDE_ACTIONS[0]  # concede 0%, i.e. repeat the last offer
TOLERANCE = 1e-4  # observations are float32


@pytest.fixture
def no_noise(monkeypatch):
    """Sellers concede at exactly their rate, so the rules can be checked exactly."""
    monkeypatch.setattr(bazaar_env, "RATE_NOISE", (1.0, 1.0))


def sellers(personality, count):
    """The first `count` sellers with this personality, as (env, first observation)."""
    env = BazaarEnv()
    found = 0
    for seed in itertools.count():
        obs, _ = env.reset(seed=seed)
        if env._personality == personality:
            yield env, obs
            found += 1
            if found == count:
                return


def minimum(env):
    """The seller's hidden minimum as a fraction of V (tests may peek)."""
    return env._reservation_price / env._item_value


def at_minimum(env, obs):
    """Whether the observed ask is exactly the seller's minimum. The env divides
    in float64 and stores float32, so the same steps give the same value."""
    return obs[OBS_ASK] <= np.float32(minimum(env))


def measured_rates(env, obs):
    """Open at 40% (never insulting), keep repeating that offer, and measure the
    share of the ask-offer gap the seller gives up each round, until their
    minimum price gets in the way."""
    rates = []
    action = OPEN_40
    done = False
    while not done:
        ask_before = obs[OBS_ASK]
        obs, _, done, _, _ = env.step(action)
        if at_minimum(env, obs):
            break
        rates.append(obs[OBS_LAST_CONCESSION] / (ask_before - obs[OBS_LAST_OFFER]))
        action = HOLD
    return rates


def within(value, bounds):
    return bounds[0] - TOLERANCE <= value <= bounds[1] + TOLERANCE


# ------------------------------------------------------------- the seller mix


def test_all_personalities_are_equally_likely():
    env = BazaarEnv()
    counts = Counter()
    for seed in range(4000):
        env.reset(seed=seed)
        counts[env._personality] += 1
    assert set(counts) == set(PERSONALITIES)
    for personality in PERSONALITIES:
        assert 0.21 < counts[personality] / 4000 < 0.29


def test_same_seed_gives_the_same_seller():
    a, b = BazaarEnv(), BazaarEnv()
    a.reset(seed=5)
    b.reset(seed=5)
    assert a._personality == b._personality
    assert a._concession_rate == b._concession_rate


def test_passes_gymnasium_env_checker():
    check_env(BazaarEnv(), skip_render_check=True)


def test_observations_stay_inside_the_observation_space():
    env = BazaarEnv()
    buyer = RandomBuyer(seed=0)
    for seed in range(500):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            assert env.observation_space.contains(obs)
            obs, _, terminated, truncated, _ = env.step(buyer.select_action(obs))
            done = terminated or truncated
        assert env.observation_space.contains(obs)


# ------------------------------------------------------- how each one concedes


def test_stubborn_sellers_give_up_a_small_constant_share(no_noise):
    for env, obs in sellers("stubborn", 20):
        rates = measured_rates(env, obs)
        assert len(rates) >= 2
        assert within(rates[0], STUBBORN_RATE_RANGE)
        assert np.allclose(rates, rates[0], atol=TOLERANCE)


def test_desperate_sellers_give_up_more_every_round(no_noise):
    for env, obs in sellers("desperate", 20):
        rates = measured_rates(env, obs)
        assert within(rates[0], DESPERATE_START_RATE_RANGE)
        for before, after in zip(rates, rates[1:]):
            expected = min(before + DESPERATE_RATE_GROWTH, MAX_CONCESSION_RATE)
            assert abs(after - expected) < TOLERANCE


def test_noise_varies_the_concessions_but_stays_within_its_range():
    low, high = bazaar_env.RATE_NOISE
    for env, obs in sellers("stubborn", 20):
        rates = measured_rates(env, obs)
        assert within(min(rates), (STUBBORN_RATE_RANGE[0] * low, np.inf))
        assert within(max(rates), (0.0, STUBBORN_RATE_RANGE[1] * high))
        assert len(set(np.round(rates, 6))) > 1


def test_a_bluffer_holds_its_price_and_claims_it_is_final(monkeypatch):
    monkeypatch.setattr(bazaar_env, "BLUFF_CHANCE", 1.0)
    for env, obs in sellers("bluffer", 10):
        opening = obs[OBS_ASK]
        obs, _, done, _, _ = env.step(OPEN_40)
        assert not done
        assert obs[OBS_ASK] == opening
        assert obs[OBS_FINAL_PRICE_FLAG] == 1.0
        assert obs[OBS_ASK] > minimum(env)  # so the claim was a bluff


def test_a_bluffer_that_does_not_bluff_concedes_normally(monkeypatch):
    monkeypatch.setattr(bazaar_env, "BLUFF_CHANCE", 0.0)
    for env, obs in sellers("bluffer", 10):
        opening = obs[OBS_ASK]
        obs, _, _, _, _ = env.step(OPEN_40)
        assert obs[OBS_ASK] < opening
        assert obs[OBS_FINAL_PRICE_FLAG] == at_minimum(env, obs)


def test_only_bluffers_claim_a_final_price_above_their_minimum():
    env = BazaarEnv()
    buyer = RuleBasedBuyer()
    for seed in range(300):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(buyer.select_action(obs))
            done = terminated or truncated
            if env._personality != "bluffer":
                assert obs[OBS_FINAL_PRICE_FLAG] == at_minimum(env, obs)


def test_tourist_spotter_raises_its_price_after_an_eager_opening():
    raised = 0
    for env, obs in sellers("tourist_spotter", 20):
        opening = obs[OBS_ASK]
        obs, _, done, _, info = env.step(OPEN_50)
        if done:  # half the ask was already above their minimum, so they took it
            assert info["outcome"] == "deal"
            continue
        assert obs[OBS_ASK] == pytest.approx(opening * (1 + TOURIST_MARKUP), rel=1e-6)
        assert obs[OBS_LAST_CONCESSION] < 0
        raised += 1
    assert raised > 0


def test_tourist_spotter_concedes_after_a_low_opening():
    for env, obs in sellers("tourist_spotter", 10):
        opening = obs[OBS_ASK]
        obs, _, _, _, _ = env.step(OPEN_40)
        assert obs[OBS_ASK] < opening


def test_other_sellers_never_raise_their_price():
    for personality in ("stubborn", "desperate", "bluffer"):
        for env, obs in sellers(personality, 10):
            opening = obs[OBS_ASK]
            obs, _, _, _, _ = env.step(OPEN_50)
            assert obs[OBS_ASK] <= opening


def test_an_insulting_offer_costs_a_round_of_patience():
    insulted = calm = 0
    for env, obs in sellers("stubborn", 60):
        patience = env._patience
        offer = 0.3 * obs[OBS_ASK]
        env.step(OPEN_30)
        if offer < 0.5 * minimum(env) - TOLERANCE:
            assert env._patience == patience - 1
            insulted += 1
        elif offer > 0.5 * minimum(env) + TOLERANCE:
            assert env._patience == patience
            calm += 1
    assert insulted > 0 and calm > 0  # both cases were actually checked


# --------------------------------------------------------------- call-backs


def test_a_seller_can_call_the_buyer_back_once(monkeypatch):
    monkeypatch.setattr(bazaar_env, "CALLBACK_CHANCE", 100.0)  # always call back
    for env, obs in sellers("stubborn", 10):
        obs, _, _, _, _ = env.step(OPEN_40)
        ask, offer = obs[OBS_ASK], obs[OBS_LAST_OFFER]

        obs, _, done, _, info = env.step(WALK_AWAY)
        assert not done
        assert obs[OBS_CALLED_BACK_FLAG] == 1.0
        assert obs[OBS_ASK] == pytest.approx(max(minimum(env), (ask + offer) / 2), rel=1e-5)

        obs, _, done, _, info = env.step(WALK_AWAY)
        assert done
        assert info["outcome"] == "buyer_walked"


def test_no_call_back_before_any_offer_or_when_unlucky(monkeypatch):
    monkeypatch.setattr(bazaar_env, "CALLBACK_CHANCE", 100.0)
    env, _ = next(sellers("desperate", 1))
    _, _, done, _, _ = env.step(WALK_AWAY)
    assert done  # there was no offer to come back to

    monkeypatch.setattr(bazaar_env, "CALLBACK_CHANCE", 0.0)
    env, _ = next(sellers("desperate", 1))
    env.step(OPEN_40)
    _, _, done, _, info = env.step(WALK_AWAY)
    assert done
    assert info["outcome"] == "buyer_walked"


# ------------------------------------------------------- what the buyer sees


def test_seller_secrets_are_revealed_only_when_the_negotiation_ends():
    env = BazaarEnv()
    obs, info = env.reset(seed=3)
    assert obs.shape == (OBS_SIZE,)
    assert info["item_value"] == env._item_value
    assert "personality" not in info and "reservation_price" not in info

    action = OPEN_40
    done = False
    while not done:
        obs, _, done, _, info = env.step(action)
        if not done:
            assert "personality" not in info and "reservation_price" not in info
        action = HOLD
    assert info["personality"] == env._personality
    assert info["reservation_price"] == pytest.approx(minimum(env))


# --------------------------------------------------------------- evaluation


def test_results_are_grouped_by_personality():
    results = [
        EpisodeResult(reward=0.18, length=2, deal_price=0.8, personality="stubborn"),
        EpisodeResult(reward=-0.31, length=1, deal_price=None, personality="desperate"),
        EpisodeResult(reward=0.27, length=3, deal_price=0.7, personality="stubborn"),
    ]
    by_personality = summarize_by_personality(results)
    assert by_personality["stubborn"]["episodes"] == 2
    assert by_personality["stubborn"]["deals"] == 2
    assert by_personality["desperate"]["deals"] == 0


def test_evaluation_records_each_sellers_personality():
    results = evaluate(RuleBasedBuyer().select_action, 50, first_seed=0)
    assert {r.personality for r in results} == set(PERSONALITIES)
