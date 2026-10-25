import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_space_limit
from gymnasium.utils.passive_env_checker import (
    check_action_space,
    check_observation_space,
)

from bazaar import env as env_module
from bazaar.env import (
    ACCEPT,
    ACTION_MEANINGS,
    CONCEDE_ACTIONS,
    CONCEDE_FRACTIONS,
    MAX_PRICE,
    N_ACTIONS,
    OBS_ASK,
    OBS_AVG_CONCESSION,
    OBS_CALLED_BACK_FLAG,
    OBS_FINAL_PRICE_FLAG,
    OBS_LAST_CONCESSION,
    OBS_LAST_OFFER,
    OBS_OPENING_ASK,
    OBS_ROUND,
    OBS_SIZE,
    OPEN_ACTIONS,
    OPEN_FRACTIONS,
    WALK_AWAY,
    BazaarEnv,
)

PRICE_FEATURES = [OBS_ASK, OBS_OPENING_ASK, OBS_LAST_OFFER, OBS_LAST_CONCESSION, OBS_AVG_CONCESSION]
UNIT_FEATURES = [OBS_ROUND, OBS_FINAL_PRICE_FLAG, OBS_CALLED_BACK_FLAG]

# Seller attributes the buyer must never observe (IDEAS.md 7.1).
HIDDEN_TERMS = ("reservation", "personality", "patience", "rate", "bluff")


def observation_constants():
    """The OBS_* index constants, skipping OBS_SIZE and non-index ones like OBS_FEATURES."""
    return {
        name: value
        for name, value in vars(env_module).items()
        if name.startswith("OBS_") and name != "OBS_SIZE" and isinstance(value, int)
    }


def test_env_can_be_constructed():
    assert isinstance(BazaarEnv(), gym.Env)


# Space checks that gymnasium's check_env runs before calling reset() and
# step(). The full check_env cannot pass until those are implemented.
def test_spaces_pass_gymnasium_space_checks():
    env = BazaarEnv()
    check_observation_space(env.observation_space)
    check_space_limit(env.observation_space, "observation")
    check_action_space(env.action_space)
    check_space_limit(env.action_space, "action")


# ------------------------------------------------------------- observation


def test_observation_space_shape_and_dtype():
    space = BazaarEnv().observation_space
    assert isinstance(space, gym.spaces.Box)
    assert space.shape == (OBS_SIZE,)
    assert space.dtype == np.float32


def test_every_observation_index_is_defined_exactly_once():
    indices = sorted(observation_constants().values())
    assert indices == list(range(OBS_SIZE))
    assert sorted(PRICE_FEATURES + UNIT_FEATURES) == indices


def test_observation_bounds():
    space = BazaarEnv().observation_space
    concessions = [OBS_LAST_CONCESSION, OBS_AVG_CONCESSION]
    others = [i for i in range(OBS_SIZE) if i not in concessions]
    assert np.all(space.low[concessions] == np.float32(env_module.MIN_CONCESSION))
    assert np.all(space.low[others] == 0.0)
    assert np.all(space.high[PRICE_FEATURES] == np.float32(MAX_PRICE))
    assert np.all(space.high[UNIT_FEATURES] == 1.0)


def test_price_bound_matches_agreed_ranges():
    # Highest opening ask: 2.5 x R with R at most 0.9 V, raised 20% by a tourist-spotter.
    assert MAX_PRICE == pytest.approx(2.7)


def test_observation_excludes_hidden_seller_information():
    # The observation is defined by the OBS_* indices; none may name a hidden
    # seller attribute. Behavioural checks need step() and come later.
    for name in observation_constants():
        assert not any(term in name.lower() for term in HIDDEN_TERMS), name


# ----------------------------------------------------------------- actions


def test_action_space_is_discrete():
    space = BazaarEnv().action_space
    assert isinstance(space, gym.spaces.Discrete)
    assert space.n == N_ACTIONS
    assert space.start == 0


def test_every_action_index_is_used_exactly_once():
    actions = list(OPEN_ACTIONS) + list(CONCEDE_ACTIONS) + [ACCEPT, WALK_AWAY]
    assert sorted(actions) == list(range(N_ACTIONS))


def test_every_action_has_a_documented_meaning():
    assert len(ACTION_MEANINGS) == N_ACTIONS
    assert all(meaning.strip() for meaning in ACTION_MEANINGS)
    assert len(set(ACTION_MEANINGS)) == N_ACTIONS


def test_action_meanings_match_action_groups():
    for action in OPEN_ACTIONS:
        assert ACTION_MEANINGS[action].startswith("open")
    for action in CONCEDE_ACTIONS:
        assert ACTION_MEANINGS[action].startswith("concede")
    assert ACTION_MEANINGS[ACCEPT].startswith("accept")
    assert ACTION_MEANINGS[WALK_AWAY] == "walk away"


def test_action_fractions_match_ideas():
    assert OPEN_FRACTIONS == (0.30, 0.40, 0.50, 0.60)
    assert CONCEDE_FRACTIONS == (0.00, 0.10, 0.25, 0.50)
    assert len(OPEN_FRACTIONS) == len(OPEN_ACTIONS)
    assert len(CONCEDE_FRACTIONS) == len(CONCEDE_ACTIONS)
