"""Tests for BazaarEnv's observation and action space definitions.

These deliberately do not call reset() or step(): seller behaviour, rewards
and termination are not part of the agreed spec yet.
"""

import gymnasium as gym
import numpy as np

from bazaar import env as bazaar_env
from bazaar import BazaarEnv

OBS_INDEX_NAMES = (
    "OBS_ASK",
    "OBS_OPENING_ASK",
    "OBS_LAST_OFFER",
    "OBS_ROUND",
    "OBS_LAST_CONCESSION",
    "OBS_AVG_CONCESSION",
    "OBS_FINAL_PRICE_FLAG",
    "OBS_CALLED_BACK_FLAG",
)
PRICE_FEATURES = ("OBS_ASK", "OBS_OPENING_ASK", "OBS_LAST_OFFER", "OBS_LAST_CONCESSION", "OBS_AVG_CONCESSION")
UNIT_FEATURES = ("OBS_ROUND", "OBS_FINAL_PRICE_FLAG", "OBS_CALLED_BACK_FLAG")
HIDDEN_TERMS = ("reservation", "minimum", "personality", "concession_rate", "patience")


def test_env_imports():
    assert issubclass(BazaarEnv, gym.Env)


def test_observation_space_type_and_shape():
    space = BazaarEnv().observation_space
    assert isinstance(space, gym.spaces.Box)
    assert space.shape == (bazaar_env.OBS_SIZE,) == (8,)
    assert space.dtype == np.float32


def test_observation_indices_cover_the_vector_once():
    indices = sorted(getattr(bazaar_env, name) for name in OBS_INDEX_NAMES)
    assert indices == list(range(bazaar_env.OBS_SIZE))
    assert len(bazaar_env.OBS_FEATURES) == bazaar_env.OBS_SIZE


def test_observation_bounds():
    space = BazaarEnv().observation_space
    assert np.isclose(bazaar_env.MAX_PRICE, 2.25)
    assert np.all(space.low == 0.0)
    for name in PRICE_FEATURES:
        assert np.isclose(space.high[getattr(bazaar_env, name)], bazaar_env.MAX_PRICE)
    for name in UNIT_FEATURES:
        assert space.high[getattr(bazaar_env, name)] == 1.0


def test_hidden_seller_information_is_not_observed():
    names = [name.lower() for name in bazaar_env.OBS_FEATURES]
    names += [name.lower() for name in dir(bazaar_env) if name.startswith("OBS_")]
    for name in names:
        for term in HIDDEN_TERMS:
            assert term not in name, f"{name!r} looks like hidden seller information"


def test_action_space_type_and_count():
    space = BazaarEnv().action_space
    assert isinstance(space, gym.spaces.Discrete)
    assert space.n == bazaar_env.N_ACTIONS == 10
    assert space.start == 0


def test_action_groups_partition_the_action_space():
    actions = [*bazaar_env.OPEN_ACTIONS, *bazaar_env.CONCEDE_ACTIONS, bazaar_env.ACCEPT, bazaar_env.WALK_AWAY]
    assert sorted(actions) == list(range(bazaar_env.N_ACTIONS))
    assert len(bazaar_env.OPEN_FRACTIONS) == len(bazaar_env.OPEN_ACTIONS)
    assert len(bazaar_env.CONCEDE_FRACTIONS) == len(bazaar_env.CONCEDE_ACTIONS)


def test_every_action_has_a_documented_meaning():
    meanings = bazaar_env.ACTION_MEANINGS
    assert len(meanings) == bazaar_env.N_ACTIONS
    assert all(isinstance(m, str) and m for m in meanings)
    assert len(set(meanings)) == len(meanings)
