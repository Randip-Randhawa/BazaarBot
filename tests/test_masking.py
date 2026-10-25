import numpy as np
import pytest
import torch
from torch import nn

from bazaar.dqn_agent import DQNAgent
from bazaar.env import (
    ACCEPT,
    CONCEDE_ACTIONS,
    N_ACTIONS,
    OBS_LAST_OFFER,
    OBS_SIZE,
    OPEN_ACTIONS,
    WALK_AWAY,
    action_mask,
)

BEFORE_OFFER = {*OPEN_ACTIONS, ACCEPT, WALK_AWAY}
AFTER_OFFER = {*CONCEDE_ACTIONS, ACCEPT, WALK_AWAY}


def observation(last_offer):
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    obs[OBS_LAST_OFFER] = last_offer
    return obs


class FixedQ(nn.Module):
    """A stand-in network that gives the same Q-values for every observation."""

    def __init__(self, q_values):
        super().__init__()
        self.q_values = torch.tensor(q_values, dtype=torch.float32)

    def forward(self, obs):
        return self.q_values.expand(*obs.shape[:-1], N_ACTIONS).clone()


def test_before_the_first_offer_only_opening_accepting_and_walking_are_valid():
    assert set(np.flatnonzero(action_mask(observation(0.0)))) == BEFORE_OFFER


def test_after_an_offer_only_conceding_accepting_and_walking_are_valid():
    assert set(np.flatnonzero(action_mask(observation(0.45)))) == AFTER_OFFER


def test_mask_works_on_a_batch():
    batch = np.stack([observation(0.0), observation(0.5), observation(0.0)])
    masks = action_mask(batch)
    assert masks.shape == (3, N_ACTIONS)
    assert set(np.flatnonzero(masks[0])) == BEFORE_OFFER
    assert set(np.flatnonzero(masks[1])) == AFTER_OFFER


def test_exploration_only_picks_valid_actions():
    agent = DQNAgent(seed=0, epsilon=1.0)
    for obs, valid in ((observation(0.0), BEFORE_OFFER), (observation(0.5), AFTER_OFFER)):
        picked = {agent.select_action(obs) for _ in range(500)}
        assert picked == valid  # every valid action gets tried, and nothing else


def test_greedy_choice_ignores_invalid_actions():
    agent = DQNAgent(seed=0)
    q_values = [1.0] * N_ACTIONS
    q_values[CONCEDE_ACTIONS[2]] = 9.0  # best overall, but invalid before an offer
    q_values[OPEN_ACTIONS[1]] = 5.0
    agent.online_net = FixedQ(q_values)
    assert agent.greedy_action(observation(0.0)) == OPEN_ACTIONS[1]
    assert agent.greedy_action(observation(0.5)) == CONCEDE_ACTIONS[2]


def test_td_target_uses_the_best_valid_next_action():
    agent = DQNAgent(seed=0, gamma=0.9)
    q_values = [5.0] * len(OPEN_ACTIONS) + [1.0] * len(CONCEDE_ACTIONS) + [2.0, 0.0]
    agent.target_net = FixedQ(q_values)

    next_states = np.stack([observation(0.5), observation(0.0), observation(0.5)])
    targets = agent.td_targets(rewards=[0.1, 0.1, 0.1], next_states=next_states, dones=[0, 0, 1])

    # After an offer the opening actions (Q = 5) are invalid, so the best is
    # accepting (Q = 2); before an offer they are valid; at the end, no future.
    assert targets.tolist() == pytest.approx([0.1 + 0.9 * 2.0, 0.1 + 0.9 * 5.0, 0.1])
