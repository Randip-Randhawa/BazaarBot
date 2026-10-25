import csv
import math

import pytest
import torch

from bazaar.baseline import RuleBasedBuyer
from bazaar.dqn_agent import DQNAgent
from bazaar.env import NO_DEAL_PENALTY, STEP_COST
from bazaar.evaluation import EpisodeResult, evaluate, evaluate_dqn, mean_and_std, run_episode, save_csv, summarize


class ScriptedEnv:
    """Stand-in env that replays fixed (reward, terminated, truncated, info) steps."""

    def __init__(self, steps):
        self.steps = steps

    def reset(self, seed=None):
        self.remaining = list(self.steps)
        return 0, {}

    def step(self, action):
        reward, terminated, truncated, info = self.remaining.pop(0)
        return 0, reward, terminated, truncated, info


def always(action):
    return lambda obs: action


def test_episode_ending_in_a_deal():
    env = ScriptedEnv([(-0.01, False, False, {}), (0.19, True, False, {"deal_price": 0.8})])
    result = run_episode(env, always(0), seed=0)
    assert result == EpisodeResult(reward=pytest.approx(0.18), length=2, deal_price=0.8)
    assert result.surplus == pytest.approx(0.2)


def test_episode_ending_without_a_deal():
    env = ScriptedEnv([(-0.31, True, False, {"outcome": "buyer_walked"})])
    result = run_episode(env, always(9), seed=0)
    assert result.deal_price is None
    assert result.surplus == 0.0
    assert result.length == 1


def test_truncation_ends_the_episode_and_keeps_its_reward():
    env = ScriptedEnv([(-0.01, False, False, {}), (-0.01, False, True, {}), (5.0, True, False, {})])
    result = run_episode(env, always(4), seed=0)
    assert result.length == 2
    assert result.reward == pytest.approx(-0.02)


def test_summary_metrics():
    results = [
        EpisodeResult(reward=0.18, length=2, deal_price=0.8),
        EpisodeResult(reward=0.27, length=3, deal_price=0.7),
        EpisodeResult(reward=-0.31, length=1, deal_price=None),
    ]
    summary = summarize(results)
    assert summary["episodes"] == 3
    assert summary["deals"] == 2
    assert summary["deal_rate"] == pytest.approx(2 / 3)
    assert summary["avg_reward"] == pytest.approx(0.14 / 3)
    assert summary["avg_deal_price"] == pytest.approx(0.75)  # the failed episode has no price
    assert summary["avg_surplus"] == pytest.approx((0.2 + 0.3 + 0.0) / 3)  # failed episode counts as 0
    assert summary["avg_length"] == pytest.approx(2.0)


def test_mean_and_sample_std():
    mean, std = mean_and_std([1.0, 2.0, 3.0, 4.0])
    assert mean == pytest.approx(2.5)
    assert std == pytest.approx(1.2909944)  # sample std, ddof=1


def test_same_seeds_give_same_results():
    policy = RuleBasedBuyer().select_action
    assert evaluate(policy, 50, first_seed=123) == evaluate(policy, 50, first_seed=123)
    assert evaluate(policy, 50, first_seed=123) != evaluate(policy, 50, first_seed=456)


def test_reward_matches_surplus_minus_haggling_costs_for_both_policies():
    agent = DQNAgent(seed=0)
    for results in (evaluate(RuleBasedBuyer().select_action, 100, 0), evaluate_dqn(agent, 100, 0)):
        for r in results:
            no_deal = r.deal_price is None
            expected = r.surplus - STEP_COST * r.length - NO_DEAL_PENALTY * no_deal
            assert r.reward == pytest.approx(expected, abs=1e-6)


def test_dqn_evaluation_is_greedy_and_leaves_the_agent_unchanged():
    agent = DQNAgent(seed=0, epsilon=1.0)  # fully random if exploration were used
    params_before = {k: v.clone() for k, v in agent.online_net.state_dict().items()}
    rng_before = agent.rng.bit_generator.state

    def fail(*args):
        raise AssertionError("evaluation must not train the agent")

    agent.update = fail
    first = evaluate_dqn(agent, 50, first_seed=7)

    assert first == evaluate_dqn(agent, 50, first_seed=7)
    assert agent.rng.bit_generator.state == rng_before  # no exploration draws
    assert agent.epsilon == 1.0
    assert agent.online_net.training  # mode restored afterwards
    assert len(agent.optimizer.state) == 0  # no optimizer step ever ran
    for name, value in agent.online_net.state_dict().items():
        assert torch.equal(value, params_before[name])


def test_surplus_captured_compares_with_paying_exactly_the_minimum():
    results = [
        EpisodeResult(reward=0.19, length=2, deal_price=0.8, reservation_price=0.7),  # got 0.2 of a possible 0.3
        EpisodeResult(reward=-0.31, length=1, deal_price=None, reservation_price=0.75),  # got 0 of 0.25
    ]
    assert summarize(results)["surplus_captured"] == pytest.approx(0.2 / 0.55)


def test_surplus_captured_is_unknown_without_the_minimum():
    results = [EpisodeResult(reward=0.19, length=2, deal_price=0.8)]
    assert math.isnan(summarize(results)["surplus_captured"])


def test_real_episodes_record_the_revealed_minimum():
    for r in evaluate(RuleBasedBuyer().select_action, 50, first_seed=0):
        assert 0.6 <= r.reservation_price <= 0.9
        assert r.surplus <= r.best_surplus + 1e-9  # nobody pays less than the minimum


def test_save_csv_writes_a_header_and_rows(tmp_path):
    path = tmp_path / "out" / "table.csv"
    save_csv(path, [{"policy": "a", "reward": 0.1}, {"policy": "b", "reward": -0.2}])
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows == [{"policy": "a", "reward": "0.1"}, {"policy": "b", "reward": "-0.2"}]
