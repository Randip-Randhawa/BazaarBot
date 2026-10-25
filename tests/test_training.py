import csv

from bazaar.dqn_agent import DQNAgent
from train import train
from tune_baseline import tune


def test_short_training_run_returns_an_agent_and_writes_its_learning_curve(tmp_path):
    curve = tmp_path / "curve.csv"
    agent = train(episodes=3, seed=0, show_progress=False, curve_path=curve)

    assert isinstance(agent, DQNAgent)
    with open(curve, newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1  # a run shorter than the validation interval is checked once, at the end
    assert rows[0]["episode"] == "3"
    assert set(rows[0]) == {"episode", "env_steps", "epsilon", "validation_reward", "validation_deal_rate"}


def test_saved_agent_loads_with_the_same_choices(tmp_path):
    agent = DQNAgent(seed=0)
    agent.save(tmp_path / "model.pt")
    loaded = DQNAgent.load(tmp_path / "model.pt")
    for weights, loaded_weights in zip(agent.online_net.parameters(), loaded.online_net.parameters()):
        assert weights.equal(loaded_weights)


def test_baseline_tuning_ranks_every_setting_best_first():
    rows = tune([0.4, 0.6], [0.1, 0.25], episodes=20, first_seed=0)
    assert len(rows) == 4
    rewards = [row["validation_reward"] for row in rows]
    assert rewards == sorted(rewards, reverse=True)
