"""Tests for the dialogue helpers and the terminal demo, with scripted players."""

import pytest

from bazaar.baseline import RuleBasedBuyer
from bazaar.dialogue import negotiation_dialogue
from bazaar.env import ACCEPT, CONCEDE_ACTIONS, OBS_CALLED_BACK_FLAG, OBS_FINAL_PRICE_FLAG, OPEN_ACTIONS, WALK_AWAY, BazaarEnv
from play import sell

OPEN_40 = OPEN_ACTIONS[1]


def scripted(actions):
    """A buyer policy that plays fixed actions and remembers what it saw."""
    seen = []

    def policy(obs):
        seen.append(obs.copy())
        return actions[len(seen) - 1]

    return policy, seen


def scripted_input(replies):
    replies = iter(replies)
    return lambda prompt: next(replies)


def test_dialogue_reads_like_a_negotiation():
    env = BazaarEnv()
    for seed in range(30):
        lines, info = negotiation_dialogue(env, RuleBasedBuyer().select_action, seed)
        assert lines[0].startswith("An item worth Rs")
        assert lines[1].startswith('Seller: "Rs')
        assert any(line.startswith("Buyer:") for line in lines)
        assert lines[-1].startswith("(The seller was")
        assert "Rs" in lines[-1]


def test_selling_with_a_final_price_claim():
    # You value the item at Rs 1000, keep Rs 700 secret, and open at Rs 1500.
    # The bot offers 40% (Rs 600); you say "Rs 900, final"; the bot accepts.
    policy, seen = scripted([OPEN_40, ACCEPT])
    said = []
    info = sell(policy, read=scripted_input(["1000", "700", "1500", "f 900"]), say=said.append)

    assert seen[1][OBS_FINAL_PRICE_FLAG] == 1.0
    assert info["outcome"] == "deal"
    assert info["deal_price"] == pytest.approx(0.9)
    assert said[-1].startswith("Sold for Rs 900")


def test_calling_the_bot_back():
    policy, seen = scripted([OPEN_40, WALK_AWAY, ACCEPT])
    info = sell(policy, read=scripted_input(["1000", "700", "1500", "1200", "1000"]), say=lambda text: None)

    assert seen[2][OBS_CALLED_BACK_FLAG] == 1.0
    assert info["outcome"] == "deal"
    assert info["deal_price"] == pytest.approx(1.0)


def test_walking_off_ends_without_a_deal():
    policy, _ = scripted([OPEN_40])
    said = []
    info = sell(policy, read=scripted_input(["1000", "700", "1500", "q"]), say=said.append)
    assert info["outcome"] == "seller_left"
    assert said[-1] == "No deal."


def test_bad_replies_are_asked_again():
    policy, _ = scripted([OPEN_40, CONCEDE_ACTIONS[0], ACCEPT])
    said = []
    replies = ["abc", "1000", "700", "1500",  # "abc" is not a price
               "500",                          # below the bot's offer of Rs 600
               "1300", "a"]
    info = sell(policy, read=scripted_input(replies), say=said.append)
    assert "Please type a number." in said
    assert any("needs to be above Rs 600" in line for line in said)
    assert info["outcome"] == "deal"
    assert info["deal_price"] == pytest.approx(0.6)  # you accepted the bot's repeated offer
