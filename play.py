"""Terminal demo for BazaarBot.

    python play.py watch    the trained bot haggles with a simulated seller
    python play.py sell     you play the seller and try to get a good price out of the bot
"""

import argparse
import time
from pathlib import Path

from bazaar.dialogue import negotiation_dialogue, rupees
from bazaar.dqn_agent import DQNAgent
from bazaar.env import ACCEPT, MAX_PRICE, MAX_ROUNDS, OBS_ASK, WALK_AWAY, BazaarEnv

DEFAULT_MODEL = "models/dqn_seed10.pt"  # the run with the best validation score

REPLY_HELP = "a = accept, a number = your new price, f <number> = call it your final price, q = walk off"


class HumanSellerEnv(BazaarEnv):
    """BazaarEnv with a person playing the seller by typing replies.

    Only the seller's side changes; the bot still sees the same kind of
    observations it was trained on.
    """

    def __init__(self, read=input, say=print):
        super().__init__()
        self.read = read
        self.say = say

    def reset(self, *, seed=None, options=None):
        super(BazaarEnv, self).reset(seed=seed)  # skip BazaarEnv's random seller
        self.say("You are the seller. The bot is the buyer.")
        value = self._read_price("What is the item worth to the buyer? The bot knows this. Rs ")
        self._reservation_price = self._read_price("Your secret lowest price? The bot never sees it. Rs ")
        opening = self._read_price("Your opening price? Rs ", highest=MAX_PRICE * value)
        self._personality = "human"
        self._patience = MAX_ROUNDS  # you decide when to give up, by typing q
        return self._start_negotiation(value, opening)

    def _seller_accepts(self, offer):
        self.say(f'Bot: "Rs {offer:,.0f}?"')  # inside the env, prices are in rupees
        while True:
            reply = self.read(f"Your reply ({REPLY_HELP}): ").strip().lower()
            if reply in ("a", "accept"):
                return True
            if reply in ("q", "quit"):
                self._reply = None
                return False
            claims_final = reply.startswith("f")
            price = self._parse_price(reply.removeprefix("f"), lowest=offer)
            if price is not None:
                self._reply = (price, claims_final)
                return False

    def _seller_counters(self, offer, is_opening):
        if self._reply is None:
            self._patience = self._round  # you walked off, so the negotiation ends now
            return
        price, claims_final = self._reply
        self._set_ask(price, claims_final_price=claims_final)

    def _seller_calls_back(self):
        if self._called_back or self._last_offer is None or self._round >= MAX_ROUNDS:
            return False
        while True:
            reply = self.read("The bot walks away. Call it back with a new price, or press Enter to let it go: Rs ")
            if not reply.strip():
                return False
            price = self._parse_price(reply, lowest=self._last_offer)
            if price is not None:
                self._set_ask(price)
                self._called_back = True
                return True

    def _read_price(self, prompt, highest=float("inf")):
        while True:
            price = self._parse_price(self.read(prompt), lowest=0.0, highest=highest)
            if price is not None:
                return price

    def _parse_price(self, text, lowest, highest=None):
        """A price in rupees from the text, or None (after saying why) if it won't do."""
        if highest is None:
            highest = MAX_PRICE * self._item_value
        try:
            price = float(text.replace(",", "").strip())
        except ValueError:
            self.say("Please type a number.")
            return None
        if not lowest < price <= highest:
            self.say(f"The price needs to be above Rs {lowest:,.0f} and at most Rs {highest:,.0f}.")
            return None
        return price


def watch(policy, seed=None, delay=0.0, say=print):
    lines, _ = negotiation_dialogue(BazaarEnv(), policy, seed)
    for line in lines:
        say(line)
        time.sleep(delay)


def sell(policy, read=input, say=print):
    """You play the seller against `policy`. Returns the final info."""
    env = HumanSellerEnv(read, say)
    obs, info = env.reset()
    value = info["item_value"]

    done = False
    while not done:
        action = policy(obs)
        if action == ACCEPT:
            say(f'Bot: "Fine, {rupees(obs[OBS_ASK], value)}. Deal."')
        elif action == WALK_AWAY:
            say('Bot: "Too expensive." The bot walks away.')
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    if info["outcome"] == "deal":
        price = info["deal_price"] * value
        minimum = info["reservation_price"] * value
        say(f"Sold for Rs {price:,.0f}: Rs {price - minimum:,.0f} above your secret minimum, "
            f"and {info['deal_price']:.0%} of what the item is worth to the bot.")
    else:
        say("No deal.")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("watch", "sell"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, help="watch mode: which simulated seller to face")
    parser.add_argument("--delay", type=float, default=0.8, help="watch mode: seconds between lines")
    args = parser.parse_args()
    if not Path(args.model).exists():
        parser.error(f"no model at {args.model}; train one with train.py or evaluate.py first")

    agent = DQNAgent.load(args.model)
    if args.mode == "watch":
        watch(agent.greedy_action, args.seed, args.delay)
    else:
        sell(agent.greedy_action)


if __name__ == "__main__":
    main()
