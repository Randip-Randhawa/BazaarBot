"""Turn negotiations into readable dialogue, for the demo and for inspecting agents.

Prices are shown in rupees by multiplying the observation's prices (which are
fractions of V) back up by the item value.
"""

from bazaar.env import (
    ACCEPT,
    OBS_ASK,
    OBS_FINAL_PRICE_FLAG,
    OBS_LAST_CONCESSION,
    OBS_LAST_OFFER,
    WALK_AWAY,
)

NO_DEAL_ENDINGS = {
    "buyer_walked": "The buyer leaves without the item.",
    "seller_left": 'Seller: "I have other customers." The seller turns away.',
    "max_rounds": "The market closes before they agree.",
}


def rupees(fraction_of_value, item_value):
    return f"Rs {fraction_of_value * item_value:,.0f}"


def buyer_line(action, ask_before, obs, item_value):
    if action == ACCEPT:
        return f'Buyer:  "Fine, {rupees(ask_before, item_value)}."'
    if action == WALK_AWAY:
        return 'Buyer:  "Too expensive." The buyer walks away.'
    return f'Buyer:  "{rupees(obs[OBS_LAST_OFFER], item_value)}?"'


def seller_line(obs, item_value):
    """What the seller says, judging from the observation after their reply."""
    ask = rupees(obs[OBS_ASK], item_value)
    if obs[OBS_LAST_CONCESSION] < 0:
        return f'Seller: "Ah, a tourist! For you, {ask}."'
    if obs[OBS_FINAL_PRICE_FLAG]:
        return f'Seller: "{ask}. Final price, take it or leave it!"'
    return f'Seller: "{ask}, my best price."'


def ending_line(info, item_value):
    if info["outcome"] == "deal":
        return f"Sold for {rupees(info['deal_price'], item_value)}."
    return NO_DEAL_ENDINGS[info["outcome"]]


def reveal_line(info, item_value):
    personality = info["personality"].replace("_", "-")
    minimum = rupees(info["reservation_price"], item_value)
    return f"(The seller was {personality}, with a secret minimum of {minimum}.)"


def negotiation_dialogue(env, policy, seed=None):
    """Play one negotiation and return it as dialogue lines, plus the final info."""
    obs, info = env.reset(seed=seed)
    value = info["item_value"]
    lines = [
        f"An item worth {rupees(1.0, value)} to the buyer.",
        f'Seller: "{rupees(obs[OBS_ASK], value)}, best price for you!"',
    ]

    done = False
    while not done:
        action = policy(obs)
        ask_before = obs[OBS_ASK]
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if info["invalid_action"]:
            lines.append("Buyer:  (hesitates)")
        else:
            lines.append(buyer_line(action, ask_before, obs, value))

        if action == WALK_AWAY and not done:
            lines.append(f'Seller: "Wait, come back! {rupees(obs[OBS_ASK], value)}, just for you."')
        elif not done and not info["invalid_action"]:
            lines.append(seller_line(obs, value))

    lines.append(ending_line(info, value))
    lines.append(reveal_line(info, value))
    return lines, info
