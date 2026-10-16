import gymnasium as gym
import numpy as np

# Price ranges agreed in IDEAS.md 7.1.
MIN_ITEM_VALUE = 100.0  # V in rupees
MAX_ITEM_VALUE = 2000.0
MIN_RESERVATION_FRACTION = 0.6  # R is at least 0.6 V
MAX_RESERVATION_FRACTION = 0.9  # R is at most 0.9 V
MIN_OPENING_MULTIPLIER = 1.5  # opening ask is at least 1.5 R
MAX_OPENING_MULTIPLIER = 2.5  # opening ask is at most 2.5 R
# Highest possible opening ask relative to V (2.25), the upper bound of every
# price feature. PROVISIONAL: assumes the seller never raises the ask, so it
# stays below the opening ask and concessions are never negative. A
# tourist-spotter raise (undecided, IDEAS.md 6.5) would break both bounds.
MAX_PRICE = MAX_RESERVATION_FRACTION * MAX_OPENING_MULTIPLIER

# Rewards agreed in IDEAS.md 7.3.
STEP_COST = 0.01  # charged every round
NO_DEAL_PENALTY = 0.3  # charged once when the negotiation ends without a deal

# PROVISIONAL generic seller parameters; the real ranges are undecided
# (IDEAS.md 6.5).
MIN_CONCESSION_RATE = 0.1  # c: fraction of the ask-offer gap the seller closes per counter
MAX_CONCESSION_RATE = 0.4
MIN_PATIENCE = 5  # rounds before the seller leaves, inclusive
MAX_PATIENCE = 10
MAX_ROUNDS = 10  # hard cap on rounds per negotiation

# Observation layout (IDEAS.md 7.3). Every price is divided by V. The seller's
# reservation price R and personality are hidden and never included.
OBS_ASK = 0  # seller's current ask, [0, MAX_PRICE]. Required.
OBS_OPENING_ASK = 1  # seller's opening ask, [0, MAX_PRICE]. Required.
OBS_LAST_OFFER = 2  # buyer's last offer, [0, MAX_PRICE]; 0 = no offer yet. Required.
OBS_ROUND = 3  # rounds played / max rounds, [0, 1]. Required.
OBS_LAST_CONCESSION = 4  # drop in the ask at the seller's last counter, [0, MAX_PRICE]. Required.
OBS_AVG_CONCESSION = 5  # mean drop per seller counter so far, [0, MAX_PRICE]. Required.
OBS_FINAL_PRICE_FLAG = 6  # 1 if the seller claimed "final price", else 0. Reserved until bluffing exists.
OBS_CALLED_BACK_FLAG = 7  # 1 if the seller called the buyer back, else 0. Reserved until callbacks exist.
OBS_SIZE = 8

# Action layout (IDEAS.md 7.3). The fractions are confirmed. PROVISIONAL: the
# total of 10 actions (IDEAS.md suggests about 6-8), and how open actions are
# limited to the first offer (IDEAS.md 6.6).
OPEN_FRACTIONS = (0.30, 0.40, 0.50, 0.60)  # first offer = fraction x current ask
CONCEDE_FRACTIONS = (0.00, 0.10, 0.25, 0.50)  # offer moves this fraction of (ask - last offer)
OPEN_ACTIONS = (0, 1, 2, 3)  # open at 30 / 40 / 50 / 60% of the ask
CONCEDE_ACTIONS = (4, 5, 6, 7)  # concede 0 / 10 / 25 / 50% of the gap
ACCEPT = 8  # buy at the seller's current ask
WALK_AWAY = 9  # leave without a deal
N_ACTIONS = 10

ACTION_MEANINGS = (
    "open at 30% of the ask",
    "open at 40% of the ask",
    "open at 50% of the ask",
    "open at 60% of the ask",
    "concede 0% of the gap (hold the last offer)",
    "concede 10% of the gap",
    "concede 25% of the gap",
    "concede 50% of the gap",
    "accept the current ask",
    "walk away",
)


class BazaarEnv(gym.Env):
    """Single-item price negotiation against a seller with a hidden minimum price."""

    metadata = {"render_modes": []}

    def __init__(self):
        high = np.full(OBS_SIZE, MAX_PRICE, dtype=np.float32)
        high[OBS_ROUND] = 1.0
        high[OBS_FINAL_PRICE_FLAG] = 1.0
        high[OBS_CALLED_BACK_FLAG] = 1.0
    
        self.observation_space = gym.spaces.Box(
            low=np.zeros(OBS_SIZE, dtype=np.float32), high=high, dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

        self._done = True  # step() requires a reset() first

    def reset(self, *, seed=None, options=None):
        # `options` is accepted for the Gymnasium API but not used yet.
        super().reset(seed=seed)
        rng = self.np_random

        # The buyer knows V. R, c and patience are the seller's private values
        # and must never reach the observation or info.
        self._item_value = rng.uniform(MIN_ITEM_VALUE, MAX_ITEM_VALUE)
        self._reservation_price = self._item_value * rng.uniform(
            MIN_RESERVATION_FRACTION, MAX_RESERVATION_FRACTION
        )
        self._opening_ask = self._reservation_price * rng.uniform(
            MIN_OPENING_MULTIPLIER, MAX_OPENING_MULTIPLIER
        )
        self._concession_rate = rng.uniform(MIN_CONCESSION_RATE, MAX_CONCESSION_RATE)
        self._patience = int(rng.integers(MIN_PATIENCE, MAX_PATIENCE + 1))

        self._ask = self._opening_ask
        self._last_offer = None  # no offer yet
        self._round = 0
        self._last_concession = 0.0
        self._n_counters = 0
        self._done = False

        return self._observation(), self._info("ongoing")

    def step(self, action):
        if self._done:
            raise RuntimeError("Episode has ended (or never started); call reset() first.")
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action!r} is outside the action space {self.action_space}.")
        action = int(action)

        self._round += 1
        reward = -STEP_COST

        if action == ACCEPT:
            return self._end(reward, "deal", price=self._ask)
        if action == WALK_AWAY:
            return self._end(reward, "buyer_walked")

        has_offered = self._last_offer is not None
        if (action in OPEN_ACTIONS and has_offered) or (
            action in CONCEDE_ACTIONS and not has_offered
        ):
            # PROVISIONAL invalid-action policy (IDEAS.md 6.6): the turn is
            # wasted. The round still counts, no offer is made and the seller
            # does not respond.
            return self._after_seller_turn(reward, invalid_action=True)

        if action in OPEN_ACTIONS:
            offer = OPEN_FRACTIONS[action] * self._ask
        else:
            fraction = CONCEDE_FRACTIONS[action - CONCEDE_ACTIONS[0]]
            offer = self._last_offer + fraction * (self._ask - self._last_offer)
        self._last_offer = offer

        if offer >= self._reservation_price:
            return self._end(reward, "deal", price=offer)

        # Generic seller counter: close a fraction c of the gap, never below R.
        new_ask = max(
            self._reservation_price,
            self._ask - self._concession_rate * (self._ask - offer),
        )
        self._last_concession = self._ask - new_ask
        self._ask = new_ask
        self._n_counters += 1

        return self._after_seller_turn(reward)

    def _after_seller_turn(self, reward, invalid_action=False):
        # Both limits are rules of the task, so they end the episode with
        # terminated=True. truncated is left for external time limits.
        if self._round >= self._patience:
            return self._end(reward, "seller_left", invalid_action=invalid_action)
        if self._round >= MAX_ROUNDS:
            return self._end(reward, "max_rounds", invalid_action=invalid_action)
        info = self._info("ongoing", invalid_action=invalid_action)
        return self._observation(), float(reward), False, False, info

    def _end(self, reward, outcome, price=None, invalid_action=False):
        # Terminal reward, applied exactly once: _done blocks further steps.
        if outcome == "deal":
            reward += 1.0 - price / self._item_value
        else:
            reward -= NO_DEAL_PENALTY
        self._done = True
        info = self._info(outcome, invalid_action=invalid_action)
        if price is not None:
            info["deal_price"] = price / self._item_value
        return self._observation(), float(reward), True, False, info

    def _observation(self):
        v = self._item_value
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        obs[OBS_ASK] = self._ask / v
        obs[OBS_OPENING_ASK] = self._opening_ask / v
        obs[OBS_LAST_OFFER] = 0.0 if self._last_offer is None else self._last_offer / v
        obs[OBS_ROUND] = self._round / MAX_ROUNDS
        obs[OBS_LAST_CONCESSION] = self._last_concession / v
        if self._n_counters:
            obs[OBS_AVG_CONCESSION] = (self._opening_ask - self._ask) / self._n_counters / v
        # OBS_FINAL_PRICE_FLAG and OBS_CALLED_BACK_FLAG stay 0 until bluffing
        # and callbacks exist.
        return obs

    def _info(self, outcome, invalid_action=False):
        # Buyer-visible information only.
        return {"round": self._round, "outcome": outcome, "invalid_action": invalid_action}
