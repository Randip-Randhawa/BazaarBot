import gymnasium as gym
import numpy as np

ITEM_VALUE_RANGE = (100.0, 2000.0)  # V, in rupees
RESERVATION_RANGE = (0.6, 0.9)  # seller's minimum price R, as a fraction of V
OPENING_ASK_RANGE = (1.5, 2.5)  # seller's first asking price, as a multiple of R
PATIENCE_RANGE = (5, 10)  # rounds before the seller loses interest
MAX_ROUNDS = 10

# Each seller has a hidden personality that decides how they concede. The
# concession rate is the share of the ask-offer gap they give up per counter.
PERSONALITIES = ("stubborn", "desperate")
STUBBORN_RATE_RANGE = (0.05, 0.15)  # small, and never changes
DESPERATE_START_RATE_RANGE = (0.20, 0.35)
DESPERATE_RATE_GROWTH = 0.10  # added after every counter
MAX_CONCESSION_RATE = 0.9

# Highest price / V; assumes the ask never rises
MAX_PRICE = RESERVATION_RANGE[1] * OPENING_ASK_RANGE[1]

STEP_COST = 0.01
NO_DEAL_PENALTY = 0.3

# Buyer's view, prices / V; seller's R and personality stay hidden
OBS_ASK = 0  # current ask
OBS_OPENING_ASK = 1  # opening ask
OBS_LAST_OFFER = 2  # buyer's last offer, 0 if none
OBS_ROUND = 3  # rounds played / MAX_ROUNDS
OBS_LAST_CONCESSION = 4  # seller's last price drop
OBS_AVG_CONCESSION = 5  # seller's mean price drop
OBS_FINAL_PRICE_FLAG = 6  # reserved, always 0
OBS_CALLED_BACK_FLAG = 7  # reserved, always 0
OBS_SIZE = 8

OBS_FEATURES = (
    "current_ask",
    "opening_ask",
    "last_offer",
    "round_progress",
    "last_concession",
    "avg_concession",
    "final_price_flag",
    "called_back_flag",
)

OPEN_FRACTIONS = (0.30, 0.40, 0.50, 0.60)  # of the current ask
CONCEDE_FRACTIONS = (0.00, 0.10, 0.25, 0.50)  # of the ask-offer gap
OPEN_ACTIONS = (0, 1, 2, 3)  # first offer only
CONCEDE_ACTIONS = (4, 5, 6, 7)
ACCEPT = 8
WALK_AWAY = 9
N_ACTIONS = 10

ACTION_MEANINGS = (
    "open at 30% of the ask",
    "open at 40% of the ask",
    "open at 50% of the ask",
    "open at 60% of the ask",
    "concede 0% of the gap (repeat the last offer)",
    "concede 10% of the gap",
    "concede 25% of the gap",
    "concede 50% of the gap",
    "accept the current ask",
    "walk away",
)


class BazaarEnv(gym.Env):
    """A buyer haggling with a seller who has a hidden minimum price."""

    def __init__(self):
        high = np.full(OBS_SIZE, MAX_PRICE, dtype=np.float32)
        high[[OBS_ROUND, OBS_FINAL_PRICE_FLAG, OBS_CALLED_BACK_FLAG]] = 1.0
        self.observation_space = gym.spaces.Box(np.zeros(OBS_SIZE, dtype=np.float32), high, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self._done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        self._item_value = rng.uniform(*ITEM_VALUE_RANGE)
        self._reservation_price = self._item_value * rng.uniform(*RESERVATION_RANGE)
        self._opening_ask = self._reservation_price * rng.uniform(*OPENING_ASK_RANGE)
        self._personality = str(rng.choice(PERSONALITIES))
        if self._personality == "stubborn":
            self._concession_rate = rng.uniform(*STUBBORN_RATE_RANGE)
        else:
            self._concession_rate = rng.uniform(*DESPERATE_START_RATE_RANGE)
        self._patience = int(rng.integers(PATIENCE_RANGE[0], PATIENCE_RANGE[1] + 1))

        self._ask = self._opening_ask
        self._last_offer = None
        self._round = 0
        self._last_concession = 0.0
        self._n_counters = 0
        self._done = False
        return self._observation(), self._info("ongoing")

    def step(self, action):
        if self._done:
            raise RuntimeError("The negotiation is over. Call reset() to start a new one.")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        action = int(action)

        self._round += 1
        reward = -STEP_COST

        if action == ACCEPT:
            return self._finish(reward, "deal", price=self._ask)
        if action == WALK_AWAY:
            return self._finish(reward, "buyer_walked")

        # Opening twice, or conceding before opening, wastes the turn.
        if action in OPEN_ACTIONS:
            if self._last_offer is not None:
                return self._next_round(reward, invalid_action=True)
            offer = OPEN_FRACTIONS[action] * self._ask
        else:
            if self._last_offer is None:
                return self._next_round(reward, invalid_action=True)
            fraction = CONCEDE_FRACTIONS[action - CONCEDE_ACTIONS[0]]
            offer = self._last_offer + fraction * (self._ask - self._last_offer)
        self._last_offer = offer

        if offer >= self._reservation_price:
            return self._finish(reward, "deal", price=offer)

        # The seller gives up part of the gap, but never goes below their minimum.
        new_ask = max(self._reservation_price, self._ask - self._concession_rate * (self._ask - offer))
        self._last_concession = self._ask - new_ask
        self._ask = new_ask
        self._n_counters += 1

        # A desperate seller gives up more with every round that passes.
        if self._personality == "desperate":
            self._concession_rate = min(self._concession_rate + DESPERATE_RATE_GROWTH, MAX_CONCESSION_RATE)
        return self._next_round(reward)

    def _next_round(self, reward, invalid_action=False):
        if self._round >= self._patience:
            return self._finish(reward, "seller_left", invalid_action=invalid_action)
        if self._round >= MAX_ROUNDS:
            return self._finish(reward, "max_rounds", invalid_action=invalid_action)
        return self._observation(), reward, False, False, self._info("ongoing", invalid_action)

    def _finish(self, reward, outcome, price=None, invalid_action=False):
        self._done = True
        info = self._info(outcome, invalid_action)
        # Revealed only once the negotiation is over, so evaluation can break
        # results down by personality. The agent learns from observations only.
        info["personality"] = self._personality
        if price is None:
            reward -= NO_DEAL_PENALTY
        else:
            reward += 1.0 - price / self._item_value
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
        if self._n_counters > 0:
            obs[OBS_AVG_CONCESSION] = (self._opening_ask - self._ask) / self._n_counters / v
        return obs

    def _info(self, outcome, invalid_action=False):
        return {"round": self._round, "outcome": outcome, "invalid_action": invalid_action}
