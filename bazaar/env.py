import gymnasium as gym
import numpy as np

ITEM_VALUE_RANGE = (100.0, 2000.0)  # V, in rupees
RESERVATION_RANGE = (0.6, 0.9)  # seller's minimum price R, as a fraction of V
OPENING_ASK_RANGE = (1.5, 2.5)  # seller's first asking price, as a multiple of R
PATIENCE_RANGE = (5, 10)  # rounds before the seller loses interest
MAX_ROUNDS = 10

# Each seller has a hidden personality that decides how they concede. The
# concession rate is the share of the ask-offer gap they give up per counter.
PERSONALITIES = ("stubborn", "desperate", "bluffer", "tourist_spotter")
STUBBORN_RATE_RANGE = (0.05, 0.15)  # small, and never changes
DESPERATE_START_RATE_RANGE = (0.20, 0.35)
DESPERATE_RATE_GROWTH = 0.10  # added after every counter
NORMAL_RATE_RANGE = (0.10, 0.30)  # bluffers and tourist-spotters
START_RATE_RANGES = {
    "stubborn": STUBBORN_RATE_RANGE,
    "desperate": DESPERATE_START_RATE_RANGE,
    "bluffer": NORMAL_RATE_RANGE,
    "tourist_spotter": NORMAL_RATE_RANGE,
}
MAX_CONCESSION_RATE = 0.9
RATE_NOISE = (0.8, 1.2)  # every counter's rate is scaled by a random factor in this range

BLUFF_CHANCE = 0.3  # per counter, while a bluffer is still above their minimum
EAGER_OPENING = 0.5  # a first offer at this share of the ask or more looks like a tourist
TOURIST_MARKUP = 0.2  # how much a tourist-spotter raises the ask after an eager opening
INSULT_FRACTION = 0.5  # offers below this share of R cost a round of the seller's patience
CALLBACK_CHANCE = 0.8  # multiplied by (last offer / R) when the buyer walks away

# Prices in the observation are divided by V. The highest one possible is a
# tourist-spotter's raise on the highest opening ask, and that raise is also
# the most negative "concession" a seller can make.
MAX_OPENING_ASK = RESERVATION_RANGE[1] * OPENING_ASK_RANGE[1]
MAX_PRICE = MAX_OPENING_ASK * (1 + TOURIST_MARKUP)
MIN_CONCESSION = -MAX_OPENING_ASK * TOURIST_MARKUP

STEP_COST = 0.01
NO_DEAL_PENALTY = 0.3

# Buyer's view, prices / V; seller's R and personality stay hidden
OBS_ASK = 0  # current ask
OBS_OPENING_ASK = 1  # opening ask
OBS_LAST_OFFER = 2  # buyer's last offer, 0 if none
OBS_ROUND = 3  # rounds played / MAX_ROUNDS
OBS_LAST_CONCESSION = 4  # seller's last price drop (negative if they raised it)
OBS_AVG_CONCESSION = 5  # seller's mean price drop
OBS_FINAL_PRICE_FLAG = 6  # 1 if the seller's latest reply claimed "final price"
OBS_CALLED_BACK_FLAG = 7  # 1 once the seller has called the buyer back
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


def action_mask(observations):
    """True for the actions that make sense, for one observation or a batch.

    Opening is only possible before the buyer's first offer, and conceding only
    after it. Accepting and walking away are always possible.
    """
    observations = np.asarray(observations)
    has_offered = observations[..., OBS_LAST_OFFER] > 0
    mask = np.ones(observations.shape[:-1] + (N_ACTIONS,), dtype=bool)
    mask[..., list(OPEN_ACTIONS)] = ~has_offered[..., None]
    mask[..., list(CONCEDE_ACTIONS)] = has_offered[..., None]
    return mask


class BazaarEnv(gym.Env):
    """A buyer haggling with a seller who has a hidden minimum price and personality."""

    def __init__(self):
        low = np.zeros(OBS_SIZE, dtype=np.float32)
        low[[OBS_LAST_CONCESSION, OBS_AVG_CONCESSION]] = MIN_CONCESSION
        high = np.full(OBS_SIZE, MAX_PRICE, dtype=np.float32)
        high[[OBS_ROUND, OBS_FINAL_PRICE_FLAG, OBS_CALLED_BACK_FLAG]] = 1.0
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self._done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        item_value = rng.uniform(*ITEM_VALUE_RANGE)
        self._reservation_price = item_value * rng.uniform(*RESERVATION_RANGE)
        opening_ask = self._reservation_price * rng.uniform(*OPENING_ASK_RANGE)
        self._personality = str(rng.choice(PERSONALITIES))
        self._concession_rate = rng.uniform(*START_RATE_RANGES[self._personality])
        self._patience = int(rng.integers(PATIENCE_RANGE[0], PATIENCE_RANGE[1] + 1))
        return self._start_negotiation(item_value, opening_ask)

    def _start_negotiation(self, item_value, opening_ask):
        self._item_value = item_value
        self._opening_ask = opening_ask
        self._ask = opening_ask
        self._last_offer = None
        self._round = 0
        self._last_concession = 0.0
        self._n_counters = 0
        self._claims_final_price = False
        self._called_back = False
        self._done = False

        info = self._info("ongoing")
        info["item_value"] = item_value  # the buyer knows what the item is worth to them
        return self._observation(), info

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
            if self._seller_calls_back():
                return self._next_round(reward)
            return self._finish(reward, "buyer_walked")

        # Opening twice, or conceding before opening, wastes the turn.
        is_opening = self._last_offer is None
        if action in OPEN_ACTIONS:
            if not is_opening:
                return self._next_round(reward, invalid_action=True)
            offer = OPEN_FRACTIONS[action] * self._ask
        else:
            if is_opening:
                return self._next_round(reward, invalid_action=True)
            fraction = CONCEDE_FRACTIONS[action - CONCEDE_ACTIONS[0]]
            offer = self._last_offer + fraction * (self._ask - self._last_offer)
        self._last_offer = offer

        if self._seller_accepts(offer):
            return self._finish(reward, "deal", price=offer)
        self._seller_counters(offer, is_opening)
        return self._next_round(reward)

    # The seller's side of the negotiation. The demo replaces these three
    # methods so that a person can play the seller instead.

    def _seller_accepts(self, offer):
        return offer >= self._reservation_price

    def _seller_counters(self, offer, is_opening):
        if offer < INSULT_FRACTION * self._reservation_price:
            self._patience -= 1  # an insulting offer wears the seller's patience down

        if self._personality == "tourist_spotter" and is_opening and offer >= EAGER_OPENING * self._ask:
            # An eager first offer gives the buyer away as a tourist.
            self._set_ask(self._ask * (1 + TOURIST_MARKUP))
            return

        bluffing = self._personality == "bluffer" and self._ask > self._reservation_price
        if bluffing and self.np_random.random() < BLUFF_CHANCE:
            self._set_ask(self._ask, claims_final_price=True)  # it isn't really final
            return

        rate = min(self._concession_rate * self.np_random.uniform(*RATE_NOISE), MAX_CONCESSION_RATE)
        new_ask = max(self._reservation_price, self._ask - rate * (self._ask - offer))
        # Every seller is honest about having reached their minimum.
        self._set_ask(new_ask, claims_final_price=new_ask <= self._reservation_price)

        # A desperate seller gives up more with every round that passes.
        if self._personality == "desperate":
            self._concession_rate = min(self._concession_rate + DESPERATE_RATE_GROWTH, MAX_CONCESSION_RATE)

    def _seller_calls_back(self):
        """When the buyer walks away, the seller may call them back once with a
        better price. The closer the buyer's last offer was to R, the likelier."""
        out_of_time = self._round >= min(self._patience, MAX_ROUNDS)
        if self._called_back or self._last_offer is None or out_of_time:
            return False
        if self.np_random.random() >= CALLBACK_CHANCE * self._last_offer / self._reservation_price:
            return False

        # Meet the buyer halfway, but never go below the minimum.
        new_ask = max(self._reservation_price, (self._ask + self._last_offer) / 2)
        self._set_ask(new_ask, claims_final_price=new_ask <= self._reservation_price)
        self._called_back = True
        return True

    def _set_ask(self, new_ask, claims_final_price=False):
        self._last_concession = self._ask - new_ask
        self._ask = new_ask
        self._n_counters += 1
        self._claims_final_price = claims_final_price

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
        # results down by seller. The agent learns from observations only.
        info["personality"] = self._personality
        info["reservation_price"] = self._reservation_price / self._item_value
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
        obs[OBS_FINAL_PRICE_FLAG] = float(self._claims_final_price)
        obs[OBS_CALLED_BACK_FLAG] = float(self._called_back)
        return obs

    def _info(self, outcome, invalid_action=False):
        return {"round": self._round, "outcome": outcome, "invalid_action": invalid_action}
