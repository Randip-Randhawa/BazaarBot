import gymnasium as gym
import numpy as np

MAX_RESERVATION_FRACTION = 0.9  
MAX_OPENING_MULTIPLIER = 2.5  
MAX_PRICE = MAX_RESERVATION_FRACTION * MAX_OPENING_MULTIPLIER


OBS_ASK = 0  
OBS_OPENING_ASK = 1  
OBS_LAST_OFFER = 2  
OBS_ROUND = 3  
OBS_LAST_CONCESSION = 4  
OBS_AVG_CONCESSION = 5  
OBS_FINAL_PRICE_FLAG = 6 
OBS_CALLED_BACK_FLAG = 7 
OBS_SIZE = 8

OPEN_FRACTIONS = (0.30, 0.40, 0.50, 0.60)  # first offer = fraction x current ask
CONCEDE_FRACTIONS = (0.00, 0.10, 0.25, 0.50)  # offer moves this fraction of (ask - last offer)
OPEN_ACTIONS = (0, 1, 2, 3)  # open at 30 / 40 / 50 / 60% of the ask
CONCEDE_ACTIONS = (4, 5, 6, 7)  # concede 0 / 10 / 25 / 50% of the gap
ACCEPT = 8  # buy at the seller's current ask
WALK_AWAY = 9  # leave without a deal
N_ACTIONS = 10


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

    def reset(self, *, seed=None, options=None):
        raise NotImplementedError("BazaarEnv.reset() is not implemented yet.")

    def step(self, action):
        raise NotImplementedError("BazaarEnv.step() is not implemented yet.")
