"""Measuring a fixed buyer policy, with no learning or exploration.

A policy here is just a function from an observation to an action, so the DQN
and the rule-based baseline go through exactly the same code and metrics.

Metric definitions (all prices are fractions of the item value V):
    deal rate        deals / episodes
    avg reward       mean total environment reward per episode
    avg deal price   mean price over deals only; episodes without a deal have
                     no price and are left out
    avg surplus      mean of (1 - deal price) over all episodes, counting an
                     episode without a deal as 0
    avg length       mean number of env.step() calls per episode

Reward and surplus are not independent. The environment's reward is the
surplus minus the costs of haggling:
    reward = surplus - STEP_COST * length - NO_DEAL_PENALTY * (no deal)
"""

from dataclasses import dataclass

import numpy as np

from bazaar.env import BazaarEnv


@dataclass
class EpisodeResult:
    reward: float  # summed over every step, including the last one
    length: int  # env.step() calls
    deal_price: float | None  # None when the episode ended without a deal
    personality: str | None = None  # revealed by the env when the episode ends

    @property
    def surplus(self):
        return 0.0 if self.deal_price is None else 1.0 - self.deal_price


def run_episode(env, policy, seed):
    """Play one negotiation from a fresh reset and stop as soon as it ends,
    whether by termination or truncation."""
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    length = 0

    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        total_reward += reward
        length += 1
        if terminated or truncated:
            return EpisodeResult(total_reward, length, info.get("deal_price"), info.get("personality"))


def evaluate(policy, n_episodes, first_seed):
    """Run `n_episodes`, seeding episode i with `first_seed + i`.

    The seller is drawn entirely in reset(), so two policies evaluated with the
    same `first_seed` meet exactly the same sellers.
    """
    env = BazaarEnv()
    return [run_episode(env, policy, first_seed + i) for i in range(n_episodes)]


def evaluate_dqn(agent, n_episodes, first_seed):
    """Evaluate the agent's greedy policy without changing the agent.

    No epsilon, no replay buffer and no optimizer are involved; greedy_action()
    runs under torch.no_grad(). The network is put in eval mode for the run and
    then returned to whatever mode it was in.
    """
    was_training = agent.online_net.training
    agent.online_net.eval()
    try:
        return evaluate(agent.greedy_action, n_episodes, first_seed)
    finally:
        agent.online_net.train(was_training)


def summarize(results):
    deal_prices = [r.deal_price for r in results if r.deal_price is not None]
    return {
        "episodes": len(results),
        "deals": len(deal_prices),
        "deal_rate": len(deal_prices) / len(results),
        "avg_reward": float(np.mean([r.reward for r in results])),
        "avg_deal_price": float(np.mean(deal_prices)) if deal_prices else float("nan"),
        "avg_surplus": float(np.mean([r.surplus for r in results])),
        "avg_length": float(np.mean([r.length for r in results])),
    }


def summarize_by_personality(results):
    """One summary per seller personality, in the order they first appear."""
    groups = {}
    for r in results:
        groups.setdefault(r.personality, []).append(r)
    return {personality: summarize(group) for personality, group in groups.items()}


def mean_and_std(values):
    """Mean and sample standard deviation (ddof=1) across independent runs."""
    values = np.asarray(values, dtype=float)
    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    return float(values.mean()), std
