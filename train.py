import argparse
import copy
from collections import deque

import numpy as np

from bazaar.dqn_agent import DQNAgent
from bazaar.env import BazaarEnv
from bazaar.evaluation import evaluate_dqn, summarize
from bazaar.replay_buffer import ReplayBuffer

BUFFER_CAPACITY = 20_000
BATCH_SIZE = 64
LEARNING_STARTS = 500  # transitions to collect before the first gradient step
TARGET_UPDATE_EVERY = 250  # environment steps between target-network syncs

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 5_000

LOG_EVERY = 100  # episodes

# Training performance swings up and down, so the network is checked on a
# fixed set of validation sellers and the best version is kept. These seeds
# are kept well away from evaluate.py's test sellers (10_000 onwards).
VALIDATE_EVERY = 500  # episodes
VALIDATION_EPISODES = 500
VALIDATION_SEED = 20_000


def epsilon_at(step):
    """Fall linearly from EPSILON_START to EPSILON_END, then stay there."""
    progress = min(step / EPSILON_DECAY_STEPS, 1.0)
    return EPSILON_START + progress * (EPSILON_END - EPSILON_START)


def train(episodes, seed, show_progress=True):
    # Separate seeds, so the agent's exploration and the buffer's sampling
    # don't draw the same random numbers.
    env = BazaarEnv()
    agent = DQNAgent(epsilon=EPSILON_START, seed=seed)
    buffer = ReplayBuffer(BUFFER_CAPACITY, seed=seed + 1)

    total_steps = 0
    gradient_updates = 0
    target_syncs = 0
    recent_rewards = deque(maxlen=LOG_EVERY)
    recent_deals = deque(maxlen=LOG_EVERY)
    recent_losses = deque(maxlen=LOG_EVERY)
    best_reward = -float("inf")
    best_episode = None
    best_weights = None

    for episode in range(1, episodes + 1):
        # Only the first reset is seeded; later ones continue the same random
        # stream, so every episode gets a new seller but the run is repeatable.
        obs, _ = env.reset(seed=seed + 2 if episode == 1 else None)
        episode_reward = 0.0
        done = False

        while not done:
            agent.epsilon = epsilon_at(total_steps)
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Store `terminated`, not `done`: if an episode is only cut short
            # by truncation, its next state still has value worth learning.
            buffer.add(obs, action, reward, next_obs, terminated)
            obs = next_obs
            episode_reward += reward
            total_steps += 1
            done = terminated or truncated

            if len(buffer) >= LEARNING_STARTS:
                recent_losses.append(agent.update(*buffer.sample(BATCH_SIZE)))
                gradient_updates += 1
            if total_steps % TARGET_UPDATE_EVERY == 0:
                agent.update_target()
                target_syncs += 1

        recent_rewards.append(episode_reward)
        recent_deals.append(info["outcome"] == "deal")

        if show_progress and episode % LOG_EVERY == 0:
            # No loss to report until learning has started.
            loss_text = f"{np.mean(recent_losses):.4f}" if recent_losses else "  n/a "
            print(
                f"episode {episode:5d} | "
                f"avg reward {np.mean(recent_rewards):+.3f} | "
                f"deal rate {np.mean(recent_deals):4.0%} | "
                f"epsilon {agent.epsilon:.2f} | "
                f"steps {total_steps} | "
                f"updates {gradient_updates} | "
                f"loss {loss_text} | "
                f"target syncs {target_syncs}"
            )

        # Also check after the last episode, so even a short run is validated.
        if episode % VALIDATE_EVERY == 0 or episode == episodes:
            reward = summarize(evaluate_dqn(agent, VALIDATION_EPISODES, VALIDATION_SEED))["avg_reward"]
            if reward > best_reward:
                best_reward, best_episode = reward, episode
                best_weights = copy.deepcopy(agent.online_net.state_dict())
            if show_progress:
                print(f"validation reward {reward:+.3f} (best {best_reward:+.3f} at episode {best_episode})")

    agent.online_net.load_state_dict(best_weights)
    agent.update_target()
    if show_progress:
        print(f"kept the network from episode {best_episode}, validation reward {best_reward:+.3f}")
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the BazaarBot DQN buyer.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", help="where to save the model (default: models/dqn_seed<SEED>.pt)")
    args = parser.parse_args()

    agent = train(args.episodes, args.seed)
    path = args.save or f"models/dqn_seed{args.seed}.pt"
    agent.save(path)
    print(f"saved to {path}")
