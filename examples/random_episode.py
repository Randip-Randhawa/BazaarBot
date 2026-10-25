"""Show BazaarEnv's observation and action spaces.

BazaarEnv is currently a skeleton: reset() and step() are not implemented,
so an episode cannot be played yet.

Run from the repository root:
    python -m examples.random_episode
"""

from bazaar import BazaarEnv


def main() -> None:
    env = BazaarEnv()
    print("Observation space:", env.observation_space)
    print("Action space:     ", env.action_space)


if __name__ == "__main__":
    main()
