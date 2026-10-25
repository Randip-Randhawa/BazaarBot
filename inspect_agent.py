"""Look inside a trained buyer: how it behaves against each seller personality,
next to the tuned rule-based baseline, plus an example negotiation per personality.
"""

import argparse
from collections import Counter
from pathlib import Path

from bazaar.baseline import TUNED_CONCESSION_STEP, TUNED_OPENING_TARGET, RuleBasedBuyer
from bazaar.dialogue import negotiation_dialogue
from bazaar.dqn_agent import DQNAgent
from bazaar.env import (
    ACCEPT,
    CONCEDE_ACTIONS,
    OBS_ASK,
    OBS_CALLED_BACK_FLAG,
    OBS_FINAL_PRICE_FLAG,
    OPEN_ACTIONS,
    OPEN_FRACTIONS,
    PERSONALITIES,
    WALK_AWAY,
    BazaarEnv,
)

DEFAULT_MODEL = "models/dqn_seed10.pt"  # the run with the best validation score
HOLD = CONCEDE_ACTIONS[0]  # concede 0%, i.e. repeat the last offer


def play(env, policy, seed):
    """One negotiation, returning every (observation, action) pair, the final
    observation and the final info."""
    obs, _ = env.reset(seed=seed)
    steps = []
    done = False
    while not done:
        action = policy(obs)
        steps.append((obs, action))
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return steps, obs, info


def count_episode(counts, steps, final_obs, info):
    counts["episodes"] += 1
    counts["opening " + str(steps[0][1])] += 1
    counts["walked"] += any(action == WALK_AWAY for _, action in steps)
    counts["called back"] += bool(final_obs[OBS_CALLED_BACK_FLAG])
    if info["outcome"] == "deal":
        counts["deals"] += 1
        counts["deal rounds"] += info["round"]
        counts["buyer closed"] += steps[-1][1] == ACCEPT

    for obs, action in steps:
        if action in CONCEDE_ACTIONS:
            counts["concessions"] += 1
            counts["holds"] += action == HOLD
        if obs[OBS_FINAL_PRICE_FLAG]:
            # Whether the claim was true is only known now that R is revealed.
            honest = obs[OBS_ASK] <= info["reservation_price"] * (1 + 1e-5)
            kind = "honest" if honest else "bluff"
            counts[kind + " claims"] += 1
            counts[kind + " accepted"] += action == ACCEPT


def behaviour(policy, n_episodes, first_seed):
    env = BazaarEnv()
    by_personality = {personality: Counter() for personality in PERSONALITIES}
    for i in range(n_episodes):
        steps, final_obs, info = play(env, policy, first_seed + i)
        count_episode(by_personality[info["personality"]], steps, final_obs, info)
    return by_personality


def share(part, whole):
    return f"{part / whole:.0%}" if whole else "-"


def print_behaviour(label, by_personality):
    print(f"\n{label}")
    print(f"  {'seller':<16} {'deals':>6} {'round':>6} {'holds':>6} {'buyer closes':>13} "
          f"{'walks':>6} {'called back':>12} {'takes honest final':>19} {'takes bluff':>12}")
    for personality, c in by_personality.items():
        avg_round = f"{c['deal rounds'] / c['deals']:.1f}" if c["deals"] else "-"
        print(
            f"  {personality:<16} {share(c['deals'], c['episodes']):>6} {avg_round:>6} "
            f"{share(c['holds'], c['concessions']):>6} {share(c['buyer closed'], c['deals']):>13} "
            f"{share(c['walked'], c['episodes']):>6} {share(c['called back'], c['walked']):>12} "
            f"{share(c['honest accepted'], c['honest claims']):>19} {share(c['bluff accepted'], c['bluff claims']):>12}"
        )

    total = sum((c for c in by_personality.values()), Counter())
    openings = ", ".join(
        f"{fraction:.0%}: {share(total['opening ' + str(action)], total['episodes'])}"
        for action, fraction in zip(OPEN_ACTIONS, OPEN_FRACTIONS)
    )
    print(f"  opening offer (share of the ask): {openings}")


def print_examples(policy, first_seed):
    """One example negotiation against each personality."""
    env = BazaarEnv()
    shown = set()
    seed = first_seed
    while len(shown) < len(PERSONALITIES):
        lines, info = negotiation_dialogue(env, policy, seed)
        if info["personality"] not in shown:
            shown.add(info["personality"])
            print(f"\n--- test seller {seed} ---")
            print("\n".join(lines))
        seed += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-seed", type=int, default=100_000)
    args = parser.parse_args()
    if not Path(args.model).exists():
        parser.error(f"no model at {args.model}; train one with train.py or evaluate.py first")

    agent = DQNAgent.load(args.model)
    baseline = RuleBasedBuyer(TUNED_OPENING_TARGET, TUNED_CONCESSION_STEP)
    print(f"{args.episodes} test sellers from seed {args.eval_seed}")
    print_behaviour(f"DQN ({args.model})", behaviour(agent.greedy_action, args.episodes, args.eval_seed))
    print_behaviour("rule-based (tuned)", behaviour(baseline.select_action, args.episodes, args.eval_seed))
    print_examples(agent.greedy_action, args.eval_seed)


if __name__ == "__main__":
    main()
