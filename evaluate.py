"""Compare greedy DQN buyers against the rule-based baseline.

Each DQN is trained from scratch with its own training seed (there are no saved
models), then evaluated with exploration off. Every policy is evaluated on the
same evaluation episodes, so they all face the same sellers.
"""

import argparse
import time

from bazaar.baseline import RuleBasedBuyer
from bazaar.evaluation import evaluate, evaluate_dqn, mean_and_std, summarize
from train import train

# train() uses seed, seed + 1 and seed + 2 internally, so training seeds are
# spaced out to keep different models' random streams apart.
SEED_SPACING = 10

METRICS = ("deal_rate", "avg_reward", "avg_deal_price", "avg_surplus", "avg_length")
HEADER = f"{'policy':<22} {'deal rate':>9} {'reward':>8} {'deal price':>10} {'surplus':>8} {'length':>7}   deals"


def format_row(label, summary):
    return (
        f"{label:<22} {summary['deal_rate']:>9.1%} {summary['avg_reward']:>+8.3f} "
        f"{summary['avg_deal_price']:>10.3f} {summary['avg_surplus']:>8.3f} {summary['avg_length']:>7.2f}"
        f"   {summary['deals']}/{summary['episodes']}"
    )


def format_aggregate(label, summaries):
    cells = []
    for metric in METRICS:
        mean, std = mean_and_std([s[metric] for s in summaries])
        cells.append(f"{mean:.3f}+/-{std:.3f}")
    return f"{label:<22} " + "  ".join(f"{name}={cell}" for name, cell in zip(METRICS, cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("both", "baseline", "dqn"), default="both")
    parser.add_argument("--train-seeds", type=int, default=5, help="number of independently trained DQNs")
    parser.add_argument("--base-seed", type=int, default=0, help=f"training seeds are base, base+{SEED_SPACING}, ...")
    parser.add_argument("--train-episodes", type=int, default=10000, help="training episodes per DQN")
    parser.add_argument("--eval-episodes", type=int, default=1000, help="evaluation episodes per policy")
    parser.add_argument("--eval-seed", type=int, default=10_000, help="episode i is seeded with eval-seed + i")
    args = parser.parse_args()

    train_seeds = [args.base_seed + SEED_SPACING * k for k in range(args.train_seeds)]
    eval_seeds = f"{args.eval_seed}..{args.eval_seed + args.eval_episodes - 1}"
    print(f"evaluation: {args.eval_episodes} episodes per policy, env seeds {eval_seeds}, no exploration")
    if args.policy != "baseline":
        print(f"training:   {len(train_seeds)} DQNs x {args.train_episodes} episodes, seeds {train_seeds}")
    print()
    print(HEADER)

    baseline = None
    if args.policy != "dqn":
        baseline = summarize(evaluate(RuleBasedBuyer().select_action, args.eval_episodes, args.eval_seed))
        print(format_row("rule-based baseline", baseline))

    if args.policy == "baseline":
        return

    dqn_summaries = []
    for seed in train_seeds:
        start = time.perf_counter()
        agent = train(args.train_episodes, seed, show_progress=False)
        summary = summarize(evaluate_dqn(agent, args.eval_episodes, args.eval_seed))
        dqn_summaries.append(summary)
        print(format_row(f"DQN train seed {seed}", summary) + f"   ({time.perf_counter() - start:.0f}s)")

    print()
    print("Mean +/- sample std across the independently trained DQNs (not across episodes):")
    print(format_aggregate("DQN", dqn_summaries))
    if baseline is not None:
        gaps = [s["avg_reward"] - baseline["avg_reward"] for s in dqn_summaries]
        mean, std = mean_and_std(gaps)
        wins = sum(gap > 0 for gap in gaps)
        print(f"DQN reward - baseline reward: {mean:+.3f} +/- {std:.3f}  (DQN ahead for {wins} of {len(gaps)} seeds)")


if __name__ == "__main__":
    main()
