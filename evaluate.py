"""Compare greedy DQN buyers against the baseline buyers.

Each DQN is trained from scratch with its own training seed, saved to models/,
then evaluated with exploration off. Every policy is evaluated on the same test
sellers, so they all face exactly the same negotiations. Tables are printed and
also written as CSV files to results/, for the report.
"""

import argparse
import time
from pathlib import Path

from bazaar.baseline import (
    TUNED_CONCESSION_STEP,
    TUNED_OPENING_TARGET,
    RandomBuyer,
    RuleBasedBuyer,
    SplitTheDifferenceBuyer,
    SteadyConcessionBuyer,
    TouristBuyer,
)
from bazaar.env import PERSONALITIES
from bazaar.evaluation import evaluate, evaluate_dqn, mean_and_std, save_csv, summarize, summarize_by_personality
from train import train

# train() uses seed, seed + 1 and seed + 2 internally, so training seeds are
# spaced out to keep different models' random streams apart.
SEED_SPACING = 10

# The DQN is compared in detail against the strongest baseline.
REFERENCE_BASELINE = "rule-based (tuned)"

METRICS = ("deal_rate", "avg_reward", "avg_deal_price", "avg_surplus", "surplus_captured", "avg_length")
HEADER = (
    f"{'policy':<24} {'deal rate':>9} {'reward':>8} {'deal price':>10} {'surplus':>8} "
    f"{'captured':>8} {'length':>7}   deals"
)


def make_baselines():
    return {
        "random": RandomBuyer(seed=0).select_action,
        "tourist": TouristBuyer().select_action,
        "split the difference": SplitTheDifferenceBuyer().select_action,
        "steady concession": SteadyConcessionBuyer().select_action,
        "rule-based (hand-set)": RuleBasedBuyer().select_action,
        REFERENCE_BASELINE: RuleBasedBuyer(TUNED_OPENING_TARGET, TUNED_CONCESSION_STEP).select_action,
    }


def format_row(label, summary):
    return (
        f"{label:<24} {summary['deal_rate']:>9.1%} {summary['avg_reward']:>+8.3f} "
        f"{summary['avg_deal_price']:>10.3f} {summary['avg_surplus']:>8.3f} "
        f"{summary['surplus_captured']:>8.1%} {summary['avg_length']:>7.2f}"
        f"   {summary['deals']}/{summary['episodes']}"
    )


def format_aggregate(label, summaries):
    cells = []
    for metric in METRICS:
        mean, std = mean_and_std([s[metric] for s in summaries])
        cells.append(f"{mean:.3f}+/-{std:.3f}")
    return f"{label:<24} " + "  ".join(f"{name}={cell}" for name, cell in zip(METRICS, cells))


def format_personality_row(label, results):
    """Reward (and deal rate) against each personality."""
    by_personality = summarize_by_personality(results)
    cells = []
    for personality in PERSONALITIES:
        s = by_personality.get(personality)
        cell = "n/a" if s is None else f"{s['avg_reward']:+.3f} ({s['deal_rate']:.0%})"
        cells.append(f"{cell:>16}")
    return f"{label:<24} " + " ".join(cells)


def average_reward(results, personality=None):
    """Average reward over all episodes, or only those against one personality."""
    if personality is None:
        return summarize(results)["avg_reward"]
    return summarize_by_personality(results)[personality]["avg_reward"]


def print_gaps(dqn_results, reference_results):
    """How far ahead of the reference baseline each DQN is, overall and per personality."""
    print(f"DQN reward minus {REFERENCE_BASELINE} reward (mean +/- std across DQNs):")
    seen = summarize_by_personality(reference_results)  # all four, unless there are very few episodes
    for personality in [None, *[p for p in PERSONALITIES if p in seen]]:
        reference = average_reward(reference_results, personality)
        gaps = [average_reward(results, personality) - reference for results in dqn_results.values()]
        mean, std = mean_and_std(gaps)
        wins = sum(gap > 0 for gap in gaps)
        name = personality or "overall"
        print(f"  {name:<16} {mean:+.3f} +/- {std:.3f}  (DQN ahead for {wins} of {len(gaps)} seeds)")


def save_results(results_dir, all_results, dqn_seeds):
    summary_rows, personality_rows = [], []
    for label, results in all_results.items():
        seed = dqn_seeds.get(label, "")
        summary_rows.append({"policy": label, "train_seed": seed, **summarize(results)})
        for personality, summary in summarize_by_personality(results).items():
            personality_rows.append({"policy": label, "train_seed": seed, "personality": personality, **summary})
    save_csv(results_dir / "summary.csv", summary_rows)
    save_csv(results_dir / "by_personality.csv", personality_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("both", "baseline", "dqn"), default="both")
    parser.add_argument("--train-seeds", type=int, default=5, help="number of independently trained DQNs")
    parser.add_argument("--base-seed", type=int, default=0, help=f"training seeds are base, base+{SEED_SPACING}, ...")
    parser.add_argument("--train-episodes", type=int, default=10000, help="training episodes per DQN")
    parser.add_argument("--eval-episodes", type=int, default=1000, help="evaluation episodes per policy")
    parser.add_argument("--eval-seed", type=int, default=100_000, help="episode i is seeded with eval-seed + i")
    parser.add_argument("--models-dir", default="models", help="where the trained DQNs are saved")
    parser.add_argument("--results-dir", default="results", help="where the CSV results are written")
    args = parser.parse_args()

    models_dir, results_dir = Path(args.models_dir), Path(args.results_dir)
    train_seeds = [args.base_seed + SEED_SPACING * k for k in range(args.train_seeds)]
    eval_seeds = f"{args.eval_seed}..{args.eval_seed + args.eval_episodes - 1}"
    print(f"evaluation: {args.eval_episodes} episodes per policy, env seeds {eval_seeds}, no exploration")
    if args.policy != "baseline":
        print(f"training:   {len(train_seeds)} DQNs x {args.train_episodes} episodes, seeds {train_seeds}")
    print()
    print(HEADER)

    all_results = {}
    if args.policy != "dqn":
        for label, policy in make_baselines().items():
            all_results[label] = evaluate(policy, args.eval_episodes, args.eval_seed)
            print(format_row(label, summarize(all_results[label])))

    dqn_results, dqn_seeds = {}, {}
    if args.policy != "baseline":
        for seed in train_seeds:
            start = time.perf_counter()
            curve_path = results_dir / f"learning_curve_seed{seed}.csv"
            agent = train(args.train_episodes, seed, show_progress=False, curve_path=curve_path)
            agent.save(models_dir / f"dqn_seed{seed}.pt")

            label = f"DQN seed {seed}"
            dqn_results[label] = evaluate_dqn(agent, args.eval_episodes, args.eval_seed)
            dqn_seeds[label] = seed
            print(format_row(label, summarize(dqn_results[label])) + f"   ({time.perf_counter() - start:.0f}s)")
        all_results.update(dqn_results)

    if dqn_results:
        print()
        print("Mean +/- sample std across the independently trained DQNs (not across episodes):")
        print(format_aggregate("DQN", [summarize(results) for results in dqn_results.values()]))

    print()
    print("Reward (deal rate) by seller personality:")
    print(f"{'policy':<24} " + " ".join(f"{personality:>16}" for personality in PERSONALITIES))
    for label, results in all_results.items():
        print(format_personality_row(label, results))

    if dqn_results and REFERENCE_BASELINE in all_results:
        print()
        print_gaps(dqn_results, all_results[REFERENCE_BASELINE])

    save_results(results_dir, all_results, dqn_seeds)
    print()
    print(f"results written to {results_dir}/" + (f", models to {models_dir}/" if dqn_results else ""))

    print()
    print("By seller personality (deal rate, reward, deal price):")
    print(f"{'policy':<22} " + "    ".join(f"{personality:<21}" for personality in PERSONALITIES))
    if baseline_results is not None:
        print(format_personality_row("rule-based baseline", baseline_results))
    for seed, results in dqn_results.items():
        print(format_personality_row(f"DQN train seed {seed}", results))

    if dqn_results and baseline_results is not None:
        baseline_by = summarize_by_personality(baseline_results)
        for personality in PERSONALITIES:
            if personality not in baseline_by:  # possible with very few evaluation episodes
                continue
            gaps = [
                summarize_by_personality(results)[personality]["avg_reward"] - baseline_by[personality]["avg_reward"]
                for results in dqn_results.values()
            ]
            mean, std = mean_and_std(gaps)
            wins = sum(gap > 0 for gap in gaps)
            print(f"DQN - baseline reward vs {personality}: {mean:+.3f} +/- {std:.3f}  (DQN ahead for {wins} of {len(gaps)} seeds)")


if __name__ == "__main__":
    main()
