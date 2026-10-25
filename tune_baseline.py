"""Find the best settings for the rule-based baseline.

The DQN keeps whichever checkpoint scores best on the validation sellers, so to
keep the comparison fair the baseline is tuned on exactly the same sellers.
The test sellers in evaluate.py are never used here.
"""

import argparse
import itertools

import numpy as np

from bazaar.baseline import RuleBasedBuyer
from bazaar.evaluation import evaluate, save_csv, summarize
from train import VALIDATION_EPISODES, VALIDATION_SEED

OPENING_TARGETS = np.round(np.arange(0.30, 0.901, 0.05), 2)
CONCESSION_STEPS = np.round(np.arange(0.05, 0.601, 0.05), 2)


def tune(opening_targets, concession_steps, episodes, first_seed):
    """Score every combination of settings; return the rows, best first."""
    rows = []
    for opening_target, concession_step in itertools.product(opening_targets, concession_steps):
        buyer = RuleBasedBuyer(float(opening_target), float(concession_step))
        reward = summarize(evaluate(buyer.select_action, episodes, first_seed))["avg_reward"]
        rows.append({
            "opening_target": float(opening_target),
            "concession_step": float(concession_step),
            "validation_reward": round(reward, 5),
        })
    return sorted(rows, key=lambda row: row["validation_reward"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/baseline_tuning.csv")
    args = parser.parse_args()

    rows = tune(OPENING_TARGETS, CONCESSION_STEPS, VALIDATION_EPISODES, VALIDATION_SEED)
    save_csv(args.out, rows)

    print(f"{len(rows)} settings scored on {VALIDATION_EPISODES} validation sellers; top 5:")
    for row in rows[:5]:
        print(f"  opening {row['opening_target']:.2f}  step {row['concession_step']:.2f}  "
              f"reward {row['validation_reward']:+.3f}")
    print(f"saved all scores to {args.out}")


if __name__ == "__main__":
    main()
