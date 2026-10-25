# BazaarBot

A reinforcement learning buyer that haggles in a simulated Indian bazaar.

Every seller has a **hidden minimum price** and a **hidden personality**. The
buyer never sees either; it has to read them from how the seller behaves (how
fast they drop their price, whether a "final price" is really final, whether
they call it back after it walks away) and buy as cheaply as it can without
losing the deal.

The buyer is a Deep Q-Network written from scratch in PyTorch. No RL library is
used: the replay buffer, training loop and DQN update are all in this repo.
Rule-based buyers are included only as baselines to compare against.

## Setup

Requires Python 3.12.

Create the virtual environment:

```bash
python -m venv myenv
```

Activate it:

```powershell
# Windows (PowerShell)
myenv\Scripts\Activate.ps1
```

```bash
# Windows (Git Bash)
source myenv/Scripts/activate

# macOS / Linux
source myenv/bin/activate
```

If PowerShell blocks the activation script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

The tests also need pytest (`python -m pip install pytest`).

## Usage

```bash
python play.py watch                 # watch the trained bot haggle with a simulated seller
python play.py sell                  # you play the seller against the bot

python train.py --seed 0             # train one DQN (10,000 episodes), saved to models/dqn_seed0.pt
python evaluate.py                   # train 5 DQNs and compare them with every baseline
python tune_baseline.py              # re-tune the rule-based baseline on the validation sellers
python inspect_agent.py              # how the trained bot behaves against each personality

python -m pytest                     # run the tests
```

`play.py` and `inspect_agent.py` use `models/dqn_seed10.pt` by default (the run
with the best validation score); pass `--model` to use another one.

## How it works

### The negotiation (environment)

One episode is one negotiation for one item (`bazaar/env.py`, a Gymnasium
environment). At the start:

- the item is worth **V** ~ U[Rs 100, Rs 2000] to the buyer, who knows V;
- the seller's hidden minimum is **R** = V x U[0.6, 0.9];
- the seller opens at R x U[1.5, 2.5], and will wait 5 to 10 rounds;
- the seller gets one of four hidden personalities, each with probability 1/4.

Each round the buyer acts, then the seller answers. The seller accepts any
offer of at least R; otherwise they give up a share of the gap between their
ask and the offer, never going below R.

| personality | behaviour |
|---|---|
| stubborn | gives up a small, fixed share of the gap (5-15%) |
| desperate | starts at 20-35% and gives up 10 points more every round |
| bluffer | gives up 10-30%, but 30% of the time holds its price and falsely says "final price" |
| tourist-spotter | gives up 10-30%, but raises its price 20% if the buyer's first offer is eager (50% of the ask or more) |

Every seller honestly says "final price" once they reach R. Each round's
concession is scaled by random noise (x U[0.8, 1.2]), an offer below 0.5 R
costs a round of the seller's patience, and a buyer who walks away after making
an offer may be called back once, at a better price, with probability
0.8 x (last offer / R).

### State, actions and reward

**State** (8 numbers, all prices divided by V so one policy works at any price):
current ask, opening ask, buyer's last offer (0 if none), rounds played / 10,
seller's last price drop, seller's average price drop, "final price" flag,
"called back" flag. R and the personality are never in the state.

**Actions** (10, discrete):

| index | action |
|---|---|
| 0-3 | open at 30 / 40 / 50 / 60% of the ask (first offer only) |
| 4-7 | concede 0 / 10 / 25 / 50% of the gap between the last offer and the ask |
| 8 | accept the current ask |
| 9 | walk away |

**Reward:** -0.01 every round, plus, when the negotiation ends, 1 - price / V
for a deal (the share of the item's value saved) or -0.3 for no deal.

### The agent

`bazaar/dqn_agent.py`, trained by `train.py`:

- Q-network: 8 -> 128 -> 128 -> 10, ReLU, Adam (learning rate 1e-3), Huber loss
- target network synced every 250 steps; discount 0.99
- replay buffer of 20,000 transitions, batches of 64, learning from step 500
- epsilon-greedy, falling linearly from 1.0 to 0.05 over the first 5,000 steps
- **action masking**: the agent only picks (and bootstraps from) actions that
  are valid in the current state, so it never wastes a turn opening twice
- every 500 episodes the greedy policy is scored on 500 fixed **validation**
  sellers, and the best-scoring network is the one kept

### Baselines

`bazaar/baseline.py`: a random buyer; a "tourist" who accepts the first price;
"split the difference" (open at 50%, then always meet halfway); "steady
concession" (open at 40%, then concede 25% a round); and a rule-based buyer that
opens near a target fraction of V and moves part of the way towards V each
round. The rule-based buyer appears twice: with hand-set settings, and tuned by
`tune_baseline.py` on the same validation sellers the DQN uses to pick its
checkpoint.

### Keeping the evaluation honest

Three separate sets of sellers are used. Training draws a fresh seller every
episode; checkpoint selection, baseline tuning and the DQN's own settings use
the validation sellers (seeds 20000-20499); final results use test sellers
(seeds 100000-100999) that were not used for any decision. Every policy meets
exactly the same test sellers, and the DQN is trained 5 times with different
seeds to show how much results vary between runs.

## Reproducing the results

```bash
python tune_baseline.py   # writes results/baseline_tuning.csv (~10 s)
python evaluate.py        # writes models/ and results/ (took about 10 minutes here)
python inspect_agent.py   # behaviour table and example negotiations
```

Training is deterministic: it uses one CPU thread on purpose, so the same seed
gives the same network on any machine.

`results/` holds CSV files meant to be plotted directly in the report:
`summary.csv` and `by_personality.csv` (every policy's metrics),
`learning_curve_seed*.csv` (validation reward during training) and
`baseline_tuning.csv` (every baseline setting tried).

## Results

From `python evaluate.py`: 1,000 test sellers (seeds 100000-100999), greedy
policies, and 5 DQNs trained for 10,000 episodes each (seeds 0, 10, 20, 30, 40).
Prices are fractions of V. "Captured" is the share of the best possible surplus,
which is paying exactly the seller's minimum R (an oracle that knew R would get
100%). Even that oracle would average only about +0.23 reward here.

| policy | deal rate | reward | deal price | captured | rounds |
|---|---|---|---|---|---|
| random | 73.4% | -0.162 | 1.086 | -25% | 1.87 |
| tourist (accepts first price) | 100% | -0.512 | 1.502 | -201% | 1.00 |
| split the difference | 100% | +0.071 | 0.914 | 35% | 1.51 |
| steady concession | 95.3% | +0.111 | 0.840 | 61% | 2.77 |
| rule-based, hand-set | 97.1% | +0.169 | 0.781 | 85% | 3.48 |
| rule-based, tuned | 99.1% | +0.179 | 0.792 | 83% | 2.45 |
| **DQN** (mean +/- std of 5 runs) | **99.3% +/- 0.2** | **+0.181 +/- 0.002** | 0.787 +/- 0.003 | 85% +/- 1 | 2.86 +/- 0.22 |

Reward by seller personality (DQN: mean of 5 runs):

| seller (test count) | rule-based, hand-set | rule-based, tuned | DQN |
|---|---|---|---|
| stubborn (266) | +0.163 | +0.175 | +0.178 +/- 0.002 |
| desperate (242) | +0.184 | +0.189 | +0.190 +/- 0.007 |
| bluffer (264) | +0.154 | +0.167 | +0.172 +/- 0.001 |
| tourist-spotter (228) | +0.178 | +0.187 | +0.186 +/- 0.003 |

What this shows:

- The DQN, which starts with no strategy at all and learns only from reward,
  clearly beats every simple baseline and the hand-set rule-based buyer
  (+0.012 reward, ahead in all 5 runs).
- Against the rule-based buyer **tuned on the validation sellers**, it is level
  or slightly ahead: +0.002 +/- 0.002, ahead in all 5 runs, but that margin is
  within the noise (paired t = 2.6 over 5 runs). Its clearest edge is against
  bluffers (+0.005, ahead in all 5 runs).
- It closes 99.3% of deals, about as many as the tuned rule (99.1%), at a far
  better price than the one baseline that always closes (split the difference,
  0.914 V).
- It treats sellers differently. Averaged over the 5 runs, it pays desperate
  sellers 0.772 V and stubborn ones 0.794 V, a gap of 0.022 V. The tuned rule's
  gap is only 0.009 V (0.785 against 0.794 V), and that much comes from
  desperate sellers simply dropping their price faster.
- The seed 10 model (`inspect_agent.py`) holds its offer in 12% of its
  concessions against desperate sellers but 5% against stubborn ones. It never
  accepted a bluffed "final price" (neither did the tuned rule), but accepted
  78% of the honest ones from desperate sellers, against 38% for the tuned rule.
- The seed 10 model never walks away, so call-backs never come into play: with
  a 0.3 penalty for no deal, walking away is not worth the risk.

Training is noisy (after episode 2,000 the validation reward still dropped by up
to 0.016-0.059 between consecutive checks, depending on the run), which is why
the best validation checkpoint is kept. `models/dqn_seed10.pt` is the run
with the best validation score, and is the one the demo uses.

## Project layout

```
bazaar/
  env.py            the negotiation environment (Gymnasium)
  dqn_agent.py      Q-network and DQN agent
  replay_buffer.py  experience replay
  baseline.py       rule-based and other baseline buyers
  evaluation.py     metrics shared by every script
  dialogue.py       turns negotiations into readable dialogue
train.py            trains one DQN
evaluate.py         trains several DQNs and compares them with the baselines
tune_baseline.py    tunes the rule-based baseline on validation sellers
inspect_agent.py    behaviour breakdown and example negotiations
play.py             terminal demo
models/             trained networks
results/            CSV results for the report
tests/              tests (run with python -m pytest)
```
