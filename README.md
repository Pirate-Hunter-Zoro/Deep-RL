# Deep Reinforcement Learning Algorithms

This repository contains implementations of various Reinforcement Learning algorithms, ranging from basic multi-armed bandits to deep policy gradients. The code investigates the performance of these agents across different environments including custom Multi-Armed Bandits, the Wumpus World (Tabular RL), CartPole (DQN), and Pendulum (VPG).

## Directory Structure

The project is organized into four main modules, each targeting a specific class of RL problems:

```text
.
├── bandit_sim/           # Multi-Armed Bandit implementations (Epsilon Greedy, UCB, Thompson)
├── mdp/           # Tabular RL (Q-Learning, SARSA) in a Wumpus World environment
├── dqn/             # Deep Q-Networks (DQN, Double DQN) for CartPole-v1
├── vpg/             # Vanilla Policy Gradient (VPG) for Pendulum-v1
├── results/              # Generated plots and performance metrics
└── requirements.txt      # Python dependencies

```

## Installation

Ensure you have a Python environment capable of sustaining these constructs.

```bash
# Install dependencies
pip install -r requirements.txt

```

*Requires: `numpy`, `matplotlib`, `gymnasium`, `torch`.*

## Usage

Run the experiments as modules from the root directory.

### 1. Multi-Armed Bandit

Compares **Epsilon-Greedy**, **Thompson Sampling**, and **Upper Confidence Bound (UCB)** agents across 5, 10, and 20 arms.

```bash
python -m bandit_sim.bandit_experiment

```

### 2. Tabular RL (Wumpus World)

Evaluates **Q-Learning** and **SARSA** agents in three custom Wumpus World layouts (Book World, Corner World, Middle World).

```bash
python -m mdp.mdp_experiment

```

### 3. Deep Q-Network (CartPole)

Trains a **DQN** agent on `CartPole-v1`. Includes implementations for:

* Standard DQN
* Double DQN
* Soft Target Updates

```bash
python -m dqn.dqn_train

```

### 4. Vanilla Policy Gradient (Pendulum)

Trains a **Continuous VPG** agent on `Pendulum-v1` using a squashed Gaussian policy. Experiments include:

* VPG with Baseline (Critic)
* VPG without Baseline
* VPG with varying Epsilon (masked exploration)

```bash
python -m vpg.vpg_train

```

## Results

Training curves and regret plots are automatically generated and stored in the `results/` directory.

* **Bandits**: Regret and Best Arm Estimates.
* **Tabular**: Smoothed rewards over episodes.
* **DQN**: Total reward comparisons between DQN variants.
* **VPG**: Performance of the agent with and without baseline variance reduction.
