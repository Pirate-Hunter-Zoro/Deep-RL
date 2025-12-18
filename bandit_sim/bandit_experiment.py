from bandit_sim.bandit_sim import Bandit_Sim
from bandit_sim.bandit_algorithms import EpsilonGreedy, ThompsonSampling, UCB
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def run_experiments():
    for arm_count in [5, 10, 20]:
        print(f"Running multi-arm bandit experiment with {arm_count} arms...", flush=True)
        horizon = 500 * arm_count
        bandit = Bandit_Sim(n_arms=arm_count)
        
        agents = {
            '0.1Epsilon' : EpsilonGreedy(epsilon=0.1),
            '0.2Epsilon' : EpsilonGreedy(epsilon=0.2),
            '0.3Epsilon' : EpsilonGreedy(epsilon=0.3),
            'Thompson' : ThompsonSampling(),
            'UCB' : UCB()
        }
        
        results = {agent: None for agent in agents.keys()}
        
        for agent_label, agent in agents.items():
            # Run the agent
            history, cumulative_rewards = agent.run(bandit, horizon)
            
            # Calculate regret - at time step t, expected best score is (t+1)*best_average
            regret = (np.arange(start=1, stop=horizon+1, step=1) * max(bandit.arm_means)) - cumulative_rewards
            
            results[agent_label] = (history, regret)
            
        results_path = Path(f"results/multi_arm_bandit/{arm_count}_arms")
        os.makedirs(results_path, exist_ok=True)
            
        # Plot regret
        plt.figure(figsize=(10,6))
        for label, performance in results.items():
            regret = performance[1]
            plt.plot(regret, label=label)
        plt.title("Cumulative Regret")
        plt.xlabel("Time Step")
        plt.ylabel("Regret")
        plt.legend()
        plt.grid()
        plt.savefig(results_path / f"multi_arm_bandit_regret_{arm_count}_arms.png")
        
        plt.close()
        
        # Plot estimated best arm
        plt.figure(figsize=(10,6))
        plt.yticks(ticks=range(arm_count))
        for label, performance in results.items():
            history = performance[0]
            # At each point in time, the agent had an estimated best arm
            best = np.argmax(history, axis=1)
            plt.scatter(x=range(horizon), y=best, label=label, alpha=0.7)
        plt.title("Expected Best Arm Over Horizon")
        plt.xlabel("Time Step")
        plt.ylabel("Predicted Best Arm")
        plt.legend()
        plt.grid()
        plt.savefig(results_path / f"multi_arm_bandit_best_arm_estimate_{arm_count}.png")
        
if __name__=="__main__":
    run_experiments()