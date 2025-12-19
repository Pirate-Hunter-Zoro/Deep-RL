from pendulum.vpg import VPGAgent
import gymnasium as gym
from gymnasium.wrappers import (
    NormalizeObservation, 
    NormalizeReward, 
    TransformObservation, 
    TransformReward,
    RecordEpisodeStatistics
)
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

SMOOTHING_WINDOW = 500
EPOCHS = 5000
BATCH_SIZE = 10

def train_vpg(experiment_name: str, use_baseline: bool, num_runs: int, epsilon: float=0.1):
    env = gym.make('Pendulum-v1')
    env = RecordEpisodeStatistics(env)
    env = NormalizeObservation(env)
    env = TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space) # clamps each normalized observation
    env = NormalizeReward(env)
    env = TransformReward(env, lambda r: np.clip(r, -10, 10)) # clamps each normalized reward
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    for i in range(num_runs):
        agent = VPGAgent(state_dim=state_dim, action_dim=action_dim, use_baseline=use_baseline)
        episode_rewards = np.zeros(shape=(EPOCHS,)) # for plotting
        for e in range(EPOCHS):
            obs, _ = env.reset()
            done = False
            while not done:
                action = agent.act(state=obs, epsilon=epsilon)
                next_state, reward, terminated, truncated, info = env.step([action]) # Because pendulum wants a list; not float
                agent.rewards.append(reward) # necessary for VPG
                obs = next_state
                done = terminated or truncated
                
            # Update after an episode
            agent.finish_episode(last_state=next_state, truncated=truncated)
            # If batch is full, perform a training step
            if ((e+1) % BATCH_SIZE == 0):
                agent.train()
            
            # Save the raw reward for plotting later
            episode_rewards[e] = info['episode']['r']
            if ((e+1) % (int(EPOCHS/5))) == 0:
                print(f"Experiment {experiment_name}, Episode {e+1}/{EPOCHS}, Total Reward: {info['episode']['r']}, Length: {info['episode']['l']}", flush=True)
                
        # Plot smoothed episode rewards
        weights = 1/SMOOTHING_WINDOW * np.ones(shape=(SMOOTHING_WINDOW,), dtype=np.float32)
        episode_rewards = np.convolve(a=episode_rewards, v=weights, mode='valid')
        plt.figure()
        plt.plot(episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Total Reward by Episode")
        fig_path = Path(f"results/vpg/{experiment_name}/pendulum_run_{i}_{experiment_name}.png")
        os.makedirs(fig_path.parent, exist_ok=True)
        plt.savefig(str(fig_path))
        plt.close()
        
def main():
    train_vpg(experiment_name="vpg_no_baseline", use_baseline=False, num_runs=3)
    train_vpg(experiment_name="vpg_baseline", use_baseline=True, num_runs=3)
    train_vpg(experiment_name="vpg_baseline_0_epsilon", use_baseline=True, num_runs=1, epsilon=0.0)
    train_vpg(experiment_name="vpg_baseline_0.01_epsilon", use_baseline=True, num_runs=1, epsilon=0.01)
    train_vpg(experiment_name="vpg_baseline_0.1_epsilon", use_baseline=True, num_runs=1, epsilon=0.1)
    
if __name__=="__main__":
    main()