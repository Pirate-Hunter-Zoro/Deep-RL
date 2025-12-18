import gymnasium as gym
from cartpole.dqn import DQNAgent
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np

SMOOTHING_WINDOW = 50
EPOCHS = 500

def run_experiment(name: str, double_dqn: bool=False, soft_update: bool=False, tau: float=0.01, target_update_freq: int=1):   
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n
    agent = DQNAgent(state_dim=state_size, num_actions=num_actions, double_dqn=double_dqn, soft_update=soft_update, tau=tau)
    episode_rewards = [] # to plot
    
    for e in range(EPOCHS):
        # Gather an episode of experience
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.act(state=obs)
            next_state, reward, terminated, truncated, _ = env.step(action=action)
            total_reward += reward
            done = terminated or truncated
            agent.remember(state=obs, action=action, reward=reward, next_state=next_state, done=done)
            obs = next_state
            agent.replay() # Updates the policy network if enough experiences have been remembered
        
        # Sync the target network if the episode count falls on the update frequency
        if (((e + 1) % target_update_freq) == 0) and (not soft_update):
            agent.update_target_network()
            
        # Record reward
        episode_rewards.append(total_reward)
        if ((e+1) % (int(EPOCHS/5))) == 0:
            print(f"Experiment {name}, Episode {e+1}/{EPOCHS}, Total Reward: {total_reward}")
    
    # Plot smoothed episode rewards
    weights = 1/SMOOTHING_WINDOW * np.ones(shape=(SMOOTHING_WINDOW,), dtype=np.float32)
    episode_rewards = np.convolve(a=np.array(episode_rewards), v=weights, mode='valid')
    plt.figure()
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward by Episode")
    fig_path = Path(f"results/dqn/dqn_cartpole_{name}.png")
    os.makedirs(fig_path.parent, exist_ok=True)
    plt.savefig(str(fig_path))
    plt.close()

def main():
    run_experiment("plain_dqn")
    run_experiment("delayed_double_dqn", double_dqn=True, target_update_freq=32)
    run_experiment("soft_update", double_dqn=True, soft_update=True)
        
if __name__ == '__main__':
    main()