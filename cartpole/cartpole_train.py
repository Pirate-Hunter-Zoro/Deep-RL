import gymnasium as gym
from cartpole.dqn import DQNAgent
import matplotlib.pyplot as plt
from pathlib import Path
import os

EPOCHS = 500

def main():
    
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n
    agent = DQNAgent(state_dim=state_size, num_actions=num_actions)
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
        # Now that the episode is done, let's sync the target network
        agent.update_target_network()
        episode_rewards.append(total_reward)
        if ((e+1) % (int(EPOCHS/10))) == 0:
            print(f"Episode {e+1}/{EPOCHS}, Total Reward: {total_reward}")
    
    # Plot episode rewards
    plt.figure()
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward by Episode")
    fig_path = Path(f"results/dqn_cartpole.png")
    os.makedirs(fig_path.parent, exist_ok=True)
    plt.savefig(str(fig_path))
    plt.close()
        
if __name__ == '__main__':
    main()