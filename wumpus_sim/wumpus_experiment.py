from wumpus_sim.wumpus import WumpusMDP
from wumpus_sim.tabular_rl import QLearningAgent, SARSAAgent
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

EPISODES = 5000
SMOOTHING_WINDOW = 100
START_EPSILON = 1.0
END_EPSILON = 0.05
DECAY_DURATION = int(EPISODES * 0.8) # decay over 80 percent of episodes
MAX_STEPS = 1000 # hard limit on steps per episode before we truncate

# NOTE - I had Gemini make the following functions to create the three wumpus worlds, but all other code came from my fingertips

def get_book_world():
    """
    The 3x4 World described in the Textbook/PDF.
    Dimensions: Width=3, Height=4.
    """
    mdp = WumpusMDP(3, 4)
    # Row 3 (Top): Empty, Empty, Gold
    mdp.add_obstacle('goal', [2, 3]) 
    # Row 2: Pit, Wumpus, Empty
    mdp.add_obstacle('pit', [0, 2], -10)
    mdp.add_obstacle('wumpus', [1, 2], -100)
    # Row 1: Empty, Empty, Empty (Safe)
    # Row 0: Start is usually (0,0)
    return "Book_World_3x4", mdp

def get_corner_world():
    """
    10x10 World with Reward at (10,10).
    Note: In 0-indexed Python, (10,10) is index [9, 9].
    """
    mdp = WumpusMDP(10, 10)
    mdp.add_obstacle('goal', [9, 9])
    return "Corner_World_10x10", mdp

def get_middle_world():
    """
    10x10 World with Reward at (5,5).
    """
    mdp = WumpusMDP(10, 10)
    mdp.add_obstacle('goal', [5, 5])
    return "Middle_World_10x10", mdp

def run_experiment_on_world(mdp: WumpusMDP, experiment_name: str):
    # Calculate epsilon decay
    epsilon_decay_step = (START_EPSILON - END_EPSILON) / DECAY_DURATION
    
    # Q-learning loop
    q_learning_rewards = np.zeros(shape=(EPISODES,), dtype=np.float32) # rewards by the episode
    agent = QLearningAgent(mdp=mdp, discount_factor=0.99, epsilon=START_EPSILON)
    for i in range(EPISODES):
        # Decay epsilon
        if i < DECAY_DURATION:
            agent.epsilon -= epsilon_decay_step
        
        state = mdp.initial_state
        total_reward = 0.0
        steps = 0
        while not mdp.is_terminal(state=state) and steps < MAX_STEPS:
            action = agent.choose_action(state=state)
            next_state, reward = mdp.act(state=state, action=action)
            agent.update(state=state, action=action, reward=reward, next_state=next_state)
            total_reward += reward
            state = next_state
            steps += 1
        q_learning_rewards[i] = total_reward
        if i % 500 == 0: print(f"Episode {i}: Reward {total_reward:.1f}, Epsilon {agent.epsilon:.3f}")
    
    # SARSA loop
    sarsa_rewards = np.zeros(shape=(EPISODES,), dtype=np.float32)
    agent = SARSAAgent(mdp=mdp, discount_factor=0.99, epsilon=START_EPSILON)
    for i in range(EPISODES):
        if i < DECAY_DURATION:
            agent.epsilon -= epsilon_decay_step
        
        state = mdp.initial_state
        total_reward = 0.0
        action = agent.choose_action(state=state)
        steps = 0
        while not mdp.is_terminal(state=state) and steps < MAX_STEPS:
            next_state, reward = mdp.act(state=state, action=action)
            next_action = agent.choose_action(state=next_state)
            agent.update(state=state, action=action, reward=reward, next_state=next_state, next_action=next_action)
            total_reward += reward
            state = next_state
            action = next_action
            steps += 1
        sarsa_rewards[i] = total_reward
        if i % 500 == 0: print(f"Episode {i}: Reward {total_reward:.1f}, Epsilon {agent.epsilon:.3f}")
        
    # Smooth rewards
    weights = 1/SMOOTHING_WINDOW * np.ones(shape=(SMOOTHING_WINDOW,), dtype=np.float32)
    q_learning_rewards = np.convolve(a=q_learning_rewards, v=weights, mode='valid')
    sarsa_rewards = np.convolve(a=sarsa_rewards, v=weights, mode='valid')
    
    # Plot smoothed rewards
    plt.figure()
    plt.plot(q_learning_rewards, label='Q-Learning', alpha=0.7)
    plt.plot(sarsa_rewards, label='SARSA', alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Reward")
    plt.title("Smoothed Reward by Episode")
    plt.legend()
    fig_path = Path(f"results/tabular_rl/wumpus_experiment_{experiment_name}.png")
    os.makedirs(fig_path.parent, exist_ok=True)
    plt.savefig(str(fig_path))
    plt.close()

def main():
    name, mdp = get_book_world()
    run_experiment_on_world(mdp=mdp, experiment_name=name)
    name, mdp = get_corner_world()
    run_experiment_on_world(mdp=mdp, experiment_name=name)
    name, mdp = get_middle_world()
    run_experiment_on_world(mdp=mdp, experiment_name=name)

if __name__=="__main__":
    main()