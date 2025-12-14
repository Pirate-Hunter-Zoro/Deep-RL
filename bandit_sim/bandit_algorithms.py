from bandit_sim.bandit_sim import Bandit_Sim
import numpy as np
import random
from typing import Tuple


class EpsilonGreedy:
    """Class which samples arms based on an epsilon-greedy policy
    """
    
    def __init__(self, epsilon: float):
        """Initialization for an epsilon-greedy agent

        Args:
            epsilon (float): Probability of taking a random action
        """
        self.epsilon = epsilon
        
    def run(self, bandit: Bandit_Sim, horizon: int) -> Tuple[np.array, np.array]:
        """Method to run the multi-arm bandit epsilon greedy algorithm with the given simulator and the number of horizon

        Args:
            bandit (Bandit_Sim): Bandit simulator
            horizon (int): Number of steps

        Returns:
            Tuple[np.array, np.array]: History and cumulative rewards
        """
        n_arms = bandit.n_arms
        # Number of times each arm has been pulled
        counts = np.zeros(shape=(n_arms,), dtype=np.int32)
        # Estimated mean value for each arm
        values = np.zeros(shape=(n_arms,), dtype=np.float32)
        
        # The following is needed for plotting estimated best action
        history = np.zeros(shape=(horizon, n_arms))
        
        # Cumulative sum of rewards over the horizon
        cumulative_rewards = np.zeros(shape=(horizon,), dtype=np.float32)
        
        # Total reward achieved so far
        total_reward = 0.0
        
        # Loop over the horizon, pulling arms and recording rewards
        for t in range(horizon):
            # We need to pull every arm once
            if t < n_arms:
                action = t
            else:
                if random.random() < self.epsilon:
                    action = int(random.random()*n_arms)
                else:
                    action = np.argmax(a=values)
            
            # Now that we know which arm we will pull
            reward = bandit.pull_arm(n=action)
            # Calculate the new average reward for that arm and update its count
            values[action] = (counts[action]*values[action] + reward) / (counts[action] + 1)
            counts[action] += 1
            
            # Increment total reward
            total_reward += reward
            cumulative_rewards[t] = total_reward
            
            # Store current payouts for arms in history
            history[t] = values
        
        # Return history and cumulative rewards
        return (history, cumulative_rewards)
    

class ThompsonSampling:
    """Class which samples arms based on Thompson sampling
    """
    
    def __init__(self):
        """No argments needed for constructor
        """
        pass
    
    def run(self, bandit: Bandit_Sim, horizon: int) -> Tuple[np.array, np.array]:
        """Method to run the Thompson Sampling algorithm with the given simulator and the number of horizon

        Args:
            bandit (Bandit_Sim): Bandit simulator
            horizon (int): Number of steps

        Returns:
            Tuple[np.array, np.array]: History and cumulative rewards
        """
        n_arms = bandit.n_arms
        # Parameters of a beta distribution (an array of each, because each arm has one beta distribution)
        alphas, betas = np.ones(shape=(n_arms,), dtype=np.float32), np.ones(shape=(n_arms,), dtype=np.float32)
       
        # The following is needed for plotting estimated best action
        history = np.zeros(shape=(horizon, n_arms))
        
        # Cumulative sum of rewards over the horizon
        cumulative_rewards = np.zeros(shape=(horizon,), dtype=np.float32)
        
        # Total reward achieved so far
        total_reward = 0.0
        
        # Loop over the horizon
        for t in range(horizon):
            # For each arm, draw a random sample from its beta distribution
            arm_samples = np.random.beta(a=alphas, b=betas) # Handy-dandy one-shot return numpy call
            
            # Select the arm with the highest sample
            action = np.argmax(a=arm_samples)
            
            # Now that we know which arm we will pull
            reward = bandit.pull_arm(n=action)
            # Update the beta distribution for this arm
            alphas[action] += reward
            betas[action] += (1 - reward)
            
            # Increment total reward
            total_reward += reward
            cumulative_rewards[t] = total_reward
            
            # Store current estimated payouts for arms in history - for each arm that is this value: alpha_arm / (alpha_arm + beta_arm)
            history[t] = alphas / (alphas + betas)
            
        # Return history and cumulative rewards
        return (history, cumulative_rewards)
    

class UCB:
    
    def __init__(self, c: float=np.sqrt(2)):
        """Initialization for upper confidence bound agent

        Args:
            c (float, optional): Exploration parameter. Defaults to np.sqrt(2).
        """
        self.c = c
        
    def run(self, bandit: Bandit_Sim, horizon: int) -> Tuple[np.array, np.array]:
        """Method to run the upper confidence bound algorithm with the given simulator and the number of horizon

        Args:
            bandit (Bandit_Sim): Bandit simulator
            horizon (int): Number of steps

        Returns:
            Tuple[np.array, np.array]: History and cumulative rewards
        """
        n_arms = bandit.n_arms
        counts = np.zeros(shape=(n_arms,), dtype=np.int32)
        values = np.zeros(shape=(n_arms,), dtype=np.float32)
        history = np.zeros(shape=(horizon, n_arms))
        cumulative_rewards = np.zeros(shape=(horizon,), dtype=np.float32)
        total_reward = 0.0
        
        for t in range(horizon):
            # We must pull each arm exactly once
            if t < n_arms:
                action = t
            else:
                # Find upper confidence bound for all arms
                ucb = values + self.c * np.sqrt(np.log(t)*np.reciprocal(counts))
                action = np.argmax(a=ucb)
            
            # Now that we have our arm
            reward = bandit.pull_arm(n=action)
            total_reward += reward
            cumulative_rewards[t] = total_reward
            
            # Update value for arm
            values[action] = (counts[action]*values[action] + reward) / (counts[action] + 1)
            counts[action] += 1
            
            # Record history
            history[t] = values
            
        return (history, cumulative_rewards)