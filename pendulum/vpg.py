import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Tuple


class PolicyNetwork(nn.Module):
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU()
        )
        # Now we need two different heads for the means and standard deviations of our actions
        self.mean_head = nn.Linear(in_features=128, out_features=action_dim)
        self.log_std_param = nn.Parameter(torch.zeros(1,action_dim)) # Learn log std parameter because that can be negative
        
        # Initialize weights to be small
        with torch.no_grad():
            self.mean_head.weight.mul_(0.1)
            self.mean_head.bias.zero_()
        
    
    def forward(self, state: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        x = self.layers(state)
        # Mean action value
        action_mean = self.mean_head(x)
        action_std = self.log_std_param.exp()
        return action_mean, action_std
    

class ValueNetwork(nn.Module):
    
    def __init__(self, state_dim: int):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1) # We just want the raw value of the state if we play optimally
        )
        
    def forward(self, state: torch.tensor) -> torch.tensor:
        return self.layers(state)
    
    
class VPGAgent:
    
    def __init__(self, state_dim: int, action_dim: int, lr: float=1e-4, gamma: float=0.99, use_baseline: bool=False):
        self.policy_net = PolicyNetwork(state_dim=state_dim, action_dim=action_dim)
        self.policy_optim = optim.Adam(params=self.policy_net.parameters(), lr=lr)
        self.action_dim = action_dim
        self.gamma = gamma
        self.value_net = None
        self.value_optim = None
        self.value_loss_fn = None
        if use_baseline:
            self.value_net = ValueNetwork(state_dim=state_dim)
            self.value_optim = optim.Adam(params=self.value_net.parameters(), lr=lr)
            self.value_loss_fn = nn.MSELoss()
            
        self.saved_log_probs = [] # Action probabilities over time
        self.saved_values = [] # Value prediction for states over time
        self.rewards = [] # Accumulated rewards over time
        
    def act(self, state: np.array, epsilon: float=0.0) -> int:
        state = torch.tensor(state, dtype=torch.float32)
        action_mean, action_std = self.policy_net(state)
        # Create a probability distribution out of the action probabilities and sample from it
        dist = torch.distributions.Normal(action_mean, action_std)
        if self.value_net != None:
            # Use baseline
            estimated_value = self.value_net(state)
            self.saved_values.append(estimated_value)
        
        if random.random() < epsilon:
            # Random uniformly sampled action
            action = torch.distributions.Uniform(low=-2.0,high=2.0).sample()
        else:
            # Sample an action according to the distribution
            action = dist.sample()
            
        # Find the action's probability with respect to our policy - even if we took a completely uniformly random action, the probability of that action according to our policy still needs to be stored
        self.saved_log_probs.append(dist.log_prob(action))
            
        # Return the action
        action = torch.clamp(action, -2.0, 2.0)
        return action.item()
    
    def update(self):
        # Obtain cumulative rewards from the episode
        R = 0
        cumulative_rewards = []
        for i in range(len(self.rewards)-1, -1, -1):
            R = self.rewards[i] + self.gamma * R
            cumulative_rewards.insert(0, R)
        normalized_rewards = torch.tensor(cumulative_rewards, dtype=torch.float32) # shape (T,)
        normalized_rewards = (normalized_rewards - torch.mean(normalized_rewards)) / (torch.std(normalized_rewards) + 1e-9)
        advantage = normalized_rewards
        
        if self.value_net != None:
            # Calculate baseline
            values = torch.stack(self.saved_values).squeeze() # shape (T,)
            advantage = normalized_rewards - values.detach() # Policy should treat baseline as fixed number and not mess with the value network's parameters
        
        # Calculate policy loss
        log_probs = torch.stack(self.saved_log_probs).squeeze() # shape (T,)
        policy_loss = -torch.mean(log_probs * advantage) # We want to maximize the probability of taking actions with high advantage
        self.policy_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_optim.step()
        
        if self.value_net != None:
            # Calculate value loss
            value_loss = self.value_loss_fn(values, normalized_rewards)
            self.value_optim.zero_grad()
            value_loss.backward()
            self.value_optim.step()
            
        # Clear memory for next episode
        self.saved_log_probs = [] 
        self.saved_values = [] 
        self.rewards = [] 