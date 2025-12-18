import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Tuple


class PolicyNetwork(nn.Module):
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        
        def layer_init(layer: nn.Linear, std: float=np.sqrt(2), bias_const: float=0.0) -> nn.Linear:
            nn.init.orthogonal_(layer.weight, std)
            nn.init.constant_(layer.bias, bias_const)
            return layer
        
        self.layers = nn.Sequential(
            layer_init(nn.Linear(in_features=state_dim, out_features=128)),
            nn.ReLU(),
            layer_init(nn.Linear(in_features=128, out_features=128)),
            nn.ReLU()
        )
        # Now we need two different heads for the means and standard deviations of our actions
        self.mean_head = layer_init(nn.Linear(in_features=128, out_features=action_dim), std=0.01)
        self.log_std_param = nn.Parameter(torch.zeros(1,action_dim)) # Learn log std parameter because that can be negative
        
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
            
        self.saved_log_probs = [] # Action probabilities over an episode
        self.saved_values = [] # Value prediction for states over an episode
        self.rewards = [] # Accumulated rewards over an episode
        
        # The following three buffers are the same except for MULTIPLE episodes
        self.batch_log_probs = [] 
        self.batch_values = []
        self.batch_rewards = []
        
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
            # Don't let this affect the policy gradient
            self.saved_log_probs.append(torch.zeros(size=dist.log_prob(action).shape, device=action_mean.device))
        else:
            # Sample an action according to the distribution
            action = dist.sample()
            # Find the action's probability with respect to our policy
            self.saved_log_probs.append(dist.log_prob(action))
            
        # Return the action
        action = torch.clamp(action, -2.0, 2.0)
        return action.item()
    
    def finish_episode(self):
        # Obtain cumulative rewards from the episode
        R = 0
        cumulative_rewards = []
        for i in range(len(self.rewards)-1, -1, -1):
            R = self.rewards[i] + self.gamma * R
            cumulative_rewards.insert(0, R)
        cumulative_rewards = torch.tensor(cumulative_rewards, dtype=torch.float32) # shape (T,)
            
        # Add to batch memory
        self.batch_log_probs.extend(self.saved_log_probs)
        self.batch_values.extend(self.saved_values)
        self.batch_rewards.extend(cumulative_rewards)
            
        # Clear memory for next episode
        self.saved_log_probs = [] 
        self.saved_values = [] 
        self.rewards = [] 
        
    def train(self):
        if len(self.batch_log_probs) > 0:
            # Calculate policy loss
            batch_log_probs = torch.stack(self.batch_log_probs).squeeze() # shape (B,)
            batch_returns = torch.stack(self.batch_rewards).squeeze()
            advantage = batch_returns
            
            if self.value_net != None:
                # We are using the advantage calculation
                batch_values = torch.stack(self.batch_values).squeeze() 
                # Calculate advantage WITHOUT affecting the value function's gradient
                advantage = batch_returns - batch_values.detach()
                
            # Normalize advantage over all our different trajectories in our batch
            norm_advantage = (advantage - torch.mean(advantage)) / (torch.std(advantage) + 1e-9)
            
            # We want to maximize the probability of taking actions with high advantage
            policy_loss = -torch.mean(batch_log_probs * norm_advantage)
            self.policy_optim.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.policy_optim.step()
            
            if self.value_net != None:
                # Calculate value loss
                value_loss = self.value_loss_fn(batch_values, batch_returns)
                self.value_optim.zero_grad()
                value_loss.backward()
                self.value_optim.step()
            
            self.batch_log_probs = []
            self.batch_rewards = []
            self.batch_values = []