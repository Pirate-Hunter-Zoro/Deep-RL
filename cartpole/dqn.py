import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class QNetwork(nn.Module):
    
    def __init__(self, state_dim: int=4, num_actions: int=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=24),
            nn.ReLU(),
            nn.Linear(in_features=24, out_features=24),
            nn.ReLU(),
            nn.Linear(in_features=24, out_features=num_actions)
        )
        
    def forward(self, state: torch.tensor) -> torch.tensor:
        # Pass the state through the layers of our network and get the RAW q-values
        return self.layers(state)
    
    
class DQNAgent():
    
    def __init__(self, state_dim: int=4, num_actions: int=2, gamma: float=0.99, epsilon: float=1.0, epsilon_min: float=0.01, epsilon_decay: float=0.995, batch_size: int=64, lr: float=3e-4, double_dqn: bool=False, soft_update: bool=False, tau: float=0.01):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.buffer = deque(maxlen=2000)
        self.loss_fn = nn.MSELoss()
        self.double_dqn = double_dqn
        self.soft_update = soft_update
        self.tau = tau
        
        # We need our policy network and target network
        self.policy_network = QNetwork(state_dim=state_dim, num_actions=num_actions)
        self.target_network = QNetwork(state_dim=state_dim, num_actions=num_actions)
        self.optimizer = optim.Adam(params=self.policy_network.parameters(), lr=lr)
    
    def update_target_network(self):
        if not self.soft_update:
            # hard update
            self.target_network.load_state_dict(state_dict=self.policy_network.state_dict())
        else:
            # soft update
            for target_param, policy_param in zip(self.target_network.parameters(), self.policy_network.parameters()):
                target_param.data.copy_((self.tau * policy_param.data) + ((1-self.tau) * target_param.data))
    
    def act(self, state: np.array) -> int:
        # Given a state, select the next action
        if random.random() < self.epsilon:
            # random action
            return 0 if random.random() < 0.5 else 1
        else:
            # greedy action
            state = torch.from_numpy(state).float().unsqueeze(0) # convert to tensor and add batch dimension
            q_value_logits = self.policy_network(state)
            return torch.argmax(q_value_logits, dim=-1).item()
        
    def remember(self, state: np.array, action: int, reward: float, next_state: np.array, done: bool):
        self.buffer.append((state, action, reward, next_state, 1 if done else 0))
        
    def replay(self):
        if len(self.buffer) >= self.batch_size:
            batch = random.sample(population=self.buffer, k=self.batch_size)
            states = torch.tensor(np.array([t[0] for t in batch]), dtype=torch.float)
            actions = torch.tensor(np.array([t[1] for t in batch]), dtype=torch.long)
            rewards = torch.tensor(np.array([t[2] for t in batch]), dtype=torch.float)
            next_states = torch.tensor(np.array([t[3] for t in batch]), dtype=torch.float)
            dones = torch.tensor(np.array([t[4] for t in batch]), dtype=torch.float)
            
            # Calculate our target values
            if not self.double_dqn:
                # Target network both picks next actions and evaluates them
                max_future_q = torch.max(self.target_network(next_states), axis=-1)[0]
            else:
                # Policy network picks next actions and target network evaluates them
                next_actions = torch.argmax(self.policy_network(next_states), axis=-1)
                target_all_q_values = self.target_network(next_states)
                max_future_q = torch.gather(input=target_all_q_values, dim=-1, index=next_actions.unsqueeze(dim=-1)).squeeze(1)
            
            target_q_values = (rewards + (self.gamma * max_future_q * (1-dones)))
            target_q_values = target_q_values.unsqueeze(dim=-1)
            
            # Now pass the current state into the policy network to see our actual q_values
            q_values_all_actions = self.policy_network(states)
            q_values_taken_actions = torch.gather(input=q_values_all_actions, dim=-1, index=actions.unsqueeze(dim=-1))
            
            # Calculate the MSE loss
            loss = self.loss_fn(q_values_taken_actions, target_q_values)
            
            # Perform training step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # If using soft update, update immediately
            if self.soft_update:
                self.update_target_network()
            
            self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)