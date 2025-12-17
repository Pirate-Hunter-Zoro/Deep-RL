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
    
    def __init__(self, state_dim: int=4, num_actions: int=2, gamma: float=0.99, epsilon: float=1.0, epsilon_min: float=0.01, epsilon_decay: float=0.995, batch_size: int=64, lr: float=3e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.buffer = deque(maxlen=2000)
        self.loss_fn = nn.MSELoss()
        
        # We need our policy network and target network
        self.policy_network = QNetwork(state_dim=state_dim, num_actions=num_actions)
        self.target_network = QNetwork(state_dim=state_dim, num_actions=num_actions)
        self.optimizer = optim.Adam(params=self.policy_network.parameters(), lr=lr)
    
    def update_target_network(self):
        self.target_network.load_state_dict(state_dict=self.policy_network.state_dict())
    
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
            states = torch.tensor([t[0] for t in batch])
            actions = torch.tensor([t[1] for t in batch], dtype=torch.long)
            rewards = torch.tensor([t[2] for t in batch])
            next_states = torch.tensor([t[3] for t in batch])
            dones = torch.tensor([t[4] for t in batch])
            
            # Calculate our target values
            max_future_q = torch.max(self.target_network(next_states), axis=-1)[0]
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
            
            self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)