# dqn_core.py (최소 재현용)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_actions=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))
    def __len__(self):
        return len(self.buffer)

def compute_reward(pred, label):
    """간단 최소 보상: Accuracy만 사용"""
    return 1.0 if pred == label else 0.0

def train_dqn(X, y, epochs=10, batch_size=128, gamma=0.99, lr=1e-3):
    n_actions = 2
    input_dim = X.shape[1]
    model = DQNetwork(input_dim, n_actions=n_actions)
    target_model = DQNetwork(input_dim, n_actions=n_actions)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=lr)
    buffer = ReplayBuffer(10000)
    epsilon = 1.0

    for epoch in range(epochs):
        for idx in np.random.permutation(len(X)):
            state = X[idx]
            label = y[idx]
            # Epsilon-greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(n_actions)
            else:
                with torch.no_grad():
                    action = model(torch.from_numpy(state).float().unsqueeze(0)).argmax().item()
            reward = compute_reward(action, label)
            buffer.push(state, action, reward, state, True)
            # batch update
            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states = torch.from_numpy(states).float()
                actions = torch.from_numpy(actions).long()
                rewards = torch.from_numpy(rewards).float()
                next_states = torch.from_numpy(next_states).float()
                dones = torch.from_numpy(dones).float()
                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target_model(next_states).max(1)[0]
                    target = rewards + gamma * next_q * (1 - dones)
                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print(f"Epoch {epoch} completed, epsilon={epsilon:.3f}")
        epsilon = max(epsilon*0.995, 0.05)
        target_model.load_state_dict(model.state_dict())

    return model
