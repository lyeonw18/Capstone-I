import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

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


from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=10000):  # 여기서 capacity 받도록 수정
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


def compute_reward(pred, label, age, priors, X, model, idx, k=5):
    # 1. 기본 정확도 보상
    acc_reward = 1.0 if pred == label else 0.0

    # 2. Demographic Parity: 나이 그룹별로 유죄 예측률이 과도하게 차이날 경우 패널티
    # 특정 나이 그룹이 유죄(1)를 더 많이 예측받는 것을 억제
    age_penalty = 0.0
    if pred == 1:  # 유죄로 예측한 경우만
        if age <= 25 or age > 45:
            age_penalty = 0.2  # 연령 그룹에 따라 페널티 부과

    # 3. Individual Fairness: 주변의 유사한 샘플과 예측이 다르면 패널티
    from sklearn.metrics.pairwise import euclidean_distances
    dists = euclidean_distances([X[idx]], X)
    nearest_idx = np.argsort(dists[0])[1:k+1]
    with torch.no_grad():
        neighbor_preds = model(torch.from_numpy(X[nearest_idx]).float()).argmax(dim=1).cpu().numpy()
    diff_count = np.sum(neighbor_preds != pred)
    fairness_penalty = 0.1 * (diff_count / k)

    # 4. 최종 보상
    total_reward = acc_reward - age_penalty - fairness_penalty
    return total_reward


def train_dqn(X, y, age, priors, epochs=10, batch_size=128, gamma=0.99, lr=1e-3, device='cpu'):
    n_actions = 2
    input_dim = X.shape[1]
    model = DQNetwork(input_dim, hidden_dim=128, n_actions=n_actions).to(device)
    target_model = DQNetwork(input_dim, hidden_dim=128, n_actions=n_actions).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=10000)
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    X_tensor = torch.from_numpy(X).float().to(device)
    y_np = y.astype(np.int64)
    age_np = age
    priors_np = priors

    n = len(X)
    for epoch in range(epochs):
        idxs = np.random.permutation(n)
        for idx in idxs:
            state = X[idx]
            label = y_np[idx]
            age_val = age_np[idx]
            priors_val = priors_np[idx]
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            # Epsilon-greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(n_actions)
            else:
                with torch.no_grad():
                    q_values = model(state_tensor)
                    action = q_values.argmax().item()
            # 보상 계산
            reward = compute_reward(action, label, age_val, priors_val, X, model, idx)  # 보상 계산 수정
            # DQN은 next_state와 done 필요 (여기선 단일 샘플이라 self-loop)
            next_state = state
            done = True
            buffer.push(state, action, reward, next_state, done)
            # 학습
            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states = torch.from_numpy(states).float().to(device)
                actions = torch.from_numpy(actions).long().to(device)
                rewards = torch.from_numpy(rewards).float().to(device)
                next_states = torch.from_numpy(next_states).float().to(device)
                dones = torch.from_numpy(dones).float().to(device)

                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_model(next_states).max(1)[0]
                    targets = rewards + gamma * next_q_values * (1 - dones)
                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # 타깃 네트워크 업데이트
        if (epoch+1) % 2 == 0:
            target_model.load_state_dict(model.state_dict())
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print(f"Epoch {epoch}: epsilon={epsilon:.3f}")
    return model


def evaluate_dqn(model, X, y, age, priors, device='cpu'):
    model.eval()
    X_tensor = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        q_values = model(X_tensor)
        preds = torch.argmax(q_values, dim=1).cpu().numpy()

    # 정확도 계산
    acc = accuracy_score(y, preds)

    # Demographic Parity: 나이, 전과 그룹별 유죄 예측 비율 차이
    age_young = preds[age <= 25]
    age_old = preds[age > 45]
    age_bias = np.abs(age_young.mean() - age_old.mean()) if len(age_young) > 0 and len(age_old) > 0 else 0.0

    prior_low = preds[priors == 0]
    prior_high = preds[priors >= 3]
    prior_bias = np.abs(prior_low.mean() - prior_high.mean()) if len(prior_low) > 0 and len(prior_high) > 0 else 0.0

    # Individual Fairness: 유사한 특성 가진 샘플들 예측 값 비교
    from sklearn.metrics.pairwise import euclidean_distances
    dists = euclidean_distances(X, X)
    k = 5  # 유사한 k개의 샘플
    fairness_penalty = 0.0

    for i in range(len(X)):
        nearest_idx = np.argsort(dists[i])[1:k+1]
        neighbor_preds = preds[nearest_idx]
        # 예측값이 다른 샘플들과 차이를 가지면 패널티
        fairness_penalty += np.sum(neighbor_preds != preds[i]) / k

    fairness_penalty /= len(X)  # 평균 패널티

    # 결과 출력
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Age Bias: {age_bias:.4f}")
    print(f"Prior Bias: {prior_bias:.4f}")
    print(f"Individual Fairness Penalty: {fairness_penalty:.4f}")

    return acc, age_bias, prior_bias, fairness_penalty
