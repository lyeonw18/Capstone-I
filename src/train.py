# train.py

import numpy as np
import pandas as pd
from dqn_core import DQNetwork, ReplayBuffer, train_dqn, evaluate_dqn


# 데이터 불러오기 (예시)
# X: feature matrix, y: labels, age: 나이, priors: 전과 수
X = np.load("data/X.npy")
y = np.load("data/y.npy")
age = np.load("data/age.npy")
priors = np.load("data/priors.npy")


# 학습
model = train_dqn(
    X, y, age, priors,
    epochs=10,
    batch_size=128,
    gamma=0.99,
    lr=1e-3,
    device='cpu'
)


# 평가
acc, age_bias, prior_bias, fairness_penalty = evaluate_dqn(model, X, y, age, priors, device='cpu')


# 결과 저장
results = pd.DataFrame({
    "accuracy": [acc],
    "age_bias": [age_bias],
    "prior_bias": [prior_bias],
    "fairness_penalty": [fairness_penalty]
})
results.to_csv("results/evaluation.csv", index=False)
