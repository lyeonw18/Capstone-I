# src/train.py
import numpy as np
import pandas as pd
from dqn_core import train_dqn, evaluate_dqn

# 전처리된 데이터 불러오기 (또는 data_processing.py에서 로드)
X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")
age_train = np.load("data/age_train.npy")       # 필요하면
priors_train = np.load("data/priors_train.npy") # 필요하면
age_test = np.load("data/age_test.npy")
priors_test = np.load("data/priors_test.npy")

# DQN 학습
model = train_dqn(X_train, y_train, age_train, priors_train,
                  epochs=10, batch_size=128, device='cpu')

# 평가
acc, age_bias, prior_bias, fairness_penalty = evaluate_dqn(
    model, X_test, y_test, age_test, priors_test
)

# 결과 저장
import os
os.makedirs("results", exist_ok=True)
pd.DataFrame({
    "accuracy": [acc],
    "age_bias": [age_bias],
    "prior_bias": [prior_bias],
    "fairness_penalty": [fairness_penalty]
}).to_csv("results/evaluation.csv", index=False)
