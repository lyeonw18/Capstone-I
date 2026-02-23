# Capstone I

## Overview
This project was conducted as part of the undergraduate capstone course.
The goal of this project was to analyze the bias issues of the COMPAS algorithm and suggest ways to improve the model to strengthen the fairness and ethical responsibility of AI-based decision-making systems.

## Dataset
- Source:
- Number of samples:
- Key features:

## Method
1. Data preprocessing
  - Filtered COMPAS dataset based on screening date and validity conditions
  - Defined “current offense” per individual using temporal logic
  - Encoded categorical variables (Label Encoding)
  - Balanced violent vs non-violent samples using downsampling
  - Train/Test split (70/30)
2. Baseline Models
  - Logistic Regression
  - Random Forest (GridSearchCV tuning)
  - XGBoost (Hyperparameter tuning)
  - SVM
  - Linear Regression
- Evaluation metric:
  - RMSE
  - Accuracy
  - Balanced Accuracy
3.  Proposed Ethical RL Framework
   - Implemented Deep Q-Network (DQN)
   - Experience Replay Buffer
   - Target Network stabilization
   - Epsilon-greedy exploration strategy
### Ethical Reward Design
  Designed a custom reward function integrating:
  1. **Group Fairness**
   - Demographic Parity penalty
   - Equalized Odds constraint

  2. **Responsibility**
   - Penalization of disproportionate false positives for vulnerable groups

  3. **Individual Fairness**
   - k-NN-based similarity consistency penalty

  Final integrated reward:

  Integrated Reward = (Fairness + Responsibility + Consistency) / 3

4. Evaluation Metrics
   - Accuracy
   - Balanced Accuracy
   - Demographic Parity Difference
   - Individual Fairness score
   - Age Bias / Prior Bias


## Results
- Best model:
- Performance:
- Key findings:

## What I Learned
- 
- 
- 

## Repository Structure
- `src/` : core source code
- `notebooks/` : exploratory analysis
- `results/` : experiment outputs




## Key Contributions

- Proposed an ethically-aware Deep Q-Network for recidivism prediction
- Integrated fairness constraints directly into the reinforcement learning reward
- Compared traditional ML models with policy-based optimization
- Evaluated trade-offs between accuracy and fairness
