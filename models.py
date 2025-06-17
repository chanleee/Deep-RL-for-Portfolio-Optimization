# models.py (PPO를 위한 ActorCritic 모델로 수정)

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init, fc1_units=256, fc2_units=128):
        super(ActorCritic, self).__init__()

        # --- Actor (정책 신경망) ---
        self.actor = nn.Sequential(
            nn.Linear(state_dim, fc1_units),
            nn.Tanh(),
            nn.Linear(fc1_units, fc2_units),
            nn.Tanh(),
            nn.Linear(fc2_units, action_dim),
            nn.Softmax(dim=-1) # Softmax를 사용하여 가중치의 합이 1이 되도록 강제
        )

        # --- Critic (가치 신경망) ---
        self.critic = nn.Sequential(
            nn.Linear(state_dim, fc1_units),
            nn.Tanh(),
            nn.Linear(fc1_units, fc2_units),
            nn.Tanh(),
            nn.Linear(fc2_units, 1)
        )
        
        # 행동의 표준편차 (탐색의 범위를 결정하는 학습 가능한 파라미터)
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init)

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.actor[-2].out_features,), new_action_std * new_action_std)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        # 주어진 상태(state)에 대한 행동(action)과 로그 확률(log_prob)을 반환
        action_probs = self.actor(state)
        # 공분산 행렬 생성 (대각 행렬로 가정하여 자산 간 독립적인 노이즈 추가)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_probs, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        # 행동값이 0과 1 사이를 벗어나지 않도록 클리핑하고, 합이 1이 되도록 정규화
        action = torch.clamp(action, 0, 1)
        action = action / (action.sum(dim=-1, keepdim=True) + 1e-8)
        
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        # 주어진 상태(state)와 행동(action)에 대한 가치(value), 로그 확률(log_prob), 엔트로피(entropy)를 반환
        action_probs = self.actor(state)
        cov_mat = torch.diag(self.action_var)
        dist = MultivariateNormal(action_probs, cov_mat)
        
        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprob, state_values, dist_entropy