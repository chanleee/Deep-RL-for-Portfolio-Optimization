# agent.py

import os
from time import sleep
from collections import deque, namedtuple

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from IPython import display

# 로컬 모듈 임포트: 새로운 models와 memory를 사용합니다.
from memory import ReplayBuffer # 수정된 memory.py의 ReplayBuffer 클래스
from models import Actor, Critic # 수정된 models.py의 Actor, Critic 클래스

# GPU 사용 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 하이퍼파라미터 ---
BUFFER_SIZE = int(1e6)  # 리플레이 버퍼 크기
BATCH_SIZE = 128        # 미니배치 크기
GAMMA = 0.99            # 할인 계수 (Discount factor)
TAU = 1e-3              # 타겟 네트워크 소프트 업데이트 계수
LR_ACTOR = 1e-4         # Actor 학습률
LR_CRITIC = 1e-3        # Critic 학습률
WEIGHT_DECAY = 0        # L2 가중치 감쇠 (Critic)
EXPLORATION_STD = 0.1   # 행동 탐색을 위한 가우시안 노이즈의 표준편차
LEARN_EVERY = 1         # 몇 스텝마다 학습할지 결정
LEARN_NUM = 1           # 한 번 학습할 때 몇 번의 배치를 학습할지 결정

class Agent():
    """DDPG 에이전트: 환경과 상호작용하고 학습합니다."""

    def __init__(self, state_size, action_size, random_seed):
        """
        Agent 객체를 초기화합니다.
        
        Args:
            state_size (int): 상태 공간의 차원
            action_size (int): 행동 공간의 차원
            random_seed (int): 난수 시드
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Actor Network (로컬 및 타겟)
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (로컬 및 타겟)
        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # 타겟 네트워크와 로컬 네트워크의 가중치를 동일하게 초기화
        self.hard_update(self.actor_local, self.actor_target)
        self.hard_update(self.critic_local, self.critic_target)

        # 리플레이 메모리 초기화
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device, random_seed)
        
        # 타임스텝 카운터
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """경험을 리플레이 버퍼에 저장하고, 주기적으로 학습을 수행합니다."""
        # 경험 저장
        self.memory.add(state, action, reward, next_state, done)

        # 정해진 스텝마다 학습 수행
        self.t_step = (self.t_step + 1) % LEARN_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                for _ in range(LEARN_NUM):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    def act(self, state, use_exploration=True):
        """주어진 상태에 대해 행동(포트폴리오 가중치)을 반환합니다."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            # action은 이제 (action_size,) 형태의 벡터
            action = self.actor_local(state).cpu().data.numpy().flatten()
        self.actor_local.train()

        if use_exploration:
            # OU Noise 대신 가우시안 노이즈 추가
            noise = np.random.normal(0, EXPLORATION_STD, size=self.action_size)
            action += noise
        
        # 행동 후처리: 모델 출력층에 Softmax가 있더라도 노이즈로 인해 제약이 깨질 수 있으므로,
        # clip과 정규화는 안전장치로 유지하는 것이 좋습니다.
        action = np.clip(action, 0, 1)
        action /= (action.sum() + 1e-8) # 0으로 나누는 것 방지

        return action

    def learn(self, experiences, gamma):
        """
        주어진 경험 배치(batch)를 사용하여 가치 함수와 정책을 업데이트합니다.
        (이 부분의 로직은 표준 DDPG 알고리즘으로, 벡터 데이터에서도 동일하게 작동합니다.)
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------- CRITIC 업데이트 ---------------- #
        # 타겟 Actor로부터 다음 행동을 예측
        actions_next = self.actor_target(next_states)
        # 타겟 Critic으로부터 다음 상태-행동 쌍의 Q-value를 계산
        Q_targets_next = self.critic_target(next_states, actions_next)
        # 현재 상태에 대한 Q-target 계산 (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # 로컬 Critic으로부터 현재 Q-value 예측
        Q_expected = self.critic_local(states, actions)
        # Critic 손실 계산 (MSE)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Critic 업데이트
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) # Gradient Clipping
        self.critic_optimizer.step()

        # ---------------- ACTOR 업데이트 ----------------- #
        # 로컬 Actor로부터 행동 예측
        actions_pred = self.actor_local(states)
        # Actor 손실 계산 (Critic이 예측한 Q-value를 최대화하는 방향으로)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Actor 업데이트
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------------- TARGET 네트워크 업데이트 ----------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        타겟 네트워크 가중치를 소프트 업데이트합니다.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def hard_update(self, local_model, target_model):
        """타겟 네트워크 가중치를 로컬 네트워크와 동일하게 하드 업데이트합니다."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)