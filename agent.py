import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm

from memory import ReplayBuffer
from models import Actor, Critic
from evaluation import evaluate_for_validation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- TD3 하이퍼파라미터 ---
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4  # 학습률 조정
LR_CRITIC = 1e-3 # 학습률 조정

# --- TD3 관련 신규 하이퍼파라미터 ---
POLICY_NOISE = 0.2      # 타겟 정책 스무딩 노이즈
NOISE_CLIP = 0.5        # 스무딩 노이즈 클리핑 범위
POLICY_FREQ = 2         # 정책(Actor) 업데이트 주기 (2번의 Critic 업데이트마다 1번 업데이트)

class Agent():
    """TD3 에이전트: DDPG의 안정성을 개선한 버전."""

    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Actor Network (로컬 및 타겟)
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Networks (쌍둥이 Critic: 1, 2)
        self.critic_1_local = Critic(state_size, action_size).to(device)
        self.critic_1_target = Critic(state_size, action_size).to(device)
        self.critic_1_target.load_state_dict(self.critic_1_local.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1_local.parameters(), lr=LR_CRITIC)

        self.critic_2_local = Critic(state_size, action_size).to(device)
        self.critic_2_target = Critic(state_size, action_size).to(device)
        self.critic_2_target.load_state_dict(self.critic_2_local.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2_local.parameters(), lr=LR_CRITIC)
        
        # 리플레이 메모리 초기화
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device, random_seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > BATCH_SIZE:
            self.t_step += 1
            self.learn(self.memory.sample(), GAMMA)

    def act(self, state, exploration_std=0.1):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy().flatten()
        self.actor_local.train()
        
        # 행동에 탐색 노이즈 추가
        noise = np.random.normal(0, exploration_std, size=self.action_size)
        action = (action + noise).clip(0, 1)
        action /= (action.sum() + 1e-8)
        return action

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # ---------------- CRITIC 업데이트 ---------------- #
        # 1. 타겟 정책 스무딩: 다음 행동에 노이즈 추가
        noise = torch.randn_like(actions).data.normal_(0, POLICY_NOISE).to(device).clamp(-NOISE_CLIP, NOISE_CLIP)
        actions_next = (self.actor_target(next_states) + noise).clamp(0, 1)
        
        # 2. 쌍둥이 Critic: 다음 상태의 Q-value 계산 후 더 작은 값 선택
        Q1_targets_next = self.critic_1_target(next_states, actions_next)
        Q2_targets_next = self.critic_2_target(next_states, actions_next)
        Q_targets_next = torch.min(Q1_targets_next, Q2_targets_next)
        
        # 현재 상태에 대한 최종 Q-target 계산
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Critic 1 업데이트
        Q1_expected = self.critic_1_local(states, actions)
        critic_1_loss = F.mse_loss(Q1_expected, Q_targets.detach())
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        # Critic 2 업데이트
        Q2_expected = self.critic_2_local(states, actions)
        critic_2_loss = F.mse_loss(Q2_expected, Q_targets.detach())
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # ---------------- ACTOR 지연 업데이트 ---------------- #
        if self.t_step % POLICY_FREQ == 0:
            # Actor 손실 계산
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_1_local(states, actions_pred).mean()
            
            # Actor 업데이트
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Target 네트워크 소프트 업데이트
            self.soft_update(self.critic_1_local, self.critic_1_target, TAU)
            self.soft_update(self.critic_2_local, self.critic_2_target, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    # train 함수는 조기 종료 로직을 포함하여 더 개선될 수 있습니다.
    # 이전 답변의 train 함수를 기반으로 수정하여 사용하시는 것을 권장합니다.
    def train(self, train_env, validation_env, n_episodes, model_path, initial_exploration_std=0.2, min_exploration_std=0.05):
        scores_deque = deque(maxlen=10)
        best_validation_score = -np.inf
        
        # 탐색 노이즈 감소(Decay)를 위한 설정
        exploration_decay_rate = (initial_exploration_std - min_exploration_std) / n_episodes
        
        progress_bar = tqdm(range(1, n_episodes + 1), desc="Training Progress (TD3)")
        for i_episode in progress_bar:
            state = train_env.reset()
            score = 0
            done = False
            
            # 현재 에피소드의 탐색 노이즈 크기 계산
            current_exploration_std = initial_exploration_std - (i_episode * exploration_decay_rate)
            
            while not done:
                action = self.act(state, exploration_std=current_exploration_std)
                next_state, reward, done, _ = train_env.step(action)
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done: break
            
            scores_deque.append(score)
            avg_score = np.mean(scores_deque)
            progress_bar.set_postfix({"Avg Train Score": f"{avg_score:.2f}", "Exploration": f"{current_exploration_std:.3f}"})

            if i_episode % 5 == 0:
                validation_score = evaluate_for_validation(validation_env, self)
                tqdm.write(f"\nEpisode {i_episode}\tAvg Train Score: {avg_score:.2f}\tValidation Sharpe Ratio: {validation_score:.4f}")
                if validation_score > best_validation_score:
                    best_validation_score = validation_score
                    tqdm.write(f"🎉 New best model found! Saving model to {model_path}")
                    torch.save(self.actor_local.state_dict(), os.path.join(model_path, 'best_actor.pth'))
                    torch.save(self.critic_1_local.state_dict(), os.path.join(model_path, 'best_critic.pth')) # Critic은 하나만 저장해도 무방