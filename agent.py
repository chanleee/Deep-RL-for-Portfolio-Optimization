# agent.py (PPO 알고리즘으로 전면 수정)

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import os
from tqdm import tqdm
from evaluation import evaluate_for_validation
from models import ActorCritic
import numpy as np

# GPU 사용 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    # PPO는 On-Policy 알고리즘이므로, 매 업데이트마다 메모리를 비웁니다.
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init):
        
        self.action_std = action_std_init
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def act(self, state, memory=None):
        """
        주어진 상태에 대해 행동을 결정합니다.
        훈련 시(memory != None): 확률적으로 행동을 샘플링하고 메모리에 저장합니다.
        평가 시(memory == None): 가장 확률 높은 행동(평균)을 결정론적으로 반환합니다.
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action_mean = self.policy_old.actor(state)
            cov_mat = torch.diag(self.policy_old.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)

            # 훈련 시에는 샘플링, 평가 시에는 평균값(가장 가능성 높은 행동) 사용
            if memory is not None:
                action = dist.sample()
                action_logprob = dist.log_prob(action)
                memory.states.append(state)
                memory.actions.append(action)
                memory.logprobs.append(action_logprob)
            else:
                action = action_mean

        # 정규화하여 가중치의 합이 1이 되도록 함
        action = torch.clamp(action, 0, 1)
        action = action / (action.sum(dim=-1, keepdim=True) + 1e-8)

        return action.detach().cpu().numpy().flatten()

    def evaluate(self, state, action):
        return self.policy.evaluate(state, action)

    def update(self, memory):
        # 몬테카를로 방식으로 보상-투-고(Reward-to-go) 계산
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(device)
        
        # K 에포크 동안 정책 최적화
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)
            
            advantages = rewards - state_values.detach()
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # --- 수정된 부분: state_values.squeeze()로 차원 축소 ---
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values.squeeze(), rewards) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        memory.clear_memory()

    def train(self, train_env, validation_env, max_training_timesteps, update_timestep, model_path):
        # PPO는 에피소드 기반이 아닌 타임스텝 기반으로 훈련하는 것이 일반적
        memory = Memory()
        
        timestep = 0
        best_validation_score = -np.inf

        progress_bar = tqdm(range(1, int(max_training_timesteps)+1), desc="Training Timesteps (PPO)")
        
        while timestep < max_training_timesteps:
            state = train_env.reset()
            done = False
            
            while not done and timestep < max_training_timesteps:
                # 현재 타임스텝에서 행동 결정
                action = self.act(state, memory)
                state, reward, done, _ = train_env.step(action)
                
                # 메모리에 보상과 종료 여부 저장
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                
                timestep += 1
                progress_bar.update(1)
                
                # 일정 타임스텝마다 정책 업데이트
                if timestep % update_timestep == 0:
                    self.update(memory)

                # 주기적으로 검증 및 모델 저장
                if timestep % (update_timestep * 5) == 0: # 5번 업데이트마다 검증
                    validation_score = evaluate_for_validation(validation_env, self)
                    tqdm.write(f"\nTimestep {timestep}\tValidation Sharpe Ratio: {validation_score:.4f}")
                    if validation_score > best_validation_score:
                        best_validation_score = validation_score
                        tqdm.write(f"🎉 New best model found! Saving model to {model_path}")
                        torch.save(self.policy.state_dict(), os.path.join(model_path, 'best_ppo_model.pth'))