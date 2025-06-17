import torch
import torch.nn as nn
import torch.nn.functional as F

# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, fc1_units=256, fc2_units=128):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(state_dim, fc1_units)
#         self.fc2 = nn.Linear(fc1_units, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, action_dim)

#     def forward(self, state):
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         # 출력층에 Softmax를 적용하여 가중치의 합이 1이 되도록 함
#         return F.softmax(self.fc3(x), dim=-1)

# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim, fc1_units=256, fc2_units=128):
#         super(Critic, self).__init__()
#         # 상태(state) 처리 경로
#         self.fcs1 = nn.Linear(state_dim, fc1_units)
#         # 상태와 행동(action)을 결합한 후의 경로
#         self.fc2 = nn.Linear(fc1_units + action_dim, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, 1)

#     def forward(self, state, action):
#         xs = F.relu(self.fcs1(state))
#         # 상태 처리 결과와 행동을 결합
#         x = torch.cat((xs, action), dim=1)
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)

# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, fc1_units=256, fc2_units=128):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(state_dim, fc1_units)
#         self.bn1 = nn.BatchNorm1d(fc1_units) # 배치 정규화 추가
#         self.fc2 = nn.Linear(fc1_units, fc2_units)
#         self.bn2 = nn.BatchNorm1d(fc2_units) # 배치 정규화 추가
#         self.fc3 = nn.Linear(fc2_units, action_dim)

#     def forward(self, state):
#         # state가 단일 샘플 (batch_size=1)일 때 BN 에러 방지
#         if state.dim() == 1:
#             state = state.unsqueeze(0)
            
#         x = F.relu(self.bn1(self.fc1(state)))
#         x = F.relu(self.bn2(self.fc2(x)))
#         return F.softmax(self.fc3(x), dim=-1)

# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim, fc1_units=256, fc2_units=128):
#         super(Critic, self).__init__()
#         self.fcs1 = nn.Linear(state_dim, fc1_units)
#         self.bns1 = nn.BatchNorm1d(fc1_units) # 배치 정규화 추가
#         self.fc2 = nn.Linear(fc1_units + action_dim, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, 1)

#     def forward(self, state, action):
#         # state가 단일 샘플 (batch_size=1)일 때 BN 에러 방지
#         if state.dim() == 1:
#             state = state.unsqueeze(0)

#         xs = F.relu(self.bns1(self.fcs1(state)))
#         x = torch.cat((xs, action), dim=1)
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_units=256, fc2_units=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_dim)

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        
        # --- 수정된 부분: softmax 제거 ---
        # 최종 가중치가 아닌, 각 행동에 대한 원시 점수(logit)를 반환
        return self.fc3(x)

class Critic(nn.Module):
    # (Critic은 수정 없음, 기존 코드와 동일)
    def __init__(self, state_dim, action_dim, fc1_units=256, fc2_units=128):
        super(Critic, self).__init__()
        self.fcs1 = nn.Linear(state_dim, fc1_units)
        self.bns1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_dim, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

    def forward(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)

        xs = F.relu(self.bns1(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)