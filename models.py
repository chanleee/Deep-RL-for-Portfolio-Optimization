import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_units=256, fc2_units=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # 출력층에 Softmax를 적용하여 가중치의 합이 1이 되도록 함
        return F.softmax(self.fc3(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_units=256, fc2_units=128):
        super(Critic, self).__init__()
        # 상태(state) 처리 경로
        self.fcs1 = nn.Linear(state_dim, fc1_units)
        # 상태와 행동(action)을 결합한 후의 경로
        self.fc2 = nn.Linear(fc1_units + action_dim, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

    def forward(self, state, action):
        xs = F.relu(self.fcs1(state))
        # 상태 처리 결과와 행동을 결합
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)