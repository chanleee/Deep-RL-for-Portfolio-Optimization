import random
from collections import namedtuple, deque
import numpy as np
import torch

# GPU 사용 설정 (agent.py와 동일하게)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """에이전트의 경험을 저장하고 샘플링하기 위한 고정 크기 버퍼 (Uniform Sampling)."""

    def __init__(self, buffer_size, batch_size, device, seed):
        """
        ReplayBuffer 객체를 초기화합니다.
        
        Args:
            buffer_size (int): 버퍼의 최대 크기
            batch_size (int): 한 번에 샘플링할 배치의 크기
            device (torch.device): 연산에 사용할 디바이스 (CPU or GPU)
            seed (int): 재현성을 위한 난수 시드
        """
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = device
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """새로운 경험(Transition)을 메모리에 추가합니다."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """
        메모리에서 무작위로 경험의 배치를 샘플링하고 PyTorch 텐서로 변환합니다.
        
        Returns:
            Tuple[torch.Tensor]: (states, actions, rewards, next_states, dones) 튜플
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        # 각 experience의 필드들을 NumPy 배열로 변환 후 vstack을 통해 수직으로 쌓습니다.
        # 이 방식은 state와 action이 다차원 벡터일 때도 완벽하게 작동합니다.
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """현재 메모리에 저장된 경험의 수를 반환합니다."""
        return len(self.memory)