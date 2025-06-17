import numpy as np
import pandas as pd

def calculate_initial_weights(assets, method='equal'):
    """
    초기 포트폴리오 가중치를 계산합니다.
    현재는 '동일 비중' 방식만 구현되어 있습니다.

    Args:
        assets (list): 자산 티커 리스트.
        method (str): 가중치 계산 방식. 'equal'은 동일 비중을 의미합니다.

    Returns:
        np.array: 각 자산과 현금에 대한 가중치 벡터 (n_assets + 1,).
                  env.py의 action_space 정의에 따라 현금(cash) 항목이 마지막에 추가됩니다.
    """
    n_assets = len(assets)
    if n_assets == 0:
        return np.array([1.0]) # 자산이 없으면 현금 100%

    if method == 'equal':
        # 각 자산에 동일한 가중치를 할당하고, 현금 비중은 0으로 시작합니다.
        asset_weights = np.full(n_assets, 1.0 / n_assets)
        # 현금(cash) 가중치(0.0)를 배열의 끝에 추가합니다.
        initial_weights = np.append(asset_weights, 0.0)
    else:
        # 다른 메소드가 추가될 경우를 대비해 예외 처리
        raise NotImplementedError(f"The method '{method}' is not implemented.")
        
    return initial_weights

def build_ou_process(T=10000, n_assets=1, theta=0.1, sigma=0.1, random_state=None):
    """
    여러 자산에 대한 Ornstein-Uhlenbeck (OU) 프로세스 신호를 생성합니다.
    (참고: 현재 프로젝트의 메인 파이프라인(summary.py)에서는 사용되지 않지만,
     향후 시뮬레이션 환경 테스트 등을 위해 다중 자산 처리가 가능하도록 수정되었습니다.)
    
    Args:
        T (int): 신호의 길이.
        n_assets (int): 시뮬레이션할 자산의 수.
        theta (float): OU 프로세스 파라미터.
        sigma (float): OU 프로세스 파라미터.
        random_state (int, optional): 재현성을 위한 난수 시드.

    Returns:
        np.array: 생성된 OU 신호. 형태는 (T, n_assets).
    """
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random

    # 다중 자산을 위한 2D 배열 생성
    X = np.empty((T, n_assets))
    x = np.zeros(n_assets) # 각 자산의 초기값은 0
    
    # 각 자산, 각 시점에 대한 노이즈를 한 번에 생성
    normals = rng.normal(0, 1, (T, n_assets))

    for t in range(T):
        # NumPy 벡터 연산을 통해 모든 자산을 동시에 업데이트
        x += -x * theta + sigma * normals[t]
        X[t] = x
        
    # 모든 자산에 대해 정규화
    X /= sigma * np.sqrt(1.0 / (2.0 * theta))
    
    return X


def get_returns(signal, random_state=None):
    """
    주어진 신호에 가우시안 노이즈를 추가하여 수익률을 계산합니다.
    (참고: 현재 프로젝트의 메인 파이프라인(summary.py)에서는 사용되지 않습니다.)
    
    Args:
        signal (np.array): 1D 또는 2D 형태의 신호.
        random_state (int, optional): 재현성을 위한 난수 시드.

    Returns:
        np.array: 입력 신호와 동일한 형태의 수익률 배열.
    """
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random
    
    # signal의 형태(shape)와 동일한 형태의 노이즈를 생성하여 더합니다.
    # 이로 인해 입력이 1D(단일 자산)든 2D(다중 자산)든 모두 처리 가능합니다.
    noise = rng.normal(size=signal.shape)
    
    return signal + noise