import gym
from gym import spaces
import numpy as np

class MultiAssetPortfolioEnv(gym.Env):
    def __init__(self, df, assets, initial_weights, initial_portfolio_value=1.0, lookback_window=30, transaction_cost_pct=0.001, risk_aversion_coeff=0.05):
        """
        환경 객체를 초기화합니다.

        Args:
            df (pd.DataFrame): 시장 데이터
            assets (list): 자산 티커 리스트
            initial_weights (np.array): 초기 포트폴리오 가중치
            initial_portfolio_value (float): 초기 포트폴리오 가치
            lookback_window (int): 상태를 구성할 과거 데이터 기간
            transaction_cost_pct (float): 거래 비용 비율
            risk_aversion_coeff (float): 위험 회피 계수 (보상 함수에 사용)
        """
        super(MultiAssetPortfolioEnv, self).__init__()

        self.df = df
        # 인자로 받은 assets 리스트를 직접 사용하도록 수정
        self.assets = assets
        self.n_assets = len(self.assets)
        self.lookback_window = lookback_window
        self.transaction_cost_pct = transaction_cost_pct
        self.risk_aversion_coeff = risk_aversion_coeff # risk_aversion_coeff 초기화

        # 상태 공간 (State Space) 정의
        # 1. 시장 상태 (Market State): (lookback_window, n_assets, n_features) -> 수익률, 변동성, MA(3), RSI(1)
        # 2. 상관관계 상태 (Correlation State): 자산 간 상관관계 행렬의 상단 삼각형 부분
        # 3. 에이전트 상태 (Agent State): 현재 포트폴리오 가중치 (현금 포함)
        
        # 1. 시장 상태 피처 수 (자산 당)
        n_market_features = 6  # log_return, vol_20d, ma_5d, ma_20d, ma_60d, rsi_14d
        
        # 2. 상관관계 피처 수 (고유한 자산 쌍의 수)
        n_correlation_features = self.n_assets * (self.n_assets - 1) // 2
        
        # 3. 에이전트 상태 피처 수 (현금 포함 가중치)
        n_agent_features = self.n_assets + 1
        
        # 전체 상태 공간의 크기
        state_shape = (self.lookback_window * self.n_assets * n_market_features) + n_correlation_features + n_agent_features
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_shape,), dtype=np.float32)

        # 행동 공간 (Action Space) 정의
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32)

        # 초기 설정
        self.initial_weights = np.array(initial_weights)
        self.initial_portfolio_value = initial_portfolio_value # 초기 포트폴리오 가치 설정
        self.current_step = 0
        self.portfolio_value = self.initial_portfolio_value # 포트폴리오 가치 초기화
        self.weights = self.initial_weights

    def reset(self):
        # 환경 초기화
        self.current_step = self.lookback_window
        # 하드코딩된 값(1.0) 대신 초기 포트폴리오 가치로 리셋하도록 수정
        self.portfolio_value = self.initial_portfolio_value
        self.weights = self.initial_weights
        return self._get_state()

    def _get_state(self):
        # 현재 시점의 상태를 반환
        start = self.current_step - self.lookback_window
        end = self.current_step
        
        # 1. 시장 상태 (Market State)
        market_feature_cols = []
        for asset in self.assets:
            market_feature_cols.extend([
                f"{asset}_log_return", f"{asset}_vol_20d", f"{asset}_ma_5d",
                f"{asset}_ma_20d", f"{asset}_ma_60d", f"{asset}_rsi_14d"
            ])
        market_state = self.df[market_feature_cols].iloc[start:end].values.flatten()
        
        # 2. 상관관계 상태 (Correlation State)
        # lookback 기간 동안의 자산별 로그 수익률로 상관관계 계산
        correlation_df = self.df[[f"{asset}_log_return" for asset in self.assets]].iloc[start:end]
        correlation_matrix = correlation_df.corr().values
        # 중복을 피하기 위해 상관관계 행렬의 상단 삼각형(upper triangle) 부분만 사용 (대각선 제외)
        # np.triu_indices는 상단 삼각형의 인덱스를 반환
        triu_indices = np.triu_indices(self.n_assets, k=1)
        correlation_state = correlation_matrix[triu_indices].flatten()

        # 3. 에이전트 상태 (Agent State)
        agent_state = self.weights
        
        # 모든 상태를 하나로 결합
        return np.concatenate((market_state, correlation_state, agent_state))

    def step(self, action):
        target_weights = action

        prev_weights_rebalanced = self._get_rebalanced_weights()
        
        turnover = np.sum(np.abs(target_weights - prev_weights_rebalanced))
        transaction_costs = turnover * self.transaction_cost_pct * self.portfolio_value
        
        self.portfolio_value -= transaction_costs
        
        returns_at_next_step = self.df[[f"{asset}_log_return" for asset in self.assets]].iloc[self.current_step]
        asset_returns = np.append(returns_at_next_step.values, 0)
        
        portfolio_log_return = np.dot(target_weights, asset_returns)
        
        # 보상(Reward) 계산
        
        # 최근 20일간의 수익률로 포트폴리오 변동성(위험) 계산
        # self.current_step이 lookback_window보다 작을 경우를 대비
        start_idx = max(0, self.current_step - self.lookback_window)
        historical_returns = self.df[[f"{asset}_log_return" for asset in self.assets]].iloc[start_idx:self.current_step]
        
        # 현금(0)을 포함한 자산별 수익률에 현재 가중치를 곱하여 포트폴리오 과거 수익률 계산
        historical_portfolio_returns = np.dot(historical_returns, target_weights[:-1]) # 현금 제외 가중치 곱
        
        # 포트폴리오 변동성 (위험)
        portfolio_volatility = np.std(historical_portfolio_returns) if len(historical_portfolio_returns) > 1 else 0

        # 수정된 보상: (수익률 - 거래비용) - (위험회피계수 * 변동성)
        # self.risk_aversion_coeff는 환경 초기화 시 설정 (예: 0.05)
        reward = portfolio_log_return - (self.transaction_cost_pct * turnover) - (self.risk_aversion_coeff * portfolio_volatility)
        
        # 포트폴리오 가치 업데이트
        self.portfolio_value *= np.exp(portfolio_log_return)
        
        self.current_step += 1
        self.weights = target_weights
        done = self.current_step >= len(self.df) - 1
        
        next_state = self._get_state() if not done else np.zeros(self.observation_space.shape)
        
        return next_state, reward, done, {'portfolio_value': self.portfolio_value, 'turnover': turnover}

    def _get_rebalanced_weights(self):
        # 이전 스텝의 자산 가격 변동을 반영하여 현재 가중치를 재계산
        # self.current_step이 0일 경우를 대비하여 max(0, ...) 추가
        prev_step_idx = max(0, self.current_step - 1)
        prev_returns = self.df[[f"{asset}_log_return" for asset in self.assets]].iloc[prev_step_idx]
        asset_returns = np.exp(np.append(prev_returns.values, 0))
        
        rebalanced_values = self.weights * asset_returns
        rebalanced_weights = rebalanced_values / np.sum(rebalanced_values)
        return rebalanced_weights