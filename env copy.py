import gym
from gym import spaces

class MultiAssetPortfolioEnv(gym.Env):
    def __init__(self, df, initial_weights, lookback_window=30, transaction_cost_pct=0.001):
        super(MultiAssetPortfolioEnv, self).__init__()

        self.df = df
        self.assets = [col.split('_') for col in df.columns if col.endswith('_Adj_Close')]
        self.n_assets = len(self.assets)
        self.lookback_window = lookback_window
        self.transaction_cost_pct = transaction_cost_pct

        # 1. 상태 공간 (State Space) 정의
        # 시장 상태 (과거 데이터) + 에이전트 상태 (현재 가중치)
        # 시장 상태: (lookback_window, n_assets, n_features) -> 여기서는 수익률과 변동성(2)
        # 에이전트 상태: n_assets + 1 (현금 포함)
        state_shape = (lookback_window * self.n_assets * 2) + (self.n_assets + 1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_shape,), dtype=np.float32)

        # 2. 행동 공간 (Action Space) 정의
        # 각 자산 + 현금에 대한 목표 가중치 벡터
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32)

        # 초기 설정
        self.initial_weights = np.array(initial_weights)
        self.current_step = 0
        self.portfolio_value = 1.0 # 초기 포트폴리오 가치는 1로 정규화

    def reset(self):
        # 환경 초기화
        self.current_step = self.lookback_window
        self.portfolio_value = 1.0
        # 사용자 프로필 기반 초기 가중치로 설정
        self.weights = self.initial_weights
        return self._get_state()

    def _get_state(self):
        # 현재 시점의 상태를 반환
        start = self.current_step - self.lookback_window
        end = self.current_step
        
        # 시장 상태: 수익률과 변동성 데이터를 lookback_window만큼 잘라냄
        market_state_cols = [f"{asset}_{feature}" for asset in self.assets for feature in ['log_return', 'vol_20d']]
        market_state = self.df[market_state_cols].iloc[start:end].values.flatten()
        
        # 에이전트 상태: 현재 포트폴리오 가중치
        agent_state = self.weights
        
        return np.concatenate((market_state, agent_state))

    def step(self, action):
        # 행동(action)은 Actor 신경망에서 나온 raw_action
        # Softmax는 Actor 모델 자체에서 처리되므로, 여기서는 합이 1이라고 가정
        target_weights = action

        # 이전 스텝의 가중치 (가격 변동으로 재조정된 후)
        prev_weights_rebalanced = self._get_rebalanced_weights()
        
        # 거래 비용 계산
        turnover = np.sum(np.abs(target_weights - prev_weights_rebalanced))
        transaction_costs = turnover * self.transaction_cost_pct * self.portfolio_value
        
        # 포트폴리오 가치 업데이트
        self.portfolio_value -= transaction_costs
        
        # 다음 스텝의 수익률 계산
        returns_at_next_step = self.df[[f"{asset}_log_return" for asset in self.assets]].iloc[self.current_step]
        # 현금 수익률은 0으로 가정
        asset_returns = np.append(returns_at_next_step.values, 0) 
        
        # 포트폴리오 로그 수익률
        portfolio_log_return = np.dot(target_weights, asset_returns)

        # 포트폴리오 위험(분산) 계산
        # 과거 lookback_window 기간 동안의 포트폴리오 수익률 분산
        historical_returns = self.df[[f"{asset}_log_return" for asset in self.assets]].iloc[self.current_step-self.lookback_window:self.current_step]
        portfolio_historical_returns = historical_returns.dot(target_weights[:-1]) # 현금 제외
        portfolio_variance = portfolio_historical_returns.var()
        
        # 포트폴리오 가치 최종 업데이트
        self.portfolio_value *= np.exp(portfolio_log_return)
        
        # 최종 보상(Reward) 계산
        # reward = 수익 - (위험 페널티) - (거래비용 페널티)
        reward = portfolio_log_return \
                 - self.risk_aversion_coeff * portfolio_variance \
                 - self.transaction_cost_coeff * turnover

        # 다음 상태로 이동
        self.current_step += 1
        self.weights = target_weights
        done = self.current_step >= len(self.df) - 1
        
        next_state = self._get_state()
        
        return next_state, reward, done, {'portfolio_value': self.portfolio_value, 'turnover': turnover}

    def _get_rebalanced_weights(self):
        # 이전 스텝의 자산 가격 변동을 반영하여 현재 가중치를 재계산
        prev_returns = self.df[[f"{asset}_log_return" for asset in self.assets]].iloc[self.current_step - 1]
        asset_returns = np.exp(np.append(prev_returns.values, 0)) # 현금 수익률 1 (exp(0))
        
        rebalanced_values = self.weights * asset_returns
        rebalanced_weights = rebalanced_values / np.sum(rebalanced_values)
        return rebalanced_weights