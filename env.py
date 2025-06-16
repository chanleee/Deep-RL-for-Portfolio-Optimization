import gym
from gym import spaces
import numpy as np
import pandas as pd

class MultiAssetPortfolioEnv(gym.Env):
    def __init__(self, df, assets, initial_weights, 
                 initial_portfolio_value=1.0, 
                 lookback_window=30, 
                 transaction_cost_pct=0.001, 
                 risk_aversion_coeff=0.05,
                 hhi_penalty_coeff=0.1): # HHI 패널티 계수 추가
        
        super(MultiAssetPortfolioEnv, self).__init__()

        self.df = df
        self.assets = assets
        self.n_assets = len(self.assets)
        self.lookback_window = lookback_window
        self.transaction_cost_pct = transaction_cost_pct
        self.risk_aversion_coeff = risk_aversion_coeff
        self.hhi_penalty_coeff = hhi_penalty_coeff # HHI 계수 초기화

        # (상태/행동 공간 정의 및 reset, _get_state 메서드는 이전과 동일)
        n_market_features = 6
        n_correlation_features = self.n_assets * (self.n_assets - 1) // 2
        n_agent_features = self.n_assets + 1
        state_shape = (self.lookback_window * self.n_assets * n_market_features) + n_correlation_features + n_agent_features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_shape,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32)
        
        self.initial_weights = np.array(initial_weights)
        self.initial_portfolio_value = initial_portfolio_value
        self.current_step = 0
        self.portfolio_value = self.initial_portfolio_value
        self.weights = self.initial_weights

    def reset(self):
        self.current_step = self.lookback_window
        self.portfolio_value = self.initial_portfolio_value
        self.weights = self.initial_weights
        return self._get_state()

    def _get_state(self):
        start = self.current_step - self.lookback_window
        end = self.current_step
        market_feature_cols = []
        for asset in self.assets:
            market_feature_cols.extend([f"{asset}_log_return", f"{asset}_vol_20d", f"{asset}_ma_5d", f"{asset}_ma_20d", f"{asset}_ma_60d", f"{asset}_rsi_14d"])
        market_state = self.df[market_feature_cols].iloc[start:end].values.flatten()
        correlation_df = self.df[[f"{asset}_log_return" for asset in self.assets]].iloc[start:end]
        correlation_matrix = correlation_df.corr().values
        triu_indices = np.triu_indices(self.n_assets, k=1)
        correlation_state = correlation_matrix[triu_indices].flatten()
        agent_state = self.weights
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
        
        start_idx = max(0, self.current_step - self.lookback_window)
        historical_returns = self.df[[f"{asset}_log_return" for asset in self.assets]].iloc[start_idx:self.current_step]
        historical_portfolio_returns = np.dot(historical_returns, target_weights[:-1])
        portfolio_volatility = np.std(historical_portfolio_returns) if len(historical_portfolio_returns) > 1 else 0
        
        # 기본 보상 (수익률 - 거래비용 - 변동성 패널티)
        base_reward = portfolio_log_return - (self.transaction_cost_pct * turnover) - (self.risk_aversion_coeff * portfolio_volatility)
        
        # HHI 계산 (현금 제외 자산들의 가중치 제곱의 합)
        hhi = np.sum(np.square(target_weights[:-1])) 
        # 집중도에 대한 패널티
        diversification_penalty = self.hhi_penalty_coeff * hhi
        
        # 최종 보상
        reward = base_reward - diversification_penalty
        
        self.portfolio_value *= np.exp(portfolio_log_return)
        self.current_step += 1
        self.weights = target_weights
        done = self.current_step >= len(self.df) - 1
        next_state = self._get_state() if not done else np.zeros(self.observation_space.shape)
        return next_state, reward, done, {'portfolio_value': self.portfolio_value, 'turnover': turnover}

    def _get_rebalanced_weights(self):
        prev_step_idx = max(0, self.current_step - 1)
        prev_returns = self.df[[f"{asset}_log_return" for asset in self.assets]].iloc[prev_step_idx]
        asset_returns = np.exp(np.append(prev_returns.values, 0))
        rebalanced_values = self.weights * asset_returns
        rebalanced_weights = rebalanced_values / np.sum(rebalanced_values)
        return rebalanced_weights