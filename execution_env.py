# execution_env.py (수정된 코드)

import gym
from gym import spaces
import numpy as np

class DailyExecutionEnv(gym.Env):
    # --- 수정된 부분: __init__ 함수에 lookback_window 인자 추가 ---
    def __init__(self, df, assets, initial_portfolio_value, transaction_cost_pct, lookback_window=30, trade_size_pct=0.05):
        super(DailyExecutionEnv, self).__init__()

        self.df = df
        self.assets = assets
        self.n_assets = len(assets)
        self.transaction_cost_pct = transaction_cost_pct
        self.trade_size_pct = trade_size_pct
        # --- 수정된 부분: lookback_window를 인자로 받아 먼저 정의 ---
        self.lookback_window = lookback_window 

        # 상태 공간 정의 (이제 lookback_window가 정의되었으므로 오류 없음)
        n_market_features = 6 
        n_correlation_features = self.n_assets * (self.n_assets - 1) // 2
        # lower_state_dim 계산 시 시장 데이터 부분에 lookback_window를 사용
        market_state_dim = (self.lookback_window * self.n_assets * n_market_features) + n_correlation_features
        # 전체 상태 차원 = 시장 상태 + 현재 가중치 + 목표 가중치
        state_dim = market_state_dim + (self.n_assets + 1) * 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

        self.action_space = spaces.Discrete(1 + 2 * self.n_assets)
        self.initial_portfolio_value = initial_portfolio_value
        
    def set_target(self, initial_weights, target_weights):
        # (기존 코드와 동일)
        self.weights = initial_weights
        self.target_weights = target_weights
        self.portfolio_value = self.initial_portfolio_value

    def reset(self, start_step):
        # (기존 코드와 동일)
        self.current_step = start_step
        return self._get_state()

    def _get_state(self):
        # (기존 코드와 동일)
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

        return np.concatenate((market_state, correlation_state, self.weights, self.target_weights))

    def step(self, action_idx):
        # (step 함수 로직은 이전과 동일하므로 생략)
        trade_asset_idx = (action_idx - 1) // 2
        trade_type = (action_idx - 1) % 2 
        
        prev_tracking_error = np.sum(np.abs(self.weights - self.target_weights))
        transaction_cost = 0

        if action_idx > 0:
            trade_amount = self.portfolio_value * self.trade_size_pct
            transaction_cost = trade_amount * self.transaction_cost_pct
            self.portfolio_value -= transaction_cost

            if trade_type == 0:
                self.weights[trade_asset_idx] += self.trade_size_pct
                self.weights[-1] -= self.trade_size_pct
            else:
                self.weights[trade_asset_idx] -= self.trade_size_pct
                self.weights[-1] += self.trade_size_pct
            
            self.weights = np.clip(self.weights, 0, 1)
            self.weights /= np.sum(self.weights)

        daily_returns = self.df[[f"{asset}_log_return" for asset in self.assets]].iloc[self.current_step]
        asset_returns = np.append(daily_returns.values, 0)
        portfolio_log_return = np.dot(self.weights, asset_returns)
        self.portfolio_value *= np.exp(portfolio_log_return)

        current_tracking_error = np.sum(np.abs(self.weights - self.target_weights))
        reward = (prev_tracking_error - current_tracking_error) * 100
        reward -= transaction_cost * 0.1

        self.current_step += 1
        done = True
        next_state = self._get_state()

        return next_state, reward, done, {}