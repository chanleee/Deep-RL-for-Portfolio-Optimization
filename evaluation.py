# evaluation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Matplotlib 스타일 설정
plt.style.use('seaborn-v0_8-darkgrid')

def run_backtest(env, agent):
    state = env.reset()
    done = False
    
    portfolio_values = [env.initial_portfolio_value]
    portfolio_weights = [env.weights]
    # 날짜 추적을 위해 df.index가 존재하는지 확인
    dates = [env.df.index[env.current_step]] if hasattr(env.df, 'index') else []

    while not done:
        # --- 수정된 부분: act 함수 호출 방식을 새로운 TD3 agent에 맞게 변경 ---
        # use_exploration 인자 대신 exploration_std=0.0으로 고정하여 노이즈 비활성화
        action = agent.act(state, exploration_std=0.0)
        
        next_state, reward, done, info = env.step(action)
        
        portfolio_values.append(info.get('portfolio_value', env.portfolio_value))
        portfolio_weights.append(env.weights)
        
        if hasattr(env.df, 'index'):
            if not done:
                dates.append(env.df.index[env.current_step])
            else:
                # 마지막 step에서는 index가 범위를 벗어날 수 있으므로 조정
                dates.append(env.df.index[min(env.current_step, len(env.df.index)-1)])
            
        state = next_state
        
    final_dates = dates[:len(portfolio_values)] if dates else pd.RangeIndex(len(portfolio_values))

    return pd.DataFrame({
        'portfolio_value': portfolio_values,
        'weights': portfolio_weights
    }, index=final_dates)

def calculate_performance_metrics(portfolio_value_series):
    if portfolio_value_series.empty or len(portfolio_value_series) < 2:
        return {"CAGR": 0, "Annualized Volatility": 0, "Sharpe Ratio": 0, "Max Drawdown (MDD)": 0}
    returns = portfolio_value_series.pct_change().dropna()
    if returns.empty:
        return {"CAGR": 0, "Annualized Volatility": 0, "Sharpe Ratio": 0, "Max Drawdown (MDD)": 0}
    cagr = (portfolio_value_series.iloc[-1] / portfolio_value_series.iloc[0]) ** (252 / len(portfolio_value_series)) - 1
    annualized_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = cagr / annualized_volatility if annualized_volatility > 1e-9 else 0
    cumulative_max = portfolio_value_series.cummax()
    drawdown = (portfolio_value_series - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    return {"CAGR": cagr, "Annualized Volatility": annualized_volatility, "Sharpe Ratio": sharpe_ratio, "Max Drawdown (MDD)": max_drawdown}

def evaluate_for_validation(env, agent):
    """
    조기 종료 검증을 위한 경량화된 평가 함수.
    백테스트를 실행하고 샤프 지수만 반환합니다.
    """
    # 수정된 run_backtest 함수 호출
    backtest_results = run_backtest(env, agent)
    metrics = calculate_performance_metrics(backtest_results['portfolio_value'])
    return metrics['Sharpe Ratio']

def plot_evaluation_results(rl_results, benchmark_results, assets):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    ax1.set_title('Cumulative Portfolio Value (Log Scale)', fontsize=14)
    rl_normalized = rl_results['portfolio_value'] / rl_results['portfolio_value'].iloc[0]
    ax1.plot(rl_normalized.index, rl_normalized, label="TD3 Agent", color='royalblue', linewidth=2)
    benchmark_normalized = benchmark_results['portfolio_value'] / benchmark_results['portfolio_value'].iloc[0]
    ax1.plot(benchmark_normalized.index, benchmark_normalized, label="Equal Weight Benchmark", color='grey', linestyle='--')
    ax1.set_ylabel("Normalized Value")
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", linewidth=0.5)
    ax2.set_title('Asset Allocation Over Time', fontsize=14)
    weights_df = pd.DataFrame(rl_results['weights'].tolist(), index=rl_results.index, columns=assets + ['Cash'])
    colors = plt.cm.viridis(np.linspace(0, 1, len(assets) + 1))
    ax2.stackplot(weights_df.index, weights_df.T, labels=weights_df.columns, colors=colors, alpha=0.8)
    ax2.set_ylabel("Portfolio Weights")
    ax2.set_xlabel("Date")
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(assets)//2 + 1, fancybox=True, shadow=True)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

def evaluate_agent(env, agent):
    """
    에이전트의 성능을 종합적으로 평가하고 지표와 그래프를 출력합니다.
    """
    print("--- Running Backtest for TD3 Agent ---") # DDPG -> TD3로 변경
    # 수정된 run_backtest 함수 호출
    rl_backtest_results = run_backtest(env, agent)
    rl_metrics = calculate_performance_metrics(rl_backtest_results['portfolio_value'])
    
    print("\n--- Running Backtest for Equal Weight Benchmark ---")
    n_assets = env.n_assets
    # 동일 비중 계산 시 현금 제외
    equal_weights = np.array([1 / n_assets] * n_assets + [0.0])
    
    benchmark_env = env
    state = benchmark_env.reset() # reset 호출 후 state를 받아야 함
    done = False
    benchmark_values = [benchmark_env.initial_portfolio_value]
    
    while not done:
        _, _, done, info = benchmark_env.step(equal_weights)
        benchmark_values.append(info.get('portfolio_value'))

    benchmark_df = pd.DataFrame({'portfolio_value': benchmark_values}, index=rl_backtest_results.index[:len(benchmark_values)])
    benchmark_metrics = calculate_performance_metrics(benchmark_df['portfolio_value'])

    print("\n" + "="*50)
    print(" " * 15 + "PERFORMANCE SUMMARY")
    print("="*50)
    summary_df = pd.DataFrame({'TD3 Agent': rl_metrics, 'Benchmark': benchmark_metrics}).T # DDPG -> TD3로 변경
    for col in ['CAGR', 'Annualized Volatility', 'Max Drawdown (MDD)']:
        summary_df[col] = summary_df[col].apply(lambda x: f"{x:.2%}")
    summary_df['Sharpe Ratio'] = summary_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
    print(summary_df)
    print("="*50 + "\n")
    
    plot_evaluation_results(rl_backtest_results, benchmark_df, env.assets)
    
    return rl_metrics