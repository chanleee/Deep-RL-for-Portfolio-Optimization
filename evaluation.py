# evaluation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Matplotlib 스타일 설정
plt.style.use('seaborn-v0_8-darkgrid')

def run_backtest(env, agent, use_exploration=False):
    """
    훈련된 에이전트를 사용하여 백테스트를 실행하고 시계열 결과를 반환합니다.

    Args:
        env (gym.Env): 포트폴리오 환경 객체.
        agent (Agent): 훈련된 DDPG 에이전트.
        use_exploration (bool): 탐색 노이즈 사용 여부.

    Returns:
        pd.DataFrame: 날짜별 포트폴리오 가치 및 자산별 가중치를 포함하는 데이터프레임.
    """
    state = env.reset()
    done = False
    
    # 결과를 저장할 리스트 초기화
    portfolio_values = [env.initial_portfolio_value]
    portfolio_weights = [env.weights]
    dates = [env.df.index[env.current_step]] # 날짜 추적

    while not done:
        action = agent.act(state, use_exploration=use_exploration)
        next_state, reward, done, info = env.step(action)
        
        portfolio_values.append(info.get('portfolio_value', env.portfolio_value))
        portfolio_weights.append(env.weights)
        
        # 마지막 스텝에서는 날짜 인덱스를 넘지 않도록 처리
        if not done:
            dates.append(env.df.index[env.current_step])
        else:
            dates.append(env.df.index[env.current_step - 1])
            
        state = next_state
        
    # 마지막 날짜가 중복될 수 있으므로, portfolio_values 길이에 맞춰 dates를 슬라이싱
    final_dates = dates[:len(portfolio_values)]

    return pd.DataFrame({
        'portfolio_value': portfolio_values,
        'weights': portfolio_weights
    }, index=final_dates)

def calculate_performance_metrics(portfolio_value_series):
    """
    포트폴리오 가치 시계열로부터 주요 성과 지표를 계산합니다.
    
    Args:
        portfolio_value_series (pd.Series): 포트폴리오 가치의 시계열 데이터.

    Returns:
        dict: 계산된 성과 지표 딕셔너리.
    """
    if portfolio_value_series.empty:
        return {
            "CAGR": 0, "Annualized Volatility": 0,
            "Sharpe Ratio": 0, "Max Drawdown (MDD)": 0
        }
    
    returns = portfolio_value_series.pct_change().dropna()
    
    # 연평균 복리 수익률 (CAGR)
    cagr = (portfolio_value_series.iloc[-1] / portfolio_value_series.iloc[0]) ** (252 / len(portfolio_value_series)) - 1
    
    # 연율화 변동성
    annualized_volatility = returns.std() * np.sqrt(252)
    
    # 샤프 지수 (무위험 수익률 0 가정)
    sharpe_ratio = cagr / annualized_volatility if annualized_volatility != 0 else 0
    
    # 최대 낙폭 (MDD)
    cumulative_max = portfolio_value_series.cummax()
    drawdown = (portfolio_value_series - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    
    return {
        "CAGR": cagr,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown (MDD)": max_drawdown
    }

def plot_evaluation_results(rl_results, benchmark_results, assets):
    """
    백테스트 결과를 시각화합니다. (누적수익률, 자산배분)

    Args:
        rl_results (pd.DataFrame): RL 에이전트의 백테스트 결과.
        benchmark_results (pd.DataFrame): 벤치마크의 백테스트 결과.
        assets (list): 자산 티커 리스트.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # --- 1. 누적 수익률 그래프 ---
    ax1.set_title('Cumulative Portfolio Value (Log Scale)', fontsize=14)
    
    # RL 에이전트 성과
    rl_normalized = rl_results['portfolio_value'] / rl_results['portfolio_value'].iloc[0]
    ax1.plot(rl_normalized.index, rl_normalized, label="DDPG Agent", color='royalblue', linewidth=2)

    # 벤치마크 성과
    benchmark_normalized = benchmark_results['portfolio_value'] / benchmark_results['portfolio_value'].iloc[0]
    ax1.plot(benchmark_normalized.index, benchmark_normalized, label="Equal Weight Benchmark", color='grey', linestyle='--')
    
    ax1.set_ylabel("Normalized Value")
    ax1.set_yscale('log') # 장기적인 성과 비교를 위해 로그 스케일 사용
    ax1.legend()
    ax1.grid(True, which="both", ls="--", linewidth=0.5)

    # --- 2. 자산 배분 그래프 ---
    ax2.set_title('Asset Allocation Over Time', fontsize=14)
    weights_df = pd.DataFrame(rl_results['weights'].tolist(), index=rl_results.index, columns=assets + ['Cash'])
    
    # 컬러맵 설정
    colors = plt.cm.viridis(np.linspace(0, 1, len(assets) + 1))
    ax2.stackplot(weights_df.index, weights_df.T, labels=weights_df.columns, colors=colors, alpha=0.8)

    ax2.set_ylabel("Portfolio Weights")
    ax2.set_xlabel("Date")
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(assets)//2 + 1, fancybox=True, shadow=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # 레전드 공간 확보
    plt.show()

def evaluate_agent(env, agent):
    """
    에이전트의 성능을 종합적으로 평가하고 지표와 그래프를 출력합니다.

    Args:
        env (gym.Env): 평가에 사용할 환경.
        agent (Agent): 평가할 에이전트.

    Returns:
        dict: RL 에이전트의 평균 성과 지표.
    """
    print("--- Running Backtest for DDPG Agent ---")
    rl_backtest_results = run_backtest(env, agent, use_exploration=False)
    rl_metrics = calculate_performance_metrics(rl_backtest_results['portfolio_value'])
    
    print("\n--- Running Backtest for Equal Weight Benchmark ---")
    # 벤치마크(동일 비중) 실행
    # 간단하게 매일 동일 비중으로 리밸런싱한다고 가정
    n_assets = env.n_assets
    equal_weights = np.array([1 / n_assets] * n_assets + [0]) # 현금 제외
    
    # 벤치마크 환경은 액션을 무시하고 동일 비중을 사용하도록 모킹
    benchmark_env = env
    benchmark_env.reset()
    done = False
    benchmark_values = [benchmark_env.initial_portfolio_value]
    
    while not done:
        # 벤치마크는 에이전트의 action 대신 동일 비중 사용
        _, _, done, info = benchmark_env.step(equal_weights)
        benchmark_values.append(info.get('portfolio_value'))

    benchmark_df = pd.DataFrame({'portfolio_value': benchmark_values}, index=rl_backtest_results.index[:len(benchmark_values)])
    benchmark_metrics = calculate_performance_metrics(benchmark_df['portfolio_value'])

    # --- 결과 출력 ---
    print("\n" + "="*50)
    print(" " * 15 + "PERFORMANCE SUMMARY")
    print("="*50)
    summary_df = pd.DataFrame({'DDPG Agent': rl_metrics, 'Benchmark': benchmark_metrics}).T
    summary_df['CAGR'] = summary_df['CAGR'].apply(lambda x: f"{x:.2%}")
    summary_df['Annualized Volatility'] = summary_df['Annualized Volatility'].apply(lambda x: f"{x:.2%}")
    summary_df['Sharpe Ratio'] = summary_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
    summary_df['Max Drawdown (MDD)'] = summary_df['Max Drawdown (MDD)'].apply(lambda x: f"{x:.2%}")
    print(summary_df)
    print("="*50 + "\n")
    
    # --- 시각화 ---
    plot_evaluation_results(rl_backtest_results, benchmark_df, env.assets)
    
    return rl_metrics, rl_backtest_results