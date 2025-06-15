# data_process.py

import yfinance as yf
import pandas as pd
import numpy as np
import talib # RSI 계산을 위해 TA-Lib 임포트

def get_market_data(tickers, start_date, end_date):
    """
    yfinance를 통해 지정된 티커의 시장 데이터를 다운로드하고, 기술적 지표를 포함하여 전처리합니다.

    Args:
        tickers (list): 티커 심볼 리스트.
        start_date (str): 데이터 시작일 (YYYY-MM-DD).
        end_date (str): 데이터 종료일 (YYYY-MM-DD).

    Returns:
        pandas.DataFrame: 전처리된 시장 데이터.
    """
    # yfinance로부터 'Close'와 'Adj Close' 데이터를 모두 다운로드
    # MA, RSI는 'Close' 기준으로, 수익률은 'Adj Close' 기준으로 계산하기 위함
    
    # 1. 수익률 계산을 위한 수정주가 데이터 다운로드
    # auto_adjust=True로 설정하면 'Close' 컬럼에 수정주가가 담겨 나옵니다.
    adj_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    adj_close_data = adj_data['Close']

    # 2. 기술적 지표 계산을 위한 수정되지 않은 원본 데이터 다운로드
    raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    close_data = raw_data['Close']
    
    # 데이터 클리닝: 누락된 값(NaN) 채우기
    adj_close_data.fillna(method='ffill', inplace=True)
    adj_close_data.fillna(method='bfill', inplace=True)
    close_data.fillna(method='ffill', inplace=True)
    close_data.fillna(method='bfill', inplace=True)

    # --- 기존 피처 계산 ---
    # 로그 수익률 ('Adj Close' 기반)
    log_returns = np.log(adj_close_data / adj_close_data.shift(1))
    # 20일 이동 변동성 ('Adj Close' 기반)
    volatility = log_returns.rolling(window=20).std() * np.sqrt(252)

    # --- 신규 피처 계산 ('Close' 기반) ---
    ma_5 = {}
    ma_20 = {}
    ma_60 = {}
    rsi_14 = {}

    for ticker in tickers:
        # 이동 평균 (Moving Averages)
        ma_5[f"{ticker}_ma_5d"] = close_data[ticker].rolling(window=5).mean()
        ma_20[f"{ticker}_ma_20d"] = close_data[ticker].rolling(window=20).mean()
        ma_60[f"{ticker}_ma_60d"] = close_data[ticker].rolling(window=60).mean()
        # 상대강도지수 (RSI)
        rsi_14[f"{ticker}_rsi_14d"] = talib.RSI(close_data[ticker], timeperiod=14)

    # 데이터프레임으로 변환
    ma_5_df = pd.DataFrame(ma_5)
    ma_20_df = pd.DataFrame(ma_20)
    ma_60_df = pd.DataFrame(ma_60)
    rsi_14_df = pd.DataFrame(rsi_14)

    # 데이터 병합
    processed_data = pd.concat([
        adj_close_data.rename(columns={ticker: f"{ticker}_Adj_Close" for ticker in tickers}),
        log_returns.rename(columns={ticker: f"{ticker}_log_return" for ticker in tickers}),
        volatility.rename(columns={ticker: f"{ticker}_vol_20d" for ticker in tickers}),
        ma_5_df,
        ma_20_df,
        ma_60_df,
        rsi_14_df
    ], axis=1)

    # 초기 데이터 생성 기간으로 인한 NaN 값 제거
    processed_data.dropna(inplace=True)

    return processed_data

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "AMZN", "GOOG", "SPY", "BIL", "TLT"]

    market_data = get_market_data(
        tickers=tickers,
        start_date="2010-01-01",
        end_date="2023-12-31"
    )
    print("--- Processed Data with New Features ---")
    print(market_data.head())
    print("\nColumns:")
    print(market_data.columns)