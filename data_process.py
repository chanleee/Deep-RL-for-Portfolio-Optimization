import yfinance as yf
import pandas as pd
import numpy as np

def get_market_data(tickers, start_date, end_date):
    """
    yfinance를 통해 지정된 티커의 시장 데이터를 다운로드하고 전처리합니다.

    Args:
        tickers (list): 티커 심볼 리스트.
        start_date (str): 데이터 시작일 (YYYY-MM-DD).
        end_date (str): 데이터 종료일 (YYYY-MM-DD).

    Returns:
        pandas.DataFrame: 전처리된 시장 데이터.
    """
    # 수정 종가(Adj Close) 데이터 다운로드
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

    # 데이터 클리닝: 누락된 값(NaN)을 이전 값으로 채우기 (forward-fill)
    data.fillna(method='ffill', inplace=True)
    # 그래도 남은 초기 NaN 값은 이후 값으로 채우기 (backward-fill)
    data.fillna(method='bfill', inplace=True)

    # 로그 수익률 계산 (기하 평균을 산술 평균으로 다루기 위함)
    log_returns = np.log(data / data.shift(1))

    # 피처 엔지니어링: 20일 이동 변동성 계산
    volatility = log_returns.rolling(window=20).std() * np.sqrt(252) # 연율화

    # 데이터 병합
    # 컬럼 이름 재정의
    adj_close_cols = {ticker: f"{ticker}_Adj_Close" for ticker in tickers}
    log_return_cols = {ticker: f"{ticker}_log_return" for ticker in tickers}
    volatility_cols = {ticker: f"{ticker}_vol_20d" for ticker in tickers}

    processed_data = pd.concat([
        data.rename(columns=adj_close_cols),
        log_returns.rename(columns=log_return_cols),
        volatility.rename(columns=volatility_cols)
    ], axis=1)

    # 초기 데이터 생성 기간으로 인한 NaN 값 제거
    processed_data.dropna(inplace=True)

    return processed_data

# --- 실행 예시 ---
tickers = ["AAPL", "MSFT", "AMZN", "GOOG", "SPY", "BIL", "TLT"]

market_data = get_market_data(
    tickers=tickers,
    start_date="2010-01-01",
    end_date="2023-12-31"
)
print(market_data.head())