import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import joblib
import ta
import os
import tensorflow as tf
from datetime import timedelta


# def lstm_model(target_date_str, ticker, full_df=None):
#     """최적화된 LSTM 모델 - 외부 데이터프레임 사용"""
#     try:
#         target_date = pd.to_datetime(target_date_str)
        
#         # 모델 로드
#         model_path = f'LSTM_MODEL_H5_WIN/{ticker}.h5'
#         scaler_path = f'LSTM_MODEL_H5_WIN/{ticker}_scaler.joblib'
        
#         if not all(os.path.exists(p) for p in [model_path, scaler_path]):
#             return None
            
#         model = tf.keras.models.load_model(model_path, compile=False)
#         scaler = joblib.load(scaler_path)
        
#         # 데이터 처리
#         if full_df is None:
#             return None
            
#         df = full_df[['Open','High','Low','Close','Volume']].copy()
#         available_dates = df.index[df.index <= target_date]
        
#         if len(available_dates) == 0:
#             return None
            
#         actual_date = available_dates[-1]
#         end_idx = df.index.get_loc(actual_date)
        
#         if end_idx < 49:
#             return None
            
#         # 기술적 지표 추가
#         df = df.copy()
#         # 컬럼 처리 (간혹 멀티인덱스일 수 있으므로 처리)
#         if isinstance(df.columns, pd.MultiIndex):
#             df.columns = df.columns.droplevel(1)
#         df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

#         df['MA20'] = ta.trend.sma_indicator(df['Close'], window=20)
#         bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
#         df['Upper'] = bb.bollinger_hband()
#         df['Lower'] = bb.bollinger_lband()
#         df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
#         df.dropna(inplace=True)
        
#         # 특징 추출
#         features = ['Open','High','Low','Close','Volume','MA20','Upper','Lower','RSI']
#         X_scaled = scaler.transform(df.iloc[end_idx-49:end_idx+1][features])
        
#         # 예측
#         pred = model.predict(np.expand_dims(X_scaled, axis=0))
#         dummy = np.zeros((1, len(features)))
#         dummy[:, features.index('Close')] = pred.flatten()
#         pred_close = scaler.inverse_transform(dummy)[0, features.index('Close')]
        
#         return float(pred_close)
        
#     except Exception as e:
#         print(f"LSTM 예측 오류 ({ticker}): {str(e)}")
#         return None
    



# ORIGINAL CODE
def lstm_model(target_date, ticker):
    try:
        # 문자열 -> Timestamp
        target_date = pd.to_datetime(target_date)

        # 1. 모델과 스케일러 동적 로딩
        model_path = os.path.join('LSTM_MODEL_H5_WIN', f'{ticker}.h5')
        scaler_path = os.path.join('LSTM_MODEL_H5_WIN', f'{ticker}_scaler.joblib')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"{ticker}의 모델 또는 스케일러 파일이 없습니다.")
            return None

        model = tf.keras.models.load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)

        # 2. 데이터 준비 (target_date 기준으로 과거 100일 확보)
        df = yf.download(ticker, start=target_date - timedelta(days=150), end=target_date + timedelta(days=1), interval='1d')

        if df.empty:
            print(f"{ticker}: 데이터 없음")
            return None

        # 컬럼 처리 (간혹 멀티인덱스일 수 있으므로 처리)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

        # 3. 기술적 지표 추가
        df['MA20'] = ta.trend.sma_indicator(df['Close'], window=20)
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['Upper'] = bb.bollinger_hband()
        df['Lower'] = bb.bollinger_lband()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df.dropna(inplace=True)

        # 4. target_date 기준 유효한 가장 가까운 거래일 찾기
        available_dates = df.index[df.index <= target_date]
        if len(available_dates) == 0:
            print(f"{ticker}: {target_date.date()} 이전 거래일 없음")
            return None

        actual_date = available_dates[-1]
        end_idx = df.index.get_loc(actual_date)

        if end_idx < 49:
            print(f"{ticker}: {actual_date.date()} 기준 50일 이상 데이터 부족")
            return None

        # 5. 특징 추출 및 스케일링
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'Upper', 'Lower', 'RSI']
        df_slice = df.iloc[end_idx - 49:end_idx + 1]  # 50일 확보
        X_scaled = scaler.transform(df_slice[features])

        # 6. 예측 수행
        X_input = np.expand_dims(X_scaled, axis=0)
        pred = model.predict(X_input)

        # 7. 역변환
        dummy = np.zeros((1, len(features)))
        close_idx = features.index('Close')
        dummy[:, close_idx] = pred.flatten()
        pred_close = scaler.inverse_transform(dummy)[0, close_idx]

        # print(f"예상 다음날 종가 ({ticker} 기준 {actual_date.date()}): ${pred_close:.2f}")
        return pred_close

    except Exception as e:
        print(f"{ticker} 처리 중 오류: {e}")
        return None



'''
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import keras
import joblib
import ta
import os
import tensorflow as tf

def cnn_model(target_date, ticker):
    # 1. 모델과 스케일러 불러오기
    model = tf.keras.models.load_model(os.path.join('LSTM_MODEL_H5_WIN', 'AOS.h5'), compile=False)
    scaler = joblib.load('LSTM_MODEL_H5_WIN/AOS_scaler.joblib')

    # 2. 데이터 준비
    symbol = 'AOS'
    df = yf.download(symbol, period='100d', interval='1d')
    df.columns = df.columns.droplevel(1)

    # 3. 기술적 지표 추가 (학습 당시 동일하게)
    df['MA20'] = ta.trend.sma_indicator(df['Close'], window=20)
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['Upper'] = bb.bollinger_hband()
    df['Lower'] = bb.bollinger_lband()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df.dropna(inplace=True)  # 지표 계산으로 인한 NaN 제거

    # 4. 특징 추출 및 스케일링
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'Upper', 'Lower', 'RSI']
    X_recent = df[features].tail(50)  # 최근 50일
    X_scaled = scaler.transform(X_recent)  # 주의: fit_transform 금지

    # 5. 예측 수행
    X_input = np.expand_dims(X_scaled, axis=0)
    pred = model.predict(X_input)  # compile=False여도 predict()는 가능

    # 6. 역변환
    dummy = np.zeros((1, len(features)))
    close_idx = features.index('Close')
    dummy[:, close_idx] = pred.flatten()
    pred_close = scaler.inverse_transform(dummy)[0, close_idx]

    print(f"예상 다음날 종가 ({symbol}): ${pred_close:.2f}")

    return pred_close

'''
