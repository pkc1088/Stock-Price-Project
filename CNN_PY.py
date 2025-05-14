import io
import os
import yfinance as yf
import time
import pandas as pd
import mplfinance as mpf
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import timedelta
from tensorflow.keras.preprocessing import image # type: ignore


# model = tf.keras.models.load_model('pattern_classification_model.h5')

# def cnn_model(target_date_str, ticker, full_df=None):
#     """최적화된 CNN 모델 - 외부 데이터프레임 사용"""
#     try:
#         target_date = pd.to_datetime(target_date_str)
#         N_CANDLES = 20
        
#         if full_df is None:
#             return None, None
            
#         # 대상 기간 데이터 추출
#         df = full_df[['Open','High','Low','Close']].copy()
#         available_dates = df.index[df.index <= target_date]
        
#         if len(available_dates) == 0:
#             return None, None
            
#         actual_date = available_dates[-1]
#         end_idx = df.index.get_loc(actual_date)
        
#         if end_idx < N_CANDLES:
#             return None, None
            
#         # 차트 생성
#         df_slice = df.iloc[end_idx-N_CANDLES:end_idx+1].copy()
#         buf = io.BytesIO()
        
#         mpf.plot(df_slice, type='candle', style='charles', 
#                 axisoff=True, savefig=buf, closefig=True)
#         buf.seek(0)
        
#         # 이미지 처리
#         img = Image.open(buf).resize((224,224))
#         img_array = tf.keras.preprocessing.image.img_to_array(img)/255.0
#         pred = model.predict(np.expand_dims(img_array, axis=0))[0]
        
#         label = ['drop','neutral','surge'][np.argmax(pred)]
#         prob = np.max(pred)
        
#         buf.close()
#         return label, prob
        
#     except Exception as e:
#         print(f"CNN 예측 오류 ({ticker}): {str(e)}")
#         return None, None



# ORIGINAL CODE
model = tf.keras.models.load_model('pattern_classification_model.h5')

def cnn_model(target_date_str, ticker):
    target_date = pd.to_datetime(target_date_str)
    N_CANDLES = 20

    try:
        # print(f"\n{ticker} 데이터 다운로드 중...")
        time.sleep(1)
        df = yf.download(
            ticker,
            start=target_date - timedelta(days=40),
            end=target_date + timedelta(days=2),
            group_by='column'
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        if df.empty:
            print(f"{ticker}: 데이터 없음")
        else:
            required_cols = ["Open", "High", "Low", "Close"]
            if not all(col in df.columns for col in required_cols):
                print(f"{ticker}: OHLC 컬럼 누락")
            else:
                available_dates = df.index[df.index <= target_date]
                if len(available_dates) == 0:
                    print(f"{ticker}: {target_date.date()} 이전 거래일 없음")
                else:
                    actual_date = available_dates[-1]
                    end_idx = df.index.get_loc(actual_date)

                    if end_idx < N_CANDLES:
                        print(f"{ticker}: 20봉 이상 데이터 부족")
                    else:
                        df_slice = df.iloc[end_idx - N_CANDLES:end_idx].copy()
                        df_slice.index = pd.to_datetime(df_slice.index)
                        df_slice = df_slice[required_cols].copy().astype("float64").dropna()

                        if len(df_slice) < N_CANDLES:
                            print(f"{ticker}: 정제 후 유효한 캔들 수 부족")
                        else:
                            mc = mpf.make_marketcolors(up='g', down='r', edge='black', wick='black', volume='gray')
                            s = mpf.make_mpf_style(marketcolors=mc, rc={'axes.grid': False})

                            # 메모리 버퍼 사용
                            buf = io.BytesIO()
                            mpf.plot(
                                df_slice,
                                type='candle',
                                style=s,
                                volume=False,
                                axisoff=True,
                                tight_layout=True,
                                savefig=buf  # 직접 메모리로 저장
                            )
                            buf.seek(0)  # 버퍼의 시작 위치로 이동

                            # 이미지 처리
                            img = Image.open(buf).convert("RGB").resize((224, 224))

                            # 예측 수행
                            predicted_label, prob = predict_image(img)
                            # print(f"예측된 레이블: {predicted_label}, 확률: {prob:.4f}")

                            buf.close()
                            return predicted_label, prob

    except Exception as e:
        print(f"{ticker} 처리 중 오류: {e}")
        return None, None


def predict_image(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    # print(f"예측 확률: {prediction[0]}")
    
    predicted_class = np.argmax(prediction)
    predicted_label = ["drop", "neutral", "surge"][predicted_class]
    prob = prediction[0][predicted_class]

    # print(f"예측 확률: {prob:.4f}, 예측 레이블: {predicted_label}")

    return predicted_label, prob






'''
import os
import yfinance as yf
import time
import pandas as pd
import mplfinance as mpf
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import timedelta
from tensorflow.keras.preprocessing import image # type: ignore

# 모델 불러오기
model = tf.keras.models.load_model('pattern_classification_model.h5')

def cnn_model(target_date_str, ticker):
    # 기준 날짜를 입력으로 받음
    # target_date_str = input("기준 날짜 입력 (YYYY-MM-DD): ")
    target_date = pd.to_datetime(target_date_str)

    # 저장 경로 설정
    # base_dir = "./data/test"
    # os.makedirs(base_dir, exist_ok=True)

    N_CANDLES = 20
    # saved_count = 0

    # 종목 티커를 입력받기
    # ticker = input("종목 티커 입력: ")

    try:
        print(f"\n{ticker} 데이터 다운로드 중...")
        time.sleep(1)
        # 종목에 대한 데이터 다운로드
        df = yf.download(
            ticker,
            start=target_date - timedelta(days=40),  # 40일의 데이터를 가져옴
            end=target_date + timedelta(days=2),     # target_date 포함
            group_by='column'
        )
        # MultiIndex 컬럼 평탄화
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        if df.empty:
            print(f"{ticker}: 데이터 없음")
        else:
            required_cols = ["Open", "High", "Low", "Close"]
            if not all(col in df.columns for col in required_cols):
                print(f"{ticker}: OHLC 컬럼 누락")
            else:
                available_dates = df.index[df.index <= target_date]
                if len(available_dates) == 0:
                    print(f"{ticker}: {target_date.date()} 이전 거래일 없음")
                else:
                    actual_date = available_dates[-1]
                    end_idx = df.index.get_loc(actual_date)

                    if end_idx < N_CANDLES:
                        print(f"{ticker}: 20봉 이상 데이터 부족")
                    else:
                        df_slice = df.iloc[end_idx - N_CANDLES:end_idx].copy()

                        # 강제 정제: 인덱스 타입, 컬럼 선택, float64 변환, NaN 제거
                        df_slice.index = pd.to_datetime(df_slice.index)
                        df_slice = df_slice[required_cols].copy()
                        df_slice = df_slice.astype("float64")
                        df_slice.dropna(inplace=True)

                        if len(df_slice) < N_CANDLES:
                            print(f"{ticker}: 정제 후 유효한 캔들 수 부족")
                        else:
                            close_today = df.at[df.index[end_idx], "Close"]
                            close_yesterday = df.at[df.index[end_idx - 1], "Close"]
                            pct_change = ((close_today - close_yesterday) / close_yesterday) * 100

                            filename = f"{ticker}_{actual_date.date()}.png"
                            # out_path = os.path.join(base_dir, filename)

                            # 차트 스타일 설정
                            mc = mpf.make_marketcolors(up='g', down='r', edge='black', wick='black', volume='gray')
                            s = mpf.make_mpf_style(marketcolors=mc, rc={'axes.grid': False})

                            # 차트 저장
                            mpf.plot(
                                df_slice,
                                type='candle',
                                style=s,
                                volume=False,
                                axisoff=True,
                                tight_layout=True,
                                # savefig=out_path
                            )

                            # 이미지 크기 조정
                            img = Image.open(out_path)
                            img = img.resize((224, 224))
                            img.save(out_path)

                            # print(f"저장 완료: {out_path}")
                            # saved_count += 1

                            # 예측 수행
                            predicted_label, prob = predict_image(out_path)  # 저장된 이미지에 대해 예측 수행
                            print(f"예측된 레이블: {predicted_label}, 확률: {prob:.4f}")
    except Exception as e:
        print(f"{ticker} 처리 중 오류: {e}")
    

# 예측 함수 (가장 큰 확률값을 가진 클래스를 선택)
def predict_image(img_src):
    img = img_src
    # image.load_img(img_path, target_size=(224, 224))  # 모델 입력 크기에 맞춰 이미지 크기 변경
    img_array = image.img_to_array(img)  # 이미지를 배열로 변환
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    img_array /= 255.0  # 정규화

    # 예측 수행
    prediction = model.predict(img_array)

    # 각 클래스에 대한 확률을 출력
    print(f"예측 확률: {prediction[0]}")  # 각 클래스에 대한 확률 (Drop, Neutral, Surge)
    
    # 가장 큰 확률을 가진 클래스를 선택 (softmax 출력에서 가장 큰 값 선택)
    predicted_class = np.argmax(prediction)  # 가장 큰 확률을 가진 클래스 인덱스
    predicted_label = ["drop", "neutral", "surge"][predicted_class]  # 클래스 레이블 지정
    prob = prediction[0][predicted_class]  # 해당 클래스의 확률

    # 예측 결과 출력
    print(f"예측 확률: {prob:.4f}, 예측 레이블: {predicted_label}")

    return predicted_label, prob

'''