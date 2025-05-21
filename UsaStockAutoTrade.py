import requests
import json
import time
import yaml
import yfinance as yf

with open('config.yaml', encoding='UTF-8') as f:
    _cfg = yaml.load(f, Loader=yaml.FullLoader)
APP_KEY = _cfg['APP_KEY']
APP_SECRET = _cfg['APP_SECRET']
ACCESS_TOKEN = ""
CANO = _cfg['CANO']
ACNT_PRDT_CD = _cfg['ACNT_PRDT_CD']
URL_BASE = _cfg['URL_BASE']

def get_access_token():
    """토큰 발급"""
    headers = {"content-type":"application/json"}
    body = {
    "grant_type":"client_credentials",
    "appkey":APP_KEY, 
    "appsecret":APP_SECRET}
    PATH = "oauth2/tokenP"
    URL = f"{URL_BASE}/{PATH}"
    res = requests.post(URL, headers=headers, data=json.dumps(body))
    ACCESS_TOKEN = res.json()["access_token"]
    return ACCESS_TOKEN
    
def hashkey(datas):
    """암호화"""
    PATH = "uapi/hashkey"
    URL = f"{URL_BASE}/{PATH}"
    headers = {
    'content-Type' : 'application/json',
    'appKey' : APP_KEY,
    'appSecret' : APP_SECRET,
    }
    res = requests.post(URL, headers=headers, data=json.dumps(datas))
    hashkey = res.json()["HASH"]
    return hashkey

# 기업명 (string)
def get_long_name(code):
    return yf.Ticker(code).info['longName']

# struct Holding: Equatable, Codable {
#     var name: String
#     var symbol: String
#     var quantity: Int
#     var avgPrice: Double
#   }
def make_holding():
    """주식 잔고 조회"""
    PATH = "/uapi/overseas-stock/v1/trading/inquire-balance"
    URL = f"{URL_BASE}/{PATH}"
    headers = {"Content-Type":"application/json",
        "authorization":f"Bearer {ACCESS_TOKEN}",
        "appKey":APP_KEY,
        "appSecret":APP_SECRET,
        "tr_id":"VTTS3012R",
        "custtype":"P"
    }
    params = {
        "CANO": CANO,
        "ACNT_PRDT_CD": ACNT_PRDT_CD,
        "OVRS_EXCG_CD": "NASD",
        "TR_CRCY_CD": "USD",
        "CTX_AREA_FK200": "",
        "CTX_AREA_NK200": ""
    }
    res = requests.get(URL, headers=headers, params=params)
    stock_list = res.json()['output1']

    stock_dict = {}
    for stock in stock_list:
        if int(stock['ovrs_cblc_qty']) > 0:
            symbol = stock['ovrs_pdno']
            quantity = int(stock['ovrs_cblc_qty'])  # 보유량
            avgPrice = float(stock['pchs_avg_pric'])  # 평균 매입가

            stock_dict[symbol] = {
                "name": get_long_name(symbol),
                "symbol": symbol,
                "quantity": int(quantity),
                "avgPrice": float(avgPrice),
            }

            time.sleep(0.1)

    return stock_dict

# struct Stock: Equatable, Codable, Hashable {
#     let name: String
#     let symbol: String
#     var price: Double
#     var prices: [PricePoint]
#     var predictedPrice: Double
#     var sentimentAnalysis: String
# }
def make_stock(symbol, market ="NAS"):
    """현재가 조회"""
    PATH = "uapi/overseas-price/v1/quotations/price"
    URL = f"{URL_BASE}/{PATH}"
    headers = {"Content-Type":"application/json",
            "authorization": f"Bearer {ACCESS_TOKEN}",
            "appKey":APP_KEY,
            "appSecret":APP_SECRET,
            "tr_id":"HHDFS00000300"}
    params = {
        "AUTH": "",
        "EXCD":market,
        "SYMB":symbol,
    }
    res = requests.get(URL, headers=headers, params=params)
    current_price = res.json()['output']['last']
    # prices = make_price_point(symbol)
    # # TODO 예측값과 감성분석 임시로 설정해놓음
    # predicted_price = 0.0
    # sentiment = "Neutral"

    # stock = {
    #     "name": get_long_name(symbol),
    #     "symbol": symbol,
    #     "price": float(current_price),
    #     "prices": prices,
    #     "predictedPrice": float(predicted_price),
    #     "sentimentAnalysis": sentiment
    # }

    return float(current_price)

# struct PricePoint: Codable, Equatable, Hashable {
#     let date: String
#     let price: Double
# }
def make_price_point(symbol):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="1y")  # 1년치 일간 시세

    price_points = [
        {"date": date.strftime("%Y-%m-%d"), "price": round(close_price, 2)}
        for date, close_price in zip(df.index, df['Close'])
    ]
    return price_points

# TODO 예상가격 (float)
# def get_target_price(code="AAPL", market="NAS"):
#     """멀티모달 모델 이용 매수 목표가 조회"""
#     target_price = TODO
#     return target_price

# def buy(market="NASD", code="AAPL", qty="1", price="0"):
#     """미국 주식 지정가 매수"""
#     PATH = "uapi/overseas-stock/v1/trading/order"
#     URL = f"{URL_BASE}/{PATH}"
#     data = {
#         "CANO": CANO,
#         "ACNT_PRDT_CD": ACNT_PRDT_CD,
#         "OVRS_EXCG_CD": market,
#         "PDNO": code,
#         "ORD_DVSN": "00",
#         "ORD_QTY": str(int(qty)),
#         "OVRS_ORD_UNPR": f"{round(price,2)}",
#         "ORD_SVR_DVSN_CD": "0"
#     }
#     headers = {"Content-Type":"application/json",
#         "authorization":f"Bearer {ACCESS_TOKEN}",
#         "appKey":APP_KEY,
#         "appSecret":APP_SECRET,
#         "tr_id":"JTTT1002U",
#         "custtype":"P",
#         "hashkey" : hashkey(data)
#     }
#     res = requests.post(URL, headers=headers, data=json.dumps(data))
#     if res.json()['rt_cd'] == '0':
#         send_message(f"[매수 성공]{str(res.json())}")
#         return True
#     else:
#         send_message(f"[매수 실패]{str(res.json())}")
#         return False

# def sell(market="NASD", code="AAPL", qty="1", price="0"):
#     """미국 주식 지정가 매도"""
#     PATH = "uapi/overseas-stock/v1/trading/order"
#     URL = f"{URL_BASE}/{PATH}"
#     data = {
#         "CANO": CANO,
#         "ACNT_PRDT_CD": ACNT_PRDT_CD,
#         "OVRS_EXCG_CD": market,
#         "PDNO": code,
#         "ORD_DVSN": "00",
#         "ORD_QTY": str(int(qty)),
#         "OVRS_ORD_UNPR": f"{round(price,2)}",
#         "ORD_SVR_DVSN_CD": "0"
#     }
#     headers = {"Content-Type":"application/json",
#         "authorization":f"Bearer {ACCESS_TOKEN}",
#         "appKey":APP_KEY,
#         "appSecret":APP_SECRET,
#         "tr_id":"JTTT1006U",
#         "custtype":"P",
#         "hashkey" : hashkey(data)
#     }
#     res = requests.post(URL, headers=headers, data=json.dumps(data))
#     if res.json()['rt_cd'] == '0':
#         send_message(f"[매도 성공]{str(res.json())}")
#         return True
#     else:
#         send_message(f"[매도 실패]{str(res.json())}")
#         return False

# 자동매매 시작
try:
    ACCESS_TOKEN = get_access_token()

    symbol = "AAPL"

    # print(make_holding())
    # print(make_stock(symbol))
    # print(make_price_point(symbol))

except Exception as e:
    print("예외 발생:", e)
    time.sleep(1)