from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import requests
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import google.generativeai as genai
import json
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from CNN_PY import cnn_model
from LSTM_PY import lstm_model
from UsaStockAutoTrade import make_holding, get_long_name, make_price_point, make_stock
import os
from typing import List
import tensorflow as tf, joblib, numpy as np
from dateutil.parser import parse


# marketaux API-Key
MARKETAUX_API_KEY = "hMCAuYSbfahDopkGP1DmoEZdn90ky3LPWL6kJzoq"
# Gemini API-Key
API_KEY = "AIzaSyAKUehrFeC1tiZG8swNGO94VI4q3Fer0Bw"
# NewsAPI
News_API_KEY = "97f1e601562e49ccbe6b33f2dcf1ebc3"

# BBC
def extract_bbc_article_text(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    main_content = soup.find('main', id='main-content')
    if not main_content:
        print("main-content를 찾을 수 없습니다.")
        return None

    article = main_content.find('article')
    if not article:
        print("article을 찾을 수 없습니다.")
        return None

    text_blocks = article.find_all('div', attrs={'data-component': 'text-block'})
    if not text_blocks:
        print("text-block을 찾을 수 없습니다.")
        return None

    paragraphs = []
    for block in text_blocks:
        p_tags = block.find_all('p')
        for p in p_tags:
            text = p.get_text(strip=True)
            if text: 
                paragraphs.append(text)

    if not paragraphs:
        print("본문을 찾을 수 없습니다.")
        return None

    full_text = "\n".join(paragraphs)
    return full_text


# CBS
def extract_cbs_news_text(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    article_section = soup.select_one("article section.content__body")

    if not article_section:
        print("본문 섹션을 찾을 수 없습니다.")
        return None

    paragraphs = article_section.find_all("p")
    article_text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

    return article_text


# CNN
def extract_cnn_news_text(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    try:
        article_root = soup.select_one("body div section article[data-uri] section main")
        if not article_root:
            print("본문 루트(main)를 찾지 못했습니다.")
            return None

        paragraphs = article_root.find_all("p")
        article_text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

        return article_text

    except Exception as e:
        print(f"오류 발생: {e}")
        return None


# Yahoo
def extract_yahoo_finance_article_text(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    svelte_div = soup.find('div', id='svelte')
    if not svelte_div:
        print("div#svelte tag error")
        return None

    main_tag = svelte_div.find('main')
    if not main_tag:
        print("<main> tag error")
        return None

    section_tag = main_tag.find('section')
    if not section_tag:
        print("<section> tag error")
        return None

    paragraphs = []
    for p in section_tag.find_all('p'):
        text = p.get_text(strip=True)
        if text:
            paragraphs.append(text)

    if not paragraphs:
        print("<p> tag error")
        return None

    return '\n'.join(paragraphs)


# URL Adapter
def extract_article_text(url):
    url = url.lower() 

    if 'bbc' in url:
        return extract_bbc_article_text(url)
    elif 'yahoo' in url:
        return extract_yahoo_finance_article_text(url)
    elif 'cbsnews' in url:
        return extract_cbs_news_text(url)
    elif 'cnn' in url:
        return extract_cnn_news_text(url)
    else:
        print("URL support error")
        return None
    

# text parser
def collect_all_news_texts(url_list):
    all_texts = []

    for url in url_list:
        text = extract_article_text(url)
        if text:
            all_texts.append(text.strip())

    return "\n\n".join(all_texts)
    

# Yahoo Dynamic Crawling
def extract_yahoo_finance_news_links(ticker_url):
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")  
    chrome_options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(options=chrome_options)
    driver.get(ticker_url)

    # 페이지 로딩 대기
    time.sleep(5) 

    soup = BeautifulSoup(driver.page_source, 'html.parser')

    news_panel = soup.find('div', id='tabpanel-news')
    if not news_panel:
        print("tabpanel-news를 찾을 수 없습니다.")
        driver.quit()
        return []

    links = []

    stream_items = news_panel.find_all('div', class_=['stream-item', 'yf-186c5b2'])

    for item in stream_items:
        section = item.find('section')
        if section:
            a_tag = section.find('a', class_='subtle-link', href=True)
            if a_tag:
                links.append(a_tag['href'])

    driver.quit()
    return links


# Marketaux API
def fetch_sentiments(symbol):
    marketaux_url = "https://api.marketaux.com/v1/news/all"
    params = {
        "symbols": symbol,
        "filter_entities": "true",
        "published_after": "2025-04-22",
        "limit": 3,
        "language": "en",
        "api_token": MARKETAUX_API_KEY,
    }

    response = requests.get(marketaux_url, params=params)
    if response.status_code != 200:
        return None

    marketaux_data = response.json()
    sentiments = []

    for article in marketaux_data.get('data', []):
        for entity in article.get('entities', []):
            if entity.get('sentiment_score') is not None:
                sentiments.append(entity['sentiment_score'])

    return sentiments


#  NewsAPI
def fetch_news_from_newsapi(query, from_date, sources, api_key, page_size=10):
    
    target_date  = datetime.strptime(from_date, "%Y-%m-%d")
    from_date = target_date - timedelta(days=3)
    from_date = from_date.strftime("%Y-%m-%d")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "sources": ",".join(sources),
        "apiKey": api_key,
        "language": "en",
        # "pageSize": page_size
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"NewsAPI request failed: {response.status_code} - {response.text}")

    data = response.json()
    urls = [article['url'] for article in data.get('articles', [])]
    return urls


# Gemini API
def analyze_sentiment_with_gemini(news_text, stock_name):
    
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    combined_news = "\n\n".join(news_text)

    prompt = (
        f"Read the following news reports about {stock_name} "
        "and judge the overall sentiment between -1 (very negative) and 1 (very positive).\n"
        "Respond strictly in the following format:\n"
        "Line 1: A single float number representing the sentiment score (e.g., 0.35)\n"
        "Line 2: A short explanation (1-3 sentences) explaining why you gave that score.\n"
        "Do not include any headings, markdown, or extra formatting.\n\n"
        f"News:\n{combined_news}"
    )

    response = model.generate_content(prompt)
    lines = response.text.strip().split('\n')

    score = None
    explanation = ""
    if len(lines) >= 2:
        match = re.match(r"^(-?\d+(\.\d+)?)$", lines[0].strip())
        if match:
            try:
                score = float(match.group(0))
            except ValueError:
                raise ValueError("Invalid score format returned.")
        else:
            raise ValueError("Sentiment score not found in the first line.")
        explanation = "\n".join(lines[1:]).strip()
    else:
        raise ValueError("Response does not contain enough lines.")

    return score, explanation





app = FastAPI(title="Stock backend")

# ---------- pydantic 요청/응답 스키마 ----------
class StockRequest(BaseModel):
    symbol: str              
    name:   str

class PricePoint(BaseModel): 
    date: str
    price: float

class StockResponse(BaseModel):
    name: str
    symbol: str
    price: float # 현재가
    prices: List[PricePoint]
    predictedPrice: float
    sentimentScore: float
    sentimentAnalysis: str

# --------------------------------------------

class TradeRequest(BaseModel):
    symbol: str              
    name:   str
    predictedPrice: float

class TradeResponse(BaseModel):
    status: str

# --------------------------------------------

class HoldingItemResponse(BaseModel):
    name: str
    symbol: str
    quantity: int
    avgPrice: float

class HoldingsResponse(BaseModel):
    holdings: List[HoldingItemResponse]


# ----------------- 엔드포인트 ------------------

@app.get("/holding", response_model=HoldingsResponse) 
def holding():
    try:
        stock_dict = make_holding()
        holdings_list = [
            HoldingItemResponse(
                name=item["name"],
                symbol=item["symbol"],
                quantity=item["quantity"],
                avgPrice=item["avgPrice"]
            )
            for item in stock_dict.values()
        ]
        return HoldingsResponse(holdings=holdings_list)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"잔고 조회 중 서버 오류 발생: {str(e)}")
    

@app.post("/trade", response_model=TradeResponse)
def trade(stock_req : TradeRequest):
    try : 
        name = stock_req.name
        symbol = stock_req.symbol
        predictedPrice = stock_req.predictedPrice
        return TradeResponse(status="ok")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"자동 매매 등록 중 오류 발생: {str(e)}")


@app.post("/predict", response_model=StockResponse)
def predict(stock_req: StockRequest):

    target_date = datetime.today().strftime("%Y-%m-%d")

    sources = ["cbs-news", "bbc-news", "cnn"]
    current_price = make_stock(stock_req.symbol) # 1
    prices = get_price_history(stock_req.symbol, target_date) # 2
    predictedPrice = mlp_price(stock_req.symbol, target_date) # 3

    # NewsAPI 
    news_url_lists = fetch_news_from_newsapi(stock_req.name, target_date, sources, News_API_KEY)
    print(f"[INFO] NewsAPI에서 가져온 URL 개수: {len(news_url_lists)}")
    # Yahoo Finance URL crawling
    yahoo_url = f"https://finance.yahoo.com/quote/{stock_req.symbol}"
    yahoo_news_links = extract_yahoo_finance_news_links(yahoo_url)
    print(f"[INFO] Yahoo Finance에서 가져온 뉴스 링크 개수: {len(yahoo_news_links)}")
    news_url_lists.extend(yahoo_news_links)
    print(f"[INFO] 전체 뉴스 URL 개수 (합산): {len(news_url_lists)}")
    # News full contents
    full_news_text = collect_all_news_texts(news_url_lists)
    print(f"[INFO] 수집한 뉴스 본문 전체 길이: {len(full_news_text)}자")
    # Gemini 
    score, explanation = analyze_sentiment_with_gemini(news_text=full_news_text, stock_name=stock_req.name)
    print(f"[INFO] Gemini 분석 결과 - Score: {score}, Explanation: {explanation}")
    # Marketaux
    mkx_sentiment = fetch_sentiments(stock_req.symbol)
    print(f"[INFO] Marketaux 분석 결과 - Score: {mkx_sentiment}")
    
    sentimentScore = score * 0.8 + sum(mkx_sentiment) / 3 * 0.2 # 4
    sentimentAnaysis = explanation # 5 

    return StockResponse(
        name = stock_req.name, 
        symbol = stock_req.symbol,
        price = current_price,              # 1
        prices = prices,                    # 2
        predictedPrice = predictedPrice,    # 3
        sentimentScore = sentimentScore,    # 4    
        sentimentAnalysis=sentimentAnaysis  # 5
    )


def get_price_history(symbol: str, target_date: str) -> List[PricePoint]:
    '''
    # # from_date 이전의 충분한 일수 확보 (ex. 150일 확보)
    # from_date = datetime.strptime(target_date, "%Y-%m-%d") #parse(target_date)
    # start_date = from_date - timedelta(days=150)

    # # yfinance는 자동으로 거래일만 포함
    # data = yf.download(symbol, start=start_date.strftime("%Y-%m-%d"), end=from_date.strftime("%Y-%m-%d"))
    # data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
    
    # # 종가만 추출
    # closes = data["Close"].dropna().tail(100)  # 마지막 100 거래일만 추출
    # print(closes.tail(10))
    
    # if isinstance(closes.index, pd.MultiIndex):
    #     closes = closes.reset_index(level=0, drop=True)  # Ticker 레벨 제거

    # prices = [
    #     PricePoint(date=index.strftime("%Y-%m-%d"), price=round(price, 2))
    #     for index, price in closes.items()
    # ]
    '''
    return make_price_point(symbol)


def mlp_price(symbol: str, target_date: str) -> float:
    # 1. yfinance로 데이터 불러오기
    # datetime.strptime(target_date, "%Y-%m-%d")
    from_date = datetime.strptime(target_date, "%Y-%m-%d") #pd.to_datetime(target_date) #parse(target_date)
    start_dt = from_date - timedelta(days=150)

    df = yf.download(symbol, start=start_dt.strftime("%Y-%m-%d"), end=from_date.strftime("%Y-%m-%d"))
    
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(symbol, axis=1, level=1)

    df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

    # 2. 모델 및 스케일러 불러오기
    model = tf.keras.models.load_model('MLP_model_3.h5', compile=False)
    X_scaler = joblib.load('MLP_X_scaler_3.joblib')
    y_scaler = joblib.load('MLP_y_scaler_3.joblib')

    # 3. LSTM, CNN 모델로부터 예측값 얻기
    Close = lstm_model(target_date, symbol, df)
    Label = cnn_model(target_date, symbol, df)

    # 4. 예측할 새 데이터 구성
    Yes_Close = df['Close'].iloc[-1]  # 전일 종가를 마지막 값으로 사용
    
    
    # new_data = np.array([[Close, Label, Yes_Close]])
    # # 5. 스케일링 및 예측
    # new_data_scaled = X_scaler.transform(new_data)
    # prediction_scaled = model.predict(new_data_scaled)
    # prediction = y_scaler.inverse_transform(prediction_scaled)
    # print(f"예측된 최종 종가값: {prediction[0][0]:.2f}")
    # return float(prediction[0][0])
    # 5. Close와 Yes_Close만 스케일링
    features_to_scale = np.array([[Close, Yes_Close]])
    features_scaled = X_scaler.transform(features_to_scale)

    # 6. Label은 정규화 없이 원본 그대로 넣기
    input_for_model = np.hstack([features_scaled, np.array([[Label]])])  # shape (1,3)

    # 7. 예측 및 역변환
    prediction_scaled = model.predict(input_for_model)
    prediction = y_scaler.inverse_transform(prediction_scaled)

    print(f"예측된 최종 종가값: {prediction[0][0]:.2f}")
    return float(prediction[0][0])



# -----------------------------
# uvicorn main:app --reload  로 실행

        
        # holdings_list_for_response = [{'name':'Amazon', 'symbol':'AMZN', 'quantity':100, 'avgPrice':200}]
        # # for symbol, data in holdings_dict.items():
        # #     try:
        # #         holdings_list_for_response.append(HoldingItemResponse(
        # #             name='Apple', #data['name'],
        # #             symbol='AAPL', #data['symbol'],
        # #             quantity=data['quantity'],
        # #             avgPrice=data['avgPrice']
        # #         ))
        # #     except Exception as e:
        # #         # 개별 주식 항목 파싱 중 오류 발생 시 로깅 또는 처리
        # #         print(f"Error parsing holding item for symbol {symbol}: {e}")
        # #         # 오류가 발생한 항목은 건너뛰고 나머지 정상 항목만 포함
        # #         continue
        # return HoldingsResponse(holdings=holdings_list_for_response)