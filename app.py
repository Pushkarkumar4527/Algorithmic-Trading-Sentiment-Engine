import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import requests
import xml.etree.ElementTree as ET
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# --- BACKEND FUNCTIONS ---

def get_ticker_suggestions(query):
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        if 'quotes' in data and len(data['quotes']) > 0:
            suggestions = []
            for q in data['quotes'][:3]:
                symbol = q.get('symbol', '')
                name = q.get('shortname', 'Unknown Company')
                if symbol:
                    suggestions.append(f"**{symbol}** ({name})")
            return suggestions
    except:
        return []
    return []

def add_technical_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df = df.dropna()
    return df

@st.cache_data
def fetch_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    prices = stock.history(start="2010-01-01", end=yesterday)
    if prices.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    prices = prices[['Close', 'Volume']].reset_index()
    prices['Date'] = pd.to_datetime(prices['Date']).dt.tz_localize(None).dt.date
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker_symbol}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    root = ET.fromstring(response.text)
    clean_news = []
    for item in root.findall('.//item'):
        title = item.find('title').text
        pub_date = item.find('pubDate').text
        date_only = pd.to_datetime(pub_date).tz_localize(None).date()
        sentiment = TextBlob(title).sentiment.polarity
        clean_news.append({'Date': date_only, 'Headline': title, 'Sentiment_Score': sentiment})
    if len(clean_news) == 0:
        news = pd.DataFrame(columns=['Date', 'Headline', 'Sentiment_Score'])
        final_df = prices.copy()
        final_df['Sentiment_Score'] = 0.0 
    else:
        news = pd.DataFrame(clean_news)
        daily_sentiment = news.groupby('Date')['Sentiment_Score'].mean().reset_index()
        final_df = pd.merge(prices, daily_sentiment, on='Date', how='left')
        final_df['Sentiment_Score'] = final_df['Sentiment_Score'].fillna(0.0)
    final_df = add_technical_indicators(final_df)
    return prices, news, final_df

# --- USER INTERFACE ---
st.set_page_config(page_title="Trading AI", layout="wide")
st.title("🤖 Algorithmic Trading Sentiment Engine")

st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TCS.NS):", "AAPL").upper()
run_button = st.sidebar.button("Run AI Prediction")

if run_button:
    with st.spinner(f"Processing {ticker}..."):
        try:
            prices, news, final_df = fetch_data(ticker)
            if final_df.empty:
                suggestions = get_ticker_suggestions(ticker)
                if suggestions:
                    st.warning(f"⚠️ No data found for '{ticker}'. Suggestions: " + " | ".join(suggestions))
                else:
                    st.error(f"❌ No data found for '{ticker}'.")
            else:
                st.success(f"Data Loaded for {ticker}")
                
                # AI Logic
                ml_data = final_df[['Close', 'Sentiment_Score', 'SMA_50', 'EMA_200', 'MACD', 'RSI']].values
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(ml_data)
                
                window_size = 5
                X, y = [], []
                for i in range(window_size, len(scaled_data)):
                    X.append(scaled_data[i-window_size:i]) 
                    y.append(scaled_data[i, 0])            
                X, y = np.array(X), np.array(y)
                X_flat = X.reshape(X.shape[0], -1)

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_flat, y)
                
                # Prediction
                last_5 = scaled_data[-window_size:].reshape(1, -1)
                raw_pred = model.predict(last_5)[0]
                dummy = np.zeros((1, 6))
                dummy[0, 0] = raw_pred
                final_pred = scaler.inverse_transform(dummy)[0, 0]
                
                current = final_df['Close'].iloc[-1]
                delta = final_pred - current

                # Results
                st.metric(label=f"Predicted Price", value=f"${final_pred:.2f}", delta=f"${delta:.2f}")
                if final_pred > current:
                    st.success("📈 BULLISH")
                else:
                    st.error("📉 BEARISH")
                    
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Historical Quant Data")
                    st.dataframe(final_df.tail(10))
                with col2:
                    st.subheader("Market Sentiment")
                    st.dataframe(news.head(10))
        except Exception as e:
            st.error(f"Error: {e}")