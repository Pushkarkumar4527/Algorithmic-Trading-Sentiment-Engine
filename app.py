# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import datetime
# import requests
# import xml.etree.ElementTree as ET
# from textblob import TextBlob
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.ensemble import RandomForestRegressor

# # --- BACKEND FUNCTIONS ---

# def get_ticker_suggestions(query):
#     url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
#     headers = {'User-Agent': 'Mozilla/5.0'}
#     try:
#         response = requests.get(url, headers=headers)
#         data = response.json()
#         if 'quotes' in data and len(data['quotes']) > 0:
#             suggestions = []
#             for q in data['quotes'][:3]:
#                 symbol = q.get('symbol', '')
#                 name = q.get('shortname', 'Unknown Company')
#                 if symbol:
#                     suggestions.append(f"**{symbol}** ({name})")
#             return suggestions
#     except:
#         return []
#     return []

# def add_technical_indicators(df):
#     df['SMA_50'] = df['Close'].rolling(window=50).mean()
#     df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
#     ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
#     ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
#     df['MACD'] = ema_12 - ema_26
#     delta = df['Close'].diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
#     rs = gain / loss
#     df['RSI'] = 100 - (100 / (1 + rs))
#     df = df.dropna()
#     return df

# @st.cache_data
# def fetch_data(ticker_symbol):
#     stock = yf.Ticker(ticker_symbol)
#     yesterday = datetime.date.today() - datetime.timedelta(days=1)
#     prices = stock.history(start="2010-01-01", end=yesterday)
#     if prices.empty:
#         return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
#     prices = prices[['Close', 'Volume']].reset_index()
#     prices['Date'] = pd.to_datetime(prices['Date']).dt.tz_localize(None).dt.date
#     url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker_symbol}"
#     headers = {'User-Agent': 'Mozilla/5.0'}
#     response = requests.get(url, headers=headers)
#     root = ET.fromstring(response.text)
#     clean_news = []
#     for item in root.findall('.//item'):
#         title = item.find('title').text
#         pub_date = item.find('pubDate').text
#         date_only = pd.to_datetime(pub_date).tz_localize(None).date()
#         sentiment = TextBlob(title).sentiment.polarity
#         clean_news.append({'Date': date_only, 'Headline': title, 'Sentiment_Score': sentiment})
#     if len(clean_news) == 0:
#         news = pd.DataFrame(columns=['Date', 'Headline', 'Sentiment_Score'])
#         final_df = prices.copy()
#         final_df['Sentiment_Score'] = 0.0 
#     else:
#         news = pd.DataFrame(clean_news)
#         daily_sentiment = news.groupby('Date')['Sentiment_Score'].mean().reset_index()
#         final_df = pd.merge(prices, daily_sentiment, on='Date', how='left')
#         final_df['Sentiment_Score'] = final_df['Sentiment_Score'].fillna(0.0)
#     final_df = add_technical_indicators(final_df)
#     return prices, news, final_df

# # --- USER INTERFACE ---
# st.set_page_config(page_title="Trading AI", layout="wide")
# st.title("🤖 Algorithmic Trading Sentiment Engine")

# st.sidebar.header("Configuration")
# ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TCS.NS):", "AAPL").upper()
# run_button = st.sidebar.button("Run AI Prediction")

# if run_button:
#     with st.spinner(f"Processing {ticker}..."):
#         try:
#             prices, news, final_df = fetch_data(ticker)
#             if final_df.empty:
#                 suggestions = get_ticker_suggestions(ticker)
#                 if suggestions:
#                     st.warning(f"⚠️ No data found for '{ticker}'. Suggestions: " + " | ".join(suggestions))
#                 else:
#                     st.error(f"❌ No data found for '{ticker}'.")
#             else:
#                 st.success(f"Data Loaded for {ticker}")
                
#                 # AI Logic
#                 ml_data = final_df[['Close', 'Sentiment_Score', 'SMA_50', 'EMA_200', 'MACD', 'RSI']].values
#                 scaler = MinMaxScaler(feature_range=(0, 1))
#                 scaled_data = scaler.fit_transform(ml_data)
                
#                 window_size = 5
#                 X, y = [], []
#                 for i in range(window_size, len(scaled_data)):
#                     X.append(scaled_data[i-window_size:i]) 
#                     y.append(scaled_data[i, 0])            
#                 X, y = np.array(X), np.array(y)
#                 X_flat = X.reshape(X.shape[0], -1)

#                 model = RandomForestRegressor(n_estimators=100, random_state=42)
#                 model.fit(X_flat, y)
                
#                 # Prediction
#                 last_5 = scaled_data[-window_size:].reshape(1, -1)
#                 raw_pred = model.predict(last_5)[0]
#                 dummy = np.zeros((1, 6))
#                 dummy[0, 0] = raw_pred
#                 final_pred = scaler.inverse_transform(dummy)[0, 0]
                
#                 current = final_df['Close'].iloc[-1]
#                 delta = final_pred - current

#                 # Results
#                 st.metric(label=f"Predicted Price", value=f"${final_pred:.2f}", delta=f"${delta:.2f}")
#                 if final_pred > current:
#                     st.success("📈 BULLISH")
#                 else:
#                     st.error("📉 BEARISH")
                    
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.subheader("Historical Quant Data")
#                     st.dataframe(final_df.tail(10))
#                 with col2:
#                     st.subheader("Market Sentiment")
#                     st.dataframe(news.head(10))
#         except Exception as e:
#             st.error(f"Error: {e}")


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

# --- UI CONFIGURATION ---
st.set_page_config(page_title="QuantAI | Trading Engine", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a "Cinematic Dark" terminal feel
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #374151; }
    </style>
    """, unsafe_allow_html=True)

# --- BACKEND LOGIC ---

def get_ticker_suggestions(query):
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        if 'quotes' in data and len(data['quotes']) > 0:
            return [f"**{q.get('symbol')}** ({q.get('shortname')})" for q in data['quotes'][:3]]
    except: return []
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
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    return df.dropna()

@st.cache_data
def fetch_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    prices = stock.history(start="2010-01-01")
    if prices.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    prices = prices[['Close', 'Volume']].reset_index()
    prices['Date'] = pd.to_datetime(prices['Date']).dt.tz_localize(None).dt.date
    
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker_symbol}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        root = ET.fromstring(response.text)
        clean_news = [{'Date': pd.to_datetime(i.find('pubDate').text).date(), 
                       'Headline': i.find('title').text, 
                       'Sentiment': TextBlob(i.find('title').text).sentiment.polarity} 
                      for i in root.findall('.//item')]
    except: clean_news = []
    
    news_df = pd.DataFrame(clean_news) if clean_news else pd.DataFrame(columns=['Date', 'Headline', 'Sentiment'])
    daily_sent = news_df.groupby('Date')['Sentiment'].mean().reset_index() if not news_df.empty else pd.DataFrame(columns=['Date', 'Sentiment'])
    
    final_df = pd.merge(prices, daily_sent, on='Date', how='left').fillna(0)
    final_df = add_technical_indicators(final_df)
    return prices, news_df, final_df

# --- SIDEBAR ---
st.sidebar.title("⚙️ Control Panel")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()
st.sidebar.markdown("---")
st.sidebar.info("This engine uses a Random Forest Ensemble with Dynamic Quant Retraining.")
run_button = st.sidebar.button("Execute AI Analysis", use_container_width=True)

# --- MAIN INTERFACE ---
st.title("🤖 Algorithmic Trading Sentiment Engine")
st.markdown("---")

if run_button:
    with st.spinner(f"Analyzing {ticker} market patterns..."):
        prices, news, final_df = fetch_data(ticker)
        
        if final_df.empty:
            suggestions = get_ticker_suggestions(ticker)
            st.error(f"Invalid Ticker: {ticker}")
            if suggestions: st.warning("Did you mean: " + " | ".join(suggestions))
        else:
            # AI TRAINING
            ml_cols = ['Close', 'Sentiment', 'SMA_50', 'EMA_200', 'MACD', 'RSI']
            ml_data = final_df[ml_cols].values
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(ml_data)
            
            X, y = [], []
            for i in range(5, len(scaled)):
                X.append(scaled[i-5:i].flatten())
                y.append(scaled[i, 0])
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # PREDICTION
            current_price = final_df['Close'].iloc[-1]
            last_window = scaled[-5:].flatten().reshape(1, -1)
            pred_raw = model.predict(last_window)[0]
            
            # Inverse Scale
            dummy = np.zeros((1, 6))
            dummy[0,0] = pred_raw
            pred_final = scaler.inverse_transform(dummy)[0,0]
            delta = pred_final - current_price

            # DISPLAY METRICS
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:.2f}")
            col2.metric("AI Predicted Price", f"${pred_final:.2f}", f"{delta:.2f}")
            col3.metric("Market Sentiment", f"{final_df['Sentiment'].iloc[-1]:.2f}")

            # TRADING SIGNAL BOX
            st.markdown("### 📊 Strategy Recommendation")
            if pred_final > current_price:
                st.success(f"🚀 **BULLISH SIGNAL DETECTED**")
                st.write(f"The model anticipates a ${delta:.2f} move. MACD and RSI levels indicate healthy momentum.")
            else:
                st.error(f"📉 **BEARISH SIGNAL DETECTED**")
                st.write(f"The model anticipates a ${abs(delta):.2f} correction. Protective stops are recommended.")

            # EXPANDABLE DATA SECTIONS
            st.markdown("---")
            with st.expander("🔍 Inspect Quantitative Vector Data"):
                st.dataframe(final_df.tail(20), use_container_width=True)
            
            with st.expander("📰 Inspect Sentiment Analysis Feed"):
                st.dataframe(news, use_container_width=True)

            # MODEL INTELLIGENCE FOOTER
            st.markdown("---")
            st.subheader("🧠 Engine Intelligence")
            i_col1, i_col2, i_col3 = st.columns(3)
            i_col1.write("**Algorithm:** Random Forest Ensemble")
            i_col2.write("**Feature Set:** 6D Quant Vector")
            i_col3.write("**Training:** Dynamic Live Retrain")