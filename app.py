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
import os

# --- TENSORFLOW IMPORT WITH CRASH PROTECTION ---
# We wrap this in a try/except block just in case your local Windows PC 
# throws that Protobuf error again!
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except Exception as e:
    TENSORFLOW_AVAILABLE = False
    TENSORFLOW_ERROR = e

# --- BACKEND FUNCTIONS ---

@st.cache_resource
def load_deep_learning_model():
    """Loads the pre-trained LSTM file from your hard drive."""
    if os.path.exists('lstm_quant_engine.keras'):
        return load_model('lstm_quant_engine.keras')
    return None

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
    
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker_symbol}&region=US&lang=en-US"
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
st.title("🤖 Dual-Engine Algorithmic Trading System")

st.sidebar.header("System Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TCS.NS):", "AAPL").upper()

# ---> NEW: THE DUAL ENGINE SELECTOR <---
engine_choice = st.sidebar.selectbox(
    "Select AI Prediction Engine:", 
    ["Random Forest (Dynamic Quant)", "LSTM Deep Learning (Pre-Trained)"]
)

run_button = st.sidebar.button("Run AI Pipeline")

if run_button:
    with st.spinner(f"Connecting to data feeds and spinning up {engine_choice} for {ticker}..."):
        try:
            prices, news, final_df = fetch_data(ticker)
            
            if final_df.empty:
                suggestions = get_ticker_suggestions(ticker)
                if suggestions:
                    suggestion_text = " | ".join(suggestions)
                    st.warning(f"⚠️ No data found for '{ticker}'. **Did you mean one of these tickers?**")
                    st.info(suggestion_text)
                else:
                    st.error(f"❌ No data found for '{ticker}'. Please check your spelling.")
            else:
                st.success(f"Data ingested successfully! Formatting for {engine_choice}...")
                
                # --- UNIVERSAL DATA PREP ---
                ml_data = final_df[['Close', 'Sentiment_Score', 'SMA_50', 'EMA_200', 'MACD', 'RSI']].values
                dynamic_scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = dynamic_scaler.fit_transform(ml_data) 
                
                window_size = 5
                last_5_days = scaled_data[-window_size:]
                
                raw_prediction = 0
                prediction_success = True
                
                # --- ENGINE 1: LSTM ---
                if engine_choice == "LSTM Deep Learning (Pre-Trained)":
                    if not TENSORFLOW_AVAILABLE:
                        st.error(f"🚨 Local Environment Crash Prevented: TensorFlow could not load due to Windows Protobuf conflicts. Please select the Random Forest engine instead, or deploy this app to Streamlit Cloud.")
                        prediction_success = False
                    else:
                        lstm_model = load_deep_learning_model()
                        if lstm_model is None:
                            st.error("🚨 Missing File: Could not find 'lstm_quant_engine.keras'. Make sure it is in the exact same folder as your app.py file!")
                            prediction_success = False
                        else:
                            # LSTMs require 3D Tensors: (1 Sample, 5 Time Steps, 6 Features)
                            model_input = last_5_days.reshape(1, window_size, 6)
                            raw_prediction = lstm_model.predict(model_input)[0][0]
                            
                # --- ENGINE 2: RANDOM FOREST ---
                else:
                    X, y = [], []
                    for i in range(window_size, len(scaled_data)):
                        X.append(scaled_data[i-window_size:i]) 
                        y.append(scaled_data[i, 0])            

                    X, y = np.array(X), np.array(y)
                    X_train_flat = X.reshape(X.shape[0], -1)

                    dynamic_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    dynamic_model.fit(X_train_flat, y)
                    
                    # Random Forest requires 2D Flattened arrays
                    model_input = last_5_days.reshape(1, -1)
                    raw_prediction = dynamic_model.predict(model_input)[0]
                
                
                # --- DISPLAY RESULTS (IF AI WORKED) ---
                if prediction_success:
                    # Un-scale to US Dollars
                    dummy_array = np.zeros((1, 6))
                    dummy_array[0, 0] = raw_prediction
                    final_usd_prediction = dynamic_scaler.inverse_transform(dummy_array)[0, 0]
                    
                    current_price = final_df['Close'].iloc[-1]
                    delta = final_usd_prediction - current_price

                    st.markdown("---")
                    st.subheader(f"Engine Running: {engine_choice}")
                    st.metric(label=f"Predicted Future Price for {ticker}", 
                              value=f"${final_usd_prediction:.2f}", 
                              delta=f"${delta:.2f}")
                    
                    if final_usd_prediction > current_price:
                        st.success("📈 **AI FORECAST: BULLISH (UPWARD TREND)**")
                    else:
                        st.error("📉 **AI FORECAST: BEARISH (DOWNWARD TREND)**")
                        
                    # --- DISPLAY RAW DATA ---
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader(f"Quantitative Vector Data ({ticker})")
                        st.dataframe(final_df[['Date', 'Close', 'SMA_50', 'EMA_200', 'MACD', 'RSI', 'Sentiment_Score']].tail(10), use_container_width=True)
                    with col2:
                        st.subheader("Live Market Sentiment")
                        st.dataframe(news[['Date', 'Headline', 'Sentiment_Score']].head(10), use_container_width=True)

        except Exception as e:
            st.error(f"System Error: {e}")