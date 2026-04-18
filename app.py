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

# Custom CSS for a professional dark dashboard feel
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #374151; }
    footer {visibility: hidden;}
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
    # Trend Indicators
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # Momentum (MACD)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    return df.dropna()

@st.cache_data(ttl=3600)
def fetch_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    prices = stock.history(start="2010-01-01")
    if prices.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    prices = prices[['Close', 'Volume']].reset_index()
    prices['Date'] = pd.to_datetime(prices['Date']).dt.tz_localize(None).dt.date
    
    # News & Sentiment
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
st.caption("Pushkar Kumar | B.Tech Computer Science | ML Project")
st.markdown("---")

if run_button:
    with st.spinner(f"Analyzing {ticker} market patterns..."):
        prices, news, final_df = fetch_data(ticker)
        
        if final_df.empty:
            suggestions = get_ticker_suggestions(ticker)
            st.error(f"Invalid Ticker: {ticker}")
            if suggestions: st.warning("Did you mean: " + " | ".join(suggestions))
        else:
            # --- AI TRAINING LOGIC ---
            # Define labels exactly as they appear in the data slice
            feat_labels = ['Close', 'Sentiment', 'SMA_50', 'EMA_200', 'MACD', 'RSI']
            ml_data = final_df[feat_labels].values
            
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(ml_data)
            
            X, y = [], []
            window_size = 5
            for i in range(window_size, len(scaled)):
                X.append(scaled[i-window_size:i].flatten())
                y.append(scaled[i, 0])
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # --- PREDICTION ---
            current_price = final_df['Close'].iloc[-1]
            last_window = scaled[-window_size:].flatten().reshape(1, -1)
            pred_raw = model.predict(last_window)[0]
            
            # Inverse Scale
            dummy = np.zeros((1, len(feat_labels)))
            dummy[0,0] = pred_raw
            pred_final = scaler.inverse_transform(dummy)[0,0]
            delta = pred_final - current_price

            # --- DISPLAY METRICS ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:.2f}")
            col2.metric("AI Predicted Price", f"${pred_final:.2f}", f"{delta:.2f}")
            col3.metric("Current RSI", f"{final_df['RSI'].iloc[-1]:.2f}")

            # --- TRADING SIGNAL BOX ---
            st.markdown("### 📊 Strategy Recommendation")
            if pred_final > current_price:
                st.success(f"🚀 **BULLISH SIGNAL DETECTED**")
                st.write(f"The model anticipates an upward move of ${delta:.2f}. Momentum indicators suggest potential growth.")
            else:
                st.error(f"📉 **BEARISH SIGNAL DETECTED**")
                st.write(f"The model anticipates a correction of ${abs(delta):.2f}. Caution or stop-loss implementation is advised.")

            # --- VISUAL CHART SECTION ---
            st.markdown("---")
            st.subheader(f"📈 {ticker} Trend Analysis")
            chart_data = final_df[['Close', 'SMA_50', 'EMA_200']].tail(100)
            st.line_chart(chart_data)
            st.caption("Visualizing Closing Price against 50-day SMA and 200-day EMA.")

            # --- EXPLAINABLE AI SECTION ---
            st.markdown("---")
            with st.expander("🧠 Deep Dive: How the AI made this decision"):
                st.write("The chart below shows which features the Random Forest model prioritized for this specific prediction.")
                
                # Fetch importance from model
                importances = model.feature_importances_
                
                # We used a flattened window of 5 days, so we average importance across the window for display
                # There are 30 total importance values (6 features * 5 days). We sum them by feature.
                reshaped_importances = importances.reshape(window_size, len(feat_labels))
                avg_importances = np.mean(reshaped_importances, axis=0)
                
                importance_df = pd.DataFrame({
                    'Feature': feat_labels, 
                    'Importance': avg_importances
                }).sort_values(by='Importance', ascending=False)
                
                st.bar_chart(importance_df.set_index('Feature'))
                st.info("The model weights technical data (RSI/MACD) and sentiment to find the most likely next-day price.")

            # --- DATA SECTIONS ---
            st.markdown("---")
            with st.expander("🔍 Inspect Quantitative Vector Data"):
                st.dataframe(final_df.tail(20), use_container_width=True)
            
            with st.expander("📰 Inspect Sentiment Analysis Feed"):
                st.dataframe(news, use_container_width=True)

            # --- FOOTER ---
            st.markdown("---")
            st.subheader("⚙️ Engine Architecture")
            i_col1, i_col2, i_col3 = st.columns(3)
            i_col1.write("**Algorithm:** Random Forest Ensemble")
            i_col2.write("**Input:** 6-Feature Quant Vector")
            i_col3.write("**Training:** Dynamic (On-the-Fly)")