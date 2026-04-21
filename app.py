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

# Professional dashboard styling
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

@st.cache_data(ttl=3600)
def fetch_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    
    # Try-Except block to handle Yahoo Finance Rate Limits/Crashes
    try:
        prices = stock.history(start="2010-01-01")
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    if prices.empty: 
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
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

# --- SESSION STATE INITIALIZATION ---
if 'res' not in st.session_state:
    st.session_state.res = None

# --- SIDEBAR ---
st.sidebar.title("⚙️ Control Panel")
ticker_input = st.sidebar.text_input("Stock Ticker", "AAPL").upper()
lookback_input = st.sidebar.slider("AI Lookback Window (Days)", 3, 20, 5)
st.sidebar.markdown("---")
st.sidebar.info("This engine uses a Random Forest Ensemble with Dynamic Quant Retraining.")
run_button = st.sidebar.button("Execute AI Analysis", use_container_width=True)

# --- MAIN INTERFACE ---
st.title("🤖 Algorithmic Trading Sentiment Engine")
st.caption("Pushkar Kumar | B.Tech Computer Science | ML Project")
st.markdown("---")

# 1. RUN LOGIC
if run_button:
    with st.spinner(f"Analyzing {ticker_input}..."):
        prices, news, final_df = fetch_data(ticker_input)
        if final_df.empty:
            suggestions = get_ticker_suggestions(ticker_input)
            st.error(f"Data unavailable for {ticker_input}. This may be an invalid ticker, or Yahoo Finance is temporarily rate-limiting the server. Please try again in a few minutes.")
            if suggestions: st.warning("Did you mean: " + " | ".join(suggestions))
        else:
            # Training Logic
            feat_labels = ['Close', 'Sentiment', 'SMA_50', 'EMA_200', 'MACD', 'RSI']
            ml_data = final_df[feat_labels].values
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(ml_data)
            X, y = [], []
            for i in range(lookback_input, len(scaled)):
                X.append(scaled[i-lookback_input:i].flatten())
                y.append(scaled[i, 0])
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Prediction Logic
            current_price = final_df['Close'].iloc[-1]
            last_window = scaled[-lookback_input:].flatten().reshape(1, -1)
            pred_raw = model.predict(last_window)[0]
            dummy = np.zeros((1, len(feat_labels)))
            dummy[0,0] = pred_raw
            pred_final = scaler.inverse_transform(dummy)[0,0]

            # Save to Session State
            st.session_state.res = {
                'ticker': ticker_input,
                'current_price': current_price,
                'pred_final': pred_final,
                'final_df': final_df,
                'news': news,
                'model': model,
                'feat_labels': feat_labels,
                'trained_window': lookback_input # Save the window size used for training
            }

# 2. DISPLAY LOGIC (Checking if results exist in memory)
if st.session_state.res:
    r = st.session_state.res
    
    # Calculation
    delta = r['pred_final'] - r['current_price']
    pct = (delta / r['current_price']) * 100

    # Top Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${r['current_price']:.2f}")
    col2.metric("AI Predicted Price", f"${r['pred_final']:.2f}", f"{delta:.2f} ({pct:.2f}%)")
    col3.metric("Current RSI", f"{r['final_df']['RSI'].iloc[-1]:.2f}")

    # Confidence Score
    align = 0
    if (r['pred_final'] > r['current_price'] and r['final_df']['MACD'].iloc[-1] > 0): align += 1
    if (r['pred_final'] > r['current_price'] and r['final_df']['RSI'].iloc[-1] < 70): align += 1
    if (r['pred_final'] < r['current_price'] and r['final_df']['MACD'].iloc[-1] < 0): align += 1
    if (r['pred_final'] < r['current_price'] and r['final_df']['RSI'].iloc[-1] > 30): align += 1
    st.progress((60 + (align * 15)) / 100, text=f"🤖 Model Confidence: {60 + (align * 15)}%")

    st.markdown("### 📊 Strategy Recommendation")
    if pct >= 1.5: st.success("🟢 **ACTION: STRONG BUY**")
    elif pct <= -1.5: st.error("🔴 **ACTION: STRONG SELL**")
    else: st.warning("🟡 **ACTION: HOLD / NEUTRAL**")

    # Risk Management Engine
    st.markdown("---")
    st.subheader("🛡️ Risk Management Engine")
    with st.expander("Calculate Position Size", expanded=True):
        acct = st.number_input("Account Balance ($)", value=10000.0, step=1000.0)
        risk_input = st.slider("Risk Per Trade (%)", 0.5, 5.0, 1.0, 0.5)
        sl_def = r['current_price'] * 0.98 if r['pred_final'] > r['current_price'] else r['current_price'] * 1.02
        sl = st.number_input("Stop Loss Price ($)", value=float(sl_def))
        
        dist = abs(r['current_price'] - sl)
        if dist > 0:
            size = int((acct * (risk_input/100)) / dist)
            st.metric("Suggested Position Size (Shares)", size)
            rr = abs(r['pred_final'] - r['current_price']) / dist
            st.write(f"**Risk/Reward Ratio:** 1 : {rr:.2f}")

    # Charts and Explainable AI
    st.markdown("---")
    st.subheader(f"📈 {r['ticker']} Trend Analysis")
    st.line_chart(r['final_df'][['Close', 'SMA_50', 'EMA_200']].tail(100))

    with st.expander("🧠 Deep Dive: Explainable AI"):
        imps = r['model'].feature_importances_
        reshaped = imps.reshape(r['trained_window'], len(r['feat_labels']))
        avg_imps = np.mean(reshaped, axis=0)
        imp_df = pd.DataFrame({'Feature': r['feat_labels'], 'Importance': avg_imps}).sort_values('Importance', ascending=False)
        st.bar_chart(imp_df.set_index('Feature'))

    # INSPECT DATA SECTIONS
    st.markdown("---")
    with st.expander("🔍 Inspect Quantitative Vector Data"):
        st.dataframe(r['final_df'].tail(20), use_container_width=True)
    
    with st.expander("📰 Inspect Sentiment Analysis Feed"):
        st.dataframe(r['news'], use_container_width=True)

    # Export Report
    csv_data = r['final_df'].to_csv(index=False).encode('utf-8')
    st.download_button("Download Quant Data as CSV", data=csv_data, file_name=f"{r['ticker']}_report.csv")
    
    # Engine Footer
    st.markdown("---")
    st.subheader("⚙️ Engine Architecture")
    f1, f2, f3 = st.columns(3)
    f1.write("**Algorithm:** Random Forest Ensemble")
    f2.write("**Input:** 6-Feature Quant Vector")
    f3.write(f"**Parameters:** {r['trained_window']}-Day Dynamic Training")