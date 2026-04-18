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

# Dynamic User-Controlled Window Size
window_size = st.sidebar.slider("AI Lookback Window (Days)", min_value=3, max_value=20, value=5, help="How many days of history the AI uses to predict tomorrow.")

st.sidebar.markdown("---")
st.sidebar.info("This engine uses a Random Forest Ensemble with Dynamic Quant Retraining.")
run_button = st.sidebar.button("Execute AI Analysis", use_container_width=True)

# --- MAIN INTERFACE ---
st.title("🤖 Algorithmic Trading Sentiment Engine")
st.caption("Pushkar Kumar | B.Tech Computer Science | ML Project")
st.markdown("---")

if run_button:
    with st.spinner(f"Analyzing {ticker} market patterns over {window_size}-day sequences..."):
        prices, news, final_df = fetch_data(ticker)
        
        if final_df.empty:
            suggestions = get_ticker_suggestions(ticker)
            st.error(f"Invalid Ticker: {ticker}")
            if suggestions: st.warning("Did you mean: " + " | ".join(suggestions))
        else:
            # --- AI TRAINING LOGIC ---
            feat_labels = ['Close', 'Sentiment', 'SMA_50', 'EMA_200', 'MACD', 'RSI']
            ml_data = final_df[feat_labels].values
            
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(ml_data)
            
            X, y = [], []
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
            percent_change = (delta / current_price) * 100

            # --- DISPLAY METRICS ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:.2f}")
            col2.metric("AI Predicted Price", f"${pred_final:.2f}", f"{delta:.2f} ({percent_change:.2f}%)")
            col3.metric("Current RSI", f"{final_df['RSI'].iloc[-1]:.2f}")

            # --- CONFIDENCE SCORE ---
            alignment = 0
            if (pred_final > current_price and final_df['MACD'].iloc[-1] > 0): alignment += 1
            if (pred_final > current_price and final_df['RSI'].iloc[-1] < 70): alignment += 1
            if (pred_final < current_price and final_df['MACD'].iloc[-1] < 0): alignment += 1
            if (pred_final < current_price and final_df['RSI'].iloc[-1] > 30): alignment += 1
            
            confidence = 60 + (alignment * 15) # Base 60%, +15% for each aligning indicator
            st.progress(confidence / 100, text=f"🤖 Model Confidence: {confidence}% (Based on Indicator Alignment)")

            # --- TRADING SIGNAL BOX ---
            st.markdown("### 📊 Strategy Recommendation")
            if percent_change >= 1.5:
                st.success(f"🟢 **ACTION: STRONG BUY**")
                st.write(f"The model anticipates a significant upward move of ${delta:.2f} (+{percent_change:.2f}%). Momentum indicators and sentiment align for potential growth.")
            elif percent_change <= -1.5:
                st.error(f"🔴 **ACTION: STRONG SELL**")
                st.write(f"The model anticipates a downward correction of ${abs(delta):.2f} ({percent_change:.2f}%). Liquidating or implementing tight stop-losses is advised.")
            else:
                st.warning(f"🟡 **ACTION: HOLD / NEUTRAL**")
                st.write(f"The model predicts minimal movement (${delta:.2f} or {percent_change:.2f}%). The asset is currently consolidating. Wait for a clearer trend breakout before deploying capital.")

            # --- REAL-WORLD RISK MANAGEMENT ENGINE ---
            st.markdown("---")
            st.subheader("🛡️ Risk Management Engine")
            with st.expander("Calculate Position Size based on AI Prediction", expanded=False):
                r_col1, r_col2 = st.columns(2)
                
                with r_col1:
                    account_balance = st.number_input("Total Account Balance ($)", value=10000.0, step=1000.0)
                    risk_pct = st.slider("Risk Per Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
                
                with r_col2:
                    # Default stop loss is 2% below/above current price depending on trend
                    stop_loss_default = current_price * 0.98 if pred_final > current_price else current_price * 1.02
                    stop_loss = st.number_input("Stop Loss Price ($)", value=float(stop_loss_default), step=1.0)
                
                # Calculations
                risk_amount = account_balance * (risk_pct / 100)
                sl_distance = abs(current_price - stop_loss)
                
                if sl_distance > 0:
                    position_size = int(risk_amount / sl_distance)
                    capital_required = position_size * current_price
                    reward_distance = abs(pred_final - current_price)
                    rr_ratio = reward_distance / sl_distance if sl_distance > 0 else 0
                    
                    st.markdown("#### Execution Plan")
                    e_col1, e_col2, e_col3 = st.columns(3)
                    e_col1.metric("Position Size (Shares)", position_size)
                    e_col2.metric("Capital Required", f"${capital_required:,.2f}")
                    e_col3.metric("Risk/Reward Ratio", f"1 : {rr_ratio:.2f}")
                    
                    if rr_ratio >= 2.0:
                        st.success("✅ **Favorable Setup:** The AI predicts a reward at least twice as large as your defined risk.")
                    elif rr_ratio >= 1.0:
                        st.warning("⚠️ **Marginal Setup:** The predicted reward barely covers the risk. Proceed with caution.")
                    else:
                        st.error("🛑 **Poor Setup:** The AI does not predict enough upward movement to justify your stop-loss risk. Do not enter.")
                else:
                    st.error("Stop Loss cannot equal Entry Price.")

            # --- VISUAL CHART SECTION ---
            st.markdown("---")
            st.subheader(f"📈 {ticker} Trend Analysis")
            chart_data = final_df[['Close', 'SMA_50', 'EMA_200']].tail(100)
            st.line_chart(chart_data)
            st.caption("Visualizing Closing Price against 50-day SMA and 200-day EMA.")

            # --- EXPLAINABLE AI SECTION ---
            st.markdown("---")
            with st.expander("🧠 Deep Dive: How the AI made this decision"):
                st.write(f"The chart below shows which features the Random Forest model prioritized across the {window_size}-day window for this specific prediction.")
                
                # Fetch importance from model and average across the window size
                importances = model.feature_importances_
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

            # --- EXPORT REPORT ---
            st.markdown("---")
            st.subheader("📥 Export Financial Data")
            csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Quant Data as CSV",
                data=csv,
                file_name=f'{ticker}_quant_report.csv',
                mime='text/csv',
            )

            # --- FOOTER ---
            st.markdown("---")
            st.subheader("⚙️ Engine Architecture")
            i_col1, i_col2, i_col3 = st.columns(3)
            i_col1.write("**Algorithm:** Random Forest Ensemble")
            i_col2.write("**Input:** 6-Feature Quant Vector")
            i_col3.write(f"**Parameters:** {window_size}-Day Dynamic Training")