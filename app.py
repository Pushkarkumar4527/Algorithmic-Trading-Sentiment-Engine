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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from fpdf import FPDF

def create_pdf_report(r, pct, action, align, risk_size, sl):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=16, style="B")
    pdf.cell(0, 10, f"QuantAI Trading Report: {r['ticker']}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=12)
    pdf.cell(0, 10, "="*70, align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("helvetica", size=14, style="B")
    pdf.cell(0, 10, "1. AI Strategy Overview", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=12)
    pdf.cell(0, 8, f"Current Price: ${r['current_price']:.2f}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"AI Predicted Target: ${r['pred_final']:.2f} ({pct:.2f}%)", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Action: {action}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Model Confidence: {60 + (align * 15)}%", new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(5)
    pdf.set_font("helvetica", size=14, style="B")
    pdf.cell(0, 10, "2. Risk Management Protocol", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=12)
    pdf.cell(0, 8, f"Suggested Position Size: {risk_size} Shares", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Stop Loss Level: ${sl:.2f}", new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(5)
    pdf.set_font("helvetica", size=14, style="B")
    pdf.cell(0, 10, "3. Recent Market Sentiment", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=10)
    if not r['news'].empty:
        for idx, row in r['news'].head(5).iterrows():
            clean_headline = str(row['Headline']).encode('ascii', 'ignore').decode()
            pdf.cell(0, 6, f"- {clean_headline}", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.cell(0, 6, "No recent news found.", new_x="LMARGIN", new_y="NEXT")
        
    pdf.ln(10)
    pdf.set_font("helvetica", size=9, style="I")
    pdf.cell(0, 10, "Disclaimer: This report is AI-generated and not financial advice.", align="C", new_x="LMARGIN", new_y="NEXT")
    
    return bytes(pdf.output())

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
    
    prices = prices[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
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

@st.cache_data(ttl=3600)
def fetch_fundamentals(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        fundamentals = {
            'Name': info.get('longName', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'Market Cap': info.get('marketCap', 'N/A'),
            'P/E Ratio': info.get('trailingPE', 'N/A'),
            'Dividend Yield': info.get('dividendYield', 'N/A'),
            'Profit Margin': info.get('profitMargins', 'N/A'),
            'Summary': info.get('longBusinessSummary', '')
        }
        if isinstance(fundamentals['Market Cap'], (int, float)):
            if fundamentals['Market Cap'] >= 1e12: fundamentals['Market Cap'] = f"${fundamentals['Market Cap']/1e12:.2f}T"
            elif fundamentals['Market Cap'] >= 1e9: fundamentals['Market Cap'] = f"${fundamentals['Market Cap']/1e9:.2f}B"
            else: fundamentals['Market Cap'] = f"${fundamentals['Market Cap']/1e6:.2f}M"
        
        if isinstance(fundamentals['Dividend Yield'], (int, float)): fundamentals['Dividend Yield'] = f"{fundamentals['Dividend Yield']*100:.2f}%"
        if isinstance(fundamentals['Profit Margin'], (int, float)): fundamentals['Profit Margin'] = f"{fundamentals['Profit Margin']*100:.2f}%"
        if isinstance(fundamentals['P/E Ratio'], (int, float)): fundamentals['P/E Ratio'] = f"{fundamentals['P/E Ratio']:.2f}"
            
        return fundamentals
    except:
        return None

@st.cache_data(ttl=3600)
def get_peer_tickers(ticker):
    url = f"https://query2.finance.yahoo.com/v6/finance/recommendationsbysymbol/{ticker}"
    try:
        data = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).json()
        recs = data['finance']['result'][0]['recommendedSymbols']
        return [r['symbol'] for r in recs[:3]] # Top 3 peers
    except:
        return []

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
        fundamentals = fetch_fundamentals(ticker_input)
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
                
            # --- EVALUATION METRICS LOGIC ---
            # Split data sequentially (80% train, 20% test) for evaluation
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train evaluation model
            eval_model = RandomForestRegressor(n_estimators=100, random_state=42)
            eval_model.fit(X_train, y_train)
            y_pred = eval_model.predict(X_test)
            
            # Unscale to calculate errors in actual Dollars
            dummy_y = np.zeros((len(y_test), len(feat_labels)))
            dummy_pred = np.zeros((len(y_pred), len(feat_labels)))
            dummy_y[:, 0] = y_test
            dummy_pred[:, 0] = y_pred
            y_test_dollars = scaler.inverse_transform(dummy_y)[:, 0]
            y_pred_dollars = scaler.inverse_transform(dummy_pred)[:, 0]
            
            mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
            rmse = np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars))
            r2 = r2_score(y_test_dollars, y_pred_dollars)
            
            # --- ACTUAL DEPLOYMENT MODEL ---
            # Train the real model on 100% of the data for tomorrow's prediction
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Prediction Logic
            current_price = final_df['Close'].iloc[-1]
            last_window = scaled[-lookback_input:].flatten().reshape(1, -1)
            pred_raw = model.predict(last_window)[0]
            dummy = np.zeros((1, len(feat_labels)))
            dummy[0,0] = pred_raw
            pred_final = scaler.inverse_transform(dummy)[0,0]

            # Peer Analysis
            peers = get_peer_tickers(ticker_input)
            peer_data = []
            if peers:
                st.info(f"Analyzing competitors: {', '.join(peers)}...")
                for p in peers:
                    p_prices, p_news, p_df = fetch_data(p)
                    p_fund = fetch_fundamentals(p)
                    if not p_df.empty and p_fund:
                        p_feat = p_df[feat_labels].values
                        p_scaler = MinMaxScaler()
                        p_scaled = p_scaler.fit_transform(p_feat)
                        p_X, p_y = [], []
                        for i in range(lookback_input, len(p_scaled)):
                            p_X.append(p_scaled[i-lookback_input:i].flatten())
                            p_y.append(p_scaled[i, 0])
                        p_model = RandomForestRegressor(n_estimators=50, random_state=42) # 50 trees for speed
                        p_model.fit(p_X, p_y)
                        p_last_window = p_scaled[-lookback_input:].flatten().reshape(1, -1)
                        p_pred_raw = p_model.predict(p_last_window)[0]
                        p_dummy = np.zeros((1, len(feat_labels)))
                        p_dummy[0,0] = p_pred_raw
                        p_pred_final = p_scaler.inverse_transform(p_dummy)[0,0]
                        p_current = p_df['Close'].iloc[-1]
                        p_delta = ((p_pred_final - p_current) / p_current) * 100
                        peer_data.append({
                            'Ticker': p,
                            'Name': p_fund['Name'],
                            'Market Cap': p_fund['Market Cap'],
                            'P/E Ratio': p_fund['P/E Ratio'],
                            'Current Price': f"${p_current:.2f}",
                            'AI Target': f"${p_pred_final:.2f}",
                            'Expected Return': f"{p_delta:+.2f}%"
                        })

            # Save to Session State
            st.session_state.res = {
                'ticker': ticker_input,
                'current_price': current_price,
                'pred_final': pred_final,
                'final_df': final_df,
                'news': news,
                'model': model,
                'feat_labels': feat_labels,
                'trained_window': lookback_input,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'fundamentals': fundamentals,
                'peer_data': peer_data
            }

# 2. DISPLAY LOGIC (Checking if results exist in memory)
if st.session_state.res:
    r = st.session_state.res
    
    # Fundamentals Dashboard
    if r.get('fundamentals'):
        f = r['fundamentals']
        st.markdown(f"### 🏢 {f['Name']} ({r['ticker']}) | {f['Sector']}")
        fund_col1, fund_col2, fund_col3, fund_col4 = st.columns(4)
        fund_col1.metric("Market Cap", f['Market Cap'])
        fund_col2.metric("P/E Ratio", f['P/E Ratio'])
        fund_col3.metric("Profit Margin", f['Profit Margin'])
        fund_col4.metric("Dividend Yield", f['Dividend Yield'])
        if f['Summary']:
            with st.expander("📖 Company Overview & Business Summary"):
                st.write(f['Summary'])
        st.markdown("---")
    
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
    action_text = "HOLD / NEUTRAL"
    if pct >= 1.5: 
        st.success("🟢 **ACTION: STRONG BUY**")
        action_text = "STRONG BUY"
    elif pct <= -1.5: 
        st.error("🔴 **ACTION: STRONG SELL**")
        action_text = "STRONG SELL"
    else: 
        st.warning("🟡 **ACTION: HOLD / NEUTRAL**")

    # Risk Management Engine
    st.markdown("---")
    st.subheader("🛡️ Risk Management Engine")
    with st.expander("Calculate Position Size", expanded=True):
        acct = st.number_input("Account Balance ($)", value=10000.0, step=1000.0)
        risk_input = st.slider("Risk Per Trade (%)", 0.5, 5.0, 1.0, 0.5)
        sl_def = r['current_price'] * 0.98 if r['pred_final'] > r['current_price'] else r['current_price'] * 1.02
        sl = st.number_input("Stop Loss Price ($)", value=float(sl_def))
        
        dist = abs(r['current_price'] - sl)
        size = 0
        if dist > 0:
            size = int((acct * (risk_input/100)) / dist)
            st.metric("Suggested Position Size (Shares)", size)
            rr = abs(r['pred_final'] - r['current_price']) / dist
            st.write(f"**Risk/Reward Ratio:** 1 : {rr:.2f}")

    # Peer Comparison
    if r.get('peer_data'):
        st.markdown("---")
        st.subheader("🥊 Industry Peer Comparison")
        f = r.get('fundamentals', {})
        main_stock = {
            'Ticker': r['ticker'] + " (Target)",
            'Name': f.get('Name', 'N/A'),
            'Market Cap': f.get('Market Cap', 'N/A'),
            'P/E Ratio': f.get('P/E Ratio', 'N/A'),
            'Current Price': f"${r['current_price']:.2f}",
            'AI Target': f"${r['pred_final']:.2f}",
            'Expected Return': f"{pct:+.2f}%"
        }
        comp_df = pd.DataFrame([main_stock] + r['peer_data'])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Charts and Explainable AI
    st.markdown("---")
    st.subheader(f"📈 {r['ticker']} Interactive Trend Analysis")
    
    # Plotly Candlestick Chart
    plot_df = r['final_df'].tail(150)
    fig = go.Figure(data=[go.Candlestick(x=plot_df['Date'],
                    open=plot_df['Open'],
                    high=plot_df['High'],
                    low=plot_df['Low'],
                    close=plot_df['Close'],
                    name='Price')])
    
    # Add Moving Averages
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['SMA_50'], line=dict(color='orange', width=1.5), name='SMA 50'))
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['EMA_200'], line=dict(color='cyan', width=1.5), name='EMA 200'))
    
    fig.update_layout(template='plotly_dark', margin=dict(l=0, r=0, t=30, b=0), height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🧠 Deep Dive: Explainable AI"):
        imps = r['model'].feature_importances_
        reshaped = imps.reshape(r['trained_window'], len(r['feat_labels']))
        avg_imps = np.mean(reshaped, axis=0)
        imp_df = pd.DataFrame({'Feature': r['feat_labels'], 'Importance': avg_imps}).sort_values('Importance', ascending=False)
        st.bar_chart(imp_df.set_index('Feature'))

    with st.expander("🧪 Model Evaluation Metrics"):
        st.write("These metrics are calculated using an 80/20 Time-Series split on historical data. Errors are represented in real Dollar amounts.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Mean Absolute Error (MAE)", f"${r['mae']:.2f}")
        m2.metric("Root Mean Squared Error (RMSE)", f"${r['rmse']:.2f}")
        m3.metric("R² Score (Accuracy)", f"{r['r2']:.4f}")

    # INSPECT DATA SECTIONS
    st.markdown("---")
    with st.expander("🔍 Inspect Quantitative Vector Data"):
        st.dataframe(r['final_df'].tail(20), use_container_width=True)
    
    with st.expander("📰 Inspect Sentiment Analysis Feed"):
        st.dataframe(r['news'], use_container_width=True)

    # Export Report
    st.markdown("---")
    st.subheader("📥 Export & Reports")
    ex_col1, ex_col2 = st.columns(2)
    
    csv_data = r['final_df'].to_csv(index=False).encode('utf-8')
    ex_col1.download_button("Download Quant Data (CSV)", data=csv_data, file_name=f"{r['ticker']}_data.csv", use_container_width=True)
    
    try:
        pdf_bytes = create_pdf_report(r, pct, action_text, align, size, sl)
        ex_col2.download_button("Download Executive Summary (PDF)", data=pdf_bytes, file_name=f"{r['ticker']}_report.pdf", mime="application/pdf", use_container_width=True)
    except Exception as e:
        ex_col2.error(f"PDF generation failed: {str(e)}")
    
    # Engine Footer
    st.markdown("---")
    st.subheader("⚙️ Engine Architecture")
    f1, f2, f3 = st.columns(3)
    f1.write("**Algorithm:** Random Forest Ensemble")
    f2.write("**Input:** 6-Feature Quant Vector")
    f3.write(f"**Parameters:** {r['trained_window']}-Day Dynamic Training")