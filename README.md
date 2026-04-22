# 🤖 Algorithmic Trading Sentiment Engine
### A Dynamic Machine Learning Pipeline for Quant-Driven Stock Prediction

---

## 📌 Project Overview

This project is a sophisticated financial tool that bridges the gap between **Market Psychology (Sentiment Analysis)** and **Quantitative Data Engineering**.

Unlike traditional static models that fail when applied to new stocks, this engine performs **Dynamic On-The-Fly Training**. It allows users to input any global stock ticker and receive a real-time prediction by analyzing up to **16 years of market patterns** combined with **live news sentiment**.

Beyond simple price prediction, the engine includes a **complete Risk Management suite**, transforming it from a machine learning experiment into a practical **algorithmic trading dashboard**.

---

## 🚀 Key Features

### 🔹 Dynamic Ensemble Learning
- Uses a **Random Forest Regressor**
- Retrains dynamically for each ticker
- Adapts to asset-specific volatility and price scale

### 🔹 Explainable AI (XAI)
- Feature Importance visualization
- Shows whether prediction relies more on:
  - Technical indicators (RSI, MACD, etc.)
  - News sentiment

### 🔹 Risk Management Engine
- Position sizing calculator
- Inputs:
  - Account balance
  - Risk tolerance (%)
  - Stop-loss
- Outputs:
  - Shares to buy
  - Risk/Reward ratio

### 🔹 Sentiment Scraper
- Parses Yahoo Finance RSS feeds
- Uses NLP via TextBlob
- Generates sentiment polarity scores

### 🔹 Quant Library (Auto Feature Engineering)
- **SMA & EMA** → Trend detection
- **MACD** → Momentum & reversals
- **RSI** → Overbought/Oversold signals

### 🔹 Interactive Control Panel
- Adjustable **Lookback Window**
- Test short-term vs long-term strategies

### 🔹 Data Export
- Download processed dataset as CSV
- Useful for backtesting and research

---

## 🛠️ Tech Stack

| Category            | Tools                          |
|--------------------|--------------------------------|
| Language           | Python 3.10+                  |
| UI Framework       | Streamlit                     |
| AI / ML            | Scikit-Learn, NumPy           |
| Data Engineering   | Pandas, YFinance              |
| NLP                | TextBlob                      |
| Deployment         | GitHub, Streamlit Cloud       |

---

## 📈 System Architecture

### 1️⃣ Data Ingestion
- Historical OHLCV data (YFinance)
- Real-time news headlines

### 2️⃣ Feature Engineering
- Technical indicators + sentiment
- Creates a 6D Feature Vector

### 3️⃣ Preprocessing
- Data normalization using MinMaxScaler

### 4️⃣ Model Training
- Random Forest with 100 decision trees

### 5️⃣ Inference
- Predicting the Next-Day closing price

### 6️⃣ Strategy Execution
- Generates Buy / Hold / Sell signals
- Calculates capital exposure via Risk Management Engine

---

## 💻 Local Setup

1.  Clone the repository
```bash
git clone https://github.com/Pushkarkumar4527/trading-ai-engine.git
cd trading-ai-engine
2.  Install Dependencies
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Run the application
```bash
streamlit run app.py
```
