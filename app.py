import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime, timedelta
import requests
from textblob import TextBlob
import os
import snscrape.modules.twitter as sntwitter  # Free alternative to Twitter API

# Page Configuration
st.set_page_config(page_title="Stock Insight Pro", layout="wide", page_icon="📊")

# Custom CSS
st.markdown("""
<style>
.metric-container {
    background-color:#f0f2f6;
    border-radius:10px;
    padding:15px;
    margin:10px 0;
    transition: transform 0.2s;
}
.tab-container {border-radius: 10px; padding:15px; margin:10px 0;}
</style>
""", unsafe_allow_html=True)

# Title
st.title("📈 Stock Insight Pro + Social Pulse")

# Sidebar
st.sidebar.header("Analysis Parameters")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()
period = st.sidebar.selectbox("Historical Period", ["1M", "6M", "YTD", "1Y", "5Y"], index=4)
ta_options = st.sidebar.multiselect("Technical Indicators", ["SMA 50", "SMA 200", "RSI", "MACD", "Bollinger Bands"])

print("os env", os.environ.get("STOCKGEIST_API_KEY"))
print("st secret", st.secrets['STOCKGEIST_API_KEY'])
# Sentiment API Configuration (Using StockGeist from search results:cite[4])
STOCKGEIST_API_KEY = st.secrets['STOCKGEIST_API_KEY']  # Get from stockgeist.ai

# Date Range Calculation
def get_date_range(period):
    today = datetime.now()
    if period == "1M": return today - timedelta(days=30)
    elif period == "6M": return today - timedelta(days=180)
    elif period == "YTD": return datetime(today.year, 1, 1)
    elif period == "1Y": return today - timedelta(days=365)
    else: return today - timedelta(days=5*365)
        
# Fetch Data Functions
@st.cache_data
def load_stock_data(ticker, period):
    try:
        start_date = get_date_range(period)
        stock_data = yf.download(ticker, start=start_date)
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

@st.cache_data
def get_social_sentiment(ticker):
    try:
        url = f"https://api.stockgeist.ai/sentiment/{ticker}"
        headers = {"Authorization": f"Bearer {STOCKGEIST_API_KEY}"}
        response = requests.get(url, headers=headers)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Error fetching sentiment: {e}")
        return None

# Modified Social Feed Handling
def get_recent_tweets(ticker, limit=15):
    tweets = []
    try:
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(
            f'${ticker} lang:en since:{datetime.now().strftime("%Y-%m-%d")}').get_items()):
            if i >= limit:
                break
            tweets.append({
                "date": tweet.date,
                "content": tweet.content,
                "username": tweet.user.username,
                "sentiment": TextBlob(tweet.content).sentiment.polarity
            })
    except Exception as e:
        st.error(f"Tweet fetch error: {e}")
    return pd.DataFrame(tweets) if tweets else pd.DataFrame()

# Technical Indicators (Enhanced)
def calculate_technical_indicators(data, ta_options):
    if "SMA 50" in ta_options:
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
    if "SMA 200" in ta_options:
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
    if "RSI" in ta_options:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
    if "MACD" in ta_options:
        exp12 = data['Close'].ewm(span=12, adjust=False).mean()
        exp26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp12 - exp26
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    if "Bollinger Bands" in ta_options:
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['STD_20'] = data['Close'].rolling(window=20).std()
        data['Upper Band'] = data['SMA_20'] + (data['STD_20'] * 2)
        data['Lower Band'] = data['SMA_20'] - (data['STD_20'] * 2)
    if 'Close' in data:
        data['Price_Change'] = data['Close'].pct_change()
    return data

# Load Data
stock_data = load_stock_data(ticker, period)
sentiment_data = get_social_sentiment(ticker)
tweet_df = get_recent_tweets(ticker) if st.sidebar.checkbox("Show Live Social Feed") else None

if stock_data is not None:
    stock_data = calculate_technical_indicators(stock_data, ta_options)
    
    # Create Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Technical", "Sentiment", "Social Feed"])

    with tab1:  # Enhanced Dashboard
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            # Fix deprecated index access
            st.metric("Current Price", f"${stock_data['Close'].iloc[-1]:.2f}")
            st.metric("Volume (Avg)", f"{stock_data['Volume'].mean():,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            # Add error handling for sentiment data
            if sentiment_data:
                st.metric("Social Sentiment", 
                         f"{sentiment_data.get('score', 0):.2f}",
                         delta=f"{sentiment_data.get('change', 0):.2%}")
            else:
                st.metric("Social Sentiment", "N/A")
            
            # Add null check for tweet_df
            if tweet_df is not None and not tweet_df.empty:
                avg_sentiment = tweet_df['sentiment'].mean()
                sentiment_label = "Positive ▲" if avg_sentiment > 0 else "Negative ▼"
                st.metric("Recent News Tone", sentiment_label)
            else:
                st.metric("Recent News Tone", "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Price Chart with Sentiment Overlay
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=stock_data.index, open=stock_data['Open'],
                                    high=stock_data['High'], low=stock_data['Low'],
                                    close=stock_data['Close'], name='Price'))
        if sentiment_data:
            fig.add_trace(go.Scatter(x=stock_data.index, y=sentiment_data['historical'],
                                    name='Sentiment Score', yaxis='y2'))
        fig.update_layout(title=f'{ticker} Price & Sentiment', yaxis2=dict(overlaying='y', side='right'))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:  # Technical Analysis
        st.subheader("Technical Indicators")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Close Price'))
        
        if "SMA 50" in ta_options:
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_50'], name='SMA 50'))
        if "SMA 200" in ta_options:
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_200'], name='SMA 200'))
        if "Bollinger Bands" in ta_options:
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Upper Band'], name='Upper Band'))
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Lower Band'], name='Lower Band'))
        
        fig.update_layout(title='Technical Indicators', 
                        xaxis_title='Date', 
                        yaxis_title='Price')
        st.plotly_chart(fig, use_container_width=True)
        
        if "RSI" in ta_options:
            st.subheader("Relative Strength Index (RSI)")
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], name='RSI'))
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
            rsi_fig.update_layout(yaxis_range=[0,100])
            st.plotly_chart(rsi_fig, use_container_width=True)
        
        if "MACD" in ta_options:
            st.subheader("MACD")
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD'], name='MACD'))
            macd_fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Signal'], name='Signal'))
            st.plotly_chart(macd_fig, use_container_width=True)
    

    with tab3:  # Sentiment Analysis
        if sentiment_data:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Sentiment Breakdown")
                fig = px.pie(values=[sentiment_data['positive'], sentiment_data['neutral'], sentiment_data['negative']],
                            names=['Positive', 'Neutral', 'Negative'], hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Sentiment Sources")
                sources = pd.DataFrame({
                    'Source': ['Twitter', 'News', 'Reddit', 'Blogs'],
                    'Contribution': [sentiment_data['twitter_weight'], 
                                    sentiment_data['news_weight'],
                                    sentiment_data['reddit_weight'],
                                    sentiment_data['blog_weight']]
                })
                st.bar_chart(sources.set_index('Source'))

    with tab4:
        if tweet_df is not None and not tweet_df.empty:
            st.subheader(f"Latest Social Discussions about ${ticker}")
            for _, tweet in tweet_df.iterrows():
                # Add null check for sentiment value
                sentiment = tweet.get('sentiment', 0)
                sentiment_color = "#90EE90" if sentiment > 0 else "#FFCCCB" if sentiment < 0 else "#FFFFFF"
                st.markdown(f"""
                <div style="border-left: 4px solid {sentiment_color}; padding: 8px; margin: 4px;">
                    <b>@{tweet.get('username', '')}</b> · {tweet.get('date', '').strftime('%Y-%m-%d %H:%M')}<br>
                    {tweet.get('content', '')}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recent social discussions found")

else:
    st.error("Unable to fetch stock data")
