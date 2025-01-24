import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime, timedelta
import requests
from textblob import TextBlob
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

# Sentiment API Configuration (Using StockGeist from search results:cite[4])
STOCKGEIST_API_KEY = os.environ.get('STOCKGEIST_API_KEY')  # Get from stockgeist.ai

# Fetch Data Functions
@st.cache_data
def load_stock_data(ticker, period):
    # ... (same as before)

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
    return pd.DataFrame(tweets)

# Technical Indicators (Enhanced)
def calculate_technical_indicators(data, ta_options):
    # ... (same SMA/RSI/MACD/Bollinger logic as before)
    # Add Sentiment Score Column
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
            st.metric("Current Price", f"${stock_data['Close'][-1]:.2f}")
            st.metric("Volume (Avg)", f"{stock_data['Volume'].mean():,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Social Sentiment", 
                     f"{sentiment_data['score']:.2f}" if sentiment_data else "N/A",
                     delta=f"{sentiment_data['change']:.2%}" if sentiment_data else "")
            st.metric("Recent News Tone", "Positive ▲" if (tweet_df['sentiment'].mean() > 0) else "Negative ▼" if tweet_df else "N/A")
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
        # ... (same technical charts as before)

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

    with tab4:  # Social Feed
        if tweet_df is not None and not tweet_df.empty:
            st.subheader(f"Latest Social Discussions about ${ticker}")
            for _, tweet in tweet_df.iterrows():
                sentiment_color = "#90EE90" if tweet['sentiment'] > 0 else "#FFCCCB" if tweet['sentiment'] < 0 else "#FFFFFF"
                st.markdown(f"""
                <div style="border-left: 4px solid {sentiment_color}; padding: 8px; margin: 4px;">
                    <b>@{tweet['username']}</b> · {tweet['date'].strftime('%Y-%m-%d %H:%M')}<br>
                    {tweet['content']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recent social discussions found")

else:
    st.error("Unable to fetch stock data")
