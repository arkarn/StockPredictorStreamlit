import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Stock Insight Pro", layout="wide")

# Custom CSS for enhanced styling
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight:bold;
    color:#2c3e50;
}
.metric-container {
    background-color:#f0f2f6;
    border-radius:10px;
    padding:15px;
    margin:10px 0;
}
</style>
""", unsafe_allow_html=True)

# Title and Description
st.title("📈 Stock Insight Pro: Advanced Market Analysis")
st.markdown("### Your Comprehensive Stock Performance Dashboard")

# Sidebar for Inputs
st.sidebar.header("Stock Analysis Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())
prediction_years = st.sidebar.slider("Forecast Years", 1, 5, 3)

# Fetch Stock Data
@st.cache_data
def load_stock_data(ticker, start, end):
    try:
        stock_data = yf.download(ticker, start=start, end=end)
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Fetch Company Info
@st.cache_data
def get_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except Exception as e:
        st.error(f"Error fetching company info: {e}")
        return None

# Load Data
stock_data = load_stock_data(ticker, start_date, end_date)
company_info = get_company_info(ticker)

if stock_data is not None and not stock_data.empty:
    # Layout Columns
    col1, col2, col3 = st.columns(3)
    
    # Performance Metrics
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Current Price", f"${stock_data['Close'][-1]:.2f}")
        st.metric("Daily Change", f"{(stock_data['Close'][-1] - stock_data['Close'][-2]):.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("52-Week High", f"${stock_data['High'].max():.2f}")
        st.metric("52-Week Low", f"${stock_data['Low'].min():.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Company Information
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        if company_info:
            st.metric("Market Cap", f"${company_info.get('marketCap', 'N/A'):,}")
            st.metric("Sector", company_info.get('sector', 'N/A'))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Stock Price Visualization
    st.subheader("Stock Price History")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title=f'{ticker} Stock Price', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast Section
    st.subheader("Price Forecast using Prophet")
    forecast_data = stock_data[['Close']].reset_index()
    forecast_data.columns = ['ds', 'y']
    
    model = Prophet()
    model.fit(forecast_data)
    
    future = model.make_future_dataframe(periods=prediction_years*365)
    forecast = model.predict(future)
    
    # Forecast Visualization
    forecast_fig = plot_plotly(model, forecast)
    st.plotly_chart(forecast_fig, use_container_width=True)
    
    # Confidence Interval Visualization
    st.subheader("Forecast Uncertainty")
    uncertainty_fig = go.Figure()
    uncertainty_fig.add_trace(go.Scatter(
        x=forecast['ds'], 
        y=forecast['yhat'], 
        name='Forecast',
        line=dict(color='blue')
    ))
    uncertainty_fig.add_trace(go.Scatter(
        x=forecast['ds'], 
        y=forecast['yhat_upper'], 
        name='Upper Bound',
        line=dict(color='red', dash='dot')
    ))
    uncertainty_fig.add_trace(go.Scatter(
        x=forecast['ds'], 
        y=forecast['yhat_lower'], 
        name='Lower Bound',
        line=dict(color='green', dash='dot')
    ))
    st.plotly_chart(uncertainty_fig, use_container_width=True)
    
else:
    st.error("Unable to fetch stock data. Please check the ticker symbol.")
