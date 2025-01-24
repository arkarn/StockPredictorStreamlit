import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime, timedelta
from scipy.stats import norm

# Page Configuration
st.set_page_config(page_title="Stock Insight Pro", layout="wide", page_icon="📊")

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
    transition: transform 0.2s;
}
.metric-container:hover {
    transform: scale(1.02);
}
.tab-container {
    background-color: white;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Title and Description
st.title("📈 Stock Insight Pro")
st.markdown("### AI-Powered Market Intelligence Platform")

# Sidebar for Inputs
st.sidebar.header("Stock Analysis")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()
comparison_tickers = st.sidebar.text_input("Compare with (comma-separated)", "MSFT,GOOG").upper().split(',')
prediction_years = st.sidebar.slider("Forecast Years", 1, 5, 3)
period = st.sidebar.selectbox("Historical Data Period", ["1M", "6M", "YTD", "1Y", "5Y"], index=4)

# Technical Analysis Options
ta_options = st.sidebar.multiselect("Technical Indicators", 
                                  ["SMA 50", "SMA 200", "RSI", "MACD", "Bollinger Bands"])

# Risk Analysis Options
mc_simulations = st.sidebar.slider("Monte Carlo Simulations", 100, 1000, 500)
confidence_level = st.sidebar.slider("Confidence Level (%)", 90, 99, 95)

# Date Range Calculation
def get_date_range(period):
    today = datetime.now()
    if period == "1M": return today - timedelta(days=30)
    elif period == "6M": return today - timedelta(days=180)
    elif period == "YTD": return datetime(today.year, 1, 1)
    elif period == "1Y": return today - timedelta(days=365)
    else: return today - timedelta(days=5*365)

# Fetch Stock Data
@st.cache_data
def load_stock_data(ticker, period):
    try:
        start_date = get_date_range(period)
        stock_data = yf.download(ticker, start=start_date)
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

# Technical Indicators Calculations
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
    return data

# Monte Carlo Simulation
def monte_carlo_simulation(data, simulations, days):
    returns = np.log(1 + data['Close'].pct_change())
    mu = returns.mean()
    sigma = returns.std()
    
    S = data['Close'][-1]
    dt = 1
    results = np.zeros((days, simulations))
    
    for i in range(simulations):
        prices = [S]
        for _ in range(days-1):
            shock = np.random.normal(mu, sigma)
            price = prices[-1] * np.exp(shock)
            prices.append(price)
        results[:,i] = prices
    return results

# Load Data
stock_data = load_stock_data(ticker, period)
company_info = get_company_info(ticker)

if stock_data is not None and not stock_data.empty:
    # Calculate Technical Indicators
    stock_data = calculate_technical_indicators(stock_data, ta_options)
    
    # Create Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Technical Analysis", "Forecast", "Risk Analysis", "Comparison"])
    
    with tab1:  # Overview Tab
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Company Profile")
            if company_info:
                st.markdown(f"""
                **Sector**: {company_info.get('sector', 'N/A')}  
                **Industry**: {company_info.get('industry', 'N/A')}  
                **Employees**: {company_info.get('fullTimeEmployees', 'N/A'):,}  
                **P/E Ratio**: {company_info.get('trailingPE', 'N/A'):.2f}  
                **Dividend Yield**: {company_info.get('dividendYield', 'N/A') or 'N/A'}
                """)
        
        with col2:
            st.subheader("Key Metrics")
            cols = st.columns(4)
            metrics = {
                "Market Cap": f"${company_info.get('marketCap', 'N/A'):,}",
                "52W High": f"${stock_data['High'].max():.2f}",
                "52W Low": f"${stock_data['Low'].min():.2f}",
                "Volume (Avg)": f"{stock_data['Volume'].mean():,.0f}"
            }
            for col, (key, value) in zip(cols, metrics.items()):
                with col:
                    st.markdown(f'<div class="metric-container">{key}<br><h3>{value}</h3></div>', 
                              unsafe_allow_html=True)
        
        st.subheader("Interactive Price Chart")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=stock_data.index,
                                    open=stock_data['Open'],
                                    high=stock_data['High'],
                                    low=stock_data['Low'],
                                    close=stock_data['Close'],
                                    name='Price'))
        fig.update_layout(title=f'{ticker} Price Action', 
                        xaxis_title='Date', 
                        yaxis_title='Price',
                        xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:  # Technical Analysis Tab
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
    
    with tab3:  # Forecast Tab
        st.subheader("Price Forecast using Prophet")
        with st.spinner("Training forecasting model..."):
            forecast_data = stock_data[['Close']].reset_index()
            forecast_data.columns = ['ds', 'y']
            
            model = Prophet(interval_width=confidence_level/100)
            model.fit(forecast_data)
            
            future = model.make_future_dataframe(periods=prediction_years*365)
            forecast = model.predict(future)
            
            forecast_fig = plot_plotly(model, forecast)
            st.plotly_chart(forecast_fig, use_container_width=True)
    
    with tab4:  # Risk Analysis Tab
        st.subheader("Risk Metrics")
        
        # Calculate daily returns
        returns = stock_data['Close'].pct_change().dropna()
        
        # Value at Risk Calculation
        var = norm.ppf(1 - confidence_level/100, returns.mean(), returns.std())
        
        cols = st.columns(3)
        with cols[0]:
            st.metric("Annualized Volatility", f"{returns.std() * np.sqrt(252):.2%}")
        with cols[1]:
            st.metric(f"{confidence_level}% Daily VaR", f"{var:.2%}")
        with cols[2]:
            st.metric("Sharpe Ratio", f"{returns.mean()/returns.std() * np.sqrt(252):.2f}")
        
        st.subheader("Monte Carlo Simulations")
        simulation_days = prediction_years * 252
        mc_results = monte_carlo_simulation(stock_data, mc_simulations, simulation_days)
        
        mc_fig = go.Figure()
        for i in range(min(100, mc_simulations)):
            mc_fig.add_trace(go.Scatter(x=list(range(simulation_days)), 
                                      y=mc_results[:,i], 
                                      line=dict(width=0.5)))
        mc_fig.update_layout(title=f"{mc_simulations} Price Simulations",
                           xaxis_title="Trading Days",
                           yaxis_title="Price")
        st.plotly_chart(mc_fig, use_container_width=True)
    
    with tab5:  # Comparison Tab
        st.subheader("Stock Comparison")
        compare_data = {}
        
        for ct in [ticker] + comparison_tickers:
            data = load_stock_data(ct, period)
            if data is not None and not data.empty:
                compare_data[ct] = data['Close']
        
        if len(compare_data) > 1:
            compare_df = pd.DataFrame(compare_data)
            compare_normalized = compare_df / compare_df.iloc[0]
            
            fig = px.line(compare_normalized, title="Normalized Price Comparison")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Add valid tickers for comparison")

else:
    st.error("Unable to fetch stock data. Please check the ticker symbol.")
