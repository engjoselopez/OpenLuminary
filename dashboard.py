import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Any

# Import project components
from data_processing.data_connector import DataConnectorFactory
from portfolio_management.optimizer import PortfolioOptimizer
from src.risk_assessment.advanced_risk import AdvancedRiskAssessment

# Page configuration
st.set_page_config(
    page_title="OpenLuminary - Financial Analysis Platform",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def get_data_provider():
    return DataConnectorFactory.get_connector("yahoo")

@st.cache_resource
def get_portfolio_optimizer():
    return PortfolioOptimizer()

@st.cache_resource
def get_risk_assessment():
    return AdvancedRiskAssessment()

data_provider = get_data_provider()
portfolio_optimizer = get_portfolio_optimizer()
risk_assessment = get_risk_assessment()

# App title and description
st.title("OpenLuminary")
st.markdown("### Open-source AI-powered financial analysis platform")

# Sidebar configuration
st.sidebar.title("Configuration")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Market Data", "Portfolio Optimization", "Risk Assessment", "AI Analysis"]
)

# Default tickers
default_tickers = "AAPL,MSFT,GOOGL,AMZN,META,TSLA,JPM,V"

# Date range
today = datetime.now()
default_start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')
default_end_date = today.strftime('%Y-%m-%d')

# Market Data Page
if page == "Market Data":
    st.header("Market Data")
    
    # Input parameters
    tickers = st.sidebar.text_input("Stock Tickers (comma-separated)", default_tickers).split(",")
    start_date = st.sidebar.date_input("Start Date", datetime.strptime(default_start_date, '%Y-%m-%d'))
    end_date = st.sidebar.date_input("End Date", datetime.strptime(default_end_date, '%Y-%m-%d'))
    
    # Fetch data
    with st.spinner("Fetching market data..."):
        market_data = data_provider.get_historical_prices(
            tickers, 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        )
        
        # Get current prices
        current_prices = data_provider.get_current_prices(tickers)
    
    # Display current prices
    st.subheader("Current Prices")
    
    # Create a DataFrame for current prices
    current_price_data = []
    for symbol, price in current_prices.items():
        if price is not None:
            # Get historical data for this symbol
            hist_data = market_data.get(symbol)
            if hist_data is not None and not hist_data.empty:
                prev_close = hist_data['Close'].iloc[-2] if len(hist_data) > 1 else None
                change = price - prev_close if prev_close is not None else None
                change_pct = (change / prev_close * 100) if prev_close is not None and prev_close != 0 else None
                
                current_price_data.append({
                    "Symbol": symbol,
                    "Price": price,
                    "Change": change,
                    "Change %": f"{change_pct:.2f}%" if change_pct is not None else "N/A"
                })
    
    if current_price_data:
        current_price_df = pd.DataFrame(current_price_data)
        st.dataframe(current_price_df, use_container_width=True)
    else:
        st.warning("No current price data available")
    
    # Display price charts
    st.subheader("Price Charts")
    
    # Create tabs for different chart types
    chart_tabs = st.tabs(["Line Chart", "Candlestick", "Returns", "Correlation"])
    
    with chart_tabs[0]:  # Line Chart
        # Select tickers to display
        selected_tickers = st.multiselect("Select tickers to display", tickers, default=tickers[:5])
        
        if selected_tickers:
            # Create DataFrame with close prices
            close_prices = pd.DataFrame()
            for symbol in selected_tickers:
                data = market_data.get(symbol)
                if data is not None and not data.empty:
                    close_prices[symbol] = data['Close']
            
            if not close_prices.empty:
                # Normalize prices to 100 at start
                normalized_prices = close_prices / close_prices.iloc[0] * 100
                
                # Create line chart
                fig = px.line(
                    normalized_prices, 
                    title="Normalized Price Performance (Base = 100)",
                    labels={"value": "Normalized Price", "variable": "Symbol"}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No price data available for selected tickers")
    
    with chart_tabs[1]:  # Candlestick
        # Select ticker for candlestick chart
        selected_ticker = st.selectbox("Select ticker for candlestick chart", tickers)
        
        data = market_data.get(selected_ticker)
        if data is not None and not data.empty:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=selected_ticker
            ))
            fig.update_layout(
                title=f"{selected_ticker} Price Movement",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No price data available for {selected_ticker}")
    
    with chart_tabs[2]:  # Returns
        # Select tickers for returns analysis
        selected_tickers = st.multiselect("Select tickers for returns analysis", tickers, default=tickers[:5])
        
        if selected_tickers:
            # Create DataFrame with close prices
            close_prices = pd.DataFrame()
            for symbol in selected_tickers:
                data = market_data.get(symbol)
                if data is not None and not data.empty:
                    close_prices[symbol] = data['Close']
            
            if not close_prices.empty:
                # Calculate returns
                returns = close_prices.pct_change().dropna()
                
                # Display returns distribution
                fig = px.box(
                    returns, 
                    title="Returns Distribution",
                    labels={"value": "Daily Return", "variable": "Symbol"}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate and display annualized metrics
                trading_days = 252
                annualized_returns = returns.mean() * trading_days
                annualized_volatility = returns.std() * np.sqrt(trading_days)
                sharpe_ratio = annualized_returns / annualized_volatility
                
                metrics_df = pd.DataFrame({
                    "Annualized Return": annualized_returns * 100,
                    "Annualized Volatility": annualized_volatility * 100,
                    "Sharpe Ratio": sharpe_ratio
                })
                
                st.subheader("Performance Metrics")
                st.dataframe(metrics_df.style.format({
                    "Annualized Return": "{:.2f}%",
                    "Annualized Volatility": "{:.2f}%",
                    "Sharpe Ratio": "{:.2f}"
                }), use_container_width=True)
            else:
                st.warning("No price data available for selected tickers")
    
    with chart_tabs[3]:  # Correlation
        # Select tickers for correlation analysis
        selected_tickers = st.multiselect("Select tickers for correlation analysis", tickers, default=tickers[:5])
        
        if selected_tickers:
            # Create DataFrame with close prices
            close_prices = pd.DataFrame()
            for symbol in selected_tickers:
                data = market_data.get(symbol)
                if data is not None and not data.empty:
                    close_prices[symbol] = data['Close']
            
            if not close_prices.empty:
                # Calculate returns
                returns = close_prices.pct_change().dropna()
                
                # Calculate correlation matrix
                correlation = returns.corr()
                
                # Display correlation heatmap
                fig = px.imshow(
                    correlation,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    title="Correlation Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No price data available for selected tickers")

# Portfolio Optimization Page
elif page == "Portfolio Optimization":
    st.header("Portfolio Optimization")
    
    # Input parameters
    tickers = st.sidebar.text_input("Stock Tickers (comma-separated)", default_tickers).split(",")
    start_date = st.sidebar.date_input("Start Date", datetime.strptime(default_start_date, '%Y-%m-%d'))
    end_date = st.sidebar.date_input("End Date", datetime.strptime(default_end_date, '%Y-%m-%d'))
    
    # Fetch data
    with st.spinner("Fetching market data..."):
        market_data = data_provider.get_historical_prices(
            tickers, 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        )
    
    # Extract close prices and calculate returns
    close_prices = pd.DataFrame()
    for symbol, data in market_data.items():
        if data is not None and not data.empty:
            close_prices[symbol] = data['Close']
    
    if not close_prices.empty:
        returns = close_prices.pct_change().dropna()
        
        # Portfolio optimization options
        st.subheader("Portfolio Optimization")
        
        optimization_tabs = st.tabs(["Equal Weight", "Maximum Sharpe", "Minimum Volatility", "Custom Weights"])
        
        with optimization_tabs[0]:  # Equal Weight
            # Equal weight portfolio
            n_assets = len(returns.columns)
            weights_equal = np.array([1.0 / n_assets] * n_assets)
            
            # Calculate performance
            performance_equal = portfolio_optimizer.calculate_portfolio_performance(returns, weights_equal)
            
            # Display metrics
            st.subheader("Equal Weight Portfolio")
            
            metrics_cols = st.columns(3)
            metrics_cols[0].metric("Expected Annual Return", f"{performance_equal['return']:.2%}")
            metrics_cols[1].metric("Annual Volatility", f"{performance_equal['volatility']:.2%}")
            metrics_cols[2].metric("Sharpe Ratio", f"{performance_equal['sharpe_ratio']:.2f}")
            
            # Display weights
            weights_df = pd.DataFrame({
                "Asset": returns.columns,
                "Weight": weights_equal * 100
            })
            
            fig = px.pie(
                weights_df, 
                values="Weight", 
                names="Asset",
                title="Portfolio Allocation"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with optimization_tabs[1]:  # Maximum Sharpe
            # Optimize for maximum Sharpe ratio
            with st.spinner("Optimizing portfolio..."):
                result = portfolio_optimizer.optimize_sharpe_ratio(returns)
            
            # Display metrics
            st.subheader("Maximum Sharpe Ratio Portfolio")
            
            metrics_cols = st.columns(3)
            metrics_cols[0].metric("Expected Annual Return", f"{result['performance']['return']:.2%}")
            metrics_cols[1].metric("Annual Volatility", f"{result['performance']['volatility']:.2%}")
            metrics_cols[2].metric("Sharpe Ratio", f"{result['performance']['sharpe_ratio']:.2f}")
            
            # Display weights
            weights_df = pd.DataFrame({
                "Asset": list(result["weights"].keys()),
                "Weight": [w * 100 for w in result["weights"].values()]
            })
            
            fig = px.pie(
                weights_df, 
                values="Weight", 
                names="Asset",
                title="Portfolio Allocation"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with optimization_tabs[2]:  # Minimum Volatility
            # Optimize for minimum volatility
            with st.spinner("Optimizing portfolio..."):
                result = portfolio_optimizer.optimize_minimum_volatility(returns)
            
            # Display metrics
            st.subheader("Minimum Volatility Portfolio")
            
            metrics_cols = st.columns(3)
            metrics_cols[0].metric("Expected Annual Return", f"{result['performance']['return']:.2%}")
            metrics_cols[1].metric("Annual Volatility", f"{result['performance']['volatility']:.2%}")
            metrics_cols[2].metric("Sharpe Ratio", f"{result['performance']['sharpe_ratio']:.2f}")
            
            # Display weights
            weights_df = pd.DataFrame({
                "Asset": list(result["weights"].keys()),
                "Weight": [w * 100 for w in result["weights"].values()]
            })
            
            fig = px.pie(
                weights_df, 
                values="Weight", 
                names="Asset",
                title="Portfolio Allocation"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with optimization_tabs[3]:  # Custom Weights
            # Custom weights
            st.subheader("Custom Portfolio Weights")
            
            # Create sliders for weights
            custom_weights = {}
            for i, ticker in enumerate(returns.columns):
                custom_weights[ticker] = st.slider(f"{ticker} Weight (%)", 0, 100, 100 // len(returns.columns))
            
            # Normalize weights
            total = sum(custom_weights.values())
            if total > 0:
                normalized_weights = {k: v/total for k, v in custom_weights.items()}
                
                # Convert to numpy array
                weights_array = np.array([normalized_weights[ticker] for ticker in returns.columns])
                
                # Calculate performance
                performance = portfolio_optimizer.calculate_portfolio_performance(returns, weights_array)
                
                # Display metrics
                metrics_cols = st.columns(3)
                metrics_cols[0].metric("Expected Annual Return", f"{performance['return']:.2%}")
                metrics_cols[1].metric("Annual Volatility", f"{performance['volatility']:.2%}")
                metrics_cols[2].metric("Sharpe Ratio", f"{performance['sharpe_ratio']:.2f}")
                
                # Display weights
                weights_df = pd.DataFrame({
                    "Asset": list(normalized_weights.keys()),
                    "Weight": [w * 100 for w in normalized_weights.values()]
                })
                
                fig = px.pie(
                    weights_df, 
                    values="Weight", 
                    names="Asset",
                    title="Portfolio Allocation"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Total weight must be greater than 0")
        
        # Generate and display efficient frontier
        st.subheader("Efficient Frontier")
        
        with st.spinner("Generating efficient frontier..."):
            efficient_frontier = portfolio_optimizer.generate_efficient_frontier(returns, n_points=30)
        
        # Plot efficient frontier
        fig = go.Figure()
        
        # Add efficient frontier
        fig.add_trace(go.Scatter(
            x=efficient_frontier["volatility"],
            y=efficient_frontier["return"],
            mode="lines",
            name="Efficient Frontier",
            line=dict(color="blue", width=2)
        ))
        
        # Add maximum Sharpe ratio portfolio
        max_sharpe_result = portfolio_optimizer.optimize_sharpe_ratio(returns)
        fig.add_trace(go.Scatter(
            x=[max_sharpe_result["performance"]["volatility"]],
            y=[max_sharpe_result["performance"]["return"]],
            mode="markers",
            marker=dict(size=12, color="red"),
            name="Maximum Sharpe Ratio"
        ))
        
        # Add minimum volatility portfolio
        min_vol_result = portfolio_optimizer.optimize_minimum_volatility(returns)
        fig.add_trace(go.Scatter(
            x=[min_vol_result["performance"]["volatility"]],
            y=[min_vol_result["performance"]["return"]],
            mode="markers",
            marker=dict(size=12, color="green"),
            name="Minimum Volatility"
        ))
        
        # Add equal weight portfolio
        equal_weight_performance = portfolio_optimizer.calculate_portfolio_performance(returns, weights_equal)
        fig.add_trace(go.Scatter(
            x=[equal_weight_performance["volatility"]],
            y=[equal_weight_performance["return"]],
            mode="markers",
            marker=dict(size=12, color="purple"),
            name="Equal Weight"
        ))
        
        # Update layout
        fig.update_layout(
            title="Efficient Frontier",
            xaxis_title="Annualized Volatility",
            yaxis_title="Annualized Return",
            height=600,
            xaxis=dict(tickformat=".0%"),
            yaxis=dict(tickformat=".0%")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No price data available for the selected tickers")

# Risk Assessment Page
elif page == "Risk Assessment":
    st.header("Risk Assessment")
    
    # Input parameters
    tickers = st.sidebar.text_input("Stock Tickers (comma-separated)", default_tickers).split(",")
    start_date = st.sidebar.date_input("Start Date", datetime.strptime(default_start_date, '%Y-%m-%d'))
    end_date = st.sidebar.date_input("End Date", datetime.strptime(default_end_date, '%Y-%m-%d'))
    confidence_level = st.sidebar.slider("Confidence Level", 0.9, 0.99, 0.95, 0.01)
    
    # Portfolio selection
    portfolio_type = st.sidebar.radio(
        "Portfolio Type",
        ["Equal Weight", "Maximum Sharpe", "Minimum Volatility", "Custom Weights"]
    )
    
    # Fetch data
    with st.spinner("Fetching market data..."):
        market_data = data_provider.get_historical_prices(
            tickers, 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        )
    
    # Extract close prices and calculate returns
    close_prices = pd.DataFrame()
    for symbol, data in market_data.items():
        if data is not None and not data.empty:
            close_prices[symbol] = data['Close']
    
    if not close_prices.empty:
        returns = close_prices.pct_change().dropna()
        
        # Determine portfolio weights based on selection
        if portfolio_type == "Equal Weight":
            n_assets = len(returns.columns)
            weights_array = np.array([1.0 / n_assets] * n_assets)
            weights_dict = {ticker: 1.0 / n_assets for ticker in returns.columns}
        
        elif portfolio_type == "Maximum Sharpe":
            result = portfolio_optimizer.optimize_sharpe_ratio(returns)
            weights_dict = result["weights"]
            weights_array = np.array([weights_dict[ticker] for ticker in returns.columns])
        
        elif portfolio_type == "Minimum Volatility":
            result = portfolio_optimizer.optimize_minimum_volatility(returns)
            weights_dict = result["weights"]
            weights_array = np.array([weights_dict[ticker] for ticker in returns.columns])
        
        else:  # Custom Weights
            # Create sliders for weights
            st.sidebar.subheader("Custom Weights")
            custom_weights = {}
            for ticker in returns.columns:
                custom_weights[ticker] = st.sidebar.slider(f"{ticker} Weight (%)", 0, 100, 100 // len(returns.columns))
            
            # Normalize weights
            total = sum(custom_weights.values())
            if total > 0:
                weights_dict = {k: v/total for k, v in custom_weights.items()}
                weights_array = np.array([weights_dict[ticker] for ticker in returns.columns])
            else:
                st.warning("Total weight must be greater than 0")
                weights_dict = {ticker: 1.0 / len(returns.columns) for ticker in returns.columns}
                weights_array = np.array([1.0 / len(returns.columns)] * len(returns.columns))
        
        # Display portfolio allocation
        st.subheader("Portfolio Allocation")
        
        weights_df = pd.DataFrame({
            "Asset": list(weights_dict.keys()),
            "Weight": [w * 100 for w in weights_dict.values()]
        })
        
        fig = px.pie(
            weights_df, 
            values="Weight", 
            names="Asset",
            title="Portfolio Allocation"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate risk metrics
        risk_metrics = risk_assessment.calculate_risk_metrics(returns, weights_array)
        
        # Display risk metrics
        st.subheader("Risk Metrics")
        
        metrics_cols1 = st.columns(3)
        metrics_cols1[0].metric("Expected Annual Return", f"{risk_metrics['mean_return']:.2%}")
        metrics_cols1[1].metric("Annual Volatility", f"{risk_metrics['volatility']:.2%}")
        metrics_cols1[2].metric("Sharpe Ratio", f"{risk_metrics['sharpe_ratio']:.2f}")
        
        metrics_cols2 = st.columns(3)
        metrics_cols2[0].metric("Sortino Ratio", f"{risk_metrics['sortino_ratio']:.2f}")
        metrics_cols2[1].metric("Maximum Drawdown", f"{risk_metrics['max_drawdown']:.2%}")
        metrics_cols2[2].metric("Calmar Ratio", f"{risk_metrics['calmar_ratio']:.2f}")
        
        metrics_cols3 = st.columns(2)
        metrics_cols3[0].metric("Skewness", f"{risk_metrics['skewness']:.2f}")
        metrics_cols3[1].metric("Kurtosis", f"{risk_metrics['kurtosis']:.2f}")
        
        # Calculate VaR and CVaR
        st.subheader(f"Value at Risk (VaR) and Conditional VaR at {confidence_level:.0%} Confidence")
        
        var_historical = risk_assessment.calculate_var(returns, weights_array, confidence_level, method="historical")
        var_parametric = risk_assessment.calculate_var(returns, weights_array, confidence_level, method="parametric")
        var_monte_carlo = risk_assessment.calculate_var(returns, weights_array, confidence_level, method="monte_carlo")
        
        cvar_historical = risk_assessment.calculate_cvar(returns, weights_array, confidence_level, method="historical")
        
        var_cols = st.columns(4)
        var_cols[0].metric("Historical VaR (1-day)", f"{var_historical['var']:.2%}")
        var_cols[1].metric("Parametric VaR (1-day)", f"{var_parametric['var']:.2%}")
        var_cols[2].metric("Monte Carlo VaR (1-day)", f"{var_monte_carlo['var']:.2%}")
        var_cols[3].metric("Historical CVaR (1-day)", f"{cvar_historical['cvar']:.2%}")
        
        # Calculate drawdown
        drawdown_result = risk_assessment.calculate_drawdown(returns, weights_array)
        
        # Plot drawdown
        st.subheader("Drawdown Analysis")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown_result["drawdown_series"].index,
            y=drawdown_result["drawdown_series"].values * 100,
            mode="lines",
            name="Drawdown",
            line=dict(color="red")
        ))
        fig.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Stress testing
        st.subheader("Stress Testing")
        
        # Define stress scenarios
        stress_scenarios = {
            "Market Crash": {ticker: 0.85 for ticker in returns.columns},
            "Tech Sector Decline": {
                ticker: 0.90 if ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "META"] else 0.98 
                for ticker in returns.columns
            },
            "Interest Rate Hike": {
                ticker: 0.95 if ticker in ["JPM", "BAC", "C", "WFC", "GS"] else 0.97
                for ticker in returns.columns
            },
            "Economic Boom": {ticker: 1.05 for ticker in returns.columns}
        }
        
        # Perform stress tests
        stress_results = risk_assessment.perform_stress_test(returns, weights_array, stress_scenarios)
        
        # Display stress test results
        stress_df = pd.DataFrame({
            "Scenario": list(stress_results.keys()),
            "Return": [result["return"] * 100 for result in stress_results.values()],
            "Volatility": [result["volatility"] * 100 for result in stress_results.values()],
            "VaR": [result["var"] * 100 for result in stress_results.values()],
            "CVaR": [result["cvar"] * 100 for result in stress_results.values()],
            "Max Drawdown": [result["max_drawdown"] * 100 for result in stress_results.values()]
        })
        
        fig = px.bar(
            stress_df,
            x="Scenario",
            y="Return",
            title="Stress Test Results - Expected Return",
            labels={"Return": "Expected Return (%)"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(stress_df.style.format({
            "Return": "{:.2f}%",
            "Volatility": "{:.2f}%",
            "VaR": "{:.2f}%",
            "CVaR": "{:.2f}%",
            "Max Drawdown": "{:.2f}%"
        }), use_container_width=True)
    else:
        st.warning("No price data available for the selected tickers")

# AI Analysis Page
elif page == "AI Analysis":
    st.header("AI-Powered Financial Analysis")
    
    # Input parameters
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Portfolio Optimization", "Risk Assessment", "Market Prediction", "Company Analysis"]
    )
    
    use_thinking_mode = st.sidebar.checkbox("Use Thinking Mode", True)
    
    # Disclaimer about Qwen3 model
    st.info(
        "This feature requires the Qwen3 model to be available. "
        "In a production environment, this would connect to a fine-tuned Qwen3 model. "
        "For demonstration purposes, we'll show sample outputs."
    )
    
    if analysis_type == "Portfolio Optimization":
        st.subheader("AI-Powered Portfolio Optimization")
        
        # Input parameters
        tickers = st.text_input("Stock Tickers (comma-separated)", default_tickers).split(",")
        investment_amount = st.number_input("Investment Amount ($)", min_value=1000, value=100000, step=1000)
        risk_tolerance = st.select_slider(
            "Risk Tolerance",
            options=["Very Conservative", "Conservative", "Moderate", "Aggressive", "Very Aggressive"],
            value="Moderate"
        )
        investment_horizon = st.select_slider(
            "Investment Horizon",
            options=["Short-term (< 1 year)", "Medium-term (1-5 years)", "Long-term (> 5 years)"],
            value="Medium-term (1-5 years)"
        )
        
        constraints = st.text_area(
            "Additional Constraints (optional)",
            "Maximum allocation per asset: 30%\nMinimum allocation per asset: 5%"
        )
        
        if st.button("Generate AI-Powered Portfolio Recommendation"):
            with st.spinner("Analyzing data and generating recommendations..."):
                # In a real implementation, this would call the Qwen3 model
                # For demonstration, we'll show a sample output
                
                # Sample AI response
                ai_response = f"""
                # Portfolio Optimization Recommendation

                Based on your investment parameters:
                - Investment Amount: ${investment_amount:,}
                - Risk Tolerance: {risk_tolerance}
                - Investment Horizon: {investment_horizon}
                - Tickers: {', '.join(tickers)}

                ## Recommended Portfolio Allocation

                | Asset | Allocation (%) | Amount ($) |
                |-------|---------------|------------|
                | AAPL  | 22.5%         | ${investment_amount * 0.225:,.2f} |
                | MSFT  | 20.0%         | ${investment_amount * 0.20:,.2f} |
                | GOOGL | 15.0%         | ${investment_amount * 0.15:,.2f} |
                | AMZN  | 12.5%         | ${investment_amount * 0.125:,.2f} |
                | META  | 10.0%         | ${investment_amount * 0.10:,.2f} |
                | TSLA  | 7.5%          | ${investment_amount * 0.075:,.2f} |
                | JPM   | 7.5%          | ${investment_amount * 0.075:,.2f} |
                | V     | 5.0%          | ${investment_amount * 0.05:,.2f} |

                ## Expected Performance

                - Expected Annual Return: 12.3%
                - Annual Volatility: 18.7%
                - Sharpe Ratio: 0.66
                - Maximum Drawdown: -22.4%

                ## Rationale

                This portfolio is optimized for a {risk_tolerance.lower()} risk tolerance with a {investment_horizon.lower()} investment horizon. The allocation balances growth potential with risk management through the following approach:

                1. **Technology Core (80%)**: Overweight in technology stocks to capture growth, with larger allocations to established companies (AAPL, MSFT) and smaller positions in higher-volatility names (TSLA).

                2. **Financial Stability (12.5%)**: JPM and V provide exposure to the financial sector with lower correlation to tech stocks, improving diversification.

                3. **Risk Management**: The portfolio respects your constraints while maintaining diversification across different market segments.

                ## Rebalancing Recommendation

                For a {investment_horizon.lower()} horizon, I recommend quarterly rebalancing to maintain target allocations while minimizing transaction costs.
                """
                
                st.markdown(ai_response)
    
    elif analysis_type == "Risk Assessment":
        st.subheader("AI-Powered Risk Assessment")
        
        # Input parameters
        tickers = st.text_input("Stock Tickers (comma-separated)", default_tickers).split(",")
        
        # Portfolio weights
        st.write("Enter portfolio weights (%):")
        weights = {}
        cols = st.columns(4)
        for i, ticker in enumerate(tickers):
            weights[ticker] = cols[i % 4].number_input(f"{ticker}", min_value=0, max_value=100, value=100 // len(tickers))
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            normalized_weights = {k: v/total for k, v in weights.items()}
        else:
            normalized_weights = {k: 0 for k in weights}
        
        market_scenario = st.selectbox(
            "Market Scenario to Analyze",
            ["Current Market Conditions", "Rising Interest Rates", "Economic Recession", "Inflationary Environment", "Tech Sector Correction"]
        )
        
        if st.button("Generate AI-Powered Risk Assessment"):
            with st.spinner("Analyzing portfolio risks..."):
                # In a real implementation, this would call the Qwen3 model
                # For demonstration, we'll show a sample output
                
                # Sample AI response
                ai_response = f"""
                # Comprehensive Risk Assessment

                ## Portfolio Overview

                I've analyzed your portfolio consisting of {len(tickers)} assets with the following allocation:
                
                {', '.join([f"{ticker}: {normalized_weights[ticker]:.1%}" for ticker in tickers])}

                ## Key Risk Metrics

                | Metric | Value | Interpretation |
                |--------|-------|---------------|
                | Value at Risk (95%) | 2.8% | There's a 5% chance of losing 2.8% or more in a single day |
                | Expected Shortfall | 3.9% | In the worst 5% of cases, the average loss would be 3.9% |
                | Maximum Drawdown | 32.4% | The portfolio could experience a 32.4% decline from peak to trough |
                | Beta | 1.18 | The portfolio is 18% more volatile than the overall market |
                | Downside Deviation | 12.3% | The portfolio's volatility considering only negative returns |

                ## Scenario Analysis: {market_scenario}

                Under a {market_scenario.lower()} scenario, the portfolio would likely experience:

                - Expected Return: -15.3%
                - Volatility: +45% (relative to normal conditions)
                - Most Affected Holdings: {tickers[0]}, {tickers[2]}, {tickers[4]}
                - Most Resilient Holdings: {tickers[1]}, {tickers[5]}

                ## Risk Factor Exposure

                1. **Technology Sector Concentration (68% of portfolio)**
                   - High exposure to technology regulation risk
                   - Vulnerable to sector rotation away from growth stocks

                2. **Interest Rate Sensitivity**
                   - Portfolio duration estimate: 8.2 years
                   - A 1% increase in rates could result in approximately 8.2% portfolio decline

                3. **Liquidity Risk**
                   - 92% of holdings are highly liquid
                   - 8% may face liquidity challenges in stressed markets

                ## Recommendations to Improve Risk Profile

                1. **Diversification Opportunities**
                   - Consider adding exposure to defensive sectors (utilities, consumer staples)
                   - Increase international diversification (current international exposure: 12%)

                2. **Hedging Strategies**
                   - Implementing a 15% allocation to inverse ETFs could reduce portfolio beta to 0.95
                   - Options collar strategy could limit downside to -15% with 12% cap on upside

                3. **Position Sizing Adjustments**
                   - Reduce {tickers[0]} position by 8-10% to decrease technology concentration
                   - Increase allocation to {tickers[5]} by 5-7% to improve defensive positioning
                """
                
                st.markdown(ai_response)
    
    elif analysis_type == "Market Prediction":
        st.subheader("AI-Powered Market Prediction")
        
        # Input parameters
        market_index = st.selectbox(
            "Market Index/Asset",
            ["S&P 500", "NASDAQ Composite", "Dow Jones Industrial Average", "Russell 2000", "Bitcoin", "Gold", "10-Year Treasury Yield"]
        )
        
        timeframe = st.selectbox(
            "Prediction Timeframe",
            ["1 Month", "3 Months", "6 Months", "1 Year"]
        )
        
        factors_to_consider = st.multiselect(
            "Factors to Consider",
            ["Monetary Policy", "Fiscal Policy", "Inflation", "Economic Growth", "Corporate Earnings", "Geopolitical Events", "Market Sentiment", "Technical Indicators"],
            ["Monetary Policy", "Inflation", "Economic Growth", "Corporate Earnings"]
        )
        
        if st.button("Generate AI-Powered Market Prediction"):
            with st.spinner("Analyzing market conditions and generating prediction..."):
                # In a real implementation, this would call the Qwen3 model
                # For demonstration, we'll show a sample output
                
                # Sample AI response
                ai_response = f"""
                # Market Prediction: {market_index} over the next {timeframe}

                ## Executive Summary

                Based on comprehensive analysis of current market conditions, economic indicators, and the factors you specified, my prediction for the {market_index} over the next {timeframe.lower()} is:

                **Moderately Bullish (+7-10% expected return)**

                Confidence level: 72%

                ## Key Drivers

                ### Monetary Policy
                The Federal Reserve is likely to maintain its current stance with a possible 25-50 basis point cut within the forecast period. This accommodative environment should continue to support asset prices, though less dramatically than during the 2020-2021 period.

                ### Inflation
                Inflation appears to be moderating, with core CPI expected to stabilize around 2.8-3.2% by the end of the forecast period. This "soft landing" scenario is positive for equities, particularly for the {market_index}.

                ### Economic Growth
                GDP growth projections of 2.1-2.4% suggest continued economic expansion, albeit at a slower pace than the post-pandemic recovery. This moderate growth environment historically favors the {market_index}, particularly if accompanied by stable inflation.

                ### Corporate Earnings
                Q2 and Q3 earnings forecasts suggest 7-9% year-over-year growth for companies in the {market_index}, with particularly strong performance in technology, healthcare, and consumer discretionary sectors.

                ## Technical Analysis

                The {market_index} is currently trading:
                - Above its 50-day moving average (bullish)
                - Above its 200-day moving average (bullish)
                - With RSI at 62 (moderately overbought but not extreme)
                - With positive MACD momentum

                Key support levels: 4,250, 4,120, 3,980
                Key resistance levels: 4,580, 4,720, 4,850

                ## Risk Factors

                Several factors could alter this outlook:

                1. **Inflation Resurgence**: A significant uptick in inflation could force more aggressive monetary tightening
                2. **Geopolitical Tensions**: Escalation in current conflicts or new geopolitical crises
                3. **Valuation Concerns**: Current P/E ratios are 15% above historical averages
                4. **Earnings Disappointments**: Particularly in heavily-weighted sectors

                ## Sector Outlook

                | Sector | Outlook | Key Drivers |
                |--------|---------|------------|
                | Technology | Strong Positive | AI investment, cloud growth, semiconductor demand |
                | Healthcare | Positive | Aging demographics, innovation pipeline, defensive characteristics |
                | Financials | Neutral | Interest rate environment, loan growth, regulatory changes |
                | Energy | Negative | Supply/demand imbalances, renewable transition |
                | Consumer Discretionary | Positive | Strong consumer balance sheets, pent-up demand |

                ## Actionable Insights

                1. **Consider overweighting** technology and healthcare sectors
                2. **Implement dollar-cost averaging** rather than lump-sum investment given elevated volatility expectations
                3. **Set trailing stop orders** at 7-8% below current levels to protect against unexpected downside
                4. **Consider partial portfolio hedging** through options or inverse ETFs (10-15% allocation)
                """
                
                st.markdown(ai_response)
    
    elif analysis_type == "Company Analysis":
        st.subheader("AI-Powered Company Analysis")
        
        # Input parameters
        ticker = st.text_input("Company Ticker", "AAPL")
        
        analysis_focus = st.multiselect(
            "Analysis Focus Areas",
            ["Financial Health", "Growth Prospects", "Competitive Position", "Valuation", "ESG Factors", "Management Quality", "Risk Factors"],
            ["Financial Health", "Growth Prospects", "Valuation"]
        )
        
        if st.button("Generate AI-Powered Company Analysis"):
            with st.spinner("Analyzing company data..."):
                # In a real implementation, this would call the Qwen3 model
                # For demonstration, we'll show a sample output
                
                # Sample AI response for Apple
                if ticker.upper() == "AAPL":
                    ai_response = """
                    # Apple Inc. (AAPL) - Comprehensive Analysis

                    ## Company Overview
                    Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide. The company offers iPhone, Mac, iPad, and wearables, home, and accessories comprising AirPods, Apple TV, Apple Watch, Beats products, and HomePod.

                    ## Financial Health
                    
                    Apple maintains an exceptionally strong financial position:

                    - **Cash & Investments**: $162.1 billion
                    - **Total Debt**: $110.7 billion
                    - **Net Cash Position**: $51.4 billion
                    - **Free Cash Flow (TTM)**: $97.5 billion
                    - **Operating Margin**: 30.1%
                    - **Return on Equity**: 147.9%

                    Apple's financial health is among the strongest in the technology sector, with substantial cash reserves providing flexibility for investments, acquisitions, and shareholder returns. The company's debt is easily manageable given its cash flow generation capacity.

                    ## Growth Prospects

                    Apple's growth outlook remains positive, though moderating from historical levels:

                    - **Revenue Growth (Forward)**: 6.2% projected
                    - **EPS Growth (Forward)**: 8.7% projected
                    - **Key Growth Drivers**:
                      1. Services ecosystem expansion (currently 23.6% of revenue with higher margins)
                      2. Wearables category (growing at 14.3% annually)
                      3. India market penetration (46% YoY growth)
                      4. AI integration across product lineup
                      5. Potential new product categories (AR/VR, automotive)

                    The company's transition toward services helps stabilize revenue and improve margins, while its ecosystem approach continues to drive customer retention and lifetime value.

                    ## Valuation

                    Current valuation metrics:

                    - **P/E Ratio**: 28.4x (vs. 5-year average of 24.2x)
                    - **Forward P/E**: 26.1x
                    - **PEG Ratio**: 3.0x
                    - **EV/EBITDA**: 21.3x
                    - **Dividend Yield**: 0.54%

                    Apple trades at a premium to its historical averages and the broader market, reflecting investor confidence in its ecosystem strength and services transition. However, current valuation leaves limited margin of safety and assumes successful execution of growth initiatives.

                    ## Discounted Cash Flow Analysis
                    
                    Based on a 5-year DCF model with the following assumptions:
                    - 7% revenue CAGR for 5 years, declining to 4% terminal growth
                    - Gradual margin expansion to 32% operating margin
                    - 9% weighted average cost of capital (WACC)
                    
                    **Fair Value Estimate**: $172 per share (representing approximately 5% downside from current price)

                    ## Investment Thesis

                    ### Strengths
                    - Unparalleled ecosystem integration and customer loyalty
                    - Transition to higher-margin services business
                    - Strong brand premium allowing above-industry margins
                    - Substantial capital return program ($90B+ annual buybacks)
                    - Innovation pipeline and R&D investments

                    ### Risks
                    - Elongating smartphone replacement cycles
                    - Regulatory scrutiny of App Store practices
                    - Concentrated revenue in iPhone product line
                    - China market exposure and geopolitical tensions
                    - Premium pricing strategy in potential economic downturn

                    ## Recommendation

                    **HOLD** with a price target of $172

                    Apple remains a high-quality company with exceptional financial strength and ecosystem advantages. However, current valuation appears to fully reflect these strengths, limiting potential upside in the near term. The company's transition to services and potential new product categories provide long-term growth potential, but investors may find better entry points during market volatility.

                    For existing shareholders, Apple continues to be a core holding with strong capital return programs. New investors should consider dollar-cost averaging or waiting for more attractive valuation levels.
                    """
                else:
                    ai_response = f"""
                    # {ticker.upper()} - Comprehensive Analysis

                    I've analyzed {ticker.upper()} based on the latest available financial data, market positioning, and industry trends.

                    ## Financial Health

                    {ticker.upper()} demonstrates {['concerning', 'adequate', 'solid', 'strong', 'exceptional'][3]} financial health:

                    - **Cash & Investments**: $8.7 billion
                    - **Total Debt**: $12.3 billion
                    - **Debt-to-EBITDA**: 2.4x
                    - **Interest Coverage Ratio**: 8.2x
                    - **Free Cash Flow (TTM)**: $3.2 billion
                    - **Operating Margin**: 18.4% (vs. industry average of 15.7%)

                    The company's balance sheet is well-structured with manageable debt levels and sufficient liquidity to fund operations and strategic initiatives.

                    ## Growth Prospects

                    {ticker.upper()}'s growth trajectory appears {['concerning', 'below average', 'moderate', 'promising', 'exceptional'][3]}:

                    - **Revenue Growth (TTM)**: 12.3%
                    - **Revenue Growth (Forward)**: 9.7% projected
                    - **EPS Growth (Forward)**: 13.2% projected
                    - **Key Growth Drivers**:
                      1. Expansion into adjacent markets
                      2. Digital transformation initiatives
                      3. New product launches planned for Q3-Q4
                      4. International expansion (particularly APAC region)
                      5. Margin improvement through operational efficiency

                    The company is successfully executing its strategic plan with particular strength in digital offerings, which now represent 42% of total revenue (up from 28% three years ago).

                    ## Valuation

                    Current valuation metrics:

                    - **P/E Ratio**: 22.1x (vs. industry average of 19.4x)
                    - **Forward P/E**: 19.3x
                    - **PEG Ratio**: 1.5x
                    - **EV/EBITDA**: 14.2x
                    - **Dividend Yield**: 1.8%

                    {ticker.upper()} trades at a slight premium to industry peers, justified by its above-average growth rate and operational execution. The PEG ratio of 1.5x suggests reasonable valuation relative to growth expectations.

                    ## Competitive Position

                    {ticker.upper()} holds a {['weak', 'tenable', 'solid', 'strong', 'dominant'][3]} competitive position in its core markets:

                    - **Market Share**: 17.3% in core segments (up from 16.1% year-over-year)
                    - **Competitive Advantages**: 
                      1. Proprietary technology platform with high switching costs
                      2. Extensive distribution network
                      3. Brand reputation for quality and reliability
                      4. Scale economies in manufacturing and procurement
                      5. Strong patent portfolio (320+ active patents)

                    The company faces intensifying competition from both established players and new entrants, but its technological edge and customer relationships provide meaningful differentiation.

                    ## Investment Thesis

                    ### Strengths
                    - Consistent execution against strategic objectives
                    - Diversified revenue streams across multiple segments
                    - Margin expansion potential through automation and AI integration
                    - Strong R&D pipeline with several promising initiatives
                    - Experienced management team with industry expertise

                    ### Risks
                    - Increasing raw material costs impacting margins
                    - Regulatory changes in key markets
                    - Technology disruption potential in core segments
                    - Customer concentration (top 10 customers = 32% of revenue)
                    - Cyclical exposure to industrial sector

                    ## Recommendation

                    **BUY** with a price target of $87 (representing approximately 18% upside from current price)

                    {ticker.upper()} offers an attractive combination of reasonable valuation, solid growth prospects, and improving competitive position. The company's strategic initiatives are gaining traction, and its financial health provides flexibility to weather potential economic headwinds while investing in growth opportunities.
                    """
                
                st.markdown(ai_response)

### 7. Set Up Continuous Integration with GitHub Actions

#Create a GitHub Actions workflow file:

#**.github/workflows/ci.yml**:
