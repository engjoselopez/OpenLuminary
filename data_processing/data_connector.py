import os
import pandas as pd
import numpy as np
import requests
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import logging
import yfinance as yf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataConnector(ABC):
    """Abstract base class for financial data connectors."""
    
    @abstractmethod
    def get_historical_prices(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Get historical price data for a list of symbols."""
        pass
    
    @abstractmethod
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for a list of symbols."""
        pass
    
    @abstractmethod
    def get_financial_statements(self, symbols: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Get financial statements for a list of symbols."""
        pass

class YahooFinanceConnector(DataConnector):
    """Data connector for Yahoo Finance."""
    
    def __init__(self):
        """Initialize the Yahoo Finance connector."""
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = timedelta(hours=1)
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from cache if available and not expired."""
        if key in self.cache and datetime.now() < self.cache_expiry.get(key, datetime.min):
            return self.cache[key]
        return None
    
    def _store_in_cache(self, key: str, data: Any):
        """Store data in cache with expiry time."""
        self.cache[key] = data
        self.cache_expiry[key] = datetime.now() + self.cache_duration
    
    def get_historical_prices(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical price data for a list of symbols.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date in format YYYY-MM-DD
            end_date: End date in format YYYY-MM-DD
            interval: Data interval (1d, 1wk, 1mo)
            
        Returns:
            Dictionary mapping symbols to DataFrames with historical data
        """
        results = {}
        
        for symbol in symbols:
            cache_key = f"hist_{symbol}_{start_date}_{end_date}_{interval}"
            cached_data = self._get_from_cache(cache_key)
            
            if cached_data is not None:
                results[symbol] = cached_data
                continue
            
            try:
                data = yf.download(
                    symbol, 
                    start=start_date, 
                    end=end_date, 
                    interval=interval,
                    progress=False
                )
                
                if not data.empty:
                    results[symbol] = data
                    self._store_in_cache(cache_key, data)
                else:
                    logger.warning(f"No data returned for {symbol}")
                    results[symbol] = None
            except Exception as e:
                logger.error(f"Error fetching historical data for {symbol}: {e}")
                results[symbol] = None
        
        return results
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for a list of symbols.
        
        Args:
            symbols: List of ticker symbols
            
        Returns:
            Dictionary mapping symbols to their current prices
        """
        cache_key = f"current_{'_'.join(symbols)}"
        cached_data = self._get_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        results = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d")
                
                if not data.empty:
                    results[symbol] = data['Close'].iloc[-1]
                else:
                    logger.warning(f"No current price data for {symbol}")
                    results[symbol] = None
            except Exception as e:
                logger.error(f"Error fetching current price for {symbol}: {e}")
                results[symbol] = None
        
        self._store_in_cache(cache_key, results)
        return results
    
    def get_financial_statements(self, symbols: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Get financial statements for a list of symbols.
        
        Args:
            symbols: List of ticker symbols
            
        Returns:
            Dictionary mapping symbols to their financial statements
        """
        results = {}
        
        for symbol in symbols:
            cache_key = f"financials_{symbol}"
            cached_data = self._get_from_cache(cache_key)
            
            if cached_data is not None:
                results[symbol] = cached_data
                continue
            
            try:
                ticker = yf.Ticker(symbol)
                
                financials = {
                    "income_statement": ticker.income_stmt,
                    "balance_sheet": ticker.balance_sheet,
                    "cash_flow": ticker.cashflow,
                    "quarterly_income_statement": ticker.quarterly_income_stmt,
                    "quarterly_balance_sheet": ticker.quarterly_balance_sheet,
                    "quarterly_cash_flow": ticker.quarterly_cashflow
                }
                
                results[symbol] = financials
                self._store_in_cache(cache_key, financials)
            except Exception as e:
                logger.error(f"Error fetching financial statements for {symbol}: {e}")
                results[symbol] = None
        
        return results
    
    def get_company_info(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get company information for a list of symbols.
        
        Args:
            symbols: List of ticker symbols
            
        Returns:
            Dictionary mapping symbols to company information
        """
        results = {}
        
        for symbol in symbols:
            cache_key = f"info_{symbol}"
            cached_data = self._get_from_cache(cache_key)
            
            if cached_data is not None:
                results[symbol] = cached_data
                continue
            
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                results[symbol] = info
                self._store_in_cache(cache_key, info)
            except Exception as e:
                logger.error(f"Error fetching company info for {symbol}: {e}")
                results[symbol] = None
        
        return results

class AlphaVantageConnector(DataConnector):
    """Data connector for Alpha Vantage API."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the Alpha Vantage connector.
        
        Args:
            api_key: Alpha Vantage API key (defaults to environment variable ALPHA_VANTAGE_API_KEY)
        """
        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            logger.warning("Alpha Vantage API key not provided. Set ALPHA_VANTAGE_API_KEY environment variable or pass to constructor.")
        
        self.base_url = "https://www.alphavantage.co/query"
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = timedelta(hours=1)
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from cache if available and not expired."""
        if key in self.cache and datetime.now() < self.cache_expiry.get(key, datetime.min):
            return self.cache[key]
        return None
    
    def _store_in_cache(self, key: str, data: Any):
        """Store data in cache with expiry time."""
        self.cache[key] = data
        self.cache_expiry[key] = datetime.now() + self.cache_duration
    
    def _make_request(self, params: Dict[str, str]) -> Dict[str, Any]:
        """Make a request to Alpha Vantage API."""
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")
        
        params["apikey"] = self.api_key
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to Alpha Vantage: {e}")
            return {}
    
    def get_historical_prices(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str,
        interval: str = "daily"
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical price data for a list of symbols.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date in format YYYY-MM-DD
            end_date: End date in format YYYY-MM-DD
            interval: Data interval (daily, weekly, monthly)
            
        Returns:
            Dictionary mapping symbols to DataFrames with historical data
        """
        results = {}
        
        for symbol in symbols:
            cache_key = f"av_hist_{symbol}_{start_date}_{end_date}_{interval}"
            cached_data = self._get_from_cache(cache_key)
            
            if cached_data is not None:
                results[symbol] = cached_data
                continue
            
            function = {
                "daily": "TIME_SERIES_DAILY_ADJUSTED",
                "weekly": "TIME_SERIES_WEEKLY_ADJUSTED",
                "monthly": "TIME_SERIES_MONTHLY_ADJUSTED"
            }.get(interval, "TIME_SERIES_DAILY_ADJUSTED")
            
            params = {
                "function": function,
                "symbol": symbol,
                "outputsize": "full"
            }
            
            data = self._make_request(params)
            
            if not data or "Error Message" in data:
                logger.error(f"Error fetching data for {symbol}: {data.get('Error Message', 'Unknown error')}")
                results[symbol] = None
                continue
            
            time_series_key = [k for k in data.keys() if "Time Series" in k]
            if not time_series_key:
                logger.error(f"No time series data found for {symbol}")
                results[symbol] = None
                continue
            
            time_series = data[time_series_key[0]]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")
            
            # Convert column names
            column_map = {
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. adjusted close": "Adj Close",
                "6. volume": "Volume",
                "7. dividend amount": "Dividend",
                "8. split coefficient": "Split"
            }
            df.rename(columns=column_map, inplace=True)
            
            # Convert data types
            for col in df.columns:
                if col != "Split":  # Split can be a float
                    df[col] = pd.to_numeric(df[col])
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            
            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            # Sort by date
            df.sort_index(inplace=True)
            
            results[symbol] = df
            self._store_in_cache(cache_key, df)
        
        return results
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for a list of symbols.
        
        Args:
            symbols: List of ticker symbols
            
        Returns:
            Dictionary mapping symbols to their current prices
        """
        results = {}
        
        for symbol in symbols:
            cache_key = f"av_current_{symbol}"
            cached_data = self._get_from_cache(cache_key)
            
            if cached_data is not None:
                results[symbol] = cached_data
                continue
            
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol
            }
            
            data = self._make_request(params)
            
            if not data or "Error Message" in data or "Global Quote" not in data:
                logger.error(f"Error fetching current price for {symbol}")
                results[symbol] = None
                continue
            
            quote = data["Global Quote"]
            if "05. price" in quote:
                price = float(quote["05. price"])
                results[symbol] = price
                self._store_in_cache(cache_key, price)
            else:
                logger.error(f"No price data found for {symbol}")
                results[symbol] = None
        
        return results
    
    def get_financial_statements(self, symbols: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Get financial statements for a list of symbols.
        
        Args:
            symbols: List of ticker symbols
            
        Returns:
            Dictionary mapping symbols to their financial statements
        """
        results = {}
        
        for symbol in symbols:
            cache_key = f"av_financials_{symbol}"
            cached_data = self._get_from_cache(cache_key)
            
            if cached_data is not None:
                results[symbol] = cached_data
                continue
            
            financials = {}
            
            # Income Statement
            params = {
                "function": "INCOME_STATEMENT",
                "symbol": symbol
            }
            
            income_data = self._make_request(params)
            
            if "annualReports" in income_data:
                annual_income = pd.DataFrame(income_data["annualReports"])
                annual_income.set_index("fiscalDateEnding", inplace=True)
                annual_income.index = pd.to_datetime(annual_income.index)
                
                quarterly_income = pd.DataFrame(income_data.get("quarterlyReports", []))
                if not quarterly_income.empty:
                    quarterly_income.set_index("fiscalDateEnding", inplace=True)
                    quarterly_income.index = pd.to_datetime(quarterly_income.index)
                
                financials["income_statement"] = annual_income
                financials["quarterly_income_statement"] = quarterly_income
            
            # Balance Sheet
            params = {
                "function": "BALANCE_SHEET",
                "symbol": symbol
            }
            
            balance_data = self._make_request(params)
            
            if "annualReports" in balance_data:
                annual_balance = pd.DataFrame(balance_data["annualReports"])
                annual_balance.set_index("fiscalDateEnding", inplace=True)
                annual_balance.index = pd.to_datetime(annual_balance.index)
                
                quarterly_balance = pd.DataFrame(balance_data.get("quarterlyReports", []))
                if not quarterly_balance.empty:
                    quarterly_balance.set_index("fiscalDateEnding", inplace=True)
                    quarterly_balance.index = pd.to_datetime(quarterly_balance.index)
                
                financials["balance_sheet"] = annual_balance
                financials["quarterly_balance_sheet"] = quarterly_balance
            
            # Cash Flow
            params = {
                "function": "CASH_FLOW",
                "symbol": symbol
            }
            
            cash_flow_data = self._make_request(params)
            
            if "annualReports" in cash_flow_data:
                annual_cash_flow = pd.DataFrame(cash_flow_data["annualReports"])
                annual_cash_flow.set_index("fiscalDateEnding", inplace=True)
                annual_cash_flow.index = pd.to_datetime(annual_cash_flow.index)
                
                quarterly_cash_flow = pd.DataFrame(cash_flow_data.get("quarterlyReports", []))
                if not quarterly_cash_flow.empty:
                    quarterly_cash_flow.set_index("fiscalDateEnding", inplace=True)
                    quarterly_cash_flow.index = pd.to_datetime(quarterly_cash_flow.index)
                
                financials["cash_flow"] = annual_cash_flow
                financials["quarterly_cash_flow"] = quarterly_cash_flow
            
            if financials:
                results[symbol] = financials
                self._store_in_cache(cache_key, financials)
            else:
                logger.error(f"No financial data found for {symbol}")
                results[symbol] = None
        
        return results

class DataConnectorFactory:
    """Factory for creating data connectors."""
    
    @staticmethod
    def get_connector(connector_type: str, **kwargs) -> DataConnector:
        """
        Get a data connector by type.
        
        Args:
            connector_type: Type of connector ('yahoo', 'alphavantage')
            **kwargs: Additional arguments for the connector
            
        Returns:
            DataConnector instance
        """
        if connector_type.lower() == "yahoo":
            return YahooFinanceConnector()
        elif connector_type.lower() == "alphavantage":
            return AlphaVantageConnector(**kwargs)
        else:
            raise ValueError(f"Unknown connector type: {connector_type}")
