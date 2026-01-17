import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

class MarketDataProvider:
    """Provider for real-time and historical market data."""
    
    def __init__(self):
        """Initialize the market data provider."""
        self.data_cache = {}
    
    def get_historical_data(
        self, 
        symbols: List[str], 
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical market data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            Dictionary mapping symbols to their historical data DataFrames
        """
        results = {}
        for symbol in symbols:
            cache_key = f"{symbol}_{period}_{interval}"
            
            # Check if we have cached data less than 1 hour old
            if cache_key in self.data_cache:
                timestamp, data = self.data_cache[cache_key]
                if datetime.now() - timestamp < timedelta(hours=1):
                    results[symbol] = data
                    continue
            
            # Fetch new data
            try:
                data = yf.download(symbol, period=period, interval=interval, progress=False)
                results[symbol] = data
                self.data_cache[cache_key] = (datetime.now(), data)
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                results[symbol] = None
        
        return results
    
    def get_realtime_quote(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get real-time quotes for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            
        Returns:
            Dictionary mapping symbols to their quote data
        """
        results = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                # Get the current market price
                quote = ticker.info
                results[symbol] = {
                    "price": quote.get("regularMarketPrice"),
                    "change": quote.get("regularMarketChange"),
                    "change_percent": quote.get("regularMarketChangePercent"),
                    "volume": quote.get("regularMarketVolume"),
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                print(f"Error fetching real-time quote for {symbol}: {e}")
                results[symbol] = {"error": str(e)}
        
        return results
    
    def get_financial_statements(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Get financial statements for a company.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Dictionary containing income statement, balance sheet, and cash flow
        """
        try:
            ticker = yf.Ticker(symbol)
            return {
                "income_statement": ticker.income_stmt,
                "balance_sheet": ticker.balance_sheet,
                "cash_flow": ticker.cashflow,
            }
        except Exception as e:
            print(f"Error fetching financial statements for {symbol}: {e}")
            return {"error": str(e)}
