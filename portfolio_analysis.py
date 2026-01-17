# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 14:20:14 2025

@author: USER
"""

# portfolio_analysis.py

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data_processing.data_connector import DataConnectorFactory
from portfolio_management.optimizer import PortfolioOptimizer
import pandas as pd

# -------------------------------
# 1. Inicializar componentes
# -------------------------------
data_connector = DataConnectorFactory.get_connector("yahoo")
portfolio_optimizer = PortfolioOptimizer()

# -------------------------------
# 2. Obtener datos históricos
# -------------------------------
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
start_date = "2022-01-01"
end_date = "2023-01-01"

market_data = data_connector.get_historical_prices(symbols, start_date, end_date)

# -------------------------------
# 3. Extraer precios de cierre y calcular retornos
# -------------------------------
close_prices = pd.DataFrame()

for symbol, data in market_data.items():
    if data is not None and not data.empty:
        close_prices[symbol] = data['Close']

returns = close_prices.pct_change().dropna()

# -------------------------------
# 4. Optimizar portafolio
# -------------------------------
result = portfolio_optimizer.optimize_sharpe_ratio(returns)

# -------------------------------
# 5. Mostrar resultados
# -------------------------------
print("✅ Optimized Portfolio Weights:")
for asset, weight in result["weights"].items():
    print(f"{asset}: {weight:.2%}")

print("\n📈 Portfolio Performance:")
print(f"Expected Annual Return: {result['performance']['return']:.2%}")
print(f"Annual Volatility: {result['performance']['volatility']:.2%}")
print(f"Sharpe Ratio: {result['performance']['sharpe_ratio']:.2f}")
