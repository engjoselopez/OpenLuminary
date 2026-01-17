# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 17:07:14 2025

@author: USER
"""

from models.qwen_interface import Qwen3Interface

# Initialize the Qwen3 interface 

qwen = Qwen3Interface(use_thinking_mode=True)

# Perform portfolio optimization analysis 

analysis_data = { "portfolio": { "AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.25, "AMZN": 0.25 }, "risk_tolerance": "moderate", "investment_horizon": "long-term", "constraints": "maximum allocation per asset: 40%" }

result = qwen.analyze_financial_data( data=analysis_data, analysis_type="portfolio_optimization" )

print(result)