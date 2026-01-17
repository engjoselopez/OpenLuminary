import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedRiskAssessment:
    """Advanced risk assessment for financial portfolios."""
    
    def __init__(self):
        """Initialize the advanced risk assessment module."""
        pass
    
    def calculate_var(
        self, 
        returns: pd.DataFrame, 
        weights: np.ndarray,
        confidence_level: float = 0.95,
        time_horizon: int = 1,
        method: str = "historical"
    ) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) using different methods.
        
        Args:
            returns: DataFrame of asset returns
            weights: Array of asset weights
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days
            method: Method to calculate VaR ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            Dictionary with VaR values
        """
        # Calculate portfolio returns
        portfolio_returns = returns.dot(weights)
        
        if method == "historical":
            # Historical VaR
            var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
            var_scaled = var * np.sqrt(time_horizon)
            return {"var": -var_scaled}
        
        elif method == "parametric":
            # Parametric VaR (assuming normal distribution)
            mean = portfolio_returns.mean()
            std = portfolio_returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mean + z_score * std
            var_scaled = var * np.sqrt(time_horizon)
            return {"var": -var_scaled}
        
        elif method == "monte_carlo":
            # Monte Carlo VaR
            mean = portfolio_returns.mean()
            std = portfolio_returns.std()
            
            # Generate random returns
            np.random.seed(42)  # For reproducibility
            n_simulations = 10000
            simulated_returns = np.random.normal(mean, std, n_simulations)
            
            # Calculate VaR
            var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
            var_scaled = var * np.sqrt(time_horizon)
            return {"var": -var_scaled}
        
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    def calculate_cvar(
        self, 
        returns: pd.DataFrame, 
        weights: np.ndarray,
        confidence_level: float = 0.95,
        time_horizon: int = 1,
        method: str = "historical"
    ) -> Dict[str, float]:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        Args:
            returns: DataFrame of asset returns
            weights: Array of asset weights
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days
            method: Method to calculate CVaR ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            Dictionary with CVaR values
        """
        # Calculate portfolio returns
        portfolio_returns = returns.dot(weights)
        
        if method == "historical":
            # Historical CVaR
            var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
            cvar = portfolio_returns[portfolio_returns <= var].mean()
            cvar_scaled = cvar * np.sqrt(time_horizon)
            return {"cvar": -cvar_scaled}
        
        elif method == "parametric":
            # Parametric CVaR (assuming normal distribution)
            mean = portfolio_returns.mean()
            std = portfolio_returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mean + z_score * std
            
            # For normal distribution, CVaR is:
            # E[X | X <= VaR] = mean + std * phi(-z) / (1-confidence_level)
            # where phi is the PDF of the standard normal distribution
            phi_z = stats.norm.pdf(z_score)
            cvar = mean - std * phi_z / (1 - confidence_level)
            cvar_scaled = cvar * np.sqrt(time_horizon)
            return {"cvar": -cvar_scaled}
        
        elif method == "monte_carlo":
            # Monte Carlo CVaR
            mean = portfolio_returns.mean()
            std = portfolio_returns.std()
            
            # Generate random returns
            np.random.seed(42)  # For reproducibility
            n_simulations = 10000
            simulated_returns = np.random.normal(mean, std, n_simulations)
            
            # Calculate VaR
            var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
            
            # Calculate CVaR
            cvar = simulated_returns[simulated_returns <= var].mean()
            cvar_scaled = cvar * np.sqrt(time_horizon)
            return {"cvar": -cvar_scaled}
        
        else:
            raise ValueError(f"Unknown CVaR method: {method}")
    
    def calculate_drawdown(self, returns: pd.DataFrame, weights: np.ndarray) -> Dict[str, Any]:
        """
        Calculate drawdown metrics.
        
        Args:
            returns: DataFrame of asset returns
            weights: Array of asset weights
            
        Returns:
            Dictionary with drawdown metrics
        """
        # Calculate portfolio returns
        portfolio_returns = returns.dot(weights)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative_returns.cummax()
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Calculate maximum drawdown
        max_drawdown = drawdown.min()
        
        # Calculate average drawdown
        avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
        
        # Calculate drawdown duration
        is_drawdown = drawdown < 0
        drawdown_start = is_drawdown.astype(int).diff() > 0
        drawdown_end = is_drawdown.astype(int).diff() < 0
        
        # Find start and end dates of drawdowns
        drawdown_periods = []
        current_start = None
        
        for date, value in drawdown_start.items():
            if value:
                current_start = date
            
            if current_start and drawdown_end.get(date, False):
                drawdown_periods.append({
                    "start": current_start,
                    "end": date,
                    "duration": (date - current_start).days,
                    "max_drawdown": drawdown[current_start:date].min()
                })
                current_start = None
        
        # If we're still in a drawdown, add it
        if current_start:
            drawdown_periods.append({
                "start": current_start,
                "end": drawdown.index[-1],
                "duration": (drawdown.index[-1] - current_start).days,
                "max_drawdown": drawdown[current_start:].min()
            })
        
        # Calculate average drawdown duration
        avg_duration = np.mean([period["duration"] for period in drawdown_periods]) if drawdown_periods else 0
        
        return {
            "max_drawdown": max_drawdown,
            "avg_drawdown": avg_drawdown,
            "avg_duration": avg_duration,
            "drawdown_periods": drawdown_periods,
            "drawdown_series": drawdown
        }
    
    def calculate_risk_metrics(
        self, 
        returns: pd.DataFrame, 
        weights: np.ndarray,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate various risk metrics.
        
        Args:
            returns: DataFrame of asset returns
            weights: Array of asset weights
            benchmark_returns: Optional benchmark returns for comparison
            
        Returns:
            Dictionary with risk metrics
        """
        # Calculate portfolio returns
        portfolio_returns = returns.dot(weights)
        
        # Calculate annualization factor
        periods_per_year = self._estimate_periods_per_year(returns)
        
        # Calculate basic statistics
        mean_return = portfolio_returns.mean() * periods_per_year
        volatility = portfolio_returns.std() * np.sqrt(periods_per_year)
        
        # Calculate skewness and kurtosis
        skewness = stats.skew(portfolio_returns)
        kurtosis = stats.kurtosis(portfolio_returns)
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0
        
        # Calculate Sortino ratio (downside risk)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
        sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0
        
        # Calculate maximum drawdown
        drawdown_result = self.calculate_drawdown(returns, weights)
        max_drawdown = drawdown_result["max_drawdown"]
        
        # Calculate Calmar ratio
        calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Calculate beta and alpha if benchmark is provided
        beta = alpha = r_squared = tracking_error = information_ratio = None
        
        if benchmark_returns is not None:
            # Align dates
            aligned_returns = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
            if len(aligned_returns) > 0:
                portfolio_aligned = aligned_returns.iloc[:, 0]
                benchmark_aligned = aligned_returns.iloc[:, 1]
                
                # Calculate beta
                covariance = portfolio_aligned.cov(benchmark_aligned)
                benchmark_variance = benchmark_aligned.var()
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                # Calculate alpha
                benchmark_mean = benchmark_aligned.mean() * periods_per_year
                alpha = mean_return - beta * benchmark_mean
                
                # Calculate R-squared
                correlation = portfolio_aligned.corr(benchmark_aligned)
                r_squared = correlation ** 2
                
                # Calculate tracking error
                tracking_diff = portfolio_aligned - benchmark_aligned
                tracking_error = tracking_diff.std() * np.sqrt(periods_per_year)
                
                # Calculate information ratio
                information_ratio = (mean_return - benchmark_mean) / tracking_error if tracking_error > 0 else 0
        
        return {
            "mean_return": mean_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "beta": beta,
            "alpha": alpha,
            "r_squared": r_squared,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio
        }
    
    def perform_stress_test(
        self, 
        returns: pd.DataFrame, 
        weights: np.ndarray,
        scenarios: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform stress testing on a portfolio.
        
        Args:
            returns: DataFrame of asset returns
            weights: Array of asset weights
            scenarios: Dictionary mapping scenario names to dictionaries of asset return modifiers
            
        Returns:
            Dictionary mapping scenario names to dictionaries of stress test results
        """
        results = {}
        
        for scenario_name, scenario_modifiers in scenarios.items():
            # Apply scenario modifiers to historical returns
            scenario_returns = returns.copy()
            
            for asset, modifier in scenario_modifiers.items():
                if asset in scenario_returns.columns:
                    scenario_returns[asset] = scenario_returns[asset] * modifier
            
            # Calculate portfolio return under this scenario
            portfolio_return = scenario_returns.dot(weights).mean() * self._estimate_periods_per_year(returns)
            portfolio_volatility = scenario_returns.dot(weights).std() * np.sqrt(self._estimate_periods_per_year(returns))
            
            # Calculate VaR and CVaR
            var_result = self.calculate_var(scenario_returns, weights)
            cvar_result = self.calculate_cvar(scenario_returns, weights)
            
            # Calculate drawdown
            drawdown_result = self.calculate_drawdown(scenario_returns, weights)
            
            results[scenario_name] = {
                "return": portfolio_return,
                "volatility": portfolio_volatility,
                "var": var_result["var"],
                "cvar": cvar_result["cvar"],
                "max_drawdown": drawdown_result["max_drawdown"]
            }
        
        return results
    
    #def _estimate_periods_per_year(self, returns: pd.DataFrame) -> int:
        """
     #   Estimate the number of periods per year from the returns data.
        
      #  Args:
            returns: DataFrame of asset returns
            
        Returns:
            Estimated number of periods per year
        """
       # if returns.index.freq:
            # If the DataFrame has a frequency set
            
        #    if freq.name == 'D':
          #      return 252  # Trading days
         #   elif freq.name == 'W':
             
            #elif freq
