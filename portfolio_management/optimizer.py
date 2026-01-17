import numpy as np
import pandas as pd
import scipy.optimize as sco
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """Portfolio optimization using modern portfolio theory and extensions."""
    
    def __init__(self):
        """Initialize the portfolio optimizer."""
        pass
    
    def calculate_portfolio_performance(
        self, 
        returns: pd.DataFrame, 
        weights: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.
        
        Args:
            returns: DataFrame of asset returns
            weights: Array of asset weights
            
        Returns:
            Dictionary of performance metrics
        """
        # Annualization factors
        periods_per_year = self._estimate_periods_per_year(returns)
        
        # Calculate portfolio return
        portfolio_return = np.sum(returns.mean() * weights) * periods_per_year
        
        # Calculate portfolio volatility
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(returns.cov() * periods_per_year, weights))
        )
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            "return": portfolio_return,
            "volatility": portfolio_volatility,
            "sharpe_ratio": sharpe_ratio
        }
    
    def _estimate_periods_per_year(self, returns: pd.DataFrame) -> int:
        """
        Estimate the number of periods per year from the returns data.
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            Estimated number of periods per year
        """
        if returns.index.freq:
            # If the DataFrame has a frequency set
            freq = returns.index.freq
            if freq.name == 'D':
                return 252  # Trading days
            elif freq.name == 'W':
                return 52  # Weeks
            elif freq.name == 'M':
                return 12  # Months
            elif freq.name == 'Q':
                return 4   # Quarters
            elif freq.name == 'Y':
                return 1   # Years
        
        # Try to infer from the date range and number of observations
        if isinstance(returns.index, pd.DatetimeIndex):
            date_range = returns.index.max() - returns.index.min()
            days = date_range.days
            observations = len(returns)
            
            if days > 0:
                avg_days_between_observations = days / observations
                
                if avg_days_between_observations < 2:  # Daily data
                    return 252
                elif avg_days_between_observations < 10:  # Weekly data
                    return 52
                elif avg_days_between_observations < 45:  # Monthly data
                    return 12
                elif avg_days_between_observations < 120:  # Quarterly data
                    return 4
                else:  # Annual data
                    return 1
        
        # Default to daily if we can't determine
        logger.warning("Could not determine frequency of returns data. Assuming daily (252 periods per year).")
        return 252
    
    def _portfolio_volatility(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        """
        Calculate portfolio volatility.
        
        Args:
            weights: Array of asset weights
            returns: DataFrame of asset returns
            
        Returns:
            Portfolio volatility
        """
        periods_per_year = self._estimate_periods_per_year(returns)
        return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * periods_per_year, weights)))
    
    def _portfolio_return(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        """
        Calculate portfolio return.
        
        Args:
            weights: Array of asset weights
            returns: DataFrame of asset returns
            
        Returns:
            Portfolio return
        """
        periods_per_year = self._estimate_periods_per_year(returns)
        return np.sum(returns.mean() * weights) * periods_per_year
    
    def _portfolio_sharpe(self, weights: np.ndarray, returns: pd.DataFrame, risk_free_rate: float = 0) -> float:
        """
        Calculate portfolio Sharpe ratio.
        
        Args:
            weights: Array of asset weights
            returns: DataFrame of asset returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Portfolio Sharpe ratio
        """
        portfolio_return = self._portfolio_return(weights, returns)
        portfolio_volatility = self._portfolio_volatility(weights, returns)
        
        return (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
    
    def _negative_sharpe(self, weights: np.ndarray, returns: pd.DataFrame, risk_free_rate: float = 0) -> float:
        """
        Calculate negative Sharpe ratio for minimization.
        
        Args:
            weights: Array of asset weights
            returns: DataFrame of asset returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Negative portfolio Sharpe ratio
        """
        return -self._portfolio_sharpe(weights, returns, risk_free_rate)
    
    def optimize_sharpe_ratio(
        self, 
        returns: pd.DataFrame, 
        risk_free_rate: float = 0,
        constraints: List[Dict[str, Any]] = None,
        bounds: List[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Optimize portfolio weights to maximize Sharpe ratio.
        
        Args:
            returns: DataFrame of asset returns
            risk_free_rate: Risk-free rate
            constraints: List of optimization constraints
            bounds: List of (min, max) tuples for each asset weight
            
        Returns:
            Dictionary with optimized weights and performance metrics
        """
        n_assets = len(returns.columns)
        
        # Initial guess: equal weights
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Default bounds: 0 to 1 for each asset
        if bounds is None:
            bounds = [(0, 1) for _ in range(n_assets)]
        
        # Default constraint: weights sum to 1
        if constraints is None:
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Optimize
        result = sco.minimize(
            self._negative_sharpe,
            initial_weights,
            args=(returns, risk_free_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result['success']:
            logger.warning(f"Optimization failed: {result['message']}")
        
        # Get the optimized weights
        weights = result['x']
        
        # Calculate performance metrics
        performance = self.calculate_portfolio_performance(returns, weights)
        
        # Create result dictionary
        asset_weights = {returns.columns[i]: weights[i] for i in range(n_assets)}
        
        return {
            "weights": asset_weights,
            "performance": performance,
            "optimization_success": result['success'],
            "optimization_message": result['message']
        }
    
    def optimize_minimum_volatility(
        self, 
        returns: pd.DataFrame,
        constraints: List[Dict[str, Any]] = None,
        bounds: List[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Optimize portfolio weights to minimize volatility.
        
        Args:
            returns: DataFrame of asset returns
            constraints: List of optimization constraints
            bounds: List of (min, max) tuples for each asset weight
            
        Returns:
            Dictionary with optimized weights and performance metrics
        """
        n_assets = len(returns.columns)
        
        # Initial guess: equal weights
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Default bounds: 0 to 1 for each asset
        if bounds is None:
            bounds = [(0, 1) for _ in range(n_assets)]
        
        # Default constraint: weights sum to 1
        if constraints is None:
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Optimize
        result = sco.minimize(
            self._portfolio_volatility,
            initial_weights,
            args=(returns,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result['success']:
            logger.warning(f"Optimization failed: {result['message']}")
        
        # Get the optimized weights
        weights = result['x']
        
        # Calculate performance metrics
        performance = self.calculate_portfolio_performance(returns, weights)
        
        # Create result dictionary
        asset_weights = {returns.columns[i]: weights[i] for i in range(n_assets)}
        
        return {
            "weights": asset_weights,
            "performance": performance,
            "optimization_success": result['success'],
            "optimization_message": result['message']
        }
    
    def optimize_target_return(
        self, 
        returns: pd.DataFrame,
        target_return: float,
        constraints: List[Dict[str, Any]] = None,
        bounds: List[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Optimize portfolio weights to minimize volatility for a target return.
        
        Args:
            returns: DataFrame of asset returns
            target_return: Target portfolio return
            constraints: List of optimization constraints
            bounds: List of (min, max) tuples for each asset weight
            
        Returns:
            Dictionary with optimized weights and performance metrics
        """
        n_assets = len(returns.columns)
        
        # Initial guess: equal weights
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Default bounds: 0 to 1 for each asset
        if bounds is None:
            bounds = [(0, 1) for _ in range(n_assets)]
        
        # Default constraints: weights sum to 1 and return equals target
        if constraints is None:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: self._portfolio_return(x, returns) - target_return}
            ]
        
        # Optimize
        result = sco.minimize(
            self._portfolio_volatility,
            initial_weights,
            args=(returns,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result['success']:
            logger.warning(f"Optimization failed: {result['message']}")
        
        # Get the optimized weights
        weights = result['x']
        
        # Calculate performance metrics
        performance = self.calculate_portfolio_performance(returns, weights)
        
        # Create result dictionary
        asset_weights = {returns.columns[i]: weights[i] for i in range(n_assets)}
        
        return {
            "weights": asset_weights,
            "performance": performance,
            "optimization_success": result['success'],
            "optimization_message": result['message']
        }
    
    def generate_efficient_frontier(
        self, 
        returns: pd.DataFrame,
        n_points: int = 50,
        constraints: List[Dict[str, Any]] = None,
        bounds: List[Tuple[float, float]] = None
    ) -> pd.DataFrame:
        """
        Generate the efficient frontier.
        
        Args:
            returns: DataFrame of asset returns
            n_points: Number of points on the efficient frontier
            constraints: List of optimization constraints
            bounds: List of (min, max) tuples for each asset weight
            
        Returns:
            DataFrame with efficient frontier points (return, volatility, sharpe)
        """
        # Find the minimum volatility portfolio
        min_vol_result = self.optimize_minimum_volatility(returns, constraints, bounds)
        min_return = min_vol_result["performance"]["return"]
        
        # Find the maximum return portfolio (100% in the asset with highest return)
        max_return_asset = returns.mean().idxmax()
        max_return = returns.mean()[max_return_asset] * self._estimate_periods_per_year(returns)
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, n_points)
        
        # Calculate efficient frontier
        efficient_frontier = []
        
        for target_return in target_returns:
            try:
                result = self.optimize_target_return(returns, target_return, constraints, bounds)
                if result["optimization_success"]:
                    efficient_frontier.append({
                        "return": result["performance"]["return"],
                        "volatility": result["performance"]["volatility"],
                        "sharpe_ratio": result["performance"]["sharpe_ratio"]
                    })
            except Exception as e:
                logger.warning(f"Failed to optimize for target return {target_return}: {e}")
        
        return pd.DataFrame(efficient_frontier)
