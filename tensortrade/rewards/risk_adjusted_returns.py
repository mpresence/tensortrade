class RiskAdjustedReturns(RewardScheme):
    """A reward scheme that rewards the agent for increasing its net worth, while penalizing more volatile strategies.
    """

    def __init__(self,
                 return_algorithm: str = 'sharpe',
                 risk_free_rate: float = 0.,
                 target_returns: float = 0.,
                 window_size: int = 1):
        """
        Args:
            return_algorithm (optional): The risk-adjusted return metric to use. Options are 'sharpe' and 'sortino'. Defaults to 'sharpe'.
            risk_free_rate (optional): The risk free rate of returns to use for calculating metrics. Defaults to 0.
            target_returns (optional): The target returns per period for use in calculating the sortino ratio. Default to 0.
        """
        algorithm = self.default('return_algorithm', return_algorithm)

        self._return_algorithm = self._return_algorithm_from_str(algorithm)
        self._risk_free_rate = self.default('risk_free_rate', risk_free_rate)
        self._target_returns = self.default('target_returns', target_returns)
        self._window_size = self.default('window_size', window_size)

    def _return_algorithm_from_str(self, algorithm_str: str) -> Callable[[pd.DataFrame], float]:
        assert algorithm_str in ['sharpe', 'sortino', 'omega']

        if algorithm_str == 'sharpe':
            return self._sharpe_ratio
        elif algorithm_str == 'sortino':
            return self._sortino_ratio
        elif algorithm_str == 'omega':
            return self._omega_ratio

    def _sharpe_ratio(self, returns: pd.Series) -> float:
        """Return the sharpe ratio for a given series of a returns.
        References:
            - https://en.wikipedia.org/wiki/Sharpe_ratio
        """
        return (np.mean(returns) - self._risk_free_rate + 1E-9) / (np.std(returns) + 1E-9)

    def _sortino_ratio(self, returns: pd.Series) -> float:
        """Return the sortino ratio for a given series of a returns.
        References:
            - https://en.wikipedia.org/wiki/Sortino_ratio
        """
        downside_returns = returns.copy()
        downside_returns[returns < self._target_returns] = returns ** 2

        expected_return = np.mean(returns)
        downside_std = np.sqrt(np.std(downside_returns))

        return (expected_return - self._risk_free_rate + 1E-9) / (downside_std + 1E-9)

    def _omega_ratio(self, returns: pd.Series) -> float:
        """Return the sharpe ratio for a given series of a returns.
        References:
            - https://en.wikipedia.org/wiki/Omega_ratio
        """
        return omega_ratio(returns, risk_free=self._risk_free_rate, required_return=self._target_returns, annualization=1)

    def get_reward(self, portfolio: 'Portfolio') -> float:
        """Return the reward corresponding to the selected risk-adjusted return metric."""
        returns = portfolio.performance['net_worth'][-(self._window_size + 1):].pct_change().dropna()
        risk_adjusted_return = self._return_algorithm(returns)

        return risk_adjusted_return
