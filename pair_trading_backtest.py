"""
Backtest of a pair trading strategy for 2 assets.
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from backtest_base import BacktestBase

class PairTradingBacktest(BacktestBase):
    """
    Backtest a statistical arbitrage (pairs trading) strategy between two assets.

    This class implements data retrieval, cointegration testing, hedge ratio estimation,
    trade execution logic, and performance evaluation for a pairs trading strategy.

    Attributes
    ----------
    ticker1 : str
        Symbol of the first asset.
    ticker2 : str
        Symbol of the second asset.
    start : str
        Start date (format 'YYYY-MM-DD') for historical data.
    end : str
        End date (format 'YYYY-MM-DD') for historical data.
    initial_amount : float
        Initial portfolio capital in dollars.
    amount : float
        Current cash balance.
    ftc : float
        Fixed transaction cost per trade.
    ptc : float
        Proportional transaction cost (as fraction of trade value).
    units1 : float
        Number of units held of asset 1.
    units2 : float
        Number of units held of asset 2.
    position : int
        Current portfolio position: 0 = no position, 
            1 = long asset1/short asset2, -1 = long asset2/short asset1.
    trades : int
        Total number of trades executed.
    verbose : bool
        If True, prints trading and portfolio information.
    data : pd.DataFrame
        Price series, log returns, signals, and strategy results.

    Methods
    -------
    run_pair_trading_strategy(lags, threshold)
        Execute the pairs trading strategy over the historical data.
    plot_cadf_results()
        Plot the p-values from the dynamic cointegration tests.
    plot_time_series(threshold)
        Plot the standardized residuals used for generating trading signals.
    """
    def _cadf(self, y, x):
        """
        Perform cointegration test using OLS regression and Augmented Dickey-Fuller.

        Parameters
        ----------
        y : pd.Series
            Dependent variable (usually asset1).
        x : pd.Series
            Independent variable (usually asset2).

        Returns
        -------
        tuple
            (alpha: float, beta: float, residuals: np.ndarray, pvalue: float)
        """
        model = sm.OLS(y, sm.add_constant(x)).fit()
        alpha = model.params[0]
        beta = model.params[1]
        residuals = model.resid.values
        return alpha, beta, residuals, adfuller(residuals)[1]

    def _dynamic_cadf(self, lags):
        """
        Compute rolling CADF cointegration test over a moving window.

        Parameters
        ----------
        lags : int
            Number of observations per rolling window.

        Returns
        -------
        list
            Rolling p-values of the CADF test.
        """
        dynamic_alpha = [np.nan]*(lags-1)
        dynamic_beta = [np.nan]*(lags-1)
        dynamic_resid = [np.nan]*(lags-1)
        dynamic_cadf_results = [np.nan]*(lags-1)

        for start_i in range(len(self.data)-lags+1):
            end_i = start_i + lags
            lags_a1 = self.data['close1'].iloc[start_i:end_i]
            lags_a2 = self.data['close2'].iloc[start_i:end_i]

            alpha, beta, residuals, cadf_results = self._cadf(y=lags_a1, x=lags_a2)

            dynamic_alpha.append(alpha)
            dynamic_beta.append(beta)
            dynamic_resid.append(residuals[-1])
            dynamic_cadf_results.append(cadf_results)

        return dynamic_cadf_results

    def _dynamic_hedge_ratio(self, lags):
        """
        Compute rolling hedge ratios using OLS regression.

        Parameters
        ----------
        lags : int
            Number of observations per rolling window.

        Returns
        -------
        tuple
            (dynamic_hr: list of float, dynamic_resid_hr: list of float)
        """
        dynamic_hr = [np.nan]*(lags-1)
        dynamic_resid_hr = [np.nan]*(lags-1)
        for start_i in range(len(self.data)-lags+1):
            end_i = start_i + lags
            lags_a1 = self.data['close1'].iloc[start_i:end_i]
            lags_a2 = self.data['close2'].iloc[start_i:end_i]

            model = sm.OLS(lags_a1, lags_a2).fit()
            beta = model.params[0]
            residuals = model.resid.values

            dynamic_hr.append(beta)
            dynamic_resid_hr.append(residuals[-1])

        return dynamic_hr, dynamic_resid_hr

    def run_pair_trading_strategy(self, lags, threshold):
        """
        Execute the pair trading backtest with dynamic hedge ratios and signals.

        Parameters
        ----------
        lags : int
            Lookback window length for CADF and hedge ratio estimation.
        threshold : float
            Entry threshold for z-score signal.

        Returns
        -------
        None
            Executes trades, updates portfolio value, and prints performance.
        """
        msg  = "\n\nRunning pair trading strategy | "
        msg += f"Number of lags={lags} & thr={threshold}"
        msg += f"\nfixed costs {self.ftc} | "
        msg += f"proportional costs {self.ptc}"
        print(msg)
        print("-"*55)
        self.position = 0
        self.trades = 0
        self.amount = self.initial_amount

        self.data['dynamic_cadf_results'] = self._dynamic_cadf(lags)
        self.data['dynamic_hr'], self.data['dynamic_hr_resid'] = self._dynamic_hedge_ratio(lags)
        print(self.data)
        self.data['signals'] = (self.data['dynamic_hr_resid'] - self.data['dynamic_hr_resid'].mean()) / self.data['dynamic_hr_resid'].std()

        for candle in range(lags, len(self.data)):

            if self.position == 0:
                if (self.data['dynamic_cadf_results'].iloc[candle] < 0.05
                    and self.data['signals'].iloc[candle] < -threshold):
                    amount = self.amount*0.2
                    self._go_long(candle, asset=1, amount=amount)
                    self._go_short(candle, asset=2, amount=amount)
                    self.position = 1

                elif (self.data['dynamic_cadf_results'].iloc[candle] < 0.05
                      and self.data['signals'].iloc[candle] > threshold):
                    amount = self.amount*0.2
                    self._go_long(candle, asset=2, amount=amount)
                    self._go_short(candle, asset=1, amount=amount)
                    self.position = -1

            elif self.position == 1:
                if self.data['signals'].iloc[candle] >= 0:
                    self._place_sell_order(candle, asset=1, units=self.units1)
                    self._place_buy_order(candle, asset=2, units=self.units2)
                    self.position = 0

            elif self.position == -1:
                if self.data['signals'].iloc[candle] <= 0:
                    self._place_sell_order(candle, asset=2, units=self.units2)
                    self._place_buy_order(candle, asset=1, units=self.units1)
                    self.position = 0

        self._close_out(candle)

    def plot_cadf_results(self):
        """
        Plot rolling CADF p-values and significance threshold.

        Returns
        -------
        None
        """
        plt.plot(self.data['dynamic_cadf_results'], c='gray', label='Dynamic CADF p-value')
        plt.axhline(y=0.05, ls='--', c='blue', label='p-value')
        plt.legend()
        plt.show()

    def plot_time_series(self, threshold):
        """
        Plot standardized residuals (signals) with trading thresholds.

        Parameters
        ----------
        threshold : float
            Threshold used to generate long/short signals.

        Returns
        -------
        None
        """
        plt.plot(self.data['signals'], c='gray', label='Time series')
        plt.axhline(y=threshold, ls='--', c='blue', label='threshold')
        plt.axhline(y=-threshold, ls='--', c='blue', label='threshold')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    start_strat = '2020-01-01'
    end_strat = '2025-05-01'
    pair_trading_strategy = PairTradingBacktest(ticker1='BTC-USD', ticker2='ETH-USD', start=start_strat, end=end_strat, amount=1000)
    pair_trading_strategy.run_pair_trading_strategy(lags=31, threshold=2)
    pair_trading_strategy.plot_cadf_results()
    pair_trading_strategy.plot_time_series(threshold=2)

    # With transaction costs
    print("\n\nWith transaction costs")
    pair_trading_strategy_wtr = PairTradingBacktest(ticker1='BTC-USD', ticker2='ETH-USD', start=start_strat, end=end_strat, amount=1000, ftc=4.0, ptc=0.03)
    pair_trading_strategy_wtr.run_pair_trading_strategy(lags=31, threshold=2)
    pair_trading_strategy_wtr.plot_cadf_results()
    pair_trading_strategy_wtr.plot_time_series(threshold=2)
