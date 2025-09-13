"""
Backtest base class for pair trading strategy.
"""

import numpy as np
import pandas as pd
import yfinance as yf

class BacktestBase():
    """
    Base class for backtesting a pair trading strategy between two assets.

    This class implements data retrieval, trade execution logic, and performance evaluation.

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
    """
    def __init__(self,
                 ticker1: str,
                 ticker2:str,
                 start: str,
                 end: str,
                 amount: float,
                 ftc=0.0,
                 ptc=0.0,
                 verbose=True):
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.start = start
        self.end = end
        self.initial_amount = amount
        self.amount = amount
        self.ftc = ftc
        self.ptc = ptc
        self.units1 = 0
        self.units2 = 0
        self.position = 0
        self.trades = 0
        self.verbose = verbose
        self._get_data()

    def _get_data(self):
        """
        Download and preprocess historical price data for the two assets.

        Retrieves daily OHLCV data from Yahoo Finance, renames columns,
        and computes log returns for each asset.

        Returns
        -------
        None
            Updates the `self.data` attribute with prices and returns.
        """
        asset1 = yf.download(tickers=self.ticker1, start=self.start, end=self.end)
        asset1.columns = ['close1', 'high1', 'low1', 'open1', 'volume1']
        asset2 = yf.download(tickers=self.ticker2, start=self.start, end=self.end)
        asset2.columns = ['close2', 'high2', 'low2', 'open2', 'volume2']

        df = pd.concat([asset1['close1'], asset2['close2']], axis=1)
        df['log_returns1'] = np.log(df['close1'] / df['close1'].shift(1))
        df['log_returns2'] = np.log(df['close2'] / df['close2'].shift(1))
        self.data = df

    def _get_date_price(self, candle):
        """
        Get the date and closing prices of both assets at a given bar index.

        Parameters
        ----------
        candle : int
            Row index in the historical DataFrame.

        Returns
        -------
        tuple
            (date: str, price1: float, price2: float)
        """
        date = str(self.data.index[candle])
        price1 = self.data['close1'].iloc[candle]
        price2 = self.data['close2'].iloc[candle]
        return date, price1, price2

    def _print_balance(self, candle):
        """
        Print the current available cash balance.

        Parameters
        ----------
        candle : int
            Row index in the historical DataFrame.

        Returns
        -------
        None
        """
        date = self._get_date_price(candle)[0]
        print(f"{date}, current balance: {self.amount:.2f}")

    def _print_net_wealth(self, candle):
        """
        Print the current net wealth (cash + asset holdings).

        Parameters
        ----------
        candle : int
            Row index in the historical DataFrame.

        Returns
        -------
        None
        """
        date, price1, price2 = self._get_date_price(candle)
        net_wealth = self.units1*price1 + self.units2*price2 + self.amount
        print(f"{date}, current net wealth: {net_wealth:.2f}")

    def _place_buy_order(self, candle, asset:int, units=None, amount=None):
        """
        Execute a buy order for asset 1 or 2.

        Parameters
        ----------
        candle : int
            Row index in the historical DataFrame.
        asset : int
            1 to buy asset1, 2 to buy asset2.
        units : float, optional
            Number of units to purchase. If None, computed from `amount`.
        amount : float, optional
            Dollar amount to invest. Ignored if `units` is provided.

        Returns
        -------
        None
        """
        date, price1, price2 = self._get_date_price(candle)
        if asset in [1, 2]:
            price = price1 if asset == 1 else price2

            if units is None:
                units = float(amount / price)
            self.amount -= (units*price) * (1+self.ptc) + self.ftc

            if asset == 1:
                self.units1 += units
            else:
                self.units2 += units
            self.trades += 1

            if self.verbose:
                print(f"{date}, buying {units} units at {price:.2f}")
                self._print_balance(candle)
                self._print_net_wealth(candle)
        else:
            raise ValueError("Enter number 1 or 2 to choose the asset to buy.")

    def _place_sell_order(self, candle, asset:int, units=None, amount=None):
        """
        Execute a sell order for asset 1 or 2.

        Parameters
        ----------
        candle : int
            Row index in the historical DataFrame.
        asset : int
            1 to sell asset1, 2 to sell asset2.
        units : float, optional
            Number of units to sell. If None, computed from `amount`.
        amount : float, optional
            Dollar amount to liquidate. Ignored if `units` is provided.

        Returns
        -------
        None
        """
        date, price1, price2 = self._get_date_price(candle)
        if asset in [1, 2]:
            price = price1 if asset == 1 else price2

            if units is None:
                units = float(amount / price)
            self.amount += (units*price) * (1-self.ptc) - self.ftc

            if asset == 1:
                self.units1 -= units
            else:
                self.units2 -= units
            self.trades += 1

            if self.verbose:
                print(f"{date}, selling {units} units at {price:.2f}")
                self._print_balance(candle)
                self._print_net_wealth(candle)
        else:
            raise ValueError("Enter number 1 or 2 to choose the asset to sell.")

    def _go_long(self, candle, asset:int, units=None, amount=None):
        """
        Enter a long position in one asset (buy), closing shorts if necessary.

        Parameters
        ----------
        candle : int
            Row index in the historical DataFrame.
        asset : int
            1 = go long asset1, 2 = go long asset2.
        units : float, optional
            Number of units to buy. If None, computed from `amount`.
        amount : float, optional
            Dollar amount to invest.

        Returns
        -------
        None
        """
        # first check if there already a short position, and if yes liquidate it.
        if self.position == -1:
            self._place_buy_order(candle, asset, units=-units)

        if units:
            self._place_buy_order(candle, asset, units=units)
        elif amount:
            self._place_buy_order(candle, asset, amount=amount)
        else:
            raise ValueError("Enter either units or an amount.")

    def _go_short(self, candle, asset:int, units=None, amount=None):
        """
        Enter a short position in one asset (sell), closing longs if necessary.

        Parameters
        ----------
        candle : int
            Row index in the historical DataFrame.
        asset : int
            1 = short asset1, 2 = short asset2.
        units : float, optional
            Number of units to short. If None, computed from `amount`.
        amount : float, optional
            Dollar amount to short.

        Returns
        -------
        None
        """
        # first check if there already a long position, and if yes liquidate it.
        if self.position == 1:
            self._place_sell_order(candle, asset, units=units)

        if units:
            self._place_sell_order(candle, asset, units=units)
        elif amount:
            self._place_sell_order(candle, asset, amount=amount)
        else:
            raise ValueError("Enter either units or an amount.")

    def _close_out(self, candle):
        """
        Liquidate all positions and compute final performance metrics.

        Parameters
        ----------
        candle : int
            Row index in the historical DataFrame.

        Returns
        -------
        None
        """
        date, price1, price2 = self._get_date_price(candle)
        self.amount += self.units1 * price1
        self.amount += self.units2 * price2
        self.units1 = 0
        self.units2 = 0
        self.trades += 1

        if self.verbose:
            print(f"{date}, inventory {self.units1} units at {price1:.2f}")
            print(f"{date}, inventory {self.units2} units at {price2:.2f}")
            print("-"*55)
            print(f"Final balance  [$] {self.amount:.2f}")
            perf = (self.amount - self.initial_amount) / self.initial_amount*100
            print(f"Net performance  [%] {perf:.2f}")
            print(f"Trades Executed  [#] {self.trades:.2f}")
            print("-"*55)
