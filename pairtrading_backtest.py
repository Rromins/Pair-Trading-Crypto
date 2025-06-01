import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

class PairTradingBacktest(object):
    def __init__(self,
                 ticker: str,
                 start: str,
                 end: str,
                 amount: float,
                 ftc=0.0,
                 ptc=0.0,
                 verbose=True):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.initial_amount = amount
        self.amount = amount
        self.ftc = ftc
        self.ptc = ptc
        self.units = 0
        self.position = 0
        self.trades = 0
        self.verbose = verbose
        self.get_data()

    def get_data(self):
        '''
        Get the data with yahoo finance api, and calculate the needed features. 
        '''
        df = yf.download(tickers=self.ticker1, start=self.start, end=self.end)
        df.columns = ['close', 'high', 'low', 'open', 'volume']
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        self.data = df

    def plot_data(self):
        '''
        Plot the stock price data.
        '''
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.data['close'], color='gray', label='close asset')
        ax.legend()
        ax.set_title(f"{self.ticker} close prices")
        plt.plot()

    def get_date_price(self, bar):
        '''
        Return date and price for a given bar
        '''
        date = str(self.data.index[bar])
        price = self.data['close'].iloc[bar]
        return date, price
    
    def print_balance(self, bar):
        '''
        Print current cash balance info
        '''
        date, price = self.get_date_price(bar)
        print(f"{date}, current balance: {self.amount:.2f}")

    def print_net_wealth(self, bar):
        '''
        Print current net wealth info
        '''
        date, price = self.get_date_price(bar)
        net_wealth = self.units * price + self.amount
        print(f"{date}, current net wealth: {net_wealth:.2f}")

    def place_buy_order(self, bar, units=None, amount=None):
        '''
        Place buy order
        '''
        date, price = self.get_date_price(bar)
        if units is None:
            units = float(amount / price)
        self.amount -= (units * price) * (1+self.ptc) + self.ftc
        self.units += units
        self.trades += 1
        if self.verbose:
            print(f"{date}, buying {units} units at {price:.2f}")
            self.print_balance(bar)
            self.print_net_wealth(bar)

    def place_sell_order(self, bar, units=None, amount=None):
        '''
        Place a sell order
        '''
        date, price = self.get_date_price(bar)
        if units is None:
            units = float(amount / price)
        self.amount += (units*price) * (1-self.ptc) - self.ftc
        self.units -= units
        self.trades += 1
        if self.verbose:
            print(f"{date}, selling {units} units at {price:.2f}")
            self.print_balance(bar)
            self.print_net_wealth(bar)

    def close_out(self, bar):
        '''
        Closing out a long or short position
        '''
        date, price = self.get_date_price(bar)
        self.amount += self.units * price
        self.units = 0
        self.trades += 1
        if self.verbose:
            print(f"{date}, inventory {self.units} units at {price:.2f}")
            print("-"*55)
            print(f"Final balance  [$] {self.amount:.2f}")
            perf = ((self.amount - self.initial_amount) / self.initial_amount*100)
            print(f"Net performance  [%] {perf:.2f}")
            print(f"Trades Executed  [#] {self.trades:.2f}")
            print("-"*55)

    def go_long(self, bar, units=None, amount=None):
        # first check if there already a short position, and if yes liquidate it.
        if self.position == -1:
            self.place_buy_order(bar, units=-self.units)
        
        if units:
            self.place_buy_order(bar, units=units)
        elif amount:
            self.place_buy_order(bar, amount=amount)
        else:
            print("Enter either units or an amount.")

    def go_short(self, bar, units=None, amount=None):
        # first check if there already a long position, and if yes liquidate it.
        if self.position == 1:
            self.place_sell_order(bar, units=self.units)
        
        if units:
            self.place_sell_order(bar, units=units)
        elif amount:
            self.place_sell_order(bar, amount=amount)
        else:
            print("Enter either units or an amount.")

    # ------------------------------------------------------------------------

    def cadf(self):
        '''
        Perform a linear regression of x on y, and a Augmented Dickey-Fuller test on the residuals to get the cointegration test p-value.
        Returns: alpha, beta 
        '''
        df = self.data.copy()
        model = sm.OLS(df['close1'], sm.add_constant(df['close2']).fit())
        alpha = model.params[0]
        beta = model.params[1]
        residuals = model.resid.values
        return alpha, beta, residuals, adfuller(residuals)[1]
    
    def dynamic_cadf(self):
        df = self.data.copy()
        total_len = max(len(df['close1']), len(df['close2']))
        dynamic_alpha = [np.nan]*(self.lags-1)
        dynamic_beta = [np.nan]*(self.lags-1)
        dynamic_resid = [np.nan]*(self.lags-1)
        dynamic_cadf_results = [np.nan]*(self.lags-1)
        for start in range(total_len-self.lags+1):
            end = start + self.lags
            lags_a1 = df['close1'].iloc[start:end]
            lags_a2 = df['close2'].iloc[start:end]
            alpha, beta, residuals, cadf_results = self.cadf()
            dynamic_alpha.append(alpha)
            dynamic_beta.append(beta)
            dynamic_resid.append(residuals[-1])
            dynamic_cadf_results.append(cadf_results)
        return dynamic_alpha, dynamic_beta, dynamic_resid, dynamic_cadf_results