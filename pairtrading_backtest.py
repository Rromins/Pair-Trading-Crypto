import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

class PairTradingBacktest(object):
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
        self.get_data()

    def get_data(self):
        '''
        Get the data with yahoo finance api, and calculate the needed features. 
        '''
        asset1 = yf.download(tickers=self.ticker1, start=self.start, end=self.end)
        asset1.columns = ['close1', 'high1', 'low1', 'open1', 'volume1']
        asset2 = yf.download(tickers=self.ticker2, start=self.start, end=self.end)
        asset2.columns = ['close2', 'high2', 'low2', 'open2', 'volume2']
        
        df = pd.concat([asset1['close1'], asset2['close2']], axis=1)
        df['log_returns1'] = np.log(df['close1'] / df['close1'].shift(1))
        df['log_returns2'] = np.log(df['close2'] / df['close2'].shift(1))
        self.data = df

    def plot_data(self):
        '''
        Plot the stock price data.
        '''
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.data['close1'], color='gray', label='close asset 1')
        ax.plot(self.data['close2'], color='black', label='close asset 2')
        ax.legend()
        ax.set_title(f"{self.ticker1} and {self.ticker2} close prices")
        plt.plot()

    def get_date_price(self, bar):
        '''
        Return date and price of asset 1 and 2 for a given bar
        '''
        date = str(self.data.index[bar])
        price1 = self.data['close1'].iloc[bar]
        price2 = self.data['close2'].iloc[bar]
        return date, price1, price2
    
    def print_balance(self, bar):
        '''
        Print current cash balance info
        '''
        date, price1, price2 = self.get_date_price(bar)
        print(f"{date}, current balance: {self.amount:.2f}")

    def print_net_wealth(self, bar):
        '''
        Print current net wealth info
        '''
        date, price1, price2 = self.get_date_price(bar)
        net_wealth = self.units1*price1 + self.units2*price2 + self.amount
        print(f"{date}, current net wealth: {net_wealth:.2f}")

    def place_buy_order(self, bar, asset:int, units=None, amount=None):
        '''
        Place buy order for a given asset 1 or 2
        '''
        date, price1, price2 = self.get_date_price(bar)
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
                self.print_balance(bar)
                self.print_net_wealth(bar)
        else:
            print("Enter number 1 or 2 to choose the asset to buy.")

    def place_sell_order(self, bar, asset:int, units=None, amount=None):
        '''
        Place a sell order for a given asset 1 or 2
        '''
        date, price1, price2 = self.get_date_price(bar)
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
                self.print_balance(bar)
                self.print_net_wealth(bar)
        else:
            print("Enter number 1 or 2 to choose the asset to sell.")

    def close_out(self, bar):
        '''
        Closing out a long or short position
        '''
        date, price1, price2 = self.get_date_price(bar)
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
            perf = ((self.amount - self.initial_amount) / self.initial_amount*100)
            print(f"Net performance  [%] {perf:.2f}")
            print(f"Trades Executed  [#] {self.trades:.2f}")
            print("-"*55)

    def go_long(self, bar, asset:int, units=None, amount=None):
        # first check if there already a short position, and if yes liquidate it.
        if self.position == -1:
            self.place_buy_order(bar, asset, units=-self.units)
        
        if units:
            self.place_buy_order(bar, asset, units=units)
        elif amount:
            self.place_buy_order(bar, asset, amount=amount)
        else:
            print("Enter either units or an amount.")

    def go_short(self, bar, asset:int, units=None, amount=None):
        # first check if there already a long position, and if yes liquidate it.
        if self.position == 1:
            self.place_sell_order(bar, asset, units=self.units)
        
        if units:
            self.place_sell_order(bar, asset, units=units)
        elif amount:
            self.place_sell_order(bar, asset, amount=amount)
        else:
            print("Enter either units or an amount.")

    def cadf(self, y, x):
        '''
        Perform a linear regression of x on y, and a Augmented Dickey-Fuller test on the residuals to get the cointegration test p-value.
        Returns: alpha, beta 
        '''
        model = sm.OLS(y, sm.add_constant(x)).fit()
        alpha = model.params[0]
        beta = model.params[1]
        residuals = model.resid.values
        return alpha, beta, residuals, adfuller(residuals)[1]
    
    def dynamic_cadf(self, lags):
        dynamic_alpha = [np.nan]*(lags-1)
        dynamic_beta = [np.nan]*(lags-1)
        dynamic_resid = [np.nan]*(lags-1)
        dynamic_cadf_results = [np.nan]*(lags-1)
        for start in range(len(self.data)-lags+1):
            end = start + lags
            lags_a1 = self.data['close1'].iloc[start:end]
            lags_a2 = self.data['close2'].iloc[start:end]
            alpha, beta, residuals, cadf_results = self.cadf(y=lags_a1, x=lags_a2)
            dynamic_alpha.append(alpha)
            dynamic_beta.append(beta)
            dynamic_resid.append(residuals[-1])
            dynamic_cadf_results.append(cadf_results)
        return dynamic_cadf_results
    
    def dynamic_hedge_ratio(self, lags):
        dynamic_hr = [np.nan]*(lags-1)
        dynamic_resid_hr = [np.nan]*(lags-1)
        for start in range(len(self.data)-lags+1):
            end = start + lags
            lags_a1 = self.data['close1'].iloc[start:end]
            lags_a2 = self.data['close2'].iloc[start:end]
            model = sm.OLS(lags_a1, lags_a2).fit()
            beta = model.params[0]
            residuals = model.resid.values
            dynamic_hr.append(beta)
            dynamic_resid_hr.append(residuals[-1])
        return dynamic_hr, dynamic_resid_hr
    
    def run_pairTrading_strategy(self, lags, threshold):
        '''
        Run a pair trading strategy for 2 assets. 
        '''
        msg  = f"\n\nRunning pair trading strategy | "
        msg += f"Number of lags={lags} & thr={threshold}"
        msg += f"\nfixed costs {self.ftc} | "
        msg += f"proportional costs {self.ptc}"
        print(msg)
        print("-"*55)
        self.position = 0
        self.trades = 0
        self.amount = self.initial_amount

        self.data['dynamic_cadf_results'] = self.dynamic_cadf(lags)
        self.data['dynamic_hr'], self.data['dynamic_hr_resid'] = self.dynamic_hedge_ratio(lags)
        print(self.data)
        self.data['signals'] = (self.data['dynamic_hr_resid'] - self.data['dynamic_hr_resid'].mean()) / self.data['dynamic_hr_resid'].std()
        
        for bar in range(lags, len(self.data)):
            if self.position == 0:
                if (self.data['dynamic_cadf_results'].iloc[bar] < 0.05 and self.data['signals'].iloc[bar] < -threshold):
                    amount = self.amount*0.2
                    self.go_long(bar, asset=1, amount=amount)
                    self.go_short(bar, asset=2, amount=amount)
                    self.position = 1
                elif (self.data['dynamic_cadf_results'].iloc[bar] < 0.05 and self.data['signals'].iloc[bar] > threshold):
                    amount = self.amount*0.2
                    self.go_long(bar, asset=2, amount=amount)
                    self.go_short(bar, asset=1, amount=amount)
                    self.position = -1
            elif self.position == 1:
                if (self.data['signals'].iloc[bar] >= 0):
                    self.place_sell_order(bar, asset=1, units=self.units1)
                    self.place_buy_order(bar, asset=2, units=self.units2)
                    self.position = 0
            elif self.position == -1:
                if (self.data['signals'].iloc[bar] <= 0):
                    self.place_sell_order(bar, asset=2, units=self.units2)
                    self.place_buy_order(bar, asset=1, units=self.units1)
                    self.position = 0
        self.close_out(bar)

    def plot_cadf_results(self):
        '''
        Plot the p-value of the dynamic CADF test. 
        '''
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.data['dynamic_cadf_results'], c='gray', label='Dynamic CADF p-value')
        ax.axhline(y=0.05, ls='--', c='blue', label='p-value')
        ax.legend()
        plt.show()

    def plot_time_series(self, threshold):
        '''
        Plot the time series used to compute the signals
        '''
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.data['signals'], c='gray', label='Time series')
        ax.axhline(y=threshold, ls='--', c='blue', label='threshold')
        ax.axhline(y=-threshold, ls='--', c='blue', label='threshold')
        ax.legend()
        plt.show()

if __name__ == '__main__':
    start = '2020-01-01'
    end = '2025-05-01'
    pairTrading_strategy = PairTradingBacktest(ticker1='BTC-USD', ticker2='ETH-USD', start=start, end=end, amount=1000)
    pairTrading_strategy.run_pairTrading_strategy(lags=31, threshold=2)
    pairTrading_strategy.plot_cadf_results()
    pairTrading_strategy.plot_time_series(threshold=2)