import yfinance as yf
import backtrader as bt
import matplotlib.pyplot as plt


class SmaCross(bt.SignalStrategy):
    def __init__(self):
        sma1 = bt.ind.SMA(period=160)
        sma2 = bt.ind.SMA(period=400)
        crossover = bt.ind.CrossOver(sma1, sma2)
        self.signal_add(bt.SIGNAL_LONG, crossover)

class HLD(bt.SignalStrategy):
    def __init__(self):
        hilo_diff = self.data.high - self.data.low


class MyStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Initialize the indicators
   #     self.sma = bt.ind.SMA(self.data,period=15)
        self.wlmr = bt.ind.WilliamsR(self.data)
        self.LH = bt.ind.FindLastIndexHighest(self.data.high,period=10)
        self.LL = bt.ind.FindLastIndexLowest(self.data.close,period=10)
                # Initialize lows in a row tracker
        self.lowsinarow = 0
        self.previous_low = None

    def next(self):

        # Set conditions.
        buy_size = 100
        sell_size = 100
        # Signal logic is placed here inside the next method

        if self.previous_low is None or (self.data.low[0] < self.previous_low) or (self.previous_low *1.05)  < self.data.high[0]:
            self.previous_low = self.data.low[0];

        # Define the buy signal condition
#        buy_sig = (self.lowsinarow > 3) and (self.data.close[0] *1.005 > self.previous_low) and (self.data.close[0] > self.data.open[0])
        buy_sig = self.LH > 3 and self.LL > 2 #and self.data.close[0] > self.data.close[-int(self.LL)] * 1.15

        # Define the buy signal condition
        sell_sig = self.wlmr > -5


     # Check for consecutive lower lows
        if self.previous_low is not None and self.data.close[0] < self.data.open[0]:
            self.lowsinarow += 1
        else:
            self.lowsinarow = 0  # Reset if not lower
       


        # Execute the buy order if the buy signal is true
        if buy_sig:
            self.buy(size=buy_size)
            print('BUY CREATE',self.data.close[0],self.LH.index,self.LL.index)
            print(self.broker.getposition(self.data))
        if sell_sig and self.broker.getposition(self.data).size >= sell_size:
            self.sell(size=sell_size)
            self.log('SELL CREATE, %.2f' % self.data.close[0])



# Initialize Cerebro engine
cerebro = bt.Cerebro()
cerebro.broker.set_cash(20000)
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
cerebro.addstrategy(MyStrategy)

# Download data from Yahoo Finance
try:

    data = yf.download('ebay', start='2024-08-13', end='2024-08-14', interval='5m', auto_adjust=True)
    if data.empty:
        raise ValueError("No data fetched. Please check the ticker and date range.")
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)
except Exception as e:
    print(f"Error downloading data: {e}")
    exit()

# Run the strategy
results = cerebro.run()

# Access the strategy instance from results
strategy_instance = results[0]

# Plot with Backtrader's built-in plotting
cerebro.plot(style='candle')
