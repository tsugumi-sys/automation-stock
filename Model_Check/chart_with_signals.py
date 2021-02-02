# Simple Moving Average (SMA) Strategy Tutorial
# url: https://towardsdatascience.com/data-science-in-finance-56a4d99279f7

# conda create --name unko python=3.7 >> yfinance はバージョン3.7まで対応（最新pythonは3.9）
# conda install -c conda-forge pandas, matplotlib, mplfinance
# conda install -c anaconda numpy  >> pandasライブラリに含まれているかも?
# pip install yfinance

import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import mplfinance as mpf
from matplotlib.ticker import MultipleLocator


# import financial data
# you can search your favorite company code here >> https://finance.yahoo.com/lookup/ 
start = '2018-01-01'
end = dt.datetime.now()
stock_code = '^GSPC' 
df = yf.download(stock_code, start, end, interval='1d') 
df.head()

# calculation SMA
short_sma = 20
long_sma = 50
SMAs = [short_sma, long_sma]
for i in SMAs:
    df["SMA_"+str(i)] = df.iloc[:,4].rolling(window=i).mean()
df.tail(3)

# judge Up trend or Down trend by SMA
# Basics: if shorter SMA is higher than longer SMA, its now in up-trend area. Down-trend area if it's in the opposite condition.
position = 0 # 1 means we ave already entered position, 0 means not already entered.
counter = 0
buy_days = []
sell_days = []
percentChange = []
for i in df.index:
    SMA_short = df['SMA_20']
    SMA_long = df['SMA_50']
    close = df['Adj Close'][i]
    
    if (SMA_short[i] > SMA_long[i]):
        #print('Up trend')
        if (position == 0):
            buy_price = close
            position = 1
            buy_days.append(i)
            #print('Buy as the price {}'.format(buy_price))
    elif (SMA_short[i] < SMA_long[i]):
        #print('Down trend')
        if (position == 1):
            position = 0
            sell_price = close
            sell_days.append(i)
            #print("Sell at the price {}".format(sell_price))
            percent = (sell_price / buy_price - 1) * 100
            percentChange.append(percent)
    if (counter == df['Adj Close'].count() - 1 and position == 1):
            position = 0
            sell_price = close
            #print('Sell at the price {}'.format(sell_price))
            percent = (sell_price / buy_price - 1) * 100
            
            
    counter += 1
    #print(percentChange)

print(len(percentChange))
print(buy_days)
print(sell_days)
buy_signals = []
sell_signals = []
for i in df.index:
    if i in buy_days:
        buy_signals.append(df['Adj Close'][i])
    else:
        buy_signals.append(np.nan)
    if i in sell_days:
        sell_signals.append(df['Adj Close'][i])
    else:
        sell_signals.append(np.nan)

# plot
adp = [
    mpf.make_addplot(buy_signals, type='scatter', markersize=200, marker='^', color="b"),
    mpf.make_addplot(sell_signals, type='scatter', markersize=200, marker='v', color="r")
]
fig, ax = mpf.plot(df, type='candle', figratio=(45, 15),
        mav=(short_sma, long_sma),
        addplot=adp,
        volume=True, title=str(stock_code),
        style='starsandstripes', returnfig=True)
legend = ['Short_SMA(20)', 'Long_SMA(50)']
ax[0].legend(legend, fontsize=16)
print(ax[0])
fig.savefig('./gspc.png')