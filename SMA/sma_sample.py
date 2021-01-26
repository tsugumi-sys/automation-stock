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


# import financial data
# you can search your favorite company code here >> https://finance.yahoo.com/lookup/ 
start = '2016-01-01'
end = dt.datetime.now()
stock_code = 'AMD' 
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
percentChange = []
for i in df.index:
    SMA_short = df['SMA_20']
    SMA_long = df['SMA_50']
    close = df['Adj Close'][i]
    
    if (SMA_short[i] > SMA_long[i]):
        print('Up trend')
        if (position == 0):
            buy_price = close
            position = 1
            print('Buy as the price {}'.format(buy_price))
    elif (SMA_short[i] < SMA_long[i]):
        print('Down trend')
        if (position == 1):
            position = 0
            sell_price = close
            print("Sell at the price {}".format(sell_price))
            percent = (sell_price / buy_price - 1) * 100
            percentChange.append(percent)
    if (counter == df['Adj Close'].count() - 1 and position == 1):
            position = 0
            sell_price = close
            print('Sell at the price {}'.format(sell_price))
            percent = (sell_price / buy_price - 1) * 100
            
    counter += 1
    print(percentChange)

    # statistics
gains = 0
numGains = 0
losses = 0
numLosses = 0
total_return = 1
for i in percentChange:
    if i > 0:
        gains += i
        numGains += 1
    else:
        losses += i
        numLosses += 1
    total_return = total_return * ((i / 100) + 1)

total_return = round((total_return - 1)*100, 2)
print('This statistics is from {} up to {} with {} trades:'.format(df.index[0], df.index[-1], numGains+numLosses))
print('SMAS used: {}'.format(SMAs))
print('Total return over {} trades: {}%'.format(numGains+numLosses, total_return))

if (numGains > 0):
    average_gain = gains / numGains
    max_return = str(max(percentChange))
else:
    average_gain = 0
    max_return = 'unknown'

if (numLosses > 0):
    average_loss = losses / numLosses
    max_loss = str(min(percentChange))
    risk_reward_ratio = str(- average_gain / average_loss)
else:
    average_loss = 0
    max_loss = 'unknown'
    risk_reward_ratio = 'inf'

if (numGains > 0 or numLosses > 0):
    batting_ave = numGains / (numGains + numLosses)
else:
    batting_ave = 0

print('Average Gain: {}'.format(average_gain))
print('Average Loss: {}'.format(average_loss))
print('Max Return: {}'.format(max_return))
print('Max Loss: {}'.format(max_loss))
print('Gain/Loss ratio: {}'.format(risk_reward_ratio))
print('Batting Average: {}'.format(batting_ave))

# plot
fig, ax = mpf.plot(df, type='candle', figratio=(16, 6),
        mav=(short_sma, long_sma),
        volume=True, title=str(stock_code),
        style='mike', returnfig=True)
legend = ['Short_SMA', 'Long_SMA']
ax[0].legend(legend, fontsize=16)
fig.savefig('./AMD.png')