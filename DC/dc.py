import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import mplfinance as mpf

start = '2016-01-01'
end = dt.datetime.now()
stock_code = 'AMD'
df = yf.download(stock_code, start, end, interval='1d')
print(df.head())
term = 20

# chek the highest price in the past {term} times
df['Highest'+str(term)] = df.iloc[:, 1].rolling(window=term).max()
# chek the highest price in the past {term} times
df['Lowest'+str(term)] = df.iloc[:, 2].rolling(window=term).min()

# judge U trend or Down trend by DC
position = 0 # 1 means entered and 0 means not already entered
counter = 0 
percentChange = []
for i in range(1, len(df)):
    Highest = df['Highest'+str(term)][i-1]
    Lowest = df['Lowest'+str(term)][i-1]
    high_price = df['High'][i]
    low_price = df['Low'][i]
    close = df['Adj Close'][i]
    
    # avoid NaN data 
    if np.isnan(Highest):
        continue
    else:
        if (high_price > Highest):
            print('Up trend')
            if (position == 0):
                position = 1
                buy_price = close
                print('Buy at the price {}'.format(buy_price))

        elif (low_price < Lowest):
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

# statistic
gains = 0
numGains = 0
losses = 0
numLosses = 0
total_return = 1

for i in percentChange:
    if i > 0:
        numGains += 1
        gains += i
    else:
        numLosses += 1
        losses += i
    total_return = total_return * ((i / 100) + 1)

total_return = round((total_return - 1)*100, 2)

if numGains > 0:
    average_gain = gains / numGains
    max_return = max(percentChange)
else:
    average_gain = 0
    max_return = 'unknown'
    
if numLosses > 0:
    average_loss = losses / numLosses
    max_loss = min(percentChange)
    risk_reward_retio = - average_gain / average_loss
else:
    average_loss = 0
    max_loss = 'unknown'
    risk_reward_retio = 'inf'
    
if numGains > 0 or numLosses > 0:
    batting_ratio = numGains / (numGains + numLosses)
else:
    batting_ratio = 0
    
print('The period is from {} up to {}'.format(df.index[0], df.index[-1]))
print('Trades: {}'.format(numGains+numLosses))
print('Total return: {}%'.format(total_return))
print('Average Gain: {}'.format(average_gain))
print('Average Loss: {}'.format(average_loss))
print('Max Return: {}'.format(max_return))
print('Max Loss: {}'.format(max_loss))
print('Gain/Loss Ratio: {}'.format(risk_reward_retio))
print('Batting Average: {}'.format(batting_ratio))