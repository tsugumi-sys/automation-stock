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


def SMA(stock_code, start='2016-01-01', end=dt.datetime.now()):
    print('-------------------------- {} ------------------------------'.format(stock_code))
    df = yf.download(stock_code, start, end, interval='1d') 
    df.head()

    # calculation SMA
    short_sma = 50
    mid_sma = 200
    long_sma = 300
    SMAs = [short_sma, mid_sma, long_sma]
    for i in SMAs:
        df["SMA_"+str(i)] = df.iloc[:,4].rolling(window=i).mean()
    df.tail(3)

    # judge Up trend or Down trend by SMA
    # Basics: if shorter SMA is higher than longer SMA, its now in up-trend area. Down-trend area if it's in the opposite condition.
    position = 0 # 1 means we ave already entered position, 0 means not already entered.
    counter = 0
    percentChange = []
    for i in df.index:
        SMA_short = df['SMA_'+str(short_sma)][i]
        SMA_middle = df['SMA_'+str(mid_sma)][i]
        SMA_long = df['SMA_'+str(long_sma)][i]
        close = df['Adj Close'][i]
        open_price = df['Open'][i]
        if np.isnan(SMA_long):
            continue
        else:
            if (SMA_short > SMA_middle and SMA_short > SMA_long and SMA_middle > SMA_long):
                if (position == 0):
                    buy_price = close
                    position = 1
            elif (SMA_short < SMA_long):
                if (position == 1):
                    position = 0
                    sell_price = close
                    percent = (sell_price / buy_price - 1) * 100
                    percentChange.append(percent)
            # loss cut when open price < buy_price * 0.94
            elif position == 1 and buy_price * 0.94 > open_price:
                position = 0
                percent = (open_price / buy_price - 1) * 100
                percentChange.append(percent)
            # loss cut
            elif position == 1 and (1 - close / buy_price) * 100 > 6:
                position = 0
                sell_price = buy_price * 0.94
                percent = (sell_price/buy_price - 1) * 100
                percentChange.append(percent)
            if (i == df.index[-1] and position == 1):
                    position = 0
                    sell_price = close
                    percent = (sell_price / buy_price - 1) * 100
                    percentChange.append(percent)

        counter += 1

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

    total_return = (total_return - 1)*100

    if (numGains > 0):
        average_gain = gains / numGains
        max_return = max(percentChange)
    else:
        average_gain = 0
        max_return = 0

    if (numLosses > 0):
        average_loss = losses / numLosses
        max_loss = min(percentChange)
        risk_reward_ratio = - average_gain / average_loss
    else:
        average_loss = 0
        max_loss = 0
        risk_reward_ratio = 0

    if (numGains > 0 or numLosses > 0):
        batting_ave = numGains / (numGains + numLosses)
    else:
        batting_ave = 0

    trades = numGains + numLosses
    
    return [trades, total_return, average_gain, average_loss, max_return, max_loss, risk_reward_ratio, batting_ave]
    
# init values
results = []
stock_codes = pd.read_csv('../symbols/sandp500.csv')
for stock_code in stock_codes['symbol']:
    items = SMA(stock_code=stock_code)
    results.append(items)

columns = ['trades', 'Total return', 'Average Gain', 'Average Loss', 'Max Return', 'Max Loss', 'Gain/Loss Ratio', 'Batting Average']
df = pd.DataFrame(results, columns=columns, index=stock_codes)
df.to_csv('./results/sandp500-50-100.csv')
print('Completed')