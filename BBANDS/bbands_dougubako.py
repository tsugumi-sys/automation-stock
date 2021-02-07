import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpl
import yfinance as yf
import datetime as dt


def stock_validation(stock_code, start="2016-01-01", end=dt.datetime.now(), interval="1d"):
    df = yf.download(stock_code, start, end, interval='1d')

    df['MA'] = df['Adj Close'].rolling(window=100).mean()
    df['STD'] = df['Adj Close'].rolling(window=100).std()
    df['Upper'] = df['MA'] + (df['STD'] * 2.5)
    df['Lower'] = df['MA'] - (df['STD'] * 2.5)


    position = 0
    counter = 0
    percentChange = []
    for i in df.index:
        if np.isnan(df['MA'][i]):
            continue
        else:
            upper = df['High'][i]
            lower = df['Low'][i]
            close = df['Adj Close'][i]
            if close > df['Upper'][i]:
                # up trend
                if position == 0:
                    position = 1
                    buy_price = close

            elif close < df['MA'][i]:
                # down trend
                if position == 1:
                    position = 0
                    sell_price = close
                    percent = (sell_price / buy_price - 1)*100
                    percentChange.append(percent)
            elif position == 1 and (1 - close/buy_price) * 100 > 6:
                position = 0
                sell_price = close
                percent = (sell_price / buy_price - 1)*100
                percentChange.append(percent)
            if counter == df['Adj Close'].count() - 1 and position == 1:
                position = 0


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
        total_return = total_return*((i/100) + 1)
    total_return = round((total_return - 1)*100, 2)

    #print('Total return over {} trades: {}%'.format(numGains+numLosses, total_return))

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

results = []
#result = stock_validation(stock_code='UAA')
print(result)
stock_codes = pd.read_csv('../symbols/nasdaq100.csv')
for stock_code in stock_codes['symbol']:
    results.append(stock_validation(stock_code=stock_code))

columns = ['trades', 'Total return', 'Average Gain', 'Average Loss', 'Max Return', 'Max Loss', 'Gain/Loss Ratio', 'Batting Average']
df = pd.DataFrame(results, columns=columns, index=stock_codes)
df.to_csv('./results/result-nasdaq100-dougubako.csv')
print('Saved')