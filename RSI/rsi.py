import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import mplfinance as mpl


def RSI(stock_code, term=14, start="2016-01-01", end=dt.datetime.now()):
    # load data
    df = yf.download(stock_code, start, end, interval='1d')
    # calculate stock price difference between yesterday and today.
    terms = [25, 50]
    for term in terms:
        df['SMA'+str(term)] = df['Adj Close'].rolling(window=term).mean()

    def diff(x):
        return x[-1] - x[0]

    df['Change'] = df['Adj Close'].rolling(window=2).apply(diff)

    # calculate rsi
    def rsi(x):
        negative_list, positive_list = [i for i in x if i < 0], [i for i in x if i > 0 or i == 0]
        if len(negative_list) == 0:
            return 100
        elif len(positive_list) == 0:
            return 0
        else:
            negative_ave, positive_ave = -sum(negative_list)/len(negative_list), sum(positive_list)/len(positive_list)
            return positive_ave/(negative_ave+positive_ave) * 100

    df['RSI'] = df['Change'].rolling(window=14).apply(rsi)

    # calculate trend
    term = 3
    name = 'SMA25 Trend'
    df[name] = df['SMA25'].rolling(window=term).apply(diff)

    position = 0
    percentChange = []
    for i in df.index:
        rsi = df['RSI'][i]
        close = df['Adj Close'][i]
        sma_short = df['SMA25'][i]
        sma_long = df['SMA50'][i]
        sma_trend = df['SMA25 Trend'][i]
        if np.isnan(rsi):
            continue
        else:
            if rsi > 50 or close > sma_long or sma_trend > 0:
                if position == 0:
                    position = 1
                    buy_price = close
            elif close < sma_long:
                if position == 1:
                    position = 0
                    sell_price = close
                    percent = (sell_price / buy_price - 1) * 100
                    percentChange.append(percent)
                    
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
        total_return = total_return * ((i/100) + 1)
    print(total_return)
    total_return = round((total_return - 1)*100, 2)
    if numGains > 0:
        ave_gain = gains / numGains
        max_return = max(percentChange)
    else:
        ave_gain = 0
        max_return = 'unknown'
    
    if numLosses > 0:
        ave_loss = losses / numLosses
        max_loss = min(percentChange)
        risk_reward_retio = - ave_gain / ave_loss
    else:
        ave_loss = 0
        max_loss = 'unknown'
        risk_reward_retio = 'inf'
    
    if numGains > 0 or numLosses > 0:
        batting_retio = numGains / (numGains + numLosses)
    else:
        batting_retio = 0
    
    trades = numGains + numLosses
    
    
    results = [trades, total_return, ave_gain, ave_loss, max_return, max_loss, risk_reward_retio, batting_retio]
    if 'unknown' in results or 'inf' in results:
        return None
    else:
        return results
df = pd.read_csv('../symbols/s&p-symbols.csv')
#print(df['symbol'])
results = []
symbols = []
for symbol in df['symbol']:
    result = RSI(stock_code=symbol)
    if not result is None:
        symbols.append(symbol)
        results.append(result)

print(len(results))

columns = ['trades', 'Total return', 'Average Gain', 'Average Loss', 'Max Return', 'Max Loss', 'Gain/Loss Ratio', 'Batting Average']
df = pd.DataFrame(results, columns=columns, index=symbols)
df.to_csv('./result.csv')

# failed to download
# BRK.B, BF.B, CTL, ETFC, MYL, NBL,