import numpy as np
import pandas as ps
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import mplfinance as mpl

start = '2016-01-01'
end = dt.datetime.now()
stock_code = '^GSPC' # AMD: 988.6%
df = yf.download(stock_code, start, end, interval='1d')
term = 14

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
trends = [3]
def linear_approx(n_sample, data):
    grad, bias = np.polyfit(n_sample, data, 1)
    return grad
for term in trends:
    name = 'RSI Trend'
    df[name] = df['RSI'].rolling(window=term).apply(lambda x: linear_approx(n_sample=list(range(1, term+1)), data=x))
    # normalize gradient from -1 to 1 range
    x_std = (df[name] - df[name].min()) / (df[name].max() - df[name].min())
    df[name] = x_std * 2 - 1


# buy if RSI < 30% and RSI Trends5 > 0
# sell if RSI > 70% and RSI Trends5 < 0
position = 0
percentChange = []
for i in df.index:
    rsi = df['RSI'][i]
    rsi_trend = df['RSI Trend'][i]
    close = df['Adj Close'][i]
    sma_short = df['SMA25'][i]
    sma_long = df['SMA50'][i]
    if np.isnan(rsi):
        continue
    else:
        if rsi < 30:
            if position == 0:
                position = 1
                buy_price = close
        elif rsi > 70:
            if position == 1:
                position = 0
                sell_price = close
                percent = (sell_price / buy_price - 1) * 100
                percentChange.append(percent)

print(percentChange)
total_return = 1
for i in percentChange:
    total_return = total_return * ((i/100) + 1)
total_return = round((total_return - 1)*100, 2)
print('Total return: {}%'.format(total_return))
# plot chart
df['70'] = 70
df['30'] = 30
df['50'] = 50
df['0'] = 0
apds = [
    mpl.make_addplot(df[['RSI', '30', '50', '70']], panel=1, type='line', ylabel='RSI'),
    mpl.make_addplot(df[['RSI Trend', '0']], panel=2, type='line', ylabel='RSI Trends5')
]
fig = mpl.plot(df, type='candle',figsize=(30, 15), style='mike', addplot=apds, title=str(stock_code))