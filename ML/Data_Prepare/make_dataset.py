import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import time
import requests
import urllib3

def RSI(df, term=14):
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

    up_trend_days = []
    down_trend_days = []
    neutral_days = []
    for i in df.index:
        rsi = df['RSI'][i]
        close = df['Adj Close'][i]
        sma_short = df['SMA25'][i]
        sma_long = df['SMA50'][i]
        sma_trend = df['SMA25 Trend'][i]
        if np.isnan(rsi):
            continue
        else:
            if rsi > 50 and close > sma_long or sma_trend > 0:
                up_trend_days.append(i)
            elif close < sma_long:
                down_trend_days.append(i)
            else:
                neutral_days.append(i)
    signals = []
    for i in df.index:
        if i in up_trend_days:
            signals.append(1)
        elif i in down_trend_days:
            signals.append(0)
        elif i in neutral_days:
            signals.append(0.5)
        else:
            signals.append(np.nan)
            
    return signals

def SMA(df, short_sma=20, long_sma=50):
    # calculation SMA
    SMAs = [short_sma, long_sma]
    for i in SMAs:
        df["SMA_"+str(i)] = df.iloc[:,4].rolling(window=i).mean()

    # judge Up trend or Down trend by SMA
    # Basics: if shorter SMA is higher than longer SMA, its now in up-trend area. Down-trend area if it's in the opposite condition.
    position = 0 # 1 means we ave already entered position, 0 means not already entered.
    counter = 0
    up_trend_days = []
    down_trend_days = []
    neutral_days = []
    for i in df.index:
        SMA_short = df['SMA_20']
        SMA_long = df['SMA_50']

        if np.isnan(SMA_long[i]):
            continue
        else:
            if (SMA_short[i] > SMA_long[i]):
                up_trend_days.append(i)

            elif (SMA_short[i] < SMA_long[i]):
                down_trend_days.append(i)
            else:
                neutral_days.append(i)


    signals = []
    for i in df.index:
        if i in up_trend_days:
            signals.append(1)
        elif i in down_trend_days:
            signals.append(0)
        elif i in neutral_days:
            signals.append(0.5)
        else:
            signals.append(np.nan)
    return signals

def make_label(df):
    def diff(x):
        subt = x[-1] - x[int(len(x) / 2)-1]
        if subt > 0:
            return 1
        else:
            return 0
        
    df['label'] = df.iloc[:, 4].rolling(60, center=True).apply(diff)
    
    up_trend_days = []
    down_trend_days = []
    for i in df.index:
        pin = df['label'][i]
        if pin == 1:
            up_trend_days.append(i)
        elif pin == 0:
            down_trend_days.append(i)
    
    signals = []
    for i in df.index:
        if i in up_trend_days:
            signals.append(1)
        elif i in down_trend_days:
            signals.append(0)
        else:
            signals.append(np.nan)
    return signals


def make_dataset(symbols, start, end=dt.datetime.now()):
    results = []
    count = 0
    for symbol in symbols:
        try:
            df = yf.download(symbol, start='2016-01-01', end=dt.datetime.now(), interval='1d')

            sma_signals = SMA(df)
            rsi_signals = RSI(df)
            label_signals = make_label(df)
            for i in range(len(sma_signals)):
                row = []
                if not np.isnan(sma_signals[i]) and not np.isnan(rsi_signals[i]) and not np.isnan(label_signals[i]):
                    row.append(sma_signals[i])
                    row.append(rsi_signals[i])
                    row.append(label_signals[i])
                    results.append(row)

            del sma_signals, rsi_signals, label_signals, df
            time.sleep(3)
        except OSError:
            print('OSError at downloading of {}'.format(symbol))
            continue
        except urllib3.exceptions.MaxRetryError:
            print('urllib3.exceptions.MaxRetryError at downloading of {}'.format(symbol))
            continue
        except requests.exceptions.ConnectionError:
            print('requests.exceptions.ConnectionError at downloading of {}'.format(symbol))
            continue
        count += 1
        print(count)

    df = pd.DataFrame(results, columns=['SMA', 'RSI', 'LABEL'])
    df.to_csv('./dataset_sandp4.csv')
    return print('Successfully made data.')
df = pd.read_csv('../../symbols/sAndp500.csv')
symbols = df['symbol'][150:200]
start = '2018-01-01'
make_dataset(symbols=symbols, start=start)