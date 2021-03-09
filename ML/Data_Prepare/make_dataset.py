import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt

# create lags data
# 
# Parameters
# ==========
# lags: int, the number of lags classification
# data: tick data from yfinance
# cols: list of sting, the names of lag type
def create_lags(data):
    global cols
    cols = []
    for lag in range(1, lags+1):
        col = 'lag_{}'.format(lag)
        data[col] = data['returns'].shift(lag)
        cols.append(col)
    
    for lag in range(1, lags+1):
        col = 'sma1_lag_{}'.format(lag)
        data[col] = data['sma1_log'].shift(lag)
        cols.append(col)
        
    for lag in range(1, lags+1):
        col = 'sma2_lag_{}'.format(lag)
        data[col] = data['sma2_log'].shift(lag)
        cols.append(col)
    cols.append('direction')

# create input and label data for machine leaning training
#
# Parameters
# ==========
# data: tick data from finance
# output: ndarray, input_data and label data
def create_data(data):
    data['sma1'] = data['Adj Close'].rolling(40).mean()
    data['sma2'] = data['Adj Close'].rolling(190).mean()
    data['sma1_log'] = np.log(data['sma1'] / data['sma1'].shift(1))
    data['sma2_log'] = np.log(data['sma2'] / data['sma2'].shift(1))
    data['returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    data['direction'] = np.sign(data['returns'])
    create_lags(data)
    data.dropna(inplace=True)
    data_ = (data - data.mean()) / data.std() # standarization
    data_['direction'] = np.where(data['direction'] == 1, 1, 0)
    return data_[cols]

lags = 5
symbols = pd.read_csv('../../symbols/sandp500.csv')
first_data = yf.download(symbols['symbol'][0], start='2016-01-01', end=dt.datetime.now(), interval='1d')
df = create_data(first_data)
for symbol in symbols['symbol'][1:]:
    data = yf.download(symbol, start='2016-01-01', end=dt.datetime.now(), interval='1d')
    ml_data = create_data(data)
    df = df.append(ml_data)
df.to_csv('./lags5_with_sma_dataset.csv')