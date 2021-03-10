# N-Lags model check
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('./models/model4/model.h5')
lags = 5
def create_lags(data):
    global cols
    cols = []
    for lag in range(1, lags+1):
        col = 'lag_{}'.format(lag)
        data[col] = data['returns'].shift(lag)
        cols.append(col)

def evaluate_model(data):
    data['returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    data['direction'] = np.sign(data['returns'])
    create_lags(data)
    data.dropna(inplace=True)
    data_ = (data - data.mean()) / data.std() # standarization
    data_['direction'] = np.where(data['direction'] == 1, 1, 0)
    data['position'] = np.where(model.predict(data_[cols]) > 0.5, 1, 0)
    valid = data['position'] == data_['direction']
    accuracy = len(valid[valid == True]) / len(valid)
    
    position = 0
    percentChange = []
    holding_periods = []
    buy_days = []
    sell_days = []
    loss_line = [0.94, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    for i in data.index:
        close = data['Adj Close'][i]
        signal = data['position'][i]
        if position == 0 and signal == 1:
            position = 1
            buy_price = close
            buy_day = i
            buy_days.append(i)
            
        elif position == 1 and signal == 0:
            position = 0
            sell_price = close
            percent = (sell_price/buy_price - 1) * 100
            percentChange.append(percent)
            holding_periods.append((i - buy_day).days)
            sell_days.append(i)
            
        elif position == 1 and buy_price < close * loss_line[0]:
            position = 0
            sell_price = close
            percent = (sell_price/buy_price - 1) * 100
            percentChange.append(percent)
            holding_periods.append((i - buy_day).days)
            sell_days.append(i)
            
        if i == data.index[-1] and position == 1:
            position = 0
            sell_price = close
            percent = (sell_price/buy_price - 1) * 100
            percentChange.append(percent)
            holding_periods.append((i - buy_day).days)
            sell_days.append(i)
            
    gains = 0
    numGains = 0
    losses = 0
    numLosses = 0
    total_return = 1
    for i in percentChange:
        if i > 0:
            gains += i
            numGains += 1
        elif i < 0:
            losses += i
            numLosses += 1
        total_return = total_return * ((i/100) + 1)
        
    total_return = (total_return - 1) * 100
    trades = len(percentChange)
    
    if numGains > 0:
        average_gain = gains / numGains
        max_return = max(percentChange)
    else:
        average_gain = np.nan
        max_return = np.nan
    
    if numLosses > 0:
        average_loss = losses / numLosses
        max_loss = min(percentChange)
    else:
        average_loss = np.nan
        max_loss = np.nan
    
    if numGains > 0 and numLosses > 0:
        risk_reward_ratio = - average_gain / average_loss
        
    elif numGains == 0 and numLosses > 0:
        risk_reward_ratio = 0
        
    elif numGains > 0 and numLosses == 0:
        risk_reward_ratio = average_gain
    
    else:
        risk_reward_ratio = np.nan
        
    if numGains > 0 or numLosses > 0:
        batting_ave = numGains / (numGains + numLosses)
    else:
        batting_ave = np.nan
    
    if len(holding_periods) > 0:
        average_period = sum(holding_periods) / len(holding_periods)
    else:
        average_period = np.nan
        
    
    return [trades, total_return, average_gain, average_loss, max_return, max_loss, risk_reward_ratio, batting_ave, average_period]

def show_results(results):
    labels = ['Trades', 'Total Return', 'Average Gain', 'Average Loss', 'Max Return', 'Max Loss', 'Risk Reward Ratio', 'Batting Average', 'Average Holding Periods']
    for i in range(len(results)):
        print('%30s | %8.3f' % (labels[i], results[i]))
    

symbols = pd.read_csv('../../../symbols/nikkei255.csv')
results = []
# for stock_code in stock_codes['symbol']:
#     items = SMA(stock_code=stock_code)
#     results.append(items)

# columns = ['trades', 'Total return', 'Average Gain', 'Average Loss', 'Max Return', 'Max Loss', 'Gain/Loss Ratio', 'Batting Average', "Average Period"]
# df = pd.DataFrame(results, columns=columns, index=stock_codes)
# df.to_csv('./results/nikkei255-50-200-300-lossLevel.csv')
# print('Completed')
for symbol in symbols['symbol']:
    data = yf.download(symbol, start='2016-01-01', end=dt.datetime.now(), interval='1d')
    item = evaluate_model(data)
    print('--- {} ---'.format(symbol))
    show_results(item)
    print('-'*30)
    results.append(item)
columns = ['trades', 'Total return', 'Average Gain', 'Average Loss', 'Max Return', 'Max Loss', 'Gain/Loss Ratio', 'Batting Average', "Average Period"]
df = pd.DataFrame(results, columns=columns, index=symbols['symbol'])
df.to_csv('./results/model1-nikkei255.csv')