import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
import traceback
import os
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

def Zero_One_Scale(df):
    df_scaled = (df - df.min()) / (df.max() - df.min())
    return df_scaled

def One_One_Scale(df):
    df_scaled = 2 * (df - df.min()) / (df.max() - df.min()) - 1
    return df_scaled


def RSI(x):
    up, down = [i for i in x if i > 0], [i for i in x if i <= 0]
    if len(down) == 0:
        return 100
    elif len(up) == 0:
        return 0
    else:
        up_average = sum(up) / len(up)
        down_average = - sum(down) / len(down)
        return 100 * up_average / (up_average + down_average)
    
def SlowK(x):
    min_price = min(x)
    max_price = max(x)
    k = (x[-1] - min_price) / (max_price - min_price)
    return k

one_one_cols = ["return", "volume-change", "amount-change", "sma7-FP", "sma7", "sma25-FP", "sma25", "roc-FP",
 "rsi-FP", "slow-k-FP", "slow-d-FP", "price-change", "price-change-percentage"]

zero_one_cols = ["slow-k", "slow-d", "adosc", "ar26", "br26"]

cols = ["volume-change", "amount-change", "sma7-FP", "sma7", "sma25-FP", "sma25", "roc-FP",
 "rsi-FP", "slow-k-FP", "slow-d-FP", "price-change", "price-change-percentage"]

n_features = len(one_one_cols) - 1


def make_prediction(df, model):
    df = df.copy()
    before_len = len(df)
    df['return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    df['direction'] = np.where(df['return'] > 0, 1, -1)
    df['direction'] = df['direction'].shift(-1)
    df['return'] = df['return'].shift(-1)
    # feature calculation
    # basic information
    df['price-change'] = df['Adj Close'] - df['Adj Close'].shift(1)
    df['price-change-percentage'] = df['Adj Close'] / df['Adj Close'].shift(1)
    # volume
    volume_mid = df['Volume'].median()
    df['Volume'] = df['Volume'].apply(lambda x: volume_mid if x == 0 else x)
    df['volume-change'] = np.log(df['Volume'] / df['Volume'].shift(1))
    # amount
    df['amount'] = df['Adj Close'] * df['Volume']
    df['amount-change'] = np.log(df['amount'] / df['amount'].shift(1))
    # simple moving average
    df['sma7'] = df['Adj Close'].rolling(7).mean()
    df['sma7-FP'] = (df['sma7'] - df['sma7'].shift(1)) / df['sma7'].shift(1)
    df['sma7'] = np.log(df['sma7'] / df["sma7"].shift(1))
    
    df['sma25'] = df['Adj Close'].rolling(25).mean()
    df['sma25-FP'] = (df['sma25'] - df['sma25'].shift(1)) / df['sma25'].shift(1)
    df['sma25'] = np.log(df['sma25']/df['sma25'].shift(1))
    
    # simple moving average difference
    df['smaDiff7-25'] = df['sma7'] - df['sma25']
    df['smaDiff7-25'] = np.where(df['smaDiff7-25'] > 0, 1, -1)

    # Moving Average Convergence Divergence
    df['macd'] = df['Adj Close'].rolling(12).mean() - df['Adj Close'].rolling(26).mean()
    df['macd-SG'] = df['macd'].rolling(9).mean()
    df['macd-histogram'] = df['macd'] - df['macd-SG']
    df['macd-histogram'] = np.where(df['macd-histogram'] > 0, 1, -1)
    df['macd-SG'] = np.where(df['macd-SG'] > 0, 1, -1)
    df['macd'] = np.where(df['macd'] > 0, 1, -1)
    # Commodity Channel Index in 24 days
    df['typical-price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['sma-cci'] = df['typical-price'].rolling(24).mean()
    df['mean-deviation'] = np.abs(df['typical-price'] - df['sma-cci'])
    df['mean-deviation'] = df['mean-deviation'].rolling(24).mean()
    df['cci'] = (df['typical-price'] - df['sma-cci']) / (0.015 * df['mean-deviation'])
    df['cci-SG'] = np.where(df['cci'] > 0, 1, -1)
    # MTM 10
    df['mtm10'] = df['Adj Close'] - df['Adj Close'].shift(10)
    df['mtm10'] = np.where(df['mtm10'] > 0, 1, -1)
    # Rate of Change in 10 days
    df['roc'] = (df['Adj Close'] - df['Adj Close'].shift(10)) / df['Adj Close'].shift(10)
    df['roc-SG'] = np.where(df['roc'] > 0, 1, -1)
    df['roc-FP'] = (df['roc'] - df['roc'].shift(1))
    # Relative Strength Index in 5 days
    df['rsi'] = df['price-change'].rolling(5).apply(RSI) / 100
    df['rsi-FP'] = (df['rsi'] - df['rsi'].shift(1))
    # Slow K and Slow D
    df['slow-k'] = df['Adj Close'].rolling(14).apply(SlowK)
    df['slow-d'] = df['slow-k'].rolling(14).mean()
    df['slow-k-FP'] = df['slow-k'] - df['slow-k'].shift(1)
    df['slow-d-FP'] = df['slow-d'] - df['slow-d'].shift(1)
    # ADOSC
    df['adosc'] = ((2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low'])) * df['Volume']
    df['adosc'] = df['adosc'].cumsum()
    df['adosc-ema3'] = df['adosc'].ewm(span=3, adjust=False).mean()
    df['adosc-ema10'] = df['adosc'].ewm(span=10, adjust=False).mean()
    df['adosc-SG'] = np.where((df['adosc-ema3'] - df['adosc-ema10']) > 0, 1, -1)
    # AR 26
    hp_op = (df['High'] - df['Open']).rolling(26).sum()
    op_lp = (df['Open'] - df['Low']).rolling(26).sum()
    df['ar26'] = hp_op / op_lp
    # BR 26
    hp_cp = (df['High'] - df['Close']).rolling(26).sum()
    cp_lp = (df['Close'] - df['Low']).rolling(26).sum()
    df['br26'] = hp_cp / cp_lp
    
    # BIAS 20
    sma20 = df['Adj Close'].rolling(20).mean()
    df['bias20'] = (df['Adj Close'] - sma20) / sma20
    df['bias20'] = np.where(df['bias20'] > 0, 1, -1)
    
    # drop row contains NaN
    df.dropna(inplace=True)
    after_len = len(df)
    
    
    # Normalization
    X = df.copy()[one_one_cols]
    y = X.pop('return')
    
    # make prediction
    count = 0
    result = [np.nan for i in range(30 + (before_len - after_len))]
    for i in range(30,len(df)):
        data = X.copy()[count:i]
        for col in cols:
            data[col] = One_One_Scale(data[col])
        data = np.reshape(data.values, (1, 30, n_features))
        y = model.predict(data)
        count += 1
        result.append(y[0][-1])
    
    
    return result



def evaluate_model(data):
    pred = make_prediction(data, model)
    data['pred'] = pred
    
    position = 0
    percentChange = []
    holding_periods = []
    buy_days = []
    sell_days = []
    loss_line = [0.94, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    for i in data.index:
        close = data['Adj Close'][i]
        ml_signal = data['pred'][i]
        if position == 0 and ml_signal > 0:
            position = 1
            buy_price = close
            buy_day = i
            buy_days.append(i)
            
        elif position == 1 and ml_signal < 0:
            position = 0
            sell_price = close
            percent = (sell_price/buy_price - 1) * 100
            percentChange.append(percent)
            holding_periods.append((i - buy_day).days)
            sell_days.append(i)
            
        elif position == 1 and close < buy_price * loss_line[0]:
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

def send_line_notify(notification_message):
    """
    LINEに通知する
    """
    line_notify_token = os.getenv('LINE_TOKEN')
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {notification_message}'}
    requests.post(line_notify_api, headers = headers, data = data)

try:
    model_num = 12
    model = load_model(f'./models/model{model_num}/model.h5')

    # nikkei 255
    symbols = pd.read_csv('../../../symbols/nikkei255.csv')
    results = []
    for symbol in symbols['symbol']:
        data = yf.download(symbol, start='2020-01-01', end='2020-12-31', interval='1d')
        item = evaluate_model(data)
        print('--- Nikkei255 {} ---'.format(symbol))
        show_results(item)
        print('-'*30)
        results.append(item)
    columns = ['trades', 'Total return', 'Average Gain', 'Average Loss', 'Max Return', 'Max Loss', 'Gain/Loss Ratio', 'Batting Average', "Average Period"]
    df = pd.DataFrame(results, columns=columns, index=symbols['symbol'])
    df.to_csv(f'./results/model{model_num}-nikkei255.csv')

    #nasdaq100
    symbols = pd.read_csv('../../../symbols/nasdaq100.csv')
    results = []
    for symbol in symbols['symbol']:
        data = yf.download(symbol, start='2020-01-01', end='2020-12-31', interval='1d')
        item = evaluate_model(data)
        print('--- NASDAQ100 {} ---'.format(symbol))
        show_results(item)
        print('-'*30)
        results.append(item)
    columns = ['trades', 'Total return', 'Average Gain', 'Average Loss', 'Max Return', 'Max Loss', 'Gain/Loss Ratio', 'Batting Average', "Average Period"]
    df = pd.DataFrame(results, columns=columns, index=symbols['symbol'])
    df.to_csv(f'./results/model{model_num}-nasdaq100.csv')

    #s and p 500
    symbols = pd.read_csv('../../../symbols/sandp500.csv')
    
    results = []
    for symbol in symbols['symbol']:
        data = yf.download(symbol, start='2020-01-01', end='2020-12-31', interval='1d')
        item = evaluate_model(data)
        print('--- S&P500 {} ---'.format(symbol))
        show_results(item)
        print('-'*30)
        results.append(item)
    columns = ['trades', 'Total return', 'Average Gain', 'Average Loss', 'Max Return', 'Max Loss', 'Gain/Loss Ratio', 'Batting Average', "Average Period"]
    df = pd.DataFrame(results, columns=columns, index=symbols['symbol'])
    df.to_csv(f'./results/model{model_num}-sandp500.csv')

    send_line_notify('Successfully Completed!!')

except:
    send_line_notify("Process has Stopped with some error!!!")
    send_line_notify(traceback.format_exc())
    print(traceback.format_exc())
