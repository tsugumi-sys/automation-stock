import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import os
import requests
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

def Normalize(df):
    df_normalized = (df - df.mean(axis=0)) / df.std(axis=0)

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

def create_data(symbol):
    # Loading data
    df = yf.download(symbol, start='2016-01-01', end='2019-12-31', interval='1d')
    # set return and direction (label)
    df['return'] = (df['Adj Close'].shift(-10) / df['Adj Close']) - 1
    df['direction'] = np.where(df['return'] > 0, 1, -1)
    df['direction'] = df['direction'].shift(-1)
    #df['return'] = One_One_Scale(df['return'])
    # feature calculation
    # basic information
    df['price-change'] = df['Adj Close'] - df['Adj Close'].shift(1)
    df['price-change-percentage'] = df['Adj Close'] / df['Adj Close'].shift(1)
    # volume
    volume_mid = df['Volume'].median()
    df['Volume'] = df['Volume'].apply(lambda x: volume_mid if x == 0 else x)
    df['volume-change'] = np.log(df['Volume'] / df['Volume'].shift(1))
    #df['volume-change'] = One_One_Scale(df['volume-change'])
    # amount
    df['amount'] = df['Adj Close'] * df['Volume']
    df['amount-change'] = np.log(df['amount'] / df['amount'].shift(1))
    #df['amount-change'] = One_One_Scale(df['amount-change'])
    # simple moving average
    df['sma7'] = df['Adj Close'].rolling(7).mean()
    df['sma7-FP'] = (df['sma7'] - df['sma7'].shift(1)) / df['sma7'].shift(1)
    #df['sma7-FP'] = One_One_Scale(df['sma7-FP'])
    df['sma7'] = np.log(df['sma7']/df['sma7'].shift(1))
    
    df['sma25'] = df['Adj Close'].rolling(25).mean()
    df['sma25-FP'] = (df['sma25'] - df['sma25'].shift(1)) / df['sma25'].shift(1)
    #df['sma25-FP'] = One_One_Scale(df['sma25-FP'])
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
    #df['roc-FP'] = One_One_Scale(df['roc-FP'])
    # Relative Strength Index in 5 days
    df['rsi'] = df['price-change'].rolling(5).apply(RSI) / 100
    df['rsi-FP'] = (df['rsi'] - df['rsi'].shift(1))
    #df['rsi-FP'] = One_One_Scale(df['rsi-FP'])
    # Slow K and Slow D
    df['slow-k'] = df['Adj Close'].rolling(14).apply(SlowK)
    df['slow-d'] = df['slow-k'].rolling(14).mean()
    df['slow-k-FP'] = df['slow-k'] - df['slow-k'].shift(1)
    df['slow-d-FP'] = df['slow-d'] - df['slow-d'].shift(1)
    #df['slow-k'] = Zero_One_Scale(df['slow-k'])
    #df['slow-d'] = Zero_One_Scale(df['slow-d'])
    #df['slow-k-FP'] = One_One_Scale(df['slow-k-FP'])
    #df['slow-d-FP'] = One_One_Scale(df['slow-d-FP'])
    # ADOSC
    df['adosc'] = ((2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low'])) * df['Volume']
    df['adosc'] = df['adosc'].cumsum()
    df['adosc-ema3'] = df['adosc'].ewm(span=3, adjust=False).mean()
    df['adosc-ema10'] = df['adosc'].ewm(span=10, adjust=False).mean()
    df['adosc-SG'] = np.where((df['adosc-ema3'] - df['adosc-ema10']) > 0, 1, -1)
    #df['adosc'] = Zero_One_Scale(df['adosc'])
    # AR 26
    hp_op = (df['High'] - df['Open']).rolling(26).sum()
    op_lp = (df['Open'] - df['Low']).rolling(26).sum()
    df['ar26'] = hp_op / op_lp
    #df['ar26'] = Zero_One_Scale(df['ar26'])
    # BR 26
    hp_cp = (df['High'] - df['Close']).rolling(26).sum()
    cp_lp = (df['Close'] - df['Low']).rolling(26).sum()
    df['br26'] = hp_cp / cp_lp
    #df['br26'] = Zero_One_Scale(df['br26'])
    # VR 26
    
    # BIAS 20
    sma20 = df['Adj Close'].rolling(20).mean()
    df['bias20'] = (df['Adj Close'] - sma20) / sma20
    df['bias20'] = np.where(df['bias20'] > 0, 1, -1)
    
    
    
    #df['price-change'] = One_One_Scale(df['price-change'])
    #df['price-change-percentage'] = One_One_Scale(df['price-change-percentage'])
    # drop row contains NaN
    df.dropna(inplace=True)
    
    # adjust the length of the data, which should be a multiple number of 30.
    length = len(df) // 5
    
    return df[:5 * length]


def save_csv(df, sym, save_path):
    sym = sym.replace('.T', '')
    df.to_csv(save_path+'{}.csv'.format(sym))
    print('{}.csv Saved'.format(sym))

cols = ['return', 'price-change', 'price-change-percentage', 'volume-change', 'amount-change', 'sma7', 'sma7-FP', 'sma25', 'sma25-FP', 'smaDiff7-25',
        'macd', 'macd-SG', 'macd-histogram', 'cci-SG', 'mtm10', 'roc-SG', 'roc-FP', 'rsi', 'rsi-FP', 'slow-k', 'slow-d',
        'slow-k-FP', 'slow-d-FP', 'adosc', 'adosc-SG', 'ar26', 'br26', 'bias20']


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
    # save directory path
    save_path = '../../Data/LSTM_8/'

    # check directory path
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # load Raw Data
    symbol = pd.read_csv('../../../symbols/sandp500.csv')
    for symbol in symbol['symbol']:
        df = create_data(symbol)
        save_csv(df[cols], symbol, save_path)

    symbol = pd.read_csv('../../../symbols/nikkei255.csv')
    for symbol in symbol['symbol']:
        print(symbol)
        df = create_data(symbol)
        save_csv(df[cols], symbol, save_path)

    symbol = pd.read_csv('../../../symbols/nasdaq100.csv')
    for symbol in symbol['symbol']:
        df = create_data(symbol)
        save_csv(df[cols], symbol, save_path)
    send_line_notify('Success!!!!!!!!!!!!!!')
except:
    import traceback
    send_line_notify("Process has Stopped with some error!!!")
    send_line_notify(traceback.format_exc())