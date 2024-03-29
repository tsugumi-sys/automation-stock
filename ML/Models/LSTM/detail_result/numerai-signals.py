import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests as re
import numerapi

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


def make_prediction(df, model):
    df = df.copy()
    before_len = len(df)
    df['return'] = np.log(df['Adj Close'].shift(10) / df['Adj Close'])
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
    
    
    cols = ["volume-change", "amount-change", "sma7-FP", "sma7", "sma25-FP", "sma25", "roc-FP",
 "rsi-FP", "slow-k-FP", "slow-d-FP", "price-change", "price-change-percentage"]

    # Make dataset
    X = df.copy()[cols]
    
    # make prediction
    count = 0
    result = [np.nan for i in range(30 + (before_len - after_len))]
    for i in range(30,len(df)):
        data = X.copy()[count:i]
        for col in cols:
            data[col] = One_One_Scale(data[col])
        data = np.reshape(data.values, (1, 30, len(cols)))
        y = model.predict(data)
        result.append(y[0][-1])
        count += 1
    
    return result


def lstm():
    model = load_model('../models/model14/model.h5')
    pred_df = pd.DataFrame()
    failed_symbol = []
    
    # Load Tickers
    napi = numerapi.SignalsAPI()
    eligible_tickers = pd.Series(napi.ticker_universe(), name='numerai_ticker')
    ticker_map = pd.read_csv('https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_ticker_map_w_bbg.csv')
    yfinance_tickers = eligible_tickers.map(
        dict(zip(ticker_map['bloomberg_ticker'], ticker_map['yahoo']))
    ).dropna()
    numerai_tickers = ticker_map['bloomberg_ticker']
    yfinance_tickers_csv = yfinance_tickers.apply(lambda x: x.replace('.', ''))
    
    for i in yfinance_tickers.index:
        symbol = yfinance_tickers[i]
        if '.TWO' in symbol:
            symbol = symbol.replace('.TWO', '.T')
        elif '.TW' in symbol:
            symbol = symbol.replace('.TW', '.T')
        symbol_for_csv = yfinance_tickers_csv[i]
        symbol_numerai = numerai_tickers[i]
        print('-'*50, symbol, '-'*50)
        try:
            df = yf.download(symbol, start=dt.datetime.now() - dt.timedelta(days=140), end=dt.datetime.now(), interval='1d')
            preds = make_prediction(df, model)
            pred = preds[-1] if len(preds) > 0 else None
            if pred:
                print(round(pred, 5))
                pred_df.loc[symbol, 'signal'] = round(pred, 5)
                pred_df.loc[symbol, 'bloomberg_ticker'] = symbol_numerai
        except:
            print('Failed ', symbol)
            failed_symbol.append(symbol)
            continue

    pred_df.to_csv('numerai-pred.csv')
    failed_df = pd.DataFrame({'symbol': failed_symbol})
    failed_df.to_csv('numerai-fail.csv')
    
lstm()