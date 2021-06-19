import pandas as pd
import yfinance as yf
import datetime as dt
import os
import requests
import numpy as np
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path('../.env')
load_dotenv(dotenv_path=dotenv_path)

# Technical Indicators
def RSI(close: pd.DataFrame, period: int=14) -> pd.Series:
    delta = close.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    _gain = up.ewm(com=(period-1), min_periods=period).mean()
    _loss = down.ewm(com=(period-1), min_periods=period).mean()
    RS = _gain / _loss
    return pd.Series(100 - (100 / (1 + RS)))

def MACD(close: pd.DataFrame, span1: int=12, span2: int=26):
    exp1 = close.ewm(span=span1, adjust=True).mean()
    exp2 = close.ewm(span=span2, adjust=True).mean()
    macd = exp1 - exp2
    return macd

def feature_enginieering(ts: np.ndarray):
    # Fill na
    ts.fillna(method='ffill', inplace=True)
    ts.fillna(method='bfill', inplace=True)

    feats = pd.DataFrame(index=ts.index)
    feats['price'] = ts

    features = ['price']
    new_feats = []

    for i,f in enumerate(features):
        for x in [7, 14, 26, 50, 99]:
            # Return
            feats[f"{f}_return_{x}days"] = feats[f].pct_change(x)

            # Volatility
            feats[f"{f}_volatility_{x}days"] = (
                np.log(feats[f])
                  .pct_change()
                  .rolling(x)
                  .std()
            )

            # Gap Mean
            feats[f"{f}_MA_gap_{x}days"] = feats[f] / (
                feats[f].rolling(x).mean()
            )

            # Features to use
            new_feats += [
                f"{f}_return_{x}days",
                f"{f}_volatility_{x}days",
                f"{f}_MA_gap_{x}days"
            ]

    # RSI
    rsi_vec = RSI(feats['price'], 14)
    feats['RSI'] = rsi_vec.values
    new_feats += ['RSI']

    # MACD
    feats['MACD'] = MACD(feats['price'], 12, 26)
    new_feats += ['MACD']

    # Target
    feats['Target'] = (feats['price'].shift(-10) / feats['price']) - 1
    new_feats += ['Target']

    feats = feats[new_feats]

    # Drop NAN
    feats.dropna(inplace=True)

    # Length
    length = len(feats) // 5

    return feats[:5*length]

def save_csv(df: pd.DataFrame, sym: str, save_path: str):
    #sym = sym.replace('.T', '')
    df.to_csv(save_path + f'{sym}.csv')
    print(f'{sym}.csv Saved')


# Line Notify
def send_line(msg: str):
    token = os.getenv('LINE_TOKEN')
    end_p = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {token}'}
    data = {'message': msg}
    res = requests.post(end_p, headers=headers, data=data)
    return res.text

# Main Process
if __name__ == '__main__':
    #try:
    save_path = 'dataset/'
    start = '2016-01-01'
    end = '2019-12-31'

    paths = ['../../../../symbols/sandp500.csv', '../../../../symbols/nasdaq100.csv']
    gspc = yf.download('^GSPC', start=start, end=end, interval='1d')
    feats_gspc = feature_enginieering(gspc['Adj Close'])

    for path in paths:
        symbols = pd.read_csv(path)
        for sym in symbols['symbol']:
            print('-' * 60)
            print(sym)
            df = yf.download(sym, start=start, end=end, interval='1d')
            df = feature_enginieering(df['Adj Close'])
            for col in df.columns:
                df[f'GSPC_{col}'] = feats_gspc[col]
                df[f'SUBT_{col}'] = feats_gspc[col] - df[col]
            save_csv(df, sym, save_path)

    send_line('Make Dataset of IDX model has ended')
    # except:
    #     import traceback
    #     send_line('Error while making dataset of IDX model:')
    #     send_line(traceback.format_exc())