import pandas as pd
import numpy as np
import datetime as dt
import requests
import os


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


def create_data(df):
    df = df.copy()
    # basic information
    df['return'] = np.log(df['Close'].shift(-5) / df['Close'])
    df['price-change'] = df['Close'] - df['Close'].shift(1)
    df['price-change-percentage'] = df['Close'] / df['Close'].shift(1)
    # volume
    volume_mid = df['Volume'].median()
    df['Volume'] = df['Volume'].apply(lambda x: volume_mid if x == 0 else x)
    df['volume-change'] = np.log(df['Volume'] / df['Volume'].shift(1))
    # amount
    df['amount'] = df['Close'] * df['Volume']
    df['amount-change'] = np.log(df['amount'] / df['amount'].shift(1))
    # simple moving average
    df['sma7'] = df['Close'].rolling(7).mean()
    df['sma7-FP'] = (df['sma7'] - df['sma7'].shift(1)) / df['sma7'].shift(1)
    df['sma7'] = np.log(df['sma7'] / df["sma7"].shift(1))
    
    df['sma25'] = df['Close'].rolling(25).mean()
    df['sma25-FP'] = (df['sma25'] - df['sma25'].shift(1)) / df['sma25'].shift(1)
    df['sma25'] = np.log(df['sma25']/df['sma25'].shift(1))
    
    # simple moving average difference
    df['smaDiff7-25'] = df['sma7'] - df['sma25']
    df['smaDiff7-25'] = np.where(df['smaDiff7-25'] > 0, 1, -1)

    # Moving Average Convergence Divergence
    df['macd'] = df['Close'].rolling(12).mean() - df['Close'].rolling(26).mean()
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
    df['mtm10'] = df['Close'] - df['Close'].shift(10)
    df['mtm10'] = np.where(df['mtm10'] > 0, 1, -1)
    # Rate of Change in 10 days
    df['roc'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)
    df['roc-SG'] = np.where(df['roc'] > 0, 1, -1)
    df['roc-FP'] = (df['roc'] - df['roc'].shift(1))
    # Relative Strength Index in 5 days
    df['rsi'] = df['price-change'].rolling(5).apply(RSI) / 100
    df['rsi-FP'] = (df['rsi'] - df['rsi'].shift(1))
    # Slow K and Slow D
    df['slow-k'] = df['Close'].rolling(14).apply(SlowK)
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
    sma20 = df['Close'].rolling(20).mean()
    df['bias20'] = (df['Close'] - sma20) / sma20
    df['bias20'] = np.where(df['bias20'] > 0, 1, -1)
    
    # drop row contains NaN
    df.dropna(inplace=True)
    
    length = len(df) // 5
    
    return df[:5 * length]

def save_to_csv(df, sym, save_path):
    df.to_csv(save_path + f'{sym}.csv')
    print(f'{sym}.csv Saved')

def convertUnixToDateTime(x):
    return dt.datetime.fromtimestamp(int(x))

def load_data(market, symbol, freq):
    try:
        url = f'https://api.cryptowat.ch/markets/{market}/{symbol}/ohlc'
        after = dt.datetime.now() - dt.timedelta(days=800)
        res = requests.get(url, params={ 'periods': freq, 'after': int(after.timestamp())})
        item = res.json()
        df = pd.DataFrame(item['result'][str(freq)],
                        columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'QuoteVolume'],
                        )

        df['Time'] = df['Time'].apply(convertUnixToDateTime)
        df = df.set_index('Time')
        df.to_csv(f'./crypto_data/{symbol}-{freq}.csv')
        return df
    except:
        print(f'Loading {symbol} data is failded.')
        return None

def send_line_notify(notification_message):
    """
    LINEに通知する
    """
    line_notify_token = 'HvPqtdmp53Cl6tZyKMIVkMjmBOWOWGyR6W7FG5Np31y'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {notification_message}'}
    requests.post(line_notify_api, headers = headers, data = data)

def main():
    try:
        save_path = '../../../Data/LSTM_Crypto_2/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        cols = ['return', 'price-change', 'price-change-percentage', 'volume-change', 'amount-change', 'sma7', 'sma7-FP', 'sma25', 'sma25-FP', 'smaDiff7-25',
            'macd', 'macd-SG', 'macd-histogram', 'cci-SG', 'mtm10', 'roc-SG', 'roc-FP', 'rsi', 'rsi-FP', 'slow-k', 'slow-d',
            'slow-k-FP', 'slow-d-FP', 'adosc', 'adosc-SG', 'ar26', 'br26', 'bias20']
        symbols = ['btcusdt', 'etcusdt', 'ethusdt', 'bnbusdt', 'xmrusdt', 'ltcusdt', 'adausdt', 'dotusdt', 'linkusdt', 'xlmusdt', 'xrpusdt', 'bchusdt', 'iostusdt', 'dogeusdt', 'solusdt', 'maticusdt', 'zilusdt']
        for sym in symbols:
            df = load_data(market='binance', symbol=sym, freq=7200)
            tech_df = create_data(df)
            save_to_csv(tech_df[cols], sym, save_path)
        send_line_notify('Suncess')
    except:
        send_line_notify('Failed!!!')


if __name__ == '__main__':
    #main()
    symbols = ['btcusdt', 'etcusdt', 'ethusdt', 'bnbusdt', 'xmrusdt', 'ltcusdt', 'adausdt', 'dotusdt', 'linkusdt', 'xlmusdt', 'xrpusdt', 'bchusdt', 'iostusdt', 'dogeusdt', 'solusdt', 'maticusdt', 'zilusdt']
    for sym in symbols:
        load_data(market='binance', symbol=sym, freq=3600)
    for sym in symbols:
        load_data(market='binance', symbol=sym, freq=7200)