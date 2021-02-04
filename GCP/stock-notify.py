import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import requests

def send_line_notify(notification_message):
    """
    LINEに通知する
    """
    line_notify_token = 'HvPqtdmp53Cl6tZyKMIVkMjmBOWOWGyR6W7FG5Np31y'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {notification_message}'}
    requests.post(line_notify_api, headers = headers, data = data)

    
def RSI(stock_code, term=14, start=dt.datetime.now()-dt.timedelta(days=100), end=dt.datetime.now()):
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
    
    # check the conditions
    rsi = df['RSI'][-1]
    close = df['Adj Close'][-1]
    sma_short = df['SMA25'][-1]
    sma_long = df['SMA50'][-1]
    sma_trend = df['SMA25 Trend'][-1]
    if rsi > 50 and close > sma_long or sma_trend > 0:
        buy_price = close
        content = "\n\n{} 👍 Buy at the price ${}".format(stock_code, round(close, 5))
    elif close < sma_long:
        sell_price = close
        content = "\n\n{} 👎 Sell at the price ${}".format(stock_code, round(close, 5))
    else:
        content = "\n\n{} is neutral".format(stock_code)
        
    return content

def main():
    content = '\n\nToday`s stock report!!! @' + str(dt.date.today())
    symbols = ['AMD', 'TEAM', 'CHTR', 'NVDA', 'TSLA', 'APH', 'AJG', 'DXC', 'CE', 'LOW', 'TGT', 'TYL']
    for symbol in symbols:
        try:
            report = RSI(symbol)
            content += '\n'+report
        except:
            content += '\n\n\n 💀 FAIL to Download {} data.'.format(symbol)
            continue
    send_line_notify(content)
    return 'Mission Completed!!'