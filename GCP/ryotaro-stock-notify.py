import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import requests

def send_line_notify(notification_message):
    """
    LINEに通知する
    """
    line_notify_token = 'UzQC6Oh3W3rALdiXFnkEkFf14cxG3lLFNLhkYE22Vlm'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {notification_message}'}
    requests.post(line_notify_api, headers = headers, data = data)

    
def RSI(stock_code, term=14, start=dt.datetime.now()-dt.timedelta(days=160), end=dt.datetime.now()):
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
    df['Highest80'] = df['Adj Close'].rolling(window=80).max()

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
    highest = df['Highest80'][-2]
    if rsi > 50 and close > sma_long and close > highest*0.95 and close < highest and sma_trend > 0:
        buy_price = close
        content = "\n\n{} 👍 \nBuy Price Tommorow: ${}\nPrevious Close Price: ${}".format(stock_code, round(highest, 5), round(close, 5))
    elif close < sma_long:
        sell_price = close
        content = "\n\n{} 👎 Sell".format(stock_code)
    else:
        content = 'unko'
        
    return content

def main():
    content = '\n\nToday`s stock report!!!' + str(dt.date.today())
    symbols = ['6976.T', '6036.T', '6758.T', '6966.T', '6544.T', '2980.T', '7190.T', '3768.T']
    for symbol in symbols:
        try:
            report = RSI(symbol)
            if not report == 'unko':
                content += '\n'+report
            else:
                continue
        except:
            content += '\n\n 💀 FAIL to Download {} data.'.format(symbol)
            continue
    send_line_notify(content)
    return 'Mission Completed!!'
main()