import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpl
import yfinance as yf
import datetime as dt

stock_codes = ['AAL','AAPL','AMD','ALGN','ADBE','ADI','ADP','ADSK','ALXN','AMAT',
'AMGN','AMZN','ASML','ATVI','BIDU','BIIB','BMRN','BKNG','CDNS',
'CERN','CHKP','CMCSA','CSCO','CSX','CTAS','CTSH',
'CTXS','DLTR','EA','EBAY','EXPE','FAST','FB','FISV','FOXA',
'GILD','GOOG','GOOGL','HAS','HSIC','HOLX','IDXX','ILMN','INCY',
'INTC','ISRG','JBHT','JD','KHC','KLAC','LBTYB','LBTYA',
'LBTYK','LULU','LILA','LILAK','LRCX','MAR','MCHP','MELI','MNST','MU','MXIM','MELI','NFLX','NTES','NVDA','NXPI',
'ORLY','PAYX','PCAR','PYPL','PEP','QCOM','REGN','ROST','SBUX',
'SNPS','SIRI','SWKS','TMUS','TTWO','TSLA','KHC',
'ULTA','UAL','VRSN','VRSK','VRTX','WBA','WDC','WLTW','WDAY','XEL']

def stock_validation(stock_code, start="2016-01-01", end=dt.datetime.now(), interval="1d"):
    df = yf.download(stock_code, start, end, interval='1d')

    df['MA'] = df['Adj Close'].rolling(window=20).mean()
    df['STD'] = df['Adj Close'].rolling(window=20).std()
    df['Upper'] = df['MA'] + (df['STD'] * 2)
    df['Lower'] = df['MA'] - (df['STD'] * 2)


    position = 0
    counter = 0
    percentChange = []
    for i in df.index:
        if np.isnan(df['MA'][i]):
            continue
        else:
            upper = df['High'][i]
            lower = df['Low'][i]
            close = df['Adj Close'][i]
            if lower < df['Lower'][i]:
                # up trend
                if position == 0:
                    position = 1
                    buy_price = close

            elif upper > df['Upper'][i]:
                # down trend
                if position == 1:
                    position = 0
                    sell_price = close
                    percent = (sell_price / buy_price - 1)*100
                    percentChange.append(percent)
            if counter == df['Adj Close'].count() - 1 and position == 1:
                position = 0


    gains = 0
    numGains = 0
    losses = 0
    numLosses = 0
    total_return = 1
    for i in percentChange:
        if i > 0:
            gains += i
            numGains += 1
        else:
            losses += i
            numLosses += 1
        total_return = total_return*((i/100) + 1)
    total_return = round((total_return - 1)*100, 2)

    #print('Total return over {} trades: {}%'.format(numGains+numLosses, total_return))

    if (numGains > 0):
        average_gain = gains / numGains
        max_return = max(percentChange)
    else:
        average_gain = 0
        max_return = 'unknown'

    if (numLosses > 0):
        average_loss = losses / numLosses
        max_loss = min(percentChange)
        risk_reward_ratio = - average_gain / average_loss
    else:
        average_loss = 0
        max_loss = 'unknown'
        risk_reward_ratio = 'inf'

    if (numGains > 0 or numLosses > 0):
        batting_ave = numGains / (numGains + numLosses)
    else:
        batting_ave = 0
    trades = numGains + numLosses
        
    return [trades, total_return, average_gain, average_loss, max_return, max_loss, risk_reward_ratio, batting_ave]

results = []
for stock_code in stock_codes:
    results.append(stock_validation(stock_code=stock_code))

columns = ['trades', 'Total return', 'Average Gain', 'Average Loss', 'Max Return', 'Max Loss', 'Gain/Loss Ratio', 'Batting Average']
df = pd.DataFrame(results, columns=columns, index=stock_codes)
df.to_csv('./result.csv')

trades_ave = df['trades'].mean()
total_return_ave = df['Total return'].mean()
average_gain_ave = df['Average Gain'].mean()
average_loss_ave = df['Average Loss'].mean()
average_max_return = df['Max Return'].mean()
average_max_loss = df['Max Loss'].mean()
average_risk_reward_ratio = df['Gain/Loss Ratio'].mean()
average_batting_ratio = df['Batting Average'].mean()
print('trade average: {}'.format(trades_ave))
print('total return average: {}'.format(total_return_ave))
print('gain average: {}'.format(average_gain_ave))
print('loss average: {}'.format(average_loss_ave))
print('max return average: {}'.format(average_max_return))
print('max loss average: {}'.format(average_max_loss))
print('Gain/Loss ratio average: {}'.format(average_risk_reward_ratio))
print('Batting ratio average: {}'.format(average_batting_ratio))