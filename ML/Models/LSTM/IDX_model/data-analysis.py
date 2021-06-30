import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import yfinance as yf
import datetime as dt

# Load Data
symbols = ['^GSPC', '^IXIC', 'GOOG', 'MSFT', 'AMZN', 'AAPL', 'MGM', 'APA', 'PXD', 'HUBS']
data = {}
for sym in symbols:
    df = yf.download(sym, start='2019-12-01', end=dt.datetime.now(), interval='1d')
    data[sym] = df

# Daily return
for sym in symbols:
    d = data[sym]
    d['daily_return'] = d['Adj Close'].shift(-1) / d['Adj Close'] - 1


# Visualize return corralation
fig, axs = plt.subplots(2, 5, figsize=(20, 10))
axs = axs.flatten()
count = 0
for sym in symbols:
    ax = axs[count]
    ax.set_xlabel(sym)
    ax.set_ylabel('^GSPC')
    df = pd.DataFrame({
        f'{sym}_DR': data[sym]['daily_return'],
        '^GSPC_DR': data['^GSPC']['daily_return'] 
    })
    sns.regplot(x=f'{sym}_DR', y='^GSPC_DR', data=df, ax=ax)
    count += 1
plt.tight_layout()
plt.savefig('return_plot.png')


# Calcualte technical indicators
# Technical Indicators
def RSI(close: pd.DataFrame, period: int=14) -> pd.Series:
    delta = close.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    _gain = up.ewm(com=(period - 1), min_periods=period).mean()
    _loss = down.abs().ewm(com=(period - 1), min_periods=period).mean()
    RS = _gain / _loss
    return pd.Series(100 - (100 / (1 + RS)))

def MACD(close: pd.DataFrame, span1=12, span2=26):
    exp1 = close.ewm(span=span1, adjust=False).mean()
    exp2 = close.ewm(span=span2, adjust=False).mean()
    macd = exp1 - exp2
    return macd

def feature_enginieering(ts: np.ndarray):
    # Fill na
    ts.fillna(method='ffill', inplace=True)
    ts.fillna(method='bfill', inplace=True)
    
    feats = pd.DataFrame(index=ts.index)
    feats['price'] = ts
    
    features = [
        'price'
    ]
    
    new_feats = []
    for i, f in enumerate(features):
        for x in [7, 14, 26, 50, 99]:
            # Return
            feats[f"{f}_return_{x}days"] = feats[
                f
            ].pct_change(x)
            
            # volatility
            feats[f"{f}_volatility_{x}days"] = (
                np.log(feats[f])
                  .pct_change()
                  .rolling(x)
                  .std()
            )
            
            # gap mean
            feats[f"{f}_MA_gap_{x}days"] = feats[f] / (
                feats[f].rolling(x).mean()
            )
            
            # features to use
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
    
    # Drop nan
    feats.dropna(inplace=True)
    
    return feats

feats_data = {}

for sym in symbols:
    d = data[sym]
    feats_data[sym] = feature_enginieering(d['Adj Close'])
    

# Combine Index (^GSPC) and other data
gspc = feats_data['^GSPC']
company_symbols = [i for i in symbols if i not in ['^GSPC', '^IXIC']]
for sym in company_symbols:
    d = feats_data[sym]
    for col in gspc.columns:
        d[f'^GSPC_{col}'] = gspc[col]
        d[f'IDX_Subt_{col}'] = d[f'^GSPC_{col}'] - d[col]


# visualize the correlations of symbol and SP500 features
# Overview
df = feats_data['AAPL']
fig, axs = plt.subplots(3, 6, figsize=(20, 20))
axs = axs.flatten()
count = 0
for col in feats_data['^GSPC'].columns:
    plot_data = pd.DataFrame({
        'x': df[col],
        'y': df[f"^GSPC_{col}"]
    })
    sns.scatterplot(x='x', y='y', data=plot_data, ax=axs[count])
    axs[count].set_xlabel(col)
    axs[count].set_ylabel(f"^GSPC_{col}")
    count += 1
plt.tight_layout()
plt.savefig('features_plot.png')


# Visualize the correlations of Target (return in ten days) and the features
fig, axs = plt.subplots(9, 6, figsize=(20, 25))
axs = axs.flatten()
count = 0
df = feats_data['MSFT']
for col in df.columns:
    plot_data = pd.DataFrame({
        'x': df['Target'],
        'y': df[col]
    })
    sns.regplot(x='x', y='y', data=plot_data, ax=axs[count])
    axs[count].set_xlabel('Target')
    axs[count].set_ylabel(col)
    count += 1
plt.tight_layout()


# Visualize time series graphs of symbol, SP500 and these subtraction.
fig, axs = plt.subplots(4, 5, figsize=(20, 20))
axs = axs.flatten()
df = feats_data['AMZN']
count = 0
for col in df.columns:
    if not '^GSPC' in col and not 'IDX' in col:
        ax = axs[count]
        sns.lineplot(data=df[[col, f"^GSPC_{col}", f"IDX_Subt_{col}"]], ax=ax)
        ax.set_xlabel('Time')
        ax.set_ylabel(col)
        ax.legend()
        count += 1

plt.tight_layout()
plt.savefig('timeplot.png')


# Visualize the distributions of the features
fig, axs = plt.subplots(9, 6, figsize=(20, 25))
axs = axs.flatten()
df = feats_data['HUBS']
count = 0
for col in df.columns:
    ax = axs[count]
    sns.histplot(df[col], ax=ax)
    ax.set_xlabel(col)
    ax.set_ylabel('count')
    count += 1
plt.tight_layout()