import os
import pandas as pd
files = os.listdir('../../Data/LSTM_Five')
exclude_file = []
for i in files:
    try:
        if len(pd.read_csv('../../Data/LSTM_Five/' + i)) == 0:
            exclude_file.append(i)
    except:
        exclude_file.append(i)

print(exclude_file)
# print(len(files))
exclude_file = ['.ipynb_checkpoints', 'CTL.csv', 'NBL.csv', 'BF.B.csv', 'ETFC.csv', 'BRK.B.csv']
files = [i for i in files if not i in exclude_file]
print(len(files))

# path = './models/model2/'

# if not os.path.exists(path):
#     os.mkdir(path)

# cols = ['return', 'price-change', 'price-change-percentage', 'volume', 'amount', 'sma7', 'sma7-FP', 'sma25', 'sma25-FP', 'smaDiff7-25',
#         'macd', 'macd-SG', 'macd-histogram', 'cci-SG', 'mtm10', 'roc-SG', 'roc-FP', 'rsi', 'rsi-FP', 'slow-k', 'slow-d',
#         'slow-k-FP', 'slow-d-FP', 'adosc', 'adosc-SG', 'ar26', 'br26', 'bias20']

# print(len(cols))

# import yfinance as yf
# import numpy as np
# df = yf.download('4151.T', start='2016-01-01', end='2019-12-31', interval='1d')
# volume_mid = df['Volume'].median()
# df['Volume'] = df['Volume'].apply(lambda x: volume_mid if x == 0 else x)
# df['return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
# print(df.loc[df['Volume'] == 0])