import os
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split

def Zero_One_Scale(df):
    df_scaled = (df - df.min()) / (df.max() - df.min())
    return df_scaled

def One_One_Scale(df):
    df_scaled = 2 * (df - df.min()) / (df.max() - df.min()) - 1
    return df_scaled

def Custom_Scale(df, min_val=-0.1, max_val=0.1):
    df = np.where(df > max_val, max_val, df)
    df = np.where(df < min_val, min_val, df)
    if df.max() > max_val or df.min() < min_val:
        print(df.max(), df.min())
    df_scaled = 2 * ((df - min_val) / (max_val - min_val)) - 1
    return df_scaled

one_one_cols = ["return", "volume-change", "amount-change", "sma7-FP", "sma7", "sma25-FP", "sma25", "roc-FP",
 "rsi-FP", "slow-k-FP", "slow-d-FP", "price-change", "price-change-percentage"]

zero_one_cols = ["slow-k", "slow-d", "adosc", "ar26", "br26"]

cols = ["return", "volume-change", "amount-change", "sma7-FP", "sma7", "sma25-FP", "sma25", "roc-FP",
 "rsi-FP", "slow-k-FP", "slow-d-FP", "price-change", "price-change-percentage", "Date"]

n_features = len(one_one_cols) - 1

def load_data(symbol):
    path = '../../Data/LSTM_Seven/{}'.format(symbol)
    data = pd.read_csv(path)
    df = pd.DataFrame()
    max_iter = len(data) // 5
    count = 0
    for i in list(range(30, max_iter * 5, 5)):
        dataset = data.copy()[count:i]
        for col in one_one_cols:
            if col == 'return':
                dataset[col] = Custom_Scale(dataset[col])
            else:
                dataset[col] = One_One_Scale(dataset[col])
        
        df = df.append(dataset[cols], ignore_index=True)
        count += 5
    return df.drop('Date', axis=1)

def create_train_data():
    #files = os.listdir('../../Data/LSTM_Six')
    symbols = pd.read_csv('../../../symbols/sandp500.csv')
    file_names = [str(x) + '.csv' for x in symbols['symbol']]

    exclude_file = ['.ipynb_checkpoints', 'MYL.csv', 'CTL.csv', 'UA.csv', 'QUALCOMM.csv', 'NBL.csv', 'HWM.csv', 'CARR.csv', 'TIF.csv', 'BF.B.csv', 'ETFC.csv', 'CXO.csv', 'AMCR.csv', 'OTIS.csv', 'ANSYS.csv', 'BRK.B.csv']
    files = [i for i in file_names if not i in exclude_file]
    df = pd.DataFrame()
    for file in files:
        try:
            data = load_data(file)
            print('{} has {} data.'.format(file, len(data)))
            df = df.append(data)
        except:
            print(file, 'has some error')
            continue

    X = df.copy()
    y = X.pop('return')
    #float_cols = [i for i in X.columns if X[i].dtype == float]
    #int_cols = [i for i in X.columns if X[i].dtype == int]
    # Normalization
    #X[float_cols] = (X[float_cols] - X[float_cols].mean(axis=0)) / X[float_cols].std(axis=0)
    
    X = np.reshape(X.values, (len(X)//30 , 30, n_features))
    y = np.reshape(y.values, (len(y)//30, 30))

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=11)
    return X_train, X_valid, y_train, y_valid

def create_model():
    model = keras.Sequential([
        layers.LSTM(1024, return_sequences=True, input_shape=(30, n_features)),
        layers.LSTM(1024, return_sequences=True),
        layers.LSTM(1024),
        layers.Dense(30)
    ])

    model.compile(
        optimizer='adam',
        loss='mae',
        metrics=['mae']
    )
    model.summary()
    return model

def train_model(model, X_train, X_valid, y_train, y_valid):
    early_stoping = callbacks.EarlyStopping(
        min_delta=0.0001,
        patience=20,
        restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=500,
        batch_size=32,
        callbacks=[early_stoping],
        verbose=1
    )

    path = './models/model18/'

    if not os.path.exists(path):
        os.mkdir(path)
    
    hist = pd.DataFrame(history.history)
    hist.to_csv(path + 'history.csv')
    model.save(path + 'model.h5')
    print('Model Suceccfully saved.')
    
def send_line_notify(notification_message):
    """
    LINEに通知する
    """
    line_notify_token = 'HvPqtdmp53Cl6tZyKMIVkMjmBOWOWGyR6W7FG5Np31y'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {notification_message}'}
    requests.post(line_notify_api, headers = headers, data = data)

try:
    X_train, X_valid, y_train, y_valid = create_train_data()
    model = create_model()
    train_model(model, X_train, X_valid, y_train, y_valid)
    send_line_notify('Successfully Completed.')
except:
    import traceback
    send_line_notify("Process has Stopped with some error!!!")
    send_line_notify(traceback.format_exc())
    print(traceback.format_exc())