import os
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
import mlflow
from mlflow import pyfunc
import mlflow.tensorflow
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path('../.env')
load_dotenv(dotenv_path=dotenv_path)

def One_One_Scale(df: np.ndarray):
    df_scaled = 2 * (df - df.min()) / (df.max() - df.min()) - 1
    return df_scaled

def load_data(path: str):
    data = pd.read_csv(path, index_col='Date')
    df= pd.DataFrame()
    f_cols = []
    GSPC_cols = []
    SUBT_cols = []
    for x in [7, 14, 26, 50, 99]:
        f_cols += [
            f"price_return_{x}days",
            f"price_volatility_{x}days",
            f"price_MA_gap_{x}days"
        ]
        GSPC_cols += [
            f"GSPC_price_return_{x}days",
            f"GSPC_price_volatlity_{x}days",
            f"GSPC_price_MA_gap_{x}days"
        ]
        SUBT_cols += [
            f"SUBT_price_return_{x}days",
            f"SUBT_price_volatility_{x}days",
            f"SUBT_price_MA_gap_{x}days"
        ]

    f_cols += ["MACD", "RSI", "Target"]
    GSPC_cols += ["MACD", "RSI", "Target"]
    SUBT_cols += ["MACD", "RSI", "Target"]
    max_iter = len(data) // 5
    
    count = 0
    print('Return Max, Min: ', data['Target'].max(), data['Target'].min())
    for i in list(range(30, max_iter * 5, 5)):
        dataset = data.copy()[count:i]
        for col in f_cols:
            dataset[col] = One_One_Scale(dataset[col])
        df = df.append(dataset[f_cols], ignore_index=True)
        count += 5
    print("Scaled Return Max, Min: ", df['Target'].max(), df['Target'].min())
    return df[f_cols]

def make_train_data():
    data_folder = './dataset/'

    symbols = pd.read_csv('../../../../symbols/sandp500.csv')
    file_names = [str(x) + '.csv' for x in symbols['symbol']]
    exclude_file = ['.ipynb_checkpoints', 'MYL.csv', 'CTL.csv', 'UA.csv', 'QUALCOMM.csv', 'NBL.csv', 'HWM.csv', 'CARR.csv', 'TIF.csv', 'BF.B.csv', 'ETFC.csv', 'CXO.csv', 'AMCR.csv', 'OTIS.csv', 'ANSYS.csv', 'BRK.B.csv', 'VAR.csv']
    file_names = [x for x in file_names if not x in exclude_file]

    df = pd.DataFrame()
    for file in file_names:
        try:
            print('-'*60)
            path = data_folder + file
            data = load_data(path)
            if data.isnull().values.sum() == 0:
                print(f'{file} has {len(data)} data.')
                df = df.append(data)
            else:
                print(f'{file} has {data.isnull().values.sum()} NaN data.')
                continue
        except:
            print(file, 'has some error.')
            continue

    X = df.copy()
    y = X.pop('Target')
    n_features = len(X.columns)
    print(y.shape)
    X = np.reshape(X.values, (len(X) // 30, 30, len(X.columns)))
    y = np.reshape(y.values, (len(y) // 30, 30))
    # y_uni = np.empty([len(X), 1])
    # for i in range(len(X)):
    #     y_uni[i][0] = y[i][-1]

    print("X shape: ", X.shape, "y shape: ", y.shape)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=11)
    return X_train, X_valid, y_train, y_valid, n_features


def create_model(n_features):
    model = keras.Sequential([
        layers.LSTM(514, return_sequences=True, input_shape=(30, n_features)),
        layers.LSTM(514, return_sequences=True),
        layers.LSTM(514),
        layers.Dense(30),
        #layers.Dense(1)
    ])
    model.compile(
        optimizer='adam',
        loss='mae',
        metrics=['mae']
    )
    model.summary()
    return model


def main():
    model_name = 'Base_Line_30'


    mlflow.set_experiment("LSTM Stock Prediction - IDX model")
    mlflow.tensorflow.autolog(every_n_iter=2)
    with mlflow.start_run(run_name=model_name):
        X_train, X_valid, y_train, y_valid, n_features = make_train_data()
        model = create_model(n_features)
        early_stop = callbacks.EarlyStopping(
            min_delta=0.0001,
            patience=20,
            restore_best_weights=True
        )
        history = model.fit(
            X_train, y_train,
            validation_data = (X_valid, y_valid),
            epochs=500,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )

    path = f'models/model{model_name}/'

    if not os.path.exists(path):
        os.mkdir(path)
    
    hist = pd.DataFrame(history.history)
    hist.to_csv(path + 'history.csv')
    model.save(path + 'model.h5')
    res = 'Sucessfully Saved'
    send_line(res)
    return res

def send_line(msg):
    token = os.getenv('LINE_TOKEN')
    end_p = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {token}'}
    data = {'message': msg}
    res = requests.post(end_p, headers=headers, data=data)
    return res.text


if __name__ == '__main__':
    main()
    