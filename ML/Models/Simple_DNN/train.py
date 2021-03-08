from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping


def create_model():
    model = keras.Sequential([
        layers.BatchNormalization(input_shape=[5]),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(0.3),
        layers.BatchNormalization(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

lags = 5
model = create_model()
model.summary()

import pandas as pd
df = pd.read_csv('../../Data/lags/lags5_with_sma_dataset.csv', index_col='Date')
df.describe()

from sklearn.model_selection import train_test_split

def create_cols_with_sma():
    cols = []
    for lag in range(1, lags+1):
        col = 'lag_{}'.format(lag)
        cols.append(col)
    
    for lag in range(1, lags+1):
        col = 'sma1_lag_{}'.format(lag)
        cols.append(col)
        
    for lag in range(1, lags+1):
        col = 'sma2_lag_{}'.format(lag)
        cols.append(col)
        
    return cols

cols = create_cols_with_sma()
X = df[cols[:5]]
y = df['direction']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

#cols = ['lag_{}'.format(i) for i in range(1, lags+1)]
from tensorflow.keras import layers, callbacks

early_stopping = callbacks.EarlyStopping(
        min_delta=0.001,
        patience=20,
        restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=500,
    batch_size=256,
    verbose=1,
    callbacks=[early_stopping],
)


import requests
def send_line_notify(notification_message):
    """
    LINEに通知する
    """
    line_notify_token = '***'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {notification_message}'}
    requests.post(line_notify_api, headers = headers, data = data)

send_line_notify('completed')