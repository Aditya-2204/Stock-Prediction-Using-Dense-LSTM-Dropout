import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt

from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

data = yf.download("BTC-AUD", "2016-1-1", dt.datetime.now())

data = pd.DataFrame(data['Close'])

data.index = range(1,len(data.values)+1)

X_train, X_test = data.index[:len(data.index)-50], data.index[len(data.index)-49:]
y_train, y_test = data.values[:len(data.values)-50], data.values[len(data.values)-49:]

X_train, y_train = np.reshape(X_train, (-1,1)), np.reshape(y_train, (-1,1))

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit_transform(data.values.reshape(-1,1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1))


model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, batch_size=32, epochs=25)