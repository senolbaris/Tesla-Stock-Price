import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

data = pd.read_csv("Tesla.csv - Tesla.csv.csv")
data_open = data["Open"].values

data_open = data_open.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_open)

train_data = scaled_data[0:1260, :]
test_data = scaled_data[1260: , :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
	x_train.append(train_data[i-60:i, 0])
	y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

model = Sequential()
model.add(LSTM(64,return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(64))
model.add(Dense(32))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error", metrics=["acc"])

model.fit(x_train, y_train, batch_size=50, epochs=10)

x_test = []
y_test = []

for i in range(60, len(test_data)):
	x_test.append(test_data[i-60:i, 0])
	y_test.append(test_data[i, 0])

x_test = np.array(x_test)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

pre = model.predict(x_test)
pre = scaler.inverse_transform(pre)
y_test = scaler.inverse_transform([y_test])
test_score = np.sqrt(np.mean(pre - y_test)**2)
print(test_score)
