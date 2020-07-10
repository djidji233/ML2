import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

# Data cleanup
data = pd.read_csv('yahoo.csv', index_col='Date')
closed = data.iloc[:, 4:5].values
training_set = closed[:round(len(closed)*0.8)]
testing_set = closed[round(len(closed)*0.8):]

sc = MinMaxScaler(feature_range=(0,1)) # normalization
training_set_scaled = sc.fit_transform(training_set)

x_train = []
y_train = []
for i in range(60, len(training_set)):
  x_train.append(training_set_scaled[i-60:i, 0])
  y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# Model & Training
model = Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1], 1))) 
model.add(LSTM(units=50,return_sequences=True))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2)) # prevents overfitting
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) # units = dimensionality of the output space
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs=50,batch_size=16) # batch_size = number of samples per gradient update

# Test
testing_set = testing_set.reshape(-1,1)
testing_set = sc.transform(testing_set)
x_test = []
for i in range(60, testing_set.shape[0]):
  x_test.append(testing_set[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_stock_price = model.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Plotting
plt.plot(closed[len(closed)-len(predicted_stock_price):], color = 'black', label = 'Yahoo Stock Price')
plt.plot(predicted_stock_price, color = 'red', label = 'Predicted Yahoo Stock Price')
plt.title('Yahoo Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Yahoo Stock Price')
plt.legend()
plt.show()