# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 16:42:10 2020

@author: Graciela
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, LSTM
from tensorflow.keras.layers import Dropout

training_data_unscaled = pd.read_csv('FB_training_data.csv')
training_data = training_data_unscaled.iloc[:, 1].values  #input

scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data.reshape(-1, 1))

x_training_data = []
y_training_data =[]


for i in range(40, len(training_data)):

    x_training_data.append(training_data[i-40:i, 0])

    y_training_data.append(training_data[i, 0])

x_training_data = np.array(x_training_data)
y_training_data = np.array(y_training_data)

#para que pueda ser usado por la RNA en TensorFlow

x_training_data = np.reshape(x_training_data, (x_training_data.shape[0], 
                                               x_training_data.shape[1], 
                                               1))

rnn = Sequential()
#45 neurons
#return_sequences = True - this must always be specified if you plan on 
                          #including another LSTM layer after the one you’re 
                           #adding. 
#return_sequences = False - for the last LSTM layer in your recurrent neural network.
#input_shape: the number of timesteps and the number of predictors in our 
                          #training data. In our case, we are using 40 timesteps 
                           #and only 1 predictor (stock price), so we will add

rnn.add(LSTM(units = 45, 
             return_sequences = True, 
             input_shape = (x_training_data.shape[1], 1)))


rnn.add(Dropout(0.2))

rnn.add(LSTM(units = 45, return_sequences = True))


rnn.add(Dropout(0.2))

rnn.add(LSTM(units = 45))  #by default is False

rnn.add(Dropout(0.2))

rnn.add(Dense(units = 1))

rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')

rnn.fit(x_training_data, y_training_data, epochs = 100, batch_size = 32)


#%% 
#Testing --- solo tiene 20 muestras. 
#como la ventana es de 40 se van a concatenar las ultimas 40 del training


unscaled_training_data = pd.read_csv('FB_training_data.csv')
unscaled_test_data = pd.read_csv('FB_test_data.csv')
all_data = pd.concat((unscaled_training_data['Open'], unscaled_test_data['Open']), axis = 0)

#aqui agarra las ultimas 40 del train y todas las del test
x_test_data = all_data[len(all_data) - len(unscaled_test_data) - 40:].values

x_test_data = scaler.transform(x_test_data.reshape(-1,1))

x_test_data_original = unscaled_test_data.iloc[:, 1].values  #input

# Hay que arreglarlo en las ventanitas de 40 samples

final_x_test_data = []

for i in range(40, len(x_test_data)):

    final_x_test_data.append(x_test_data[i-40:i, 0])

final_x_test_data = np.array(final_x_test_data)

#para que pueda usarse en la red 

final_x_test_data = np.reshape(final_x_test_data, (final_x_test_data.shape[0], 

                                               final_x_test_data.shape[1], 

                                               1))

predictions = rnn.predict(final_x_test_data)

unscaled_predictions = scaler.inverse_transform(predictions)


plt.plot(unscaled_predictions)

plt.plot(unscaled_predictions, color = 'red', label = "Predictions")

plt.plot(x_test_data_original , color = 'black', label = "Real Data")

plt.title('Facebook Stock Price Predictions')
