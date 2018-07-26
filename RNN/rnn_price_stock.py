#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 21:51:01 2018

@author: thiago
"""

# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values # to get only column 1. .values to create numpy array

# Feature Scaling --> To optimize the training, avoiding one input to be to strong
# Types os Feature Scaling
    # Standardisation
    # Normalisations --> Very common in RNN
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1)) # --> To scale beteween 0 & 1
training_set_scaled = sc.fit_transform(training_set)


# Creating a data structure with 60 timesteps and 1 output
# Super important --> To avoid overfitting or wrong predictions
# This 60 you get by trying. It's inportant to find the best number

# input of the network
# for each day/observation, X will contain the 60 previews days informations
# For text, we would change here for how long the text it should remember
# This is the memorization step
X_train = []

# output of the network
# Will contain the stock for next day.
# or next letter, for text problem
y_train = []

# populate X & y train
# from 60th day to 1258, than means 2012 to 2016
# We should change this, depending on your data(text for example)
# start with 60, because you need 60 day to then add to Y
for i in range(60, 1258):              # to get 60 values
    X_train.append(training_set_scaled[i-60:i, 0]) # 0 for the column
    y_train.append(training_set_scaled[i, 0]) # to get the next value, which the RNN has to guess

# convert to numpy
X_train, y_train = np.array(X_train), np.array(y_train)


# Reshaping --> Very inportant -- to numpy, if you have multiple dimensions
# This is very inportant, we add more dimensions for the dataset
# Why so inportant?
# Because we add the numeber of predictors we can use to predict what we wanna
# This means that we can use more than just the last price, but multiple indicators
# to help the prediction
# For each indicator, add new dimension
# Paramters --> Look more in keras documentation
    # object to reshape
    # new shape to convert to
        # Batchsize --> size of data -- total of observations
            # X_train.shape[0] gets how many lines
        # timesteps --> 60 in our case
            # X_train.shape[1] gets how many columns
        # input_dim --> Numb of indicators, in our case, one
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM # RNN
from keras.layers import Dropout # used for regularization, avoiding overfitting


# Initialising the RNN, as always
regressor = Sequential()


# Adding the first LSTM layer and some Dropout regularisation
# Paramters
    # LSTM --> RNN with LSTM
    # units --> number of LSM cells/Neurons/Nodes.
        # Why 50?
            # because the problem is complex. For more simples, less neurons
            # This creates the dimensionality
    # return_sequences --> = True when you know you gonna have one more LSTM layer. False as default
    # input_shape --> no need to specify X_train.shape[0], because it inserts automatic
        # just need in the first layer

    # Dropout --> dropout rate. percentage of neurons to ignore when training

    # optimizer --> which gradient descent to use
        # adam
        # RMSprop
        # etc
    # loss -->  which loss descent to use
        # mean_squared_error --> Used for regression, not classification

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2)) # to avoid overfitting

# Why to add more LSTM layers?
    # to create a robost structure

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50)) # return_sequences = False, because it's the last one
regressor.add(Dropout(0.2))


# Adding the output layer --> Like ANN
regressor.add(Dense(units = 1)) # one neuron

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Trainig Part
# Fitting the RNN to the Training set
# epochs = number of repetitions
# batch_size = how to divide the data to be trained
    # after 32 stock prices, that we gonna run the backpropagation
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()






