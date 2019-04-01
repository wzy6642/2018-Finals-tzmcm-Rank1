# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 19:09:25 2019

@author: wzy
"""
import pandas as pd
from copy import deepcopy
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
import keras as K
from keras.models import Sequential
from keras.layers import Dense

def load_train_data(filename):
    data = pd.read_csv(filename)
    ID = list(deepcopy(data['shares_ID']))
    data.drop('shares_name', axis=1, inplace=True)
    data.drop('shares_ID', axis=1, inplace=True)
    data.index = ID
    time = list(data.columns)
    end = time.index('2018-01-02')
    time = time[:end]
    data = data[time]
    data = data.fillna(0)
    data.drop(['603056.SH', '601828.SH', '601838.SH', '002925.SZ'], inplace=True)
    ID = list(deepcopy(data.index))
    return data, ID

def load_test_data(filename):
    data = pd.read_csv(filename)
    ID = list(deepcopy(data['shares_ID']))
    data.drop('shares_name', axis=1, inplace=True)
    data.drop('shares_ID', axis=1, inplace=True)
    data.index = ID
    time = list(data.columns)
    begin = time.index('2018-01-02')
    time = time[begin:]
    data = data[time]
    data = data.fillna(0)
    data.drop(['603056.SH', '601828.SH', '601838.SH', '002925.SZ'], inplace=True)
    ID = list(deepcopy(data.index))
    return data, ID

def count(list1, days):
    time_series = list(np.arange(0, len(list1), days))
    time_series_mean = []
    for i in time_series:
        temp = list1[i:i+days]
        mean = sum(temp) / days
        time_series_mean.append(mean)
    return time_series_mean
def time_step_count(data, days, ID):
    shares_value = []
    samples = []
    for i in ID:
        share_value = data.loc[[i], :]
        share_value = list(list(share_value.values)[0])
        shares_value.append(share_value)
    for i in shares_value:
        temp = deepcopy(i)
        sample = count(temp, days)
        samples.append(sample)
    samples = pd.DataFrame(samples)
    samples.index = ID
    return samples

def rmse(y_true, y_pred):
    return K.backend.sqrt(K.backend.mean(K.backend.square(y_pred - y_true), axis=-1)) 

def create_nn_model(input_dim, activation, layers, optimizer):
    model = Sequential()
    l_num = 0
    for l in layers:
        if l_num == 0:
            model.add(Dense(l, input_dim=input_dim, activation=activation, init='he_normal'))
        else:
            model.add(Dense(l, activation=activation, init='he_normal'))
        l_num = l_num + 1
    model.compile(optimizer=optimizer, loss=rmse)
    return model


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    filename = '../data/shares.csv'
    days = 5    # 1:日  5:周  22:月
    train_data, train_ID = load_train_data(filename)
    test_data, test_ID = load_test_data(filename)
    train_data_series = time_step_count(train_data, days, train_ID)
    test_data_series = time_step_count(test_data, days, test_ID)
    label_column = train_data_series.size/len(train_data_series)-1
    train_label = list(deepcopy(train_data_series[label_column]))
    train_data_series.drop(label_column, axis=1, inplace=True)
    scaler = StandardScaler().fit(train_data_series)
    train = scaler.transform(train_data_series)
    X = train
    y = train_label
    model = create_nn_model(input_dim=X.shape[1], activation='softplus', layers=[600, 1], optimizer='adagrad')
    fit = model.fit(X, y, batch_size=1, nb_epoch=400, validation_split=0.1, verbose=2)
    out1 = model.predict(train)

