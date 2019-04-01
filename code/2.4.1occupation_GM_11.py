# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 14:51:10 2019
短时期用灰色预测做
@author: wzy
"""
import GM_11
import pandas as pd
from copy import deepcopy
import warnings
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, isnan

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

def predict(data, ID, count=1):
    shares_value = []
    Result = []
    for i in ID:
        share_value = data.loc[[i], :]
        share_value = list(list(share_value.values)[0])
        shares_value.append(share_value)
    for i in shares_value:
        gf = GM_11.GrayForecast(i)
        gf.forecast(count)
        result = gf.log()
        Result.append(result)
    return Result
        
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

def show_res1dual(data, predict, ID):
    shares_value = []
    for i in ID:
        share_value = data.loc[[i], :]
        share_value = list(list(share_value.values)[0])
        shares_value.append(share_value)
    for i in range(5):
        plt.figure()
        pred = list(deepcopy((predict[i])['数据']))
        pred = pred[:-1]
        x = list(range(len(pred)))
        plt.plot(x, pred)
        plt.plot(x, shares_value[i])
        plt.grid(ls='--')
        plt.title('GM(1,1)')
        plt.xlabel('sample')
        plt.ylabel('value')
        plt.show()

def calc(list1, list2):
    sum_ = 0
    for i in range(len(list1)):
        temp = (list1[i] - list2[i])**2
        sum_ += temp
    final = sqrt(sum_ / len(list1))
    return final
def RMSE(real, predict, n, ID):
    shares_value = []
    err = []
    for i in ID:
        share_value = real.loc[[i], :]
        share_value = list(list(share_value.values)[0])
        shares_value.append(share_value)
    for i in range(len(shares_value)):
        pred = list(deepcopy((predict[i])['数据']))
        pred = pred[-n:]
        real_number = deepcopy(shares_value[i])
        real_number = real_number[:n]
        rmse_cal = calc(pred, real_number)
        err.append(rmse_cal)
    return err
        
def err_mean(err):
    sum_ = 0
    for i in range(len(err)):
        if isnan(err[i]):
            pass
        else:
            sum_ += err[i]
    return sum_ / len(err)


if __name__ == '__main__':
    # GM(1,1)在时间跨度为1的情况下，未来1个预测点的平均预测RMSE得分为：12.799377
    # GM(1,1)在时间跨度为5的情况下，未来1个预测点的平均预测RMSE得分为：10.474189
    # GM(1,1)在时间跨度为22的情况下，未来1个预测点的平均预测RMSE得分为：7.074416
    warnings.filterwarnings("ignore")
    filename = '../data/shares.csv'
    days = 22    # 1:日  5:周  22:月
    n = 3       # 针对接下来几个数据进行预测
    train_data, train_ID = load_train_data(filename)
    test_data, test_ID = load_test_data(filename)
    train_data_series = time_step_count(train_data, days, train_ID)
    test_data_series = time_step_count(test_data, days, test_ID)
    result = predict(train_data_series, train_ID, n)
    #show_res1dual(train_data_series, result, train_ID)
    err = RMSE(train_data_series, result, n, train_ID)
    mean = err_mean(err)
    print('GM(1,1)在时间跨度为%d的情况下，未来%d个预测点的平均预测RMSE得分为：%f' % (days, n, mean))
    
