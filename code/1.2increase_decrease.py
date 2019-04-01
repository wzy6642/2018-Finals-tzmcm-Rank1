# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 16:20:53 2018

@author: wzy
"""
import pandas as pd
from copy import deepcopy
from math import isnan
import matplotlib.pyplot as plt
import numpy as np
import pywt

def load_data(filename):
    data = pd.read_csv(filename)
    ID = list(deepcopy(data['shares_ID']))
    data.index = ID
    data.drop('shares_name', axis=1, inplace=True)
    data.drop('shares_ID', axis=1, inplace=True)
    data = data.fillna(0)
    return data
def load_data_plate(filename):
    data = pd.read_csv(filename)
    return data

def data_extract(data_plate, data):
    ID = list(deepcopy(data_plate['id']))
    data_sub_plate_all = []
    for i in ID:
        i = i[1:-1]
        i = i.split(', ')
        data_sub_plate = []
        for j in i:
            j = j[1:-1]
            temp = data.loc[j]
            data_sub_plate.append(temp)
        data_sub_plate_all.append(data_sub_plate)
    return data_sub_plate_all

def seris_concat_dataframe(data):
    result = []
    for i in data:
        temp = pd.DataFrame(i)
        result.append(temp)
    return result
 
def load_data_final():
    filename = '../data/shares.csv'
    data = load_data(filename)
    filename = '../result/big_plate.csv'
    big_plate = load_data_plate(filename)
    filename = '../result/num1big_plate_sub.csv'
    num1big_plate_sub = load_data_plate(filename)
    filename = '../result/num2big_plate_sub.csv'
    num2big_plate_sub = load_data_plate(filename)
    filename = '../result/num3big_plate_sub.csv'
    num3big_plate_sub = load_data_plate(filename)
    filename = '../result/num4big_plate_sub.csv'
    num4big_plate_sub = load_data_plate(filename)
    data_sub_plate1 = data_extract(num1big_plate_sub, data)
    data_sub_plate2 = data_extract(num2big_plate_sub, data)
    data_sub_plate3 = data_extract(num3big_plate_sub, data)
    data_sub_plate4 = data_extract(num4big_plate_sub, data)
    data_big_plate = data_extract(big_plate, data)
    data_1 = seris_concat_dataframe(data_sub_plate1)
    data_2 = seris_concat_dataframe(data_sub_plate2)
    data_3 = seris_concat_dataframe(data_sub_plate3)
    data_4 = seris_concat_dataframe(data_sub_plate4)
    big_plate = seris_concat_dataframe(data_big_plate)
    return data_1, data_2, data_3, data_4, big_plate

def rete_in_de(column):
    record = []
    for i in range(len(column)-1):
        temp = (column[i+1] - column[i])
        record.append(temp)
    return record
def diff(data):
    diff_data = []
    for i in data:
        shares_value = []
        shares_incr_dec = []
        for j in range(len(i)):
            share_value = i.iloc[[j]]
            share_value = list(list(share_value.values)[0])
            share_value = [i for i in share_value if not isnan(i)]
            shares_value.append(share_value)
        for share in shares_value:
            share1 = deepcopy(share)
            temp = rete_in_de(share1)
            shares_incr_dec.append(temp)
        diff_data.append(shares_incr_dec)
    return diff_data
def diff_all(data_1, data_2, data_3, data_4, big_plate):
    diff_data1 = diff(data_1)
    diff_data2 = diff(data_2)
    diff_data3 = diff(data_3)
    diff_data4 = diff(data_4)
    diff_big_plate = diff(big_plate)
    return diff_data1, diff_data2, diff_data3, diff_data4, diff_big_plate

def plot_diff(data, m):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    num = 0
    for i in data:
        num += 1
        plt.figure()
        for j in range(0, len(i), 5):
            x = list(np.arange(len(i[j])))
            plt.plot(x, i[j])
            plt.title('同一簇的股票的一阶差分随时间的关系图')
            plt.xlabel('时间')
            plt.ylabel('股票大盘的一阶差分值')
            plt.show()
        filename = '../img/1.2/diff/' + str(m) + '/' + str(m) + '_' + str(num) + '_' + '差分走势图.png'
        plt.savefig(filename)

def coeff(data):
    data_A2 = []
    data_D2 = []
    data_D1 = []
    for i in data:
        shares_value = []
        a2 = []
        d2 = []
        d1 = []
        for j in range(len(i)):
            share_value = i.iloc[[j]]
            share_value = list(list(share_value.values)[0])
            share_value = [i for i in share_value if not isnan(i)]
            shares_value.append(share_value)
        for share in shares_value:
            share1 = deepcopy(share)
            share1 = np.array(share1)
            A2, D2, D1 = pywt.wavedec(share1, 'db4', mode='sym', level=2)
            a2.append(A2)
            d2.append(D2)
            d1.append(D1)
        data_A2.append(a2)
        data_D2.append(d2)
        data_D1.append(d1)
    return data_A2, data_D2, data_D1

def plot_coeff(data, m):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    num = 0
    for i in data:
        num += 1
        plt.figure()
        for j in range(0, len(i), 5):
            x = list(np.arange(len(i[j])))
            plt.plot(x, i[j])
            plt.title('db4小波分解A2走势图')
            plt.xlabel('时间')
            plt.ylabel('db4小波分解A2值')
            plt.show()
        filename = '../img/1.2/coeff/' + str(m) + '/'+'A2/' + str(m) + '_' + str(num) + '_' + 'A2走势图.png'
        plt.savefig(filename)
        
def calcu_cov(data):
    Cov = []
    for i in data:
        ID = i.index
        value = i.values
        value2 = deepcopy(value)
        value2 = np.matrix(value2)
        value2 = value2.T
        value2 = pd.DataFrame(value2)
        value2.columns = ID
        cov = value2.corr()
        temp = cov.sum()
        temp = sum(temp)
        temp = temp / (len(i) ** 2)
        Cov.append(temp)
    return Cov

def A2_max_location(data):
    Location = []
    for i in data:
        sub_location = []
        for j in i:
            temp = list(deepcopy(j))
            location = temp.index(max(temp)) 
            sub_location.append(location)
        Location.append(sub_location)
    return Location


if __name__ == '__main__':
    data_1, data_2, data_3, data_4, big_plate = load_data_final()
    diff_data1, diff_data2, diff_data3, diff_data4, diff_big_plate = diff_all(data_1, data_2, data_3, data_4, big_plate)
    #生成一阶差分信号图并进行保存
    #plot_diff(diff_data4, 4)
    # 这里的参数为data_1, data_2, data_3, data_4
    data_A2, data_D2, data_D1 = coeff(data_4)
    #plot_coeff(data_A2, 4)
    # 这里的参数为data_1, data_2, data_3, data_4
    Cov1 = calcu_cov(data_1)
    # 指标2：DB4小波分解的峰值时间点的一致性分析A2的max的出现位置
    # 这里的参数为data_A2
    location = A2_max_location(data_A2)

