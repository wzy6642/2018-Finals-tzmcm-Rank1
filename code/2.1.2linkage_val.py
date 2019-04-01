# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 15:27:29 2018
主程序的验证部分
@author: wzy
"""
import pandas as pd
from copy import deepcopy
import numpy as np

def load_data(filename):
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
    return data, ID

def rete_in_de(column, days):
    record = []
    for i in range(len(column)-days):
        if column[i] == 0:
            temp = 0
        else:
            temp = (column[i+days] - column[i]) / column[i]
        record.append(temp)
    return record
def increase(data, days, ID):
    shares_value = []
    shares_incr_dec = []
    for i in ID:
        share_value = data.loc[[i], :]
        share_value = list(list(share_value.values)[0])
        shares_value.append(share_value)
    for share in shares_value:
        share1 = deepcopy(share)
        temp = rete_in_de(share1, days)
        shares_incr_dec.append(temp)
    shares_incr_dec = pd.DataFrame(shares_incr_dec)
    shares_incr_dec.index = ID
    return shares_incr_dec

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

def load_data_final(data):
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

def plate_describe(data):
    mean = []
    for i in data:
        temp = i.mean()
        mean.append(temp)
    return mean

def calc_high_inc_dec(data, threshold=0.02):
    Discrete = []
    for i in data:
        data_copy = list(deepcopy(i))
        discrete = []
        for j in data_copy:
            if j >= threshold:
                temp = 1
            elif j <= -threshold:
                temp = -1
            else:
                temp = 0
            discrete.append(temp)
        Discrete.append(discrete)
    return Discrete

def calc_percentage_increase(data1, data2, date_inc):
    Sum = 0
    for i in range(len(data1)-date_inc):
        if data1[i]==1 and data2[i+date_inc]==1:
            Sum += 1
    percentage = Sum / (len(data1)-date_inc)
    return percentage
def linkage_big_increase(data, date_inc=1):
    record2 = []
    for i in range(len(data)):
        record1 = []
        for j in range(len(data)):
            data1 = deepcopy(data[i])
            data2 = deepcopy(data[j])
            percentage = calc_percentage_increase(data1, data2, date_inc)
            record1.append(percentage)
        record2.append(record1)
    record2 = pd.DataFrame(record2)
    record2.index = list(np.arange(1, len(data)+1))
    record2.columns = list(np.arange(1, len(data)+1))
    return record2

def calc_percentage_decrease(data1, data2, date_inc):
    Sum = 0
    for i in range(len(data1)-date_inc):
        if data1[i]==-1 and data2[i+date_inc]==-1:
            Sum += 1
    percentage = Sum / (len(data1)-date_inc)
    return percentage
def linkage_big_decrease(data, date_inc=1):
    record2 = []
    for i in range(len(data)):
        record1 = []
        for j in range(len(data)):
            data1 = deepcopy(data[i])
            data2 = deepcopy(data[j])
            percentage = calc_percentage_decrease(data1, data2, date_inc)
            record1.append(percentage)
        record2.append(record1)
    record2 = pd.DataFrame(record2)
    record2.index = list(np.arange(1, len(data)+1))
    record2.columns = list(np.arange(1, len(data)+1))
    return record2

def rules2file(m, n, core, data, flag=1):
    values = []
    for i in list(np.arange(1, len(data)+1)):
        value = data.loc[[i], :]
        value = list(list(value.values)[0])
        values.append(value)
    if flag == 1:
        file = '../result/2.1/大涨/验证' + str(m) + '_' + str(n) + '_' + str(core) + '.txt'
        with open(file, 'wt') as f:
            for i in range(len(data)):
                for j in range(len(data)):
                    if i == j:
                        pass
                    else:
                        print('当数据时间跨度为%d，轮动规律间隔日期%d时，板块%d大涨==========>板块%d大涨的概率为：%f' % (m, n, i+1, j+1, values[i][j]), file=f)
    elif flag == 0:
        file = '../result/2.1/大跌/验证' + str(m) + '_' + str(n) + '_' + str(core) + '.txt'
        with open(file, 'wt') as f:
            for i in range(len(data)):
                for j in range(len(data)):
                    if i == j:
                        pass
                    else:
                        print('当数据时间跨度为%d，轮动规律间隔日期%d时，板块%d大跌==========>板块%d大跌的概率为：%f' % (m, n, i+1, j+1, values[i][j]), file=f)
     
    
if __name__ == '__main__':
    # 1:日
    # 5:周
    # 22:月
    num1 = 22       # 计算涨幅/跌幅的时间跨度
    num2 = 1        # 计算轮动规律的间隔日期
    filename = '../data/shares.csv'
    data, ID = load_data(filename)
    shares_incr_dec = increase(data, num1, ID)
    data_1, data_2, data_3, data_4, big_plate = load_data_final(shares_incr_dec)
    mean_1 = plate_describe(data_1)
    mean_2 = plate_describe(data_2)
    mean_3 = plate_describe(data_3)
    mean_4 = plate_describe(data_4)
    big_mean = plate_describe(big_plate)
    Discrete_1 = calc_high_inc_dec(mean_1)
    Discrete_2 = calc_high_inc_dec(mean_2)
    Discrete_3 = calc_high_inc_dec(mean_3)
    Discrete_4 = calc_high_inc_dec(mean_4)
    Discrete_big = calc_high_inc_dec(big_mean)
    linkage_inc_big = linkage_big_increase(Discrete_big, date_inc=num2)
    linkage_inc_1 = linkage_big_increase(Discrete_1, date_inc=num2)
    linkage_inc_2 = linkage_big_increase(Discrete_2, date_inc=num2)
    linkage_inc_3 = linkage_big_increase(Discrete_3, date_inc=num2)
    linkage_inc_4 = linkage_big_increase(Discrete_4, date_inc=num2)
    linkage_dec_big = linkage_big_decrease(Discrete_big, date_inc=num2)
    linkage_dec_1 = linkage_big_decrease(Discrete_1, date_inc=num2)
    linkage_dec_2 = linkage_big_decrease(Discrete_2, date_inc=num2)
    linkage_dec_3 = linkage_big_decrease(Discrete_3, date_inc=num2)
    linkage_dec_4 = linkage_big_decrease(Discrete_4, date_inc=num2)
    # 结果打印 0:大板块  1：第一个小板块  2：第二个小板块  3：第三个小板块  4：第四个小板块
    """
    时间跨度为月时，轮动规律最佳
    """
    core = 0
    rules2file(num1, num2, core, linkage_dec_big, flag=0)
    core = 1
    rules2file(num1, num2, core, linkage_dec_1, flag=0)
    core = 2
    rules2file(num1, num2, core, linkage_dec_2, flag=0)
    core = 3
    rules2file(num1, num2, core, linkage_dec_3, flag=0)
    core = 4
    rules2file(num1, num2, core, linkage_dec_4, flag=0)
    core = 0
    rules2file(num1, num2, core, linkage_inc_big, flag=1)
    core = 1
    rules2file(num1, num2, core, linkage_inc_1, flag=1)
    core = 2
    rules2file(num1, num2, core, linkage_inc_2, flag=1)
    core = 3
    rules2file(num1, num2, core, linkage_inc_3, flag=1)
    core = 4
    rules2file(num1, num2, core, linkage_inc_4, flag=1)
##########################################################
##这是验证集上的程序
##程序的运行方式：
##  num1 可取值有1  5  22     从而修改计算涨幅/跌幅的时间跨度
##  num2 可取值为任何正整数     从而修改计算轮动规律的间隔日期
##  文件命名方式参照文件夹的readme
##########################################################
