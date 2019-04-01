# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 15:36:55 2018
参考： https://stocktobe.com/book/paper.aspx?id=7&token=
@author: wzy
"""
import pandas as pd
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

def load_data(filename):
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

def show_inc_dec(file):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(len(file)):
        plt.figure()
        y = deepcopy(file[i])
        y = y[100:500]
        x = list(np.arange(len(y)))
        points = [(0, 0), (len(y), 0)]
        (xpoints, ypoints) = zip(*points)
        plt.plot(xpoints, ypoints)
        for i in range(len(y)):
            if y[i] == 1:
                plt.bar(x[i], y[i], fc='r', alpha=0.5)
            elif y[i] == 0:
                plt.bar(x[i], y[i], fc='c', alpha=0.5)
            else:
                plt.bar(x[i], y[i], fc='g', alpha=0.5)
        plt.title('股票大涨大跌情况的可视化')
        plt.xlabel('时间')
        plt.ylabel('涨跌情况')
        plt.grid(ls='--')
        plt.show()

def calc(list1, list2):
    temp = 0
    for i in range(len(list1)):
        temp += (list1[i] - list2[i])**2
    return math.sqrt(temp/len(list1))
def distance(data):
    similary = []
    for i in range(len(data)):
        loc = []
        for j in range(len(data)):
            temp = calc(data[i], data[j])
            loc.append(temp)
        similary.append(loc)
    similary = pd.DataFrame(similary)
    similary.index = list(np.arange(1, len(data)+1))
    similary.columns = list(np.arange(1, len(data)+1))
    return similary

def distance_show(data):
    sns.heatmap(data, linewidths = 0.05, cmap='rainbow', annot=True)
    plt.title('四大板块间距离相关性')
    plt.xlabel('板块名称')
    plt.ylabel('板块名称')

def find_similar(data, n=3):
    shares_value = []
    small_value = []
    similar = []
    suoyin = list(np.arange(1, len(data)+1))
    for i in suoyin:
        share_value = data.loc[[i], :]
        share_value = list(list(share_value.values)[0])
        shares_value.append(share_value)
    for i in shares_value:
        temp = deepcopy(i)
        temp.sort()
        temp = temp[:n]
        small_value.append(temp)
    for i in range(len(shares_value)):
        list1 = deepcopy(shares_value[i])
        list2 = deepcopy(small_value[i])
        location = []
        for j in list2:
            location.append(list1.index(j)+1)
        similar.append(location)
    return similar


def data_pack(index, data, n=3):
    data_use = deepcopy(data)
    index = np.array(index)
    index = index - 1
    index = list(index)
    index.sort()
    data1 = data_use[index[0]]
    data2 = data_use[index[1]]
    data3 = data_use[index[2]]
    temp = pd.concat([data1, data2, data3])
    data_use.append(temp)
    del(data_use[index[0]])
    a = index[1]-1
    del(data_use[a])
    a = index[2]-2
    del(data_use[a])
    return data_use
    
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

def rules2file(N, m, n, core, zuhe, data, flag=1):
    values = []
    for i in list(np.arange(1, len(data)+1)):
        value = data.loc[[i], :]
        value = list(list(value.values)[0])
        values.append(value)
    if flag == 1:
        file = '../result/2.3/大涨/' + str(m) + '_' + str(n) + '_' + str(core) + '_' + str(zuhe[0]) + str(zuhe[1]) + str(zuhe[2]) + '_' + str(N) + '.txt'
        with open(file, 'wt') as f:
            print('时间跨%d组，组合板块为：%d, %d, %d' % (N, zuhe[0], zuhe[1], zuhe[2]), file=f)
            for i in range(len(data)):
                for j in range(len(data)):
                    if i == j:
                        pass
                    else:
                        print('当数据时间跨度为%d，轮动规律间隔日期%d时，板块%d大涨==========>板块%d大涨的概率为：%f' % (m, n, i+1, j+1, values[i][j]), file=f)
    elif flag == 0:
        file = '../result/2.3/大跌/' + str(m) + '_' + str(n) + '_' + str(core) + '_' + str(zuhe[0]) + str(zuhe[1]) + str(zuhe[2]) + '_' + str(N) + '.txt'
        with open(file, 'wt') as f:
            print('时间跨%d组，组合板块为：%d, %d, %d' % (N, zuhe[0], zuhe[1], zuhe[2]), file=f)
            for i in range(len(data)):
                for j in range(len(data)):
                    if i == j:
                        pass
                    else:
                        print('当数据时间跨度为%d，轮动规律间隔日期%d时，板块%d大跌==========>板块%d大跌的概率为：%f' % (m, n, i+1, j+1, values[i][j]), file=f)
                 
                        
if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 1:日
    # 5:周
    # 22:月
    # N:跨度 N=1为2.2结果  N为其他正整数为2.3结果
    N = 2
    num1 = 1*N       # 计算涨幅/跌幅的时间跨度
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
    # 板块之间的涨跌一致性分析图的可视化
    # show_inc_dec(Discrete_big)
    sim_1 = distance(Discrete_1)
    sim_2 = distance(Discrete_2)
    sim_3 = distance(Discrete_3)
    sim_4 = distance(Discrete_4)
    sim_big = distance(Discrete_big)
    # 绘制相关性矩阵的heatmap
    # distance_show(sim_big)
    """
    此处全部取3可能有一定的局限性，最好进行一下后处理！
    """
    sim_samp1 = find_similar(sim_1, 3)
    sim_samp2 = find_similar(sim_2, 3)
    sim_samp3 = find_similar(sim_3, 3)
    sim_samp4 = find_similar(sim_4, 3)
    """
    以下为测试部分
    """
    # 填入0~9
    # sim_samp1  sim_samp2  sim_samp3  sim_samp4
    i = sim_samp1[9]
    data_use = data_pack(i, data_1)
    mean = plate_describe(data_use)
    Discrete = calc_high_inc_dec(mean)
    linkage_inc = linkage_big_increase(Discrete, date_inc=num2)
    linkage_dec = linkage_big_decrease(Discrete, date_inc=num2)
    # 1：第一个小板块  2：第二个小板块  3：第三个小板块  4：第四个小板块
    core = 1
    rules2file(N, num1, num2, core, i, linkage_dec, flag=1)
###############################################################
## 程序备注：
## N：时间跨度（组）
## num1： 可选值有1  5  22  分别代表计算涨幅/跌幅的时间跨度为日  周  月
## num2： 可选值有1         计算轮动规律的间隔日期
## 493行中 sim_samp1可以替换为sim_samp2 sim_samp3 sim_samp4 分别代表对4大板块内部的小板块进行分析
## 493行中 sim_samp1的索引值可以填入0~9的数字，分别代表对哪一个联动板块分析
## 500行中 core可以取值为1~4 1：第一个小板块  2：第二个小板块  3：第三个小板块  4：第四个小板块
## 501行中 flag和linkage_dec一起改变
##    linkage_dec ======>flag=0 表示计算大跌的aporiori
##    linkage_inc ======>flag=1 表示计算大涨的aporiori
## 484~487行中的3代表KNN的K为3，可以做适当的修改，亦可以做后处理
## core和sim_samp2的取值注意一致性
###############################################################

