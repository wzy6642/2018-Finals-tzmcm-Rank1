# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 23:48:49 2018
周6，7股票不交易
使用2017-12-26到2018-11-30之间的数据进行分析
参考 https://www.jianshu.com/p/ec3d893d296d
@author: wzy
"""
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

def load_data(filename):
    data = pd.read_csv(filename)
    ID = list(deepcopy(data['shares_ID']))
    time = list(data.columns)
    begin = time.index('2017-12-26')
    time = time[begin:]
    data = data[time]
    data.insert(0, 'shares_ID', ID)
    data = data.fillna(0)
    return data, ID

def rete_in_de(column, n):
    record = []
    for i in range(len(column)-n):
        temp = (column[i+n] - column[i])
        record.append(temp)
    return record
def diff(data, n):
    shares_value = []
    shares_incr_dec = []
    data.drop('shares_ID', axis=1, inplace=True)
    for i in range(len(data)):
        share_value = data.iloc[[i]]
        share_value = list(list(share_value.values)[0])
        shares_value.append(share_value)
    for share in shares_value:
        share1 = deepcopy(share)
        temp = rete_in_de(share1, n)
        shares_incr_dec.append(temp)
    shares_incr_dec = pd.DataFrame(shares_incr_dec)
    return shares_incr_dec  

def data_corr(data, ID):
    data_cpoy = deepcopy(data)
    value = data_cpoy.values
    value = np.matrix(value)
    value = value.T
    value = pd.DataFrame(value)
    value.columns = ID
    corr = value.corr(method='pearson')
    return corr
    
def calc_location_pos(List, threshold):
    num = 0
    mamory = []
    for i in List:
        if i > threshold:
            mamory.append(num)
        num += 1
    return mamory
def positive_related(corr, ID):
    pos_corr = deepcopy(corr)
    pos_location = []
    for i in ID:
        sub_corr = pos_corr[i]
        sub_corr_1 = list(deepcopy(sub_corr))
        temp = calc_location_pos(sub_corr_1, 0)
        pos_location.append(temp)
    related_name = []
    for i in pos_location:
        location_name = []
        for j in i:
            location_name.append(ID[j])
        related_name.append(location_name)
    return related_name
        
def calc_location_neg(List, threshold):
    num = 0
    mamory = []
    for i in List:
        if i < threshold:
            mamory.append(num)
        num += 1
    return mamory
def negative_related(corr, ID):
    neg_corr = deepcopy(corr)
    neg_location = []
    for i in ID:
        sub_corr = neg_corr[i]
        sub_corr_1 = list(deepcopy(sub_corr))
        temp = calc_location_neg(sub_corr_1, 0)
        neg_location.append(temp)
    related_name = []
    for i in neg_location:
        location_name = []
        for j in i:
            location_name.append(ID[j])
        related_name.append(location_name)
    return related_name

def calc_location_no(List, threshold):
    num = 0
    mamory = []
    for i in List:
        if i == threshold:
            mamory.append(num)
        num += 1
    return mamory
def no_related(corr, ID):
    no_corr = deepcopy(corr)
    no_location = []
    for i in ID:
        sub_corr = no_corr[i]
        sub_corr_1 = list(deepcopy(sub_corr))
        temp = calc_location_no(sub_corr_1, 0)
        no_location.append(temp)
    related_name = []
    for i in no_location:
        location_name = []
        for j in i:
            location_name.append(ID[j])
        related_name.append(location_name)
    return related_name

def calc_location_pos_high(List, threshold):
    num = 0
    mamory = []
    for i in List:
        if i > threshold:
            mamory.append(num)
        num += 1
    return mamory
def high_positive_related(corr, ID):
    high_pos_corr = deepcopy(corr)
    high_pos_location = []
    for i in ID:
        sub_corr = high_pos_corr[i]
        sub_corr_1 = list(deepcopy(sub_corr))
        temp = calc_location_pos_high(sub_corr_1, 0.8)
        high_pos_location.append(temp)
    related_name = []
    for i in high_pos_location:
        location_name = []
        for j in i:
            location_name.append(ID[j])
        related_name.append(location_name)
    return related_name

def calc_location_pos_low(List, threshold):
    num = 0
    mamory = []
    for i in List:
        if 0<i < threshold:
            mamory.append(num)
        num += 1
    return mamory
def low_positive_related(corr, ID):
    low_pos_corr = deepcopy(corr)
    low_pos_location = []
    for i in ID:
        sub_corr = low_pos_corr[i]
        sub_corr_1 = list(deepcopy(sub_corr))
        temp = calc_location_pos_low(sub_corr_1, 0.3)
        low_pos_location.append(temp)
    related_name = []
    for i in low_pos_location:
        location_name = []
        for j in i:
            location_name.append(ID[j])
        related_name.append(location_name)
    return related_name

def calc_location_neg_high(List, threshold):
    num = 0
    mamory = []
    for i in List:
        if -1<i < threshold:
            mamory.append(num)
        num += 1
    return mamory
def high_negative_related(corr, ID):
    high_neg_corr = deepcopy(corr)
    high_neg_location = []
    for i in ID:
        sub_corr = high_neg_corr[i]
        sub_corr_1 = list(deepcopy(sub_corr))
        temp = calc_location_neg_high(sub_corr_1, -0.8)
        high_neg_location.append(temp)
    related_name = []
    for i in high_neg_location:
        location_name = []
        for j in i:
            location_name.append(ID[j])
        related_name.append(location_name)
    return related_name

def calc_location_neg_low(List, threshold):
    num = 0
    mamory = []
    for i in List:
        if threshold<i < 0:
            mamory.append(num)
        num += 1
    return mamory
def low_negative_related(corr, ID):
    low_neg_corr = deepcopy(corr)
    low_neg_location = []
    for i in ID:
        sub_corr = low_neg_corr[i]
        sub_corr_1 = list(deepcopy(sub_corr))
        temp = calc_location_pos_low(sub_corr_1, -0.3)
        low_neg_location.append(temp)
    related_name = []
    for i in low_neg_location:
        location_name = []
        for j in i:
            location_name.append(ID[j])
        related_name.append(location_name)
    return related_name

def plot_diff(data, List, ID):
    shares_value = []
    for i in range(len(data)):
        share_value = data.iloc[[i]]
        share_value = list(list(share_value.values)[0])
        shares_value.append(share_value)
    # 截取部分
    List1 = List[6:10]
    #List2 = List[7:8]
    #List3 = List[11:13]
    List = List1 #+ List2 + List3
    for i in List:
        location = ID.index(i)
        y = data.iloc[[location]]
        y = list(list(y.values)[0])
        x = list(np.arange(len(y)))
        plt.plot(x, y, label=i)
        plt.title('-1<corr<-0.8 & day diff = 10')
        plt.xlabel('time diff')
        plt.ylabel('value diff')
        plt.grid(ls='--')
        plt.legend(loc = 'upper right')
        plt.show()
        

if __name__ == '__main__':
    filename = '../data/shares.csv'
    data, ID = load_data(filename)
    diff_data = diff(data, 10)
    diff_data.index = ID
    corr = data_corr(data, ID)
    pos_related_ID = positive_related(corr, ID)
    neg_related_ID = negative_related(corr, ID)
    no_related_ID = no_related(corr, ID)
    high_pos_related_ID = high_positive_related(corr, ID)
    low_pos_related_ID = low_positive_related(corr, ID)
    high_neg_related_ID = high_negative_related(corr, ID)
    low_neg_related_ID = low_negative_related(corr, ID)
    """
    很少有股票表现负相关,大多带有一定的正相关性
    低正相关股票会表现出分离与汇合，高正相关股票会描述板块内部走势的一致性
    corr越大，说明其走势越一致
    不同时间跨度的分析会伴随着股票板块的运动的变化而变化
    基准是5天、10天、30天
    在强相关内部，时间跨度影响对板块的分离与汇合影响较小
    而在弱相关内部，时间跨度不同则板块的分离与汇合的情况则会发生一定的改变
    """
    # 第二个参数换为data以观察大盘的一致性走向，注意更换plot里面title等标注
    plot_diff(diff_data, low_pos_related_ID[44], ID)


