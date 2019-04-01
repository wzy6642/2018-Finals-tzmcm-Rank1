# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 21:16:02 2018

@author: wzy
"""
import pandas as pd
import numpy as np
import warnings
from math import isnan
from copy import deepcopy
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from skfuzzy.cluster import cmeans
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import mpl
from scipy.stats import norm
from scipy import stats
from sklearn.manifold import TSNE

def LoadData(filename):
    data = pd.read_excel(filename)
    shares_time = data.index
    data.index = list(np.arange(len(data)))
    data.insert(0, 'time', shares_time)
    data = data.drop([1])
    shares_name = data.iloc[[0]]
    shares_name.drop('time', axis=1, inplace=True)
    shares_ID = list(shares_name.columns)
    shares_name = list(list(shares_name.values)[0])
    shares = pd.DataFrame({'shares_ID': shares_ID, 'shares_name': shares_name})
    data = data.drop([0])
    Time = list(data['time'])
    time_list = []
    for i in Time:
        i = str(i)
        time_list.append(i[:10])
    data.drop('time', axis=1, inplace=True)
    data_value = np.matrix(data.values)
    data_value = data_value.T
    data_value = pd.DataFrame(data_value)
    data_value.columns = time_list
    data2 = pd.concat([shares,data_value], axis=1, ignore_index=True)
    data2.columns = list(shares.columns) + time_list
    data.index = list(np.arange(len(data)))
    return data, data2, shares_ID

def rete_in_de(column):
    record = []
    for i in range(len(column)-1):
        temp = (column[i+1] - column[i]) / column[i]
        record.append(temp)
    return record
def increase_decrease(data, shares_ID):
    shares_value = []
    shares_incr_dec = []
    data_incr = []
    data_dec = []
    data_incr_avg = []
    data_dec_avg = []
    increase_days = []
    decrease_days = []
    for i in range(len(data)):
        share_value = data.iloc[[i]]
        share_value = list(list(share_value.values)[0])[2:]
        share_value = [i for i in share_value if not isnan(i)]
        shares_value.append(share_value)
    for share in shares_value:
        share1 = deepcopy(share)
        temp = rete_in_de(share1)
        shares_incr_dec.append(temp)
    for data in shares_incr_dec:
        increase = 0
        decrease = 0
        in_num = 0
        de_num = 0
        for i in data:
            if isnan(i):
                pass
            else:
                if i >= 0:
                    increase += i
                    in_num += 1
                else:
                    decrease += i
                    de_num += 1
        data_incr.append(increase)
        data_incr_avg.append(increase/in_num)
        data_dec.append(decrease)
        data_dec_avg.append(decrease/de_num)
        increase_days.append(in_num)
        decrease_days.append(de_num)
    feature = pd.DataFrame({'shares_ID': shares_ID, 'increase': data_incr, 
                            'decrease': data_dec, 'increase_avg': data_incr_avg, 
                            'decrease_avg': data_dec_avg, 'increase_days': increase_days, 
                            'decrease_days': decrease_days})
    feature['deal_days'] = feature['increase_days'] + feature['decrease_days']
    return feature

def average(shares, feature):
    shares_value = []
    mean = []
    Max = []
    Min = []
    median = []
    quartile_1 = []
    quartile_2 = []
    for i in range(len(shares)):
        share_value = shares.iloc[[i]]
        share_value = list(list(share_value.values)[0])[2:]
        share_value = [i for i in share_value if not isnan(i)]
        shares_value.append(share_value)
    for i in shares_value:
        temp = sum(i) / len(i)
        mean.append(temp)
    for i in shares_value:
        temp = max(i)
        Max.append(temp)
    for i in shares_value:
        temp = min(i)
        Min.append(temp)
    for i in shares_value:
        temp = deepcopy(i)
        temp.sort()
        median.append(temp[int(len(i)/2)])
        quartile_1.append(temp[int(len(i)*0.25)])
        quartile_2.append(temp[int(len(i)*0.75)])
    feature['mean'] = mean
    feature['max'] = Max
    feature['min'] = Min
    feature['range'] = feature['max'] - feature['min']
    feature['quartile_1'] = quartile_1
    feature['median'] = median
    feature['quartile_2'] = quartile_2
    return feature

def var(shares, feature):
    shares_value = []
    var = []
    std = []
    for i in range(len(shares)):
        share_value = shares.iloc[[i]]
        share_value = list(list(share_value.values)[0])[2:]
        share_value = [i for i in share_value if not isnan(i)]
        shares_value.append(share_value)
    for i in shares_value:
        i = np.array(i)
        arr_var = np.var(i)
        var.append(arr_var)
        arr_std = np.std(i, ddof=1)
        std.append(arr_std)
    feature['var'] = var
    feature['std'] = std
    return feature

def difference(shares):
    shares_value = []
    difference_matrix = []
    for i in range(len(shares)):
        share_value = shares.iloc[[i]]
        share_value = list(list(share_value.values)[0])[2:]
        share_value = [i for i in share_value if not isnan(i)]
        shares_value.append(share_value)
    for i in shares_value:
        temp = []
        for j in range(len(i)-1):
            num = i[j+1] - i[j]
            temp.append(num)
        difference_matrix.append(temp)
    return difference_matrix, shares_value
    
def diff_features(difference_matrix, feature):
    count = []
    diff_max = []
    diff_min = []
    diff_avg = []
    median = []
    quartile_1 = []
    quartile_2 = []
    for i in difference_matrix:
        num_0 = 0
        for j in i:
            if j == 0:
                num_0 += 1
        count.append(num_0)
    feature['diff_count_0'] = count
    for i in difference_matrix:
        temp = max(i)
        diff_max.append(temp)
    feature['diff_max'] = diff_max
    for i in difference_matrix:
        temp = min(i)
        diff_min.append(temp)
    feature['diff_min'] = diff_min
    feature['diff_range'] = feature['diff_max'] - feature['diff_min']
    for i in difference_matrix:
        temp = sum(i) / len(i)
        diff_avg.append(temp)
    feature['diff_avg'] = diff_avg
    for i in difference_matrix:
        temp = deepcopy(i)
        temp.sort()
        median.append(temp[int(len(i)/2)])
        quartile_1.append(temp[int(len(i)*0.25)])
        quartile_2.append(temp[int(len(i)*0.75)])
    feature['diff_quartile_1'] = quartile_1
    feature['diff_median'] = median
    feature['diff_quartile_2'] = quartile_2
    var = []
    std = []
    for i in difference_matrix:
        i = np.array(i)
        arr_var = np.var(i)
        var.append(arr_var)
        arr_std = np.std(i, ddof=1)
        std.append(arr_std)
    feature['diff_var'] = var
    feature['diff_std'] = std
    return feature

def discrete_coefficient(feature):
    feature['discrete_coefficient'] = feature['std'] / feature['mean']
    return feature

def increase(shares, n):
    shares_value = []
    increase_n_days = []
    for i in range(len(shares)):
        share_value = shares.iloc[[i]]
        share_value = list(list(share_value.values)[0])[2:]
        share_value = [i for i in share_value if not isnan(i)]
        shares_value.append(share_value)
    for i in shares_value:
        record = []
        for j in range(len(i)-n):
            temp = i[j+n] - i[j]
            record.append(temp)
        increase_n_days.append(record)
    return increase_n_days

def increase_days_describe(increase_n_days, feature, n):
    Range = []
    Avg = []
    Min = []
    Max = []
    for i in increase_n_days:
        temp = max(i) - min(i)
        Range.append(temp)
        Min.append(min(i))
        Max.append(max(i))
    Str = 'increase' + str(n) + 'range'
    feature[Str] = Range
    for i in increase_n_days:
        temp = sum(i) / len(i)
        Avg.append(temp)
    var = []
    std = []
    for i in increase_n_days:
        i = np.array(i)
        arr_var = np.var(i)
        var.append(arr_var)
        arr_std = np.std(i, ddof=1)
        std.append(arr_std)
    Str = 'increase' + str(n) + 'avg'
    feature[Str] = Avg
    Str = 'increase' + str(n) + 'min'
    feature[Str] = Min
    Str = 'increase' + str(n) + 'max'
    feature[Str] = Max
    Str = 'increase' + str(n) + 'var'
    feature[Str] = var
    Str = 'increase' + str(n) + 'std'
    feature[Str] = std
    return feature

def ADR(difference_matrix, N):
    Pos_Neg = []
    for i in difference_matrix:
        record = []
        for j in range(len(i)-N):
            positive = 0
            negative = 0
            num = 0
            while num <= N:
                if i[j+num] >= 0:
                    if num <= 10:
                        positive += 1
                    else:
                        positive = positive
                elif i[j+num] < 0:
                    negative += 1
                num += 1
            temp = (positive+1) / (negative+1)
            record.append(temp)
        Pos_Neg.append(record)
    return Pos_Neg

def Pos_Neg_describe(feature, Pos_Neg):
    Pos_Neg_range = []
    Mean = []
    Min = []
    Max = []
    for i in Pos_Neg:
        Range = max(i) - min(i)
        Pos_Neg_range.append(Range)
        Mean.append(sum(i) / len(i))
        Min.append(min(i))
        Max.append(max(i))
    feature['ADR_range'] = Pos_Neg_range
    feature['ADR_mean'] = Mean
    feature['ADR_min'] = Min
    feature['ADR_max'] = Max
    return feature

def RSI(difference_matrix, N):
    rsi = []
    for i in difference_matrix:
        temp = []
        for j in range(len(i)-N):
            num = 0
            positive = 0
            negative = 0
            while num <= N:
                if i[j+num]>=0:
                    positive += i[j+num]
                elif i[j+num]<0:
                    negative += i[j+num]
                num += 1
            RS = positive+0.001 / (-negative+0.001)
            RSI = 100 * RS / (1 + RS)
            temp.append(RSI)
        rsi.append(temp)
    return rsi

def RSI_describe(feature, rsi):
    RSI_range = []
    Mean = []
    for i in rsi:
        Range = max(i) - min(i)
        RSI_range.append(Range)
        Mean.append(sum(i) / len(i))
    feature['RSI_range'] = RSI_range
    feature['RSI_mean'] = Mean
    return feature

def BIAS(shares, N):
    BIAS__ = []
    shares_value = []
    for i in range(len(shares)):
        share_value = shares.iloc[[i]]
        share_value = list(list(share_value.values)[0])[2:]
        share_value = [i for i in share_value if not isnan(i)]
        shares_value.append(share_value)
    for i in shares_value:
        bias = []
        for j in range(len(i)-N):
            temp = i[j:j+N]
            avg = sum(temp) / len(temp)
            Bias_num = (i[j]-avg)/avg
            bias.append(Bias_num)
        BIAS__.append(bias)
    return BIAS__
        
def BIAS_describe(feature, bias, N):
    BIAS_range = []
    Mean = []
    for i in bias:
        Range = max(i) - min(i)
        BIAS_range.append(Range)
        Mean.append(sum(i) / len(i))
    str1 = 'BIAS' + str(N) + 'range'
    feature[str1] = BIAS_range
    str2 = 'BIAS' + str(N) + 'mean'
    feature[str2] = Mean
    return feature

def RSV(shares, N):
    RSV__ = []
    shares_value = []
    for i in range(len(shares)):
        share_value = shares.iloc[[i]]
        share_value = list(list(share_value.values)[0])[2:]
        share_value = [i for i in share_value if not isnan(i)]
        shares_value.append(share_value)
    for i in shares_value:
        Rsv = []
        for j in range(len(i)-N):
            temp = i[j:j+N]
            max_num = max(temp)
            min_num = min(temp)
            item = (i[j+N]-min_num+0.00001)/(max_num-min_num+0.00001)
            Rsv.append(item)
        RSV__.append(Rsv)
    return RSV__
    
def RSV_describe(feature, rsv, N):
    RSV_range = []
    Mean = []
    Min = []
    Max = []
    RSV_1_4 = []
    RSV_1_2 = []
    RSV_3_4 = []
    for i in rsv:
        Range = max(i) - min(i)
        RSV_range.append(Range)
        Mean.append(sum(i) / len(i))
        Min.append(min(i))
        Max.append(max(i))
    for i in rsv:
        temp = deepcopy(i)
        temp.sort()
        RSV_1_4.append(temp[int(len(temp)/4)])
        RSV_1_2.append(temp[int(len(temp)/2)])
        RSV_3_4.append(temp[int(len(temp)*0.75)])
    var = []
    std = []
    for i in rsv:
        i = np.array(i)
        arr_var = np.var(i)
        var.append(arr_var)
        arr_std = np.std(i, ddof=1)
        std.append(arr_std)
    str1 = 'RSV' + str(N) + 'range'
    feature[str1] = RSV_range
    str2 = 'RSV' + str(N) + 'mean'
    feature[str2] = Mean
    str3 = 'RSV' + str(N) + 'min'
    feature[str3] = Min
    str4 = 'RSV' + str(N) + 'max'
    feature[str4] = Max
    str5 = 'RSV' + str(N) + '_1_4'
    feature[str5] = RSV_1_4
    str6 = 'RSV' + str(N) + '_1_2'
    feature[str6] = RSV_1_2
    str7 = 'RSV' + str(N) + '_3_4'
    feature[str7] = RSV_3_4
    str8 = 'RSV' + str(N) + 'var'
    feature[str8] = var
    str9 = 'RSV' + str(N) + 'std'
    feature[str9] = std
    return feature

def ROC(shares, N):
    ROC__ = []
    shares_value = []
    for i in range(len(shares)):
        share_value = shares.iloc[[i]]
        share_value = list(list(share_value.values)[0])[2:]
        share_value = [i for i in share_value if not isnan(i)]
        shares_value.append(share_value)
    for i in shares_value:
        roc = []
        for j in range(N,len(i)):
            BX = i[j-N]
            AX = i[j] - BX
            ROC = AX / BX
            roc.append(ROC)
        ROC__.append(roc)
    return ROC__
   
def ROC_describe(feature, roc, N):
    ROC_range = []
    Mean = []
    for i in roc:
        Range = max(i) - min(i)
        ROC_range.append(Range)
        Mean.append(sum(i) / len(i))
    str1 = 'ROC' + str(N) + 'range'
    feature[str1] = ROC_range
    str2 = 'ROC' + str(N) + 'mean'
    feature[str2] = Mean
    return feature

def normalization(data, MIN=0.002, MAX=0.998):
    #print(data.size)
    data.drop('shares_ID', axis=1, inplace=True)
    min_max_scaler = MinMaxScaler(feature_range=(MIN, MAX))
    data_norm = min_max_scaler.fit_transform(data) 
    data_norm = pd.DataFrame(data_norm)
    data_norm.columns = data.columns
    return data_norm

def entropy(data_norm, data_id, threshold=0.8):
    feature_weight = pd.DataFrame({'temp': list(np.zeros(len(data_norm)))})
    for i in data_norm.columns:                     # 计算特征比重
        Sum = data_norm[i].sum()
        temp = data_norm[i]/Sum
        feature_weight[i] = temp
    feature_weight.drop('temp', axis=1, inplace=True)
    Entropy = {}
    for i in feature_weight.columns:                # 计算每一项指标的熵值
        Sum = 0
        column = list(deepcopy(feature_weight[i]))
        for j in range(len(feature_weight)):
            Sum += column[j] * math.log(column[j])
        Entropy[i] = (-1 / (math.log(len(feature_weight)))) * Sum
    #f = open('../result/Entropy.txt', 'w')
    #f.write(str(Entropy))
    #f.close()
    important_features = []
    for key, value in Entropy.items():
        if value <= threshold:                      # 提取重要特征进行分析,控制此处的阈值
            important_features.append(key)
    difference_coefficient = {}
    for i in important_features:                    # 计算差异系数
        difference_coefficient[i] = 1 - Entropy[i]
    #f = open('../result/difference_coefficient.txt', 'w')
    #f.write(str(difference_coefficient))
    #f.close()
    Diff_sum = sum(list(difference_coefficient.values()))
    entropy_weight = {}
    for i in important_features:                    # 计算熵权
        entropy_weight[i] = difference_coefficient[i] / Diff_sum
    f = open('../result/entropy_weight.txt', 'w')
    f.write(str(entropy_weight))
    f.close()
    feature_weight = feature_weight[important_features]
    feature_weight = np.mat(feature_weight)
    weight = np.array(list(entropy_weight.values()))
    overall_merit = weight * (feature_weight.T)     # 计算各个评价对象的综合评价值
    overall_merit = overall_merit.T
    overall_merit = np.array(overall_merit)
    overall_list = []
    for i in range(len(feature_weight)):
        overall_list.append(overall_merit[i][0])
    overall = pd.DataFrame({'eventid': data_id, 'overall': overall_list})
    overall = overall.sort_values(by=['overall'], ascending=(False))
    overall.index = list(np.arange(len(data_norm)))
    data_norm = data_norm[important_features]
    overall = overall.sort_values(by=['eventid'], ascending=(True))
    overall.index = list(np.arange(len(data_norm)))
    # data = pd.concat([data, overall], axis=1)
    # data_id = data['eventid']
    # data.drop(labels=['eventid'], axis=1, inplace = True)
    return Entropy, difference_coefficient, important_features, entropy_weight, overall
    
def concat_popular(feature):
    filename = '../result/allData.csv'
    feature_popular = pd.read_csv(filename, header = None)
    feature_popular.columns = ['pop_0', 'pop_1', 'pop_2', 'pop_3', 'pop_4', 
                               'pop_5', 'pop_6', 'pop_7', 'pop_8', 'pop_9', 
                               'pop_10']
    feature_popular.drop('pop_0', axis=1, inplace=True)
    feature = pd.concat([feature, feature_popular], axis=1)
    return feature
    
def feature_engineering(percent):
    warnings.filterwarnings("ignore")
    filename = '../data/附件.xlsx'
    data, shares, shares_ID = LoadData(filename)
    shares.to_csv('../data/shares.csv', index=False)
    feature = increase_decrease(shares, shares_ID)
    feature = average(shares, feature)
    feature = var(shares, feature)
    difference_matrix, shares_value = difference(shares)
    Pos_Neg = ADR(difference_matrix, 10)
    feature = diff_features(difference_matrix, feature)
    increase_n_days = increase(shares, 30)
    feature = increase_days_describe(increase_n_days, feature, 30)
    increase_n_days = increase(shares, 10)
    feature = increase_days_describe(increase_n_days, feature, 10)
    increase_n_days = increase(shares, 5)
    feature = increase_days_describe(increase_n_days, feature, 5)
    increase_n_days = increase(shares, 2)
    feature = increase_days_describe(increase_n_days, feature, 2)
    feature = Pos_Neg_describe(feature, Pos_Neg)
    rsi = RSI(difference_matrix, 10)
    feature = RSI_describe(feature, rsi)
    bias = BIAS(shares, 6)
    feature = BIAS_describe(feature, bias, 6)
    bias = BIAS(shares, 10)
    feature = BIAS_describe(feature, bias, 10)
    rsv = RSV(shares, 9)
    feature = RSV_describe(feature, rsv, 9)
    rsv = RSV(shares, 30)
    feature = RSV_describe(feature, rsv, 30)
    rsv = RSV(shares, 90)
    feature = RSV_describe(feature, rsv, 90)
    roc = ROC(shares, 12)
    feature = ROC_describe(feature, roc, 12)
    feature = concat_popular(feature)
    data_norm = normalization(feature, MIN=0.002, MAX=0.998)
    Entropy, difference_coefficient, important_features, entropy_weight, overall = entropy(data_norm, shares_ID, threshold=percent)
    original = deepcopy(feature)
    useful_feature = feature[important_features]
    useful_feature.insert(0, 'eventid', list(overall['eventid']))
    useful_feature['overall'] = list(overall['overall'])
    useful_feature.to_csv('../result/feature.csv', index=False)
    return useful_feature, entropy_weight, important_features, original

def KMeans_cluser(useful_feature, data_id, data_score, important_features, n):
    model = KMeans(n_clusters=n, n_jobs=4, max_iter=10000)
    model.fit(useful_feature)
    score_sil = metrics.silhouette_score(useful_feature, model.labels_, metric='euclidean')
    print("当聚为%d簇时，KMeans轮廓系数Silhouette Coefficient为：%f" % (n, score_sil))             # 计算轮廓系数
    score_cal = metrics.calinski_harabaz_score(useful_feature, model.labels_) 
    print("当聚为%d簇时，KMeans轮廓系数Calinski-Harabaz Index为：%f" % (n, score_cal))
    label_counts = pd.Series(model.labels_).value_counts()
    centers = pd.DataFrame(model.cluster_centers_)
    report = pd.concat([centers, label_counts], axis=1)
    report.columns = list(useful_feature.columns) + ['label_counts']
    kmeans_result = pd.concat([useful_feature, pd.Series(model.labels_, index=useful_feature.index)], axis=1)
    kmeans_result.columns = list(useful_feature.columns) + ['label']
    kmeans_result.insert(0, 'eventid', data_id)
    kmeans_result['overall'] = data_score
    center_overall_sum = {}
    for i in range(n):
        temp = (kmeans_result[kmeans_result.label == i])['overall'].sum()
        key_word = '第' + str(i) + '类'
        center_overall_sum[key_word] = temp
    center_overall_sum = sorted(center_overall_sum.items(), key = lambda x:x[1], reverse = True)
    Old_label = []
    New_label = []
    num = 0
    for i in center_overall_sum:
        num += 1
        Old_label.append(int(i[0][1]))
        New_label.append(num)
    label = list(deepcopy(kmeans_result['label']))
    temp = []
    for i in range(len(label)):
        for j in range(n):
            if label[i] == Old_label[j]:
                temp.append(New_label[j])
    kmeans_result.drop('label', axis=1, inplace=True)
    kmeans_result['label'] = temp
    return kmeans_result, report

def normalise_U(U):
    for i in range(0, len(U)):
        maximum = max(U[i])
        for j in range(0, len(U[0])):
            if U[i][j] != maximum:
                U[i][j] = 0
            else:
                U[i][j] = 1
    return U
def CMeans_cluser(useful_feature, n, data_id, data_score,  __m=2.0):
    data_columns = useful_feature.columns
    KFCM_result = np.matrix(useful_feature)
    KFCM_result = KFCM_result.T
    center, u, u0, d, jm, p, fpc = cmeans(KFCM_result, m=__m, c=n, error=0.00000001, maxiter=100000)
    # print('end KFCM')
    u = u.T
    final_location = normalise_U(u)
    label = []
    for i in final_location:
        i = list(i)
        temp = i.index(1)
        label.append(temp)
    score_sil = metrics.silhouette_score(useful_feature, label, metric='euclidean')
    print("当聚为%d簇时，KFCM轮廓系数Silhouette Coefficient为：%f" % (n, score_sil))             # 计算轮廓系数
    score_cal = metrics.calinski_harabaz_score(useful_feature, label) 
    print("当聚为%d簇时，KFCM轮廓系数Calinski-Harabaz Index为：%f" % (n, score_cal))
    KFCM_result = KFCM_result.T
    KFCM_result = pd.DataFrame(KFCM_result)
    KFCM_result.columns = data_columns
    KFCM_result['label'] = label
    KFCM_result['overall'] = data_score
    center_overall_sum = {}
    for i in range(n):
        temp = (KFCM_result[KFCM_result.label == i])['overall'].sum()
        key_word = '第' + str(i) + '类'
        center_overall_sum[key_word] = temp
    center_overall_sum = sorted(center_overall_sum.items(), key = lambda x:x[1], reverse=True)
    Old_label = []
    New_label = []
    num = 0
    for i in center_overall_sum:
        num += 1
        Old_label.append(int(i[0][1]))
        New_label.append(num)
    label = list(deepcopy(KFCM_result['label']))
    temp = []
    for i in range(len(label)):
        for j in range(n):
            if label[i] == Old_label[j]:
                temp.append(New_label[j])
    KFCM_result.drop('label', axis=1, inplace=True)
    KFCM_result['label'] = temp
    KFCM_result.insert(0, 'eventid', data_id)
    return KFCM_result

def scan(useful_feature):
    #useful_feature = useful_feature.drop(useful_feature[useful_feature['var'] > 2500].index) 
    #useful_feature = useful_feature.drop(useful_feature[useful_feature['diff_var'] > 5.3950].index)
    #useful_feature = useful_feature.drop(useful_feature[useful_feature['increase30var'] > 195].index)
    #useful_feature = useful_feature.drop(useful_feature[useful_feature['increase10var'] > 78.8].index)
    #useful_feature = useful_feature.drop(useful_feature[useful_feature['increase5var'] > 33.93].index)
    #useful_feature = useful_feature.drop(useful_feature[useful_feature['increase2var'] > 12.32].index)
    #useful_feature = useful_feature.drop(useful_feature[useful_feature['RSV9max'] > 408900].index)
    #useful_feature = useful_feature.drop(useful_feature[useful_feature['RSV9var'] > 1570].index)
    #useful_feature = useful_feature.drop(useful_feature[useful_feature['RSV30range'] > 595000].index)
    #useful_feature = useful_feature.drop(useful_feature[useful_feature['RSV30max'] > 352000].index)
    #useful_feature = useful_feature.drop(useful_feature[useful_feature['RSV30var'] > 1.31].index)
    #useful_feature = useful_feature.drop(useful_feature[useful_feature['RSV30std'] > 14620].index)
    #useful_feature = useful_feature.drop(useful_feature[useful_feature['RSV90range'] > 345750].index)
    #useful_feature = useful_feature.drop(useful_feature[useful_feature['RSV90max'] > 1.3e+06].index)
    #useful_feature = useful_feature.drop(useful_feature[useful_feature['RSV90var'] == 1.17195e+09].index)
    #useful_feature = useful_feature.drop(useful_feature[useful_feature['RSV90var'] == 1.12447e+09].index)
    #useful_feature = useful_feature.drop(useful_feature[useful_feature['RSV90std'] > 7500].index)
    for i in useful_feature.columns[1:-1]:
        temp = list(useful_feature[i])
        num = list(np.arange(len(temp)))
        plt.figure()
        plt.scatter(num, temp)
        plt.title(i)
        #plt.show()
    return useful_feature

def plot_heatmap(data, step):
    dfData = data.corr()
    if step == 1:
        plt.subplots(figsize=(20, 20))
        sns.heatmap(dfData, annot=True, vmax=1, square=True, fmt='.2f')
        location = '../img/feature_related_step1.pdf'
    elif step == 2:
        plt.subplots(figsize=(7, 7))
        sns.heatmap(dfData, annot=True, vmax=1, square=True, fmt='.2f')
        location = '../img/1.1/feature_related_step2.pdf'
    plt.title('feature heatmap')
    plt.savefig(location)
    plt.show()

def deal_related(data):
    data.drop('RSV90range', axis=1, inplace=True)
    data.drop('increase30var', axis=1, inplace=True)
    data.drop('increase10var', axis=1, inplace=True)
    data.drop('increase5var', axis=1, inplace=True)
    data.drop('increase2var', axis=1, inplace=True)
    data.drop('RSV30range', axis=1, inplace=True)
    important_features = data.columns
    return data, important_features
    
def plot_distribute(data, important_features):
    mpl.rcParams['font.sans-serif'] = ['FangSong']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure()
    for i in important_features:
        sns.distplot(data[i], fit=norm)
        (mu,sigma) = norm.fit(data[i])
        print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
        plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
        plt.ylabel('Frequency')
        plt.title(i)
        plt.figure()
        stats.probplot(data[i], plot=plt)
        plt.show()

def boxcox_reverse(data, feature_name):
    plt.figure()
    soft ,b = stats.boxcox(data[feature_name])
    mpl.rcParams['font.sans-serif'] = ['FangSong']
    mpl.rcParams['axes.unicode_minus'] = False
    sns.distplot(soft, fit=norm)
    (mu,sigma) = norm.fit(soft)
    print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('reverse')
    plt.figure()
    stats.probplot(soft, plot=plt)
    plt.show()

def boxcox_trans(data, feature_name):
    soft ,b = stats.boxcox(data[feature_name])
    data[feature_name] = soft
    return data

def scan_outside_samp(data):
    for i in data.columns:
        x = list(np.arange(len(data[i])))
        y = list(deepcopy(data[i]))
        plt.figure()
        plt.scatter(x, y)
        plt.title(i)
        plt.xlabel('sample')
        plt.ylabel('value')
        location = '../img/1.1/' + i + '_dis.png'
        plt.savefig(location)
        plt.show()

def add_weight(useful_feature, entropy_weight):
    weight = {}
    feature = useful_feature.columns
    for i in feature:
        weight[i] = entropy_weight[i]
    Key = []
    Value = []
    for key, value in weight.items():
        Key.append(key)
        Value.append(value)
    Sum = sum(Value)
    Value2 = []
    for i in Value:
        Value2.append(i / Sum)
    weight = {}
    for i in range(len(Value)):
        weight[Key[i]] = Value2[i]
    for i in feature:
        useful_feature[i] = useful_feature[i] * weight[i]
    return useful_feature

def big_plate_conclusion(KFCM_result, original, n):
    final = pd.DataFrame(columns = ["id"])
    final['id'] = list(deepcopy(KFCM_result['eventid']))
    final['mean'] = list(deepcopy(original['mean']))
    final['std'] = list(deepcopy(original['std']))
    final['label'] = list(deepcopy(KFCM_result['label']))
    cort_mean = []
    cort_std = []
    cort_id = []
    for i in range(1, n+1):
        label_cou = final[final['label'].isin([i])]
        mean = list(deepcopy(label_cou['mean']))
        mean = sum(mean) / len(mean)
        std = list(deepcopy(label_cou['std']))
        std = sum(std) / len(std)
        id_ = list(deepcopy(label_cou['id']))
        cort_mean.append(mean)
        cort_std.append(std)
        cort_id.append(id_)
    cort_id = np.asarray(cort_id)
    sub_label = list(np.arange(1, n+1))
    big_plate = pd.DataFrame({'mean': cort_mean, 'std': cort_std, 'id': cort_id, 'label': sub_label})
    return big_plate
    
def sub_dataframe(old_data, KFCM_result, n):
    old_data['label'] = list(deepcopy(KFCM_result['label']))
    label_cou = []
    for i in range(1, n+1):
        label_cou.append(old_data[old_data['label'].isin([i])])
    return label_cou

def buid_submodel(data, sub_n, m):
    data_id = list(deepcopy(data['eventid']))
    data_score = list(deepcopy(data['overall']))
    data.drop('eventid', axis=1, inplace=True)
    data.drop('overall', axis=1, inplace=True)
    data.drop('label', axis=1, inplace=True)
    useful_feature, important_features = deal_related(data)
    useful_feature = boxcox_trans(useful_feature, feature_name='RSV90std')
    useful_feature = boxcox_trans(useful_feature, feature_name='var')
    useful_feature = boxcox_trans(useful_feature, feature_name='diff_var')
    useful_feature = boxcox_trans(useful_feature, feature_name='RSV9max')
    useful_feature = boxcox_trans(useful_feature, feature_name='RSV9var')
    useful_feature = boxcox_trans(useful_feature, feature_name='RSV30max')
    useful_feature = boxcox_trans(useful_feature, feature_name='RSV30var')
    useful_feature = boxcox_trans(useful_feature, feature_name='RSV30std')
    useful_feature = boxcox_trans(useful_feature, feature_name='RSV90max')
    useful_feature = boxcox_trans(useful_feature, feature_name='RSV90var')
    sub_KFCM_result = CMeans_cluser(useful_feature, sub_n, data_id, data_score,  __m=m)
    return sub_KFCM_result
    
def small_plate_conclusion(cluser, original_data, n, k):
    final = pd.DataFrame(columns = ["id"])
    final['id'] = list(deepcopy(cluser['eventid']))
    final['mean'] = list(deepcopy(original_data['mean']))
    final['std'] = list(deepcopy(original_data['std']))
    final['label'] = list(deepcopy(cluser['label']))
    cort_mean = []
    cort_std = []
    cort_id = []
    for i in range(1, n+1):
        label_cou = final[final['label'].isin([i])]
        mean = list(deepcopy(label_cou['mean']))
        mean = sum(mean) / len(mean)
        std = list(deepcopy(label_cou['std']))
        std = sum(std) / len(std)
        id_ = list(deepcopy(label_cou['id']))
        cort_mean.append(mean)
        cort_std.append(std)
        cort_id.append(id_)
    cort_id = np.asarray(cort_id)
    sub_label = list(np.arange(1, n+1))
    upper_label = []
    for i in range(len(cort_mean)):
        upper_label.append(k+1)
    small_plate = pd.DataFrame({'mean': cort_mean, 'std': cort_std, 'id': cort_id, 'sub_label': sub_label, 'upper_label': upper_label})
    return small_plate

def plot(data_norm, result):
    tsne = TSNE()
    tsne.fit_transform(data_norm)                   # 流行学习降维
    tsne = pd.DataFrame(tsne.embedding_, index=data_norm.index)
    plt.rcParams['font.sans-serif'] = ['SimHei']    # 中文正确显示
    plt.rcParams['axes.unicode_minus'] = False      # 负号正确显示
    d1 = tsne[result['label'] == 1]          # 绘图
    plt.plot(d1[0], d1[1], 'r.', label=u'cluster1', alpha=0.7, markersize=5)
    d2 = tsne[result['label'] == 2]           
    plt.plot(d2[0], d2[1], 'co', label=u'cluster2', alpha=0.7, markersize=5)
    d3 = tsne[result['label'] == 3]           
    plt.plot(d3[0], d3[1], '*', color='0.6', label=u'cluster3', alpha=0.7, markersize=5)
    d4 = tsne[result['label'] == 4]           
    plt.plot(d4[0], d4[1], 'm+', label=u'cluster4', alpha=0.7, markersize=5)
    plt.legend([u'cluster1', u'cluster2', u'cluster3', u'cluster4'],
               loc='best')
    plt.title(u'股票KFCM聚类')
    plt.show()
    plt.savefig('../img/1.1/股票KFCM聚类效果分析.jpg')
    
def plate_show(data):
    font2 = {'family': 'Times New Roman','weight' :'normal','size': 20,}
    plt.figure()
    plt.grid(ls='--')
    mean = list(deepcopy(data['mean']))
    std = list(deepcopy(data['std']))
    plt.scatter(mean, std, marker='o', s=100, label='cluster point')
    plt.title('mean&std', font2)
    plt.xlabel('mean', font2)
    plt.ylabel('std', font2)
    plt.legend()
    plt.show()
    mean_s = pd.Series(mean)    # 计算相关系数
    std_s = pd.Series(std)
    print(mean_s.corr(std_s))
    
def menu_score(data):
    menu = pd.read_excel('../result/行业分类.xlsx')
    menu_label = list(deepcopy(menu['label']))
    score_sil = metrics.silhouette_score(data, menu_label, metric='euclidean')
    print("行业划分下，KFCM轮廓系数Silhouette Coefficient为：%f" % (-score_sil))             # 计算轮廓系数
    score_cal = metrics.calinski_harabaz_score(data, menu_label) 
    print("行业划分下，KFCM轮廓系数Calinski-Harabaz Index为：%f" % (score_cal))
    
    
if __name__ == '__main__':
    useful_feature, entropy_weight, important_features, original = feature_engineering(percent=0.9)
    old_data = deepcopy(useful_feature)
    original_use_for_sub = deepcopy(original)
    data_id = list(deepcopy(useful_feature['eventid']))
    data_score = list(deepcopy(useful_feature['overall']))
    useful_feature.drop('eventid', axis=1, inplace=True)
    useful_feature.drop('overall', axis=1, inplace=True)
    #特征关联度观察
    #plot_heatmap(useful_feature, step=1)
    useful_feature, important_features = deal_related(useful_feature)
    #plot_heatmap(useful_feature, step=2)
    #观察时需要这行代码
    #plot_distribute(useful_feature, important_features)
    #观察boxcox的变换后的跟随结果
    #boxcox_reverse(useful_feature, feature_name='RSV90std')
    #boxcox_reverse(useful_feature, feature_name='var')
    #boxcox_reverse(useful_feature, feature_name='diff_var')
    useful_feature = boxcox_trans(useful_feature, feature_name='RSV90std')
    useful_feature = boxcox_trans(useful_feature, feature_name='var')
    useful_feature = boxcox_trans(useful_feature, feature_name='diff_var')
    useful_feature = boxcox_trans(useful_feature, feature_name='RSV9max')
    useful_feature = boxcox_trans(useful_feature, feature_name='RSV9var')
    useful_feature = boxcox_trans(useful_feature, feature_name='RSV30max')
    useful_feature = boxcox_trans(useful_feature, feature_name='RSV30var')
    useful_feature = boxcox_trans(useful_feature, feature_name='RSV30std')
    useful_feature = boxcox_trans(useful_feature, feature_name='RSV90max')
    useful_feature = boxcox_trans(useful_feature, feature_name='RSV90var')
    #观察离群值
    #scan_outside_samp(useful_feature)
    #这个函数没有用了
    #useful_feature = scan(useful_feature)
    #加权后轮廓系数降低，最终结果不太理想，所以才用不加权的措施
    #useful_feature = add_weight(useful_feature, entropy_weight)
    #大板块簇中心数目
    #####################
    ##m做参数调整
    #####################
    n = 4
    m = 1.7
    kmeans_result, report = KMeans_cluser(useful_feature, data_id, data_score, important_features, n)
    KFCM_result = CMeans_cluser(useful_feature, n, data_id, data_score,  __m=m)
    big_plate = big_plate_conclusion(KFCM_result, original, n)
    #聚类图可视化
    #plot(useful_feature, KFCM_result)
    #第一问大板块划分的结果，发现mean和std具有正相关，说明高风险高回报
    big_plate.to_csv('../result/big_plate.csv', index=False)
    sub_data = sub_dataframe(old_data, KFCM_result, n)
    #####################
    ##sub_n、sub_m做参数调整
    #####################
    sub_n = 10
    sub_m = [1.5, 1.5, 1.7, 1.8, 1.9]  
    sub = []
    for i in range(n):
        sub_KFCM_result = buid_submodel(sub_data[i], sub_n, sub_m[i])
        sub.append(sub_KFCM_result)
    sub_orig = sub_dataframe(original_use_for_sub, KFCM_result, n)
    sub_final = []
    for i in range(n):
        sub_final.append(small_plate_conclusion(sub[i], sub_orig[i], sub_n, i))
    for i in range(n):
        sub = sub_final[i]
        file_load = '../result/num' + str(i+1) + 'big_plate_sub.csv'
        sub.to_csv(file_load, index=False)
    """
    # sub_final[0]  0.9095056666895966
    # sub_final[1]  0.9958538235378025
    # sub_final[2]  0.9477480240611946
    # sub_final[3]  0.9411166197568799
    # big_plate     0.8076826558952767
    """
    #plate_show(sub_final[0])
    #按照标准聚醋后的评分结果
    """
    当聚为4簇时，KMeans轮廓系数Silhouette Coefficient为：0.483210
    当聚为4簇时，KMeans轮廓系数Calinski-Harabaz Index为：1555.033643
    当聚为4簇时，KFCM轮廓系数Silhouette Coefficient为：0.480489
    当聚为4簇时，KFCM轮廓系数Calinski-Harabaz Index为：1549.201505
    当聚为10簇时，KFCM轮廓系数Silhouette Coefficient为：0.279574
    当聚为10簇时，KFCM轮廓系数Calinski-Harabaz Index为：116.668197
    当聚为10簇时，KFCM轮廓系数Silhouette Coefficient为：0.184802
    当聚为10簇时，KFCM轮廓系数Calinski-Harabaz Index为：56.091092
    当聚为10簇时，KFCM轮廓系数Silhouette Coefficient为：0.285989
    当聚为10簇时，KFCM轮廓系数Calinski-Harabaz Index为：389.040537
    当聚为10簇时，KFCM轮廓系数Silhouette Coefficient为：0.313228
    当聚为10簇时，KFCM轮廓系数Calinski-Harabaz Index为：90.124834
    0.8780637261835184
    行业划分下，KFCM轮廓系数Silhouette Coefficient为：0.355196
    行业划分下，KFCM轮廓系数Calinski-Harabaz Index为：5.646729
    """
    menu_score(useful_feature)

