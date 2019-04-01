#!/usr/bin/env python
# coding: utf-8
import pandas
import numpy as np
from sklearn import manifold  
from sklearn.preprocessing import scale,Imputer




dataPath="../data/shares.csv"
data=pandas.read_csv(dataPath)

# print(data.head(1)) # 取第一行
data=data.values
print(data.shape)

# 获取股票代码和名称
shareInfo=data[:,[0,1]]
print(shareInfo.shape)
# 获取股票数据
shareFeature=data[:,2:]
print(shareFeature.shape)


# 缺失值处理
indexNun=[]
for index,dt in enumerate(shareFeature):
    for mydt in dt:
        if np.isnan(mydt):
            indexNun.append(index)
            break                        
# print(indexNun)
myRange=list(range(0,data.shape[0]))
normalndex = [i for j, i in enumerate(myRange) if j not in indexNun]
unnormalFeature=shareFeature[indexNun] 
normalFeature = np.delete(shareFeature,indexNun, axis = 0)
print(normalndex)
print(normalFeature.shape)
print(indexNun)



normalFeature=scale(normalFeature) # 数据标准化
feature=manifold.TSNE(n_components=10,method='exact').fit_transform(normalFeature)
print(feature.shape)
print(feature)


#数据 合并
normald=np.insert(feature, 0, values=normalndex, axis=1)
print(normald.shape)
np.savetxt("../result/无缺失值.csv", normald, delimiter=',')


inputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
inputer.fit(unnormalFeature)
nonnormaldata=inputer.transform(unnormalFeature)
normalFeature=scale(nonnormaldata) # 数据标准化
featureNonnormal=manifold.TSNE(n_components=10,method='exact').fit_transform(nonnormaldata)
print(featureNonnormal.shape)
unnormald=np.insert(featureNonnormal, 0, values=indexNun, axis=1)
np.savetxt("../result/有缺失值.csv", unnormald, delimiter=',')


#合并
allData=np.concatenate((normald, unnormald))
print(allData.shape)
# print(allData)
allData[np.lexsort(allData[:,::-1].T)]
np.savetxt("../result/allData.csv", allData, delimiter=',')

