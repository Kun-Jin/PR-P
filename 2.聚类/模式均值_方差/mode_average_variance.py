import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['axes.unicode_minus']=False
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import pdist, squareform
import warnings
import os
warnings.filterwarnings('ignore')

import numpy as np
import math

pd.set_option('display.width', None)    #设置整体宽度
pd.set_option('display.max_rows',None)   #设置最大行数
pd.set_option('display.max_columns', None) #设置最大列数

pd.set_option('display.width', 500)    #设置整体宽度
pd.set_option('display.max_rows',500)   #设置最大行数
pd.set_option('display.max_columns', 100) #设置最大列数
#设置横坐标为整数
from matplotlib.ticker import MaxNLocator

data1 = pd.read_csv(r'./mode11.csv',encoding='gb18030')
data2 = pd.read_csv(r'./mode22.csv',encoding='gb18030')
mode1_average=np.mean(data1.iloc[:,1:],axis=0)
mode2_average=np.mean(data2.iloc[:,1:],axis=0)

mode1_variance=np.var(data1.iloc[:,1:],axis=0)
mode2_variance=np.var(data2.iloc[:,1:],axis=0)

plt.figure()
plt.plot(mode1_average,label="Pattern-1",marker="^",markersize=4,color='skyblue')
plt.plot(mode2_average,label="Pattern-2",marker="s",markersize=4,color='olive')
plt.ylabel('Order volume',fontsize=15)
plt.xlabel('Hour',fontsize=15)
plt.yticks(fontsize=15)
plt.xticks(fontsize=13)
plt.rcParams.update({'font.size': 15})
plt.legend()
plt.savefig('mode_mean.pdf')
plt.figure()
plt.plot(mode1_variance,label="Pattern-1",marker="^",markersize=4,color='skyblue')
plt.plot(mode2_variance,label="Pattern-2",marker="s",markersize=4,color='olive')
plt.ylabel('Order volume',fontsize=15)
plt.xlabel('Hour',fontsize=15)
plt.yticks(fontsize=15)
plt.xticks(fontsize=13)
plt.rcParams.update({'font.size': 15})
plt.legend()
plt.savefig('mode_variance.pdf')
