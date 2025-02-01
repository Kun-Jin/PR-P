import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
from vmdpy import VMD
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.fftpack import fft
from math import sqrt
scaler = MinMaxScaler(feature_range=(0, 1))
import math
from xgboost import plot_tree
from Interval_criterion import interval_criterion
from sklearn.neighbors import KernelDensity
import matplotlib
#设置字体为楷体
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']

error = pd.read_csv(r'./train_error.csv',encoding='gbk')
inverse_data= pd.read_csv(r'./test_predict_cluster1.csv',encoding='gbk')
data= pd.read_csv(r'./test_real_cluster1.csv',encoding='gbk')
error =error.values.flatten()
inverse_data=inverse_data.values
data=data.values


#寻找分布
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
plt.figure()
X = error  # 转化为1D array
X.sort()
plt.hist(X, bins=50)
#寻找分布
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
plt.figure()
X = error  # 转化为1D array
X.sort()
#寻找分布
from distfit import distfit
dist = distfit(todf=True)
dist.fit_transform(X)
dist.plot()
plt.xlabel("Error")
plt.title('')
#
'''这三种方法都可以得到估计的概率密度'''
scipy_kde1 = st.gaussian_kde(X,bw_method=0.3)  # 高斯核密度估计
dens1 = scipy_kde1.evaluate(X)
scipy_kde2 = st.gaussian_kde(X,bw_method=0.5)  # 高斯核密度估计
dens2 = scipy_kde2.evaluate(X)
scipy_kde3 = st.gaussian_kde(X,bw_method=0.9)  # 高斯核密度估计
dens3 = scipy_kde3.evaluate(X)
scipy_kde4 = st.gaussian_kde(X,bw_method='scott')  # 高斯核密度估计
dens4 = scipy_kde4.evaluate(X)
print("带宽:", scipy_kde4.factor)
plt.plot(X, dens1, label='h = 0.3 for KDE',linewidth =2.0,color='cyan')
plt.plot(X, dens2, label='h = 0.5 for KDE',linewidth =2.0,color='olive')
plt.plot(X, dens3, label='h = 0.9 for KDE',linewidth =2.0,color='gray')
plt.plot(X, dens4, label='h = Scott rule for KDE',linewidth =2.0,color='silver')

plt.xlabel('Error',fontsize=40)
plt.ylabel('Frequency',fontsize=40)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.xlim((-100, 100))
plt.legend(loc="upper right",fontsize=40)  # 显示图例,设置图例字体大小
plt.savefig('mode1_kde_free_deep.pdf')
'''画累积概率分布曲线'''
plt.figure()
X_integrate = []  # 存放所有点的累积概率
X_integrate.append(0)  # 第一个值是0
point = 0  # 存放单个累积概率
for i in range(1, len(X)):
    point = point + scipy_kde4.integrate_box_1d(X[i-1], X[i])  # scipy_kde.integrate_box_1d()函数的作用是计算两个边界之间的一维概率密度函数的积分，实质上得到的是概率值
    X_integrate.append(point)
plt.plot(X, X_integrate, c='b', label='累积概率')
plt.xlabel('变量')
plt.ylabel('概率分布函数')
plt.legend()
plt.show()

PINC1=0.99
neighbour1=min(X_integrate, key=lambda x: abs(x - PINC1))
index_x1=X_integrate.index(neighbour1)
neighbour2=min(X_integrate, key=lambda x: abs(x - (1-PINC1)))
index_x2=X_integrate.index(neighbour2)
w1=11.25
w2=7.85
upper1=w1*X[index_x1]
down1=w2*X[index_x2]
#绘图
plt.figure(figsize=(10,7.5))
# plt.plot(inverse_data[:,0],"r", label = 'Prediction')
# plt.scatter(np.arange(0,num),data.iloc[len(data)-num:len(data),4].values,color="b",label = 'Real data',s=5)
aa=np.maximum(inverse_data[:,0]+down1,0)
plt.plot(inverse_data[:,0]+upper1,linestyle='-',linewidth = '0.5',color='silver')
plt.plot(aa,linestyle='-',linewidth = '0.5',color='silver')
plt.xlim((0, 479))
plt.xlabel('Hour',fontsize=20)
plt.ylabel('Order volume',fontsize=20)
plt.fill_between(range(0,len(inverse_data[:,0])),inverse_data[:,0]+upper1,aa,color='silver',alpha=0.5,label='99% confidence interval')
#计算区间预测评估指标
PICP1,PINAW1,CWC1,ACE1,MPICD1=interval_criterion(PINC1,data[:,0],aa,inverse_data[:,0]+upper1)


PINC2=0.95
neighbour1=min(X_integrate, key=lambda x: abs(x - PINC2))
index_x1=X_integrate.index(neighbour1)
neighbour2=min(X_integrate, key=lambda x: abs(x - (1-PINC2)))
index_x2=X_integrate.index(neighbour2)
upper2=w1*X[index_x1]
down2=w2*X[index_x2]
#绘图
# plt.plot(inverse_data[:,0],"r", label = 'Prediction')
aa=np.maximum(inverse_data[:,0]+down2,0)
plt.plot(inverse_data[:,0]+upper2,linestyle='-',linewidth = '0.5',color='gray')
plt.plot(aa,linestyle='-',linewidth = '0.5',color='gray')
plt.fill_between(range(0,len(inverse_data[:,0])),inverse_data[:,0]+upper2,aa,color='gray',alpha=0.5,label='95% confidence interval')
# plt.scatter(np.arange(0,num),data.iloc[len(data)-num:len(data),4].values,color="b",label = 'Real data',s=5)
plt.legend(fontsize=10, loc='upper right')
#计算区间预测评估指标
PICP2,PINAW2,CWC2,ACE2,MPICD2=interval_criterion(PINC2,data[:,0],aa,inverse_data[:,0]+upper2)

PINC3=0.9
neighbour1=min(X_integrate, key=lambda x: abs(x - PINC3))
index_x1=X_integrate.index(neighbour1)
neighbour2=min(X_integrate, key=lambda x: abs(x - (1-PINC3)))
index_x2=X_integrate.index(neighbour2)
upper3=w1*X[index_x1]
down3=w2*X[index_x2]
#绘图
# plt.plot(inverse_data[:,0],"r", label = 'Prediction')
aa=np.maximum(inverse_data[:,0]+down3,0)
plt.plot(inverse_data[:,0]+upper3,linestyle='-',linewidth = '0.5',color='gray')
plt.plot(aa,linestyle='-',linewidth = '0.5',color='gray')
plt.fill_between(range(0,len(inverse_data[:,0])),inverse_data[:,0]+upper3,aa,color='gray',alpha=0.5,label='90% confidence interval')
plt.scatter(np.arange(0,len(data)),data[:,0],color="b",label = 'Actual value',s=0.1)
# plt.plot(data.iloc[len(data)-num:len(data),1].values,"w",linewidth = '0.2')
plt.legend(fontsize=10, loc='upper right')
plt.title('IKDE',fontsize=20)
#计算区间预测评估指标
PICP3,PINAW3,CWC3,ACE3,MPICD3=interval_criterion(PINC3,data[:,0],aa,inverse_data[:,0]+upper3)
plt.savefig('IKDE.pdf')
