import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
from vmdpy import VMD
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.fftpack import fft
from math import sqrt
scaler = MinMaxScaler(feature_range=(0, 1))
import math

def interval_criterion(PINC,data,down,upper):

    #计算区间预测评估指标
    #评价置信区间PICP,PINAW,CWC，PICP用来评价预测区间的覆盖率，PINAW预测区间的宽带
    count=0
    for i in range(len(down)):
        if data[i]>=down[i] and data[i]<=upper[i]:
            count=count+1

    PICP = count/len(down)
    # print("PICP",PICP)
    #对于概率性的区间预测方法，在置信度一样的情况下，预测区间越窄越好
    max0=np.max(data)
    min0=np.min(data)
    sum0=list(map(lambda x: (x[1]-x[0]) , zip(down,upper)))
    sum1=np.sum(sum0)/len(sum0)
    PINAW = 1/(max0-min0)*sum1
    # print("PINAW",PINAW)
    #综合指标的评价cwcCWC = PINAW*(1+R(PICP)*np.exp(-y(PICP-U)))
    g=90#取值在50-100
    e0=math.exp(-g*(PICP-PINC))
    if PICP>=PINC:
        r=0
    else:
        r=1
    CWC=PINAW*(1+r*PICP*e0)
    ACE=PICP-PINC
    MPICD=np.mean(np.abs((upper+down)/2-data))
    # print("CWC",CWC)
    return PICP,PINAW,CWC,ACE,MPICD

