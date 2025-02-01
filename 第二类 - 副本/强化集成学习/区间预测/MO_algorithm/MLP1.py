def picriteria (w1,w2):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Times New Roman'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from scipy.fftpack import fft
    from math import sqrt
    scaler = MinMaxScaler(feature_range=(0, 1))
    import math
    from xgboost import plot_tree
    from Interval_criterion import interval_criterion
    from sklearn.neighbors import KernelDensity

    error = pd.read_csv(r'./train_error.csv', encoding='gbk')
    inverse_data = pd.read_csv(r'./test_predict_cluster1.csv', encoding='gbk')
    data = pd.read_csv(r'./test_real_cluster1.csv', encoding='gbk')
    error = error.values.flatten()
    inverse_data = inverse_data.values
    data = data.values
    #寻找分布
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as st
    # plt.figure()
    X = error  # 转化为1D array
    X.sort()
    # plt.hist(X, bins=50)
    #寻找分布
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as st
    # plt.figure()
    X = error  # 转化为1D array
    scipy_kde = st.gaussian_kde(X,bw_method='scott')  # 高斯核密度估计
    X.sort()
    #寻找分布
    # from distfit import distfit
    # dist = distfit(todf=True)
    # dist.fit_transform(X)
    # dist.plot()
    # plt.xlabel("Error")
    # plt.title('')

    '''这三种方法都可以得到估计的概率密度'''
    dens = scipy_kde.evaluate(X)

    # plt.plot(X, dens, label='KDE function',linewidth =2.0,color='g')
    # plt.xlabel('Error')
    # plt.ylabel('Frequency')
    # plt.legend(loc="upper right")  # 显示图例,设置图例字体大小
    # plt.show()

    '''画累积概率分布曲线'''
    # plt.figure()
    X_integrate = []  # 存放所有点的累积概率
    X_integrate.append(0)  # 第一个值是0
    point = 0  # 存放单个累积概率
    for i in range(1, len(X)):
        point = point + scipy_kde.integrate_box_1d(X[i-1], X[i])  # scipy_kde.integrate_box_1d()函数的作用是计算两个边界之间的一维概率密度函数的积分，实质上得到的是概率值
        X_integrate.append(point)
    # plt.plot(X, X_integrate, c='b', label='累积概率')
    # plt.xlabel('变量')
    # plt.ylabel('概率分布函数')
    # plt.legend()
    # plt.show()

    PINC1=0.99
    neighbour1=min(X_integrate, key=lambda x: abs(x - PINC1))
    index_x1=X_integrate.index(neighbour1)
    neighbour2=min(X_integrate, key=lambda x: abs(x - (1-PINC1)))
    index_x2=X_integrate.index(neighbour2)
    upper1=w1*X[index_x1]
    down1=w2*X[index_x2]
    #绘图
    # plt.figure(figsize=(10,7.5))
    # # plt.plot(inverse_data[:,0],"r", label = 'Prediction')
    # # plt.scatter(np.arange(0,num),data.iloc[len(data)-num:len(data),4].values,color="b",label = 'Real data',s=5)
    # plt.plot(inverse_data[:,0]+upper1,linestyle='-',linewidth = '0.5',color='silver')
    # plt.plot(inverse_data[:,0],linestyle='-',linewidth = '0.5',color='silver')
    # plt.fill_between(range(0,len(inverse_data[:,0])),inverse_data[:,0]+upper1,inverse_data[:,0]+down1,color='silver',alpha=0.5,label='99% confidence interval')
    #计算区间预测评估指标
    PICP1,PINAW1,CWC1,ACE1,MPICD1=interval_criterion(PINC1,data[:,0],inverse_data[:,0]+down1,inverse_data[:,0]+upper1)


    PINC2=0.95
    neighbour1=min(X_integrate, key=lambda x: abs(x - PINC2))
    index_x1=X_integrate.index(neighbour1)
    neighbour2=min(X_integrate, key=lambda x: abs(x - (1-PINC2)))
    index_x2=X_integrate.index(neighbour2)
    upper2=w1*X[index_x1]
    down2=w2*X[index_x2]
    #绘图
    # # plt.plot(inverse_data[:,0],"r", label = 'Prediction')
    # plt.plot(inverse_data[:,0]+upper2,linestyle='-',linewidth = '0.5',color='gray')
    # plt.plot(inverse_data[:,0]+down2,linestyle='-',linewidth = '0.5',color='gray')
    # plt.fill_between(range(0,len(inverse_data[:,0])),inverse_data[:,0]+upper2,inverse_data[:,0]+down2,color='gray',alpha=0.5,label='95% confidence interval')
    # # plt.scatter(np.arange(0,num),data.iloc[len(data)-num:len(data),4].values,color="b",label = 'Real data',s=5)
    # plt.legend(fontsize=10, loc='upper right')
    #计算区间预测评估指标
    PICP2,PINAW2,CWC2,ACE2,MPICD2=interval_criterion(PINC2,data[:,0],inverse_data[:,0]+down2,inverse_data[:,0]+upper2)

    PINC3=0.9
    neighbour1=min(X_integrate, key=lambda x: abs(x - PINC3))
    index_x1=X_integrate.index(neighbour1)
    neighbour2=min(X_integrate, key=lambda x: abs(x - (1-PINC3)))
    index_x2=X_integrate.index(neighbour2)
    upper3=w1*X[index_x1]
    down3=w2*X[index_x2]
    #绘图
    # # plt.plot(inverse_data[:,0],"r", label = 'Prediction')
    # plt.plot(inverse_data[:,0]+upper3,linestyle='-',linewidth = '0.5',color='gray')
    # plt.plot(inverse_data[:,0]+down3,linestyle='-',linewidth = '0.5',color='gray')
    # plt.fill_between(range(0,len(inverse_data[:,0])),inverse_data[:,0]+upper3,inverse_data[:,0]+down3,color='gray',alpha=0.5,label='90% confidence interval')
    # plt.scatter(np.arange(0,num),data.iloc[len(data)-num:len(data),1].values,color="b",label = 'Real data',s=0.1)
    # plt.legend(fontsize=10, loc='upper right')
    #计算区间预测评估指标
    PICP3,PINAW3,CWC3,ACE3,MPICD3=interval_criterion(PINC3,data[:,0],inverse_data[:,0]+down3,inverse_data[:,0]+upper3)
    # plt.savefig('mode1_free_per_deep.pdf')
    return (PICP1+PICP2+PICP3)/3,(PINAW1+PINAW2+PINAW3)/3
