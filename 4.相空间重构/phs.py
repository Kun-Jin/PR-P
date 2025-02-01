import numpy as np
import pandas as pd
from scipy.fftpack import fft
from scipy import fftpack
import matplotlib.pyplot as plt

N_ft=430       #时频域的点数

# 计算GP算法的时间延迟参数（自相关法）
def get_tau(imf):
    if (len(imf) != N_ft):
        print('请输入指定的数据长度！')  # 需要更改，比如弹出对话框
        return 0,0,0
    elif (isinstance(imf, np.ndarray) != True):
        print('数据格式错误！')
        return 0,0,0
    else:
        j = 1  # j为固定值
        tau_max = 20
        Rall = np.zeros(tau_max)
        for tau in range(tau_max):
            R = 0
            for i in range(N_ft - j * tau):
                R += imf[i] * imf[i + j * tau]
            Rall[tau] = R / (N_ft - j * tau)
        for tau in range(tau_max):
            if Rall[tau] < (Rall[0] * 0.6321):
                break
            tauall=np.arange(tau_max)
        return tauall,Rall,tau


# 计算GP算法的嵌入维数(假近邻算法)
def get_m(imf, tau):
    if (len(imf) != N_ft):
        print('请输入指定的数据长度！')  # 需要更改，比如弹出对话框
        return 0, 0
    elif (isinstance(imf, np.ndarray) != True):
        print('数据格式错误！')
        return 0, 0
    else:
        m_max =10
        P_m_all = []  # m_max-1个假近邻点百分率
        for m in range(1, m_max + 1):
            i_num = N_ft - (m - 1) * tau
            kj_m = np.zeros((i_num, m))  # m维重构相空间
            for i in range(i_num):
                for j in range(m):
                    kj_m[i][j] = imf[i + j * tau]
            if (m > 1):
                index = np.argsort(Dist_m)
                a_m = 0  # 最近邻点数
                for i in range(i_num):
                    temp = 0
                    for h in range(i_num):
                        temp = index[i][h]
                        if (Dist_m[i][temp] > 0):
                            break
                    D = np.linalg.norm(kj_m[i] - kj_m[temp])
                    D = np.sqrt((D * D) / (Dist_m[i][temp] * Dist_m[i][temp]) - 1)
                    if (D > 10):
                        a_m += 1
                P_m_all.append(a_m / i_num)
            i_num_m = i_num - tau
            Dist_m = np.zeros((i_num_m, i_num_m))  # 两向量之间的距离
            for i in range(i_num_m):
                for k in range(i_num_m):
                    Dist_m[i][k] = np.linalg.norm(kj_m[i] - kj_m[k])
        P_m_all = np.array(P_m_all)
        m_all = np.arange(1, m_max)
        return m_all, P_m_all


# GP算法求关联维数(时频域特征)
def GP(imf, tau):
    if (len(imf) != N_ft):
        print('请输入指定的数据长度！')  # 需要更改，比如弹出对话框
        return
    elif (isinstance(imf, np.ndarray) != True):
        print('数据格式错误！')
        return
    else:
        m_max = 10 # 最大嵌入维数
        ss = 50 # r的步长
        fig = plt.figure(1)
        for m in range(1, m_max + 1):
            i_num = N_ft - (m - 1) * tau
            kj_m = np.zeros((i_num, m))  # m维重构相空间
            for i in range(i_num):
                for j in range(m):
                    kj_m[i][j] = imf[i + j * tau]
            dist_min, dist_max = np.linalg.norm(kj_m[0] - kj_m[1]), np.linalg.norm(kj_m[0] - kj_m[1])
            Dist_m = np.zeros((i_num, i_num))  # 两向量之间的距离
            for i in range(i_num):
                for k in range(i_num):
                    D = np.linalg.norm(kj_m[i] - kj_m[k])
                    if (D > dist_max):
                        dist_max = D
                    elif (D > 0 and D < dist_min):
                        dist_min = D
                    Dist_m[i][k] = D
            dr = (dist_max - dist_min) / (ss - 1)  # r的间距
            r_m = []
            Cr_m = []
            for r_index in range(ss):
                r = dist_min + r_index * dr
                r_m.append(r)
                Temp = np.heaviside(r - Dist_m, 1)
                for i in range(i_num):
                    Temp[i][i] = 0
                Cr_m.append(np.sum(Temp))
            r_m = np.log(np.array((r_m)))
            Cr_m = np.log(np.array((Cr_m)) / (i_num * (i_num - 1)))
            plt.plot(r_m, Cr_m)
        plt.xlabel('ln(r)')
        plt.ylabel('ln(C)')
        plt.show()

if __name__=='__main__':
    # 检验关联维数程序
    t = []
    f1 = 25
    f2 = 30
    for i in range(N_ft):
        t.append(i * 0.001)
    t = np.array(t)
    # yu = np.ones(M * N)
    # AEall = np.sin(t * 2 * np.pi * f1) + np.sin(t * 2 * np.pi * f2)  #在这里直接改信号
    data = pd.read_csv(r'./第二类.csv',encoding='gb18030')
    data1=data['xiadan']
    data1=data1[:430]
    AEall=np.array(data1,dtype=float)
    ss=AEall.dtype
    tauall, Rall, tau = get_tau(AEall)
    m, P = get_m(AEall, tau)
    GP(AEall, 1)
    print(tau)
    fig2 = plt.figure(2)
    yu = np.ones(len(tauall)) * Rall[0] * 0.6321
    plt.plot(tauall, Rall)
    plt.plot(tauall, yu)
    plt.xlabel('tau')
    plt.ylabel('R')
    plt.show()
    fig3 = plt.figure(3)
    plt.plot(m, P,'o',linestyle = '-')
    plt.xlabel('Lag order')
    # plt.ylabel('P')
    plt.savefig('lag2.pdf')
    plt.show()

