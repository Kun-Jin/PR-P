import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.sans-serif']=['Times New Roman'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号



data = pd.read_csv(r'./ride_hailing_1h_data_after_onehot.csv',encoding='gbk')
data_1=data['xiadan']
data_before=data_1.copy()

series = data_1
# series = series - np.mean(series)  # 中心化(非必须)

# step1 嵌入
windowLen = 20 # 嵌入窗口长度
seriesLen = len(series)  # 序列长度
K = seriesLen - windowLen + 1
X = np.zeros((windowLen, K))
for i in range(K):
    X[:, i] = series[i:i + windowLen]

# step2: svd分解， U和sigma已经按升序排序
U, sigma, VT = np.linalg.svd(X, full_matrices=False)

sum_sum = 0
weight_value = 0
sigma_number = 0
for i in range(VT.shape[0]):
    sum_sum += sigma[i]
    weight_value = sum_sum / sum(sigma)
    print(weight_value)
    sigma_number += 1
    if weight_value >= 0.9:
        break

for i in range(VT.shape[0]):
    VT[i, :] *= sigma[i]
A = VT

# 重组
rec = np.zeros((windowLen, seriesLen))
for i in range(windowLen):
    for j in range(windowLen - 1):
        for m in range(j + 1):
            rec[i, j] += A[i, j - m] * U[m, i]
        rec[i, j] /= (j + 1)
    for j in range(windowLen - 1, seriesLen - windowLen + 1):
        for m in range(windowLen):
            rec[i, j] += A[i, j - m] * U[m, i]
        rec[i, j] /= windowLen
    for j in range(seriesLen - windowLen + 1, seriesLen):
        for m in range(j - seriesLen + windowLen, windowLen):
            rec[i, j] += A[i, j - m] * U[m, i]
        rec[i, j] /= (seriesLen - j)

rrr = np.sum(rec[:sigma_number,:], axis=0)  # 选择重构的部分，这里选了全部
print(sigma_number)

plt.figure(figsize=(8, 6))
for i in range(windowLen):
    ax = plt.subplot(windowLen/2, 2, i + 1)
    ax.plot(rec[i, :],linewidth=0.3)
    ax.set_xlabel("Time (hour)")
plt.savefig("SAA.pdf")


rrr=np.maximum(np.ceil(rrr),5)
data['xiadan']=rrr

plt.figure(figsize=(8, 6))
plt.plot(data_before,label="指数平滑之前",linewidth=0.3)
plt.xlim (-10,5300)
plt.ylim (0,4900)
plt.legend()

plt.figure(figsize=(8, 6))
plt.plot(rrr,label="指数平滑之后",linewidth=0.3,color="r")
plt.xlim (-10,5300)
plt.ylim (0,4900)
plt.legend()

plt.figure()
plt.plot(data_before,label="指数平滑之前")
plt.plot(rrr,label="指数平滑之后")
plt.legend()
#保存数据画图
data_before.to_csv('指数平滑之前.csv')
pd.DataFrame(rrr).to_csv('指数平滑之后.csv')