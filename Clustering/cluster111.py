import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
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

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
pd.set_option('display.width', 500)    #设置整体宽度
pd.set_option('display.max_rows',500)   #设置最大行数
pd.set_option('display.max_columns', 100) #设置最大列数
#设置横坐标为整数
from matplotlib.ticker import MaxNLocator

data = pd.read_csv(r'./xiadan_pivot.csv',encoding='gb18030')
data_cluster=pd.DataFrame()
data_cluster['全天均值']=np.mean(data.iloc[:,1:],axis=1)
data_cluster['全天偏度']=data.iloc[:,1:].skew(axis=1)
data_cluster['19-24均值']=np.mean(data.iloc[:,18:23],axis=1)
data_cluster['8-13均值']=np.mean(data.iloc[:,7:12],axis=1)
data_cluster['20-24峰值']=data[['1','20','21','22','23']].max(axis=1)
data_cluster['0-5峰值']=data[['0','1','2','3','4','5']].max(axis=1)
data_cluster = data_cluster * [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
#聚类
from pylab import *
import pandas as pd
import numpy as np
import operator
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import normalized_mutual_info_score  # NMI
from sklearn.metrics import rand_score  # RI
from sklearn.metrics import accuracy_score  # ACC
from sklearn.metrics import f1_score  # F-measure
from sklearn.metrics import adjusted_rand_score  # ARI


# 数据保存在.csv文件中
# iris = pd.read_csv("iris.csv", header=0)  # 鸢尾花数据集 Iris  class=3
# wine = pd.read_csv("dataset/wine.csv")  # 葡萄酒数据集 Wine  class=3
# seeds = pd.read_csv("dataset/seeds.csv")  # 小麦种子数据集 seeds  class=3
# wdbc = pd.read_csv("dataset/wdbc.csv")  # 威斯康星州乳腺癌数据集 Breast Cancer Wisconsin (Diagnostic)  class=2
# glass = pd.read_csv("dataset/glass.csv")  # 玻璃辨识数据集 Glass Identification  class=6

# df = iris  # 设置要读取的数据集
# # print(df)
# columns = list(df.columns)  # 获取数据集的第一行，第一行通常为特征名，所以先取出
# features = columns[:len(columns) - 1]  # 数据集的特征名（去除了最后一列，因为最后一列存放的是标签，不是数据）
# dataset = df[features]  # 预处理之后的数据，去除掉了第一行的数据（因为其为特征名，如果数据第一行不是特征名，可跳过这一步）
# original_labels = list(df[columns[-1]])  # 原始标签（最后一列）
# attributes = len(df.columns) - 1  # 属性数量（数据集维度）

df=data
dataset = data_cluster
attributes = df.shape[1]

# 初始化模糊矩阵U
def initializeMembershipMatrix(c):
    num_samples = df.shape[0]
    membership_mat = np.random.rand(num_samples, c)
    membership_mat = membership_mat / np.sum(membership_mat, axis=1, keepdims=True)
    return membership_mat


# 计算类中心点
def calculateClusterCenter(membership_mat, c, m):
    cluster_mem_val = zip(*membership_mat)
    cluster_centers = list()
    cluster_mem_val_list = list(cluster_mem_val)
    for j in range(c):
        x = cluster_mem_val_list[j]
        x_raised = [e ** m for e in x]
        denominator = sum(x_raised)
        temp_num = list()
        for i in range(n):
            data_point = list(dataset.iloc[i])
            prod = [x_raised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, zip(*temp_num))
        center = [z / denominator for z in numerator]  # 每一维都要计算。
        cluster_centers.append(center)
    return cluster_centers


# 更新隶属度
def updateMembershipValue(membership_mat, cluster_centers, c):
    #    p = float(2/(m-1))
    data = []
    for i in range(n):
        x = list(dataset.iloc[i])  # 取出文件中的每一行数据
        data.append(x)
        distances = [np.linalg.norm(list(map(operator.sub, x, cluster_centers[j]))) for j in range(c)]
        for j in range(c):
            den = sum([math.pow(float(distances[j] / distances[k]), 2) for k in range(c)])
            membership_mat[i][j] = float(1 / den)
    return membership_mat, data


# 得到聚类结果
def getClusters(membership_mat):
    cluster_labels = list()
    for i in range(n):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels


# FCM算法
def fuzzyCMeansClustering(c, epsilon, m, T):
    start = time.time()  # 开始时间，计时
    membership_mat = initializeMembershipMatrix(c)  # 初始化隶属度矩阵
    t = 0
    while t <= T:  # 最大迭代次数
        cluster_centers = calculateClusterCenter(membership_mat, c, m)  # 根据隶属度矩阵计算聚类中心
        old_membership_mat = membership_mat.copy()  # 保留之前的隶属度矩阵，同于判断迭代条件
        membership_mat, data = updateMembershipValue(membership_mat, cluster_centers, c)  # 新一轮迭代的隶属度矩阵
        cluster_labels = getClusters(membership_mat)  # 获取标签
        if np.linalg.norm(membership_mat - old_membership_mat) < epsilon:
            break
        print("第", t, "次迭代")
        t += 1

    print("用时：{0}".format(time.time() - start))
    # print(membership_mat)
    return cluster_labels, cluster_centers, data, membership_mat


# 计算聚类指标
def clustering_indicators(labels_true, labels_pred):
    if type(labels_true[0]) != int:
        labels_true = LabelEncoder().fit_transform(df[columns[len(columns) - 1]])  # 如果标签为文本类型，把文本标签转换为数字标签
    f_measure = f1_score(labels_true, labels_pred, average='macro')  # F值
    accuracy = accuracy_score(labels_true, labels_pred)  # ACC
    normalized_mutual_information = normalized_mutual_info_score(labels_true, labels_pred)  # NMI
    rand_index = rand_score(labels_true, labels_pred)  # RI
    ARI = adjusted_rand_score(labels_true, labels_pred)
    return f_measure, accuracy, normalized_mutual_information, rand_index, ARI


# 绘制聚类结果散点图
def draw_cluster(dataset, centers, labels):
    center_array = array(centers)
    if attributes > 2:
        dataset = PCA(n_components=2).fit_transform(dataset)  # 如果属性数量大于2，降维
        center_array = PCA(n_components=2).fit_transform(center_array)  # 如果属性数量大于2，降维
    else:
        dataset = array(dataset)
    # 做散点图
    label = array(labels)
    plt.scatter(dataset[:, 0], dataset[:, 1], marker='o', c='black', s=7)  # 原图
    # plt.show()
    colors = np.array(
        ["#FF0000", "#0000FF", "#00FF00", "#FFFF00", "#00FFFF", "#FF00FF", "#800000", "#008000", "#000080", "#808000",
         "#800080", "#008080", "#444444", "#FFD700", "#008080"])
    # 循换打印k个簇，每个簇使用不同的颜色
    for i in range(c):
        plt.scatter(dataset[nonzero(label == i), 0], dataset[nonzero(label == i), 1], c=colors[i], s=7, marker='o')
    # plt.scatter(center_array[:, 0], center_array[:, 1], marker='x', color='m', s=30)  # 聚类中心
    plt.show()


# from sklearn.metrics import silhouette_score
# T = 100  # 最大迭代数
# n = len(dataset)  # 样本数
# m = 2  # 模糊参数
# epsilon = 1e-5
# # 动态选择聚类个数并绘制轮廓系数折线图
# def select_optimal_clusters(dataset, max_clusters=10):
#     silhouette_avg = []
#     # 计算不同聚类个数的轮廓系数
#     for c in range(2, max_clusters + 1):
#         # FCM 聚类
#         labels, centers, data, membership = fuzzyCMeansClustering(c, epsilon, m, T)
#         # 计算轮廓系数
#         silhouette_avg.append(silhouette_score(dataset, labels))
#     # 选择最佳聚类个数
#     optimal_c_silhouette = np.argmax(silhouette_avg) + 2  # 从2开始
#     # 绘制轮廓系数的折线图
#     plt.figure(figsize=(8, 6))
#     plt.plot(range(2, max_clusters + 1), silhouette_avg, marker='o', color='b', label="Silhouette score")
#     plt.axvline(x=optimal_c_silhouette, color='r', linestyle='--', label=f"Optimal number of clusters: {optimal_c_silhouette}")
#     # plt.title("Silhouette Scores for Different Cluster Numbers")
#     plt.xlabel("Number of clusters")
#     plt.ylabel("Silhouette score")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('silhouette_plot.pdf')
#     plt.show()
#     print(f"The optimal number of clusters based on silhouette score: {optimal_c_silhouette}")
#     return optimal_c_silhouette
# # 使用函数选择最优聚类个数并绘制图表
# optimal_c_silhouette = select_optimal_clusters(dataset, max_clusters=10)


from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score,pairwise_distances

T = 100  # 最大迭代数
n = len(dataset)  # 样本数
m = 2  # 模糊参数
epsilon = 1e-5

# 计算聚类评价指标
def clustering_indicators(dataset, labels):
    silhouette_avg = silhouette_score(dataset, labels)  # 轮廓系数
    return silhouette_avg


# 计算模糊聚类评价指标（FPC）
def fuzzy_partition_coefficient(membership_matrix):
    N, C = membership_matrix.shape  # N是样本数，C是聚类数
    FPC = np.sum(membership_matrix ** 2) / N  # 求隶属度平方和的平均值
    return FPC


# 修改评估函数，增加Dunn指数和FPC的计算
def evaluate_cluster_quality_with_dunn_and_fpc(dataset, max_clusters=10):
    silhouette_scores = []
    fpc_scores = []

    # 计算不同聚类个数的评价指标
    for c in range(2, max_clusters + 1):
        # FCM 聚类
        labels, centers, data, membership = fuzzyCMeansClustering(c, epsilon, m, T)

        # 计算评价指标
        silhouette_avg= clustering_indicators(dataset, labels)

        # 计算FPC
        fpc = fuzzy_partition_coefficient(membership)

        # 保存各个指标的值
        silhouette_scores.append(silhouette_avg)
        fpc_scores.append(fpc)

    # 选择最佳聚类个数
    optimal_c_silhouette = np.argmax(silhouette_scores) + 2  # 从2开始
    # 绘制轮廓系数的折线图
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', color='b', label="Silhouette score")
    # 绘制FPC
    plt.plot(range(2, max_clusters + 1), fpc_scores, marker='o', color='orange', label="Fuzzy partition coefficient")
    plt.axvline(x=optimal_c_silhouette, color='r', linestyle='--', label=f"Optimal number of clusters: {optimal_c_silhouette}")
    # plt.title("Silhouette Scores for Different Cluster Numbers")
    plt.xlabel("Number of clusters")
    # plt.ylabel("Silhouette score")
    plt.legend()
    plt.grid(True)
    plt.savefig('silhouette_plot_fpc.pdf')
    plt.show()
    print(f"The optimal number of clusters based on silhouette score: {optimal_c_silhouette}")
    return optimal_c_silhouette

# 使用函数选择最优聚类个数并绘制图表
evaluate_cluster_quality_with_dunn_and_fpc(dataset, max_clusters=10)

if __name__ == '__main__':
    c = 2  # 聚类簇数
    T = 100  # 最大迭代数
    n = len(dataset)  # 样本数
    m = 2  # 模糊参数
    epsilon = 1e-5
    labels, centers, data, membership = fuzzyCMeansClustering(c, epsilon, m, T)  # 运行FCM算法
    # F_measure, ACC, NMI, RI, ARI = clustering_indicators(original_labels, labels)  # 计算聚类指标
    # print("F_measure:", F_measure, "ACC:", ACC, "NMI", NMI, "RI", RI, "ARI", ARI)
    # print(membership)
    # print(centers)
    # print(dataset)
    draw_cluster(dataset, centers, labels)

    # 将每个标签重复 24 次
    repeated_labels = np.repeat(labels, 24)

    # 创建 DataFrame
    labels_df = pd.DataFrame(repeated_labels, columns=['labels'])
    labels_df.to_csv('labels_df.csv', index=False)


















