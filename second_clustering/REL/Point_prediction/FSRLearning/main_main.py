import numpy as np
import itertools
import time
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
from feature_selector import FeatureSelectorRL

# Get the pandas DataFrame
ensemble1 = pd.read_csv(r'./ensemble2.csv',encoding='gbk')
# Get the dataset with the features
X = ensemble1.drop(ensemble1.columns[-1], axis=1)

# Get the dataset with the label values
y = ensemble1.iloc[:, -1]

# Create the object of feature selection with RL
fsrl_obj = FeatureSelectorRL(X.shape[1], nb_iter=100)

# Returns the results of the selection and the ranking
results = fsrl_obj.fit_predict(X, y)

# fsrl_obj.compare_with_benchmark(X, y, results)
# fsrl_obj.get_plot_ratio_exploration()
fsrl_obj.get_feature_strengh(results)
# fsrl_obj.get_depth_of_visited_states()

df = pd.DataFrame({
    'rank': results[0][2]
})
df.to_csv('./rank_indices.csv')
model_num = df[df['rank'] > 0.01].index
#Q-learning结束，集成预测
model_num=[3,10]#最终选择的模型

data_o = ensemble1
data=data_o.copy()
data11 = scaler.fit_transform(data_o.iloc[:, :])
scaler11 = data11.shape[1]

shuru = data11[:, model_num]
shuchu=data11[:, -1 ]
test_set = 0.1
num = round(len(shuchu) * test_set)
X_train = shuru[0:len(shuru) - num]
y_train = shuchu[0:len(shuru) - num]
X_test = shuru[len(shuru) - num:len(shuru)]
y_test = shuchu[len(shuru) - num:len(shuru)]
# 定义MLP模型
mlp = MLPRegressor(max_iter=1000, random_state=42)
# 定义超参数搜索空间
param_space = {
    'hidden_layer_sizes': [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'lbfgs'],
    'learning_rate': ['constant', 'adaptive'],
    'alpha': (0.0001, 0.001, 0.01),
}

# 使用BayesSearchCV进行贝叶斯优化
opt = BayesSearchCV(
    mlp,
    param_space,
    n_iter=20,
    cv=5,
    random_state=42
)
# opt = MLPRegressor(hidden_layer_sizes=(32,32), activation='relu', solver='adam',alpha=0.001, max_iter=1000, random_state=42)
opt.fit(X_train, y_train)
y_pred = opt.predict(X_test).reshape(-1, 1)
y_pre_pre=np.tile(y_pred, scaler11)
inverse_data = scaler.inverse_transform(y_pre_pre)
inverse_data=np.ceil(inverse_data)

plt.figure()
plt.plot(inverse_data[:,-1],label="预测值")
plt.plot(data.iloc[len(data)-num:len(data),-1].values,label="真实值")
plt.legend()
# names = ['2023/3/29','2023/3/30','2023/3/31','2023/4/1','2023/4/2', '2023/4/3', '2023/4/4', '2023/4/5', '2023/4/6']
# x = range(len(names))
# plt.xticks(x, names, rotation=30)data_o
#误差
mse = np.sum((data.iloc[len(data)-num:len(data),-1].values - inverse_data[:,-1]) ** 2) / len(data.iloc[len(data)-num:len(data),-1].values)
rmse = sqrt(mse)
mae = np.sum(np.absolute(data.iloc[len(data)-num:len(data),-1].values - inverse_data[:,-1])) / num
mape = np.sum(np.absolute(data.iloc[len(data)-num:len(data),-1].values - inverse_data[:,-1])/data.iloc[len(data)-num:len(data),-1].values) / num
r2 = 1-mse/ np.var(data.iloc[len(data)-num:len(data),-1].values)#均方误差/方差
print(" mae:",mae," mape:",mape,"mse:",mse," rmse:",rmse," r2:",r2)
#保存结果
inverse_data[inverse_data[:, -1] < 0, -1] = 0
pd.DataFrame(inverse_data[:,-1]).to_csv('./test_predict_cluster1.csv')
data.iloc[len(data)-num:len(data),-1].to_csv('./test_real_cluster1.csv')
pd.DataFrame(inverse_data[:,-1]-data.iloc[len(data)-num:len(data),-1].values).to_csv('./test_error.csv')
#保存训练集数据
predict_result=opt.predict(X_train)

y_pre_pre1=predict_result.reshape(-1, 1)
y_pre_pre1=np.tile(y_pre_pre1, scaler11)
inverse_data1 = scaler.inverse_transform(y_pre_pre1)
inverse_data1=np.ceil(inverse_data1)
inverse_data1[inverse_data1[:, -1] < 0, -1] = 0
pd.DataFrame(inverse_data1[:,-1]).to_csv('./train_predict.csv')
data.iloc[0:len(data)-num,-1].to_csv('./train_real.csv')
pd.DataFrame(inverse_data1[:,-1]-data.iloc[0:len(data)-num,-1].values).to_csv('./train_error.csv')





