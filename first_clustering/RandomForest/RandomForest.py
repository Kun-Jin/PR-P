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
from xgboost import plot_tree


data_o = pd.read_csv(r'./第一类.csv',encoding='gbk')
data=data_o.drop(['time','hour','wandan','Arrival_passenger','rain',
                  'yes1h_news_score','is_workday','isweekday'],axis=1)

data11= scaler.fit_transform(data.iloc[:,0:])
scaler11=data11.shape[1]

shuru=data11[:,1:]
shuchu=data11[:,0]

test_set=0.1
num = round(len(shuchu) * test_set)
X_train=shuru[0:len(shuru)-num]
y_train = shuchu[0:len(shuru)-num]
X_test=shuru[len(shuru)-num:len(shuru)]
y_test = shuchu[len(shuru) - num:len(shuru)]

# data_1=data_1.reshape(1,-1)
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import RobustScaler
from xgboost.sklearn import XGBRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split,GridSearchCV

# 设置k折交叉验证的参数
kfolds = KFold(n_splits=5, shuffle=True, random_state=42)

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

def gen_models(model_name):
    if model_name == 'LR':
        # 线性回归
        return LinearRegression()
    elif model_name == 'Ridge':
        # ridge岭回归模型 使用二范数作为正则化项
        return make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))
    elif model_name == 'Lasso':
        # LASSO收缩模型（使用L1范数作为正则化项）（由于对目标函数的求解结果中将得到很多的零分量，它也被称为收缩模型。）
        return make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))
    elif model_name == 'ElasticNet':
        # 定义elastic-net弹性网络模型（弹性网络实际上是结合了岭回归和lasso的特点，同时使用了L1和L2作为正则化项。）
        return make_pipeline(RobustScaler(),
                             ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))
    elif model_name == 'SVR':
        # 支持向量机回归
        return make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003, kernel='rbf'))
        # sigmoid # poly # linear # rbf
        # return SVR(gamma='scale',kernel='linear')
    elif model_name == 'MLP':
        # 多层感知机
        return MLPRegressor(hidden_layer_sizes=(50, 150, 200, 150),
                            max_iter=10000,
                            random_state=42)
    elif model_name == 'XGBoost':
        return XGBRegressor(learning_rate=0.01, n_estimators=3460,
                            max_depth=3, min_child_weight=0,
                            gamma=0, subsample=0.7,
                            colsample_bytree=0.7,
                            objective='reg:squarederror', nthread=-1,
                            scale_pos_weight=1, seed=35,
                            reg_alpha=0.0006)

    elif model_name == 'GradientBoosting':
        # 定义GB梯度提升模型（展开到一阶导数）
        return GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',
                                         min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=42)
    elif model_name == 'LightGBM':
        # 定义lightgbm模型
        return LGBMRegressor(objective='regression')
    elif model_name == 'Stacking':
        rf = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
        gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',
                                        min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=42)
        xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                               max_depth=3, min_child_weight=0,
                               gamma=0, subsample=0.7,
                               colsample_bytree=0.7,
                               objective='reg:squarederror', nthread=-1,
                               scale_pos_weight=1, seed=27,
                               reg_alpha=0.00006)
        lightgbm = LGBMRegressor(objective='regression',
                                 num_leaves=4,
                                 learning_rate=0.01,
                                 n_estimators=5000,
                                 max_bin=200,
                                 # bagging_fraction=0.75,
                                 # bagging_freq=5,
                                 bagging_seed=7,
                                 # feature_fraction=0.2,
                                 feature_fraction_seed=7,
                                 verbose=-1,
                                 # min_data_in_leaf=2,
                                 # min_sum_hessian_in_leaf=11
                                 )
        return StackingCVRegressor(regressors=(rf, gbr, xgboost, lightgbm),
                                   meta_regressor=rf,
                                   use_features_in_secondary=True)
    elif model_name == 'RandomForest':
        return RandomForestRegressor(n_estimators=200, oob_score=True, random_state=42)


#X_train, X_test, y_train, y_test = train_test_split(features_select, target, test_size=round(len(target) * 0.1),shuffle=False)


# def train_test(train_features, test_features, train_y, test_y, model_name):
#     model = gen_models(model_name)
#     model.fit(train_features, train_y)  # 训练模型
#     pre_y = model.predict(test_features)
#     return pre_y,model
#
# y_pre_pre,model = train_test(X_train, X_test, y_train, y_test, 'RandomForest')

# 创建 RandomForestRegressor 模型
model = RandomForestRegressor(random_state=42)

# 定义超参数网格
param_grid = {
    'n_estimators': [50, 100],  # 树的数量
    'max_depth': [5, 10, None],  # 树的最大深度
    'min_samples_split': [2, 5],  # 最小分裂样本数
}

# 使用 GridSearchCV 进行 5 折交叉验证和超参数调优
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfolds, scoring='neg_mean_squared_error')

# 进行网格搜索和交叉验证
grid_search.fit(X_train, y_train)

# 输出最佳模型
best_model = grid_search.best_estimator_

# 用最佳模型进行预测
y_pre_pre = best_model.predict(X_test)

y_pre_pre=y_pre_pre.reshape(-1, 1)
y_pre_pre=np.tile(y_pre_pre, scaler11)
inverse_data = scaler.inverse_transform(y_pre_pre)
inverse_data=np.ceil(inverse_data)

plt.figure()
plt.plot(inverse_data[:,0],label="预测值")
plt.plot(data.iloc[len(data)-num:len(data),0].values,label="真实值")
plt.legend()
# names = ['2023/3/29','2023/3/30','2023/3/31','2023/4/1','2023/4/2', '2023/4/3', '2023/4/4', '2023/4/5', '2023/4/6']
# x = range(len(names))
# plt.xticks(x, names, rotation=30)
#误差
mse = np.sum((data.iloc[len(data)-num:len(data),0].values - inverse_data[:,0]) ** 2) / len(data.iloc[len(data)-num:len(data),0].values)
rmse = sqrt(mse)
mae = np.sum(np.absolute(data.iloc[len(data)-num:len(data),0].values - inverse_data[:,0])) / num
mape = np.sum(np.absolute(data.iloc[len(data)-num:len(data),0].values - inverse_data[:,0])/data.iloc[len(data)-num:len(data),0].values) / num
r2 = 1-mse/ np.var(data.iloc[len(data)-num:len(data),0].values)#均方误差/方差
print(" mae:",mae," mape:",mape,"mse:",mse," rmse:",rmse," r2:",r2)
#保存结果
inverse_data[inverse_data[:, 0] < 0, 0] = 0
pd.DataFrame(inverse_data[:,0]).to_csv('./predict_RandomForest.csv')
data.iloc[len(data)-num:len(data),0].to_csv('./real.csv')
pd.DataFrame(inverse_data[:,0]-data.iloc[len(data)-num:len(data),0].values).to_csv('./error_RandomForest.csv')

#保存训练集数据
predict_result = best_model.predict(X_train)

y_pre_pre1=predict_result.reshape(-1, 1)
y_pre_pre1=np.tile(y_pre_pre1, scaler11)
inverse_data1 = scaler.inverse_transform(y_pre_pre1)
inverse_data1=np.ceil(inverse_data1)
inverse_data1[inverse_data1[:, 0] < 0, 0] = 0
pd.DataFrame(inverse_data1[:,0]).to_csv('./train_predict.csv')
data.iloc[0:len(data)-num,0].to_csv('./train_real.csv')

