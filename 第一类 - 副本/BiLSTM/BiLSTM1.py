from keras.layers import Input, Dense, LSTM, Conv1D,Dropout,Bidirectional,Multiply, Activation,LSTM,Flatten
from keras.models import Model
import math

#from attention_utils import get_activations
from keras.models import *

import  pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
from math import sqrt


SINGLE_ATTENTION_VECTOR = False

def attention_3d_block2(inputs, single_attention_vector=False):
    # 如果上一层是LSTM，需要return_sequences=True
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    # 乘上了attention权重，但是并没有求和，好像影响不大
    # 如果分类任务，进行Flatten展开就可以了
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def attention_model():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))

    lstm_out = Bidirectional(LSTM(lstm_units,recurrent_activation = 'sigmoid',return_sequences=True))(inputs)
    lstm_out = Bidirectional(LSTM(lstm_units,recurrent_activation = 'sigmoid',return_sequences=True))(lstm_out)
    # attention_mul = attention_3d_block2(lstm_out)
    attention_mul = Flatten()(lstm_out)

    # output = Dense(100, activation='relu')(attention_mul)
    output = Dense(1, activation='relu')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model

data_o = pd.read_csv(r'./第一类.csv',encoding='gbk')
data=data_o.drop(['time','hour','Arrival_passenger','wandan','rain',
                  'yes1h_news_score','is_workday','isweekday'],axis=1)

data11= scaler.fit_transform(data.iloc[:,0:])
scaler11=data11.shape[1]

shuru=data11[:,1:]
shuchu=data11[:,0]

test_set=0.1
num = round(len(shuchu) * test_set)
train_data=shuru[0:len(shuru)-num]
train_pollution_data = shuchu[0:len(shuru)-num]
test_data=shuru[len(shuru)-num:len(shuru)]
test_pollution_data = shuchu[len(shuru) - num:len(shuru)]


INPUT_DIMS =train_data.shape[1]
TIME_STEPS =1
lstm_units = 64

# train_X, _ = create_dataset(train_data,TIME_STEPS)
# _ , train_Y = create_dataset(train_pollution_data,TIME_STEPS)
# test_X, _ = create_dataset(test_data,TIME_STEPS)

train_X = train_data.reshape([train_data.shape[0], 1, INPUT_DIMS])
train_Y = train_pollution_data
test_X = test_data.reshape([test_data.shape[0], 1, INPUT_DIMS])

print(train_X.shape,train_Y.shape,test_X.shape)

m = attention_model()
m.summary()
m.compile(optimizer='adam', loss='mse')
m.fit([train_X], train_Y, epochs=1000, batch_size=32,verbose=2)
#m.save("./model.h5")
#np.save("normalize.npy",normalize)
predict_result=m.predict(test_X)

y_pre_pre=predict_result.reshape(-1, 1)
y_pre_pre=np.tile(y_pre_pre, scaler11)
inverse_data = scaler.inverse_transform(y_pre_pre)
inverse_data=np.ceil(inverse_data)
# pd.DataFrame(inverse_data[:,0]).to_csv('./s18.csv')

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
mae = np.sum(np.absolute(data.iloc[len(data)-num:len(data),0].values - inverse_data[:,0])) / len(test_pollution_data)
mape = np.sum(np.absolute(data.iloc[len(data)-num:len(data),0].values - inverse_data[:,0])/data.iloc[len(data)-num:len(data),0].values) / len(test_pollution_data)
r2 = 1-mse/ np.var(data.iloc[len(data)-num:len(data),0].values)#均方误差/方差
print(" mae:",mae," mape:",mape,"mse:",mse," rmse:",rmse," r2:",r2)
#保存结果
inverse_data[inverse_data[:, 0] < 0, 0] = 0
pd.DataFrame(inverse_data[:,0]).to_csv('./predict.csv')
data.iloc[len(data)-num:len(data),0].to_csv('./real.csv')
pd.DataFrame(inverse_data[:,0]-data.iloc[len(data)-num:len(data),0].values).to_csv('./error.csv')
#保存训练集数据
predict_result=m.predict(train_X)

y_pre_pre1=predict_result.reshape(-1, 1)
y_pre_pre1=np.tile(y_pre_pre1, scaler11)
inverse_data1 = scaler.inverse_transform(y_pre_pre1)
inverse_data1=np.ceil(inverse_data1)
inverse_data1[inverse_data1[:, 0] < 0, 0] = 0
pd.DataFrame(inverse_data1[:,0]).to_csv('./train_predict.csv')
data.iloc[0:len(data)-num,0].to_csv('./train_real.csv')

