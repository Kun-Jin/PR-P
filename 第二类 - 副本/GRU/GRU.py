from matplotlib import pyplot as plt
from matplotlib.pylab import mpl
import math
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from keras.layers import LeakyReLU
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
from tensorflow.keras import Input, Model, Sequential
import pandas as pd
import numpy as np
import math
import tensorflow as tf

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
from math import sqrt


SINGLE_ATTENTION_VECTOR = False
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = inputs
    #a = Permute((2, 1))(inputs)
    #a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)

    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul

# 注意力机制的另一种写法 适合上述报错使用 来源:https://blog.csdn.net/uhauha2929/article/details/80733255
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



def create_dataset(dataset, look_back):
    '''
    对数据进行处理
    '''

    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back),:]
        dataX.append(a)
        dataY.append(dataset[i + look_back,:])
    TrainX = np.array(dataX)
    Train_Y = np.array(dataY)

    return TrainX, Train_Y

#多维归一化  返回数据和最大最小值
def NormalizeMult(data):
    #normalize 用于反归一化
    data = np.array(data)
    normalize = np.arange(2*data.shape[1],dtype='float64')

    normalize = normalize.reshape(data.shape[1],2)
    print(normalize.shape)
    for i in range(0,data.shape[1]):
        #第i列
        list = data[:,i]
        listlow,listhigh =  np.percentile(list, [0, 100])
        # print(i)
        normalize[i,0] = listlow
        normalize[i,1] = listhigh
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0,data.shape[0]):
                data[j,i]  =  (data[j,i] - listlow)/delta
                # data[j, i] = np.log(data[j, i])
    #np.save("./normalize.npy",normalize)
    return  data,normalize

#多维反归一化
def FNormalizeMult(data,normalize):
    data = np.array(data)
    for i in  range(0,data.shape[1]):
        listlow =  normalize[i,0]
        listhigh = normalize[i,1]
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0,data.shape[0]):
                data[j,i]  =  data[j,i]*delta + listlow
                # data[j, i] = math.exp(data[j, i])

    return data


def attention_model():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))

    x=LSTM(64, recurrent_activation = 'sigmoid',return_sequences=True)(inputs)
    x = LSTM(64, recurrent_activation='sigmoid')(x)

    output =Dense(units=1, activation='relu')(x)
    model = Model(inputs=[inputs], outputs=output)
    return model

data_o = pd.read_csv(r'./第二类.csv',encoding='gbk')
data=data_o.drop(['time','hour','wandan','Arrival_passenger','first_1h_confirm',
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

X_train = train_data.reshape([train_data.shape[0], 1, INPUT_DIMS])
Y_train = train_pollution_data
X_test = test_data.reshape([test_data.shape[0], 1, INPUT_DIMS])

print(X_train.shape,Y_train.shape,X_test.shape)

input_dim =X_train.shape[2]
time_steps =X_train.shape[1]
batch_size = 64

def scheduler(epoch):
    # 每隔50个epoch，学习率减小为原来的1/10
    if epoch % 50 == 0 and epoch != 0:
        lr = K.get_value(gru.optimizer.lr)
        if lr>1e-5:
            K.set_value(gru.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
    return K.get_value(gru.optimizer.lr)

reduce_lr = LearningRateScheduler(scheduler)
early_stopping = EarlyStopping(monitor='loss',
                               patience=20,
                               min_delta=1e-5,
                               mode='auto',
                               restore_best_weights=False,#是否从具有监测数量的最佳值的时期恢复模型权重
                               verbose=2)

# 特征数
input_dim = X_train.shape[2]
# 时间步长：用多少个时间步的数据来预测下一个时刻的值
time_steps = X_train.shape[1]
batch_size = 32

gru = Sequential()
input_layer =Input(batch_shape=(batch_size,time_steps,input_dim))
gru.add(input_layer)
gru.add(tf.keras.layers.GRU(64))
gru.add(tf.keras.layers.Dense(32))
gru.add(tf.keras.layers.LeakyReLU(alpha=0.3))
gru.add(tf.keras.layers.Dense(16))
gru.add(tf.keras.layers.LeakyReLU(alpha=0.3))
gru.add(tf.keras.layers.Dense(1))
gru.add(tf.keras.layers.LeakyReLU(alpha=0.3))
# 定义优化器
nadam = tf.keras.optimizers.Nadam(lr=1e-3)
gru.compile(loss = 'mse',optimizer = nadam,metrics = ['mae'])
gru.summary()

# history=gru.fit(X_train,Y_train,validation_split=0.1,epochs=200,batch_size=32,callbacks=[reduce_lr],verbose=2)
gru.fit(X_train,Y_train,epochs=1000,batch_size=32,verbose=2)
# print(history.history.keys()) #查看history中存储了哪些参数
#plt.plot(history.epoch,history.history.get('loss')) #画出随着epoch增大loss的变化图
predict = gru.predict(X_test)

y_pre_pre=predict.reshape(-1, 1)
y_pre_pre=np.tile(y_pre_pre, scaler11)
inverse_data = scaler.inverse_transform(y_pre_pre)
inverse_data=np.ceil(inverse_data)
# pd.DataFrame(inverse_data[:,0]).to_csv('./s20.csv')

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
data.iloc[len(data)-num:len(data),4].to_csv('./real.csv')
pd.DataFrame(inverse_data[:,0]-data.iloc[len(data)-num:len(data),0].values).to_csv('./error.csv')
#保存训练集数据
predict_result=gru.predict(X_train)

y_pre_pre1=predict_result.reshape(-1, 1)
y_pre_pre1=np.tile(y_pre_pre1, scaler11)
inverse_data1 = scaler.inverse_transform(y_pre_pre1)
inverse_data1=np.ceil(inverse_data1)
inverse_data1[inverse_data1[:, 0] < 0, 0] = 0
pd.DataFrame(inverse_data1[:,0]).to_csv('./train_predict.csv')
data.iloc[0:len(data)-num,0].to_csv('./train_real.csv')