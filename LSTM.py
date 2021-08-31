import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense, LSTM
from keras.models import Sequential

df = pd.read_csv("line1.csv", encoding='ANSI')
data = df['COUNTS'].tolist()


def data_processing(raw_data, scale=True):
    if scale is True:
        return (raw_data - np.mean(raw_data)) / np.std(raw_data)  # 标准化
    else:
        return (raw_data - np.min(raw_data)) / (np.max(raw_data) - np.min(raw_data))  # 极差规格化


'''样本数据生成函数'''
TIMESTEPS = 12


def generate_data(seq):
    x = []  # 初始化输入序列X
    y = []  # 初始化输出序列Y
    '''生成连贯的时间序列类型样本集，每一个X内的一行对应指定步长的输入序列，Y内的每一行对应比X滞后一期的目标数值'''
    for j in range(len(seq) - TIMESTEPS - 1):
        x.append([seq[j:j + TIMESTEPS]])  # 从输入序列第一期出发，等步长连续不间断采样
        y.append([seq[j + TIMESTEPS]])  # 对应每个X序列的滞后一位data值
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


'''对原数据进行尺度缩放'''
data = data_processing(data)

'''将所有样本来作为训练样本'''
train_X, train_y = generate_data(data)

'''将所有样本作为测试样本'''
test_X, test_y = generate_data(data)

model = Sequential()
model.add(LSTM(16, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(train_y.shape[1]))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(train_X, train_y, epochs=1000, batch_size=len(train_X), verbose=2, shuffle=False)

# scores = model.evaluate(train_X, train_y, verbose=0)
# print("Model Accuracy: %.2f%%" % (scores[1] * 100))

result = model.predict(train_X, verbose=0)  # 长度681 少了3个

'''自定义反标准化函数'''


def scale_inv(raw_data, scale=True):
    data1 = df.iloc[:, 0].tolist()
    if scale is True:
        return raw_data * np.std(data1) + np.mean(data1)
    else:
        return raw_data * (np.max(data1) - np.min(data1)) + np.min(data1)


'''绘制反标准化之前的真实值与预测值对比图'''


# plt.figure()
# plt.plot(scale_inv(result), label='predict data')
# plt.plot(scale_inv(test_y), label='true data')
# plt.title('none-normalized')
# plt.legend()


def generate_predata(seq):
    x = list()  # 初始化输入序列X
    x.append(seq)
    return np.array(x, dtype=np.float32)


datalist = data.tolist()
pre_result = []

for i in range(150):  # 这里为什么用50？

    pre_x = generate_predata(datalist[len(datalist) - TIMESTEPS:])  # 双层array len=1 12个一组
    pre_x = np.reshape(pre_x, (1, 1, TIMESTEPS))
    pre_y = model.predict(pre_x)  # 双层array len=1
    pre_result.append(pre_y.tolist()[0])
    datalist.append(pre_y.tolist()[0][0])  # datalist的长度也在改变

res = result.tolist()
res.extend(pre_result)

'''绘制反标准化之前的真实值与预测值对比图'''
plt.figure()
plt.plot(scale_inv(np.array(res)), label='predict data')
plt.plot(scale_inv(test_y), label='true data')
plt.title('none-normalized')
plt.legend()
plt.show()
