import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller  # Dickey-Fuller test

plt.rc("font", family="SimHei", size="12")  # 解决画图中文字体问题


# 数据类型
# 检测长度为852天 暑假的时间段暂定为7.1-8.31长度是62天
# riders数据有9年 这里metro数据只有2年半 在设定PDQs的时候不同
# 数据缺失 2018-02-12 数据应有长度852 实际长度833 丢失约20天


# 检测平稳
def stationarity(timeseries):
    rolmean = timeseries.rolling(365).mean()
    rolstd = timeseries.rolling(365).std()
    plt.figure('STATIONARITY', figsize=(12, 8))
    plt.plot(timeseries, color='r', label='Original')
    plt.plot(rolmean, color='g', label='Rolling Mean')
    plt.plot(rolstd, color='b', label='Rolling Std')

    # Perform Dickey-Fuller test:
    dftest = adfuller(timeseries, autolag='AIC')  # autolag : {‘AIC’, ‘BIC’, ‘t-stat’, None}
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print('Results of Dickey-Fuller Test:')
    print(dfoutput)


# 观察数据
df = pd.read_csv('line1.csv')
plt.figure('RAW DATA', figsize=(12, 8))
plt.xlabel('天数')
plt.ylabel('人数(万)')
df.COUNTS.plot(title='郑州地铁客流')

# 使用seasonal_decompose函数进行分析
decomposition_7 = seasonal_decompose(df.COUNTS, freq=1)
decomposition_365 = seasonal_decompose(df.COUNTS, freq=365)

# 可以分别获得趋势、季节性和随机性
observed = decomposition_365.observed
trend = decomposition_365.trend
seasonal = decomposition_365.seasonal
residual = decomposition_365.resid
plt.figure('DECOMPOSITION DATA', figsize=(12, 8))
plt.plot(observed)
plt.plot(trend)
plt.plot(seasonal)
plt.plot(residual)

# 模型不查分意味着原先序列是平稳的；1阶差分意味着原先序列有个固定的平均趋势；二阶差分意味着原先序列有个随时间变化的趋势
# check p-value <0.05 Test Statistic<0.01 选择第3个model
df['first_difference'] = df.COUNTS.diff(1)
df['seasonal_difference'] = df.COUNTS.diff(365)
df['seasonal_first_difference'] = df.first_difference.diff(365)
# stationarity(df.COUNTS)
# stationarity(df.first_difference.dropna(inplace=False))
# stationarity(df.seasonal_difference.dropna(inplace=False))
stationarity(df.seasonal_first_difference.dropna(inplace=False))

fig = plt.figure('ACF and PACF', figsize=(12, 8))
ax1 = fig.add_subplot(211)
sm.graphics.tsa.plot_acf(df.seasonal_first_difference.iloc[366:], ax=ax1)  # 这个lag是怎么来的
ax2 = fig.add_subplot(212)
sm.graphics.tsa.plot_pacf(df.seasonal_first_difference.iloc[366:], ax=ax2)  # 去掉lag怎么样

'''
p, d, q = model.order
P, D, Q, s = model.seasonal_order
d + D*s + max(3*q + 1, 3*Q*s + 1, p, P*s) + 1
Here, because of the samples restriction
(P, D, Q) can either be (0, 0, 0) or (1, 0, 0)
(1, 0, 0) cause MemoryError
(7, 0/1, 0) X (0, 0, 0, 365
(0, 0/1, 7) X (0, 0, 0, 365)
(7, 0/1, 7) X (0, 0, 0, 365)
查看他们的AIC BIC
'''
# mod = sm.tsa.statespace.SARIMAX(df.COUNTS, trend='n', order=(7, 1, 0), seasonal_order=(0, 0, 0, 365))
# results = mod.fit()
# print(results.summary())
#
# mod = sm.tsa.statespace.SARIMAX(df.COUNTS, trend='n', order=(0, 1, 7), seasonal_order=(0, 0, 0, 365))
# results = mod.fit()
# print(results.summary())

mod = sm.tsa.statespace.SARIMAX(df.COUNTS, trend='n', order=(7, 1, 7), seasonal_order=(0, 0, 0, 365))
results = mod.fit()
print(results.summary())

# 对之前进行预测
fig = plt.figure('PREDICT ON PAST', figsize=(12, 8))
ax = fig.add_subplot(111)
df['PREDICT'] = results.predict(start=0, end=833, dynamic=False)
df[['COUNTS', 'PREDICT']].plot(figsize=(12, 8), ax=ax)
plt.savefig('PREDICT ON PAST.png')

# 对之后的进行预测
day_list = [x for x in range(833, 1000)]
future = pd.DataFrame(index=day_list, columns=df.columns)
df = pd.concat([df, future])

fig = plt.figure('PREDICT ON FUTURE', figsize=(12, 8))
ax = fig.add_subplot(111)
df['PREDICT'] = results.predict(start=833, end=894, external=df, dynamic=False)
df[['COUNTS', 'PREDICT']].ix[833 - 894:].plot(ax=ax)

plt.savefig('PREDICT ON FUTURE.png')
plt.show()
