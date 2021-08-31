import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.graphics.api import qqplot

# 1数据准备
dta = [10930, 10318, 10595, 10972, 7706, 6756, 9092, 10551, 9722, 10913, 11151, 8186, 6422,
       6337, 11649, 11652, 10310, 12043, 7937, 6476, 9662, 9570, 9981, 9331, 9449, 6773, 6304, 9355,
       10477, 10148, 10395, 11261, 8713, 7299, 10424, 10795, 11069, 11602, 11427, 9095, 7707, 10767,
       12136, 12812, 12006, 12528, 10329, 7818, 11719, 11683, 12603, 11495, 13670, 11337, 10232,
       13261, 13230, 15535, 16837, 19598, 14823, 11622, 19391, 18177, 19994, 14723, 15694, 13248,
       9543, 12872, 13101, 15053, 12619, 13749, 10228, 9725, 14729, 12518, 14564, 15085, 14722,
       11999, 9390, 13481, 14795, 15845, 15271, 14686, 11054, 10395]

dta = np.array(dta, dtype=np.float)
dta = pd.Series(dta)
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('2001', '2090'))  # 应该是2090，不是2100 给打上时间戳
plt.figure('原始数据', figsize=(12, 8))
dta.plot()

# 2时间序列的差分d
fig = plt.figure('差分', figsize=(12, 8))
ax = fig.add_subplot(111)
diff1 = dta.diff(1)
diff1.plot(ax=ax)
# 这是观察到数据已经有周期性 所以选d=1

# 3合适的p,q
# 3.1检查平稳时间序列的自相关图和偏自相关图
diff1 = dta.diff(1)  # 我们已经知道要使用一阶差分的时间序列，之前判断差分的程序可以注释掉
fig = plt.figure('测定p,q', figsize=(12, 8))
ax1 = fig.add_subplot(211)
sm.graphics.tsa.plot_acf(dta, lags=40, ax=ax1)  # 这个图测出q
ax2 = fig.add_subplot(212)
sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)  # 这个图测出p
# 自相关图显示滞后有三个阶超出了置信边界（第一条线代表起始点，不在滞后范围内）
# 偏相关图显示在滞后1至7阶（lags 1,2,…，7）时的偏自相关系数超出了置信边界，从lag 7之后偏自相关系数值缩小至0
# ARMA(7,1)模型：即使得自相关和偏自相关都缩小至零。则是一个混合模型。
# 所以从第二个开始下降的点开始选择q 然后从第二个点开始数知道出现不在置信区域之外的点计算p


# 3.2模型选择
arma_mod70 = sm.tsa.ARMA(dta, (7, 0)).fit()
print(arma_mod70.aic, arma_mod70.bic, arma_mod70.hqic)
arma_mod30 = sm.tsa.ARMA(dta, (0, 1)).fit()
print(arma_mod30.aic, arma_mod30.bic, arma_mod30.hqic)
arma_mod71 = sm.tsa.ARMA(dta, (7, 1)).fit()
print(arma_mod71.aic, arma_mod71.bic, arma_mod71.hqic)
arma_mod80 = sm.tsa.ARMA(dta, (8, 0)).fit()
print(arma_mod80.aic, arma_mod80.bic, arma_mod80.hqic)
# 选择3个数都小的那一个


# 3.3 检验残差序列
resid = arma_mod80.resid
fig = plt.figure('残差', figsize=(12, 8))
ax1 = fig.add_subplot(211)
sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)
print(sm.stats.durbin_watson(arma_mod80.resid.values))
# 残差的ACF和PACF图，可以看到序列残差基本为白噪声 赋值很小


# 3.4 观察是否正太分布
print(stats.normaltest(resid))
plt.figure('正态分布', figsize=(12, 8))
ax = plt.gca()
qqplot(resid, line='q', ax=ax, fit=True)
# 是

# 3.5残差序列检验
r, q, p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
data = np.c_[range(1, 41), r[1:], q, p]  # 这个1-41怎么来的
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))
predict_dta = arma_mod80.predict('2090', '2100', dynamic=True)
print(predict_dta)

# 3.6预测
plt.figure('预测', figsize=(12, 8))
ax = plt.gca()
ax = dta.loc['2000':].plot(ax=ax)
arma_mod80.plot_predict('2090', '2100', dynamic=True, ax=ax, plot_insample=False)
plt.show()
