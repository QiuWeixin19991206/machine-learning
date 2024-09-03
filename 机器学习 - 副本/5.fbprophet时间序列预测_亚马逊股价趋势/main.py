# 主流时间序列算法：自回归，移动平均和整合模型
# fbprophet要求python < 3.9版本
import matplotlib.pyplot as plt
from stocker import Stocker

# microsoft = Stocker('MSFT')
# stock_history = microsoft.stock
# print(stock_history.head())

# microsoft.plot_stock()# 数据的收盘价情况
# microsoft.plot_stock(start_date = '2000-01-03', end_date = '2018-01-16',
#                      stats = ['Daily Change', 'Adj. Volume'], plot_type='pct') # Daily Change
# microsoft.buy_and_hold(start_date='1999-01-05', end_date='2002-01-03', nshares=100)
# # 显示出星期的趋势
# print(microsoft.weekly_seasonality)
# microsoft.weekly_seasonality = True
# print(microsoft.weekly_seasonality)
# model, model_data = microsoft.create_prophet_model(days=0)
# model.plot_components(model_data)
# plt.show()
#
# microsoft.changepoint_date_analysis()
# model, future = microsoft.create_prophet_model(days=180)


# 调优
amazon = Stocker('AMZN')
amazon.plot_stock()
model, model_data = amazon.create_prophet_model()
model.plot_components(model_data)
plt.show()
model, model_data = amazon.create_prophet_model(days=90)
amazon.evaluate_prediction()# 评估的指标包括：真实值和预测值之间的平均误差，上升与下降趋势，置信区间
'''Changepoint Prior Scale
该参数指定了突变点的权重，突变点就是那些突然上升，下降，或者是幅度突然变化明显的
权重大了，模型就会越符合于我们当前的训练数据集，但是过拟合也更大了。
权重小了，模型可能就欠拟合了，达不到我们预期的要求了'''
amazon.changepoint_prior_analysis(changepoint_priors=[0.001, 0.05, 0.1, 0.2])
# 评估
# 看一下模型的表现情况
amazon.changepoint_prior_validation(start_date='2016-01-04', end_date='2017-01-03', changepoint_priors=[0.001, 0.05, 0.1, 0.2])
amazon.changepoint_prior_validation(start_date='2016-01-04', end_date='2017-01-03', changepoint_priors=[0.25,0.4, 0.5, 0.6,0.7,0.8])
amazon.evaluate_prediction(nshares=1000)
amazon.evaluate_prediction(start_date = '2008-01-03', end_date = '2009-01-05', nshares=1000) # 换个时间
amazon.predict_future(days=10) # 预测未来的价格









