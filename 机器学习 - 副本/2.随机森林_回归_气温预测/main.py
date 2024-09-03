import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape

    print('平均气温误差.',np.mean(errors))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

if __name__ == '__main__':
    features = pd.read_csv('temps.csv')
    print(features.head(5))
    print(features.describe())  # 查看特征值的统计特性
    years = features['year']
    months = features['month']
    days = features['day']
    # datetime 格式
    dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
             zip(years, months, days)]
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
    print(dates[:5])

    # 指定画图风格
    plt.style.use('fivethirtyeight')
    # 设置布局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig.autofmt_xdate(rotation=45)  # 调整图表的x轴日期标签，使其自动适应图表大小，并以45度角旋转显示。
    # 标签值
    ax1.plot(dates, np.array(features['actual']))
    ax1.set_xlabel('');
    ax1.set_ylabel('temperature');
    ax1.set_title('max temp')
    ax2.plot(dates, features['temp_1'])
    ax2.set_xlabel('');
    ax2.set_ylabel('temperature');
    ax2.set_title('previous max temp')
    ax3.plot(dates, features['temp_2'])
    ax3.set_xlabel('Date');
    ax3.set_ylabel('temperature');
    ax3.set_title('two days prior max temp')
    ax4.plot(dates, features['friend'])
    ax4.set_xlabel('Date');
    ax4.set_ylabel('temperature');
    ax4.set_title('friend estimate')
    plt.tight_layout(pad=2)  # 指定了子图之间的额外间距大小为2个单位

    # 独热编码
    features = pd.get_dummies(features)
    print(features.head(5))

    # 数据处理
    labels = np.array(features['actual'])  # y
    features = features.drop('actual', axis=1)  # x
    feature_list = list(features.columns)
    features = np.array(features)  # x.np
    # 建模
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=123)
    estimator = RandomForestRegressor(n_estimators=1000, random_state=123)  # 随机森林对特值进行随机
    estimator.fit(x_train, y_train)
    # 评估
    y_predict = estimator.predict(x_test)
    errors = abs(y_predict - y_test)
    mape = np.mean(100 * (errors / y_test))
    print('MAPE:', mape)
    accuracy = 100 - mape
    print('accuracy:', round(accuracy, 2), '%') # 保留的小数位2

    '''特征重要性分析'''
    # 特征名字 值吧？
    importances = list(estimator.feature_importances_)
    # 名字，数值组合在一起
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # 排序
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # 打印出来
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
    # 画图观察
    # 指定风格
    plt.style.use('fivethirtyeight')
    # 指定位置
    fig1 = plt.subplots(figsize=(5, 5))
    x_values = list(range(len(importances)))
    # 绘图
    plt.bar(x_values, importances, orientation='vertical', color='r', edgecolor='k', linewidth=1.2)
    # x轴名字得竖着写
    plt.xticks(x_values, feature_list, rotation='vertical')
    # 图名
    plt.ylabel('Importance');
    plt.xlabel('Variable', fontsize=1);
    plt.title('Variable Importances');

    fig2 = plt.subplots(figsize=(5, 5))
    # 对特征进行排序
    sorted_importances = [importance[1] for importance in feature_importances]
    sorted_features = [importance[0] for importance in feature_importances]
    # 累计重要性
    cumulative_importances = np.cumsum(sorted_importances)
    # 绘制折线图
    plt.plot(x_values, cumulative_importances, 'g-')
    # 画一条红色虚线，0.95那
    plt.hlines(y=0.95, xmin=0, xmax=len(sorted_importances), color='r', linestyles='dashed')
    # X轴
    plt.xticks(x_values, sorted_features, rotation='vertical')
    # Y轴和名字
    plt.xlabel('Variable');
    plt.ylabel('Cumulative Importance');
    plt.title('Cumulative Importances');


    '''网格搜索 参数选择'''
    # 建立树的个数
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)] # 起 止 步长
    # 最大特征的选择方式
    max_features = ['auto', 'sqrt'] # 10, 20 , 30
    # 树的最大深度
    max_depth = [int(x) for x in np.linspace(10, 20, num=2)]
    max_depth.append(None)
    # 节点最小分裂所需样本个数
    min_samples_split = [2, 5, 10]
    # 叶子节点最小样本数，任何分裂不能让其子节点样本数少于此值
    min_samples_leaf = [1, 2, 4]
    # 样本采样方法
    bootstrap = [True, False]

    # Random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    estimator = RandomizedSearchCV(estimator=estimator, param_distributions=random_grid,
                                   n_iter=100, scoring='neg_mean_absolute_error',
                                   cv=3, verbose=2, random_state=42, n_jobs=-1) # cv几折 ver提示什么 n_jobs=-1所有cpu跑

    # 占总重要性95%的五个重要事项的名称
    important_feature_names = ['temp_1', 'average', 'temp_2', 'friend', 'year']
    # 查找最重要功能的列 序号
    important_indices = [feature_list.index(feature) for feature in important_feature_names]
    # 创建最重要的特征的训练和测试集
    important_train_features = x_train[:, important_indices]
    important_test_features = x_test[:, important_indices]
    train_features = important_train_features[:]
    test_features = important_test_features[:]

    estimator.fit(train_features, y_train)
    print(estimator.best_params_)
    best_params = estimator.best_estimator_

    print(evaluate(estimator, test_features, y_test))














    plt.show()
