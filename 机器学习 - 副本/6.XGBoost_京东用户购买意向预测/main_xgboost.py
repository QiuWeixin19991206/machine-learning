import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import preprocessing
from datetime import datetime
from datetime import timedelta
import pickle
import os
import math
import sys
import xgboost as xgb
from sklearn.model_selection import train_test_split
import operator
import time
from sklearn.model_selection import GridSearchCV
'''这里涉及到的数据集是京东最新的数据集：
JData_User.csv 用户数据集 105,321个用户
JData_Comment.csv 商品评论 558,552条记录
JData_Product.csv 预测商品集合 24,187条记录
JData_Action_201602.csv 2月份行为交互记录 11,485,424条记录
JData_Action_201603.csv 3月份行为交互记录 25,916,378条记录
JData_Action_201604.csv 4月份行为交互记录 13,199,934条记录'''

'''数据挖掘流程：
（一）.数据清洗 
1. 数据集完整性验证 
2. 数据集中是否存在缺失值 
3. 数据集中各特征数值应该如何处理 
4. 哪些数据是我们想要的，哪些是可以过滤掉的 
5. 将有价值数据信息做成新的数据源 
6. 去除无行为交互的商品和用户 
7. 去掉浏览量很大而购买量很少的用户(惰性用户或爬虫用户) 
（二）.数据理解与分析 
1. 掌握各个特征的含义 
2. 观察数据有哪些特点，是否可利用来建模 
3. 可视化展示便于分析 
4. 用户的购买意向是否随着时间等因素变化 
（三）.特征提取 
1. 基于清洗后的数据集哪些特征是有价值 
2. 分别对用户与商品以及其之间构成的行为进行特征提取 
3. 行为因素中哪些是核心？如何提取？ 
4. 瞬时行为特征or累计行为特征？
（四）.模型建立 
1. 使用机器学习算法进行预测
2. 参数设置与调节 
3. 数据集切分？'''

#定义文件名
ACTION_201602_FILE = "data/JData_Action_201602.csv"
ACTION_201603_FILE = "data/JData_Action_201603.csv"
ACTION_201604_FILE = "data/JData_Action_201604.csv"
COMMENT_FILE = "data/JData_Comment.csv"
PRODUCT_FILE = "data/JData_Product.csv"
USER_FILE = "data/JData_User.csv"
USER_TABLE_FILE = "data/User_table.csv"
ITEM_TABLE_FILE = "data/Item_table.csv"

def analysis_data():
    df_user = pd.read_csv('data/JData_User.csv', encoding='gbk')
    df_product = pd.read_csv('data/JData_Product.csv', encoding='gbk')
    df_comment = pd.read_csv('data/JData_Comment.csv', encoding='gbk')
    JD_2 = pd.read_csv('data/JData_Action_201602.csv', encoding='gbk')
    JD_3 = pd.read_csv('data/JData_Action_201603.csv', encoding='gbk')
    JD_4 = pd.read_csv('data/JData_Action_201604.csv', encoding='gbk')

    IsDuplicated = JD_2.duplicated()
    df_d = JD_2[IsDuplicated]
    df_d.groupby('type').count()
    # 发现重复数据大多数都是由于浏览（1），或者点击(6)产生

    df_user['user_reg_tm'] = pd.to_datetime(df_user['user_reg_tm'])
    print(df_user.loc[df_user.user_reg_tm >= '2016-4-15'])
    # 由于注册时间是京东系统错误造成，如果行为数据中没有在4月15号之后的数据的话，那么说明这些用户还是正常用户，并不需要删除。
    JD_4['time'] = pd.to_datetime(JD_4['time'])
    print(JD_4.loc[JD_4.time >= '2016-4-16'])
    # 结论：说明用户没有异常操作数据，所以这一批用户不删除

    return None


def data_deal():
    df_user = pd.read_csv('data/JData_User.csv', encoding='gbk')
    df_product = pd.read_csv('data/JData_Product.csv', encoding='gbk')
    df_comment = pd.read_csv('data/JData_Comment.csv', encoding='gbk')
    JD_2 = pd.read_csv('data/JData_Action_201602.csv', encoding='gbk')
    JD_3 = pd.read_csv('data/JData_Action_201603.csv', encoding='gbk')
    JD_4 = pd.read_csv('data/JData_Action_201604.csv', encoding='gbk')

    # print(df_user.head(), df_product.head(), df_comment.head(), JD_2.head())
    # user_id  age  sex  user_lv_cd user_reg_tm
    # sku_id  a1  a2  a3  cate  brand
    # dt  sku_id  comment_num  has_bad_comment  bad_comment_rate
    # user_id  sku_id   time  model_id  type  cate  brand

    # # 转换数据类型
    # JD_2['user_id'] = JD_2['user_id'].apply(lambda x: int(x))
    # print(JD_2['user_id'].dtype)
    # JD_2.to_csv('data/JData_Action_201602.csv', index=None)
    # JD_3['user_id'] = JD_3['user_id'].apply(lambda x: int(x))
    # print(JD_3['user_id'].dtype)
    # JD_3.to_csv('data\JData_Action_201603.csv', index=None)
    # JD_4['user_id'] = JD_4['user_id'].apply(lambda x: int(x))
    # print(JD_4['user_id'].dtype)
    # JD_4.to_csv('data\JData_Action_201604.csv', index=None)
    # 检查用户名是否一致
    user_action_check(df_user, JD_2, JD_3, JD_4)
    # 检查是否有重复记录
    deduplicate(JD_3, 'Mar. action', 'data/JData_Action_201603_dedup.csv')
    deduplicate(JD_4, 'Feb. action', 'data/JData_Action_201604_dedup.csv')
    deduplicate(df_comment, 'Comment', 'data/JData_Comment_dedup.csv')
    deduplicate(df_product, 'Product', 'data/JData_Product_dedup.csv')
    deduplicate(df_user, 'User', 'data/JData_User_dedup.csv')
    # 检查是否存在注册时间在2016年-4月-15号之后的用户

    # 年龄分段
    df_user['age'] = df_user['age'].apply(tranAge) # tranAge年龄分段函数
    print(df_user.groupby(df_user['age']).count())
    '''      user_id    sex  user_lv_cd  user_reg_tm
     age                                          
    -1.0    14412  14412       14412        14412
     1.0        7      7           7            7
     2.0     8797   8797        8797         8797
     3.0    46570  46570       46570        46570
     4.0    30336  30336       30336        30336
     5.0     3325   3325        3325         3325
     6.0     1871   1871        1871         1871'''
    df_user.to_csv('data/JData_User.csv', index=None)

    # 构造了简单的用户(user)行为特征和商品(item)行为特征
    user_base = get_from_jdata_user()
    user_behavior = merge_action_data()

    # 连接成一张表，类似于SQL的左连接(left join)
    user_behavior = pd.merge(user_base, user_behavior, on=['user_id'], how='left')
    # 保存为user_table.csv
    user_behavior.to_csv(USER_TABLE_FILE, index=False)

    item_base = get_from_jdata_product()
    item_behavior = merge_action_data()
    item_comment = get_from_jdata_comment()
    # SQL: left join
    item_behavior = pd.merge(item_base, item_behavior, on=['sku_id'], how='left')
    item_behavior = pd.merge(item_behavior, item_comment, on=['sku_id'], how='left')
    item_behavior.to_csv(ITEM_TABLE_FILE, index=False)

    print(item_behavior.head())

    df_user = pd.read_csv('data/User_table.csv', header=0)
    pd.options.display.float_format = '{:,.3f}'.format  # 输出格式设置，保留三位小数
    df_user.describe()
    # 由上述统计信息发现 存在用户无任何交互记录，因此可以删除上述用户。
    print(df_user[df_user['age'].isnull()])
    delete_list = df_user[df_user['age'].isnull()].index
    df_user.drop(delete_list, axis=0, inplace=True)

    # 删除无交互记录的用户
    df_naction = df_user[(df_user['browse_num'].isnull()) & (df_user['addcart_num'].isnull()) & (df_user['delcart_num'].isnull()) & (df_user['buy_num'].isnull()) & (df_user['favor_num'].isnull()) & (df_user['click_num'].isnull())]
    df_user.drop(df_naction.index, axis=0, inplace=True)
    print('删除无交互记录的用户后还有多少数据：', len(df_user))

    # 统计并删除无购买记录的用户
    # 统计无购买记录的用户
    df_bzero = df_user[df_user['buy_num'] == 0]
    # 输出购买数为0的总记录数
    print('输出购买数为0的总记录数', len(df_bzero))
    # 删除无购买记录的用户
    df_user = df_user[df_user['buy_num'] != 0]
    # 删除爬虫及惰性用户
    bindex = df_user[df_user['buy_browse_ratio'] < 0.0005].index
    print('爬虫用户数量：',len(bindex))
    df_user.drop(bindex, axis=0, inplace=True)
    cindex = df_user[df_user['buy_click_ratio'] < 0.0005].index
    print('惰性用户数量：',len(cindex))
    df_user.drop(cindex, axis=0, inplace=True)

    df_user.describe()
    # 最终数据集

    return None


def data_explore():
    # df_user = pd.read_csv('data/JData_User.csv', encoding='gbk')
    # df_product = pd.read_csv('data/JData_Product.csv', encoding='gbk')
    # df_comment = pd.read_csv('data/JData_Comment.csv', encoding='gbk')
    # JD_2 = pd.read_csv('data/JData_Action_201602.csv', encoding='gbk')
    # JD_3 = pd.read_csv('data/JData_Action_201603.csv', encoding='gbk')
    # JD_4 = pd.read_csv('data/JData_Action_201604.csv', encoding='gbk')

    # 周一到周日各天购买情况
    df_ac = []
    df_ac.append(get_from_action_data_4(fname=ACTION_201602_FILE))
    df_ac.append(get_from_action_data_4(fname=ACTION_201603_FILE))
    df_ac.append(get_from_action_data_4(fname=ACTION_201604_FILE))
    df_ac = pd.concat(df_ac, ignore_index=True)
    print(df_ac.dtypes)
    # user_id     int64
    # sku_id      int64
    # time       object
    # dtype: object

    # 将time字段转换为datetime类型
    df_ac['time'] = pd.to_datetime(df_ac['time'])
    # 使用lambda匿名函数将时间time转换为星期(周一为1, 周日为７)
    df_ac['time'] = df_ac['time'].apply(lambda x: x.weekday() + 1)
    print(df_ac.head())

    # 周一到周日每天购买用户个数
    df_user = df_ac.groupby('time')['user_id'].nunique()
    df_user = df_user.to_frame().reset_index()
    df_user.columns = ['weekday', 'user_num']
    # 周一到周日每天购买商品个数
    df_item = df_ac.groupby('time')['sku_id'].nunique()# 按照时间（'time'）对数据进行分组，计算每天的独立商品（'sku_id'）数量
    df_item = df_item.to_frame().reset_index()
    df_item.columns = ['weekday', 'item_num']
    # 周一到周日每天购买记录个数
    df_ui = df_ac.groupby('time', as_index=False).size()# 按照时间（'time'）对数据进行分组，计算每天的记录数量
    df_ui = df_ui.to_frame().reset_index()
    df_ui.columns = ['weekday', 'user_item_num']
    # 条形宽度
    bar_width = 0.2
    # 透明度
    opacity = 0.4

    plt.bar(df_user['weekday'], df_user['user_num'], bar_width,
            alpha=opacity, color='c', label='user')
    plt.bar(df_item['weekday'] + bar_width, df_item['item_num'],
            bar_width, alpha=opacity, color='g', label='item')
    plt.bar(df_ui['weekday'] + bar_width * 2, df_ui['user_item_num'],
            bar_width, alpha=opacity, color='m', label='user_item')

    plt.xlabel('weekday')
    plt.ylabel('number')
    plt.title('A Week Purchase Table')
    plt.xticks(df_user['weekday'] + bar_width * 3 / 2., (1, 2, 3, 4, 5, 6, 7))
    plt.tight_layout()
    plt.legend(prop={'size': 10})

    # ## 一个月中各天购买量
    df_ac = get_from_action_data_4(fname=ACTION_201602_FILE)

    # 将time字段转换为datetime类型并使用lambda匿名函数将时间time转换为天
    df_ac['time'] = pd.to_datetime(df_ac['time']).apply(lambda x: x.day)

    df_user = df_ac.groupby('time')['user_id'].nunique()
    df_user = df_user.to_frame().reset_index()
    df_user.columns = ['day', 'user_num']

    df_item = df_ac.groupby('time')['sku_id'].nunique()
    df_item = df_item.to_frame().reset_index()
    df_item.columns = ['day', 'item_num']

    df_ui = df_ac.groupby('time', as_index=False).size()
    df_ui = df_ui.to_frame().reset_index()
    df_ui.columns = ['day', 'user_item_num']

    # 条形宽度
    bar_width = 0.2
    # 透明度
    opacity = 0.4
    # 天数
    day_range = range(1, len(df_user['day']) + 1, 1)
    # 设置图片大小
    plt.figure(figsize=(14, 10))

    plt.bar(df_user['day'], df_user['user_num'], bar_width,
            alpha=opacity, color='c', label='user')
    plt.bar(df_item['day'] + bar_width, df_item['item_num'],
            bar_width, alpha=opacity, color='g', label='item')
    plt.bar(df_ui['day'] + bar_width * 2, df_ui['user_item_num'],
            bar_width, alpha=opacity, color='m', label='user_item')

    plt.xlabel('day')
    plt.ylabel('number')
    plt.title('February Purchase Table')
    plt.xticks(df_user['day'] + bar_width * 3 / 2., day_range)
    # plt.ylim(0, 80)
    plt.tight_layout()
    plt.legend(prop={'size': 9})

    df_ac = get_from_action_data(fname=ACTION_201603_FILE)

    # 将time字段转换为datetime类型并使用lambda匿名函数将时间time转换为天
    df_ac['time'] = pd.to_datetime(df_ac['time']).apply(lambda x: x.day)
    df_user = df_ac.groupby('time')['user_id'].nunique()
    df_user = df_user.to_frame().reset_index()
    df_user.columns = ['day', 'user_num']

    df_item = df_ac.groupby('time')['sku_id'].nunique()
    df_item = df_item.to_frame().reset_index()
    df_item.columns = ['day', 'item_num']

    df_ui = df_ac.groupby('time', as_index=False).size()
    df_ui = df_ui.to_frame().reset_index()
    df_ui.columns = ['day', 'user_item_num']
    # 条形宽度
    bar_width = 0.2
    # 透明度
    opacity = 0.4
    # 天数
    day_range = range(1, len(df_user['day']) + 1, 1)
    # 设置图片大小
    plt.figure(figsize=(14, 10))

    plt.bar(df_user['day'], df_user['user_num'], bar_width,
            alpha=opacity, color='c', label='user')
    plt.bar(df_item['day'] + bar_width, df_item['item_num'],
            bar_width, alpha=opacity, color='g', label='item')
    plt.bar(df_ui['day'] + bar_width * 2, df_ui['user_item_num'],
            bar_width, alpha=opacity, color='m', label='user_item')

    plt.xlabel('day')
    plt.ylabel('number')
    plt.title('March Purchase Table')
    plt.xticks(df_user['day'] + bar_width * 3 / 2., day_range)
    # plt.ylim(0, 80)
    plt.tight_layout()
    plt.legend(prop={'size': 9})


    df_ac = get_from_action_data(fname=ACTION_201604_FILE)

    # 将time字段转换为datetime类型并使用lambda匿名函数将时间time转换为天
    df_ac['time'] = pd.to_datetime(df_ac['time']).apply(lambda x: x.day)
    df_user = df_ac.groupby('time')['user_id'].nunique()
    df_user = df_user.to_frame().reset_index()
    df_user.columns = ['day', 'user_num']

    df_item = df_ac.groupby('time')['sku_id'].nunique()
    df_item = df_item.to_frame().reset_index()
    df_item.columns = ['day', 'item_num']

    df_ui = df_ac.groupby('time', as_index=False).size()
    df_ui = df_ui.to_frame().reset_index()
    df_ui.columns = ['day', 'user_item_num']

    # 条形宽度
    bar_width = 0.2
    # 透明度
    opacity = 0.4
    # 天数
    day_range = range(1, len(df_user['day']) + 1, 1)
    # 设置图片大小
    plt.figure(figsize=(14, 10))

    plt.bar(df_user['day'], df_user['user_num'], bar_width,
            alpha=opacity, color='c', label='user')
    plt.bar(df_item['day'] + bar_width, df_item['item_num'],
            bar_width, alpha=opacity, color='g', label='item')
    plt.bar(df_ui['day'] + bar_width * 2, df_ui['user_item_num'],
            bar_width, alpha=opacity, color='m', label='user_item')

    plt.xlabel('day')
    plt.ylabel('number')
    plt.title('April Purchase Table')
    plt.xticks(df_user['day'] + bar_width * 3 / 2., day_range)
    # plt.ylim(0, 80)
    plt.tight_layout()
    plt.legend(prop={'size': 9})


    return None
action_1_path = r'data/JData_Action_201602.csv'
action_2_path = r'data/JData_Action_201603.csv'
action_3_path = r'data/JData_Action_201604.csv'

comment_path = r'data/JData_Comment.csv'
product_path = r'data/JData_Product.csv'
user_path = r'data/JData_User.csv'

comment_date = [
        "2016-02-01", "2016-02-08", "2016-02-15", "2016-02-22", "2016-02-29",
        "2016-03-07", "2016-03-14", "2016-03-21", "2016-03-28", "2016-04-04",
        "2016-04-11", "2016-04-15"]

def features_engineering():
    test = pd.read_csv('data/JData_Action_201602.csv')
    test[['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']] = test[
        ['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']].astype('float32')

    user = pd.read_csv(user_path, encoding='gbk')
    print(user.isnull().any())
    user.dropna(axis=0, how='any', inplace=True)
    user.isnull().any()
    print(user.isnull().any())

    # 给定训练集起始日期
    train_start_date = '2016-02-01'

    # 将起始日期转换为 datetime 类型，并加上指定的天数作为训练集结束日期
    train_end_date = datetime.strptime(train_start_date, '%Y-%m-%d') + timedelta(days=3)

    # 将训练集结束日期格式化为字符串形式
    train_end_date = train_end_date.strftime('%Y-%m-%d')

    # 设定时间间隔（天数）
    day = 3

    # 根据训练集结束日期和时间间隔计算起始日期
    start_date = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=day)

    # 将起始日期格式化为字符串形式
    start_date = start_date.strftime('%Y-%m-%d')

    # 从指定路径读取评论数据
    comments = pd.read_csv(comment_path)

    # 设定评论数据的结束日期为训练集的结束日期
    comment_date_end = train_end_date

    # 选择评论数据的开始日期为最接近且早于训练集结束日期的日期
    comment_date_begin = comment_date[0]
    for date in reversed(comment_date):
        if date < comment_date_end:
            comment_date_begin = date
            break

    # 选择指定日期范围内的评论数据
    comments = comments[comments.dt == comment_date_begin]

    # 对评论数进行独热编码
    df = pd.get_dummies(comments['comment_num'], prefix='comment_num')

    # 处理不存在的评论数类别（例如测试集中未出现的情况），添加默认值为 0 的列
    for i in range(0, 5):
        if 'comment_num_' + str(i) not in df.columns:
            df['comment_num_' + str(i)] = 0

    # 保留特定的评论数类别列
    df = df[['comment_num_0', 'comment_num_1', 'comment_num_2', 'comment_num_3', 'comment_num_4']]

    # 将评论数特征与原始评论数据合并
    comments = pd.concat([comments, df], axis=1)

    # 选择需要的列作为最终的特征数据
    comments = comments[['sku_id', 'has_bad_comment', 'bad_comment_rate', 'comment_num_0', 'comment_num_1',
                         'comment_num_2', 'comment_num_3', 'comment_num_4']]
    print(comments.head(2))
    # sku id
    # has bad comment
    # bad comment rate
    # comment num 0
    # comment num 1
    # comment num 2

    # 行为特征
    # 获取指定时间范围内的用户行为数据
    actions = get_actions(start_date, train_end_date, all_actions)

    # 选择需要的列：用户ID（user_id）、商品ID（sku_id）、商品类别（cate）、行为类型（type）
    actions = actions[['user_id', 'sku_id', 'cate', 'type']]

    # 对行为类型进行独热编码，生成列名带有前缀 'action_before_%s' % 3 的哑变量列
    df = pd.get_dummies(actions['type'], prefix='action_before_%s' % 3)

    # 将独热编码后的行为类型特征与原始行为数据合并
    actions = pd.concat([actions, df], axis=1)

    # 按用户ID（user_id）、商品ID（sku_id）、商品类别（cate）分组，对各组内的行为计数进行求和
    actions = actions.groupby(['user_id', 'sku_id', 'cate'], as_index=False).sum()

    # 显示处理后的前20行数据
    actions.head(20)

    # 分组统计，用户-类别，不同用户对不同商品类别的行为计数
    user_cate = actions.groupby(['user_id', 'cate'], as_index=False).sum()
    del user_cate['sku_id']
    del user_cate['type']

    actions = pd.merge(actions, user_cate, how='left', on=['user_id', 'cate'])

    actions[before_date + '_1_y'] = actions[before_date + '_1.0_y'] - actions[before_date + '_1.0_x']

    train_start_date = '2016-02-01'
    train_end_date = datetime.strptime(train_start_date, '%Y-%m-%d') + timedelta(days=3)
    train_end_date = train_end_date.strftime('%Y-%m-%d')
    day = 3

    start_date = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=day)
    start_date = start_date.strftime('%Y-%m-%d')
    before_date = 'user_action_%s' % day

    actions = get_actions(start_date, train_end_date, all_actions)

    df = pd.get_dummies(actions['type'], prefix=before_date)

    actions['date'] = pd.to_datetime(actions['time']).apply(lambda x: x.date())
    actions = pd.concat([actions[['user_id', 'date']], df], axis=1)
    actions_date = actions.groupby(['user_id', 'date']).sum()
    actions_date.head()

    actions_date = actions_date.unstack()
    actions_date.fillna(0, inplace=True)

    actions = actions.groupby(['user_id'], as_index=False).sum()

    actions[before_date + '_1_ratio'] = np.log(1 + actions[before_date + '_4.0']) - np.log(
        1 + actions[before_date + '_1.0'])

    actions[before_date + '_1_mean'] = actions[before_date + '_1.0'] / day

    train_start_date = '2016-02-01'
    train_end_date = datetime.strptime(train_start_date, '%Y-%m-%d') + timedelta(days=3)
    train_end_date = train_end_date.strftime('%Y-%m-%d')
    day = 3

    start_date = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=day)
    start_date = start_date.strftime('%Y-%m-%d')

    actions = get_actions(start_date, train_end_date, all_actions)
    actions = actions[['user_id', 'cate', 'type']]

    df = pd.get_dummies(actions['type'], prefix='type')
    actions = pd.concat([actions[['user_id', 'cate']], df], axis=1)
    actions = actions.groupby(['user_id', 'cate']).sum()

    actions = actions.unstack()

    actions.columns = actions.columns.swaplevel(0, 1)

    actions.columns = actions.columns.droplevel()

    actions.columns = [
        'cate_4_type1', 'cate_5_type1', 'cate_6_type1', 'cate_7_type1',
        'cate_8_type1', 'cate_9_type1', 'cate_10_type1', 'cate_11_type1',
        'cate_4_type2', 'cate_5_type2', 'cate_6_type2', 'cate_7_type2',
        'cate_8_type2', 'cate_9_type2', 'cate_10_type2', 'cate_11_type2',
        'cate_4_type3', 'cate_5_type3', 'cate_6_type3', 'cate_7_type3',
        'cate_8_type3', 'cate_9_type3', 'cate_10_type3', 'cate_11_type3',
        'cate_4_type4', 'cate_5_type4', 'cate_6_type4', 'cate_7_type4',
        'cate_8_type4', 'cate_9_type4', 'cate_10_type4', 'cate_11_type4',
        'cate_4_type5', 'cate_5_type5', 'cate_6_type5', 'cate_7_type5',
        'cate_8_type5', 'cate_9_type5', 'cate_10_type5', 'cate_11_type5',
        'cate_4_type6', 'cate_5_type6', 'cate_6_type6', 'cate_7_type6',
        'cate_8_type6', 'cate_9_type6', 'cate_10_type6', 'cate_11_type6'
    ]

    actions = actions.fillna(0)
    actions['cate_action_sum'] = actions.sum(axis=1)

    actions['cate8_percentage'] = (actions['cate_8_type1'] + actions['cate_8_type2'] +
                                    actions['cate_8_type3'] + actions['cate_8_type4'] +
                                    actions['cate_8_type5'] + actions['cate_8_type6']
                                  ) / actions['cate_action_sum']
    actions['cate8_type1_percentage'] = np.log(
        1 + actions['cate_8_type1']) - np.log(
        1 + actions['cate_8_type1'] + actions['cate_4_type1'] +
        actions['cate_5_type1'] + actions['cate_6_type1'] +
        actions['cate_7_type1'] + actions['cate_9_type1'] +
        actions['cate_10_type1'] + actions['cate_11_type1'])

    train_start_date = '2016-02-01'
    train_end_date = datetime.strptime(train_start_date, '%Y-%m-%d') + timedelta(days=3)
    train_end_date = train_end_date.strftime('%Y-%m-%d')
    day = 3

    start_date = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=day)
    start_date = start_date.strftime('%Y-%m-%d')

    actions = get_actions(start_date, train_end_date, all_actions)
    df = pd.get_dummies(actions['type'], prefix='product_action')

    actions['date'] = pd.to_datetime(actions['time']).apply(lambda x: x.date())
    actions = pd.concat([actions[['sku_id', 'date']], df], axis=1)

    actions = actions.groupby(['sku_id'], as_index=False).sum()

    days_interal = (datetime.strptime(train_end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days

    actions['product_action_1_ratio'] = np.log(1 + actions['product_action_4.0']) - np.log(
        1 + actions['product_action_1.0'])

    return None

def make_model():
    data = pd.read_csv('data/train_set.csv')

    data_x = data.loc[:, data.columns != 'label']
    data_y = data.loc[:, data.columns == 'label']

    print(data_x.head())
    print(data_y.head())

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=0)
    x_val = x_test.iloc[:1500, :]# 前1500行作为验证集
    y_val = y_test.iloc[:1500, :]

    x_test = x_test.iloc[1500:, :]# 其余部分作为测试集
    y_test = y_test.iloc[1500:, :]

    print(x_val.shape)
    print(x_test.shape)

    # 删除训练集和验证集中的user_id和sku_id列
    del x_train['user_id']
    del x_train['sku_id']

    del x_val['user_id']
    del x_val['sku_id']

    print(x_train.head())

    # %% 转换为XGBoost的DMatrix格式
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dvalid = xgb.DMatrix(x_val, label=y_val)

    # %% 设置XGBoost模型参数
    param = {
        'n_estimators': 4000,  # 树的数量
        'max_depth': 3,  # 树的最大深度
        'min_child_weight': 5,  # 叶子节点最小权重和
        'gamma': 0,  # 惩罚项系数
        'subsample': 1.0,  # 采样比例
        'colsample_bytree': 0.8,  # 树的列采样比例
        'scale_pos_weight': 10,  # 正样本的权重比例
        'eta': 0.1,  # 学习率
        'silent': 1,  # 静默模式
        'objective': 'binary:logistic',  # 二分类的逻辑回归损失函数
        'eval_metric': 'auc'  # 评价指标为AUC
    }
    num_round = param['n_estimators'] # 设置训练轮数

    plst = list(param.items())  # 将参数字典转换为列表
    evallist = [(dtrain, 'train'), (dvalid, 'eval')] # 设置评估列表，包含训练集和验证集

    # %% 训练XGBoost模型
    bst = xgb.train(plst, dtrain, num_round, evallist, early_stopping_rounds=10)
    bst.save_model('bst.model')# 保存训练好的模型

    print(bst.attributes()) # 打印模型的属性

    features = list(x_train.columns[:]) # 获取特征列表
    create_feature_map(features) #%% 生成特征重要性映射 创建特征映射文件
    feature_importance(bst) #%% 计算特征重要性

    fi = pd.read_csv('data/feature_importance_10-24.csv')
    fi.sort_values("fscore", inplace=True, ascending=False) # 按照重要性分数降序排序
    print(fi.head())

    print(x_test.head())

    # %% 准备测试数据
    users = x_test[['user_id', 'sku_id', 'cate']].copy()
    del x_test['user_id']
    del x_test['sku_id']
    x_test_DMatrix = xgb.DMatrix(x_test) # 转换为DMatrix格式
    y_pred = bst.predict(x_test_DMatrix) #, ntree_limit=bst.best_ntree_limit) # 使用模型进行预测

    x_test['pred_label'] = y_pred
    print(x_test.head())

    x_test = x_test.apply(label, axis=1) # 应用标签转换函数
    print(x_test.head())

    x_test['true_label'] = y_test # 添加实际标签到测试集中

    # x_test users = x_test[['user_id', 'sku_id', 'cate']].copy()
    x_test['user_id'] = users['user_id'] # 重新添加user_id列
    x_test['sku_id'] = users['sku_id'] # 重新添加sku_id列

    # %% 计算准确率和召回率
    # 获取所有实际购买的用户ID
    all_user_set = x_test[x_test['true_label'] == 1]['user_id'].unique()
    print(len(all_user_set))
    # 获取所有预测购买的用户ID
    all_user_test_set = x_test[x_test['pred_label'] == 1]['user_id'].unique()
    print(len(all_user_test_set))
    # 获取所有预测购买的用户和商品对
    all_user_test_item_pair = x_test[x_test['pred_label'] == 1]['user_id'].map(str) + '-' + \
                              x_test[x_test['pred_label'] == 1]['sku_id'].map(str)
    all_user_test_item_pair = np.array(all_user_test_item_pair)
    print(len(all_user_test_item_pair))
    # print (all_user_test_item_pair)

    # 计算用户层面的准确率和召回率
    pos, neg = 0, 0
    for user_id in all_user_test_set:
        if user_id in all_user_set:
            pos += 1
        else:
            neg += 1
    all_user_acc = 1.0 * pos / (pos + neg)
    all_user_recall = 1.0 * pos / len(all_user_set)
    print('所有用户中预测购买用户的准确率为 ' + str(all_user_acc))
    print('所有用户中预测购买用户的召回率' + str(all_user_recall))

    #%% 计算商品对的准确率和召回率
    all_user_item_pair = x_test[x_test['true_label'] == 1]['user_id'].map(str) + '-' + \
                         x_test[x_test['true_label'] == 1]['sku_id'].map(str)
    all_user_item_pair = np.array(all_user_item_pair)
    # print (len(all_user_item_pair))
    # print(all_user_item_pair)
    pos, neg = 0, 0
    for user_item_pair in all_user_test_item_pair:
        # print (user_item_pair)
        if user_item_pair in all_user_item_pair:
            pos += 1
        else:
            neg += 1
    all_item_acc = 1.0 * pos / (pos + neg)
    all_item_recall = 1.0 * pos / len(all_user_item_pair)
    print('所有用户中预测购买商品的准确率为 ' + str(all_item_acc))
    print('所有用户中预测购买商品的召回率' + str(all_item_recall))

    # %% 计算F1分数和最终得分
    F11 = 6.0 * all_user_recall * all_user_acc / (5.0 * all_user_recall + all_user_acc)
    F12 = 5.0 * all_item_acc * all_item_recall / (2.0 * all_item_recall + 3 * all_item_acc)
    score = 0.4 * F11 + 0.6 * F12
    print('F11=' + str(F11))
    print('F12=' + str(F12))
    print('score=' + str(score))

    return None


def user_action_check(df_user, df_month2, df_month3, df_month4):
    df_sku = df_user.loc[:, 'user_id'].to_frame() #df_sku:user_id
    # 检查用户名是否一致，保证Action中的用户ID是User中的ID的子集
    print('Is action of Feb. from User file? ', len(df_month2) == len(pd.merge(df_sku, df_month2)))
    print('Is action of Mar. from User file? ', len(df_month3) == len(pd.merge(df_sku, df_month3)))
    print('Is action of Apr. from User file? ', len(df_month4) == len(pd.merge(df_sku, df_month4)))

    return None

def deduplicate(df_file, filename, newpath):
    # 检查是否有重复记录
    # 用户同时购买多件商品，同时添加多个数量的商品到购物车等
    before = df_file.shape[0] # 获取数据框中初始的行数
    df_file.drop_duplicates(inplace=True) # 删除数据框中的重复行
    after = df_file.shape[0] # 获取删除重复行后数据框的行数
    n_dup = before-after # 计算重复记录的数量，即删除的行数
    print ('No. of duplicate records for ' + filename + ' is: ' + str(n_dup)) # 输出重复记录的数量
    if n_dup != 0:
        df_file.to_csv(newpath, index=None) # 将删除重复记录后的数据框保存到新的路径 newpath
    else:
        print ('no duplicate records in ' + filename)  # 如果没有重复记录，则输出相应提示信息

    return None

def tranAge(x):
    # 年龄分段
    if x == u'15岁以下':
        x='1'
    elif x==u'16-25岁':
        x='2'
    elif x==u'26-35岁':
        x='3'
    elif x==u'36-45岁':
        x='4'
    elif x==u'46-55岁':
        x='5'
    elif x==u'56岁以上':
        x='6'
    return x

def add_type_count(group):
    # 将用户行为分类：
    behavior_type = group.type.astype(int)
    #用户行为类别
    type_cnt = Counter(behavior_type)
    # 1: 浏览 2: 加购 3: 删除
    # 4: 购买 5: 收藏 6: 点击
    group['browse_num'] = type_cnt[1]
    group['addcart_num'] = type_cnt[2]
    group['delcart_num'] = type_cnt[3]
    group['buy_num'] = type_cnt[4]
    group['favor_num'] = type_cnt[5]
    group['click_num'] = type_cnt[6]

    return group[['user_id', 'browse_num', 'addcart_num', 'delcart_num', 'buy_num', 'favor_num', 'click_num']]

def get_from_action_data(fname, chunk_size=50000):
    # 对action数据进行统计
    # 根据自己调节chunk_size大小
    reader = pd.read_csv(fname, header=0, iterator=True,encoding='gbk')
    # # reader 是通过读取指定文件路径 fname 的 CSV 文件创建的迭代器对象
    # # header=0 表示第一行是列名
    # # iterator=True 表示以迭代器模式读取文件，可以分块读取数据
    # # encoding='gbk' 指定文件编码为 GBK，适用于包含中文字符的文件
    chunks = []
    loop = True
    while loop:
        try:
            # 只读取user_id和type两个字段
            chunk = reader.get_chunk(chunk_size)[["user_id", "type"]]
            chunks.append(chunk)
        except StopIteration:# 如果迭代器到达文件末尾，抛出 StopIteration 异常，结束循环
            loop = False
            print("Iteration is stopped")
    # 将块拼接为pandas dataframe格式
    df_ac = pd.concat(chunks, ignore_index=True)
    # 按user_id分组，对每一组进行统计，as_index 表示无索引形式返回数据
    df_ac = df_ac.groupby(['user_id'], as_index=False).apply(add_type_count)
    # 删除重复的行，只保留 user_id 列独一无二的行
    df_ac = df_ac.drop_duplicates('user_id')

    return df_ac

# 将各个action数据的统计量进行聚合
def merge_action_data():
    df_ac = []
    df_ac.append(get_from_action_data(fname=ACTION_201602_FILE))
    df_ac.append(get_from_action_data(fname=ACTION_201603_FILE))
    df_ac.append(get_from_action_data(fname=ACTION_201604_FILE))

    df_ac = pd.concat(df_ac, ignore_index=True)
    # 用户在不同action表中统计量求和
    df_ac = df_ac.groupby(['user_id'], as_index=False).sum()
    # 构造转化率字段
    df_ac['buy_addcart_ratio'] = df_ac['buy_num'] / df_ac['addcart_num']
    df_ac['buy_browse_ratio'] = df_ac['buy_num'] / df_ac['browse_num']
    df_ac['buy_click_ratio'] = df_ac['buy_num'] / df_ac['click_num']
    df_ac['buy_favor_ratio'] = df_ac['buy_num'] / df_ac['favor_num']

    # 将大于１的转化率字段置为１(100%)
    df_ac.loc[df_ac['buy_addcart_ratio'] > 1., 'buy_addcart_ratio'] = 1.
    df_ac.loc[df_ac['buy_browse_ratio'] > 1., 'buy_browse_ratio'] = 1.
    df_ac.loc[df_ac['buy_click_ratio'] > 1., 'buy_click_ratio'] = 1.
    df_ac.loc[df_ac['buy_favor_ratio'] > 1., 'buy_favor_ratio'] = 1.

    return df_ac

#　从FJData_User表中抽取需要的字段
def get_from_jdata_user():
    df_usr = pd.read_csv(USER_FILE, header=0)
    df_usr = df_usr[["user_id", "age", "sex", "user_lv_cd"]]
    return df_usr

# 读取Product中商品
def get_from_jdata_product():
    df_item = pd.read_csv(PRODUCT_FILE, header=0,encoding='gbk')
    return df_item

# 对每一个商品分组进行统计
def add_type_count_product(group):
    behavior_type = group[type].astype(int)
    type_cnt = Counter(behavior_type)

    group['browse_num'] = type_cnt[1]
    group['addcart_num'] = type_cnt[2]
    group['delcart_num'] = type_cnt[3]
    group['buy_num'] = type_cnt[4]
    group['favor_num'] = type_cnt[5]
    group['click_num'] = type_cnt[6]

    return group[['sku_id', 'browse_num', 'addcart_num',
                  'delcart_num', 'buy_num', 'favor_num',
                  'click_num']]

#对action中的数据进行统计
def get_from_action_data_product(fname, chunk_size=50000):
    reader = pd.read_csv(fname, header=0, iterator=True)
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)[["sku_id", "type"]]
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")

    df_ac = pd.concat(chunks, ignore_index=True)

    df_ac = df_ac.groupby(['sku_id'], as_index=False).apply(add_type_count_product())
    # Select unique row
    df_ac = df_ac.drop_duplicates('sku_id')

    return df_ac

# 获取评论中的商品数据,如果存在某一个商品有两个日期的评论，我们取最晚的那一个
def get_from_jdata_comment():
    df_cmt = pd.read_csv(COMMENT_FILE, header=0)
    df_cmt['dt'] = pd.to_datetime(df_cmt['dt'])
    # find latest comment index
    idx = df_cmt.groupby(['sku_id'])['dt'].transform(max) == df_cmt['dt']
    df_cmt = df_cmt[idx]

    return df_cmt[['sku_id', 'comment_num', 'has_bad_comment', 'bad_comment_rate']]

# 提取购买(type=4)的行为数据
def get_from_action_data_4(fname, chunk_size=50000):
    reader = pd.read_csv(fname, header=0, iterator=True)
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)[
                ["user_id", "sku_id", "type", "time"]]
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")

    df_ac = pd.concat(chunks, ignore_index=True)
    # type=4,为购买
    df_ac = df_ac[df_ac['type'] == 4]

    return df_ac[["user_id", "sku_id", "time"]]


def get_actions_0():
    action = pd.read_csv(action_1_path)
    return action


def get_actions_1():
    action = pd.read_csv(action_1_path)
    action[['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']] = action[
        ['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']].astype('float32')
    return action


def get_actions_2():
    action = pd.read_csv(action_1_path)
    action[['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']] = action[
        ['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']].astype('float32')

    return action


def get_actions_3():
    action = pd.read_csv(action_1_path)
    action[['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']] = action[
        ['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']].astype('float32')

    return action


def get_actions_10():
    reader = pd.read_csv(action_1_path, iterator=True)
    reader[['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']] = reader[
        ['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']].astype('float32')
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(50000)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")
    action = pd.concat(chunks, ignore_index=True)

    return action


def get_actions_20():
    reader = pd.read_csv(action_2_path, iterator=True)
    reader[['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']] = reader[
        ['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']].astype('float32')
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(50000)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")
    action = pd.concat(chunks, ignore_index=True)

    return action


def get_actions_30():
    reader = pd.read_csv(action_3_path, iterator=True)
    reader[['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']] = reader[
        ['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']].astype('float32')
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(50000)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")
    action = pd.concat(chunks, ignore_index=True)

    return action


# 读取并拼接所有行为记录文件
def get_all_action():
    action_1 = get_actions_1()
    action_2 = get_actions_2()
    action_3 = get_actions_3()
    actions = pd.concat([action_1, action_2, action_3])  # type: pd.DataFrame
    # actions = pd.concat([action_1, action_2])
    #     actions = pd.read_csv(action_path)
    return actions


# 获取某个时间段的行为记录
def get_actions(start_date, end_date, all_actions):
    """
    :param start_date:
    :param end_date:
    :return: actions: pd.Dataframe
    """
    actions = all_actions[(all_actions.time >= start_date) & (all_actions.time < end_date)].copy()
    return actions

'''用户特征'''
def get_basic_user_feat():
    # 获取基本的用户特征，基于用户本身属性多为类别特征的特点，对age,sex,usr_lv_cd进行独热编码操作，对于用户注册时间暂时不处
    # 针对年龄的中文字符问题处理，首先是读入的时候编码，填充空值，然后将其数值化，最后独热编码，此外对于sex也进行了数值类型转换
    user = pd.read_csv(user_path, encoding='gbk')
    #user['age'].fillna('-1', inplace=True)
    #user['sex'].fillna(2, inplace=True)
    user.dropna(axis=0, how='any',inplace=True) # how='any'：表示只要有任何一个空值（NaN），就删除该行。
    user['sex'] = user['sex'].astype(int)
    user['age'] = user['age'].astype(int)
    le = preprocessing.LabelEncoder() # 将每个类别字符串映射到一个整数，从而为分类数据创建数值标签。
    age_df = le.fit_transform(user['age'])
#     print list(le.classes_)

    age_df = pd.get_dummies(age_df, prefix='age')# 为哑变量
    sex_df = pd.get_dummies(user['sex'], prefix='sex')# 为哑变量
    user_lv_df = pd.get_dummies(user['user_lv_cd'], prefix='user_lv_cd')# 为哑变量
    user = pd.concat([user['user_id'], age_df, sex_df, user_lv_df], axis=1) # 保留 user_id 列，并将哑变量列与之合并
    return user

'''商品特征'''
def get_basic_product_feat():
    # 根据商品文件获取基本的特征，针对属性a1,a2,a3进行独热编码，商品类别和品牌直接作为特征
    product = pd.read_csv(product_path)
    attr1_df = pd.get_dummies(product["a1"], prefix="a1")
    attr2_df = pd.get_dummies(product["a2"], prefix="a2")
    attr3_df = pd.get_dummies(product["a3"], prefix="a3")
    product = pd.concat([product[['sku_id', 'cate', 'brand']], attr1_df, attr2_df, attr3_df], axis=1)
    return product

'''评论特征'''
def get_comments_product_feat(end_date):
    # 分时间段
    # 对评论数进行独热编码

    comments = pd.read_csv(comment_path)

    # 确定评论时间范围
    comment_date_end = end_date
    comment_date_begin = comment_date[0]

    # 找到最接近且早于指定结束日期的评论日期
    for date in reversed(comment_date):
        if date < comment_date_end:
            comment_date_begin = date
            break

    # 选择指定日期的评论数据
    comments = comments[comments.dt == comment_date_begin]

    # 对评论数进行独热编码
    df = pd.get_dummies(comments['comment_num'], prefix='comment_num')

    # 为了防止某个时间段不具备评论数为0的情况（测试集出现过这种情况）
    for i in range(0, 5):
        if 'comment_num_' + str(i) not in df.columns:
            df['comment_num_' + str(i)] = 0
    df = df[['comment_num_0', 'comment_num_1', 'comment_num_2', 'comment_num_3', 'comment_num_4']]

    # 将评论数特征与原始评论数据合并
    comments = pd.concat([comments, df], axis=1)  # type: pd.DataFrame
    # del comments['dt']
    # del comments['comment_num']

    # 选择需要的列作为最终的特征
    comments = comments[['sku_id', 'has_bad_comment', 'bad_comment_rate', 'comment_num_0', 'comment_num_1',
                         'comment_num_2', 'comment_num_3', 'comment_num_4']]
    return comments

'''行为特征'''
def get_action_feat(start_date, end_date, all_actions, i):
    actions = get_actions(start_date, end_date, all_actions)
    actions = actions[['user_id', 'sku_id', 'cate','type']]
    # 不同时间累积的行为计数（3,5,7,10,15,21,30）
    df = pd.get_dummies(actions['type'], prefix='action_before_%s' %i)
    before_date = 'action_before_%s' %i
    actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
    # 分组统计，用户-类别-商品,不同用户对不同类别下商品的行为计数
    actions = actions.groupby(['user_id', 'sku_id','cate'], as_index=False).sum()
    # 分组统计，用户-类别，不同用户对不同商品类别的行为计数
    user_cate = actions.groupby(['user_id','cate'], as_index=False).sum()
    del user_cate['sku_id']
    del user_cate['type']
    actions = pd.merge(actions, user_cate, how='left', on=['user_id','cate'])
    #本类别下其他商品点击量
    # 前述两种分组含有相同名称的不同行为的计数，系统会自动针对名称调整添加后缀,x,y，所以这里作差统计的是同一类别下其他商品的行为计数
    actions[before_date+'_1.0_y'] = actions[before_date+'_1.0_y'] - actions[before_date+'_1.0_x']
    actions[before_date+'_2.0_y'] = actions[before_date+'_2.0_y'] - actions[before_date+'_2.0_x']
    actions[before_date+'_3.0_y'] = actions[before_date+'_3.0_y'] - actions[before_date+'_3.0_x']
    actions[before_date+'_4.0_y'] = actions[before_date+'_4.0_y'] - actions[before_date+'_4.0_x']
    actions[before_date+'_5.0_y'] = actions[before_date+'_5.0_y'] - actions[before_date+'_5.0_x']
    actions[before_date+'_6.0_y'] = actions[before_date+'_6.0_y'] - actions[before_date+'_6.0_x']
    # 统计用户对不同类别下商品计数与该类别下商品行为计数均值（对时间）的差值
    actions[before_date+'minus_mean_1'] = actions[before_date+'_1.0_x'] - (actions[before_date+'_1.0_x']/i)
    actions[before_date+'minus_mean_2'] = actions[before_date+'_2.0_x'] - (actions[before_date+'_2.0_x']/i)
    actions[before_date+'minus_mean_3'] = actions[before_date+'_3.0_x'] - (actions[before_date+'_3.0_x']/i)
    actions[before_date+'minus_mean_4'] = actions[before_date+'_4.0_x'] - (actions[before_date+'_4.0_x']/i)
    actions[before_date+'minus_mean_5'] = actions[before_date+'_5.0_x'] - (actions[before_date+'_5.0_x']/i)
    actions[before_date+'minus_mean_6'] = actions[before_date+'_6.0_x'] - (actions[before_date+'_6.0_x']/i)
    del actions['type']
    # 保留cate特征
#     del actions['cate']

    return actions

# 用户-行为
def get_accumulate_user_feat(end_date, all_actions, day):
    start_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=day)
    start_date = start_date.strftime('%Y-%m-%d')
    before_date = 'user_action_%s' % day

    feature = [
        'user_id', before_date + '_1', before_date + '_2', before_date + '_3',
        before_date + '_4', before_date + '_5', before_date + '_6',
        before_date + '_1_ratio', before_date + '_2_ratio',
        before_date + '_3_ratio', before_date + '_5_ratio',
        before_date + '_6_ratio', before_date + '_1_mean',
        before_date + '_2_mean', before_date + '_3_mean',
        before_date + '_4_mean', before_date + '_5_mean',
        before_date + '_6_mean', before_date + '_1_std',
        before_date + '_2_std', before_date + '_3_std', before_date + '_4_std',
        before_date + '_5_std', before_date + '_6_std'
    ]

    actions = get_actions(start_date, end_date, all_actions)
    df = pd.get_dummies(actions['type'], prefix=before_date)

    actions['date'] = pd.to_datetime(actions['time']).apply(lambda x: x.date())

    actions = pd.concat([actions[['user_id', 'date']], df], axis=1)
    # 分组统计，用户不同日期的行为计算标准差
#    actions_date = actions.groupby(['user_id', 'date']).sum()
#    actions_date = actions_date.unstack()
#   actions_date.fillna(0, inplace=True)
#    action_1 = np.std(actions_date[before_date + '_1'], axis=1)
#    action_1 = action_1.to_frame()
#    action_1.columns = [before_date + '_1_std']
#    action_2 = np.std(actions_date[before_date + '_2'], axis=1)
#    action_2 = action_2.to_frame()
#    action_2.columns = [before_date + '_2_std']
#    action_3 = np.std(actions_date[before_date + '_3'], axis=1)
#    action_3 = action_3.to_frame()
#    action_3.columns = [before_date + '_3_std']
#    action_4 = np.std(actions_date[before_date + '_4'], axis=1)
#    action_4 = action_4.to_frame()
#    action_4.columns = [before_date + '_4_std']
#    action_5 = np.std(actions_date[before_date + '_5'], axis=1)
#    action_5 = action_5.to_frame()
#    action_5.columns = [before_date + '_5_std']
#   action_6 = np.std(actions_date[before_date + '_6'], axis=1)
#    action_6 = action_6.to_frame()
#    action_6.columns = [before_date + '_6_std']
#    actions_date = pd.concat(
#        [action_1, action_2, action_3, action_4, action_5, action_6], axis=1)
#    actions_date['user_id'] = actions_date.index
    # 分组统计，按用户分组，统计用户各项行为的转化率、均值

    actions = actions.groupby(['user_id'], as_index=False).sum()
#     days_interal = (datetime.strptime(end_date, '%Y-%m-%d') -
#                     datetime.strptime(start_date, '%Y-%m-%d')).days
    # 转化率
#     actions[before_date + '_1_ratio'] = actions[before_date +
#                                                 '_4'] / actions[before_date +
#                                                                 '_1']
#     actions[before_date + '_2_ratio'] = actions[before_date +
#                                                 '_4'] / actions[before_date +
#                                                                 '_2']
#     actions[before_date + '_3_ratio'] = actions[before_date +
#                                                 '_4'] / actions[before_date +
#                                                                 '_3']
#     actions[before_date + '_5_ratio'] = actions[before_date +
#                                                 '_4'] / actions[before_date +
#                                                                 '_5']
#     actions[before_date + '_6_ratio'] = actions[before_date +
#                                                 '_4'] / actions[before_date +
#                                                                 '_6']
    actions[before_date + '_1_ratio'] =  np.log(1 + actions[before_date + '_4.0']) - np.log(1 + actions[before_date +'_1.0'])
    actions[before_date + '_2_ratio'] =  np.log(1 + actions[before_date + '_4.0']) - np.log(1 + actions[before_date +'_2.0'])
    actions[before_date + '_3_ratio'] =  np.log(1 + actions[before_date + '_4.0']) - np.log(1 + actions[before_date +'_3.0'])
    actions[before_date + '_5_ratio'] =  np.log(1 + actions[before_date + '_4.0']) - np.log(1 + actions[before_date +'_5.0'])
    actions[before_date + '_6_ratio'] =  np.log(1 + actions[before_date + '_4.0']) - np.log(1 + actions[before_date +'_6.0'])
    # 均值
    actions[before_date + '_1_mean'] = actions[before_date + '_1.0'] / day
    actions[before_date + '_2_mean'] = actions[before_date + '_2.0'] / day
    actions[before_date + '_3_mean'] = actions[before_date + '_3.0'] / day
    actions[before_date + '_4_mean'] = actions[before_date + '_4.0'] / day
    actions[before_date + '_5_mean'] = actions[before_date + '_5.0'] / day
    actions[before_date + '_6_mean'] = actions[before_date + '_6.0'] / day
    #actions = pd.merge(actions, actions_date, how='left', on='user_id')
    #actions = actions[feature]
    return actions

# 用户近期行为特征
def get_recent_user_feat(end_date, all_actions):
    # 在上面针对用户进行累积特征提取的基础上，分别提取用户近一个月、近三天的特征，然后提取一个月内用户除去最近三天的行为占据一个月的行为的比重
    actions_3 = get_accumulate_user_feat(end_date, all_actions, 3)
    actions_30 = get_accumulate_user_feat(end_date, all_actions, 30)
    actions = pd.merge(actions_3, actions_30, how='left', on='user_id')
    del actions_3
    del actions_30

    actions['recent_action1'] = np.log(1 + actions['user_action_30_1.0'] - actions['user_action_3_1.0']) - np.log(
        1 + actions['user_action_30_1.0'])
    actions['recent_action2'] = np.log(1 + actions['user_action_30_2.0'] - actions['user_action_3_2.0']) - np.log(
        1 + actions['user_action_30_2.0'])
    actions['recent_action3'] = np.log(1 + actions['user_action_30_3.0'] - actions['user_action_3_3.0']) - np.log(
        1 + actions['user_action_30_3.0'])
    actions['recent_action4'] = np.log(1 + actions['user_action_30_4.0'] - actions['user_action_3_4.0']) - np.log(
        1 + actions['user_action_30_4.0'])
    actions['recent_action5'] = np.log(1 + actions['user_action_30_5.0'] - actions['user_action_3_5.0']) - np.log(
        1 + actions['user_action_30_5.0'])
    actions['recent_action6'] = np.log(1 + actions['user_action_30_6.0'] - actions['user_action_3_6.0']) - np.log(
        1 + actions['user_action_30_6.0'])

    #     actions['recent_action1'] = (actions['user_action_30_1']-actions['user_action_3_1'])/actions['user_action_30_1']
    #     actions['recent_action2'] = (actions['user_action_30_2']-actions['user_action_3_2'])/actions['user_action_30_2']
    #     actions['recent_action3'] = (actions['user_action_30_3']-actions['user_action_3_3'])/actions['user_action_30_3']
    #     actions['recent_action4'] = (actions['user_action_30_4']-actions['user_action_3_4'])/actions['user_action_30_4']
    #     actions['recent_action5'] = (actions['user_action_30_5']-actions['user_action_3_5'])/actions['user_action_30_5']
    #     actions['recent_action6'] = (actions['user_action_30_6']-actions['user_action_3_6'])/actions['user_action_30_6']

    return actions

# 用户对同类别下各种商品的行为
#增加了用户对不同类别的交互特征
def get_user_cate_feature(start_date, end_date, all_actions):
    # 用户对各个类别的各项行为操作统计
    # 用户对各个类别操作行为统计占对所有类别操作行为统计的比重
    actions = get_actions(start_date, end_date, all_actions)
    actions = actions[['user_id', 'cate', 'type']]
    df = pd.get_dummies(actions['type'], prefix='type')
    actions = pd.concat([actions[['user_id', 'cate']], df], axis=1)
    actions = actions.groupby(['user_id', 'cate']).sum()
    actions = actions.unstack()
    actions.columns = actions.columns.swaplevel(0, 1)
    actions.columns = actions.columns.droplevel()
    actions.columns = [
        'cate_4_type1', 'cate_5_type1', 'cate_6_type1', 'cate_7_type1',
        'cate_8_type1', 'cate_9_type1', 'cate_10_type1', 'cate_11_type1',
        'cate_4_type2', 'cate_5_type2', 'cate_6_type2', 'cate_7_type2',
        'cate_8_type2', 'cate_9_type2', 'cate_10_type2', 'cate_11_type2',
        'cate_4_type3', 'cate_5_type3', 'cate_6_type3', 'cate_7_type3',
        'cate_8_type3', 'cate_9_type3', 'cate_10_type3', 'cate_11_type3',
        'cate_4_type4', 'cate_5_type4', 'cate_6_type4', 'cate_7_type4',
        'cate_8_type4', 'cate_9_type4', 'cate_10_type4', 'cate_11_type4',
        'cate_4_type5', 'cate_5_type5', 'cate_6_type5', 'cate_7_type5',
        'cate_8_type5', 'cate_9_type5', 'cate_10_type5', 'cate_11_type5',
        'cate_4_type6', 'cate_5_type6', 'cate_6_type6', 'cate_7_type6',
        'cate_8_type6', 'cate_9_type6', 'cate_10_type6', 'cate_11_type6'
    ]
    actions = actions.fillna(0)
    actions['cate_action_sum'] = actions.sum(axis=1)
    actions['cate8_percentage'] = (
        actions['cate_8_type1'] + actions['cate_8_type2'] +
        actions['cate_8_type3'] + actions['cate_8_type4'] +
        actions['cate_8_type5'] + actions['cate_8_type6']
    ) / actions['cate_action_sum']
    actions['cate4_percentage'] = (
        actions['cate_4_type1'] + actions['cate_4_type2'] +
        actions['cate_4_type3'] + actions['cate_4_type4'] +
        actions['cate_4_type5'] + actions['cate_4_type6']
    ) / actions['cate_action_sum']
    actions['cate5_percentage'] = (
        actions['cate_5_type1'] + actions['cate_5_type2'] +
        actions['cate_5_type3'] + actions['cate_5_type4'] +
        actions['cate_5_type5'] + actions['cate_5_type6']
    ) / actions['cate_action_sum']
    actions['cate6_percentage'] = (
        actions['cate_6_type1'] + actions['cate_6_type2'] +
        actions['cate_6_type3'] + actions['cate_6_type4'] +
        actions['cate_6_type5'] + actions['cate_6_type6']
    ) / actions['cate_action_sum']
    actions['cate7_percentage'] = (
        actions['cate_7_type1'] + actions['cate_7_type2'] +
        actions['cate_7_type3'] + actions['cate_7_type4'] +
        actions['cate_7_type5'] + actions['cate_7_type6']
    ) / actions['cate_action_sum']
    actions['cate9_percentage'] = (
        actions['cate_9_type1'] + actions['cate_9_type2'] +
        actions['cate_9_type3'] + actions['cate_9_type4'] +
        actions['cate_9_type5'] + actions['cate_9_type6']
    ) / actions['cate_action_sum']
    actions['cate10_percentage'] = (
        actions['cate_10_type1'] + actions['cate_10_type2'] +
        actions['cate_10_type3'] + actions['cate_10_type4'] +
        actions['cate_10_type5'] + actions['cate_10_type6']
    ) / actions['cate_action_sum']
    actions['cate11_percentage'] = (
        actions['cate_11_type1'] + actions['cate_11_type2'] +
        actions['cate_11_type3'] + actions['cate_11_type4'] +
        actions['cate_11_type5'] + actions['cate_11_type6']
    ) / actions['cate_action_sum']

    actions['cate8_type1_percentage'] = np.log(
        1 + actions['cate_8_type1']) - np.log(
            1 + actions['cate_8_type1'] + actions['cate_4_type1'] +
            actions['cate_5_type1'] + actions['cate_6_type1'] +
            actions['cate_7_type1'] + actions['cate_9_type1'] +
            actions['cate_10_type1'] + actions['cate_11_type1'])

    actions['cate8_type2_percentage'] = np.log(
        1 + actions['cate_8_type2']) - np.log(
            1 + actions['cate_8_type2'] + actions['cate_4_type2'] +
            actions['cate_5_type2'] + actions['cate_6_type2'] +
            actions['cate_7_type2'] + actions['cate_9_type2'] +
            actions['cate_10_type2'] + actions['cate_11_type2'])
    actions['cate8_type3_percentage'] = np.log(
        1 + actions['cate_8_type3']) - np.log(
            1 + actions['cate_8_type3'] + actions['cate_4_type3'] +
            actions['cate_5_type3'] + actions['cate_6_type3'] +
            actions['cate_7_type3'] + actions['cate_9_type3'] +
            actions['cate_10_type3'] + actions['cate_11_type3'])
    actions['cate8_type4_percentage'] = np.log(
        1 + actions['cate_8_type4']) - np.log(
            1 + actions['cate_8_type4'] + actions['cate_4_type4'] +
            actions['cate_5_type4'] + actions['cate_6_type4'] +
            actions['cate_7_type4'] + actions['cate_9_type4'] +
            actions['cate_10_type4'] + actions['cate_11_type4'])
    actions['cate8_type5_percentage'] = np.log(
        1 + actions['cate_8_type5']) - np.log(
            1 + actions['cate_8_type5'] + actions['cate_4_type5'] +
            actions['cate_5_type5'] + actions['cate_6_type5'] +
            actions['cate_7_type5'] + actions['cate_9_type5'] +
            actions['cate_10_type5'] + actions['cate_11_type5'])
    actions['cate8_type6_percentage'] = np.log(
        1 + actions['cate_8_type6']) - np.log(
            1 + actions['cate_8_type6'] + actions['cate_4_type6'] +
            actions['cate_5_type6'] + actions['cate_6_type6'] +
            actions['cate_7_type6'] + actions['cate_9_type6'] +
            actions['cate_10_type6'] + actions['cate_11_type6'])
    actions['user_id'] = actions.index
    actions = actions[[
        'user_id', 'cate8_percentage', 'cate4_percentage', 'cate5_percentage',
        'cate6_percentage', 'cate7_percentage', 'cate9_percentage',
        'cate10_percentage', 'cate11_percentage', 'cate8_type1_percentage',
        'cate8_type2_percentage', 'cate8_type3_percentage',
        'cate8_type4_percentage', 'cate8_type5_percentage',
        'cate8_type6_percentage'
    ]]
    return actions

# 商品-行为
def get_accumulate_product_feat(start_date, end_date, all_actions):
    feature = [
        'sku_id', 'product_action_1', 'product_action_2',
        'product_action_3', 'product_action_4',
        'product_action_5', 'product_action_6',
        'product_action_1_ratio', 'product_action_2_ratio',
        'product_action_3_ratio', 'product_action_5_ratio',
        'product_action_6_ratio', 'product_action_1_mean',
        'product_action_2_mean', 'product_action_3_mean',
        'product_action_4_mean', 'product_action_5_mean',
        'product_action_6_mean', 'product_action_1_std',
        'product_action_2_std', 'product_action_3_std', 'product_action_4_std',
        'product_action_5_std', 'product_action_6_std'
    ]

    actions = get_actions(start_date, end_date, all_actions)
    df = pd.get_dummies(actions['type'], prefix='product_action')
    # 按照商品-日期分组，计算某个时间段该商品的各项行为的标准差
    actions['date'] = pd.to_datetime(actions['time']).apply(lambda x: x.date())
    actions = pd.concat([actions[['sku_id', 'date']], df], axis=1)
#    actions_date = actions.groupby(['sku_id', 'date']).sum()
#    actions_date = actions_date.unstack()
#    actions_date.fillna(0, inplace=True)
#    action_1 = np.std(actions_date['product_action_1'], axis=1)
#    action_1 = action_1.to_frame()
#    action_1.columns = ['product_action_1_std']
#    action_2 = np.std(actions_date['product_action_2'], axis=1)
#    action_2 = action_2.to_frame()
#    action_2.columns = ['product_action_2_std']
#    action_3 = np.std(actions_date['product_action_3'], axis=1)
#    action_3 = action_3.to_frame()
#    action_3.columns = ['product_action_3_std']
#    action_4 = np.std(actions_date['product_action_4'], axis=1)
#    action_4 = action_4.to_frame()
#    action_4.columns = ['product_action_4_std']
#   action_5 = np.std(actions_date['product_action_5'], axis=1)#   action_5 = action_5.to_frame()
#   action_5.columns = ['product_action_5_std']
#  action_6 = np.std(actions_date['product_action_6'], axis=1)
#    action_6 = action_6.to_frame()
#    action_6.columns = ['product_action_6_std']
#    actions_date = pd.concat(
#        [action_1, action_2, action_3, action_4, action_5, action_6], axis=1)
#    actions_date['sku_id'] = actions_date.index

    actions = actions.groupby(['sku_id'], as_index=False).sum()
    days_interal = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days
    # 针对商品分组，计算购买转化率
#     actions['product_action_1_ratio'] = actions['product_action_4'] / actions[
#         'product_action_1']
#     actions['product_action_2_ratio'] = actions['product_action_4'] / actions[
#         'product_action_2']
#     actions['product_action_3_ratio'] = actions['product_action_4'] / actions[
#         'product_action_3']
#     actions['product_action_5_ratio'] = actions['product_action_4'] / actions[
#         'product_action_5']
#     actions['product_action_6_ratio'] = actions['product_action_4'] / actions[
#         'product_action_6']
    actions['product_action_1_ratio'] =  np.log(1 + actions['product_action_4.0']) - np.log(1 + actions['product_action_1.0'])
    actions['product_action_2_ratio'] =  np.log(1 + actions['product_action_4.0']) - np.log(1 + actions['product_action_2.0'])
    actions['product_action_3_ratio'] =  np.log(1 + actions['product_action_4.0']) - np.log(1 + actions['product_action_3.0'])
    actions['product_action_5_ratio'] =  np.log(1 + actions['product_action_4.0']) - np.log(1 + actions['product_action_5.0'])
    actions['product_action_6_ratio'] =  np.log(1 + actions['product_action_4.0']) - np.log(1 + actions['product_action_6.0'])
    # 计算各种行为的均值
    actions['product_action_1_mean'] = actions[
        'product_action_1.0'] / days_interal
    actions['product_action_2_mean'] = actions[
        'product_action_2.0'] / days_interal
    actions['product_action_3_mean'] = actions[
        'product_action_3.0'] / days_interal
    actions['product_action_4_mean'] = actions[
        'product_action_4.0'] / days_interal
    actions['product_action_5_mean'] = actions[
        'product_action_5.0'] / days_interal
    actions['product_action_6_mean'] = actions[
        'product_action_6.0'] / days_interal
    #actions = pd.merge(actions, actions_date, how='left', on='sku_id')
    #actions = actions[feature]
    return actions

# 类别特征
def get_accumulate_cate_feat(start_date, end_date, all_actions):
    feature = ['cate', 'cate_action_1', 'cate_action_2', 'cate_action_3', 'cate_action_4', 'cate_action_5',
               'cate_action_6', 'cate_action_1_ratio', 'cate_action_2_ratio',
               'cate_action_3_ratio', 'cate_action_5_ratio', 'cate_action_6_ratio', 'cate_action_1_mean',
               'cate_action_2_mean', 'cate_action_3_mean', 'cate_action_4_mean', 'cate_action_5_mean',
               'cate_action_6_mean', 'cate_action_1_std', 'cate_action_2_std', 'cate_action_3_std',
               'cate_action_4_std', 'cate_action_5_std', 'cate_action_6_std']
    actions = get_actions(start_date, end_date, all_actions)
    actions['date'] = pd.to_datetime(actions['time']).apply(lambda x: x.date())
    df = pd.get_dummies(actions['type'], prefix='cate_action')
    actions = pd.concat([actions[['cate', 'date']], df], axis=1)
    # 按照类别-日期分组计算针对不同类别的各种行为某段时间的标准差
    #    actions_date = actions.groupby(['cate','date']).sum()
    #    actions_date = actions_date.unstack()
    #    actions_date.fillna(0, inplace=True)
    #    action_1 = np.std(actions_date['cate_action_1'], axis=1)
    #    action_1 = action_1.to_frame()
    #    action_1.columns = ['cate_action_1_std']
    #    action_2 = np.std(actions_date['cate_action_2'], axis=1)
    #    action_2 = action_2.to_frame()
    #    action_2.columns = ['cate_action_2_std']
    #    action_3 = np.std(actions_date['cate_action_3'], axis=1)
    #    action_3 = action_3.to_frame()
    #    action_3.columns = ['cate_action_3_std']
    #    action_4 = np.std(actions_date['cate_action_4'], axis=1)
    #    action_4 = action_4.to_frame()
    #   action_4.columns = ['cate_action_4_std']
    #    action_5 = np.std(actions_date['cate_action_5'], axis=1)
    #    action_5 = action_5.to_frame()
    #    action_5.columns = ['cate_action_5_std']
    #    action_6 = np.std(actions_date['cate_action_6'], axis=1)
    #    action_6 = action_6.to_frame()
    #    action_6.columns = ['cate_action_6_std']
    #    actions_date = pd.concat([action_1, action_2, action_3, action_4, action_5, action_6], axis=1)
    #    actions_date['cate'] = actions_date.index
    # 按照类别分组，统计各个商品类别下行为的转化率
    actions = actions.groupby(['cate'], as_index=False).sum()
    days_interal = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days

    #     actions['cate_action_1_ratio'] = actions['cate_action_4'] / actions['cate_action_1']
    #     actions['cate_action_2_ratio'] = actions['cate_action_4'] / actions['cate_action_2']
    #     actions['cate_action_3_ratio'] = actions['cate_action_4'] / actions['cate_action_3']
    #     actions['cate_action_5_ratio'] = actions['cate_action_4'] / actions['cate_action_5']
    #     actions['cate_action_6_ratio'] = actions['cate_action_4'] / actions['cate_action_6']
    actions['cate_action_1_ratio'] = (np.log(1 + actions['cate_action_4.0']) - np.log(1 + actions['cate_action_1.0']))
    actions['cate_action_2_ratio'] = (np.log(1 + actions['cate_action_4.0']) - np.log(1 + actions['cate_action_2.0']))
    actions['cate_action_3_ratio'] = (np.log(1 + actions['cate_action_4.0']) - np.log(1 + actions['cate_action_3.0']))
    actions['cate_action_5_ratio'] = (np.log(1 + actions['cate_action_4.0']) - np.log(1 + actions['cate_action_5.0']))
    actions['cate_action_6_ratio'] = (np.log(1 + actions['cate_action_4.0']) - np.log(1 + actions['cate_action_6.0']))
    # 按照类别分组，统计各个商品类别下行为在一段时间的均值
    actions['cate_action_1_mean'] = actions['cate_action_1.0'] / days_interal
    actions['cate_action_2_mean'] = actions['cate_action_2.0'] / days_interal
    actions['cate_action_3_mean'] = actions['cate_action_3.0'] / days_interal
    actions['cate_action_4_mean'] = actions['cate_action_4.0'] / days_interal
    actions['cate_action_5_mean'] = actions['cate_action_5.0'] / days_interal
    actions['cate_action_6_mean'] = actions['cate_action_6.0'] / days_interal
    # actions = pd.merge(actions, actions_date, how ='left',on='cate')
    # actions = actions[feature]
    return actions


def get_labels(start_date, end_date, all_actions):
    actions = get_actions(start_date, end_date, all_actions)
    #     actions = actions[actions['type'] == 4]
    # 修改为预测购买了商品8的用户预测
    actions = actions[(actions['type'] == 4) & (actions['cate'] == 8)]

    actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
    actions['label'] = 1
    actions = actions[['user_id', 'sku_id', 'label']]
    return actions


def make_actions(user, product, all_actions, train_start_date):
    train_end_date = datetime.strptime(train_start_date, '%Y-%m-%d') + timedelta(days=3)
    train_end_date = train_end_date.strftime('%Y-%m-%d')
    # 修正prod_acc,cate_acc的时间跨度
    start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=30)
    start_days = start_days.strftime('%Y-%m-%d')
    print(train_end_date)
    user_acc = get_recent_user_feat(train_end_date, all_actions)
    print('get_recent_user_feat finsihed')

    user_cate = get_user_cate_feature(train_start_date, train_end_date, all_actions)
    print('get_user_cate_feature finished')

    product_acc = get_accumulate_product_feat(start_days, train_end_date, all_actions)
    print('get_accumulate_product_feat finsihed')
    cate_acc = get_accumulate_cate_feat(start_days, train_end_date, all_actions)
    print('get_accumulate_cate_feat finsihed')
    comment_acc = get_comments_product_feat(train_end_date)
    print('get_comments_product_feat finished')
    # 标记
    test_start_date = train_end_date
    test_end_date = datetime.strptime(test_start_date, '%Y-%m-%d') + timedelta(days=5)
    test_end_date = test_end_date.strftime('%Y-%m-%d')
    labels = get_labels(test_start_date, test_end_date, all_actions)
    print("get labels")

    actions = None
    for i in (3, 5, 7, 10, 15, 21, 30):
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        if actions is None:
            actions = get_action_feat(start_days, train_end_date, all_actions, i)
        else:
            # 注意这里的拼接key
            actions = pd.merge(actions, get_action_feat(start_days, train_end_date, all_actions, i), how='left',
                               on=['user_id', 'sku_id', 'cate'])

    actions = pd.merge(actions, user, how='left', on='user_id')
    actions = pd.merge(actions, user_acc, how='left', on='user_id')
    actions = pd.merge(actions, user_cate, how='left', on='user_id')
    # 注意这里的拼接key
    actions = pd.merge(actions, product, how='left', on=['sku_id', 'cate'])
    actions = pd.merge(actions, product_acc, how='left', on='sku_id')
    actions = pd.merge(actions, cate_acc, how='left', on='cate')
    actions = pd.merge(actions, comment_acc, how='left', on='sku_id')
    actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])
    # 主要是填充拼接商品基本特征、评论特征、标签之后的空值
    actions = actions.fillna(0)
    #     return actions
    # 采样
    action_postive = actions[actions['label'] == 1]
    action_negative = actions[actions['label'] == 0]
    del actions
    neg_len = len(action_postive) * 10
    action_negative = action_negative.sample(n=neg_len)
    action_sample = pd.concat([action_postive, action_negative], ignore_index=True)

    return action_sample

def make_train_set(train_start_date, setNums ,f_path, all_actions):
    train_actions = None
    #all_actions = get_all_action()
    #print ("get all actions!")
    user = get_basic_user_feat()
    print ('get_basic_user_feat finsihed')
    product = get_basic_product_feat()
    print ('get_basic_product_feat finsihed')
    # 滑窗,构造多组训练集/验证集
    for i in range(setNums):
        print (train_start_date)
        if train_actions is None:
            train_actions = make_actions(user, product, all_actions, train_start_date)
        else:
            train_actions = pd.concat([train_actions, make_actions(user, product, all_actions, train_start_date)],
                                          ignore_index=True)
        # 接下来每次移动一天
        train_start_date = datetime.strptime(train_start_date, '%Y-%m-%d') + timedelta(days=1)
        train_start_date = train_start_date.strftime('%Y-%m-%d')
        print ("round {0}/{1} over!".format(i+1, setNums))

    train_actions.to_csv(f_path, index=False)

# 构造验证集(线下测试集)
def make_val_answer(val_start_date, val_end_date, all_actions, label_val_s1_path):
    actions = get_actions(val_start_date, val_end_date, all_actions)
    actions = actions[(actions['type'] == 4) & (actions['cate'] == 8)]
    actions = actions[['user_id', 'sku_id']]
    actions = actions.drop_duplicates()
    actions.to_csv(label_val_s1_path, index=False)


def make_val_set(train_start_date, train_end_date, val_s1_path):
    # 修改时间跨度
    start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=30)
    start_days = start_days.strftime('%Y-%m-%d')
    all_actions = get_all_action()
    print("get all actions!")
    user = get_basic_user_feat()
    print('get_basic_user_feat finsihed')

    product = get_basic_product_feat()
    print('get_basic_product_feat finsihed')
    #     user_acc = get_accumulate_user_feat(train_end_date,all_actions,30)
    #     print 'get_accumulate_user_feat finished'
    user_acc = get_recent_user_feat(train_end_date, all_actions)
    print('get_recent_user_feat finsihed')
    user_cate = get_user_cate_feature(train_start_date, train_end_date, all_actions)
    print('get_user_cate_feature finished')

    product_acc = get_accumulate_product_feat(start_days, train_end_date, all_actions)
    print('get_accumulate_product_feat finsihed')
    cate_acc = get_accumulate_cate_feat(start_days, train_end_date, all_actions)
    print('get_accumulate_cate_feat finsihed')
    comment_acc = get_comments_product_feat(train_end_date)
    print('get_comments_product_feat finished')

    actions = None
    for i in (3, 5, 7, 10, 15, 21, 30):
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        if actions is None:
            actions = get_action_feat(start_days, train_end_date, all_actions, i)
        else:
            actions = pd.merge(actions, get_action_feat(start_days, train_end_date, all_actions, i), how='left',
                               on=['user_id', 'sku_id', 'cate'])

    actions = pd.merge(actions, user, how='left', on='user_id')
    actions = pd.merge(actions, user_acc, how='left', on='user_id')
    actions = pd.merge(actions, user_cate, how='left', on='user_id')
    # 注意这里的拼接key
    actions = pd.merge(actions, product, how='left', on=['sku_id', 'cate'])
    actions = pd.merge(actions, product_acc, how='left', on='sku_id')
    actions = pd.merge(actions, cate_acc, how='left', on='cate')
    actions = pd.merge(actions, comment_acc, how='left', on='sku_id')
    actions = actions.fillna(0)

    #     print actions
    # 构造真实用户购买情况作为后续验证
    val_start_date = train_end_date
    val_end_date = datetime.strptime(val_start_date, '%Y-%m-%d') + timedelta(days=5)
    val_end_date = val_end_date.strftime('%Y-%m-%d')
    make_val_answer(val_start_date, val_end_date, all_actions, 'label_' + val_s1_path)

    actions.to_csv(val_s1_path, index=False)

# 构造测试集
def make_test_set(train_start_date, train_end_date):
    start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=30)
    start_days = start_days.strftime('%Y-%m-%d')
    all_actions = get_all_action()
    print("get all actions!")

    user = get_basic_user_feat()
    print('get_basic_user_feat finsihed')

    product = get_basic_product_feat()
    print('get_basic_product_feat finsihed')


    user_acc = get_recent_user_feat(train_end_date, all_actions)
    print('get_accumulate_user_feat finsihed')


    user_cate = get_user_cate_feature(train_start_date, train_end_date, all_actions)
    print('get_user_cate_feature finished')


    product_acc = get_accumulate_product_feat(start_days, train_end_date, all_actions)
    print('get_accumulate_product_feat finsihed')

    cate_acc = get_accumulate_cate_feat(start_days, train_end_date, all_actions)
    print('get_accumulate_cate_feat finsihed')

    comment_acc = get_comments_product_feat(train_end_date)

    actions = None
    for i in (3, 5, 7, 10, 15, 21, 30):
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        if actions is None:
            actions = get_action_feat(start_days, train_end_date, all_actions, i)
        else:
            actions = pd.merge(actions, get_action_feat(start_days, train_end_date, all_actions, i), how='left',
                               on=['user_id', 'sku_id', 'cate'])

    actions = pd.merge(actions, user, how='left', on='user_id')
    actions = pd.merge(actions, user_acc, how='left', on='user_id')
    actions = pd.merge(actions, user_cate, how='left', on='user_id')
    # 注意这里的拼接key
    actions = pd.merge(actions, product, how='left', on=['sku_id', 'cate'])
    actions = pd.merge(actions, product_acc, how='left', on='sku_id')
    actions = pd.merge(actions, cate_acc, how='left', on='cate')
    actions = pd.merge(actions, comment_acc, how='left', on='sku_id')

    actions = actions.fillna(0)

    actions.to_csv("test_set.csv", index=False)

def create_feature_map(features):
    outfile = open(r'xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


def feature_importance(bst_xgb):
    importance = bst_xgb.get_fscore(fmap=r'xgb.fmap')  # 获取特征重要性分数
    importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)  # 按照分数降序排序

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])  # 将结果转换为DataFrame
    df['fscore'] = df['fscore'] / df['fscore'].sum()  # 计算每个特征的重要性占比
    file_name = 'feature_importance_' + str(datetime.now().date())[5:] + '.csv'
    df.to_csv(file_name)  # 保存特征重要性到CSV文件

def label(column):
    if column['pred_label'] > 0.5:
        #rint ('yes')
        column['pred_label'] = 1
    else:
        column['pred_label'] = 0
    return column



if __name__ =='__main__':

    # analysis_data()
    # data_deal()
    # data_explore()
    # features_engineering()

    # '''构造训练集/测试集'''
    # # 再次获取所有用户行为数据
    # all_actions = get_all_action()
    # print("get all actions!")
    #
    # # 生成训练集
    # train_start_date = '2016-02-01'
    # make_train_set(train_start_date, 20, 'train_set.csv', all_actions)
    #
    # # 生成测试集
    # sub_start_date = '2016-04-13'
    # sub_end_date = '2016-04-16'
    # make_test_set(sub_start_date, sub_end_date)



    make_model()






