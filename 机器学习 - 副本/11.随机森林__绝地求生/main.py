import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import seaborn as sns # 0.11.2版本

from sklearn.model_selection import train_test_split
from IPython.display import display
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor


def print_score(m):
    res = ['mae train', metrics.mean_absolute_error(m.predict(X_train), y_train),
           'mae test', metrics.mean_absolute_error(m.predict(X_valid), y_valid)
           ]
    print(res)

def rf_feat_importance(m,df):
    a = pd.DataFrame({'cols': df.columns, 'imp': m.feature_importances_}).sort_values('imp', ascending=False)
    # print(a)
    return a

if __name__ == '__main__':
    train = pd.read_csv('train_V2.csv')
    test = pd.read_csv('test_V2.csv')
    # print(train)

    # winPlacePerc的缺失值
    # print(train[train['winPlacePerc'].isnull()])
    # 去除缺失值
    train.drop(2744604, inplace=True)

    # print(train[train['winPlacePerc'].isnull()])
    # 去除掉线情况的比赛
    # .transform('count') 的作用是对每个比赛ID（matchId）进行分组，并统计每个分组（即每场比赛）中的行数（玩家数量），然后将这个统计结果返回到新创建的 playerJoined 列中
    train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')
    # print(train['playersJoined'])
    plt.figure(figsize=(15, 10))
    sns.countplot(train['playersJoined'])


    train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')
    plt.figure(figsize=(15, 10))
    sns.countplot(train[train['playersJoined'] >= 75]['playersJoined'])


    '''由于每把人数不一样，我们归一化匹配人数为100'''
    # 这种计算可能导致了结果不符合预期，特别是在处理大数据集时，可能会出现数值溢出或失真的情况
    # train['killsNorm'] = train['kills']/train['playersJoined']*100
    # train['damageDealtNorm'] = train['damageDealt']/train['playersJoined']*100
    # train['mathDurationNorm'] = train['mathDuration'] / train['playersJoined'] * 100
    # to_show = ['id', 'kills', 'killsNorm', 'damageDealt', 'damageDealtNorm', 'mathDuration', 'mathDurationNorm']
    # print(train[to_show][:11])

    train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100+1)
    train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100+1)
    train['matchDurationNorm'] = train['matchDuration']*((100-train['playersJoined'])/100+1)
    to_show = ['Id', 'kills','killsNorm','damageDealt', 'damageDealtNorm', 'matchDuration', 'matchDurationNorm']
    # print(train[to_show][:11])

    '''剔除异常数据 开挂的'''
    train['totalDistance'] = train['rideDistance'] + train['walkDistance'] + train['swimDistance']
    train['killWithoutMoving'] = ((train['kills'] > 0) & (train['totalDistance'] == 0))
    train['headshot_rate'] = train['headshotKills'] / train['kills']
    train['headshot_rate'] = train['headshot_rate'].fillna(0)
    train.drop(train[train['killWithoutMoving'] == True].index, inplace=True)
    train.drop(train[train['roadKills'] > 10].index, inplace=True)

    plt.figure(figsize=(15, 10))
    sns.countplot(data=train, x=train['kills']).set_title('Kills')

    train.drop(train[train['kills'] > 30].index, inplace=True)

    # plt.figure(figsize=(15, 10))
    # sns.distplot(train['headshot_rate'], bins=10)

    train['matchType'].nunique()

    train = pd.get_dummies(train, columns=['matchType'])
    train.head()

    train['matchId'] = train['matchId'].astype('category')
    train['groupId'] = train['groupId'].astype('category')
    train.drop(columns=['Id'], inplace=True)

    '''单排，双排，四排'''
    solos = train[train['numGroups'] > 50]
    duos = train[(train['numGroups'] > 25) & (train['numGroups'] <= 50)]
    squads = train[train['numGroups'] <= 25]
    # print(len(solos) / len(train))
    # print(len(duos)/len(train))

    # 可视化各种排的情况
    f, ax = plt.subplots(figsize=(20, 10))
    sns.pointplot(x='kills', y='winPlacePerc', data=solos, color='black', alpha=0.8)
    sns.pointplot(x='kills', y='winPlacePerc', data=duos, color='red', alpha=0.8)
    sns.pointplot(x='kills', y='winPlacePerc', data=squads, color='blue', alpha=0.8)
    plt.text(25, 0.5, 'Solos', color='red')
    plt.grid()

    # 相关性热力图
    '''
    annot=True: 在热力图上显示每个单元格的数值
    fmt='.1f': 数值格式为浮点数，保留一位小数
    ax=ax: 指定图形绘制在 ax 轴上，即指定的子图对象
    '''
    # f, ax = plt.subplots(figsize=(15, 15))
    # sns.heatmap(train.corr(), annot=True, linewidths=0.5, fmt='.1f', ax=ax)

    '''
    k = 5: 选择与 winPlacePerc 相关性最高的前5个特征
    train.corr().nlargest(k, 'winPlacePerc')['winPlacePerc'].index找到相关性最高的前5个特征的索引。
    计算这5个特征之间的相关系数矩阵
    设置热力图的行和列的标签为特征名，使得可以清晰地查看各特征之间的相关性
    '''
    # k = 5
    # f, ax = plt.subplots(figsize=(12, 12))
    # cols = train.corr().nlargest(k, 'winPlacePerc')['winPlacePerc'].index
    # cm = np.corrcoef(train[cols].values.T)
    # sns.heatmap(cm, annot=True, linewidths=0.5, fmt='.2f', ax=ax, yticklabels=cols.values, xticklabels=cols.values)

    # plt.show()


    ################################################################################
    '''建模'''

    sample = 50000
    df_sample = train.sample(sample)
    df_sample.drop(columns=['groupId', 'matchId'], inplace=True)
    df = df_sample.drop(columns=['winPlacePerc'])# 去掉标签 得到x
    y = df_sample['winPlacePerc'] #得到y
    X_train, X_valid, y_train, y_valid = train_test_split(df, y, random_state=1)


    transfer = RandomForestRegressor(n_estimators=50, n_jobs=-1)
    transfer.fit(X_train, y_train)
    print_score(transfer)

    # # 更换数据
    # rf_feat_importance(transfer, df)[:10].plot('cols', 'imp', figsize=(14, 6), kind='barh')
    # plt.show()
    #
    # fi = rf_feat_importance(transfer, df)
    #
    # to_keep = fi[fi.imp > 0.02].cols
    # print(to_keep)




















