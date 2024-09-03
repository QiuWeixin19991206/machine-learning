import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.interpolate import UnivariateSpline
from sklearn import linear_model
import xgboost as xgb
from ultis import *

#剔除掉一些离群点
def quantile_clip(group):
    group[group < group.quantile(.05)] = group.quantile(.05)
    group[group > group.quantile(.95)] = group.quantile(.95)
    return group

def date_trend(group):
    tmp = group.groupby('date_hour').mean().reset_index()

    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    y = tmp['travel_time'].values
    nans, x = nan_helper(y)
    if group.link_ID.values[0] in ['3377906282328510514', '3377906283328510514', '4377906280784800514',
                                   '9377906281555510514']:
        tmp['date_trend'] = group['travel_time'].median()
    else:
        regr = linear_model.LinearRegression()
        regr.fit(x(~nans).reshape(-1, 1), y[~nans].reshape(-1, 1))
        tmp['date_trend'] = regr.predict(tmp.index.values.reshape(-1, 1)).ravel()
    group = pd.merge(group, tmp[['date_trend', 'date_hour']], on='date_hour', how='left')
    return group

def minute_trend(group):
    tmp = group.groupby('hour_minute').mean().reset_index()
    #s的值越小，对数据拟合越好，但是会过拟合的危险；
    spl = UnivariateSpline(tmp.index, tmp['travel_time'].values, s=0.5)
    tmp['minute_trend'] = spl(tmp.index)
    group = pd.merge(group, tmp[['minute_trend', 'hour_minute']], on='hour_minute', how='left')

    return group

def mean_time(group):
    group['link_ID_en'] = group['travel_time'].mean()
    return group

def std(group):
    group['travel_time_std'] = np.std(group['travel_time'])
    return group

def create_lagging(df, df_original, i):
    df1 = df_original.copy()
    df1['time_interval_begin'] = df1['time_interval_begin'] + pd.DateOffset(minutes=i * 2)
    df1 = df1.rename(columns={'travel_time': 'lagging' + str(i)})
    df2 = pd.merge(df, df1[['link_ID', 'time_interval_begin', 'lagging' + str(i)]],
                   on=['link_ID', 'time_interval_begin'],
                   how='left')
    return df2


if __name__ =='__main__':
    # 道路通行时间：
    df = pd.read_csv('new_gy_contest_traveltime_training_data_second.txt', delimiter=';', dtype={'link_ID': object})
    # 道路长宽情况：
    link_df = pd.read_csv('gy_contest_link_info.txt', delimiter=';', dtype={'link_ID': object})
    # 道路连接情况：
    link_tops = pd.read_csv('gy_contest_link_top.txt', delimiter=',', dtype={'link_ID': object})

    # 数据集筛选与标签转换
    # 数据集中有些数据可能由于异常情况导致不适合建模（堵车，维修等）
    # 截取开始时间
    df['time_interval_begin'] = pd.to_datetime(df['time_interval'].map(lambda x: x[1:20]))
    df = df.drop(['time_interval'], axis=1)
    df['travel_time'] = np.log1p(df['travel_time'])

    # 对每条道路，每天执行
    df['travel_time'] = df.groupby(['link_ID', 'date'])['travel_time'].transform(quantile_clip)

    # 根据需求来选择样本数据
    df = df.loc[(df['time_interval_begin'].dt.hour.isin([6, 7, 8, 13, 14, 15, 16, 17, 18]))]

    df.to_csv('data/raw_data.txt', header=True, index=None, sep=';', mode='w')

    '''缺失值预处理'''
    df = pd.read_csv('data/raw_data.txt', delimiter=';', parse_dates=['time_interval_begin'], dtype={'link_ID': object})
    link_df = pd.read_csv('gy_contest_link_info.txt', delimiter=';', dtype={'link_ID': object})
    # 构建时间序列数据，原始数据表中没有列出的数据均需要填充
    date_range = pd.date_range("2017-03-01 00:00:00", "2017-07-31 23:58:00", freq='2min')

    new_index = pd.MultiIndex.from_product([link_df['link_ID'].unique(), date_range],
                                           names=['link_ID', 'time_interval_begin'])
    new_df = pd.DataFrame(index=new_index).reset_index()

    # 合并，出现大量缺失值
    df2 = pd.merge(new_df, df, on=['link_ID', 'time_interval_begin'], how='left')

    # 筛选所需时间段数据
    df2 = df2.loc[(df2['time_interval_begin'].dt.hour.isin([6, 7, 8, 13, 14, 15, 16, 17, 18]))]
    df2 = df2.loc[~((df2['time_interval_begin'].dt.year == 2017) & (df2['time_interval_begin'].dt.month == 7) & (
        df2['time_interval_begin'].dt.hour.isin([8, 15, 18])))]
    df2 = df2.loc[~((df2['time_interval_begin'].dt.year == 2017) & (df2['time_interval_begin'].dt.month == 3) & (
            df2['time_interval_begin'].dt.day == 31))] # ~取布尔值的反

    df2['date'] = df2['time_interval_begin'].dt.strftime('%Y-%m-%d')
    # 保存中间结果
    df2.to_csv('data/pre_training.txt', header=True, index=None, sep=';', mode='w')
    # 补全时间序列
    df = pd.read_csv('data/pre_training.txt', delimiter=';', parse_dates=['time_interval_begin'], dtype={'link_ID': object})
    df['travel_time2'] = df['travel_time']

    # 多个月统计-季节性变化
    df['date_hour'] = df.time_interval_begin.map(lambda x: x.strftime('%Y-%m-%d-%H'))
    df = df.groupby('link_ID').apply(date_trend)
    df = df.drop(['date_hour', 'link_ID'], axis=1)
    df = df.reset_index()
    df = df.drop('level_1', axis=1)
    df['travel_time'] = df['travel_time'] - df['date_trend']

    # 日变化量（分钟）
    df['hour_minute'] = df.time_interval_begin.map(lambda x: x.strftime('%H-%M'))
    df = df.groupby('link_ID').apply(minute_trend)
    df = df.drop(['hour_minute', 'link_ID'], axis=1)
    df = df.reset_index()
    df = df.drop('level_1', axis=1)
    df['travel_time'] = df['travel_time'] - df['minute_trend']

    # 选择训练特征：
    link_infos = pd.read_csv('gy_contest_link_info.txt', delimiter=';', dtype={'link_ID': object})
    link_tops = pd.read_csv('gy_contest_link_top.txt', delimiter=',', dtype={'link_ID': object})
    link_infos = pd.merge(link_infos, link_tops, on=['link_ID'], how='left')  # 合并道路信息
    link_infos['links_num'] = link_infos["in_links"] + link_infos["out_links"]  # 总和
    link_infos['area'] = link_infos['length'] * link_infos['width']  # 面积
    df = pd.merge(df, link_infos[['link_ID', 'length', 'width', 'links_num', 'area']], on=['link_ID'],
                  how='left')  # 组合特征

    # 时间相关特征
    df.loc[df['date'].isin(
        ['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
         '2017-05-28', '2017-05-29', '2017-05-30']), 'vacation'] = 1

    df.loc[~df['date'].isin(
        ['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
         '2017-05-28', '2017-05-29', '2017-05-30']), 'vacation'] = 0

    df['minute'] = df['time_interval_begin'].dt.minute
    df['hour'] = df['time_interval_begin'].dt.hour
    df['day'] = df['time_interval_begin'].dt.day
    df['week_day'] = df['time_interval_begin'].map(lambda x: x.weekday() + 1)
    df['month'] = df['time_interval_begin'].dt.month

    df = df.groupby('link_ID').apply(mean_time)
    # 同行时间长的编号大
    sorted_link = np.sort(df['link_ID_en'].unique())
    df['link_ID_en'] = df['link_ID_en'].map(lambda x: np.argmin(x >= sorted_link))

    df = df.groupby('link_ID').apply(std)
    df['travel_time'] = df['travel_time'] / df['travel_time_std']

    # 缺失时间预测

    params = {
        'learning_rate': 0.2,
        'n_estimators': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'max_depth': 10,
        'min_child_weight': 1,
        'reg_alpha': 0,
        'gamma': 0
    }
    df = pd.get_dummies(df, columns=['links_num', 'width', 'minute', 'hour', 'week_day', 'day', 'month'])
    feature = df.columns.values.tolist()
    train_feature = [x for x in feature if
                     x not in ['link_ID', 'time_interval_begin', 'travel_time', 'date', 'travel_time2', 'minute_trend',
                               'travel_time_std', 'date_trend']]

    train_df = df.loc[~df['travel_time'].isnull()]
    test_df = df.loc[df['travel_time'].isnull()].copy()

    print(train_feature)

    # 训练数据切分
    X = train_df[train_feature].values
    y = train_df['travel_time'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    eval_set = [(X_test, y_test)]

    # 训练回归模型来预测缺失值
    regressor = xgb.XGBRegressor(learning_rate=params['learning_rate'], n_estimators=params['n_estimators'],
                                 booster='gbtree', objective='reg:linear', n_jobs=-1, subsample=params['subsample'],
                                 colsample_bytree=params['colsample_bytree'], random_state=0,
                                 max_depth=params['max_depth'], gamma=params['gamma'],
                                 min_child_weight=params['min_child_weight'], reg_alpha=params['reg_alpha'])
    regressor.fit(X_train, y_train, verbose=True, early_stopping_rounds=10, eval_set=eval_set)
    print(test_df[train_feature].head())
    print(test_df[train_feature].info())
    test_df['prediction'] = regressor.predict(test_df[train_feature].values)

    df = pd.merge(df, test_df[['link_ID', 'time_interval_begin', 'prediction']], on=['link_ID', 'time_interval_begin'],
                  how='left')
    feature_vis(regressor, train_feature)

    # 还原预测结果
    df['imputation1'] = df['travel_time'].isnull()
    df['travel_time'] = df['travel_time'].fillna(value=df['prediction'])
    df['travel_time'] = (df['travel_time'] * np.array(df['travel_time_std']) + np.array(df['minute_trend'])
                         + np.array(df['date_trend']))
    # 保存时间序列数据
    print(df[['travel_time', 'prediction', 'travel_time2']].describe())
    df[['link_ID', 'date', 'time_interval_begin', 'travel_time', 'imputation1']].to_csv('data/com_training.txt',
                                                                                        header=True,
                                                                                        index=None,
                                                                                        sep=';', mode='w')
    # ### 构建特征
    df = pd.read_csv('data/com_training.txt', delimiter=';', parse_dates=['time_interval_begin'],
                     dtype={'link_ID': object})
    df1 = df.copy()
    # 平移5格
    df1['time_interval_begin'] = df1['time_interval_begin'] + pd.DateOffset(minutes=5 * 2)

    df1 = df1.rename(columns={'travel_time': 'lagging' + str(5)})

    # 合并数据集
    df2 = pd.merge(df, df1[['link_ID', 'time_interval_begin', 'lagging' + str(5)]],
                   on=['link_ID', 'time_interval_begin'], how='left')

    df1 = create_lagging(df, df, 1)

    # 构建时间序列特征
    lagging = 5
    for i in range(2, lagging + 1):
        df1 = create_lagging(df1, df, i)

    link_infos = pd.read_csv('gy_contest_link_info.txt', delimiter=';', dtype={'link_ID': object})
    link_tops = pd.read_csv('gy_contest_link_top.txt', delimiter=',', dtype={'link_ID': object})

    link_tops = link_tops.fillna(0)
    link_infos = pd.merge(link_infos, link_tops, on=['link_ID'], how='left')
    link_infos['links_num'] = link_infos["in_links"] + link_infos["out_links"]

    link_infos['area'] = link_infos['length'] * link_infos['width']
    df2 = pd.merge(df1, link_infos[['link_ID', 'length', 'width', 'links_num', 'area']], on=['link_ID'], how='left')

    # 假期特征
    df2.loc[df2['date'].isin(
        ['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
         '2017-05-28', '2017-05-29', '2017-05-30']), 'vacation'] = 1
    df2.loc[~df2['date'].isin(
        ['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
         '2017-05-28', '2017-05-29', '2017-05-30']), 'vacation'] = 0

    # 起始分钟特征
    df2.loc[df2['time_interval_begin'].dt.hour.isin([6, 7, 8]), 'minute_series'] = \
        df2['time_interval_begin'].dt.minute + (df2['time_interval_begin'].dt.hour - 6) * 60

    df2.loc[df2['time_interval_begin'].dt.hour.isin([13, 14, 15]), 'minute_series'] = \
        df2['time_interval_begin'].dt.minute + (df2['time_interval_begin'].dt.hour - 13) * 60

    df2.loc[df2['time_interval_begin'].dt.hour.isin([16, 17, 18]), 'minute_series'] = \
        df2['time_interval_begin'].dt.minute + (df2['time_interval_begin'].dt.hour - 16) * 60

    # 星期特征
    df2['day_of_week'] = df2['time_interval_begin'].map(lambda x: x.weekday() + 1)
    df2.loc[df2['day_of_week'].isin([1, 2, 3]), 'day_of_week_en'] = 1
    df2.loc[df2['day_of_week'].isin([4, 5]), 'day_of_week_en'] = 2
    df2.loc[df2['day_of_week'].isin([6, 7]), 'day_of_week_en'] = 3

    # 时间段特征
    df2.loc[df['time_interval_begin'].dt.hour.isin([6, 7, 8]), 'hour_en'] = 1
    df2.loc[df['time_interval_begin'].dt.hour.isin([13, 14, 15]), 'hour_en'] = 2
    df2.loc[df['time_interval_begin'].dt.hour.isin([16, 17, 18]), 'hour_en'] = 3

    # 星期，时间段合并特征
    df2['week_hour'] = df2["day_of_week_en"].astype('str') + "," + df2["hour_en"].astype('str')

    df2 = pd.get_dummies(df2, columns=['week_hour', 'links_num', 'width'])

    df2 = df2.groupby('link_ID').apply(mean_time)

    sorted_link = np.sort(df2['link_ID_en'].unique())

    df2['link_ID_en'] = df2['link_ID_en'].map(lambda x: np.argmin(x >= sorted_link))

    # 保存特征结果
    df2.to_csv('data/com_training.txt', header=True, index=None, sep=';', mode='w')


