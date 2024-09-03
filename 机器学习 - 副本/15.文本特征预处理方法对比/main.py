import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
# 设置 Matplotlib 图表的默认参数
mpl.rc('axes',labelsize=14)
mpl.rc('xtick',labelsize=14)
mpl.rc('ytick',labelsize=14)
np.random.seed(100)
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# 不同方式的预处理
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def data_describe():

    print(housing.head())
    print(housing['ocean_proximity'].value_counts())  # 统计靠海的情况（字符）
    print(housing.describe())

    housing.hist(bins=50, figsize=(10, 10)) # bins步长
    plt.show()

    train_set, test_set, = train_test_split(housing, test_size=0.2, random_state=100)

    '''连续值离散化'''
    housing['income_cat'] = pd.cut(housing['median_income'], bins=[0, 1.5, 3.0, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5])
    print(housing['income_cat'].value_counts()) # 统计个数

    housing['income_cat'].hist()
    plt.show()

    '''位置情况展示分析'''
    housing2 = train_set.copy()

    housing2.plot(kind='scatter', x='longitude', y='latitude') # 散点图
    plt.show()

    housing2.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)# 透视度 alpha=0.1
    plt.show()

    housing2.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
                 s=housing2['population'] / 100, # 设置散点的大小
                 label='population', figsize=(10, 8),
                 c='median_house_value',  # 根据 'median_house_value' 列的值对散点进行着色。
                 cmap=plt.get_cmap('jet'), # 选择着色的颜色映射，这里使用 'jet' 颜色映射。
                 colorbar=True) # 关闭原本的图例
    plt.legend()
    plt.show()

    '''把坐标位置加载到地图中'''
    california_img = mpimg.imread('california.png')
    ax = housing2.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
                      s=housing2['population'] / 100, label='population', figsize=(10, 8),
                      c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=False)
    # 设置一下坐标的选择区域
    plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05] # 设置背景图片的显示范围，对应经度和纬度的范围。
               , cmap=plt.get_cmap('jet'))
    plt.ylabel('latitude')
    plt.xlabel('longitude')
    price = housing2['median_house_value']
    tick_values = np.linspace(price.min(), price.max(), 11)
    cbar = plt.colorbar()
    cbar.ax.set_yticklabels(['$%dk' % (round(v / 1000)) for v in tick_values])
    cbar.set_label('median house value')
    plt.legend()
    plt.show()

    print(housing2.describe())

    '''特征相关性'''
    housing2.drop('ocean_proximity', axis=1, inplace=True)
    corr_matrix = housing2.corr()
    print(corr_matrix)
    print(corr_matrix['median_house_value'].sort_values(ascending = False))

    # 像圆的没有相关性，越解决y=x 的有线性关系
    atttibutes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
    scatter_matrix(housing[atttibutes], figsize=(10, 10))
    plt.show()

    return None

def data_make():
    housing['income_cat'] = pd.cut(housing['median_income'], bins=[0, 1.5, 3.0, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5]).astype(float)
    print(housing.info())
    train_set, test_set, = train_test_split(housing, test_size=0.2, random_state=100)
    # 特征构建
    housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
    housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
    housing['population_per_household'] = housing['population'] / housing['households']
    # 相关性
    housing_cat = housing[['ocean_proximity']]
    print('housing_cat', housing_cat)
    housing.drop('ocean_proximity', axis=1, inplace=True)
    corr_matrix = housing.corr()
    corr_matrix['median_house_value'].sort_values(ascending=False)

    housing3 = train_set.drop('median_house_value', axis=1)
    housing_label = train_set['median_house_value'].copy()
    print(housing3.shape)

    '''缺失值处理'''
    sampe_incomplete_rows = housing[housing.isnull().any(axis=1)]# 找出包含缺失值的行
    median = housing['total_bedrooms'].median()# 计算中位数
    sampe_incomplete_rows['total_bedrooms'].fillna(median, inplace=True)# 用中位数填充缺失值，推荐的方式

    imputer = SimpleImputer(strategy='median')# 创建 SimpleImputer 对象，并指定填充策略为中位数
    housing_num = housing
    imputer.fit(housing_num)# 计算中位数并填充缺失值
    print(imputer.statistics_ ) # 输出 imputer 计算的各列中位数
    print(housing_num.median().values) # 输出数据框计算的各列中位数

    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index.values)
    print(housing_tr.head())
    print(housing_tr.loc[sampe_incomplete_rows.index.values].head())

    '''字符特征处理 housing_cat = housing[['ocean_proximity']]'''
    '''1'''
    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    print(housing_cat_encoded[:10])
    '''2'''
    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    print(housing_cat_1hot[:10].toarray())

    # 创建 FunctionTransformer 对象，应用 add_extra_features 函数
    attr_adder = FunctionTransformer(add_extra_features, validate=False,
                                     kw_args={"add_bedrooms_per_room": False})
    # 使用 FunctionTransformer 对象处理房屋数据，添加额外特征
    housing_extra_attribs = attr_adder.fit_transform(housing.values)

    # 将处理后的数据转换为 DataFrame，并添加新特征的列名
    housing_extra_attribs = pd.DataFrame(housing_extra_attribs,
                                         columns=list(housing.columns) + ['rooms_per_household',
                                                                          'population_per_household'])
    # 打印处理后的数据的前几行
    print(housing_extra_attribs.head())


    return housing_num, housing_label

def add_extra_features(X, add_bedrooms_per_room=True):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household,
                     bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]

if __name__ =='__main__':
    housing = pd.read_csv('housing.csv')
    rooms_ix, bedrooms_ix, population_ix, household_ix = (3, 4, 5, 6)
    # data_describe()
    housing_num, housing_label = data_make()
    combined = pd.concat([housing_num, housing_label], axis=1)
    combined_cleaned = combined.dropna()

    # 分离处理后的 x 和 y
    housing_num = combined_cleaned.iloc[:, :8]  # 取除最后一列外的所有列
    housing_label = combined_cleaned.iloc[:, -1]  # 取最后一列作为 y

    # 创建数值特征处理流水线
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # 使用中位数填充缺失值
        ('attr_adder', FunctionTransformer(add_extra_features, validate=False)),  # 添加额外特征
        ('std_scaler', StandardScaler()),  # 标准化特征
    ])

    # 应用数值特征处理流水线到房屋数值特征数据上
    housing_num_tr = num_pipeline.fit_transform(housing_num)
    print(housing_num_tr[:5])

    # 获取数值特征列名和类别特征列名
    num_attribs = list(housing_num)  # 获取数值特征列名
    cat_attribs = ['ocean_proximity']  # 定义类别特征列名

    # 创建完整的数据预处理流水线，包括数值特征处理和类别特征处理
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),  # 数值特征流水线应用到数值特征列
        ('cat', OneHotEncoder(), cat_attribs),  # 类别特征编码处理
    ])

    # 对整个房屋数据集进行完整的数据预处理
    housing_prepared = full_pipeline.fit_transform(housing)
    print(housing_prepared[0])

    # 选择线性回归模型作为算法模型
    lin_reg = LinearRegression()

    # 使用预处理后的数据进行模型训练
    lin_reg.fit(housing_prepared, housing_label)

    # 准备输入数据进行预测
    input_data = housing[:5]
    input_label = housing_label[:5]
    input_data_prepared = full_pipeline.transform(input_data)

    # 使用训练好的线性回归模型进行预测
    predictions = lin_reg.predict(input_data_prepared)
    print(predictions)

    # 打印实际标签值
    print(input_label.values)

    # 在整个训练集上进行预测，并计算均方误差（MSE）
    housing_predictions = lin_reg.predict(housing_prepared)
    mse = mean_squared_error(housing_label, housing_predictions)
    rmse = np.sqrt(mse)
    print("Root Mean Squared Error:", rmse)






