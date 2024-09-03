import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
# # 修改前
# from sklearn.preprocessing import Imputer
# 修改后
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb
def make_data():
    ### object类型处理
    print(app_train.dtypes.value_counts())
    app_train.select_dtypes('object').apply(pd.Series.nunique, axis=0) # pd.Series.nunique计算唯一值的数量，排除缺失值


    le = LabelEncoder()
    for col in app_train:
        if app_train[col].dtype == 'object':
            if len(list(app_train[col].unique())) <= 2:
                le.fit(app_train[col])
                app_train[col] = le.transform(app_train[col])

    app_train2 = pd.get_dummies(app_train) # 307511, 243


    '''可视化分析'''
    # print((app_train2['DAYS_BIRTH']/-365).describe())
    # print((app_train2['DAYS_EMPLOYED']).describe())
    #
    # app_train2['DAYS_EMPLOYED'].plot.hist()
    # plt.show()
    #
    # correlations = app_train2.corr()['TARGET'].sort_values()
    # print(correlations)
    # print(correlations.tail())
    #
    # app_train2['DAYS_BIRTH'] = abs(app_train2['DAYS_BIRTH'])
    # print(app_train2['TARGET'].corr(app_train2['DAYS_BIRTH']))
    #
    # plt.style.use('fivethirtyeight')
    # plt.hist(app_train2['DAYS_BIRTH'] / 365, edgecolor='k', bins=25)
    # plt.show()
    #
    # plt.figure(figsize=(10, 8))
    # # KDEPLOT
    # sns.kdeplot(app_train2.loc[app_train2['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label='target==0')
    # sns.kdeplot(app_train2.loc[app_train2['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label='target==1')
    # plt.show()
    #
    # age_data = app_train2[['TARGET', 'DAYS_BIRTH']]
    # age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365
    #
    # age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins=np.linspace(20, 70, num=11))
    # print(age_data.head())
    # age_groups = age_data.groupby('YEARS_BINNED').mean()
    # print(age_groups)
    #
    # plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])
    # plt.xticks(rotation=75)
    # plt.show()
    #
    # ext_data = app_train2[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
    # ext_data_corrs = ext_data.corr()
    # print(ext_data_corrs)
    #
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(ext_data_corrs, cmap=plt.cm.RdYlBu_r, annot=True)
    # plt.show()
    #
    # plt.figure(figsize=(10, 10))
    # for i, source in enumerate(['EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1']):
    #     # 指定好子图的位置
    #     plt.subplot(3, 1, i + 1)
    #     # kdeplot
    #     sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, source] / 365, label='target==0')
    #     sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, source] / 365, label='target==1')
    #     plt.title('D of %s' % source)
    # plt.tight_layout(h_pad=2.5)
    # plt.show()

    '''特征工程'''
    poly_features = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
    # 缺失值填充
    imputer = SimpleImputer(strategy='median')
    poly_target = poly_features['TARGET']
    poly_features.drop(columns=['TARGET'], inplace=True)
    poly_features = imputer.fit_transform(poly_features)

    # 拟合并转换特征数据，生成多项式特征
    poly_transformer = PolynomialFeatures(degree=3)
    poly_transformer.fit(poly_features)
    poly_features = poly_transformer.transform(poly_features)
    print(poly_features.shape)

    # 打印生成的多项式特征的前 20 个特征名称
    print(poly_transformer.get_feature_names_out(input_features=['TARGET','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH'])[:20])

    # 将当前得到的部分特征跟总体组合在一起
    poly_features = pd.DataFrame(poly_features, columns=poly_transformer.get_feature_names_out(
        input_features=['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))
    print(poly_features.head())

    poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']
    app_train_poly = app_train.merge(poly_features, on='SK_ID_CURR', how='left')
    print(app_train_poly.head())


    # 根据实际情况来创建特征
    app_train_domain = app_train.copy()

    app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']
    app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']
    app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']
    app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']

    plt.figure(figsize=(12, 20))
    for i, feature in enumerate(
            ['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT']):
        plt.subplot(4, 1, i + 1)
        sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 0, feature], label='target == 0')
        sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 1, feature], label='target == 1')

        plt.title('Distribution of %s by Target Value' % feature)
        plt.xlabel('%s' % feature);
        plt.ylabel('Density');

    plt.tight_layout(h_pad=2.5)
    plt.show()

    # 数据预处理

    label = app_train['TARGET']
    train = app_train.drop(columns=['TARGET'])
    train, test, y_train, y_test = train_test_split(train, label, test_size=0.2, random_state=0)
    features = list(train.columns)

    imputer = SimpleImputer(strategy='median')
    std = StandardScaler()
    # 填充
    imputer.fit(train)
    train = imputer.transform(train)
    test = imputer.transform(test)
    # 标准化
    std.fit(train)
    train = std.transform(train)
    test = std.transform(test)

    print()
    return train, test, y_train, y_test

def missing_value_table(df):
    #计算所有的缺失值
    mis_val = df.isnull().sum()

    mis_val_percent = 100*df.isnull().sum()/len(df)
    #合并
    mis_val_table = pd.concat([mis_val,mis_val_percent],axis=1)
    mis_val_rename = mis_val_table.rename(columns = {0:'Missing valyes',1:'% of total values'})
    #剔除完整的并排序
    mis_val_rename = mis_val_rename[mis_val_rename.iloc[:,1]!=0].sort_values('% of total values',ascending=False)
    return mis_val_rename

def logisticRegression(train, y_train):
    log_reg = LogisticRegression(C=0.0001)
    log_reg.fit(train, y_train)
    predictions = log_reg.predict_proba(test)[:, 1]
    test_auc = roc_auc_score(y_test, predictions)
    print(test_auc)

def randomForestClassifier(train,y_train):

    random_forest = RandomForestClassifier(n_estimators=1000, random_state=10, n_jobs=-1)
    random_forest.fit(train, y_train)
    predictions = random_forest.predict_proba(test)[:, 1]
    test_auc = roc_auc_score(y_test, predictions)

def LightGBM(train, test, y_train, y_test):
    model = lgb.LGBMClassifier(n_estimators=10000, objective='binary',
                               class_weight='balanced', learning_rate=0.05,
                               reg_alpha=0.1, reg_lambda=0.1,
                               subsample=0.8, n_jobs=-1, random_state=50)

    model.fit(train, y_train, eval_metric='auc',
              eval_set=[(test, y_test), (train, y_train)],
              eval_names=['test', 'train'],
              early_stopping_rounds=100, verbose=200)

if __name__ == '__main__':
    app_train = pd.read_csv('application_train.csv')
    train, test, y_train, y_test = make_data()

    # 基础模型：逻辑回归
    logisticRegression(train, y_train)

    # 对比：随机森林
    randomForestClassifier(train,y_train)

    # LightGBM
    LightGBM(train, test, y_train, y_test)
