
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''数据分析'''
# calendar = pd.read_csv('calendar.csv.gz')
# print(calendar) # listing_id        date        available      price
# print(calendar.date.min(),calendar.date.max())
# print(calendar.isnull().sum())
# calendar = calendar.dropna() # 去除NAN
# print(calendar.shape)
#
# # 如果data不是datatime格式 转换为标准格式
# calendar['date'] = pd.to_datetime(calendar['date'])
#
# # object ==> float
# print(calendar.info())
# calendar['price'] = calendar['price'].str.replace('$', '') #去掉字符
# calendar['price'] = calendar['price'].str.replace(',','')
# print(calendar.head())
# print(calendar.info())
# calendar['price'] = calendar['price'].astype(float)
# print(calendar.info())
#
#
# # 提取时间中你需要的指标 月份
# mean_of_mouth = calendar.groupby(calendar['date'].dt.strftime('%B'))['price'].mean() # %B月份
# mean_of_mouth.plot(kind='barh', figsize=(12, 8))
# plt.show()
#
# # 改变排序
# cats = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
# mean_of_mouth = mean_of_mouth.reindex(cats)
# mean_of_mouth.plot(kind='barh', figsize=(12, 8))
# plt.show()
#
# # 提取时间中你需要的指标 星期
# calendar['dayofweek'] = calendar.date.dt.day_name()
# print(calendar.head())
#
# cats = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
# price_week = calendar.groupby(['dayofweek']).mean().reindex(cats)
# price_week.drop('listing_id',axis=1,inplace = True)
# price_week.plot()
#
# calendar['dayofweek'] = calendar.date.dt.weekday_name
# cats = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
# price_week = calendar[['dayofweek','price']]
# price_week = calendar.groupby(['dayofweek']).mean().reindex(cats)
# price_week.drop('listing_id',axis=1,inplace = True)
# price_week.plot()
# ticks = list(range(0,7,1))
# labels = 'Mon Tues Weds Thurs Fri Sat Sun'.split(' ')
# plt.xticks(ticks,labels)



# ### 房源信息
# listings = pd.read_csv('listings.csv.gz')
# listings.head()
# plt.figure(figsize=(12,6))
# sns.distplot(listings.review_scores_rating.dropna())
# sns.despine()
# plt.show()
# listings.review_scores_rating.describe()
#
# listings['price'] = listings['price'].str.replace('$','')
# listings['price'] = listings['price'].str.replace(',','')
# listings['price'] = listings['price'].astype(float)
#
# print(listings['price'].describe())
#
# high_price = listings.sort_values('price',ascending=False)
# print('high_price.head(10)')
#
# # 统计图
# listings.loc[(listings['price']<=600) & (listings['price']>0)].price.hist(bins=200)
# # 分析图
# f,ax = plt.subplots(figsize=(14,10))
# sns.boxplot(y='price',x='room_type',data=listings.loc[(listings['price']<=600) & (listings['price']>0)])



# 前10个最常见的设施
# print(listings.amenities[:5])
# listings.amenities = listings.amenities.str.replace('[{}]','').str.replace('"','')
# print(listings.amenities[:5])
# listings['amenities'].map(lambda amns:amns.split(','))[:5]
# np.concatenate(listings['amenities'].map(lambda amns:amns.split(',')))
#
# pd.Series(np.concatenate(listings['amenities'].map(lambda amns:amns.split(',')))).value_counts().head(10)
#
# f,ax = plt.subplots(figsize=(14,10))
# pd.Series(np.concatenate(listings['amenities'].map(lambda amns:amns.split(',')))).value_counts().head(10).plot(kind='bar')
# plt.show()


'''数据预处理'''
# #数值特征列
# listings = pd.read_csv('listings.csv.gz')
# listings['price'] = listings['price'].str.replace('$','')
# listings['price'] = listings['price'].str.replace(',','')
# listings['price'] = listings['price'].astype(float)
#
# print(listings.head())
#
# col = ['host_listings_count', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', 'number_of_reviews', 'review_scores_rating', 'reviews_per_month']
# print(listings.loc[(listings['price'] <= 600) & (listings['price'] > 0 )].groupby(['bathrooms', 'bedrooms']).count()['price'].reset_index())
# print(listings.loc[(listings['price']<=600) & (listings['price']>0)].groupby(['bathrooms','bedrooms']).count()['price'].reset_index().pivot(index='bathrooms', columns='bedrooms', values='price'))
#
# # cur_input = listings.loc[(listings['price']<=600) & (listings['price']>0)].groupby(['bathrooms','bedrooms']).count()['price'].reset_index().pivot(index='bathrooms', columns='bedrooms', values='price')
# # f,ax = plt.subplots(figsize=(14,10))
# # sns.heatmap(cur_input,cmap='Oranges',annot=True,linewidths=0.5,fmt='.0f')
# # plt.show()
#
#
# cur_input = listings.loc[(listings['price'] <= 600) & (listings['price'] > 0)].groupby(['bathrooms', 'bedrooms']).mean(numeric_only=True)['price'].reset_index().pivot_table(index='bathrooms', columns='bedrooms', values='price')
# f,ax = plt.subplots(figsize=(14,10))
# sns.heatmap(cur_input,cmap='Oranges',annot=True,linewidths=0.5,fmt='.0f')
# plt.show()


'''建模'''
listings = pd.read_csv('listings.csv.gz')
listings['price'] = listings['price'].str.replace('$','')
listings['price'] = listings['price'].str.replace(',','')
listings['price'] = listings['price'].astype(float)
listings = listings.loc[(listings.price <= 600) & (listings.price > 0)]

# 去除 listings.amenities 中的大括号 {} 和双引号
listings.amenities = listings.amenities.str.replace("[{}]", "").str.replace('"', "")
columns =  ['host_is_superhost', 'host_identity_verified', 'host_has_profile_pic',
                   'is_location_exact', 'requires_license', 'instant_bookable',
                   'require_guest_profile_picture', 'require_guest_phone_verification']
for c in columns: # 把 f t 转为 0 1
    listings[c] = listings[c].replace('f',0,regex=True)
    listings[c] = listings[c].replace('t',1,regex=True)

# 将'security_deposit'列中的缺失值填充为0
listings['security_deposit'] = listings['security_deposit'].fillna(value=0)

# 去掉'security_deposit'列中包含的美元符号（$）和右括号（），并将其转换为浮点数类型
listings['security_deposit'] = listings['security_deposit'].replace('[\$,)]', '', regex=True).astype(float)

# 将'cleaning_fee'列中的缺失值填充为0
listings['cleaning_fee'] = listings['cleaning_fee'].fillna(value=0)

# 去掉'cleaning_fee'列中包含的美元符号（$）和右括号（），并将其转换为浮点数类型
listings['cleaning_fee'] = listings['cleaning_fee'].replace('[\$,)]', '', regex=True).astype(float)

listings_new = listings[['host_is_superhost', 'host_identity_verified', 'host_has_profile_pic','is_location_exact',
                         'requires_license', 'instant_bookable', 'require_guest_profile_picture',
                         'require_guest_phone_verification', 'security_deposit', 'cleaning_fee',
                         'host_listings_count', 'host_total_listings_count', 'minimum_nights',
                     'bathrooms', 'bedrooms', 'guests_included', 'number_of_reviews','review_scores_rating', 'price']]

# 检查listings_new中的每一列，打印出包含缺失值的列名
for col in listings_new.columns[listings_new.isnull().any()]:
    print(col)

# 填充包含缺失值的列，使用该列的中位数
for col in listings_new.columns[listings_new.isnull().any()]:
    listings_new[col] = listings_new[col].fillna(listings_new[col].median())

# 对指定的分类特征进行独热编码，并将结果连接到listings_new中
for cat_feature in ['zipcode', 'property_type', 'room_type', 'cancellation_policy', 'neighbourhood_cleansed', 'bed_type']:
    listings_new = pd.concat([listings_new, pd.get_dummies(listings[cat_feature])], axis=1)

# listings_new = pd.concat([listings_new, df_amenities], axis=1, join='inner')
print(listings_new.head())

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
y = listings_new['price']
x = listings_new.drop('price',axis=1)
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=1)

transfer = StandardScaler()
X_train = transfer.fit_transform(X_train)
X_test = transfer.transform(X_test)

estimator = RandomForestRegressor(n_estimators=500,n_jobs=-1)
estimator.fit(X_train,y_train)
y_train_predict = estimator.predict(X_train)
y_test_predict = estimator.predict(X_test)
rmse_rf = mean_squared_error(y_test,y_test_predict)**(1/2)
print(rmse_rf)
print(r2_score(y_test,y_test_predict))


'''LightGBM建模'''
from lightgbm import LGBMRegressor

fit_params ={'early_stopping_rounds':10,
            'eval_metric':'rmse',
            'eval_set':[(X_test,y_test)],
            'eval_names':['valid'],
            'verbose':100}

estimator = LGBMRegressor(max_depth=20,learning_rate =0.01,n_estimators=1000)
estimator.fit(X_train,y_train,**fit_params)
y_pred = estimator.predict(X_test)
print(r2_score(y_test,y_pred))





