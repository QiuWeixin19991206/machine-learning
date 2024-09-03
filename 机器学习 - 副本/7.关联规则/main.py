

'''
支持度 表示项集 AA 出现的频率越高，说明这个项集在数据集中出现的普遍程度越高
置信度 指的是一个规则 A→BA→B 成功应用的概率 a发生的情况下，b也发生的概率 Pab/Pa
提升度 置信度a=>b/支持度b

'''

# import pandas as pd
# from mlxtend.frequent_patterns import apriori
# from mlxtend.frequent_patterns import association_rules
#
# data = {'ID':[1,2,3,4,5,6],
#        'Onion':[1,0,0,1,1,1],
#        'Potato':[1,1,0,1,1,1],
#        'Burger':[1,1,0,0,1,1],
#        'Milk':[0,1,1,1,0,1],
#        'Beer':[0,0,1,0,1,0]}
#
# df = pd.DataFrame(data)
# df = df[['ID', 'Onion', 'Potato', 'Burger', 'Milk', 'Beer' ]].astype(bool) #选择特定列的顺序
#
# print(df)
#
# # ### 设置支持度 *(support)* 来选择频繁项集.
# # - 选择最小支持度为50%
# # - `apriori(df, min_support=0.5, use_colnames=True)`
# frequent_itemsets = apriori(df[['Onion', 'Potato', 'Burger', 'Milk', 'Beer' ]], min_support=0.50, use_colnames=True)
#
# # 计算规则
# # association_rules(df, metric='lift', min_threshold=1)
# # 可以指定不同的衡量标准与最小阈值
#
# rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
#
# print(rules)
#
# print(':')
# print(rules [ (rules['lift'] >1.125)  & (rules['confidence']> 0.8)  ].to_string())# 打印完整.to_string()


'''实际处理数据'''
# import pandas as pd
# from mlxtend.frequent_patterns import apriori
# from mlxtend.frequent_patterns import association_rules
# retail_shopping_basket = {'ID':[1,2,3,4,5,6],
#                          'Basket':[['Beer', 'Diaper', 'Pretzels', 'Chips', 'Aspirin'],
#                                    ['Diaper', 'Beer', 'Chips', 'Lotion', 'Juice', 'BabyFood', 'Milk'],
#                                    ['Soda', 'Chips', 'Milk'],
#                                    ['Soup', 'Beer', 'Diaper', 'Milk', 'IceCream'],
#                                    ['Soda', 'Coffee', 'Milk', 'Bread'],
#                                    ['Beer', 'Chips']
#                                   ]
#                          }
# # 创建DataFrame
# retail = pd.DataFrame(retail_shopping_basket)
# print(retail)
#
# # 提取ID列和处理Basket列
# retail = retail[['ID', 'Basket']] # 提取
# pd.options.display.max_colwidth=100
#
# retail_id = retail[['ID']]
# retail_Basket = retail.Basket.str.join(',') # 将列表转换为逗号分隔的字符串
# retail_Basket = retail_Basket.str.get_dummies(',') # 使用get_dummies将每个物品转换为二进制变量
# # 合并处理后的数据
# retail = retail_id.join(retail_Basket.astype(bool)) # 将处理后的Basket数据与ID列合并
# print(retail.to_string())
# # 使用Apriori算法生成频繁项集
# frequent_itemsets_2 = apriori(retail.drop(['ID'], axis=1), use_colnames=True)
# print(frequent_itemsets_2.to_string())
# print(association_rules(frequent_itemsets_2, metric='lift').to_string())






'''电影题材关联'''
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

movies = pd.read_csv('movies.csv')
print(movies.head().to_string())
#
# a = movies.drop('genres', axis=1) # 删除 'genres' 列
# b = movies['genres'].str.get_dummies() # 对 'genres' 列进行独热编码
# c = a.join(b) # 将独热编码后的列与原数据框（删除 'genres' 列后）合并
movies_ohe = movies.drop('genres',axis=1).join(movies.genres.str.get_dummies())

# 设置显示的最大列数为100
pd.options.display.max_columns = 100
# print(movies_ohe.head(), movies_ohe.shape) #  数据集包括9125部电影，一共有20种不同类型

movies_ohe.set_index(['movieId','title'],inplace=True) # 将 'movieId' 和 'title' 列设置为索引
print(movies_ohe.head())

frequent_itemsets_movies = apriori(movies_ohe,use_colnames=True, min_support=0.025) # 计算支持度
print(frequent_itemsets_movies.head())

rules_movies =  association_rules(frequent_itemsets_movies, metric='lift', min_threshold=1.25)
print(rules_movies.head())

print(rules_movies[(rules_movies.lift>4)].sort_values(by=['lift'], ascending=False))


