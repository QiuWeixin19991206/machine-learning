# import jieba # 中文分词器
# import pandas as pd
# import numpy as np
#
# '''文本分类 去停用词 构建文本特征 贝叶斯分类 '''
#
# data = pd.read_table('path', names=['category', 'theme', 'URL', 'content'], encoding='utf-8')
#
# content = data.content.values.tolist() #把文章转为一个list
# content_s = []
# for line in content:
#     current_segment = jieba.lcut(line) # 对每一篇进行分词
#     if len(current_segment) > 1 and current_segment != '\r\n': #换行符
#         content_s.append(current_segment)
#
# '''可视化'''
# data_content = pd.DataFrame({'content_s': content_s})
# print(data_content.head(5))
#
# # stopwords选用停词表，可自己统计然后添加
# def drop_stopwords(content, stopwords):
#     '''
#     过滤掉停用词
#     :param content:
#     :param stopwords:
#     :return:
#     '''
#     contents_clean = []
#     all_words = []
#     for line in content:
#         line_clean = []
#         for word in line:
#             if word in stopwords:
#                 continue
#             line_clean.append(word)
#             all_words.append(str(word))
#         contents_clean.append(line_clean)
#     return contents_clean, all_words
#
# content = data_content.content_s.values.tolist()
# stopwords = stopwords.stopword.values.tolist()
# contents_clean, all_words = drop_stopwords(content, stopwords)
# data_all_words = pd.DataFrame({'all_words': all_words})
#
# '''TF-IDF关键词提取'''
# import jieba.analyse
# index =2400
# content_s_str = ''.join(content_s[index])
# print(content_s_str)
# print(' '.join(jieba.analyse.extract_tags(content_s_str, topK=5, withWeight=False)))# 选出五个核心词
#
# # 制作 y/标签
# x_train = pd.DataFrame({'contents_clean': contents_clean, 'label':data['category']})
# x_train.tail()
# x_train.label.unique()
#
# label_mapping = {'汽车':1 ,'时经':2, '科技':3, '健康':4,'体育':5, '教育':6, '文化':7}
# x_train['label'] = x_train['label'].map(label_mapping) # 构建一个映射方法
# print(x_train.head(5))
#
#
#贝叶斯算法
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def nb_news():
    news = fetch_20newsgroups(subset="all")#获取数据集
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)#划分数据集
    transfer = TfidfVectorizer()#文本特征抽取
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    #贝叶斯算法
    estimator = MultinomialNB()
    estimator.fit(x_train, y_train)

    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值:\n", y_test == y_predict)
    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    return None

if __name__ == "__main__":
    nb_news()















