import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import random
import cufflinks
from plotly.offline import iplot

import plotly.graph_objs as go

def get_top_n_words(corpus,n=None):
    '''
    功能：得到出现次数最多的前n个词
    corpus：文本内容
    '''

    # 创建一个 CountVectorizer 对象
    transfer = CountVectorizer()

    # 拟合数据框 df 的 'desc' 列，生成词汇表
    # 将 'desc' 列转换为词袋模型
    bag_of_words = transfer.fit_transform(corpus)
    print(bag_of_words.toarray()) # 行：文档 列：词 不同行同一列：每个文档出现该词的情况  同行不同一列：该文档出现不同词的情况

    # 每个词出现次数
    sum_words = bag_of_words.sum(axis=0)
    print(sum_words)

    # 词：出现次数
    word_freq = [(word, sum_words[0, idx]) for word, idx in transfer.vocabulary_.items()]
    print(word_freq[:5])

    # 按出现次数排序
    words_freq = sorted(word_freq,key=lambda x:x[1],reverse=True) # key=lambda x:x[1]按第二个元素排序
    return words_freq[:n]

def get_top_n_words2(corpus,n=None):
    '''
    功能：得到出现次数最多的前n个词
    corpus：文本内容
    stop_words='english'
    '''

    # 创建一个 CountVectorizer 对象
    transfer = CountVectorizer(stop_words='english')

    # 拟合数据框 df 的 'desc' 列，生成词汇表
    # 将 'desc' 列转换为词袋模型
    bag_of_words = transfer.fit_transform(corpus)
    print(bag_of_words.toarray()) # 行：文档 列：词 不同行同一列：每个文档出现该词的情况  同行不同一列：该文档出现不同词的情况

    # 每个词出现次数
    sum_words = bag_of_words.sum(axis=0)
    print(sum_words)

    # 词：出现次数
    word_freq = [(word, sum_words[0, idx]) for word, idx in transfer.vocabulary_.items()]
    print(word_freq[:5])

    # 按出现次数排序
    words_freq = sorted(word_freq,key=lambda x:x[1],reverse=True) # key=lambda x:x[1]按第二个元素排序
    return words_freq[:n]


def get_top_n_words3(corpus,n=None):
    '''
    功能：得到出现次数最多的前n个词
    corpus：文本内容
    stop_words='english'
    ngram_range=(1,3)
    '''

    # 创建一个 CountVectorizer 对象
    transfer = CountVectorizer(stop_words='english', ngram_range=(1,3))

    # 拟合数据框 df 的 'desc' 列，生成词汇表
    # 将 'desc' 列转换为词袋模型
    bag_of_words = transfer.fit_transform(corpus)
    print(bag_of_words.toarray()) # 行：文档 列：词 不同行同一列：每个文档出现该词的情况  同行不同一列：该文档出现不同词的情况

    # 每个词出现次数
    sum_words = bag_of_words.sum(axis=0)
    print(sum_words)

    # 词：出现次数
    word_freq = [(word, sum_words[0, idx]) for word, idx in transfer.vocabulary_.items()]
    print(word_freq[:5])

    # 按出现次数排序
    words_freq = sorted(word_freq,key=lambda x:x[1],reverse=True) # key=lambda x:x[1]按第二个元素排序
    return words_freq[:n]


def clean_txt(text):
    # 将文本转换为小写
    text.lower()
    # 使用正则表达式替换非数字、小写字母、空格、#、+、_ 的字符为 ''
    text = sub_replace.sub('',text)
    # 移除停用词，并重新组合成字符串
    ' '.join(word for word in text.split() if word not in stopwords)
    return text



def recommendations(name, cosine_similarity):
    # 初始化推荐酒店列表
    recommended_hotels = []

    # 获取待推荐酒店在索引中的位置
    idx = indices[indices == name].index[0]

    # 根据待推荐酒店的相似度得分，创建一个 Series 并按降序排序
    score_series = pd.Series(cosine_similarity[idx]).sort_values(ascending=False)

    # 获取相似度最高的前 10 个酒店的索引
    top_10_indexes = list(score_series[1:11].index)

    # 遍历 top 10 索引，将推荐的酒店索引添加到推荐酒店列表中
    for i in top_10_indexes:
        recommended_hotels.append(list(df.index)[i])

    # 返回推荐酒店列表
    return recommended_hotels


if __name__ == '__main__':
    df = pd.read_csv('Seattle_Hotels.csv', encoding='latin-1') # CSV文件是用 latin-1 编码保存的
    print(df) # 名字 地址 描述 dataframe[152 rows x 3 columns]

    '''得到出现次数最多的前n个词'''
    # common_words=get_top_n_words(df['desc'],20)
    # print(common_words) # list(20)
    #
    # df1 = pd.DataFrame(common_words, columns=['desc', 'count'])
    # print(df1)
    # #          desc  count
    # # 0        the   1258
    # # 1        and   1062
    #
    # '''绘图'''
    # # df1.groupby('desc').sum()['count'].sort_values().iplot(kind='barh', yTitle='Count', linecolor='black',
    # #                                                        title='top 20 before remove stopwords')
    # fig = go.Figure(data=[go.Bar(x=df1.groupby('desc').sum()['count'].sort_values().tail(20),
    #                              y=df1.groupby('desc').sum()['count'].sort_values().tail(20).index,
    #                              orientation='h',
    #                              marker=dict(color='blue'))],
    #                 layout=go.Layout(title='去除停用词前的前 20 个词汇',
    #                                  xaxis=dict(title='词频'),
    #                                  yaxis=dict(title='词汇'),
    #                                  plot_bgcolor='rgba(0,0,0,0)',
    #                                  paper_bgcolor='rgba(0,0,0,0)',
    #                                  font=dict(color='black')))
    # fig.show()
    #
    #
    # common_words = get_top_n_words2(df['desc'], 20)
    # df2 = pd.DataFrame(common_words, columns=['desc', 'count'])
    # # df2.groupby('desc').sum()['count'].sort_values().iplot(kind='barh', yTitle='Count', linecolor='black',
    # #                                                        title='top 20 after remove stopwords')
    # fig2 = go.Figure(data=[go.Bar(x=df2.groupby('desc').sum()['count'].sort_values().tail(20),
    #                              y=df2.groupby('desc').sum()['count'].sort_values().tail(20).index,
    #                              orientation='h',
    #                              marker=dict(color='blue'))],
    #                 layout=go.Layout(title='top 20 after remove stopwords',
    #                                  xaxis=dict(title='词频'),
    #                                  yaxis=dict(title='词汇'),
    #                                  plot_bgcolor='rgba(0,0,0,0)',
    #                                  paper_bgcolor='rgba(0,0,0,0)',
    #                                  font=dict(color='black')))
    # fig2.show()
    #
    #
    # common_words=get_top_n_words3(df['desc'],20)
    # df3 = pd.DataFrame(common_words,columns=['desc','count'])
    # fig3 = go.Figure(data=[go.Bar(x=df3.groupby('desc').sum()['count'].sort_values().tail(20),
    #                              y=df3.groupby('desc').sum()['count'].sort_values().tail(20).index,
    #                              orientation='h',
    #                              marker=dict(color='blue'))],
    #                 layout=go.Layout(title='top 20 before remove stopwords-ngram_range=(2,2)',
    #                                  xaxis=dict(title='词频'),
    #                                  yaxis=dict(title='词汇'),
    #                                  plot_bgcolor='rgba(0,0,0,0)',
    #                                  paper_bgcolor='rgba(0,0,0,0)',
    #                                  font=dict(color='black')))
    # fig3.show()
    #
    #
    # '''算词数多少'''
    # df['word_count'] = df['desc'].apply(lambda x:len(str(x).split())) # split将整个文本字符串分割成单词列表
    # print(df.head())
    #
    # # df['word_count'].iplot(kind='hist',bins=50)
    # # 创建直方图数据
    # data = [go.Histogram(x=df['word_count'], nbinsx=50)]
    # # 设置布局
    # layout = go.Layout(
    #     title='单词数量直方图',
    #     xaxis=dict(title='单词数量'),
    #     yaxis=dict(title='频数'),
    #     plot_bgcolor='rgba(0,0,0,0)',
    #     paper_bgcolor='rgba(0,0,0,0)',
    #     font=dict(color='black'))
    # # 创建图形对象
    # fig4 = go.Figure(data=data, layout=layout)
    # # 在本地离线模式显示图形
    # fig4.show()


    '''文本处理'''
    # 加载 'english' 停用词列表
    stopwords = set(stopwords.words('english'))

    # 定义用于替换非数字、小写字母、空格、#、+、_ 的正则表达式模式
    sub_replace = re.compile('[^0-9a-z #+_]')

    # 应用 clean_txt 函数来清洗 'desc' 列的文本数据
    df['desc_clean'] = df['desc'].apply(clean_txt)
    print(df.head(), df['desc'][0], df['desc_clean'][0])


    '''相似度计算'''
    df.set_index('name', inplace=True)
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), stop_words='english')# 创建 TF-IDF 向量化器
    tfidf_matrix = tf.fit_transform(df['desc_clean']) # 转换文本数据为 TF-IDF 矩阵
    cosine_similarity =linear_kernel(tfidf_matrix,tfidf_matrix) # 计算余弦相似度矩阵

    recommendations('Hilton Garden Seattle Downtown',cosine_similarity)







