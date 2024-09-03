import jieba
import jieba.analyse


seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("全模式: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("精确模式: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print(", ".join(seg_list))




'''添加自定义词典'''
jieba.load_userdict("./data/mydict.txt") #需UTF-8，可以在另存为里面设置

#也可以用jieba.add_word("乾清宫")

text = "故宫的著名景点包括乾清宫、太和殿和黄琉璃瓦等,炉石传说国服即将回归"

# 全模式
seg_list = jieba.cut(text, cut_all=True)
print(u"[全模式]: ", "/ ".join(seg_list))

# 精确模式
seg_list = jieba.cut(text, cut_all=False)
print(u"[精确模式]: ", "/ ".join(seg_list))


'''关键词抽取'''
seg_list = jieba.cut(text, cut_all=False)
print (u"分词结果:")
print ("/".join(seg_list))

# 获取关键词，提取文本中前 5 个关键词
tags = jieba.analyse.extract_tags(text, topK=5)
print (u"关键词:")
print (" ".join(tags))

'''查看 各词 的权重'''
tags = jieba.analyse.extract_tags(text, topK=5, withWeight=True)
for word, weight in tags:
    print(word, weight)


# 词性标注
import jieba.posseg as pseg

words = pseg.cut("我爱北京天安门")
for word, flag in words:
    print("%s %s" % (word, flag))


'''词云展示'''

import jieba
from wordcloud import WordCloud
from scipy.misc import imread
from collections import Counter
import matplotlib.pyplot as plt

data = {}

text_file = open('./data/19Congress.txt', 'r', encoding='utf-8')
text = text_file.read()
with open('./data/stopwords.txt', encoding='utf-8') as file:
    stopwords = {line.strip() for line in file}

seg_list = jieba.cut(text, cut_all=False)
for word in seg_list:
    if len(word) >= 2:
        if not data.__contains__(word):
            data[word] = 0
        data[word] += 1
# print(data)

my_wordcloud = WordCloud(
    background_color='white',  # 设置背景颜色
    max_words=400,  # 设置最大实现的字数
    font_path=r'./data/SimHei.ttf',  # 设置字体格式，如不设置显示不了中文
    mask=imread('./data/mapofChina.jpg'),  # 指定在什么图片上画
    width=1000,
    height=1000,
    stopwords=stopwords
).generate_from_frequencies(data)

plt.figure(figsize=(18, 16))
plt.imshow(my_wordcloud)
plt.axis('off')
plt.show()  # 展示词云
my_wordcloud.to_file('result.jpg')
text_file.close()






















