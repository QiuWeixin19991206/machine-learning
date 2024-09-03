

'''字符串处理'''
input_str = 'AAA今天天气不错，风和 日丽的 '
print(input_str)
print(input_str.strip())# 去两边的
print(input_str.rstrip())# 去除右边的空格
print(input_str.lstrip())# 去除左边的空格

print(input_str.strip('A'))

print(input_str.replace('今天', '昨天'))

print(input_str.replace('今天', ''))

'''查找'''
print(input_str.find('今天'))
print(input_str.isalpha())#是否全为字母
print(input_str.isdigit())#是否全为数字
'''分割 合并'''
print(input_str.split(' ')) # 空格作为分割符
print('B'.join(input_str)) # 以什么为分割符号进行合并


'''正则表达式语法'''
import re
input = '自然语言处理 。 123ABCddvd'
pattern = re.compile(r'.') #指定模板 匹配除了换行符以外所有字符
print(re.findall(pattern, input))

pattern = re.compile(r'[ABC]') # 查找指定符号
print(re.findall(pattern, input))

pattern = re.compile(r'[a-zA-Z]|[0-9]')
a = re.findall(pattern, input)
print(re.findall(pattern, input)) # list

pattern = re.compile(r'\D') # 查找指定符号
print(re.findall(pattern, input))

pattern = re.compile(r'\W') # 查找指定符号
print(re.findall(pattern, input))

pattern = re.compile(r'\S') # 查找指定符号
print(re.findall(pattern, input))

'''重复匹配'''
# \d* 匹配零个或多个数字。
# \d+ 匹配一个或多个数字。
# \d? 匹配零个或一个数字。
pattern = re.compile(r'\d*') # 查找指定符号
print(re.findall(pattern, input))

pattern = re.compile(r'\d+') # 查找指定符号
print(re.findall(pattern, input))

pattern = re.compile(r'\d?') # 查找指定符号
print(re.findall(pattern, input))

pattern = re.compile(r'\d[1, 3]') # 最少匹配1次 最多匹配3次
print(re.findall(pattern, input))


'''match search'''
input2 = '3123自然语言处理'
pattern = re.compile(r'\d')  # 编译一个正则表达式模式，查找单个数字
match = re.match(pattern, input2)  # 使用编译好的正则表达式模式在输入字符串的开头进行匹配
print(match.group())  # 如果匹配成功，打印匹配到的内容


# sub 返回一个被替换的字符串
# subn 返回一个元组，第一个元素是被替换的字符串，第二个元素是一个数字，表明产生了多少次替换
pattern = re.compile(r'\d')
print(re.sub(pattern,'数字',input2))

# split( rule , target [,maxsplit] )
# 第一个参数是正则规则，第二个参数是目标字符串，第三个参数是最多切片次数,返回一个被切完的子字符串的列表
input3 = '自然语言处理123机器学习456深度学习'
pattern = re.compile(r'\d+')
print(re.split(pattern,input3))

# <…>’ 里面是你给这个组起的名字,
# 编译一个正则表达式模式，使用命名捕获组查找数字后跟随非数字字符
pattern = re.compile(r'(?P<dota>\d+)(?P<lol>\D+)')
# 使用编译好的正则表达式模式在输入字符串中搜索第一个匹配项
m = re.search(pattern, input3)
# 如果匹配成功，打印匹配到的命名捕获组 'lol' 对应的内容
print(m.group('lol'))


# 筛选号码
input = 'number 3383-343-220'
pattern = re.compile(r'(\d\d\d-\d\d\d-\d\d\d)')
m = re.search(pattern,input)
print(m.groups())

################################################################################

import nltk
# nltk.download()
from nltk.corpus import stopwords
from nltk.text import Text
from nltk.tokenize import word_tokenize
input_str = "Today's weather is good, very windy and sunny, we have no classes in the afternoon,We have to play basketball tomorrow."
tokens = word_tokenize(input_str)# list
print(tokens)

# 转小写
tokens = [word.lower() for word in tokens]
print(tokens[:5])

# 创建一个Text对象，方便后续操作
t = Text(tokens)
t.count('good') # 1
t.index('good') # 4
# t.plot(8)

# 停用词
# # 打印 stopwords 数据的描述信息，并将换行符替换为空格，以单行形式显示
print(stopwords.readme().replace('\n', ' '))
print(stopwords.fileids())# 停用词表
print(stopwords.raw('english').replace('\n',' '))

test_words = [word.lower() for word in tokens]
# 将 test_words 列表转换为集合，去除重复元素
test_words_set = set(test_words)
# 获取 English 停用词的集合
# 计算 test_words_set 与 stopwords_set 的交集
test_words_set.intersection(set(stopwords.words('english')))

# 过滤掉停用词
filtered = [w for w in test_words_set if(w not in stopwords.words('english'))]
print(filtered)

# 词性标注
from nltk import pos_tag
tags = pos_tag(tokens)
print(tags)

# 分块
from nltk.chunk import RegexpParser
sentence = [('the','DT'),('little','JJ'),('yellow','JJ'),('dog','NN'),('died','VBD')]
grammer = "MY_NP: {<DT>?<JJ>*<NN>}"
cp = nltk.RegexpParser(grammer) #生成规则
result = cp.parse(sentence) #进行分块
print(result)

result.draw() #调用matplotlib库画出来

# 命名实体识别
from nltk import ne_chunk
sentence = "Edison went to Tsinghua University today."
# word_tokenize(sentence): 将句子分词成单词序列。
# pos_tag(...): 对分词后的单词序列进行词性标注，返回带有词性标签的列表。
# ne_chunk(...): 对词性标注后的结果进行命名实体识别，识别出句子中的命名实体
print(ne_chunk(pos_tag(word_tokenize(sentence))))

################################################################################

'''数据清洗实例 '''
import re
from nltk.corpus import stopwords
# 输入数据
s = '    RT @Amila #Test\nTom\'s newly listed Co  &amp; Mary\'s unlisted     Group to supply tech for nlTK.\nh $TSLA $AAPL https:// t.co/x34afsfQsh'

cache_english_stopwords = stopwords.words('english')


def text_clean(text):

    '''
    \$:  将$标记为正则表达式中的特殊字符
    \w*: 匹配任意长度的字母、数字或下划线
    .*: 匹配任意长度的任意字符（除换行符外）
    \b: 表示单词的边界，用于确保匹配的是整个单词
    \w{1,2}: 匹配任意长度为1到2的字母、数字或下划线
    '': 是替换的目标字符串，这里为空字符串，表示删除匹配到的内容
    '''

    print('原始数据:', text, '\n')

    # 去掉HTML标签(e.g. &amp;)
    text_no_special_entities = re.sub(r'\&\w*;|#\w*|@\w*', '', text)
    print('去掉特殊标签后的:', text_no_special_entities, '\n')

    # 去掉一些价值符号
    text_no_tickers = re.sub(r'\$\w*', '', text_no_special_entities)
    print('去掉价值符号后的:', text_no_tickers, '\n')

    # 去掉超链接
    text_no_hyperlinks = re.sub(r'https?:\/\/.*\/\w*', '', text_no_tickers)
    print('去掉超链接后的:', text_no_hyperlinks, '\n')

    # 去掉一些专门名词缩写，简单来说就是字母比较少的词
    text_no_small_words = re.sub(r'\b\w{1,2}\b', '', text_no_hyperlinks)
    print('去掉专门名词缩写后:', text_no_small_words, '\n')

    # 去掉多余的空格
    text_no_whitespace = re.sub(r'\s\s+', ' ', text_no_small_words)
    text_no_whitespace = text_no_whitespace.lstrip(' ')
    print('去掉空格后的:', text_no_whitespace, '\n')

    # 分词
    tokens = word_tokenize(text_no_whitespace)
    print('分词结果:', tokens, '\n')

    # 去停用词
    list_no_stopwords = [i for i in tokens if i not in cache_english_stopwords]
    print('去停用词后结果:', list_no_stopwords, '\n')
    # 过滤后结果
    text_filtered = ' '.join(list_no_stopwords)  # ''.join() would join without spaces between words.
    print('过滤后:', text_filtered)


text_clean(s)


################################################################################
'''spacy'''
import spacy
# python -m spacy download en
nlp = spacy.load('en')
doc = nlp('Weather is good, very windy and sunny. We have no classes in the afternoon.')

# 分词
for token in doc:
    print (token)

#分句
for sent in doc.sents:
    print (sent)

# 词性
for token in doc:
    print ('{}-{}'.format(token,token.pos_))

# 命名体识别
doc_2 = nlp("I went to Paris where I met my old friend Jack from uni.")
for ent in doc_2.ents:
    print ('{}-{}'.format(ent,ent.label_))# Paris-GPE Jack-PERSON

from spacy import displacy

doc = nlp('I went to Paris where I met my old friend Jack from uni.')
displacy.render(doc, style='ent', jupyter=True)


'''找到书中所有人物名字'''
from collections import Counter,defaultdict
def read_file(file_name):
    with open(file_name, 'r') as file:
        return file.read()

# 加载文本数据
text = read_file('./data/pride_and_prejudice.txt')
processed_text = nlp(text)
sentences = [s for s in processed_text.sents]
print (len(sentences))
print(sentences[:5])

def find_person(doc):
    c = Counter()
    for ent in processed_text.ents:
        if ent.label_ == 'PERSON': #如果词性为人
            c[ent.lemma_]+=1
    return c.most_common(10)
print (find_person(processed_text))



'''# 恐怖袭击分析'''
def read_file_to_list(file_name):
    with open(file_name, 'r') as file:
        return file.readlines()

terrorism_articles = read_file_to_list('data/rand-terrorism-dataset.txt')

terrorism_articles_nlp = [nlp(art) for art in terrorism_articles]

common_terrorist_groups = [
    'taliban',
    'al - qaeda',
    'hamas',
    'fatah',
    'plo',
    'bilad al - rafidayn'
]

common_locations = [
    'iraq',
    'baghdad',
    'kirkuk',
    'mosul',
    'afghanistan',
    'kabul',
    'basra',
    'palestine',
    'gaza',
    'israel',
    'istanbul',
    'beirut',
    'pakistan'
]

location_entity_dict = defaultdict(Counter)

for article in terrorism_articles_nlp:

    article_terrorist_groups = [ent.lemma_ for ent in article.ents if
                                ent.label_ == 'PERSON' or ent.label_ == 'ORG']  # 人或者组织
    article_locations = [ent.lemma_ for ent in article.ents if ent.label_ == 'GPE']
    terrorist_common = [ent for ent in article_terrorist_groups if ent in common_terrorist_groups]
    locations_common = [ent for ent in article_locations if ent in common_locations]

    for found_entity in terrorist_common:
        for found_location in locations_common:
            location_entity_dict[found_entity][found_location] += 1

import pandas as pd

location_entity_df = pd.DataFrame.from_dict(dict(location_entity_dict),dtype=int)
location_entity_df = location_entity_df.fillna(value = 0).astype(int)



import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12, 10))
hmap = sns.heatmap(location_entity_df, annot=True, fmt='d', cmap='YlGnBu', cbar=False)

# 添加信息
plt.title('Global Incidents by Terrorist group')
plt.xticks(rotation=30)
plt.show()



























print()


