import nltk
import re
import string
import numpy as np
import pandas as pd
import codecs
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import itertools
from sklearn.metrics import confusion_matrix
import gensim

from keras.layers import Dense, Input, Flatten, Dropout, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import LSTM, Bidirectional
from keras.models import Model

def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df


def cv(data):
    '''是 scikit-learn 库中用于文本特征提取的一个类。
    它可以将文本文档集合转换为文档-词项矩阵，其中每个文档被表示为一个向量，
    向量中的每个元素代表了相应词汇在文档中的出现次数。'''
    count_vectorizer = CountVectorizer()

    emb = count_vectorizer.fit_transform(data)

    return emb, count_vectorizer


'''Bag of Words Counts'''
def BagofWordsCounts():
    # tolist() 把(列表、数组等）转换为 Python 内置的列表（list）类型的方法
    list_corpus = clean_questions["text"].tolist()
    list_labels = clean_questions["class_label"].tolist()

    '''数据'''
    X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2,random_state=40)

    X_train_counts, count_vectorizer = cv(X_train)
    X_test_counts = count_vectorizer.transform(X_test)

    '''可视化'''
    fig = plt.figure(figsize=(10, 10))
    plot_LSA(X_train_counts, y_train)
    # plt.show()

    '''训练 预测'''
    clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',multi_class='multinomial', n_jobs=-1, random_state=40)
    clf.fit(X_train_counts, y_train)
    y_predicted_counts = clf.predict(X_test_counts)

    '''评估'''
    accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

    '''混淆矩阵检查'''
    cm = confusion_matrix(y_test, y_predicted_counts)
    fig = plt.figure(figsize=(10, 10))
    plot = plot_confusion_matrix(cm, classes=['Irrelevant', 'Disaster', 'Unsure'], normalize=False,
                                 title='Confusion matrix')
    print(cm)

    importance = get_most_important_features(count_vectorizer, clf, 10)

    top_scores = [a[0] for a in importance[1]['tops']]
    top_words = [a[1] for a in importance[1]['tops']]
    bottom_scores = [a[0] for a in importance[1]['bottom']]
    bottom_words = [a[1] for a in importance[1]['bottom']]

    plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Most important words for relevance")

    return None

def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True):
    lsa = TruncatedSVD(n_components=2) # 使用 TruncatedSVD 进行潜在语义分析，将数据降维到两个主成分
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    color_mapper = {label: idx for idx, label in enumerate(set(test_labels))} # 创建一个字典，将类别标签映射到整数索引
    color_column = [color_mapper[label] for label in test_labels] # 根据类别标签映射字典，将每个数据点的类别映射为对应的整数索引
    colors = ['orange', 'blue', 'blue'] # 定义散点图中使用的颜色列表
    if plot: # 如果 plot 参数为 True，则绘制散点图
        plt.scatter(lsa_scores[:, 0], lsa_scores[:, 1], s=8, alpha=.8, c=test_labels,
                        cmap=matplotlib.colors.ListedColormap(colors))
        # 创建图例，标识不同类别的颜色对应的含义
        red_patch = mpatches.Patch(color='orange', label='Irrelevant')
        green_patch = mpatches.Patch(color='blue', label='Disaster')
        plt.legend(handles=[red_patch, green_patch], prop={'size': 30})


'''评估'''
def get_metrics(y_test, y_predicted):
    # 计算精确率：真正例 / (真正例 + 假正例)
    precision = precision_score(y_test, y_predicted, pos_label=None, average='weighted')

    # 计算召回率：真正例 / (真正例 + 假反例)
    recall = recall_score(y_test, y_predicted, pos_label=None, average='weighted')

    # 计算 F1 分数：精确率和召回率的调和平均数
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    # 计算准确率：(真正例 + 真反例) / 总数
    accuracy = accuracy_score(y_test, y_predicted)

    return accuracy, precision, recall, f1


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.winter):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)

    return plt


def get_most_important_features(vectorizer, model, n=5):
    # 创建一个从索引到单词的映射字典
    index_to_word = {v: k for k, v in vectorizer.vocabulary_.items()}

    # 初始化一个空字典，用于存储每个类别的重要特征
    classes = {}

    # 遍历每个类别
    for class_index in range(model.coef_.shape[0]):
        # 获取每个特征（单词）的重要性分数并与单词索引进行关联
        word_importances = [(el, index_to_word[i]) for i, el in enumerate(model.coef_[class_index])]

        # 按照特征的重要性分数进行降序排序
        sorted_coeff = sorted(word_importances, key=lambda x: x[0], reverse=True)

        # 获取最重要的前 n 个特征和最不重要的前 n 个特征
        tops = sorted(sorted_coeff[:n], key=lambda x: x[0])  # 最重要的前 n 个特征
        bottom = sorted_coeff[-n:]  # 最不重要的前 n 个特征

        # 将结果存储到 classes 字典中
        classes[class_index] = {
            'tops': tops,  # 存储最重要的特征
            'bottom': bottom  # 存储最不重要的特征
        }

    return classes

def plot_important_words(top_scores, top_words, bottom_scores, bottom_words, name):
    y_pos = np.arange(len(top_words))
    top_pairs = [(a, b) for a, b in zip(top_words, top_scores)]
    top_pairs = sorted(top_pairs, key=lambda x: x[1])

    bottom_pairs = [(a, b) for a, b in zip(bottom_words, bottom_scores)]
    bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)

    top_words = [a[0] for a in top_pairs]
    top_scores = [a[1] for a in top_pairs]

    bottom_words = [a[0] for a in bottom_pairs]
    bottom_scores = [a[1] for a in bottom_pairs]

    fig = plt.figure(figsize=(10, 10))

    plt.subplot(121)
    plt.barh(y_pos, bottom_scores, align='center', alpha=0.5)
    plt.title('Irrelevant', fontsize=20)
    plt.yticks(y_pos, bottom_words, fontsize=14)
    plt.suptitle('Key words', fontsize=16)
    plt.xlabel('Importance', fontsize=20)

    plt.subplot(122)
    plt.barh(y_pos, top_scores, align='center', alpha=0.5)
    plt.title('Disaster', fontsize=20)
    plt.yticks(y_pos, top_words, fontsize=14)
    plt.suptitle(name, fontsize=16)
    plt.xlabel('Importance', fontsize=20)

    plt.subplots_adjust(wspace=0.8)
    plt.show()

'''TFIDF'''
def TFIDF():
    list_corpus = clean_questions["text"].tolist()
    list_labels = clean_questions["class_label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2,random_state=40)

    X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    fig = plt.figure(figsize=(10, 10))
    plot_LSA(X_train_tfidf, y_train)
    plt.show()

    clf_tfidf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                                   multi_class='multinomial', n_jobs=-1, random_state=40)
    clf_tfidf.fit(X_train_tfidf, y_train)
    y_predicted_tfidf = clf_tfidf.predict(X_test_tfidf)

    accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf = get_metrics(y_test, y_predicted_tfidf)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_tfidf, precision_tfidf,
                                                                           recall_tfidf, f1_tfidf))

    cm2 = confusion_matrix(y_test, y_predicted_tfidf)
    fig = plt.figure(figsize=(10, 10))
    plot = plot_confusion_matrix(cm2, classes=['Irrelevant', 'Disaster', 'Unsure'], normalize=False,
                                 title='Confusion matrix')

    print("TFIDF confusion matrix")
    print(cm2)

    # 词语的解释
    importance_tfidf = get_most_important_features(tfidf_vectorizer, clf_tfidf, 10)
    top_scores = [a[0] for a in importance_tfidf[1]['tops']]
    top_words = [a[1] for a in importance_tfidf[1]['tops']]
    bottom_scores = [a[0] for a in importance_tfidf[1]['bottom']]
    bottom_words = [a[1] for a in importance_tfidf[1]['bottom']]

    plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Most important words for relevance")


def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()

    train = tfidf_vectorizer.fit_transform(data)

    return train, tfidf_vectorizer

'''Word2vec'''
def Word2vec():
    # 我们现在考虑的是每一个词基于频率的情况，如果在新的测试环境下有些词变了呢？比如说goog和positive.有些词可能表达的意义差不多但是却长得不一样，这样我们的模型就难捕捉到了。
    # word2vec 可以解决
    list_labels = clean_questions["class_label"].tolist()

    word2vec_path = "GoogleNews-vectors-negative300.bin"
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    embeddings = get_word2vec_embeddings(word2vec, clean_questions)
    X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(embeddings, list_labels,
                                                                                            test_size=0.2,
                                                                                            random_state=40)
    fig = plt.figure(figsize=(16, 16))
    plot_LSA(embeddings, list_labels)
    # plt.show()

    '''训练 预测'''
    clf_w2v = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                                 multi_class='multinomial', random_state=40)
    clf_w2v.fit(X_train_word2vec, y_train_word2vec)
    y_predicted_word2vec = clf_w2v.predict(X_test_word2vec)

    accuracy_word2vec, precision_word2vec, recall_word2vec, f1_word2vec = get_metrics(y_test_word2vec,
                                                                                      y_predicted_word2vec)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_word2vec, precision_word2vec,
                                                                           recall_word2vec, f1_word2vec))

    cm_w2v = confusion_matrix(y_test_word2vec, y_predicted_word2vec)
    fig = plt.figure(figsize=(10, 10))
    plot = plot_confusion_matrix(cm_w2v, classes=['Irrelevant', 'Disaster', 'Unsure'], normalize=False,
                                 title='Confusion matrix')
    # plt.show()
    print("Word2Vec confusion matrix")
    print(cm_w2v)

    return None


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    # 如果 tokens_list 为空，返回一个全零向量
    if len(tokens_list) < 1:
        return np.zeros(k)

    # 如果 generate_missing 为 True，缺失的单词用随机向量表示，否则用全零向量表示
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]

    # 计算 tokens_list 的长度
    length = len(vectorized)
    # 计算所有单词向量的和
    summed = np.sum(vectorized, axis=0)
    # 计算平均向量
    averaged = np.divide(summed, length)

    return averaged


def get_word2vec_embeddings(vectors, clean_questions, generate_missing=False):
    # 对 clean_questions 数据帧中的每一个 'tokens' 列应用 get_average_word2vec 函数
    embeddings = clean_questions['tokens'].apply(lambda x: get_average_word2vec(x, vectors, generate_missing=generate_missing))
    # 返回嵌入向量列表
    return list(embeddings)

'''基于深度学习的自然语言处理（CNN与RNN）'''
def deeplearning():
    # 加载预训练的Word2Vec模型
    word2vec_path = "GoogleNews-vectors-negative300.bin"
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    # 定义嵌入维度和最大序列长度
    EMBEDDING_DIM = 300
    MAX_SEQUENCE_LENGTH = 35

    # 定义词汇表的大小
    VOCAB_SIZE = len(VOCAB)

    # 定义验证集的比例
    VALIDATION_SPLIT = 0.2

    # 创建一个Tokenizer对象
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)

    # 使用Tokenizer拟合文本数据
    tokenizer.fit_on_texts(clean_questions["text"].tolist())

    # 将文本数据转换为序列
    sequences = tokenizer.texts_to_sequences(clean_questions["text"].tolist())

    # 获取词汇索引
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # 填充序列，使其具有相同的长度
    cnn_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # 将标签转换为分类格式
    labels = to_categorical(np.asarray(clean_questions["class_label"]))

    # 打乱数据
    indices = np.arange(cnn_data.shape[0])
    np.random.shuffle(indices)
    cnn_data = cnn_data[indices]
    labels = labels[indices]

    # 计算验证集的样本数量
    num_validation_samples = int(VALIDATION_SPLIT * cnn_data.shape[0])

    # 创建嵌入矩阵
    embedding_weights = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, index in word_index.items():
        # 如果词在预训练的词向量中，则使用预训练的向量，否则使用随机向量
        embedding_weights[index, :] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
    print(embedding_weights.shape)

    x_train = cnn_data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = cnn_data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]

    model = ConvNet(embedding_weights, MAX_SEQUENCE_LENGTH, len(word_index) + 1, EMBEDDING_DIM,
                    len(list(clean_questions["class_label"].unique())), False)

    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=3, batch_size=128)



def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index, trainable=False, extra_conv=True):
    # 定义嵌入层
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                weights=[embeddings],
                                input_length=max_sequence_length,
                                trainable=trainable)

    # 定义输入层，输入为序列长度
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    # Yoon Kim模型 (https://arxiv.org/abs/1408.5882)
    convs = []
    filter_sizes = [3, 4, 5]

    # 为每个过滤器大小创建卷积层和最大池化层
    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=128, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(pool_size=3)(l_conv)
        convs.append(l_pool)

    # 将所有卷积层的输出合并
    l_merge = Concatenate(mode='concat', concat_axis=1)(convs)

    # 添加一个1D卷积层和全局最大池化层，替代Yoon Kim模型
    conv = Conv1D(filters=128, kernel_size=3, activation='relu')(embedded_sequences)
    pool = MaxPooling1D(pool_size=3)(conv)

    # 根据 extra_conv 参数选择是否使用额外的卷积层
    if extra_conv:
        x = Dropout(0.5)(l_merge)
    else:
        x = Dropout(0.5)(pool)

    # 展平层，将多维数据转换为一维
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)

    # 输出层，使用 softmax 激活函数进行多分类
    preds = Dense(labels_index, activation='softmax')(x)

    # 定义并编译模型
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy', # 多分类 交叉熵
                  optimizer='adam',
                  metrics=['acc'])

    return model


if __name__ == '__main__':
    # questions = pd.read_csv('socialmedia_relevant_cols_clean.csv') # text choose_one  class_label
    # questions.columns = ['text', 'choose_one', 'class_label']
    # print(questions.head())
    # print(questions.describe())
    #
    # questions =standardize_text(questions, 'text') # 处理text的杂乱符号
    # questions.to_csv("clean_data.csv")
    # print(questions.head())

    clean_questions = pd.read_csv("clean_data.csv")
    print(clean_questions.tail()) #默认显示最后 5 行
    print(clean_questions.groupby('class_label').count())

    '''  处理流程 分词 训练与测试集 检查与验证'''
    tokenizer = RegexpTokenizer(r'\w+') #分割文本的工具
    clean_questions["tokens"] = clean_questions["text"].apply(tokenizer.tokenize)
    print(clean_questions.head())

    '''语料库情况'''
    all_words = [word for tokens in clean_questions["tokens"] for word in tokens] # 所有词
    sentence_lengths = [len(tokens) for tokens in clean_questions["tokens"]] #每行句子长度
    VOCAB = sorted(list(set(all_words))) # 集合中不允许重复值，转回list 再排序
    print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
    print("Max sentence length is %s" % max(sentence_lengths))

    # 句子长度情况
    fig = plt.figure(figsize=(10, 10))
    plt.xlabel('Sentence length')
    plt.ylabel('Number of sentences')
    plt.hist(sentence_lengths)


    BagofWordsCounts()

    TFIDF()

    Word2vec()

    deeplearning()

    plt.show()












































