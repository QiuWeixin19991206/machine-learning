import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

data = pd.read_csv('creditcard.csv')
print(data)

'''画图观察某个key下的值分布'''
count_classes = pd.value_counts(data['Class'], sort = True).sort_index()
count_classes.plot(kind='bar')
plt.title('class')
plt.xlabel('class')
plt.ylabel('frequency')

'''某个标签太少了 过采样造数据 下采样减数据'''
from sklearn.preprocessing import StandardScaler

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))#转np转二维
data = data.drop(['Time', 'Amount'], axis=1)#去除旧数据数据
'''下采样'''
X = data.iloc[:, data.columns !='Class']
y = data.iloc[:, data.columns =='Class']

# 得到所用异常样本的索引
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)

# 得到所有正常样本的索引
normal_indices = np.array(data[data.Class == 0].index)

# 在正常中采指定个数的样本，并获取索引
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
random_normal_indices = np.array(random_normal_indices)

# 合并新的样本索引 和 样本点
under_sample_indices = np.concatenate([random_normal_indices, fraud_indices])
under_sample_data = data.iloc[under_sample_indices]

# 得到新的数据集
X_undersample = under_sample_data.iloc[:, under_sample_data.columns !='Class']
y_undersample = under_sample_data.iloc[:, under_sample_data.columns =='Class']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_undersample, y_undersample, test_size=0.3, random_state=0)

'''过采样SMOTE'''
# from imblearn.over_sampling import SMOTE
# credit_cards=pd.read_csv('creditcard.csv')
#
# columns=credit_cards.columns
# # 在特征中去除掉标签
# features_columns=columns.delete(len(columns)-1)
#
# features=credit_cards[features_columns]
# labels=credit_cards['Class']

'''交叉验证 正则化惩罚项'''
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,recall_score,classification_report

def printing_Kfold_scores(x_train_data, y_train_data):
    fold = KFold(5, shuffle=False)

    # 定义不同力度的正则化惩罚力度
    c_param_range = [0.01, 0.1, 1, 10, 100]
    # 展示结果用的表格
    results_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall score'])
    results_table['C_parameter'] = c_param_range

    # k-fold 表示K折的交叉验证，这里会得到两个索引集合: 训练集 = indices[0], 验证集 = indices[1]
    j = 0
    # 循环遍历不同的参数
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('正则化惩罚力度: ', c_param)
        print('-------------------------------------------')
        print('')

        recall_accs = []

        # 一步步分解来执行交叉验证
        for iteration, indices in enumerate(fold.split(x_train_data)):
            # 指定算法模型，并且给定参数
            lr = LogisticRegression(C=c_param, penalty='l1', solver='liblinear')

            # 训练模型，注意索引不要给错了，训练的时候一定传入的是训练集，所以X和Y的索引都是0
            lr.fit(x_train_data.iloc[indices[0], :].values, y_train_data.iloc[indices[0], :].values.ravel())#ravel 多维数组“展平”成一个一维数组

            # 建立好模型后，预测模型结果，这里用的就是验证集，索引为1
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1], :].values)

            # 有了预测结果之后就可以来进行评估了，这里recall_score需要传入预测值和真实值。
            recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample)
            # 一会还要算平均，所以把每一步的结果都先保存起来。
            recall_accs.append(recall_acc)
            print('Iteration ', iteration, ': 召回率 = ', recall_acc)

        # 当执行完所有的交叉验证后，计算平均结果
        results_table.loc[j, 'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('平均召回率 ', np.mean(recall_accs))
        print('')

    # 找到最好的参数，哪一个Recall高，自然就是最好的了。
    best_c = results_table.loc[results_table['Mean recall score'].astype('float32').idxmax()]['C_parameter']

    # 打印最好的结果
    print('*********************************************************************************')
    print('效果最好的模型所选参数 = ', best_c)
    print('*********************************************************************************')

    return best_c

best_c = printing_Kfold_scores(x_train,y_train)

'''混淆矩阵！'''
import itertools
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    绘制混淆矩阵
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

lr = LogisticRegression(C = best_c, penalty = 'l1',solver='liblinear')  # 实例化模型对象
lr.fit(x_train.values,y_train.values.ravel())
y_pred = lr.predict(x_test.values)

# 计算所需值
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)

print("召回率: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# 绘制
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')

'''不应在下采样的数据集上进行测验，所以重新划分data得到test进行测试'''
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
lr = LogisticRegression(C = best_c, penalty = 'l1', solver='liblinear')  # 实例化模型对象
lr.fit(x_train.values,y_train.values.ravel())
y_pred = lr.predict(x_test.values)
# 计算所需值
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)
print("召回率: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
# 绘制
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')

# 用之前最好的参数来进行建模
lr = LogisticRegression(C=0.01, penalty='l1', solver='liblinear')

# 训练模型，还是用下采样的数据集
lr.fit(x_train.values, y_train.values.ravel())

# 得到预测结果的概率值
y_pred_undersample_proba = lr.predict_proba(x_test.values)

# 指定不同的阈值
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

plt.figure(figsize=(10, 10))

j = 1

# 用混淆矩阵来进行展示
for i in thresholds:
    y_test_predictions_high_recall = y_pred_undersample_proba[:, 1] > i

    plt.subplot(3, 3, j)
    j += 1

    cnf_matrix = confusion_matrix(y_test, y_test_predictions_high_recall)
    np.set_printoptions(precision=2)

    print("给定阈值为:", i, "时测试集召回率: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

    class_names = [0, 1]
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Threshold >= %s' % i)

plt.show()