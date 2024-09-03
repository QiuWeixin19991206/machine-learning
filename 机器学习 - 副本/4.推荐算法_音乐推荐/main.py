import pandas as pd
import numpy as np
import time
import sqlite3
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# data_home = './'
# triplet_dataset = pd.read_csv(filepath_or_buffer=data_home + 'train_triplets.txt', sep='\t', header=None, names=['user', 'song', 'play_count'])
# print(triplet_dataset.info)

# '''统计用户播放量'''
# output_dict = {} # 初始化一个空字典，用于存储每个用户的播放次数总和
# with open(data_home+'train_triplets.txt') as f:
#     for line_number, line in enumerate(f):
#         user = line.split('\t')[0] # 通过制表符分割行内容，取出用户信息
#         play_count = int(line.split('\t')[2]) # 取出播放次数，转换为整数类型
#         if user in output_dict:
#             play_count +=output_dict[user] # 更新播放次数
#             output_dict.update({user:play_count}) # 更新字典中的播放次数
#         output_dict.update({user:play_count})
# output_list = [{'user': k, 'play_count':v} for k, v in output_dict.items()] # 将字典转换为列表
# play_count_df = pd.DataFrame(output_list)  # 将列表转换为DataFrame
# play_count_df = play_count_df.sort_values(by='play_count', ascending=False) # 按播放次数降序排序DataFrame
# play_count_df.to_csv(path_or_buf='user_playcount_df.csv', index=False)
#
# '''统计歌曲播放量'''
# output_dict = {} # 初始化一个空字典，用于存储每个用户的播放次数总和
# with open(data_home+'train_triplets.txt') as f:
#     for line_number, line in enumerate(f):
#         song = line.split('\t')[1] # 通过制表符分割行内容，取出用户信息
#         play_count = int(line.split('\t')[2]) # 取出播放次数，转换为整数类型
#         if song in output_dict:
#             play_count +=output_dict[song] # 更新播放次数
#             output_dict.update({song:play_count}) # 更新字典中的播放次数
#         output_dict.update({song:play_count})
# output_list = [{'song': k, 'play_count':v} for k, v in output_dict.items()] # 将字典转换为列表
# song_count_df = pd.DataFrame(output_list)  # 将列表转换为DataFrame
# song_count_df = song_count_df.sort_values(by='play_count', ascending=False) # 按播放次数降序排序DataFrame
# song_count_df.to_csv(path_or_buf='song_playcount_df.csv', index=False)

play_count_df = pd.read_csv(filepath_or_buffer='user_playcount_df.csv')
print(play_count_df.head(5))
song_count_df = pd.read_csv(filepath_or_buffer='song_playcount_df.csv')
print(song_count_df.head(5))

total_play_count = sum(song_count_df.play_count)
print(float((play_count_df.head(100000).play_count.sum()) / total_play_count) * 100)
play_count_subset = play_count_df.head(100000)

print(float((song_count_df.head(30000).play_count.sum()) / total_play_count) * 100)
song_count_subset = song_count_df.head(30000)

# 取10w用户 3w首歌
user_subset = list(play_count_subset.user)
song_subset = list(song_count_subset.song)
'''清除多余数据'''
data_home = './'
triplet_dataset = pd.read_csv(filepath_or_buffer=data_home + 'train_triplets.txt', sep='\t', header=None, names=['user', 'song', 'play_count'])
triplet_dataset_sub = triplet_dataset[triplet_dataset.user.isin(user_subset)]
del triplet_dataset
triplet_dataset_sub_song = triplet_dataset_sub[triplet_dataset_sub.song.isin(song_subset)]
del triplet_dataset_sub
triplet_dataset_sub_song.to_csv(path_or_buf=data_home + 'triplet_dataset_sub_song.csv', index=False)

'''加入音乐详细信息'''
conn = sqlite3.connect(data_home+'track_metadata.db')
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(cur.fetchall()) # 获取了所有查询结果，并返回一个包含所有表名的列表
track_metadata_df = pd.read_sql(con=conn, sql='select * from songs')
track_metadata_df_sub = track_metadata_df[track_metadata_df.song_id.isin(song_subset)]
track_metadata_df_sub.to_csv(path_or_buf=data_home+'track_metadata_df_sub.csv', index=False)
print(track_metadata_df_sub.shape)

del(track_metadata_df_sub['track_id'])
del(track_metadata_df_sub['artist_mbid'])
# 使用 drop_duplicates 方法基于 song_id 列去除了 DataFrame track_metadata_df_sub 中的重复行
track_metadata_df_sub = track_metadata_df_sub.drop_duplicates(['song_id'])
triplet_dataset_sub_song_merged = pd.merge(triplet_dataset_sub_song, track_metadata_df_sub, how='left', left_on='song', right_on='song_id')
triplet_dataset_sub_song_merged.rename(columns={'play_count':'listen_count'}, inplace=True)


del(triplet_dataset_sub_song_merged['artist_id'])
del(triplet_dataset_sub_song_merged['song_id'])
del(triplet_dataset_sub_song_merged['duration'])
del(triplet_dataset_sub_song_merged['artist_familiarity'])
del(triplet_dataset_sub_song_merged['artist_hotttnesss'])
del(triplet_dataset_sub_song_merged['track_7digitalid'])
del(triplet_dataset_sub_song_merged['shs_perf'])
del(triplet_dataset_sub_song_merged['shs_work'])

'''可视化'''
# # 选择了 title 和 listen_count 两列，然后按照 title 列进行分组
# popular_songs = triplet_dataset_sub_song_merged[['title', 'listen_count']].groupby('title').sum().reset_index()
# # 进行降序排序，并选取前20行
# popular_songs_top_20 = popular_songs.sort_values('listen_count', ascending=False).head(n=20)
# # 用于重置所有的rc参数为默认值
# plt.rcdefaults()
# # 提取了歌曲标题和播放次数
# objects = (list(popular_songs_top_20['title']))
# y_pos = np.arange(len(objects))
# performance = list(popular_songs_top_20['listen_count'])
# # 垂直条形图
# plt.bar(y_pos, performance, align='center', alpha=0.5)
# plt.xticks(y_pos, objects, rotation='vertical')
# plt.ylabel('Item count')
# plt.title('Most popular songs')
#
# popular_release = triplet_dataset_sub_song_merged[['release', 'listen_count']].groupby('release').sum().reset_index()
# popular_release_top_20 = popular_release.sort_values('listen_count', ascending=False).head(n=20)
# objects = (list(popular_release_top_20['release']))
# y_pos = np.arange(len(objects))
# performance = list(popular_release_top_20['listen_count'])
# plt.bar(y_pos, performance, align='center', alpha=0.5)
# plt.xticks(y_pos, objects, rotation='vertical')
# plt.ylabel('Item count')
# plt.title('Most popular Release')
#
# popular_artist = triplet_dataset_sub_song_merged[['artist_name', 'listen_count']].groupby(
#     'artist_name').sum().reset_index()
# popular_artist_top_20 = popular_artist.sort_values('listen_count', ascending=False).head(n=20)
# objects = (list(popular_artist_top_20['artist_name']))
# y_pos = np.arange(len(objects))
# performance = list(popular_artist_top_20['listen_count'])
# plt.bar(y_pos, performance, align='center', alpha=0.5)
# plt.xticks(y_pos, objects, rotation='vertical')
# plt.ylabel('Item count')
# plt.title('Most popular Artists')
# plt.show()

triplet_dataset_sub_song_merged_set = triplet_dataset_sub_song_merged
train_data, test_data = train_test_split(triplet_dataset_sub_song_merged_set, test_size = 0.40, random_state=0)


'''简单暴力，排行榜单推荐'''
# def create_popularity_recommendation(train_data, user_id, item_id):
#     # 获取每首独特歌曲的用户ID数作为推荐分数
#     train_data_grouped = train_data.groupby([item_id]).agg({user_id: 'count'}).reset_index()
#     train_data_grouped.rename(columns={user_id: 'score'}, inplace=True)
#
#     # 根据推荐分数对歌曲进行排序 降序
#     train_data_sort = train_data_grouped.sort_values(['score', item_id], ascending=[0, 1])
#
#     # 根据分数生成推荐排名 ascending=0 表示降序排列，method='first' 表示按照出现的顺序进行排名
#     train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
#
#     # 获取排名前20的推荐结果，即推荐分数最高的20首歌曲
#     popularity_recommendations = train_data_sort.head(20)
#     return popularity_recommendations
#
# recommendations = create_popularity_recommendation(triplet_dataset_sub_song_merged,'user','title')

'''基于歌曲相似度的推荐'''
# 听过A,B歌的两个人群a,b  sore=计算两个人群 交集/并集
user_subset = list(play_count_subset.user)
song_subset = list(song_count_subset.song)
triplet_dataset_sub_song_merged_sub = triplet_dataset_sub_song_merged[triplet_dataset_sub_song_merged.song.isin(song_subset)]
import Recommenders
estimator = Recommenders.item_similarity_recommender_py()
estimator.create(train_data, 'user', 'title')
user_id = list(train_data.user)[7]
user_items = estimator.get_user_items(user_id)
print(user_items)

'''SVD矩阵分解'''
# M*M M*N(特征值) N*N==> M*K K*K K*N K<N但是10%的特征值就能表示总体的99%信息
import math as mt
from scipy.sparse.linalg import * #used for matrix multiplication
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix

from scipy.sparse import coo_matrix

# 先计算歌曲被当前用户播放量 / 用户播放总量 当做分值
# 计算每个用户的总收听量并重命名列
triplet_dataset_sub_song_merged_sum_df = triplet_dataset_sub_song_merged[['user','listen_count']].groupby('user').sum().reset_index()
triplet_dataset_sub_song_merged_sum_df.rename(columns={'listen_count':'total_listen_count'},inplace=True)
# 将计算好的总收听量合并回原数据框
triplet_dataset_sub_song_merged = pd.merge(triplet_dataset_sub_song_merged,triplet_dataset_sub_song_merged_sum_df)
print(triplet_dataset_sub_song_merged[triplet_dataset_sub_song_merged.user =='d6589314c0a9bcbca4fee0c93b14bc402363afea'][['user','song','listen_count','fractional_play_count']].head())
# 复制数据并获取唯一用户和歌曲的列表
small_set = triplet_dataset_sub_song_merged
user_codes = small_set.user.drop_duplicates().reset_index()
song_codes = small_set.song.drop_duplicates().reset_index()
# 重命名列
user_codes.rename(columns={'index':'user_index'}, inplace=True)
song_codes.rename(columns={'index':'song_index'}, inplace=True)
# 为每个用户和歌曲创建索引值
song_codes['so_index_value'] = list(song_codes.index)
user_codes['us_index_value'] = list(user_codes.index)
small_set = pd.merge(small_set,song_codes,how='left')
small_set = pd.merge(small_set,user_codes,how='left')
# 准备创建稀疏矩阵的数据
mat_candidate = small_set[['us_index_value','so_index_value','fractional_play_count']]
data_array = mat_candidate.fractional_play_count.values
row_array = mat_candidate.us_index_value.values
col_array = mat_candidate.so_index_value.values
# 创建稀疏矩阵
data_sparse = coo_matrix((data_array, (row_array, col_array)),dtype=float)
# 计算SVD的函数
def compute_svd(urm, K):
    U, s, Vt = svds(urm, K)

    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        S[i, i] = mt.sqrt(s[i])

    U = csc_matrix(U, dtype=np.float32)
    S = csc_matrix(S, dtype=np.float32)
    Vt = csc_matrix(Vt, dtype=np.float32)

    return U, S, Vt

# 计算估计矩阵的函数
def compute_estimated_matrix(urm, U, S, Vt, uTest, K, test):
    rightTerm = S * Vt
    max_recommendation = 250
    estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
    recomendRatings = np.zeros(shape=(MAX_UID, max_recommendation), dtype=np.float16)
    for userTest in uTest:
        prod = U[userTest, :] * rightTerm
        estimatedRatings[userTest, :] = prod.todense()
        recomendRatings[userTest, :] = (-estimatedRatings[userTest, :]).argsort()[:max_recommendation]
    return recomendRatings

K=50
urm = data_sparse
MAX_PID = urm.shape[1]
MAX_UID = urm.shape[0]

U, S, Vt = compute_svd(urm, K)

uTest = [4,5,6,7,8,873,23]
# 获取推荐列表
uTest_recommended_items = compute_estimated_matrix(urm, U, S, Vt, uTest, K, True)
# 打印推荐结果
for user in uTest:
    print("Recommendation for user with user id {}". format(user))
    rank_value = 1
    for i in uTest_recommended_items[user,0:10]:
        song_details = small_set[small_set.so_index_value == i].drop_duplicates('so_index_value')[['title','artist_name']]
        print("The number {} recommended song is {} BY {}".format(rank_value, list(song_details['title'])[0],list(song_details['artist_name'])[0]))
        rank_value+=1