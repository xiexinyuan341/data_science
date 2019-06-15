from functools import reduce
import collections
from pprint import pprint

import pandas as pd
import numpy as np

from ContentBasedRecommenderSystem.MovieProfile import get_movie_dataset, create_movie_profile

"""
user profile画像建立：
这里将电影的便签直接作为用户的标签来创建用户画像
1. 提取用户为电影的打分列表
2. 根据观看列表和物品画像为用户匹配关键词，并统计词频
3. 根据词频排序，最多保留TOP-N个词，作为用户的标签
"""


def create_user_profile(rating_file_path, movie_profile, topN=30):
    watch_record = pd.read_csv(rating_file_path, usecols=range(2),
                               dtype={"userId": np.int32, "movieId": np.int32})

    watch_record = watch_record.groupby("userId").agg(list)

    user_profile = {}
    for uid, mids in watch_record.itertuples():
        record_movie_prifole = movie_profile.loc[list(mids)]
        counter = collections.Counter(
            reduce(lambda x, y: list(x) + list(y), record_movie_prifole["profile"].values))
        # 取出出现次数最多的前50个词
        interest_words = counter.most_common(topN)
        # 取出出现次数最多的词的出现的次数,用于下面缩放数据
        maxcount = interest_words[0][1]
        # 利用次数计算权重,将出现权重缩放到0-1的范围
        interest_words = [(w, round(c / maxcount, 4)) for w, c in interest_words]
        user_profile[uid] = interest_words

    return user_profile


if __name__ == '__main__':
    movie_dataset = get_movie_dataset("data/tags.csv", "data/movies.csv")
    movie_profile = create_movie_profile(movie_dataset)
    user_profile = create_user_profile("data/ratings.csv", movie_profile)
    pprint(user_profile)
