from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import pandas as pd
import numpy as np

from pprint import pprint

'''
- 利用tags.csv中每部电影的标签作为电影的候选关键词
- 利用TF·IDF计算每部电影的标签的tfidf值，选取TOP-N个关键词作为电影画像标签
- 并将电影的分类词直接作为每部电影的画像标签
'''


def get_movie_dataset(tag_file_path, movies_file_path):
    # 加载基于所有电影的标签
    _tags = pd.read_csv(tag_file_path, usecols=range(1, 3)).dropna()
    tags = _tags.groupby("movieId").agg(list)

    # 加载电影列表数据集
    movies = pd.read_csv(movies_file_path, index_col="movieId")
    # 将类别词分开
    movies["genres"] = movies["genres"].apply(lambda x: x.split("|"))
    # 为每部电影匹配对应的标签数据，如果没有将会是NAN
    movies_index = set(movies.index) & set(tags.index)
    new_tags = tags.loc[list(movies_index)]
    # 将标签数据加入到movie数据里面
    ret = movies.join(new_tags)

    # 构建电影数据集，包含电影Id、电影名称、类别、标签四个字段
    # 如果电影没有标签数据，那么就替换为空列表
    # map(fun,可迭代对象)
    movie_dataset = pd.DataFrame(
        list(map(
            lambda x: (x[0], x[1], x[2], x[2] + x[3]) if x[3] is not np.nan else (x[0], x[1], x[2], []),
            ret.itertuples()))
        , columns=["movieId", "title", "genres", "tags"]
    )

    movie_dataset.set_index("movieId", inplace=True)
    return movie_dataset


def create_movie_profile(movie_dataset, topN=30):
    """
    使用tfidf分析tags,然后根据tfidf的分值取topN作为电影的画像
    """
    dataset = movie_dataset["tags"].values
    # 根据数据集建立词袋，并统计词频，将所有词放入一个词典，使用索引进行获取
    dct = Dictionary(dataset)
    # 根据将每条数据，返回对应的词索引和词频
    corpus = [dct.doc2bow(line) for line in dataset]
    # 训练TF-IDF模型，即计算TF-IDF值
    model = TfidfModel(corpus)

    _movie_profile = []
    for i, data in enumerate(movie_dataset.itertuples()):
        mid = data[0]
        title = data[1]
        genres = data[2]
        vector = model[corpus[i]]  # vector每行tag的tfidf向量
        # 排序选取tfidf高的N个tag
        movie_tags = sorted(vector, key=lambda x: x[1], reverse=True)[:topN]
        # dct[x[0]]将词索引变成词的字符串
        topN_tags_weights = dict(map(lambda x: (dct[x[0]], x[1]), movie_tags))
        # 将类别词的添加进去，并设置权重值为1.0
        for g in genres:
            topN_tags_weights[g] = 1.0
        topN_tags = [i[0] for i in topN_tags_weights.items()]
        _movie_profile.append((mid, title, topN_tags, topN_tags_weights))

    movie_profile = pd.DataFrame(_movie_profile, columns=["movieId", "title", "profile", "weights"])
    movie_profile.set_index("movieId", inplace=True)
    return movie_profile


def create_inverted_table(movie_profile):
    """
    用电影画像创建倒排索引
    """
    inverted_table = {}
    for mid, weights in movie_profile["weights"].iteritems():
        for tag, weight in weights.items():
            # 到inverted_table dict 用tag作为Key去取值 如果取不到就返回[]
            _ = inverted_table.get(tag, [])
            # 将电影的id 和 权重 放到一个tuple中 添加到list中
            _.append((mid, weight))
            # 将修改后的值设置回去
            inverted_table.setdefault(tag, _)
    return inverted_table


if __name__ == '__main__':
    movie_dataset = get_movie_dataset("data/tags.csv", "data/movies.csv")
    movie_profile = create_movie_profile(movie_dataset)
    pprint(movie_profile.head())
    inverted_table = create_inverted_table(movie_profile)
    pprint(inverted_table)
