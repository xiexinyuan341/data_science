from pprint import pprint

from ContentBasedRecommenderSystem.MovieProfile import get_movie_dataset, create_movie_profile, create_inverted_table
from ContentBasedRecommenderSystem.UserProfile import create_user_profile

"""
根据用户画像选取用户喜欢的标签,
再根据电影的倒排索引获取与标签相关的电影,
然后将用户喜欢标签的程度与标签在所在电影的重要程度相乘取平均值得到用户对电影的喜欢权重
最后取topN个电影.(这里没有排除掉用户已经观看过的电影.)
"""


def recommend_movie(uid, user_profile, inverted_table, top_n=5):
    result_table = {}
    for interest_word, interest_weight in user_profile.get(uid):
        related_movies = inverted_table[interest_word]  # 获取与标签相关的电影列表
        for mid, related_weight in related_movies:
            # related_weight为tag在这个movie的td-idf值
            _ = result_table.get(mid, [])
            _.append(interest_weight * related_weight)  # 总和用户兴趣权重和电影标签tf-idf权重
            result_table.setdefault(mid, _)

    # 获取得到权重的平均值
    rs_result = map(lambda x: (x[0], sum(x[1]) / len(x[1])), result_table.items())
    # 排序取topN
    rs_result = sorted(rs_result, key=lambda x: x[1], reverse=True)[:top_n]
    return rs_result


if __name__ == '__main__':
    movie_dataset = get_movie_dataset("data/tags.csv", "data/movies.csv")
    movie_profile = create_movie_profile(movie_dataset)
    inverted_table = create_inverted_table(movie_profile)
    user_profile = create_user_profile("data/ratings.csv", movie_profile)
    pprint(recommend_movie(1, user_profile, inverted_table))
