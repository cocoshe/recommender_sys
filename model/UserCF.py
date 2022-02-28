import pandas as pd
import numpy as np


class UserCF:
    def __init__(self):
        self.file_path = 'dataset/ratings.csv'
        self.df = pd.read_csv(self.file_path)

    @staticmethod
    def _cosine_similarity(vector1, vector2):
        """
        因为MovieId大小不是量,而是代表类, 这里的余弦相似就是进行了一个类似于 hashmap 的操作
        所以用len来带入了cosine_similarity操作
        simple method for calculate cosine distance.
        e.g: x = [1 0 1 1 0], y = [0 1 1 0 1]
             cosine = (x1*y1+x2*y2+...) / [sqrt(x1^2+x2^2+...) * sqrt(y1^2+y2^2+...)]
             that means union_len(movies1, movies2) / sqrt(len(movies1)*len(movies2))
        """
        union_len = len(set(vector1) & set(vector2))  # 等价于x1*y1+x2*y2+...
        if union_len == 0:
            return 0.0
        product = len(vector1) * len(vector2)  # 等价于sqrt(x1^2+x2^2+...) * sqrt(y1^2+y2^2+...)
        cosine = union_len / np.sqrt(product)
        return cosine

    def get_top_n_users(self, target_user_id, top_n):
        """
        get top n users that similar to target_user
        获取与target_user_id最相似的top_n个用户
        依据的是用户点评过的电影的相似度
        :param target_user_id:
        :param top_n:
        :return: top n users, similarity
        """
        target_movies = self.df[self.df['UserId'] == target_user_id]['MovieId'].unique()  # target用户点评过的电影
        # print('target_movies: ', target_movies)
        other_user_ids = self.df[self.df['UserId'] != target_user_id]['UserId'].unique()  # target之外的其他用户
        # print('other_user_ids: ', other_user_ids)
        other_movies = [self.df[self.df['UserId'] == userid]['MovieId'].unique() for userid in other_user_ids]  # 其他用户点评过的电影
        # print('other_movies: ', other_movies)

        sim_list = [self._cosine_similarity(target_movies, movies) for movies in other_movies]
        sim_list = sorted(zip(other_user_ids, sim_list), key=lambda x: x[1], reverse=True)  # zip打包成元组: (userid, sim) 即userid 和 target_user_id的相似度
        return sim_list[:top_n]

    def get_top_n_movies(self, top_n_users, candidates_movies, top_n):
        """
        calculate interest of candidates movies and return top n movies
        计算候选电影的兴趣度并返回top n电影
        :param top_n_users:
        :param candidates_movies:
        :param top_n:
        :return: top_n_movies
        """
        top_n_user_data = [self.df[self.df['UserId'] == userid] for userid, _ in top_n_users]  # 得到之前相似前top_n个用户的数据, 一个df的list
        interest_list = []
        for movie in candidates_movies:  # 选一个候选的movies
            temp = []
            for user_data in top_n_user_data:  # 其他用户中选一个人, 看有没有评论过
                if movie in user_data['MovieId'].unique():  # 如果评论过了
                    temp.append(user_data[user_data['MovieId'] == movie]['Rating'].values[0] / 5)  # 就把第一次评分归一化后放进temp, 表示该用户对该电影的兴趣度
                else:  # 如果这个人没有评论过
                    temp.append(0)  # 那这个人对这个电影的兴趣度就是0
            # print(top_n_user_data)
            # print(type(top_n_user_data))
            interest = sum([top_n_users[i][1] * temp[i] for i in range(len(top_n_user_data))])  # 内循环之后, 其他所有人对这部电影的兴趣度的和, 就是target用户对这部电影的兴趣度
            interest_list.append((movie, interest))  # 存一下电影和target用户对电影的兴趣度
        interest_list = sorted(interest_list, key=lambda x: x[1], reverse=True)  # target对所有电影的兴趣度都得到了, 降序排列, 输出前top_n个
        return interest_list[:top_n]

    def get_candidates_movies(self, target_user_id):
        """
        get candidates movies
        获得候选的电影
        :param target_user_id:
        :return: candidates movies
        """
        target_movies = set(self.df[self.df['UserId'] == target_user_id]['MovieId'])  # 存一下target用户点评的过电影, 去重
        other_movies = set(self.df[self.df['UserId'] != target_user_id]['MovieId'])  # 存一下其他用户点评过的电影
        candidates_movies = list(other_movies - target_movies)  # 取其他用户点评过而且target用户没有点评过的movies作为候选
        return candidates_movies

    def calculate(self, target_user_id=1, top_n=10):
        """
        calculate top n movies for target user
        :param target_user_id:
        :param top_n:
        """
        top_n_users = self.get_top_n_users(target_user_id, top_n)
        # print(top_n_users)
        candidates_movies = self.get_candidates_movies(target_user_id)
        top_n_movies = self.get_top_n_movies(top_n_users, candidates_movies, top_n)
        return top_n_movies


