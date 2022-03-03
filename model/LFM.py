import math
import numpy as np

class LFM:
    def __init__(self):
        pass

    def RandomSelectNegativeSample(self, items, items_pool):
        """
        随机选择一些负样本: 找热门但是不感兴趣
        返回的ret是一个dict, 其中用户有过行为的item对应的value都为1,
        然后有与数量相当的items_pool中但是没有行为的负样本value为0
        :param items_pool: 候选物品的列表
        :param items: 用户已经有过行为的物品的集合
        :return: 筛选出不在用户行为物品集合中, 但是很热门(在候选物品中频率高)的物品集合
        """
        ret = dict()  # 维护用户有过行为的物品, 1为正样本, 0为负样本
        for i in items:
            ret[i] = 1
        cnt = 0
        # 需要注意的是，代码中的items_pool 为一个存储着所有items 的list，
        # 因为有重复，所以每个物品被选中的概率和它出现的频率成正比。这就完成了选取热门物品的部分。
        for i in range(0, len(items_pool) * 3):
            item = items_pool[np.random.randint(0, len(items_pool))]
            if item in ret.keys():
                continue  # 热门物品如果在用户行为中, 直接continue
            cnt += 1  # 热门物品若不在用户行为中, 负样本加一
            ret[item] = 0  # 记录value为0则是负样本
            if cnt > len(items):  # 控制负样本数量和正样本数量相等
                break
        return ret

    def LatentFactorModel(self, user_item, F, N, alpha, flambda):
        users_pool = []
        items_pool = []
        for user, items in user_item.items():
            users_pool.append(user)
            for item in items:
                items_pool.append(item)
        [P, Q] = self.InitModel(users_pool, items_pool, F)
        for step in range(0, N):
            for user, items in user_item.items():
                samples = self.RandomSelectNegativeSample(items, items_pool)
                for item, rui in samples.items():
                    eui = rui - self.Predict(P[user], Q[item])
                    P[user] += alpha * (eui * Q[item] - flambda * P[user])
                    Q[item] += alpha * (eui * P[user] - flambda * Q[item])
            alpha *= 0.9

        return P, Q


    def Predict(self, Puser, Qitem):
        """
        进行前向推理
        :param Puser:
        :param Qitem:
        :return:
        """
        res = np.sum(Puser * Qitem)
        res = 1.0 / (1 + math.exp(-res))
        return res

    def InitModel(self, users_pool, items_pool, F):
        Q = dict()
        P = dict()
        users_pool = set(users_pool)
        items_pool = set(items_pool)
        for user in users_pool:
            P[user] = np.random.rand(F)
        for item in items_pool:
            Q[item] = np.random.rand(F)
        return P, Q

    def Recommend(self, user, P, Q):
        rank = dict()
        for item in Q.keys():
            rank[item] = np.sum(P[user] * Q[item])
        # ATTENTION! 相比于Predict, 这里没有归一化
        return sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:10]


