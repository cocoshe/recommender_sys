from tqdm import tqdm
import math


class ItemCF:
    def __init__(self, train):
        self.train = train

    @staticmethod
    def list2dict(list_):
        """
        list转为dic
        :param list_: [(UserID, ItemID), (UserID, ItemID), ...]
        :return: dic
        """
        users = list(set([i[0] for i in list_]))
        dic = dict(zip(users, [[] for i in range(len(users))]))
        for key, value in list_:
            dic[key].append(value)
        return dic

    def get_item_similarity(self):
        """
        计算物体相似度
        :param :
        :return:
        """
        c = dict()  # 记录电影两两之间共同喜欢的人数
        n = dict()  # 记录电影的打分人数
        print("计算物品相似度")
        for user, items in tqdm(self.train.items()):
            for item in items:
                n[item] += 1  # item 被看的次数 +1
                for item2 in items:
                    if item == item2:
                        continue
                    c[item][item2] += 1  # / math.log(1 + len(items) * 1.0)   # 加入惩罚

        w = dict()
        for i, related_items in c.items():
            for j, cij in related_items.items():
                w[i][j] = cij / math.sqrt(n[i] * n[j])
