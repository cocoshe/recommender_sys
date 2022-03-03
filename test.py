from model.UserCF import UserCF
import numpy as np
from model.LFM import LFM

def test_UserCF():
    cf = UserCF()
    res = cf.calculate()
    print(res)


if __name__ == '__main__':
    # test_UserCF()
    # print(np.random.randn(10))
    users_item = dict({"cocoshe": [1, 1, 1, 3],
                       "qwe": [2, 2, 2],
                       "a": [2, 2, 2],
                       "s": [2, 2, 2],
                       "d": [2, 2, 2],
                       "e": [5]})

    model = LFM()
    P, Q = model.LatentFactorModel(users_item, 10, 1000, 0.01, 0.02)
    print('P: ', P)
    print('Q: ', Q)

    rank = model.Recommend('cocoshe', P, Q)
    print(rank)





