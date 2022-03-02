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
    users_item = dict({"cocoshe": ["aa", "bb", "cc"],
                       "qwe": ['a', 'b', 'c']})

    model = LFM()
    P, Q = model.LatentFactorModel(users_item, 10, 20, 0.01, 0.02)
    print('P: ', P)
    print('Q: ', Q)

    rank = model.Recommend('cocoshe', P, Q)
    print(rank)





