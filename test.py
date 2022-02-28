from model.UserCF import UserCF


def test_UserCF():
    cf = UserCF()
    res = cf.calculate()
    print(res)


if __name__ == '__main__':
    test_UserCF()
