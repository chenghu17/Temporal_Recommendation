import numpy as np
import pandas as pd
import time

if __name__ == '__main__':
    # trainPath = 'data_FM/train.tsv'

    validationPath = 'data_LastFM/validation.tsv'
    trainPath = 'data_LastFM/train.tsv'
    testPath = 'data_LastFM/test.tsv'
    # validationPath = 'data_Epinions/validation.csv'
    # validationPath = 'data_FineFoods/validation.csv'
    # df_train = pd.read_csv(trainPath, sep='\t', header=None)
    # df_validation = pd.read_csv(validationPath, sep='\t', header=None)
    df_test = pd.read_csv(testPath, sep='\t', header=None)
    # print(df_validation.head(100))
    # userSet = list(df_validation[0].drop_duplicates())
    # print(len(userSet))
    maxtime = max(df_test[3])
    mintime = min(df_test[3])


    # print(userSet)
    # a = set([1,2,2,3,4,5])
    # b = set([1])
    # print(len(a&b))

    # itemMat_18 = np.loadtxt('evolution18/itemMat30.txt')
    # userMat_18 = np.loadtxt('evolution18/userMat30.txt')
    # itemMat_stand = np.loadtxt('evolution_standard/itemMat30.txt')
    # userMat_stand = np.loadtxt('evolution_standard/userMat30.txt')


    st_max = time.localtime(maxtime)
    st_min = time.localtime(mintime)
    print(time.strftime('%Y-%m-%d %H:%M:%S', st_max))
    print(time.strftime('%Y-%m-%d %H:%M:%S', st_min))

    # st = time.strptime('2004-12-31 00:00:00','%Y-%m-%d %H:%M:%S')
    # print(time.mktime(st))

    # 观察precision值高的用户特征
    # userList = [15033, 20996, 16797, 3493, 21004, 4246, 5075, 13968]

    # for i in range(6):
    #     print(i)
    # userMat = [n for n in range(6)]
    # print(userMat)


    # userList = [215,1325,2877,4524,6050]
    # df_validation = pd.read_csv('data_Netflix/train.tsv', sep='\t', header=None)
    # file = open('observe/observe_data_2.txt', 'a')
    # for i in range(len(df_validation)):
    #     user = df_validation.iat[i, 0]
    #     if user in userList:
    #         linestr = str(df_validation.iat[i, 0]) + ":" + str(df_validation.iat[i, 3]) + '\n'
    #         file.write(linestr)
    # file.close()
    #
    # f = open('observe/observe_data_2.txt', 'r')
    # f1 = open('observe/user_observe_2.txt', 'a')
    # f2 = open('observe/item_observe_2.txt', 'a')
    #
    # while 1:
    #     lines = f.readlines(1000)
    #     if not lines:
    #         break
    #     for line in lines:
    #         content = line.strip('\n').split(':')
    #         user = content[0]
    #         timestamp = content[1]
    #         f1.write(user+'\n')
    #         f2.write(timestamp+'\n')
