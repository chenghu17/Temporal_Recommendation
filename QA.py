# author : hucheng
# This part is for question & answer

import pandas as pd
import numpy as np
import time
import random
import evolution


def itemDict(path):
    item_dict = dict()
    item_data = pd.read_csv(path)
    item = item_data.loc[:, 'movieId']
    itemNum = len(item)
    for id in range(itemNum):
        key = item.iloc[id]
        if key not in item_dict.values():
            item_dict[id] = key
    # print(item)
    # print(len(item))
    # print(item.iloc[0])
    return item_dict, itemNum


def splitData(datapath, trianpath, testpath):
    train = open(trianpath, 'a')
    test = open(testpath, 'a')
    data_df = pd.read_csv(datapath)
    # user = data_df.loc[:, 'userId']
    user = list(data_df['userId'].drop_duplicates())

    userNum = len(user)

    for user_id in user:
        split_data = data_df[data_df['userId'] == user_id]
        minIndex = min(split_data.index.values)
        maxIndex = max(split_data.index.values)
        middleNum = int((maxIndex - minIndex) * 0.8)
        train_data = split_data.iloc[:middleNum, :]
        train_data.to_csv(train, header=False, index=False)
        test_data = split_data.iloc[middleNum:, :]
        test_data.to_csv(test, header=False, index=False)

    return userNum


def prediction(self, userMat, itemMat):
    pred = list()
    true = list()
    testMat = self.testMat
    for user_id in range(len(testMat)):
        negativeList = list(np.where(testMat[user_id] == 0)[0])
        # positive
        for item_id in np.where(testMat[user_id] == 1)[0]:
            true.append(1)
            Pu = userMat[user_id]
            Qi = itemMat[item_id]
            Y = np.dot(Pu, Qi)
            pred.append(Y)
            # negative
            nega_Sampling = random.sample(negativeList, 1)
            for nega_item_id in nega_Sampling:
                true.append(0)
                Qj = itemMat[nega_item_id]
                Y = np.dot(Pu, Qj)
                pred.append(Y)
    Y_True = np.array(true)
    Y_Pred = np.array(pred)
    return Y_True, Y_Pred


def trainingData(trainpath, item_dict, userNum, itemNum):
    trainMat = np.zeros((userNum, itemNum))
    train_df = pd.read_csv(trainpath, header=None)
    train_df = train_df[[0, 1]]
    length = len(train_df)
    for i in range(length):
        user_id = train_df.iat[i, 0]
        item_index = train_df.iat[i, 1]
        item_id = item_dict[item_index]
        trainMat[user_id][item_id] = 1

    return trainMat


def timestampTOtime(datapath):
    f = open('dataset/time.txt', 'a')
    train_df = pd.read_csv(datapath)
    length = len(train_df)
    for i in range(length):
        userId = train_df.iat[i, 0]
        itemId = train_df.iat[i, 1]
        timestamp = train_df.iat[i, 3]
        st = time.localtime(timestamp)
        datatime = time.strftime('%Y-%m-%d %H:%M:%S', st)
        datatime = str(userId) + ' ' + str(itemId) + ' ' + datatime
        f.write(datatime + '\n')
    f.close()


def prediction(validationPath, userMat, itemMat, itemSet):
    pred = list()
    true = list()
    df_validation = pd.read_csv(validationPath, sep='\t', header=None)
    for u in range(len(df_validation[0])):
        df_tmp = df_validation[df_validation[0] == u]
        item_tmp = set(df_tmp[1])
        userId = df_validation.iat[u, 0]
        itemId = df_validation.iat[u, 1]
        Pu = userMat[userId]
        Qi = itemMat[itemId]
        true.append(1)
        Y = np.dot(Pu, Qi)
        pred.append(Y)
        negative_item_set = itemSet - item_tmp
        nega_item_id = random.choice(list(negative_item_set))
        Qk = itemMat[nega_item_id]
        true.append(0)
        Y = np.dot(Pu, Qk)
        pred.append(Y)
    Y_True = np.array(true)
    Y_Pred = np.array(pred)
    return Y_True, Y_Pred

def itemSet(trainPath):
    itemset= set()
    df = pd.read_csv(trainPath, sep='\t', header=None)
    for i in range(len(df)):
        itemId = df.iat[i,1]
        itemset.add(itemId)
    return itemset


if __name__ == '__main__':
    title = 'Talk is cheap, Show me the code'

    # function1：extract column of user、timestamp、item
    # df = pd.read_csv('dataset/dataset_full.tsv',sep='\t', header=None, error_bad_lines=False)
    # df = df[[0,1,4]]
    # df = df.dropna()
    # df.to_csv('dataset/dataset_tmp.csv', header=False, index=False)
    # -------------------------------

    # function2：transform user_id to number
    # df = pd.read_csv('dataset/userid.tsv',sep='\t', header=None)
    # df = df[[0]]
    # userid_dict = dict()
    # for i in range(len(df)):
    #     userid = df.iat[i,0]
    #     userid_dict[userid] = i
    # df_data = pd.read_csv('dataset/dataset_tmp.csv', header=None)
    # for line in range(len(df_data)):
    #     userid = df_data.iat[line,0]
    #     df_data.iat[line, 0] = userid_dict[userid]
    # df_data.to_csv('dataset/dataset_tmp.csv', header=False, index=False)
    # --------------------------------

    # function3：transform time to timestamp
    # df = pd.read_csv('dataset/dataset_tmp.csv',header=None)
    # for line in range(len(df)):
    #     timeStr = df.iat[line,1]
    #     # timeStr= '2008-06-09T21:08:37Z'
    #     timeStr = timeStr[:10]+' '+timeStr[11:19]
    #     timeStr = time.strptime(timeStr,'%Y-%m-%d %H:%M:%S')
    #     timestamp = time.mktime(timeStr)
    #     df.iat[line, 1] = timestamp
    # df.to_csv('dataset/dataset_tmp.csv', header=False, index=False)
    # --------------------------------

    # function：calculate the number of music, and transform itemId to number
    # df = pd.read_csv('dataset/dataset_tmp.csv',header=None)
    # item_dict = dict()
    # index = 0
    # for line in range(len(df)):
    #     itemId = df.iat[line,2]
    #     if itemId not in item_dict.keys():
    #         item_dict[itemId] = index
    #         index += 1
    # print(index)
    # --------------------------------

    # datapath = 'dataset/ratings.csv'
    # timestampTOtime(datapath)
    # --------------------------------

    # path = 'dataset/movies.csv'
    # item_dict, itemNum = itemDict(path)
    # datapath = 'dataset/ratings.csv'
    # trainpath = 'dataset/train.csv'
    # testpath = 'dataset/test.csv'
    # userNum = splitData(datapath, trainpath, testpath)
    # trainingData('dataset/train.csv', item_dict, userNum,itemNum)
    # --------------------------------

    # traindata = [
    #     [0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    #     [0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    # ]
    # traindata = np.array(traindata)
    # timeMat = []
    # userNum = traindata.shape[0]
    # itemNum = traindata.shape[1]
    # --------------------------------

    # t = 18
    # timestamp = t * 30 * 24 * 3600
    # trainPath = 'data/train.tsv'
    # df_train = pd.read_csv(trainPath, sep='\t', header=None)
    # maxnum = pd.Series.max(df_train[3])
    # minnum = pd.Series.min(df_train[3])
    # distance = maxnum - minnum
    # time_Step = int(distance / timestamp + 1)
    # print(time_Step)
    # sum = 0
    # for i in range(1, time_Step+1):
    #     level_up = int(minnum + i * timestamp)
    #     level_down = level_up - timestamp
    #     # print(df_train[3])
    #     df = df_train[(df_train[3] < level_up) & (df_train[3] >= level_down)]
    #     userSet = set(df[0])
    #     for userId in userSet:
    #         df_tmp = df[df[0]==userId]
    #     length = len(df)
    #     sum += length
    #     print(len(df))
    # print(sum)

    # function : compute item set everyone has rated in each interval
    # t = 18
    # userNum = 1000
    # timestamp = t * 30 * 24 * 3600
    # trainPath = 'data/train.tsv'
    # df_train = pd.read_csv(trainPath, sep='\t', header=None)
    # max_Timestamp = pd.Series.max(df_train[3])
    # min_Timestamp = pd.Series.min(df_train[3])
    # distance = max_Timestamp - min_Timestamp
    # time_Step = int(distance / timestamp + 1)
    # user_item_time = [0 for n in range(time_Step + 1)]
    # for t in range(1, time_Step + 1):
    #     level_down_current = min_Timestamp + (t - 1) * timestamp
    #     level_up_current = level_down_current + timestamp
    #     df_interval_current = df_train[(df_train[3] >= level_down_current) & (df_train[3] < level_up_current)]
    #     # 单个时间间隔中所包含的user集合
    #     userSet = set(df_interval_current[0])
    #     # 初始化包含单个时间间隔中每个用户所打过分的list
    #     user_item_List = [0 for n in range(userNum)]
    #     for userId in userSet:
    #         df_tmp = df_interval_current[df_interval_current[0]==userId]
    #         item_tmp = set(df_tmp[1])
    #         user_item_List[userId] = item_tmp
    #     user_item_time[t] = user_item_List

    # trainPath = 'data/train.tsv'
    # validationPath = 'data/validation.tsv'
    # itemMat = np.loadtxt('evolution/itemMat0.txt')
    # userMat = np.loadtxt('evolution/userMat0.txt')
    # itemset = itemSet(trainPath)
    #
    # Y_True, Y_Pred = prediction(validationPath,userMat,itemMat,itemset)
    # auc = evolution.AUC(Y_True, Y_Pred)
    # print('AUC:', auc)

    f = open('evolution_standard/auc.txt','a')
    for i in range(10):
        f.write(str(i)+'\n')
    f.close()

    # print(type(itemMat))
    # for step in range(10):
    #     userMat_name = 'userMat' + str(step) + '.txt'
    #     print(userMat_name)
