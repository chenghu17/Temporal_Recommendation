# author : hucheng
# This part is for question & answer

import pandas as pd
import numpy as np
import time


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
    f = open('dataset/time.txt','a')
    train_df = pd.read_csv(datapath)
    length = len(train_df)
    for i in range(length):

        timestamp = train_df.iat[i,3]
        st = time.localtime(timestamp)
        datatime = time.strftime('%Y-%m-%d %H:%M:%S', st)
        f.write(datatime+'\n')
    f.close()

if __name__ == '__main__':
    title = 'Talk is cheap, Show me the code'

    #--------------------------------
    # datapath = 'dataset/ratings.csv'
    # timestampTOtime(datapath)

    # path = 'dataset/movies.csv'
    # item_dict, itemNum = itemDict(path)
    #
    # datapath = 'dataset/ratings.csv'
    # trainpath = 'dataset/train.csv'
    # testpath = 'dataset/test.csv'
    # userNum = splitData(datapath, trainpath, testpath)
    #
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
