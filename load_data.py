# ratingMat : userId,movieId,rating,timestamp
# create rating-matrix and time-matrix
# ratingMat : user-item rating or  {0,1}
# timeMat : user-item timestamp, {1,2,3,4,5...}

import numpy as  np
import random
import pandas as pd


def itemDict(path):
    item_dict = dict()
    item_data = pd.read_csv(path)
    item = item_data.loc[:, 'movieId']
    itemNum = len(item)
    for index in range(itemNum):
        key = item.iloc[index]
        if key not in item_dict.keys():
            item_dict[key] = index
    return item_dict, itemNum


# order by time:
# 80% for training, 20% for testing
def splitData(datapath, trianpath, testpath):
    user_dict = dict()
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
    for index in range(1,userNum+1):
        user_dict[index] = index-1
    return userNum,user_dict


def trainingData(trainpath, item_dict,user_dict, userNum, itemNum):
    trainMat = np.zeros((userNum, itemNum))
    train_df = pd.read_csv(trainpath, header=None)
    train_df = train_df[[0, 1]]
    length = len(train_df)
    for i in range(length):
        user_index = train_df.iat[i, 0]
        user_id = user_dict[user_index]
        item_index = train_df.iat[i, 1]
        item_id = item_dict[item_index]
        trainMat[user_id][item_id] = 1
    timeMat = []

    return trainMat, timeMat


def test_Data(testpath, item_dict):
    test_df = pd.read_csv(testpath)

    test_data = []

    return test_data


if __name__ == '__main__':
    # a = 3
    # print(-a)
    trainingData()
