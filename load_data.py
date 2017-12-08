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
    item_Num = len(item)
    for id in range(item_Num):
        key = item.iloc[id]
        if key not in item_dict.values():
            item_dict[id] = key
    return item_dict, item_Num

# order by time:
# 80% for training, 20% for testing
def splitData(path):
    data_df = pd.read_csv(pd)





def trainingData(path, item_dict):
    traindata = [
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    ]

    # ratingMat = []
    # timeMat = []
    # return ratingMat,timeMat

    return traindata


def test_Data(test_path, item_dict):

    test_df = pd.read_csv(test_path)

    test_data = []

    return test_data


if __name__ == '__main__':
    # a = 3
    # print(-a)
    trainingData()
