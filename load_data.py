# ratingMat : userId,movieId,rating,timestamp
# create rating-matrix and time-matrix
# ratingMat : user-item rating or  {0,1}
# timeMat : user-item timestamp, {1,2,3,4,5...}

import numpy as np
import pandas as pd


# For dataset of movieLen
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
    for index in range(1, userNum + 1):
        user_dict[index] = index - 1
    train.close()
    test.close()
    return userNum, user_dict


def trainingData(trainpath, item_dict, user_dict, userNum, itemNum):
    ratingMat = np.zeros((userNum, itemNum))
    train_df = pd.read_csv(trainpath, header=None)
    train_df = train_df[[0, 1]]
    length = len(train_df)
    for i in range(length):
        user_index = train_df.iat[i, 0]
        user_id = user_dict[user_index]
        item_index = train_df.iat[i, 1]
        item_id = item_dict[item_index]
        ratingMat[user_id][item_id] = 1
    timeMat = []

    return ratingMat, timeMat


def testingData(testpath, item_dict, user_dict, userNum, itemNum):
    testMat = np.zeros((userNum, itemNum))
    train_df = pd.read_csv(testpath, header=None)
    train_df = train_df[[0, 1]]
    length = len(train_df)
    # Y_True = np.ones(length)
    for i in range(length):
        user_index = train_df.iat[i, 0]
        user_id = user_dict[user_index]
        item_index = train_df.iat[i, 1]
        item_id = item_dict[item_index]
        testMat[user_id][item_id] = 1
    # return testMat,Y_True
    return testMat


# For dataset of FM-1K
def itemSet(trainPath):
    itemset = set()
    df = pd.read_csv(trainPath, sep='\t', header=None)
    for i in range(len(df)):
        itemId = df.iat[i, 1]
        itemset.add(itemId)
    return itemset


# for data_FineFoods

# productId、userId、score、time
# --> A1D87F6ZCVE5NK B00813GRG4 1.0 1346976000
def captureData(path, capPath):
    sourceFile = open(path, 'r', encoding='ISO-8859-1')
    captureFile = open(capPath, 'w')
    for line in sourceFile.readlines():
        # print(line)
        line = line.strip()
        if not len(line):
            continue
        if 'product/productId:' in line:
            line = line.lstrip('product/productId:').strip()
            captureFile.write(line)
            continue
        if 'review/userId:' in line:
            line = line.lstrip('review/userId:').strip()
            captureFile.write('\t' + line + '\t' + str(1.0))
            continue
        if 'review/time' in line:
            line = line.lstrip('review/time:').strip()
            captureFile.writelines('\t' + line + '\n')


#
# if __name__ == '__main__':
#     path = 'data_FineFoods/finefoods.txt'
#     capPath = 'data_FineFoods/data.csv'
#     captureData(path,capPath)

if __name__ == '__main__':
    df = pd.read_csv('data_FineFoods/data.csv', sep='\t', header=None)
    userSet = set(df[1])
    count = 0
    for user in userSet:
        df_tmp = df[df[1]==user]
        # print(user)
        if len(df_tmp)>20:
            count += 1
    print(count)
    print(len(userSet))

    max_Timestamp = pd.Series.max(df[3])
    min_Timestamp = pd.Series.min(df[3])
    print(max_Timestamp)
    print(min_Timestamp)
