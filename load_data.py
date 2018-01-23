# ratingMat : userId,movieId,rating,timestamp
# create rating-matrix and time-matrix
# ratingMat : user-item rating or  {0,1}
# timeMat : user-item timestamp, {1,2,3,4,5...}

import numpy as np
import pandas as pd
import time


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


# if __name__ == '__main__':
#     path = 'data_FineFoods/finefoods.txt'
#     capPath = 'data_FineFoods/data.csv'
#     captureData(path,capPath)


# for data_Epinions
def getData(path, getPath):
    sourceFile = open(path, 'r', encoding='ISO-8859-1')
    captureFile = open(getPath, 'w')
    for line in sourceFile.readlines():
        # print(line)
        line = line.strip()
        line_list = line.split()
        if len(line_list) < 4:
            continue
        if line_list[3].isdigit() and int(line_list[3]) > 10000000:
            captureFile.writelines(line_list[0] + '\t' + line_list[1] + '\t' + str(1.0) + '\t' + line_list[3] + '\n')


# getData('data_Epinions/epinions.txt', 'data_Epinions/data.csv')


# get user data who has rated more then 20
def rebuildData(rootPath, path, k):
    rootPath = rootPath + '/user.txt'
    userSetFile = open(rootPath, 'w')
    df = pd.read_csv(path, sep='\t', header=None)
    userSet = set(df[1])

    # userSet_over20 = set()
    for user in userSet:
        df_tmp = df[df[1] == user]
        if len(df_tmp) >= k:
            print(user)
            userSetFile.writelines(user + '\n')
            # userSet_over20.add(user)


# 1、保留打分数超过k个的用户
# if __name__ == '__main__':
#
#     # threshold
#     k = 10
#
#     # rootPath = 'data_FineFoods'
#     # path = 'data_FineFoods/data.csv'
#
#     rootPath = 'data_Epinions'
#     path = 'data_Epinions/data.csv'
#
#     rebuildData(rootPath, path, k)

# 2、将user映射到数字上
# if __name__=='__main__':
#
#
#     # user_file = open('data_Epinions/user.txt','r')
#     # data_file = open('data_Epinions/data.csv','r')
#     # getData = open('data_Epinions/data_get.csv','w')
#
#     user_file = open('data_FineFoods/user.txt', 'r')
#     data_file = open('data_FineFoods/data.csv', 'r')
#     getData = open('data_FineFoods/data_get.csv', 'w')
#
#     userset = set()
#     count = 1
#     userdict = dict()
#
#     for line in user_file.readlines():
#         line = line.strip()
#         userdict[line] = count
#         count += 1
#
#     for line in data_file.readlines():
#         line = line.strip()
#         line_list = line.split()
#         if line_list[1] in userdict.keys():
#             getData.writelines(line_list[0] + '\t' + str(userdict[line_list[1]]) + '\t' + line_list[2] + '\t' + line_list[3] + '\n')


# 3、对每个打分数据的时间跨度进行计算，判断是否符合要求，将item名映射到数字
# if __name__ == '__main__':
#
#     df = pd.read_csv('data_Epinions/data_get.csv', sep='\t', header=None)
#     data_file = open('data_Epinions/data_get.csv', 'r')
#     getfile = open('data_Epinions/data_sum.csv', 'w')
#     itemList = list(df[0].drop_duplicates())
#     count = 1
#     itemdict = dict()
#     for item in itemList:
#         itemdict[item] = count
#         count += 1
#     for line in data_file.readlines():
#         line = line.strip()
#         line_list = line.split()
#         if line_list[0] in itemdict.keys():
#             getfile.writelines(
#                 line_list[1] + '\t' + str(itemdict[line_list[0]]) + '\t' + line_list[2] + '\t' + line_list[3] + '\n')
#     data_file.close()
#     getfile.close()

# 5、划分train、validation、test数据集
# if __name__ == '__main__':
#     # df = pd.read_csv('data_FineFoods/data_sum.csv', sep='\t', header=None)
#     df = pd.read_csv('data_Epinions/data_sum.csv', sep='\t', header=None)
#
#     max_Timestamp = pd.Series.max(df[3])
#     min_Timestamp = pd.Series.min(df[3])
#     middle_Timestamp = (max_Timestamp - min_Timestamp) * 0.8 + min_Timestamp
#
#     df_train = df[(df[3] >= min_Timestamp) & (df[3] < middle_Timestamp)]
#     df_train.to_csv('data_Epinions/train.csv', sep='\t', header=None, index=False)
#
#     df_other = df[(df[3] >= middle_Timestamp) & (df[3] <= max_Timestamp)]
#     df_validation = df_other.sample(frac=0.5)
#     df_tmp = df_other.append(df_validation)
#     df_test = df_tmp.drop_duplicates(keep=False)
#     df_validation.to_csv('data_Epinions/validation.csv', sep='\t', header=None, index=False)
#     df_test .to_csv('data_Epinions/test.csv', sep='\t', header=None, index=False)

# 提取item数据item.txt
# if __name__=='__main__':
#     path = 'data_Epinions/data_get.csv'
#     rootPath = 'data_Epinions/item.txt'
#     itemSetFile = open(rootPath, 'w')
#     df = pd.read_csv(path, sep='\t', header=None)
#     itemSet = set(df[0])
#
#     # userSet_over20 = set()
#     for item in itemSet:
#         itemSetFile.writelines(item + '\n')
