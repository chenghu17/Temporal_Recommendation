# ratingMat : userId,movieId,rating,timestamp
# create rating-matrix and time-matrix
# ratingMat : user-item rating or  {0,1}
# timeMat : user-item timestamp, {1,2,3,4,5...}

import numpy as np
import pandas as pd
import time
import os


# For data_MovieLen of movieLen
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


# For data_MovieLen of FM-1K
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
def getData(path, datapath):
    sourceFile = open(path, 'r')
    resultFile = open(datapath, 'a')

    while 1:
        lines = sourceFile.readlines(100000)
        if not lines:
            break
        for line in lines:
            line = line.strip()
            line_list = line.split('\t')
            if int(line_list[2]) <= 2:
                continue
            # convert line_list[4] to timestamp
            timestamp = 0
            if line_list[5] == '':
                timestamp = changetime(line_list[4])
            else:
                timestamp = changetime(line_list[5])
            content = line_list[1] + '\t' + line_list[0] + '\t' + str(1) + '\t' + str(timestamp) + '\n'
            resultFile.write(content)


def changetime(time_string):
    # change yyyy-mm-dd or to yyyy/mm/dd timestamp
    time_string = time_string + ' 00/00/00'
    st = time.strptime(time_string, '%Y/%m/%d %H/%M/%S')
    timestamp = str(time.mktime(st)).split('.')[0]
    return timestamp


def append():
    df1 = pd.read_csv('data_Netflix/test_one.csv', header=None, sep='\t')
    df2 = pd.read_csv('data_Netflix/test_two.csv', header=None, sep='\t')
    df3 = pd.read_csv('data_Netflix/test_three.csv', header=None, sep='\t')
    df4 = pd.read_csv('data_Netflix/test_four.csv', header=None, sep='\t')
    result = df1.append([df2, df3, df4])
    result.to_csv('data_Netflix/test.csv', header=None, index=False, sep='\t')


# reserve value > 400
def analyze_user(datapath, objectpath):
    # data_df = pd.read_csv(datapath)
    # # f = open(objectpath,'a')
    # user = list(data_df['userId'].drop_duplicates())
    # userNum = len(user)
    # print(userNum)
    # count = 0
    # for user_id in user:
    #     split_data = data_df[data_df['userId'] == user_id]
    #     if len(split_data)>200:
    #         count += 1
    # print(count)
    df = pd.read_csv(datapath, sep='\t', header=None)
    user_id = dict()
    for i in range(len(df)):
        userId = df.iat[i, 0]
        if userId not in user_id.keys():
            user_id[userId] = 1
        else:
            user_id[userId] += 1
    user_id_over20 = {key: value for key, value in user_id.items() if value >= 20}
    # print(len(user_id_over20.keys()))
    user = list(user_id_over20.keys())
    data_user = df[0].isin(user)
    data = df[data_user]
    data.to_csv(objectpath, header=None, sep='\t', index=False)

def analyze_item(datapath, objectpath):
    df = pd.read_csv(datapath, sep='\t', header=None)
    item_id = dict()
    for i in range(len(df)):
        itemId = df.iat[i, 1]
        if itemId not in item_id.keys():
            item_id[itemId] = 1
        else:
            item_id[itemId] += 1
    item_id_over80 = {key: value for key, value in item_id.items() if value >= 100}
    # print(len(user_id_over20.keys()))
    item = list(item_id_over80.keys())
    data_item = df[1].isin(item)
    data = df[data_item]
    data.to_csv(objectpath, header=None, sep='\t', index=False)


def gettest_users(userpath, user_number):
    file = open(userpath, 'a')
    for i in range(user_number):
        line = str(i) + '\n'
        file.write(line)
    file.close()


def updateUserId_movie(filepath, resultpath):
    df = pd.read_csv(filepath, header=None, sep='\t')
    user_id = dict()
    count = 0
    for i in range(len(df)):
        userId = str(df.iat[i, 0])
        if userId not in user_id.keys():
            user_id[userId] = count
            df.iat[i, 0] = count
            count += 1
        else:
            df.iat[i, 0] = user_id[userId]
    print('userid:', count)
    df.to_csv(resultpath, sep='\t', header=None, index=False)


def updateItemId_movie(resultpath, finalpath):
    df = pd.read_csv(resultpath, header=None, sep='\t')
    item_id = dict()
    count = 0
    for i in range(len(df)):
        itemId = str(df.iat[i, 1])
        if itemId not in item_id.keys():
            item_id[itemId] = count
            df.iat[i, 1] = count
            count += 1
        else:
            df.iat[i, 1] = item_id[itemId]
    print('itemid:', count)
    df.to_csv(finalpath, sep='\t', header=None, index=False)
    print(count)
    del df


def getMiddelTime(finalpath):
    df = pd.read_csv(finalpath, header=None, sep='\t')
    max_Timestamp = pd.Series.max(df[3])
    min_Timestamp = pd.Series.min(df[3])
    print('max_Timestamp:', max_Timestamp)
    print('min_Timestamp', min_Timestamp)
    print('usernumber:', len(df[0].drop_duplicates()))
    print('itemnumber:', len(df[1].drop_duplicates()))


# split train、validation、test data set according to timestamp
def splitTimeData_movie(finalpath):
    df = pd.read_csv(finalpath, header=None, sep='\t')
    # 12月： 31104000
    max_Timestamp = 1427784002
    min_Timestamp = 824835410
    middle_Timestamp = 1396680002
    df_train = df[(df[3] >= min_Timestamp) & (df[3] < middle_Timestamp)]
    df_train.to_csv('data_Epinions/train.tsv', sep='\t', header=None, index=False)
    df_other = df[(df[3] >= middle_Timestamp) & (df[3] <= max_Timestamp)]
    df_validation = df_other.sample(frac=0.5)
    df_tmp = df_other.append(df_validation)
    df_test = df_tmp.drop_duplicates(keep=False)
    df_validation.to_csv('data_Epinions/validation.tsv', sep='\t', header=None, index=False)
    df_test.to_csv('data_Epinions/test.tsv', sep='\t', header=None, index=False)


# check the number of users in validation and test with the sum of user numbers
def check(validation, test):
    df_val = pd.read_csv(validation, header=None, sep='\t')
    df_test = pd.read_csv(test, header=None, sep='\t')
    print(len(df_val[0].drop_duplicates()))
    print(len(df_test[0].drop_duplicates()))


if __name__ == '__main__':
    # userpath = 'data_MovieLen/test_users.tsv'
    # exists = os.path.exists(userpath)
    # if not exists:
    #     gettest_users(userpath, n)

    path = 'data_Epinions/rating.txt'
    datapath = 'data_Epinions/ratings.csv'

    exists = os.path.exists(datapath)
    if not exists:
        getData(path, datapath)
    #
    objectpath = 'data_Epinions/all_1.tsv'
    exists = os.path.exists(objectpath)
    if not exists:
        analyze_user(datapath, objectpath)
    #
    datapath = 'data_Epinions/all_1.tsv'
    objectpath = 'data_Epinions/all_2.tsv'
    exists = os.path.exists(objectpath)
    if not exists:
        analyze_item(datapath, objectpath)
    #
    resultpath = 'data_Epinions/result.tsv'
    exists = os.path.exists(resultpath)
    if not exists:
        updateUserId_movie(objectpath, resultpath)
    #
    finalpath = 'data_Epinions/final.tsv'
    exists = os.path.exists(finalpath)
    if not exists:
        updateItemId_movie(resultpath, finalpath)
    #
    getMiddelTime(finalpath)
    # # split train、validation、test

    # splitTimeData_movie(finalpath)

    # validationpath = 'data_Netflix/validation.tsv'
    # testpath = 'data_Netflix/test.tsv'
    # check(validationpath, testpath)

# if __name__ == '__main__':

# rebuildNetData(filepath1,'one')
# filepath = 'data_Netflix/data_three_stan.csv'
# rebuildNetData(filepath)
# filepath1 = 'data_Netflix/data_one.csv'
# filepath2 = 'data_Netflix/data_two.csv'
# filepath3 = 'data_Netflix/data_three.csv'
# filepath4 = 'data_Netflix/data_four.csv'
# append(filepath1, filepath2)
# append(filepath3, filepath4)
# filepath = 'data_Netflix/data_one_two.csv'
# filepath2 = 'data_Netflix/data_three_four.csv'
# changetime(filepath)
# splitTimeData(filepath)
# append()

# train = 'data_Netflix/train.csv'
# validation = 'data_Netflix/validation.csv'
# test = 'data_Netflix/test.csv'
# train_stan = 'data_Netflix/train_stan.csv'
# validation_stan = 'data_Netflix/validation_stan.csv'
# test_stan = 'data_Netflix/test_stan.csv'
# trainmiddle = 'data_Netflix/trainmiddle.csv'
# validationmiddle = 'data_Netflix/validationmiddle.csv'
# testmiddle = 'data_Netflix/testmiddle.csv'
# # # # update user id
# updateUserId(train, validation, test, 'stan')
# # # # reserve value > 500
# clearData(train_stan, validation_stan, test_stan)
# # # # update user id
# updateUserId(trainmiddle, validationmiddle, testmiddle, 'final')
# # # # update item id
# trainfinal = 'data_Netflix/train_final.csv'
# validationfinal = 'data_Netflix/validation_final.csv'
# testfinal = 'data_Netflix/test_final.csv'
# updateUserId(trainfinal, validationfinal, testfinal, 'learn')
# # # # generator test_users.tsv file
# gettest_users(23928)
#
# # convert 100028384.0 to 100028384
# f = open('data_Netflix/test_learn.csv', 'r')
# f_new = open('data_Netflix/test.tsv', 'a')
# while 1:
#     lines = f.readlines(10000)
#     if not lines:
#         break
#     for line in lines:
#         stamp = (line.strip('\n').split('\t')[3]).split('.')[0]
#         line_new = line.strip('\n').split('\t')[0] + '\t' + line.strip('\n').split('\t')[1] + '\t' + \
#                    line.strip('\n').split('\t')[2] + '\t' + stamp + '\n'
#         f_new.write(line_new)


# df = pd.read_csv('data_Netflix/test_learn.csv', header=None, sep='\t')
# for i in range(len(df)):
#     df.iloc[i, 3] = str(int(df.iloc[i, 3]))
# df.to_csv('data_Netflix/test_learn_new.csv', header=None, sep='\t', index=False)

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
