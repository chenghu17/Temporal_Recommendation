# ratingMat : userId,movieId,rating,timestamp
# create rating-matrix and time-matrix
# ratingMat : user-item rating or  {0,1}
# timeMat : user-item timestamp, {1,2,3,4,5...}

import numpy as np
import pandas as pd
import time
import json
import csv


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


# for netflix
def rebuildNetData(filepath):
    column = ['item', 'user', 'rating', 'timestamp']
    df = pd.read_csv(filepath, header=None, names=column)
    df_user = df.user
    df = df.drop('user', axis=1)
    # df = df.drop('user', axis=1)
    df.insert(0, 'user', df_user)
    df.rating = 1
    # df.insert(0,'user',df_user)
    df.to_csv('data_Netflix/data_four.csv', header=None, sep='\t', index=False)


def changetime(filepath):
    df = pd.read_csv(filepath, header=None, sep='\t')
    # change yyyy-mm-dd to timestamp
    for i in range(len(df[3])):
        time_string = df.iat[i, 3] + ' 00:00:00'
        st = time.strptime(time_string, '%Y-%m-%d %H:%M:%S')
        timestamp = time.mktime(st)
        df.iat[i, 3] = timestamp
    df.to_csv('data_Netflix/data_four_stan.csv', header=None, sep='\t', index=False)


# split train、validation、test data set according to timestamp
def splitTimeData(filepath):
    df = pd.read_csv(filepath, header=None, sep='\t')
    max_Timestamp = 1135958400
    min_Timestamp = 942249600
    middle_Timestamp = 1104422400
    df_train = df[(df[3] >= min_Timestamp) & (df[3] < middle_Timestamp)]
    df_train.to_csv('data_Netflix/train_three.csv', sep='\t', header=None, index=False)
    df_other = df[(df[3] >= middle_Timestamp) & (df[3] <= max_Timestamp)]
    df_validation = df_other.sample(frac=0.5)
    df_tmp = df_other.append(df_validation)
    df_test = df_tmp.drop_duplicates(keep=False)
    df_validation.to_csv('data_Netflix/validation_three.csv', sep='\t', header=None, index=False)
    df_test.to_csv('data_Netflix/test_three.csv', sep='\t', header=None, index=False)


def append():
    df1 = pd.read_csv('data_Netflix/test_one.csv', header=None, sep='\t')
    df2 = pd.read_csv('data_Netflix/test_two.csv', header=None, sep='\t')
    df3 = pd.read_csv('data_Netflix/test_three.csv', header=None, sep='\t')
    df4 = pd.read_csv('data_Netflix/test_four.csv', header=None, sep='\t')
    result = df1.append([df2, df3, df4])
    result.to_csv('data_Netflix/test.csv', header=None, index=False, sep='\t')


def updateUserId(filepath, validation, test,state):
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
    # print(count)
    df.to_csv('data_Netflix/train_'+state+'.csv', sep='\t', header=None, index=False)
    del df

    # data = json.dumps(user_id)
    # file = open('data_Netflix/user_dict_'+state+'.txt', 'w')
    # file.write(data)

    # file = open('data_Netflix/user_dict_'+state+'.txt', 'r')
    # data = file.read()
    # user_id = json.loads(data)

    df_validation = pd.read_csv(validation, header=None, sep='\t')
    user = list(user_id.keys())
    mask = df_validation[0].isin(user)
    val = df_validation[mask]
    del df_validation
    for i in range(len(val)):
        userId = str(val.iat[i, 0])
        val.iat[i, 0] = user_id[userId]
    val.to_csv('data_Netflix/validation_'+state+'.csv', header=None, sep='\t', index=False)

    # 删除会导致内存过大
    # 那不如新建一个文件，如果userId在dict里面，则把这一行写入到新文件中。（不适用，因为dataframe每一行不方便取）
    # file = open('data_Netflix/user_dict_'+state+'.txt', 'r')
    # data = file.read()
    # user_id = json.loads(data)
    # df_validation = pd.read_csv(validation, header=None, sep='\t')
    # for i in range(len(df_validation)):
    #     userId = str(df_validation.iat[i, 0])
    #     if userId not in user_id.keys():
    #         # delete this line
    #         df_validation.drop(df_validation.index[[i]], inplace=True)
    #         # df_validation = df_validation.drop(i)
    #     else:
    #         df_validation.iat[i, 0] = user_id[userId]
    # df_validation.to_csv('data_Netflix/validation_'+state+'.csv', header=None, sep='\t', index=False)
    # del df_validation

    # df_test = pd.read_csv(test, header=None, sep='\t')
    # for i in range(len(df_test)):
    #     userId = str(df_test.iat[i, 0])
    #     if userId not in user_id.keys():
    #         # delete this line
    #         df_test.drop(df_test.index[[i]], inplace=True)
    #     else:
    #         df_test.iat[i, 0] = user_id[userId]
    # df_test.to_csv('data_Netflix/test_'+state+'.csv', header=None, sep='\t', index=False)
    # del df_test

    df_test = pd.read_csv(test, header=None, sep='\t')
    user = list(user_id.keys())
    mask = df_test[0].isin(user)
    tes = df_test[mask]
    del df_test
    for i in range(len(tes)):
        userId = str(tes.iat[i, 0])
        tes.iat[i, 0] = user_id[userId]
    tes.to_csv('data_Netflix/test_'+state+'.csv', header=None, sep='\t', index=False)
    del tes


def updateItemId(filepath, validation, test,state):
    df = pd.read_csv(filepath, header=None, sep='\t')
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
    df.to_csv('data_Netflix/train_'+state+'.csv', sep='\t', header=None, index=False)
    del df

    df_validation = pd.read_csv(validation, header=None, sep='\t')
    item = list(item_id.keys())
    mask = df_validation[1].isin(item)
    val = df_validation[mask]
    del df_validation
    for i in range(len(val)):
        itemId = str(val.iat[i, 1])
        val.iat[i, 1] = item_id[itemId]
    val.to_csv('data_Netflix/validation_'+state+'.csv', header=None, sep='\t', index=False)

    df_test = pd.read_csv(test, header=None, sep='\t')
    item = list(item_id.keys())
    mask = df_test[1].isin(item)
    tes = df_test[mask]
    del df_test
    for i in range(len(tes)):
        itemId = str(tes.iat[i, 1])
        tes.iat[i, 1] = item_id[itemId]
    tes.to_csv('data_Netflix/test_'+state+'.csv', header=None, sep='\t', index=False)
    del tes

def gettest_users(user_number):
    file = open('data_Netflix/test_users.tsv', 'r')
    for i in range(user_number):
        line = str(i) + '\n'
        file.write(line)
    file.close()

# reserve value > 500
def clearData(filepath, validation, test):
    df = pd.read_csv(filepath, header=None, sep='\t')
    df_validation = pd.read_csv(validation, header=None, sep='\t')
    df_test = pd.read_csv(test, header=None, sep='\t')
    user_id = dict()
    for i in range(len(df)):
        userId = df.iat[i, 0]
        if userId not in user_id.keys():
            user_id[userId] = 1
        else:
            user_id[userId] += 1
    user_id_over20 = {key: value for key, value in user_id.items() if value >= 500}
    print(len(user_id_over20.keys()))

    user = list(user_id_over20.keys())
    train_user = df[0].isin(user)
    trai = df[train_user]
    validation_user = df_validation[0].isin(user)
    val = df_validation[validation_user]
    test_user = df_test[0].isin(user)
    tes = df_test[test_user]

    trai.to_csv('data_Netflix/trainmiddle.csv', header=None, sep='\t', index=False)
    val.to_csv('data_Netflix/validationmiddle.csv', header=None, sep='\t', index=False)
    tes.to_csv('data_Netflix/testmiddle.csv', header=None, sep='\t', index=False)


if __name__ == '__main__':
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

    train = 'data_Netflix/train.csv'
    validation = 'data_Netflix/validation.csv'
    test = 'data_Netflix/test.csv'
    train_stan = 'data_Netflix/train_stan.csv'
    validation_stan = 'data_Netflix/validation_stan.csv'
    test_stan = 'data_Netflix/test_stan.csv'
    trainmiddle = 'data_Netflix/trainmiddle.csv'
    validationmiddle = 'data_Netflix/validationmiddle.csv'
    testmiddle = 'data_Netflix/testmiddle.csv'

    # updateUserId(train, validation, test,'stan')
    clearData(train_stan, validation_stan, test_stan)
    updateUserId(trainmiddle, validationmiddle, testmiddle,'final')
    # update item id
    trainfinal = 'data_Netflix/train_final.csv'
    validationfinal = 'data_Netflix/train_final.csv'
    testfinal
    updateUserId(trainmiddle, validationmiddle, testmiddle,'learn')

    # 更改地址之后再调用，然后还要把user_name file 文件的注释去掉
    # train_middle = 'data_Netflix/trainmiddle.csv'
    # validation_middle = 'data_Netflix/validationmiddle.csv'
    # test_middle = 'data_Netflix/testmiddle.csv'
    # updateUserId(train_middle,validation_middle,test_middle)

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
