from sklearn.metrics import roc_auc_score as ras
import numpy as np
import pandas as pd
import heapq
import load_data


def AUC(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    auc = ras(y_true, y_pred)
    return auc

def Precision(validationPath, timestamp ,userMat, itemMat, K):
    df_validation = pd.read_csv(validationPath, sep='\t', header=None)

    max_Timestamp = pd.Series.max(df_validation[3])
    min_Timestamp = pd.Series.min(df_validation[3])
    current_Timestamp = min_Timestamp + timestamp
    level_down_current = min_Timestamp
    level_up_current = current_Timestamp if current_Timestamp < max_Timestamp else max_Timestamp

    df_interval_current = df_validation[
        (df_validation[3] >= level_down_current) & (df_validation[3] < level_up_current)]



    userSet = list(df_interval_current[0].drop_duplicates())
    count = 0
    right = 0
    precisionRate = 0
    for userId in userSet:

        # userId = df_validation.iat[u, 0]
        # itemId = df_validation.iat[u, 1]
        df_tmp = df_interval_current[df_interval_current[0] == userId]
        df_interval_currentItem = set(df_tmp[1])
        itemNum = len(df_interval_currentItem)

        Pu = userMat[userId]
        result = dict()
        for i in range(len(itemMat)):
            Qi = itemMat[i]
            pro = np.dot(Pu, Qi)
            result[i] = pro
        # 判断前k个中是否存在itemId
        top_k_values = heapq.nlargest(K, result.values())
        top_k_keys = list()
        for values in top_k_values:
            for keys in result.keys():
                if result[keys] == values:
                    top_k_keys.append(keys)
        # 取交集，计算
        TP = len(df_interval_currentItem & set(top_k_keys))
        precisionRate += float(TP) / float(K)

    precisionRate = precisionRate/len(userSet)
    return precisionRate

def reCall(validationPath, timestamp, userMat, itemMat,K):
    df_validation = pd.read_csv(validationPath, sep='\t', header=None)

    max_Timestamp = pd.Series.max(df_validation[3])
    min_Timestamp = pd.Series.min(df_validation[3])
    current_Timestamp = min_Timestamp + timestamp
    level_down_current = min_Timestamp
    level_up_current = current_Timestamp if current_Timestamp < max_Timestamp else max_Timestamp

    df_interval_current = df_validation[
        (df_validation[3] >= level_down_current) & (df_validation[3] < level_up_current)]


    userSet = list(df_interval_current[0].drop_duplicates())
    count = 0
    right = 0
    recallRate = 0
    for userId in userSet:
        df_tmp = df_interval_current[df_interval_current[0] == userId]
        df_interval_currentItem = set(df_tmp[1])
        itemNum = len(df_interval_currentItem)

        Pu = userMat[userId]
        result = dict()
        for i in range(len(itemMat)):
            Qi = itemMat[i]
            pro = np.dot(Pu, Qi)
            result[i] = pro
        # 判断前k个中是否存在itemId
        # top_k_values = heapq.nlargest(itemNum, result.values())
        top_k_values = heapq.nlargest(K, result.values())
        top_k_keys = list()
        for values in top_k_values:
            for keys in result.keys():
                if result[keys] == values:
                    top_k_keys.append(keys)
        # 取交集
        TP = len(df_interval_currentItem & set(top_k_keys))
        recallRate += float(TP) / float(itemNum)

    recallRate = float(recallRate)/len(userSet)
    return recallRate


def RMSE():
    return

def MRR():
    return

def MAR():
    return

def NGCG():
    return

if __name__=='__main__':

    trainPath = 'data_FM/train.tsv'
    # validationPath = 'data_FM/validation.tsv'
    validationPath = 'data_FM/test.tsv'
    timestamp_6 = 6 * 30 * 24 * 3600
    timestamp_9 = 9 * 30 * 24 * 3600
    timestamp_12 = 12 * 30 * 24 * 3600

    itemMat_6 = np.loadtxt('evolution6/itemMat15.txt')
    userMat_6 = np.loadtxt('evolution6/userMat15.txt')

    itemMat_9 = np.loadtxt('evolution9/itemMat10.txt')
    userMat_9 = np.loadtxt('evolution9/userMat10.txt')

    itemMat_12 = np.loadtxt('evolution12/itemMat10.txt')
    userMat_12 = np.loadtxt('evolution12/userMat10.txt')

    Precision_6 = Precision(validationPath, timestamp_6, userMat_6, itemMat_6, 50)
    RECALL_6 = reCall(validationPath, timestamp_6, userMat_6, itemMat_6, 50)
    print('Precision_6:', Precision_6)
    print('RECALL_6:', RECALL_6)

    Precision_9 = Precision(validationPath, timestamp_9, userMat_9, itemMat_9, 50)
    RECALL_9 = reCall(validationPath, timestamp_9, userMat_9, itemMat_9, 50)
    print('Precision_9:', Precision_9)
    print('RECALL_9:', RECALL_9)

    Precision_12 = Precision(validationPath, timestamp_12, userMat_12, itemMat_12, 50)
    RECALL_12 = reCall(validationPath, timestamp_12, userMat_12, itemMat_12, 50)
    print('Precision_12:', Precision_12)
    print('RECALL_12:', RECALL_12)


    # itemMat_18 = np.loadtxt('evolution18/itemMat30.txt')
    # userMat_18 = np.loadtxt('evolution18/userMat30.txt')

    # itemMat_stand = np.loadtxt('evolution_standard/itemMat48.txt')
    # userMat_stand = np.loadtxt('evolution_standard/userMat48.txt')

    # Precision_18 = Precision(validationPath,userMat_18,itemMat_18,30)
    # RECALL_18 = reCall(validationPath,userMat_18,itemMat_18,30)
    # print('Precision_18:',Precision_18)
    # print('RECALL_18:',RECALL_18)

    # Precision_stand = Precision(validationPath,userMat_stand,itemMat_stand,30)
    # RECALL_stand = reCall(validationPath,userMat_stand,itemMat_stand,30)
    # print('Precision_stand:',Precision_stand)
    # print('RECALL_stand:',RECALL_stand)

    #根据求得的前一个时间间隔itemMat和userMat,对validation数据集计算各种evolution值
