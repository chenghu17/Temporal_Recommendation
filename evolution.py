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


def Precision(validationPath, timestamp, userMat, itemMat, K):
    df_validation = pd.read_csv(validationPath, sep='\t', header=None)
    # df_interval_current = 0
    if timestamp != 0:
        max_Timestamp = pd.Series.max(df_validation[3])
        min_Timestamp = pd.Series.min(df_validation[3])
        current_Timestamp = min_Timestamp + timestamp
        level_down_current = min_Timestamp
        level_up_current = current_Timestamp if current_Timestamp < max_Timestamp else max_Timestamp
        df_interval_current = df_validation[
            (df_validation[3] >= level_down_current) & (df_validation[3] < level_up_current)]
    else:
        df_interval_current = df_validation

    userSet = list(df_interval_current[0].drop_duplicates())
    # count = 0
    # right = 0
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

    precisionRate = precisionRate / len(userSet)
    return precisionRate


def reCall(validationPath, timestamp, userMat, itemMat, K):
    df_validation = pd.read_csv(validationPath, sep='\t', header=None)
    # df_interval_current = 0
    if timestamp != 0:
        max_Timestamp = pd.Series.max(df_validation[3])
        min_Timestamp = pd.Series.min(df_validation[3])
        current_Timestamp = min_Timestamp + timestamp
        level_down_current = min_Timestamp
        level_up_current = current_Timestamp if current_Timestamp < max_Timestamp else max_Timestamp
        df_interval_current = df_validation[
            (df_validation[3] >= level_down_current) & (df_validation[3] < level_up_current)]
    else:
        df_interval_current = df_validation

    userSet = list(df_interval_current[0].drop_duplicates())
    # count = 0
    # right = 0
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

    recallRate = float(recallRate) / len(userSet)
    return recallRate


def RMSE():
    return


def MRR(validationPath, timestamp, userMat, itemMat, K):
    #添加这个度量标准
    df_validation = pd.read_csv(validationPath, sep='\t', header=None)
    # df_interval_current = 0
    if timestamp != 0:
        max_Timestamp = pd.Series.max(df_validation[3])
        min_Timestamp = pd.Series.min(df_validation[3])
        current_Timestamp = min_Timestamp + timestamp
        level_down_current = min_Timestamp
        level_up_current = current_Timestamp if current_Timestamp < max_Timestamp else max_Timestamp
        df_interval_current = df_validation[
            (df_validation[3] >= level_down_current) & (df_validation[3] < level_up_current)]
    else:
        df_interval_current = df_validation

    userSet = list(df_interval_current[0].drop_duplicates())
    MRR  = 0
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

        num = 0
        MRR_rate = 0
        for key in top_k_keys:
            if key in df_interval_currentItem:
                index = top_k_keys.index(key)
                MRR_rate += 1.0/(index+1)
                num += 1
        if num != 0:
            MRR_rate = MRR_rate / num
            MRR += MRR_rate
    MRR = MRR / len(userSet)
    return MRR


def MAR():
    return


def NGCG():
    return


if __name__ == '__main__':
    rootPath = 'data_LastFM/'
    # rootPath = 'data_Epinions/'
    # rootPath = 'data_FineFoods/'
    resultPath = 'alpha_0.02_alphaReg_0.02_gama_0.02/'
    # resultPath = 'alpha_0.02_alphaReg_0.1_gama_0.1/'
    # resultPath = 'alpha_0.002_alphaReg_0.02_gama_0.02/'
    # trainPath = rootPath + 'train.tsv'
    validationPath = rootPath + 'validation.tsv'
    # validationPath = rootPath + 'validation.csv'
    timestamp_3 = 3 * 30 * 24 * 3600
    timestamp_6 = 6 * 30 * 24 * 3600
    timestamp_9 = 9 * 30 * 24 * 3600
    timestamp_12 = 12 * 30 * 24 * 3600
    timestamp_18 = 18 * 30 * 24 * 3600
    timestamp_24 = 24 * 30 * 24 * 3600
    timestamp_standard = 0
    k = 20

    rootPath = rootPath + resultPath

    itemMat_3 = np.loadtxt(rootPath + 'evolution3/itemMat10.txt')
    userMat_3 = np.loadtxt(rootPath + 'evolution3/userMat10.txt')

    # itemMat_6 = np.loadtxt(rootPath + 'evolution6/itemMat20.txt')
    # userMat_6 = np.loadtxt(rootPath + 'evolution6/userMat20.txt')
    #
    # itemMat_9 = np.loadtxt(rootPath + 'evolution9/itemMat25.txt')
    # userMat_9 = np.loadtxt(rootPath + 'evolution9/userMat25.txt')
    #
    # itemMat_12 = np.loadtxt(rootPath + 'evolution12/itemMat90.txt')
    # userMat_12 = np.loadtxt(rootPath + 'evolution12/userMat90.txt')
    #
    # itemMat_18 = np.loadtxt(rootPath + 'evolution18/itemMat20.txt')
    # userMat_18 = np.loadtxt(rootPath + 'evolution18/userMat20.txt')
    #
    # itemMat_24 = np.loadtxt(rootPath + 'evolution24/itemMat165.txt')
    # userMat_24 = np.loadtxt(rootPath + 'evolution24/userMat165.txt')
    #
    # itemMat_stand = np.loadtxt(rootPath + 'evolution_standard/itemMat496.txt')
    # userMat_stand = np.loadtxt(rootPath + 'evolution_standard/userMat496.txt')

    Precision_3 = Precision(validationPath, timestamp_3, userMat_3, itemMat_3, k)
    RECALL_3 = reCall(validationPath, timestamp_3, userMat_3, itemMat_3, k)
    # MRR_3 = MRR(validationPath, timestamp_3, userMat_3, itemMat_3, k)
    print('Precision_3:', Precision_3)
    print('RECALL_3:', RECALL_3)
    # print('MRR_3:', MRR_3)

    # Precision_6 = Precision(validationPath, timestamp_6, userMat_6, itemMat_6, k)
    # RECALL_6 = reCall(validationPath, timestamp_6, userMat_6, itemMat_6, k)
    # print('Precision_6:', Precision_6)
    # print('RECALL_6:', RECALL_6)
    # # #
    # Precision_9 = Precision(validationPath, timestamp_9, userMat_9, itemMat_9, k)
    # RECALL_9 = reCall(validationPath, timestamp_9, userMat_9, itemMat_9, k)
    # print('Precision_9:', Precision_9)
    # print('RECALL_9:', RECALL_9)
    # #
    # Precision_12 = Precision(validationPath, timestamp_12, userMat_12, itemMat_12, k)
    # RECALL_12 = reCall(validationPath, timestamp_12, userMat_12, itemMat_12, k)
    # print('Precision_12:', Precision_12)
    # print('RECALL_12:', RECALL_12)
    #
    # Precision_18 = Precision(validationPath, timestamp_18, userMat_18, itemMat_18, k)
    # RECALL_18 = reCall(validationPath, timestamp_18, userMat_18, itemMat_18, k)
    # print('Precision_18:', Precision_18)
    # print('RECALL_18:', RECALL_18)
    #
    # Precision_24 = Precision(validationPath, timestamp_24, userMat_24, itemMat_24, k)
    # RECALL_24 = reCall(validationPath, timestamp_24, userMat_24, itemMat_24, k)
    # print('Precision_24:', Precision_24)
    # print('RECALL_24:', RECALL_24)
    #
    # Precision_stand = Precision(validationPath, timestamp_standard, userMat_stand, itemMat_stand, k)
    # RECALL_stand = reCall(validationPath, timestamp_standard, userMat_stand, itemMat_stand, k)
    # print('Precision_stand:', Precision_stand)
    # print('RECALL_stand:', RECALL_stand)

    # 根据求得的前一个时间间隔itemMat和userMat,对validation数据集计算各种evolution值
