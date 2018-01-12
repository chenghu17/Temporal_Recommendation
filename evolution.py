from sklearn.metrics import roc_auc_score as ras
import numpy as np
import pandas as pd
import heapq


def AUC(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    auc = ras(y_true, y_pred)
    return auc

def reCall(validationPath, userMat, itemMat, K):
    df_validation = pd.read_csv(validationPath, sep='\t', header=None)
    for u in range(len(df_validation[0])):
        count = 1
        right = 0
        userId = df_validation.iat[u, 0]
        itemId = df_validation.iat[u, 1]
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
        if itemId in top_k_keys:
            right += 1
        count += 1
    recall = float(right)/float(count)
    return recall

def RMSE():
    return

def MRR():
    return

def MAR():
    return

def NGCG():
    return