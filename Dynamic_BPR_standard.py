# author : hucheng

import numpy as np
import pandas as pd
import random
import time
import evolution
from scipy.stats import logistic


class DBPR():

    # d : dimensions of latent factor
    # t : size of time slices
    # steps : iteration number
    # alpha : the learning rate
    # alpha_Reg : the regularization parameter for Pu
    # gama : the regularization parameter

    def __init__(self, trainPath, validationPath, d, t, userNum, itemNum, itemSet, step, alpha, alpha_Reg, gama):
        self.trainPath = trainPath
        self.validationPath = validationPath
        self.d = d
        # t可以是时间间隔的timestamp表示
        self.t = t
        self.userNum = userNum
        self.itemNum = itemNum
        self.itemSet = itemSet
        self.step = step
        self.alpha = alpha
        self.alpha_Reg = alpha_Reg
        self.gama = gama

    def prediction(self, validationPath, userMat, itemMat, itemSet):
        pred = list()
        true = list()
        df_validation = pd.read_csv(validationPath, sep='\t', header=None)
        for u in range(len(df_validation[0])):
            df_tmp = df_validation[df_validation[0] == u]
            item_tmp = set(df_tmp[1])
            userId = df_validation.iat[u, 0]
            itemId = df_validation.iat[u, 1]
            Pu = userMat[userId]
            Qi = itemMat[itemId]
            true.append(1)
            Y = np.dot(Pu, Qi)
            pred.append(Y)
            negative_item_set = itemSet - item_tmp
            nega_item_id = random.choice(list(negative_item_set))
            Qk = itemMat[nega_item_id]
            true.append(0)
            Y = np.dot(Pu, Qk)
            pred.append(Y)
            print(u)

        Y_True = np.array(true)
        Y_Pred = np.array(pred)
        return Y_True, Y_Pred

    def Time_BPR(self):
        trainPath = self.trainPath
        validationPath = self.validationPath
        userNum = self.userNum
        itemNum = self.itemNum
        itemSet = self.itemSet
        alpha = self.alpha
        gama = self.gama
        df_train = pd.read_csv(trainPath, sep='\t', header=None)
        userMat = np.random.random((userNum, self.d))
        itemMat = np.random.random((itemNum, self.d))

        # 单个时间间隔中所包含的user集合
        userSet = set(df_train[0])
        # 初始化包含单个时间间隔中每个用户所打过分的set
        user_item_List = [0 for n in range(userNum)]
        user_item_nega_List = [0 for n in range(userNum)]
        for userId in userSet:
            df_tmp = df_train[df_train[0] == userId]
            item_tmp = set(df_tmp[1])
            item_nega = itemSet - item_tmp
            user_item_List[userId] = item_tmp
            user_item_nega_List[userId] = item_nega

        for step in range(self.step):
            starttime = time.time()
            for line in range(len(df_train)):
                userId = df_train.iat[line, 0]
                itemId = df_train.iat[line, 1]
                Pu = userMat[userId]
                Qi = itemMat[itemId]
                # tmp_train = df_train[df_train[0] == userId]
                # item_rating_set = set(tmp_train[1])
                # negative_item_set = itemSet - user_item_List[userId]
                # negative sampling, negative number is 1
                # nega_item_id = random.choice(list(negative_item_set))
                nega_item_id = random.choice(list(user_item_nega_List[userId]))
                Qk = itemMat[nega_item_id]
                eik = np.dot(Pu, Qi) - np.dot(Pu, Qk)
                logisticResult = logistic.cdf(-eik)
                # calculate every gradient
                gradient_pu = logisticResult * (Qk - Qi) + gama * Pu
                gradient_qi = logisticResult * (-Pu) + gama * Qi
                gradient_qk = logisticResult * (Pu) + gama * Qk
                # update every vector
                userMat[userId] = Pu - alpha * gradient_pu
                itemMat[itemId] = Qi - alpha * gradient_qi
                itemMat[nega_item_id] = Qk - alpha * gradient_qk

            endtime = time.time()
            print('%d step :%d' % (step, endtime - starttime))
            if step % 3 == 0:
                # Y_True, Y_Pred = self.prediction(validationPath, userMat, itemMat, itemSet)
                # auc = evolution.AUC(Y_True, Y_Pred)
                # print('AUC:', auc)
                userMat_name = 'userMat' + str(step) + '.txt'
                itemMat_name = 'itemMat' + str(step) + '.txt'
                np.savetxt('evolution_standard/' + userMat_name, userMat)
                np.savetxt('evolution_standard/' + itemMat_name, itemMat)

        return userMat, itemMat
