# author : hucheng

import numpy as np
import pandas as pd
import random
import time
import metric
from scipy.stats import logistic


class DBPR:

    # d : dimensions of latent factor
    # t : size of time slices
    # steps : iteration number
    # alpha : the learning rate
    # alpha_Reg : the regularization parameter for Pu
    # gama : the regularization parameter

    def __init__(self, rootPath, trainPath, validationPath, d, interval, userNum, itemNum, itemSet, step, alpha, alpha_Reg, gama):
        self.rootPath = rootPath
        self.trainPath = trainPath
        self.validationPath = validationPath
        self.d = d
        # t可以是时间间隔的timestamp表示
        self.interval = interval
        self.userNum = userNum
        self.itemNum = itemNum
        self.itemSet = itemSet
        self.step = step
        self.alpha = alpha
        self.alpha_Reg = alpha_Reg
        self.gama = gama

    def prediction(self, timestamp, validationPath, userMat, itemMat, itemSet):
        pred = list()
        true = list()
        df_validation = pd.read_csv(validationPath, sep='\t', header=None)

        max_Timestamp = pd.Series.max(df_validation[3])
        min_Timestamp = pd.Series.min(df_validation[3])
        current_Timestamp = min_Timestamp + timestamp
        level_down_current = min_Timestamp
        level_up_current = current_Timestamp if current_Timestamp < max_Timestamp else max_Timestamp

        df_interval_current = df_validation[
            (df_validation[3] >= level_down_current) & (df_validation[3] < level_up_current)]

        for u in range(len(df_interval_current[0])):
            df_tmp = df_interval_current[df_interval_current[0] == u]
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
            # print(u)

        Y_True = np.array(true)
        Y_Pred = np.array(pred)
        return Y_True, Y_Pred

    def Time_BPR(self):
        rootPath = self.rootPath
        trainPath = self.trainPath
        validationPath = self.validationPath
        userNum = self.userNum
        itemNum = self.itemNum
        itemSet = self.itemSet
        timestamp = self.interval * 30 * 24 * 3600
        alpha = self.alpha
        alpha_Reg = self.alpha_Reg
        gama = self.gama
        df_train = pd.read_csv(trainPath, sep='\t', header=None)
        max_Timestamp = pd.Series.max(df_train[3])
        min_Timestamp = pd.Series.min(df_train[3])
        time_distance = max_Timestamp - min_Timestamp
        # 根据需时间间隔t来对时序数据进行划分
        time_Step = int(time_distance / timestamp + 1)
        userMat = [0 for n in range(time_Step + 1)]
        itemMat = [0 for n in range(time_Step + 1)]
        for i in range(1, time_Step + 1):
            userMat[i] = np.random.random((userNum, self.d))
            itemMat[i] = np.random.random((itemNum, self.d))

        # 初始化包含每个时间间隔的list
        user_item_time = [0 for n in range(time_Step + 1)]
        for t in range(1, time_Step + 1):
            level_down_current = min_Timestamp + (t - 1) * timestamp
            level_up_current = level_down_current + timestamp
            df_interval_current = df_train[(df_train[3] >= level_down_current) & (df_train[3] < level_up_current)]
            # 单个时间间隔中所包含的user集合
            userSet = set(df_interval_current[0])
            # 初始化包含单个时间间隔中每个用户所打过分的set
            user_item_Set = [0 for n in range(userNum)]
            for userId in userSet:
                df_tmp = df_interval_current[df_interval_current[0] == userId]
                item_tmp = set(df_tmp[1])
                user_item_Set[userId] = item_tmp
            # 包含每个时间间隔中每个用户所打过分的set，用于计算未打过分的负样本
            user_item_time[t] = user_item_Set

        for step in range(self.step):
            starttime = time.time()
            for t in range(1, time_Step + 1):
                level_down_current = min_Timestamp + (t - 1) * timestamp
                level_up_current = level_down_current + timestamp
                df_interval_current = df_train[(df_train[3] >= level_down_current) & (df_train[3] < level_up_current)]
                if t == 1:
                    for line in range(len(df_interval_current)):
                        userId = df_interval_current.iat[line, 0]
                        itemId = df_interval_current.iat[line, 1]
                        Pu = userMat[t][userId]
                        Qi = itemMat[t][itemId]
                        # according to interval & userId, get rating set, and then get negative set
                        item_rating_set = user_item_time[t][userId]
                        negative_item_set = itemSet - item_rating_set
                        # negative sampling, negative number is 1
                        nega_item_id = random.choice(list(negative_item_set))
                        Qk = itemMat[t][nega_item_id]
                        eik = np.dot(Pu, Qi) - np.dot(Pu, Qk)
                        logisticResult = logistic.cdf(-eik)
                        # calculate every gradient
                        gradient_pu = logisticResult * (Qk - Qi) + gama * Pu
                        gradient_qi = logisticResult * (-Pu) + gama * Qi
                        gradient_qk = logisticResult * (Pu) + gama * Qk
                        # update every vector
                        userMat[t][userId] = Pu - alpha * gradient_pu
                        itemMat[t][itemId] = Qi - alpha * gradient_qi
                        itemMat[t][nega_item_id] = Qk - alpha * gradient_qk

                else:
                    last_t = t - 1
                    # level_down_last = level_down_current - timestamp
                    # level_up_last = level_down_current
                    # df_interval_last = df_train[(df_train[3] >= level_down_last) & (df_train[3] < level_up_last)]
                    for line in range(len(df_interval_current)):
                        # if current line == 0
                        # how is going？
                        userId = df_interval_current.iat[line, 0]
                        itemId = df_interval_current.iat[line, 1]
                        Pu = userMat[t][userId]
                        Qi = itemMat[t][itemId]
                        Pu_last = userMat[last_t][userId]
                        Qi_last = itemMat[last_t][itemId]

                        # according to interval & userId, get rating set, and then get negative set
                        # negative sampling, negative number is 1
                        item_rating_set = user_item_time[t][userId]
                        negative_item_set = itemSet - item_rating_set
                        nega_item_id = random.choice(list(negative_item_set))
                        Qk = itemMat[t][nega_item_id]

                        # sampling last interval rating item
                        last_rating_set = user_item_time[last_t][userId]
                        # 判断上一个时间间隔用户是否存在rating item
                        # 如果不存在，根据初始化的情况，该值应该会等于0
                        if last_rating_set != 0:
                            last_rating_id = random.choice(list(last_rating_set))
                            Qj_last = itemMat[last_t][last_rating_id]

                            eii = np.dot(Pu, Qi) - np.dot(Pu_last, Qi_last)
                            eik = np.dot(Pu, Qi) - np.dot(Pu, Qk)
                            ejk = np.dot(Pu_last, Qj_last) - np.dot(Pu, Qk)
                            eii_logistic = logistic.cdf(-eii)
                            eik_logistic = logistic.cdf(-eik)
                            ejk_logistic = logistic.cdf(-ejk)

                            # calculate every gradient
                            gradient_pu = eii_logistic * (-Qi) + eik_logistic * (Qk - Qi) + ejk_logistic * (
                                Qk) + alpha_Reg * (Pu - Pu_last) + gama * Pu
                            gradient_qi = eii_logistic * (-Pu) + eik_logistic * (-Pu) + gama * Qi
                            gradient_qk = eik_logistic * (Pu) + ejk_logistic * (Pu) + gama * Qk
                        else:
                            eii = np.dot(Pu, Qi) - np.dot(Pu_last, Qi_last)
                            eik = np.dot(Pu, Qi) - np.dot(Pu, Qk)
                            eii_logistic = logistic.cdf(-eii)
                            eik_logistic = logistic.cdf(-eik)

                            # calculate every gradient
                            gradient_pu = eii_logistic * (-Qi) + eik_logistic * (Qk - Qi) + alpha_Reg * (
                                    Pu - Pu_last) + gama * Pu
                            gradient_qi = eii_logistic * (-Pu) + eik_logistic * (-Pu) + gama * Qi
                            gradient_qk = eik_logistic * (Pu) + gama * Qk
                        # update every vector
                        userMat[t][userId] = Pu - alpha * gradient_pu
                        itemMat[t][itemId] = Qi - alpha * gradient_qi
                        itemMat[t][nega_item_id] = Qk - alpha * gradient_qk

            # Y_True, Y_Pred = self.prediction(validationPath, userMat[time_Step], itemMat[time_Step], itemSet)
            # auc = evolution.AUC(Y_True, Y_Pred)
            # print('AUC:', auc)
            endtime = time.time()
            # print('%d step :%d' % (step, endtime - starttime))
            if step % 3 == 0:
                userMat_name = 'userMat' + str(step) + '.txt'
                itemMat_name = 'itemMat' + str(step) + '.txt'
                np.savetxt(rootPath+'evolution'+str(self.interval)+'/' + userMat_name, userMat[time_Step])
                np.savetxt(rootPath+'evolution'+str(self.interval)+'/' + itemMat_name, itemMat[time_Step])
                #
                f = open(rootPath+'evolution'+str(self.interval)+'/auc.txt', 'a')
                Y_True, Y_Pred = self.prediction(timestamp, validationPath, userMat[time_Step], itemMat[time_Step],
                                                 itemSet)
                auc = metric.AUC(Y_True, Y_Pred)
                auc_write = str(step) + ' step,auc=' + str(auc)+'\n'
                # print(auc_write)
                f.write(auc_write)
                f.close()


        return userMat[time_Step], itemMat[time_Step]
